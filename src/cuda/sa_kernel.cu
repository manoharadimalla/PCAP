#include "cuda_common.cuh"
#include "runway/rng.hpp"
#include "runway/sim_core.hpp"
#include "runway/types.hpp"

namespace runway {
namespace cuda {

// Definition of GPU constant memory separation matrix (declared extern in cuda_common.cuh)
__constant__ float c_sep[36];

template <int BLOCK_SIZE, int MAX_N>
__global__ void sa_k2_k3_kernel(
    int* d_chain_perms,       // [C * n] current perms
    int* d_chain_bests,       // [C * n] best perms
    float* d_chain_J_bests,   // [C] best J
    uint64_t* d_rng_states,   // [C] per-chain RNG state
    int C,
    int n,
    int S,
    int I,
    float alpha,
    float initial_T,
    const float* __restrict__ d_ready,
    const float* __restrict__ d_target,
    const float* __restrict__ d_latest,
    const float* __restrict__ d_weight,
    const int* __restrict__ d_cls,
    const float* __restrict__ d_delay,
    const float* __restrict__ d_mult,
    const float* __restrict__ d_closeA,
    const float* __restrict__ d_closeB,
    float lambda_std,
    float m_penalty
) {
    int cid = blockIdx.x; // Block per chain
    if (cid >= C) return;

    int tid = threadIdx.x;

    // Shared memory allocations
    __shared__ int s_perm[MAX_N];
    __shared__ int s_prop[MAX_N];
    __shared__ int s_best[MAX_N];

    __shared__ float s_ready[MAX_N];
    __shared__ float s_target[MAX_N];
    __shared__ float s_latest[MAX_N];
    __shared__ float s_weight[MAX_N];
    __shared__ int s_cls[MAX_N];

    __shared__ float s_warp_sums[32];
    __shared__ float s_warp_sumsq[32];

    __shared__ float s_T;
    __shared__ float s_J_cur;
    __shared__ float s_J_best;
    __shared__ int s_accept;
    __shared__ uint64_t s_rng_state;

    // 1. Cooperative load of aircraft data into shared memory
    for (int i = tid; i < n; i += BLOCK_SIZE) {
        s_ready[i] = d_ready[i];
        s_target[i] = d_target[i];
        s_latest[i] = d_latest[i];
        s_weight[i] = d_weight[i];
        s_cls[i] = d_cls[i];
        s_perm[i] = d_chain_perms[cid * n + i];
        s_prop[i] = s_perm[i];
        s_best[i] = d_chain_bests[cid * n + i];
    }

    if (tid == 0) {
        s_T = initial_T;
        s_rng_state = d_rng_states[cid];
        s_J_best = d_chain_J_bests[cid];
    }
    __syncthreads();

    // 2. Evaluate initial permutation on current epoch's noise
    float init_sum = 0.0f;
    float init_sumsq = 0.0f;
    for (int s = tid; s < S; s += BLOCK_SIZE) {
        float lastT[6] = {-1e9f, -1e9f, -1e9f, -1e9f, -1e9f, -1e9f};
        float cost = 0.0f;
        float mult_s = d_mult[s];
        float closeA = d_closeA[s];
        float closeB = d_closeB[s];

        #pragma unroll 4
        for (int k = 0; k < n; ++k) {
            int a = s_perm[k];
            float t = s_ready[a] + __ldg(&d_delay[a * S + s]);
            int cls_a = s_cls[a];

            float sep_bound = -1e9f;
            #pragma unroll
            for (int c = 0; c < 6; ++c) {
                float cand = lastT[c] + mult_s * c_sep[c * 6 + cls_a];
                if (cand > sep_bound) sep_bound = cand;
            }
            if (sep_bound > t) t = sep_bound;

            t = (t >= closeA && t < closeB) ? closeB : t;
            lastT[cls_a] = t;

            float d = t - s_target[a];
            if (d > 0.0f) cost += s_weight[a] * d;
            if (t > s_latest[a]) cost += m_penalty * s_weight[a] * (t - s_latest[a]);
        }
        init_sum += cost;
        init_sumsq += cost * cost;
    }

    float b_sum = block_reduce_sum<float, BLOCK_SIZE>(init_sum, s_warp_sums);
    float b_sumsq = block_reduce_sum<float, BLOCK_SIZE>(init_sumsq, s_warp_sumsq);

    if (tid == 0) {
        float mean = b_sum / S;
        float var = (b_sumsq / S) - (mean * mean);
        float std_val = (var > 0.0f) ? sqrtf(var) : 0.0f;
        s_J_cur = mean + lambda_std * std_val;
        if (s_J_cur < s_J_best) {
            s_J_best = s_J_cur;
            for (int i = 0; i < n; ++i) s_best[i] = s_perm[i];
        }
    }
    __syncthreads();

    // 3. Main SA Search Loop (I iterations per epoch)
    XorShift64Star rng;

    for (int iter = 0; iter < I; ++iter) {
        // Thread 0 proposes neighborhood move
        if (tid == 0) {
            rng.state = s_rng_state;
            propose_move(s_perm, s_prop, n, rng);
            s_rng_state = rng.state;
            s_accept = 0;
        }
        __syncthreads();

        // Level-1 Parallelism: Threads evaluate scenarios
        float local_sum = 0.0f;
        float local_sumsq = 0.0f;

        for (int s = tid; s < S; s += BLOCK_SIZE) {
            float lastT[6] = {-1e9f, -1e9f, -1e9f, -1e9f, -1e9f, -1e9f};
            float cost = 0.0f;
            float mult_s = d_mult[s];
            float closeA = d_closeA[s];
            float closeB = d_closeB[s];

            #pragma unroll 4
            for (int k = 0; k < n; ++k) {
                int a = s_prop[k];
                float t = s_ready[a] + __ldg(&d_delay[a * S + s]);
                int cls_a = s_cls[a];

                float sep_bound = -1e9f;
                #pragma unroll
                for (int c = 0; c < 6; ++c) {
                    float cand = lastT[c] + mult_s * c_sep[c * 6 + cls_a];
                    if (cand > sep_bound) sep_bound = cand;
                }
                if (sep_bound > t) t = sep_bound;

                t = (t >= closeA && t < closeB) ? closeB : t;
                lastT[cls_a] = t;

                float d = t - s_target[a];
                if (d > 0.0f) cost += s_weight[a] * d;
                if (t > s_latest[a]) cost += m_penalty * s_weight[a] * (t - s_latest[a]);
            }
            local_sum += cost;
            local_sumsq += cost * cost;
        }

        // Warp-shuffle parallel reduction
        float cand_sum = block_reduce_sum<float, BLOCK_SIZE>(local_sum, s_warp_sums);
        float cand_sumsq = block_reduce_sum<float, BLOCK_SIZE>(local_sumsq, s_warp_sumsq);

        if (tid == 0) {
            float mean = cand_sum / S;
            float var = (cand_sumsq / S) - (mean * mean);
            float std_val = (var > 0.0f) ? sqrtf(var) : 0.0f;
            float J_prop = mean + lambda_std * std_val;

            float delta = J_prop - s_J_cur;
            bool accept = false;
            if (delta <= 0.0f) {
                accept = true;
            } else {
                float r = rng.next_float();
                if (r < expf(-delta / s_T)) {
                    accept = true;
                }
            }

            if (accept) {
                s_accept = 1;
                s_J_cur = J_prop;
                if (J_prop < s_J_best) {
                    s_J_best = J_prop;
                    for (int i = 0; i < n; ++i) s_best[i] = s_prop[i];
                }
            }
            s_T *= alpha;
            s_rng_state = rng.state;
        }
        __syncthreads();

        // If accepted, update s_perm in parallel
        if (s_accept) {
            for (int i = tid; i < n; i += BLOCK_SIZE) {
                s_perm[i] = s_prop[i];
            }
            __syncthreads();
        }
    }

    // 4. Write back chain best, current perm, and RNG state to global memory
    for (int i = tid; i < n; i += BLOCK_SIZE) {
        d_chain_perms[cid * n + i] = s_perm[i];
        d_chain_bests[cid * n + i] = s_best[i];
    }
    if (tid == 0) {
        d_chain_J_bests[cid] = s_J_best;
        d_rng_states[cid] = s_rng_state;
    }
}

// Explicit instantiations for block sizes
template __global__ void sa_k2_k3_kernel<64, 256>(
    int*, int*, float*, uint64_t*, int, int, int, int, float, float,
    const float*, const float*, const float*, const float*, const int*,
    const float*, const float*, const float*, const float*, float, float
);

template __global__ void sa_k2_k3_kernel<128, 256>(
    int*, int*, float*, uint64_t*, int, int, int, int, float, float,
    const float*, const float*, const float*, const float*, const int*,
    const float*, const float*, const float*, const float*, float, float
);

template __global__ void sa_k2_k3_kernel<256, 256>(
    int*, int*, float*, uint64_t*, int, int, int, int, float, float,
    const float*, const float*, const float*, const float*, const int*,
    const float*, const float*, const float*, const float*, float, float
);

template __global__ void sa_k2_k3_kernel<512, 256>(
    int*, int*, float*, uint64_t*, int, int, int, int, float, float,
    const float*, const float*, const float*, const float*, const int*,
    const float*, const float*, const float*, const float*, float, float
);

} // namespace cuda
} // namespace runway
