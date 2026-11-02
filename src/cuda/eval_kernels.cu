#include "cuda_common.cuh"
#include "runway/sim_core.hpp"
#include "runway/types.hpp"

namespace runway {
namespace cuda {

__constant__ float c_sep[36];

// K1 Naive Baseline: thread-per-candidate
__global__ void eval_k1_candidates_kernel(
    const int* d_perms, // [num_candidates * n]
    int num_candidates,
    int n,
    int S,
    const float* d_ready,
    const float* d_target,
    const float* d_latest,
    const float* d_weight,
    const int* d_cls,
    const float* d_delay,
    const float* d_mult,
    const float* d_closeA,
    const float* d_closeB,
    float* d_out_J,
    float lambda_std,
    float m_penalty
) {
    int cid = blockDim.x * blockIdx.x + threadIdx.x;
    if (cid >= num_candidates) return;

    const int* perm = d_perms + cid * n;

    double sum = 0.0;
    double sum_sq = 0.0;

    for (int s = 0; s < S; ++s) {
        float lastT[6] = {-1e9f, -1e9f, -1e9f, -1e9f, -1e9f, -1e9f};
        float cost = 0.0f;
        float mult_s = d_mult[s];
        float closeA = d_closeA[s];
        float closeB = d_closeB[s];

        for (int k = 0; k < n; ++k) {
            int a = perm[k];
            float t = d_ready[a] + d_delay[a * S + s];
            int cls_a = d_cls[a];

            float sep_bound = -1e9f;
            #pragma unroll
            for (int c = 0; c < 6; ++c) {
                float cand = lastT[c] + mult_s * c_sep[c * 6 + cls_a];
                if (cand > sep_bound) sep_bound = cand;
            }
            if (sep_bound > t) t = sep_bound;

            t = (t >= closeA && t < closeB) ? closeB : t;
            lastT[cls_a] = t;

            float d = t - d_target[a];
            if (d > 0.0f) cost += d_weight[a] * d;
            if (t > d_latest[a]) {
                cost += m_penalty * d_weight[a] * (t - d_latest[a]);
            }
        }
        sum += cost;
        sum_sq += (double)cost * cost;
    }

    float mean = (float)(sum / S);
    double var = (sum_sq / S) - ((double)mean * mean);
    float std_val = (var > 0.0) ? sqrtf((float)var) : 0.0f;
    d_out_J[cid] = mean + lambda_std * std_val;
}

// Single-permutation reduction kernel (1 block, BLOCK_SIZE threads)
template <int BLOCK_SIZE>
__global__ void eval_single_perm_kernel(
    const int* d_perm,
    int n,
    int S,
    const float* d_ready,
    const float* d_target,
    const float* d_latest,
    const float* d_weight,
    const int* d_cls,
    const float* d_delay,
    const float* d_mult,
    const float* d_closeA,
    const float* d_closeB,
    float* d_out_J,
    float lambda_std,
    float m_penalty
) {
    __shared__ float s_warp_sums[32];
    __shared__ float s_warp_sumsq[32];

    int tid = threadIdx.x;
    float local_sum = 0.0f;
    float local_sumsq = 0.0f;

    for (int s = tid; s < S; s += blockDim.x) {
        float lastT[6] = {-1e9f, -1e9f, -1e9f, -1e9f, -1e9f, -1e9f};
        float cost = 0.0f;
        float mult_s = d_mult[s];
        float closeA = d_closeA[s];
        float closeB = d_closeB[s];

        for (int k = 0; k < n; ++k) {
            int a = d_perm[k];
            float t = d_ready[a] + d_delay[a * S + s];
            int cls_a = d_cls[a];

            float sep_bound = -1e9f;
            #pragma unroll
            for (int c = 0; c < 6; ++c) {
                float cand = lastT[c] + mult_s * c_sep[c * 6 + cls_a];
                if (cand > sep_bound) sep_bound = cand;
            }
            if (sep_bound > t) t = sep_bound;

            t = (t >= closeA && t < closeB) ? closeB : t;
            lastT[cls_a] = t;

            float d = t - d_target[a];
            if (d > 0.0f) cost += d_weight[a] * d;
            if (t > d_latest[a]) cost += m_penalty * d_weight[a] * (t - d_latest[a]);
        }
        local_sum += cost;
        local_sumsq += cost * cost;
    }

    float block_sum = block_reduce_sum<float, BLOCK_SIZE>(local_sum, s_warp_sums);
    float block_sumsq = block_reduce_sum<float, BLOCK_SIZE>(local_sumsq, s_warp_sumsq);

    if (tid == 0) {
        float mean = block_sum / S;
        float var = (block_sumsq / S) - (mean * mean);
        float std_val = (var > 0.0f) ? sqrtf(var) : 0.0f;
        *d_out_J = mean + lambda_std * std_val;
    }
}

} // namespace cuda
} // namespace runway
