#include "cuda_common.cuh"
#include "runway/instance_io.hpp"
#include "runway/sim_core.hpp"
#include "runway/rng.hpp"
#include <chrono>
#include <cmath>
#include <algorithm>
#include <numeric>

namespace runway {
namespace cuda {

// Declare (not define) — defined in sa_kernel.cu
extern __constant__ float c_sep[36];

void generate_noise_gpu(
    float* d_delay,
    float* d_mult,
    float* d_closeA,
    float* d_closeB,
    const int* d_op,
    int n,
    int S,
    float t_min,
    float t_max,
    uint64_t seed,
    float p_close = 0.20f
);

template <int BLOCK_SIZE, int MAX_N>
extern __global__ void sa_k2_k3_kernel(
    int* d_chain_perms,
    int* d_chain_bests,
    float* d_chain_J_bests,
    uint64_t* d_rng_states,
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
);

} // namespace cuda

#if defined(RUNWAY_HAS_CUDA)

SAResult solve_sa_cuda(
    const AircraftData& data,
    const SeparationMatrix& sep,
    int C,
    int S,
    int iters,
    uint64_t seed,
    const std::string& kernel_name,
    int block_size,
    float m_penalty
) {
    auto wall_start = std::chrono::high_resolution_clock::now();
    int n = data.n;

    int E = std::max(1, std::min(10, iters / 100));
    int I = std::max(1, iters / E);
    float lambda_std = 0.0f;

    // 1. Upload separation matrix to constant memory
    CUDA_CHECK(cudaMemcpyToSymbol(cuda::c_sep, sep.data, 36 * sizeof(float)));

    // 2. Allocate and upload aircraft SoA arrays
    float *d_ready, *d_target, *d_latest, *d_weight;
    int *d_cls, *d_op;
    size_t ac_fbytes = n * sizeof(float);
    size_t ac_ibytes = n * sizeof(int);

    CUDA_CHECK(cudaMalloc(&d_ready, ac_fbytes));
    CUDA_CHECK(cudaMalloc(&d_target, ac_fbytes));
    CUDA_CHECK(cudaMalloc(&d_latest, ac_fbytes));
    CUDA_CHECK(cudaMalloc(&d_weight, ac_fbytes));
    CUDA_CHECK(cudaMalloc(&d_cls, ac_ibytes));
    CUDA_CHECK(cudaMalloc(&d_op, ac_ibytes));

    CUDA_CHECK(cudaMemcpy(d_ready, data.ready.data(), ac_fbytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_target, data.target.data(), ac_fbytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_latest, data.latest.data(), ac_fbytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_weight, data.weight.data(), ac_fbytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cls, data.cls.data(), ac_ibytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_op, data.op.data(), ac_ibytes, cudaMemcpyHostToDevice));

    // 3. Initialize chains on host
    std::vector<int> fcfs = solve_fcfs(data);
    std::vector<int> h_chain_perms(C * n);
    std::vector<int> h_chain_bests(C * n);
    std::vector<float> h_chain_J_bests(C, 1e30f);
    std::vector<uint64_t> h_rng_states(C);

    for (int c = 0; c < C; ++c) {
        XorShift64Star rng(seed ^ (static_cast<uint64_t>(c) * 0x9e3779b97f4a7c15ULL + 1ULL));
        std::vector<int> p = fcfs;
        if (c > 0) {
            int perturb = 1 + rng.next_int(5);
            std::vector<int> p_next(n);
            for (int k = 0; k < perturb; ++k) {
                propose_move(p.data(), p_next.data(), n, rng);
                p = p_next;
            }
        }
        for (int i = 0; i < n; ++i) {
            h_chain_perms[c * n + i] = p[i];
            h_chain_bests[c * n + i] = p[i];
        }
        h_rng_states[c] = rng.state;
    }

    int *d_chain_perms, *d_chain_bests;
    float* d_chain_J_bests;
    uint64_t* d_rng_states;
    size_t chain_pbytes = static_cast<size_t>(C) * n * sizeof(int);

    CUDA_CHECK(cudaMalloc(&d_chain_perms, chain_pbytes));
    CUDA_CHECK(cudaMalloc(&d_chain_bests, chain_pbytes));
    CUDA_CHECK(cudaMalloc(&d_chain_J_bests, C * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_rng_states, C * sizeof(uint64_t)));

    CUDA_CHECK(cudaMemcpy(d_chain_perms, h_chain_perms.data(), chain_pbytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_chain_bests, h_chain_bests.data(), chain_pbytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_chain_J_bests, h_chain_J_bests.data(), C * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rng_states, h_rng_states.data(), C * sizeof(uint64_t), cudaMemcpyHostToDevice));

    // 4. Allocate noise memory on GPU
    float *d_delay, *d_mult, *d_closeA, *d_closeB;
    size_t delay_bytes = static_cast<size_t>(n) * S * sizeof(float);
    size_t s_bytes = S * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_delay, delay_bytes));
    CUDA_CHECK(cudaMalloc(&d_mult, s_bytes));
    CUDA_CHECK(cudaMalloc(&d_closeA, s_bytes));
    CUDA_CHECK(cudaMalloc(&d_closeB, s_bytes));

    float t_min = *std::min_element(data.target.begin(), data.target.end());
    float t_max = *std::max_element(data.target.begin(), data.target.end());

    // Temperature calibration (same math as CPU)
    NoiseData calib_noise;
    generate_noise_cpu(calib_noise, data, S, seed + 1);
    float J_base = evaluate_schedule_cpu(data, sep, fcfs, calib_noise, m_penalty).J;
    XorShift64Star calib_rng(seed + 999);
    std::vector<int> tmp_p = fcfs;
    std::vector<int> prop_p(n);
    double w_sum = 0.0;
    int w_cnt = 0;
    for (int k = 0; k < 200; ++k) {
        propose_move(tmp_p.data(), prop_p.data(), n, calib_rng);
        float J_p = evaluate_schedule_cpu(data, sep, prop_p, calib_noise, m_penalty).J;
        float d = J_p - J_base;
        if (d > 0.0f) { w_sum += d; w_cnt++; }
    }
    float mean_w = (w_cnt > 0) ? static_cast<float>(w_sum / w_cnt) : 10.0f;
    float T0 = std::max(1e-4f, mean_w / 0.693147f);
    int total_iters = E * I;
    float alpha = std::pow(1e-3f, 1.0f / static_cast<float>(total_iters > 0 ? total_iters : 1));

    // CUDA timing events
    cudaEvent_t k_start, k_stop;
    CUDA_CHECK(cudaEventCreate(&k_start));
    CUDA_CHECK(cudaEventCreate(&k_stop));

    float current_T = T0;
    std::vector<int> global_best_perm = fcfs;
    float global_best_J = 1e30f;
    float total_kernel_ms = 0.0f;

    // 5. Epoch loop
    for (int epoch = 1; epoch <= E; ++epoch) {
        // Generate GPU noise for this epoch
        cuda::generate_noise_gpu(
            d_delay, d_mult, d_closeA, d_closeB,
            d_op, n, S, t_min, t_max,
            seed + static_cast<uint64_t>(epoch) * 1000ULL
        );

        CUDA_CHECK(cudaEventRecord(k_start));

        // Launch kernel
        if (block_size == 128) {
            cuda::sa_k2_k3_kernel<128, 256><<<C, 128>>>(
                d_chain_perms, d_chain_bests, d_chain_J_bests, d_rng_states,
                C, n, S, I, alpha, current_T,
                d_ready, d_target, d_latest, d_weight, d_cls,
                d_delay, d_mult, d_closeA, d_closeB,
                lambda_std, m_penalty
            );
        } else if (block_size == 512) {
            cuda::sa_k2_k3_kernel<512, 256><<<C, 512>>>(
                d_chain_perms, d_chain_bests, d_chain_J_bests, d_rng_states,
                C, n, S, I, alpha, current_T,
                d_ready, d_target, d_latest, d_weight, d_cls,
                d_delay, d_mult, d_closeA, d_closeB,
                lambda_std, m_penalty
            );
        } else {
            cuda::sa_k2_k3_kernel<256, 256><<<C, 256>>>(
                d_chain_perms, d_chain_bests, d_chain_J_bests, d_rng_states,
                C, n, S, I, alpha, current_T,
                d_ready, d_target, d_latest, d_weight, d_cls,
                d_delay, d_mult, d_closeA, d_closeB,
                lambda_std, m_penalty
            );
        }

        CUDA_CHECK(cudaEventRecord(k_stop));
        CUDA_CHECK(cudaEventSynchronize(k_stop));

        float k_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&k_ms, k_start, k_stop));
        total_kernel_ms += k_ms;

        current_T *= std::pow(alpha, static_cast<float>(I));

        // Epoch synchronization: retrieve chain bests
        CUDA_CHECK(cudaMemcpy(h_chain_bests.data(), d_chain_bests, chain_pbytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_chain_J_bests.data(), d_chain_J_bests, C * sizeof(float), cudaMemcpyDeviceToHost));

        // Re-score chain-bests on fresh CPU scenario set to combat winner's curse
        NoiseData sync_noise;
        generate_noise_cpu(sync_noise, data, S, seed + static_cast<uint64_t>(epoch) * 7777ULL + 500000ULL);

        struct ScorePair { int chain_idx; float score; };
        std::vector<ScorePair> scores(C);
        for (int c = 0; c < C; ++c) {
            std::vector<int> p(h_chain_bests.begin() + c * n, h_chain_bests.begin() + (c + 1) * n);
            float sc = evaluate_schedule_cpu(data, sep, p, sync_noise, m_penalty).J;
            scores[c] = {c, sc};
            if (sc < global_best_J) {
                global_best_J = sc;
                global_best_perm = p;
            }
        }

        std::sort(scores.begin(), scores.end(), [](const ScorePair& a, const ScorePair& b) {
            return a.score < b.score;
        });

        int best_chain_idx = scores[0].chain_idx;
        const int* best_p_ptr = h_chain_bests.data() + best_chain_idx * n;

        int num_replace = std::max(1, C / 4);
        for (int r = 0; r < num_replace; ++r) {
            int worst_idx = scores[C - 1 - r].chain_idx;
            CUDA_CHECK(cudaMemcpy(
                d_chain_perms + worst_idx * n,
                best_p_ptr,
                n * sizeof(int),
                cudaMemcpyHostToDevice
            ));
            CUDA_CHECK(cudaMemcpy(
                d_chain_bests + worst_idx * n,
                best_p_ptr,
                n * sizeof(int),
                cudaMemcpyHostToDevice
            ));
        }
    }

    auto wall_end = std::chrono::high_resolution_clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(wall_end - wall_start).count();

    // Clean up
    CUDA_CHECK(cudaFree(d_ready));
    CUDA_CHECK(cudaFree(d_target));
    CUDA_CHECK(cudaFree(d_latest));
    CUDA_CHECK(cudaFree(d_weight));
    CUDA_CHECK(cudaFree(d_cls));
    CUDA_CHECK(cudaFree(d_op));
    CUDA_CHECK(cudaFree(d_chain_perms));
    CUDA_CHECK(cudaFree(d_chain_bests));
    CUDA_CHECK(cudaFree(d_chain_J_bests));
    CUDA_CHECK(cudaFree(d_rng_states));
    CUDA_CHECK(cudaFree(d_delay));
    CUDA_CHECK(cudaFree(d_mult));
    CUDA_CHECK(cudaFree(d_closeA));
    CUDA_CHECK(cudaFree(d_closeB));
    CUDA_CHECK(cudaEventDestroy(k_start));
    CUDA_CHECK(cudaEventDestroy(k_stop));

    long long total_sim_evals = static_cast<long long>(C) * E * I;
    SAResult result;
    result.best_perm = global_best_perm;
    result.in_sample_J = global_best_J;
    result.kernel_time_ms = total_kernel_ms;
    result.total_time_ms = total_ms;
    result.sim_per_sec = (static_cast<double>(total_sim_evals) * S) / (total_ms / 1000.0);
    result.epochs = E;
    result.iters_per_epoch = I;
    result.chains = C;
    result.scenarios = S;

    return result;
}

#endif // RUNWAY_HAS_CUDA

} // namespace runway
