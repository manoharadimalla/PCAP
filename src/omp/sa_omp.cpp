#include "runway/instance_io.hpp"
#include "runway/sim_core.hpp"
#include "runway/rng.hpp"
#include <vector>
#include <cmath>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <iostream>

#if defined(RUNWAY_HAS_OPENMP)
#include <omp.h>
#endif

namespace runway {

#if defined(RUNWAY_HAS_OPENMP)

// Scenario-parallel evaluator (fine-grained)
static inline float evaluate_scenarios_omp(
    const int* perm,
    int n,
    const AircraftData& data,
    const SeparationMatrix& sep,
    const NoiseData& noise,
    float m_penalty = 5.0f
) {
    int S = noise.S;
    double sum = 0.0;

    #pragma omp parallel for reduction(+:sum) schedule(static)
    for (int s = 0; s < S; ++s) {
        SimMetrics m = simulate_one(
            perm, n, s, S,
            data.ready.data(), data.target.data(), data.latest.data(),
            data.weight.data(), data.cls.data(), sep.data,
            noise.delay.data(),
            noise.closeA[s], noise.closeB[s],
            m_penalty
        );
        sum += m.cost;
    }

    return static_cast<float>(sum / S);
}

// Single-thread scenario evaluator used inside coarse-grained OpenMP chains
static inline float evaluate_single_thread(
    const int* perm,
    int n,
    const AircraftData& data,
    const SeparationMatrix& sep,
    const NoiseData& noise,
    float m_penalty = 5.0f
) {
    int S = noise.S;
    double sum = 0.0;
    for (int s = 0; s < S; ++s) {
        SimMetrics m = simulate_one(
            perm, n, s, S,
            data.ready.data(), data.target.data(), data.latest.data(),
            data.weight.data(), data.cls.data(), sep.data,
            noise.delay.data(),
            noise.closeA[s], noise.closeB[s],
            m_penalty
        );
        sum += m.cost;
    }
    return static_cast<float>(sum / S);
}

SAResult solve_sa_omp(
    const AircraftData& data,
    const SeparationMatrix& sep,
    int C,
    int S,
    int iters,
    uint64_t seed,
    const std::string& mode,
    int num_threads,
    float m_penalty,
    bool progress
) {
    auto start_time = std::chrono::high_resolution_clock::now();
    int n = data.n;

    if (num_threads > 0) {
        omp_set_num_threads(num_threads);
    }

    // 1. Generate training noise once (shared across all chains/scenarios)
    NoiseData train_noise;
    generate_noise_cpu(train_noise, data, S, seed);

    // 2. Initial schedule: FCFS
    std::vector<int> fcfs_perm = solve_fcfs(data);
    float fcfs_J = evaluate_single_thread(fcfs_perm.data(), n, data, sep, train_noise, m_penalty);

    // 3. Auto-calibrate initial temperature T0
    XorShift64Star calib_rng(seed + 9999ULL);
    double sum_pos_delta = 0.0;
    int pos_delta_count = 0;
    std::vector<int> cur_p = fcfs_perm;
    std::vector<int> prop_p(n);

    for (int step = 0; step < 50; ++step) {
        propose_move(cur_p.data(), prop_p.data(), n, calib_rng);
        float J_p = evaluate_single_thread(prop_p.data(), n, data, sep, train_noise, m_penalty);
        float delta = J_p - fcfs_J;
        if (delta > 0.0f) {
            sum_pos_delta += delta;
            pos_delta_count++;
        }
        cur_p = prop_p;
    }
    float avg_delta = (pos_delta_count > 0) ? static_cast<float>(sum_pos_delta / pos_delta_count) : 100.0f;
    float T0 = -avg_delta / std::log(0.50f);
    if (T0 < 1.0f) T0 = 100.0f;

    float T_end = 0.01f;
    float alpha = std::pow(T_end / T0, 1.0f / static_cast<float>(std::max(1, iters)));

    struct ChainState {
        std::vector<int> perm;
        float J_cur;
        std::vector<int> best_perm;
        float J_best;
        XorShift64Star rng;
    };

    std::vector<ChainState> chains(C);
    for (int c = 0; c < C; ++c) {
        chains[c].perm = fcfs_perm;
        chains[c].J_cur = fcfs_J;
        chains[c].best_perm = fcfs_perm;
        chains[c].J_best = fcfs_J;
        chains[c].rng = XorShift64Star(seed + static_cast<uint64_t>(c) * 10007ULL + 1ULL);
    }

    if (mode == "chains") {
        // Coarse-grained OpenMP: whole chains parallelized across CPU threads
        #pragma omp parallel for schedule(static)
        for (int c = 0; c < C; ++c) {
            std::vector<int> proposal(n);
            float T = T0;
            for (int it = 0; it < iters; ++it) {
                propose_move(chains[c].perm.data(), proposal.data(), n, chains[c].rng);
                float J_prop = evaluate_single_thread(proposal.data(), n, data, sep, train_noise, m_penalty);

                float delta = J_prop - chains[c].J_cur;
                bool accept = false;
                if (delta <= 0.0f) {
                    accept = true;
                } else {
                    float r = chains[c].rng.next_float();
                    if (r < std::exp(-delta / T)) {
                        accept = true;
                    }
                }

                if (accept) {
                    chains[c].perm = proposal;
                    chains[c].J_cur = J_prop;
                    if (J_prop < chains[c].J_best) {
                        chains[c].J_best = J_prop;
                        chains[c].best_perm = proposal;
                    }
                }
                T *= alpha;
            }
        }
    } else {
        // Fine-grained OpenMP ("scenarios"): chains run sequentially, scenarios parallelized with OpenMP reduction
        std::vector<int> proposal(n);
        for (int c = 0; c < C; ++c) {
            float T = T0;
            for (int it = 0; it < iters; ++it) {
                propose_move(chains[c].perm.data(), proposal.data(), n, chains[c].rng);
                float J_prop = evaluate_scenarios_omp(proposal.data(), n, data, sep, train_noise, m_penalty);

                float delta = J_prop - chains[c].J_cur;
                bool accept = false;
                if (delta <= 0.0f) {
                    accept = true;
                } else {
                    float r = chains[c].rng.next_float();
                    if (r < std::exp(-delta / T)) {
                        accept = true;
                    }
                }

                if (accept) {
                    chains[c].perm = proposal;
                    chains[c].J_cur = J_prop;
                    if (J_prop < chains[c].J_best) {
                        chains[c].J_best = J_prop;
                        chains[c].best_perm = proposal;
                    }
                }
                T *= alpha;
            }
        }
    }

    // Select the best chain by training score
    int best_c = 0;
    float best_train_J = chains[0].J_best;
    for (int c = 1; c < C; ++c) {
        if (chains[c].J_best < best_train_J) {
            best_train_J = chains[c].J_best;
            best_c = c;
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();

    long long total_sim_evals = static_cast<long long>(C) * iters;
    SAResult result;
    result.best_perm = chains[best_c].best_perm;
    result.in_sample_J = best_train_J;
    result.kernel_time_ms = total_ms;
    result.total_time_ms = total_ms;
    result.sim_per_sec = (static_cast<double>(total_sim_evals) * S) / (total_ms / 1000.0);
    result.epochs = 1;
    result.iters_per_epoch = iters;
    result.chains = C;
    result.scenarios = S;

    if (progress) {
        std::cerr << "DONE " << best_train_J << " in " << (total_ms / 1000.0) << "s\n";
    }

    return result;
}

#endif // RUNWAY_HAS_OPENMP

} // namespace runway
