#include "runway/instance_io.hpp"
#include "runway/sim_core.hpp"
#include "runway/rng.hpp"
#include <vector>
#include <cmath>
#include <chrono>
#include <algorithm>
#include <numeric>

#if defined(RUNWAY_HAS_OPENMP)
#include <omp.h>
#endif

namespace runway {

#if defined(RUNWAY_HAS_OPENMP)

static inline float evaluate_scenarios_omp(
    const int* perm,
    int n,
    const AircraftData& data,
    const SeparationMatrix& sep,
    const NoiseData& noise,
    float lambda_std,
    float m_penalty
) {
    int S = noise.S;
    double sum = 0.0;
    double sum_sq = 0.0;

    #pragma omp parallel for reduction(+:sum, sum_sq) schedule(static)
    for (int s = 0; s < S; ++s) {
        SimMetrics m = simulate_one(
            perm, n, s, S,
            data.ready.data(), data.target.data(), data.latest.data(),
            data.weight.data(), data.cls.data(), sep.data,
            noise.delay.data(), noise.mult[s],
            noise.closeA[s], noise.closeB[s],
            sep.triangle_ok, m_penalty
        );
        sum += m.cost;
        sum_sq += static_cast<double>(m.cost) * m.cost;
    }

    float mean = static_cast<float>(sum / S);
    double var = (sum_sq / S) - (static_cast<double>(mean) * mean);
    float std_val = static_cast<float>(var > 0.0 ? std::sqrt(var) : 0.0);
    return mean + lambda_std * std_val;
}

static inline float evaluate_scenarios_simd(
    const int* perm,
    int n,
    const AircraftData& data,
    const SeparationMatrix& sep,
    const NoiseData& noise,
    float lambda_std,
    float m_penalty
) {
    int S = noise.S;
    double sum = 0.0;
    double sum_sq = 0.0;

    #pragma omp simd reduction(+:sum, sum_sq)
    for (int s = 0; s < S; ++s) {
        SimMetrics m = simulate_one(
            perm, n, s, S,
            data.ready.data(), data.target.data(), data.latest.data(),
            data.weight.data(), data.cls.data(), sep.data,
            noise.delay.data(), noise.mult[s],
            noise.closeA[s], noise.closeB[s],
            sep.triangle_ok, m_penalty
        );
        sum += m.cost;
        sum_sq += static_cast<double>(m.cost) * m.cost;
    }

    float mean = static_cast<float>(sum / S);
    double var = (sum_sq / S) - (static_cast<double>(mean) * mean);
    float std_val = static_cast<float>(var > 0.0 ? std::sqrt(var) : 0.0);
    return mean + lambda_std * std_val;
}

static inline float evaluate_scenarios_serial(
    const int* perm,
    int n,
    const AircraftData& data,
    const SeparationMatrix& sep,
    const NoiseData& noise,
    float lambda_std,
    float m_penalty
) {
    int S = noise.S;
    double sum = 0.0;
    double sum_sq = 0.0;

    for (int s = 0; s < S; ++s) {
        SimMetrics m = simulate_one(
            perm, n, s, S,
            data.ready.data(), data.target.data(), data.latest.data(),
            data.weight.data(), data.cls.data(), sep.data,
            noise.delay.data(), noise.mult[s],
            noise.closeA[s], noise.closeB[s],
            sep.triangle_ok, m_penalty
        );
        sum += m.cost;
        sum_sq += static_cast<double>(m.cost) * m.cost;
    }

    float mean = static_cast<float>(sum / S);
    double var = (sum_sq / S) - (static_cast<double>(mean) * mean);
    float std_val = static_cast<float>(var > 0.0 ? std::sqrt(var) : 0.0);
    return mean + lambda_std * std_val;
}

SAResult solve_sa_omp(
    const AircraftData& data,
    const SeparationMatrix& sep,
    int C,
    int S,
    int E,
    int I,
    uint64_t seed,
    const std::string& mode,
    int num_threads,
    float lambda_std,
    float m_penalty
) {
    if (num_threads > 0) {
        omp_set_num_threads(num_threads);
    }

    auto start_time = std::chrono::high_resolution_clock::now();
    int n = data.n;

    std::vector<int> fcfs = solve_fcfs(data);

    // Initial noise for calibration
    NoiseData calib_noise;
    generate_noise_cpu(calib_noise, data, S, seed + 1);

    float J_base = evaluate_scenarios_serial(fcfs.data(), n, data, sep, calib_noise, lambda_std, m_penalty);

    // T0 calibration
    XorShift64Star calib_rng(seed + 999);
    std::vector<int> tmp_perm = fcfs;
    std::vector<int> prop_perm(n);
    double worsening_sum = 0.0;
    int worsening_count = 0;

    for (int k = 0; k < 200; ++k) {
        propose_move(tmp_perm.data(), prop_perm.data(), n, calib_rng);
        float J_p = evaluate_scenarios_serial(prop_perm.data(), n, data, sep, calib_noise, lambda_std, m_penalty);
        float delta = J_p - J_base;
        if (delta > 0.0f) {
            worsening_sum += delta;
            worsening_count++;
        }
    }

    float mean_worsening = (worsening_count > 0) ? static_cast<float>(worsening_sum / worsening_count) : 10.0f;
    float T0 = mean_worsening / 0.693147f;
    if (T0 < 1e-4f) T0 = 10.0f;

    int total_iters = E * I;
    float T_end = 1e-3f * T0;
    float alpha = std::pow(T_end / T0, 1.0f / static_cast<float>(total_iters > 0 ? total_iters : 1));

    struct Chain {
        std::vector<int> perm;
        std::vector<int> best_perm;
        float J_cur = 0.0f;
        float J_best = 1e30f;
        XorShift64Star rng;
    };

    std::vector<Chain> chains(C);
    for (int c = 0; c < C; ++c) {
        chains[c].rng.set_seed(seed ^ (static_cast<uint64_t>(c) * 0x9e3779b97f4a7c15ULL + 1ULL));
        chains[c].perm = fcfs;
        if (c > 0) {
            int perturb = 1 + chains[c].rng.next_int(5);
            std::vector<int> p_next(n);
            for (int p = 0; p < perturb; ++p) {
                propose_move(chains[c].perm.data(), p_next.data(), n, chains[c].rng);
                chains[c].perm = p_next;
            }
        }
        chains[c].best_perm = chains[c].perm;
    }

    float current_T = T0;
    std::vector<int> global_best_perm = fcfs;
    float global_best_J = 1e30f;

    NoiseData epoch_noise;
    NoiseData sync_noise;

    for (int epoch = 1; epoch <= E; ++epoch) {
        generate_noise_cpu(epoch_noise, data, S, seed + static_cast<uint64_t>(epoch) * 1000ULL);

        if (mode == "chains") {
            // O1: Parallel across chains
            #pragma omp parallel for schedule(static)
            for (int c = 0; c < C; ++c) {
                std::vector<int> proposal(n);
                chains[c].J_cur = evaluate_scenarios_serial(chains[c].perm.data(), n, data, sep, epoch_noise, lambda_std, m_penalty);
                chains[c].J_best = chains[c].J_cur;
                chains[c].best_perm = chains[c].perm;

                float T = current_T;
                for (int it = 0; it < I; ++it) {
                    propose_move(chains[c].perm.data(), proposal.data(), n, chains[c].rng);
                    float J_prop = evaluate_scenarios_serial(proposal.data(), n, data, sep, epoch_noise, lambda_std, m_penalty);

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
        } else if (mode == "hybrid") {
            // O3: Chains parallel + SIMD scenarios
            #pragma omp parallel for schedule(static)
            for (int c = 0; c < C; ++c) {
                std::vector<int> proposal(n);
                chains[c].J_cur = evaluate_scenarios_simd(chains[c].perm.data(), n, data, sep, epoch_noise, lambda_std, m_penalty);
                chains[c].J_best = chains[c].J_cur;
                chains[c].best_perm = chains[c].perm;

                float T = current_T;
                for (int it = 0; it < I; ++it) {
                    propose_move(chains[c].perm.data(), proposal.data(), n, chains[c].rng);
                    float J_prop = evaluate_scenarios_simd(proposal.data(), n, data, sep, epoch_noise, lambda_std, m_penalty);

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
            // O2: Scenarios parallel (chains serial)
            std::vector<int> proposal(n);
            for (int c = 0; c < C; ++c) {
                chains[c].J_cur = evaluate_scenarios_omp(chains[c].perm.data(), n, data, sep, epoch_noise, lambda_std, m_penalty);
                chains[c].J_best = chains[c].J_cur;
                chains[c].best_perm = chains[c].perm;

                float T = current_T;
                for (int it = 0; it < I; ++it) {
                    propose_move(chains[c].perm.data(), proposal.data(), n, chains[c].rng);
                    float J_prop = evaluate_scenarios_omp(proposal.data(), n, data, sep, epoch_noise, lambda_std, m_penalty);

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

        current_T *= std::pow(alpha, static_cast<float>(I));

        // Epoch synchronization
        generate_noise_cpu(sync_noise, data, S, seed + static_cast<uint64_t>(epoch) * 7777ULL + 500000ULL);
        struct ScorePair {
            int chain_idx;
            float score;
        };
        std::vector<ScorePair> scores(C);
        for (int c = 0; c < C; ++c) {
            float sc = evaluate_scenarios_serial(chains[c].best_perm.data(), n, data, sep, sync_noise, lambda_std, m_penalty);
            scores[c] = {c, sc};
            if (sc < global_best_J) {
                global_best_J = sc;
                global_best_perm = chains[c].best_perm;
            }
        }

        std::sort(scores.begin(), scores.end(), [](const ScorePair& a, const ScorePair& b) {
            return a.score < b.score;
        });

        int best_chain_idx = scores[0].chain_idx;
        const std::vector<int>& best_p = chains[best_chain_idx].best_perm;

        int num_replace = std::max(1, C / 4);
        for (int r = 0; r < num_replace; ++r) {
            int worst_idx = scores[C - 1 - r].chain_idx;
            chains[worst_idx].perm = best_p;
            chains[worst_idx].best_perm = best_p;
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();

    long long total_sim_evals = static_cast<long long>(C) * E * I;
    SAResult result;
    result.best_perm = global_best_perm;
    result.in_sample_J = global_best_J;
    result.kernel_time_ms = total_ms;
    result.total_time_ms = total_ms;
    result.sim_per_sec = (static_cast<double>(total_sim_evals) * S) / (total_ms / 1000.0);
    result.epochs = E;
    result.iters_per_epoch = I;
    result.chains = C;
    result.scenarios = S;

    return result;
}

#endif // RUNWAY_HAS_OPENMP

} // namespace runway
