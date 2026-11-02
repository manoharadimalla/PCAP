#include "runway/instance_io.hpp"
#include "runway/sim_core.hpp"
#include <chrono>
#include <numeric>
#include <algorithm>
#include <iostream>

namespace runway {

SAResult solve_brute_force(
    const AircraftData& data,
    const SeparationMatrix& sep,
    const NoiseData& noise,
    float lambda_std,
    float m_penalty
) {
    auto start_time = std::chrono::high_resolution_clock::now();
    int n = data.n;

    std::vector<int> perm(n);
    std::iota(perm.begin(), perm.end(), 0);

    std::vector<int> best_perm = perm;
    float best_J = 1e30f;
    long long total_evals = 0;

    int S = noise.S;
    std::vector<float> costs(S);

    do {
        double sum_cost = 0.0;
        double sum_sq_cost = 0.0;
        for (int s = 0; s < S; ++s) {
            SimMetrics m = simulate_one(
                perm.data(), n, s, S,
                data.ready.data(), data.target.data(), data.latest.data(),
                data.weight.data(), data.cls.data(), sep.data,
                noise.delay.data(), noise.mult[s],
                noise.closeA[s], noise.closeB[s],
                sep.triangle_ok, m_penalty
            );
            sum_cost += m.cost;
            sum_sq_cost += static_cast<double>(m.cost) * m.cost;
        }
        float mean = static_cast<float>(sum_cost / S);
        double var = (sum_sq_cost / S) - (static_cast<double>(mean) * mean);
        float std_val = static_cast<float>(var > 0.0 ? std::sqrt(var) : 0.0);
        float J = mean + lambda_std * std_val;

        if (J < best_J) {
            best_J = J;
            best_perm = perm;
        }
        total_evals++;
    } while (std::next_permutation(perm.begin(), perm.end()));

    auto end_time = std::chrono::high_resolution_clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();

    SAResult res;
    res.best_perm = best_perm;
    res.in_sample_J = best_J;
    res.kernel_time_ms = total_ms;
    res.total_time_ms = total_ms;
    res.sim_per_sec = (static_cast<double>(total_evals) * S) / (total_ms / 1000.0);
    res.scenarios = S;
    res.iters_per_epoch = static_cast<int>(total_evals);
    res.epochs = 1;
    res.chains = 1;

    return res;
}

} // namespace runway
