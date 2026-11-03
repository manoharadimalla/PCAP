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

    do {
        double sum_cost = 0.0;
        for (int s = 0; s < S; ++s) {
            SimMetrics m = simulate_one(
                perm.data(), n, s, S,
                data.ready.data(), data.target.data(), data.latest.data(),
                data.weight.data(), data.cls.data(), sep.data,
                noise.delay.data(),
                noise.closeA[s], noise.closeB[s],
                m_penalty
            );
            sum_cost += m.cost;
        }
        float mean = static_cast<float>(sum_cost / S);

        if (mean < best_J) {
            best_J = mean;
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
