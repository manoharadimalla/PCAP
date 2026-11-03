#include "runway/instance_io.hpp"
#include "runway/sim_core.hpp"
#include "runway/rng.hpp"
#include <chrono>
#include <numeric>
#include <algorithm>

namespace runway {

SAResult solve_random_search(
    const AircraftData& data,
    const SeparationMatrix& sep,
    int S,
    int num_samples,
    uint64_t seed,
    float m_penalty
) {
    auto start_time = std::chrono::high_resolution_clock::now();
    int n = data.n;

    NoiseData noise;
    generate_noise_cpu(noise, data, S, seed);

    XorShift64Star rng(seed);
    std::vector<int> perm(n);
    std::iota(perm.begin(), perm.end(), 0);

    std::vector<int> best_perm = perm;
    float best_J = 1e30f;

    for (int iter = 0; iter < num_samples; ++iter) {
        // Fisher-Yates shuffle
        for (int i = n - 1; i > 0; --i) {
            int j = rng.next_int(i + 1);
            std::swap(perm[i], perm[j]);
        }

        EvalResult eval = evaluate_schedule_cpu(data, sep, perm, noise, m_penalty);
        if (eval.mean_cost < best_J) {
            best_J = eval.mean_cost;
            best_perm = perm;
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();

    SAResult res;
    res.best_perm = best_perm;
    res.in_sample_J = best_J;
    res.kernel_time_ms = total_ms;
    res.total_time_ms = total_ms;
    res.sim_per_sec = (static_cast<double>(num_samples) * S) / (total_ms / 1000.0);
    res.scenarios = S;
    res.iters_per_epoch = num_samples;
    res.epochs = 1;
    res.chains = 1;

    return res;
}

} // namespace runway
