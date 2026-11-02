#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "runway/types.hpp"
#include "runway/sim_core.hpp"
#include "runway/instance_io.hpp"

using namespace runway;

TEST_CASE("Serial determinism: same seed produces identical results", "[baselines]") {
    AircraftData data = generate_instance(15, 1.2f, 100);
    SeparationMatrix sep = get_default_separation_matrix();

    SAResult res1 = solve_sa_serial(data, sep, 8, 256, 3, 500, 424242);
    SAResult res2 = solve_sa_serial(data, sep, 8, 256, 3, 500, 424242);

    REQUIRE(res1.best_perm.size() == res2.best_perm.size());
    for (size_t i = 0; i < res1.best_perm.size(); ++i) {
        CHECK(res1.best_perm[i] == res2.best_perm[i]);
    }
    CHECK(res1.in_sample_J == res2.in_sample_J);
}

TEST_CASE("Brute-force optimum vs SA on n=8, S=500", "[baselines]") {
    int n = 8;
    int S = 500;
    AircraftData data = generate_instance(n, 1.2f, 777);
    SeparationMatrix sep = get_default_separation_matrix();

    NoiseData noise;
    generate_noise_cpu(noise, data, S, 12345);

    SAResult opt_brute = solve_brute_force(data, sep, noise, 1.0f, 5.0f);
    float true_opt_J = opt_brute.in_sample_J;

    int success_count = 0;
    const int NUM_SEEDS = 20;

    for (int s = 1; s <= NUM_SEEDS; ++s) {
        // SA search with fixed noise for direct comparison to brute-force optimum
        // We use C=16 chains, E=5 epochs, I=600 iterations
        SAResult sa = solve_sa_serial(data, sep, 16, S, 5, 600, static_cast<uint64_t>(s));
        // Evaluate the SA best schedule on the exact same fixed noise
        EvalResult ev = evaluate_schedule_cpu(data, sep, sa.best_perm, noise);
        float sa_J = ev.J;

        float relative_gap = (sa_J - true_opt_J) / true_opt_J;
        if (relative_gap <= 0.005f) { // Within 0.5%
            success_count++;
        }
    }

    INFO("Success count: " << success_count << " / " << NUM_SEEDS);
    // Gate: >= 90% (18/20)
    CHECK(success_count >= 18);
}
