#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "runway/types.hpp"
#include "runway/sim_core.hpp"
#include "runway/instance_io.hpp"

#if defined(RUNWAY_HAS_OPENMP)
#include <omp.h>

using namespace runway;

TEST_CASE("OpenMP evaluation equivalence to serial", "[omp]") {
    AircraftData data = generate_instance(30, 1.2f, 555);
    SeparationMatrix sep = get_default_separation_matrix();

    NoiseData noise;
    generate_noise_cpu(noise, data, 1024, 789);

    std::vector<int> perm = solve_fcfs(data);

    EvalResult serial_res = evaluate_schedule_cpu(data, sep, perm, noise);

    // Test parallel scenario evaluation with reduction
    int S = noise.S;
    double sum = 0.0;

    #pragma omp parallel for reduction(+:sum) schedule(static)
    for (int s = 0; s < S; ++s) {
        SimMetrics m = simulate_one(
            perm.data(), data.n, s, S,
            data.ready.data(), data.target.data(), data.latest.data(),
            data.weight.data(), data.cls.data(), sep.data,
            noise.delay.data(),
            noise.closeA[s], noise.closeB[s],
            5.0f
        );
        sum += m.cost;
    }

    float omp_mean = static_cast<float>(sum / S);

    // Relative error < 1e-6
    CHECK_THAT(omp_mean, Catch::Matchers::WithinRel(serial_res.mean_cost, 1e-6f));
}

#endif // RUNWAY_HAS_OPENMP
