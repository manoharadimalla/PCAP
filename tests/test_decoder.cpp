#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "runway/types.hpp"
#include "runway/sim_core.hpp"
#include "runway/instance_io.hpp"

using namespace runway;

TEST_CASE("Decoder equivalence and FCFS properties", "[decoder]") {
    SECTION("FCFS is sorted by target and breaks ties by id") {
        AircraftData data;
        Aircraft A{0, OP_ARR, WAKE_HEAVY, 0, 0.0f, 100.0f, 1000.0f, PRIO_NORMAL, 1.0f};
        Aircraft B{1, OP_ARR, WAKE_MEDIUM, 1, 0.0f, 50.0f, 1000.0f, PRIO_NORMAL, 1.0f};
        Aircraft C{2, OP_ARR, WAKE_LIGHT, 2, 0.0f, 100.0f, 1000.0f, PRIO_NORMAL, 1.0f};
        data.add(A);
        data.add(B);
        data.add(C);

        std::vector<int> fcfs = solve_fcfs(data);
        // Expect B (target 50), then A (target 100, id 0), then C (target 100, id 2)
        REQUIRE(fcfs.size() == 3);
        CHECK(fcfs[0] == 1);
        CHECK(fcfs[1] == 0);
        CHECK(fcfs[2] == 2);
    }

    SECTION("Exact decoder == fast-path decoder when triangle inequality holds") {
        // Construct a matrix where triangle inequality is strictly satisfied
        SeparationMatrix sep;
        for (int a = 0; a < 6; ++a) {
            for (int b = 0; b < 6; ++b) {
                sep.set(a, b, 60.0f); // Constant matrix trivially satisfies triangle inequality
            }
        }
        sep.check_triangle_inequality();
        REQUIRE(sep.triangle_ok);

        AircraftData data = generate_instance(30, 1.2f, 1234);
        NoiseData noise;
        generate_noise_cpu(noise, data, 50, 5678, 0.30f);
        std::vector<int> perm = solve_fcfs(data);

        for (int s = 0; s < noise.S; ++s) {
            SimMetrics m_exact = simulate_one_exact(
                perm.data(), data.n, s, noise.S,
                data.ready.data(), data.target.data(), data.latest.data(),
                data.weight.data(), data.cls.data(), sep.data,
                noise.delay.data(), noise.mult[s],
                noise.closeA[s], noise.closeB[s]
            );

            SimMetrics m_fast = simulate_one_fast(
                perm.data(), data.n, s, noise.S,
                data.ready.data(), data.target.data(), data.latest.data(),
                data.weight.data(), data.cls.data(), sep.data,
                noise.delay.data(), noise.mult[s],
                noise.closeA[s], noise.closeB[s]
            );

            CHECK_THAT(m_exact.cost, Catch::Matchers::WithinRel(m_fast.cost, 1e-5f));
            CHECK(m_exact.misses == m_fast.misses);
        }
    }
}
