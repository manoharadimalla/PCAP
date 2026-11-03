#include <catch2/catch_test_macros.hpp>
#include "runway/types.hpp"
#include "runway/sim_core.hpp"
#include "runway/instance_io.hpp"

using namespace runway;

TEST_CASE("Closure edge cases", "[closure]") {
    AircraftData data;
    Aircraft A{0, OP_ARR, WAKE_HEAVY, 0, 100.0f, 100.0f, 1000.0f, PRIO_NORMAL, 1.0f};
    Aircraft B{1, OP_ARR, WAKE_MEDIUM, 1, 150.0f, 150.0f, 1000.0f, PRIO_NORMAL, 1.0f};
    data.add(A);
    data.add(B);

    SeparationMatrix sep = get_default_separation_matrix();
    float delay[2] = {0.0f, 0.0f};
    int perm[2] = {0, 1};

    float times_base[2];
    SimMetrics m_base = simulate_one_with_times(
        perm, 2, 0, 1,
        data.ready.data(), data.target.data(), data.latest.data(),
        data.weight.data(), data.cls.data(), sep.data,
        delay, 0.0f, 0.0f,
        times_base
    );
    // A at 100, B at max(150, 100+157) = 257. Cost: 0 + (257-150) = 107.
    REQUIRE(times_base[0] == 100.0f);
    REQUIRE(times_base[1] == 257.0f);

    SECTION("Empty closure (a=b) equals no-closure result") {
        float times[2];
        SimMetrics m = simulate_one_with_times(
            perm, 2, 0, 1,
            data.ready.data(), data.target.data(), data.latest.data(),
            data.weight.data(), data.cls.data(), sep.data,
            delay, 120.0f, 120.0f,
            times
        );
        CHECK(times[0] == times_base[0]);
        CHECK(times[1] == times_base[1]);
        CHECK(m.cost == m_base.cost);
    }

    SECTION("Closure entirely before all ready times has no effect") {
        float times[2];
        SimMetrics m = simulate_one_with_times(
            perm, 2, 0, 1,
            data.ready.data(), data.target.data(), data.latest.data(),
            data.weight.data(), data.cls.data(), sep.data,
            delay, 10.0f, 50.0f,
            times
        );
        CHECK(times[0] == times_base[0]);
        CHECK(times[1] == times_base[1]);
        CHECK(m.cost == m_base.cost);
    }

    SECTION("Closure covering entire horizon pushes every aircraft to >= b") {
        float times[2];
        simulate_one_with_times(
            perm, 2, 0, 1,
            data.ready.data(), data.target.data(), data.latest.data(),
            data.weight.data(), data.cls.data(), sep.data,
            delay, 50.0f, 500.0f,
            times
        );
        CHECK(times[0] == 500.0f);
        CHECK(times[1] == 500.0f + 157.0f);
    }

    SECTION("Closure start equal to computed time pushes it ([a, b) includes a)") {
        // A would land at 100. Closure [100, 200). A must be pushed to 200.
        float times[2];
        simulate_one_with_times(
            perm, 2, 0, 1,
            data.ready.data(), data.target.data(), data.latest.data(),
            data.weight.data(), data.cls.data(), sep.data,
            delay, 100.0f, 200.0f,
            times
        );
        CHECK(times[0] == 200.0f);
        // B = max(150, 200 + 157) = 357
        CHECK(times[1] == 357.0f);
    }

    SECTION("Closure end equal to computed time does not push it ([a, b) excludes b)") {
        // A would land at 100. Closure [50, 100). A is not inside [50, 100).
        float times[2];
        simulate_one_with_times(
            perm, 2, 0, 1,
            data.ready.data(), data.target.data(), data.latest.data(),
            data.weight.data(), data.cls.data(), sep.data,
            delay, 50.0f, 100.0f,
            times
        );
        CHECK(times[0] == 100.0f);
        CHECK(times[1] == 257.0f);
    }
}
