#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "runway/types.hpp"
#include "runway/sim_core.hpp"
#include "runway/instance_io.hpp"

using namespace runway;

TEST_CASE("Worked example Section 4.7", "[worked_example]") {
    // Three arrivals A(H, cls 0), B(M, cls 1), C(L, cls 2)
    // ready=0, target=0, weight=1, latest=1000
    AircraftData data;
    Aircraft A{0, OP_ARR, WAKE_HEAVY,  0, 0.0f, 0.0f, 1000.0f, PRIO_NORMAL, 1.0f};
    Aircraft B{1, OP_ARR, WAKE_MEDIUM, 1, 0.0f, 0.0f, 1000.0f, PRIO_NORMAL, 1.0f};
    Aircraft C{2, OP_ARR, WAKE_LIGHT,  2, 0.0f, 0.0f, 1000.0f, PRIO_NORMAL, 1.0f};
    data.add(A);
    data.add(B);
    data.add(C);

    SeparationMatrix sep = get_default_separation_matrix();

    float delay[3] = {0.0f, 0.0f, 0.0f};
    float times[3] = {0.0f, 0.0f, 0.0f};

    SECTION("Order A, B, C -> times 0, 157, 288 -> cost 445") {
        int perm[3] = {0, 1, 2};
        SimMetrics m = simulate_one_with_times(
            perm, 3, 0, 1,
            data.ready.data(), data.target.data(), data.latest.data(),
            data.weight.data(), data.cls.data(), sep.data,
            delay, 1.0f, 0.0f, 0.0f,
            times, false
        );

        CHECK(times[0] == 0.0f);
        CHECK(times[1] == 157.0f);
        CHECK(times[2] == 288.0f);
        CHECK(m.cost == 445.0f);
    }

    SECTION("Order C, B, A -> times 0, 69, 129 -> cost 198") {
        int perm[3] = {2, 1, 0};
        SimMetrics m = simulate_one_with_times(
            perm, 3, 0, 1,
            data.ready.data(), data.target.data(), data.latest.data(),
            data.weight.data(), data.cls.data(), sep.data,
            delay, 1.0f, 0.0f, 0.0f,
            times, false
        );

        CHECK(times[0] == 0.0f);
        CHECK(times[1] == 69.0f);
        CHECK(times[2] == 129.0f);
        CHECK(m.cost == 198.0f);
    }

    SECTION("Closure example: order C, B, A with closure [50, 100) -> cost 260") {
        int perm[3] = {2, 1, 0};
        SimMetrics m = simulate_one_with_times(
            perm, 3, 0, 1,
            data.ready.data(), data.target.data(), data.latest.data(),
            data.weight.data(), data.cls.data(), sep.data,
            delay, 1.0f, 50.0f, 100.0f,
            times, false
        );

        CHECK(times[0] == 0.0f);
        CHECK(times[1] == 100.0f);
        CHECK(times[2] == 160.0f);
        CHECK(m.cost == 260.0f);
    }

    SECTION("Empty closure (a=b) gives 198") {
        int perm[3] = {2, 1, 0};
        SimMetrics m = simulate_one_with_times(
            perm, 3, 0, 1,
            data.ready.data(), data.target.data(), data.latest.data(),
            data.weight.data(), data.cls.data(), sep.data,
            delay, 1.0f, 50.0f, 50.0f,
            times, false
        );

        CHECK(m.cost == 198.0f);
    }
}
