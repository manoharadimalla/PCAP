#include <catch2/catch_test_macros.hpp>
#include "runway/types.hpp"
#include "runway/sim_core.hpp"
#include "runway/instance_io.hpp"

using namespace runway;

TEST_CASE("FCFS sorting properties", "[decoder]") {
    SECTION("FCFS is sorted by target and breaks ties by id") {
        AircraftData data;
        Aircraft A{0, OP_ARR, WAKE_HEAVY, 0, 0.0f, 100.0f, 1000.0f, PRIO_NORMAL, 1.0f};
        Aircraft B{1, OP_ARR, WAKE_MEDIUM, 1, 0.0f, 50.0f, 1000.0f, PRIO_NORMAL, 1.0f};
        Aircraft C{2, OP_ARR, WAKE_LIGHT, 2, 0.0f, 100.0f, 1000.0f, PRIO_NORMAL, 1.0f};
        data.add(A);
        data.add(B);
        data.add(C);

        std::vector<int> fcfs = solve_fcfs(data);
        REQUIRE(fcfs.size() == 3);
        CHECK(fcfs[0] == 1);
        CHECK(fcfs[1] == 0);
        CHECK(fcfs[2] == 2);
    }
}
