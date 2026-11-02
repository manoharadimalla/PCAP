#include <catch2/catch_test_macros.hpp>
#include "runway/types.hpp"
#include "runway/sim_core.hpp"
#include "runway/instance_io.hpp"

using namespace runway;

TEST_CASE("Validator catches violations and accepts valid schedules", "[validator]") {
    AircraftData data = generate_instance(20, 1.2f, 42);
    SeparationMatrix sep = get_default_separation_matrix();

    NoiseData noise;
    generate_noise_cpu(noise, data, 10, 12345, 0.40f);

    std::vector<int> fcfs = solve_fcfs(data);

    SECTION("FCFS is valid for all scenarios") {
        for (int s = 0; s < noise.S; ++s) {
            std::string err;
            bool ok = validate_schedule(data, sep, fcfs, noise, s, &err);
            INFO("Scenario " << s << " error: " << err);
            CHECK(ok);
        }
    }

    SECTION("Catches duplicate index in permutation") {
        std::vector<int> bad_perm = fcfs;
        bad_perm[1] = bad_perm[0];
        std::string err;
        bool ok = validate_schedule(data, sep, bad_perm, noise, 0, &err);
        CHECK_FALSE(ok);
        CHECK(err.find("Duplicate") != std::string::npos);
    }

    SECTION("Catches out-of-bounds index") {
        std::vector<int> bad_perm = fcfs;
        bad_perm[0] = 999;
        std::string err;
        bool ok = validate_schedule(data, sep, bad_perm, noise, 0, &err);
        CHECK_FALSE(ok);
        CHECK(err.find("bounds") != std::string::npos);
    }
}
