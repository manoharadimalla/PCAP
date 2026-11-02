// Test driver entry point. Catch2::Catch2WithMain provides the main() function.
#include <catch2/catch_test_macros.hpp>

TEST_CASE("Sanity check", "[sanity]") {
    REQUIRE(true);
}
