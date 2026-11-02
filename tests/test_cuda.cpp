#include <catch2/catch_test_macros.hpp>
#include "runway/types.hpp"

#if defined(RUNWAY_HAS_CUDA)
TEST_CASE("CUDA test placeholder", "[cuda]") {
    REQUIRE(true);
}
#endif
