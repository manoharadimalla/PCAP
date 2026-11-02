#include <catch2/catch_test_macros.hpp>
#include "runway/rng.hpp"
#include "runway/sim_core.hpp"
#include <vector>
#include <numeric>

using namespace runway;

TEST_CASE("Moves fuzzing: always produces valid permutation", "[moves]") {
    const int n = 30;
    std::vector<int> src(n);
    std::iota(src.begin(), src.end(), 0);
    std::vector<int> dst(n);
    XorShift64Star rng(987654321ULL);

    std::vector<bool> seen(n);
    const int FUZZ_COUNT = 1000000; // 1e6 moves

    for (int iter = 0; iter < FUZZ_COUNT; ++iter) {
        propose_move(src.data(), dst.data(), n, rng);

        std::fill(seen.begin(), seen.end(), false);
        bool valid = true;
        for (int i = 0; i < n; ++i) {
            int val = dst[i];
            if (val < 0 || val >= n || seen[val]) {
                valid = false;
                break;
            }
            seen[val] = true;
        }

        if (!valid) {
            FAIL("Invalid permutation generated at iteration " << iter);
        }

        // Periodically use dst as the new src
        if ((iter & 0x7F) == 0) {
            src = dst;
        }
    }
    SUCCEED("Successfully fuzzed 1,000,000 moves without invalid permutations.");
}
