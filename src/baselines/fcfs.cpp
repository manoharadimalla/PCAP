#include "runway/instance_io.hpp"
#include <algorithm>
#include <numeric>

namespace runway {

std::vector<int> solve_fcfs(const AircraftData& data) {
    int n = data.n;
    std::vector<int> perm(n);
    std::iota(perm.begin(), perm.end(), 0);

    // Sort by target time; tie-break by id
    std::sort(perm.begin(), perm.end(), [&](int a, int b) {
        if (data.target[a] != data.target[b]) {
            return data.target[a] < data.target[b];
        }
        return data.id[a] < data.id[b];
    });

    return perm;
}

} // namespace runway
