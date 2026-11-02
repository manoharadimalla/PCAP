#include "runway/instance_io.hpp"
#include "runway/sim_core.hpp"
#include <sstream>
#include <algorithm>

namespace runway {

bool validate_schedule(
    const AircraftData& data,
    const SeparationMatrix& sep,
    const std::vector<int>& perm,
    const NoiseData& noise,
    int scenario_idx,
    std::string* error_msg
) {
    int n = data.n;
    if (static_cast<int>(perm.size()) != n) {
        if (error_msg) *error_msg = "Permutation size does not match aircraft count";
        return false;
    }

    // Check valid permutation
    std::vector<bool> seen(n, false);
    for (int idx : perm) {
        if (idx < 0 || idx >= n) {
            if (error_msg) *error_msg = "Permutation index out of bounds: " + std::to_string(idx);
            return false;
        }
        if (seen[idx]) {
            if (error_msg) *error_msg = "Duplicate index in permutation: " + std::to_string(idx);
            return false;
        }
        seen[idx] = true;
    }

    if (scenario_idx < 0 || scenario_idx >= noise.S) {
        if (error_msg) *error_msg = "Invalid scenario index: " + std::to_string(scenario_idx);
        return false;
    }

    // Compute realized times
    std::vector<float> times(n, 0.0f);
    simulate_one_with_times(
        perm.data(), n, scenario_idx, noise.S,
        data.ready.data(), data.target.data(), data.latest.data(),
        data.weight.data(), data.cls.data(), sep.data,
        noise.delay.data(), noise.mult[scenario_idx],
        noise.closeA[scenario_idx], noise.closeB[scenario_idx],
        times.data(), sep.triangle_ok
    );

    float closeA = noise.closeA[scenario_idx];
    float closeB = noise.closeB[scenario_idx];
    float m_s = noise.mult[scenario_idx];

    // 1. Ready time checks
    for (int k = 0; k < n; ++k) {
        int a = perm[k];
        float min_ready = data.ready[a] + noise.delay[a * noise.S + scenario_idx];
        if (times[k] < min_ready - 1e-3f) {
            if (error_msg) {
                std::stringstream ss;
                ss << "Aircraft " << a << " scheduled at t=" << times[k]
                   << " before realized ready time " << min_ready;
                *error_msg = ss.str();
            }
            return false;
        }

        // 2. Runway closure check: no operation starts inside [closeA, closeB)
        if (closeA < closeB) {
            if (times[k] >= closeA - 1e-4f && times[k] < closeB - 1e-4f) {
                if (error_msg) {
                    std::stringstream ss;
                    ss << "Aircraft " << a << " scheduled at t=" << times[k]
                       << " inside closure interval [" << closeA << ", " << closeB << ")";
                    *error_msg = ss.str();
                }
                return false;
            }
        }
    }

    // 3. Separation check for ALL pairs (i < j)
    for (int i = 0; i < n; ++i) {
        int leader = perm[i];
        int cls_leader = data.cls[leader];
        for (int j = i + 1; j < n; ++j) {
            int follower = perm[j];
            int cls_follower = data.cls[follower];
            float req_sep = m_s * sep.get(cls_leader, cls_follower);
            float actual_sep = times[j] - times[i];
            if (actual_sep < req_sep - 1e-3f) {
                if (error_msg) {
                    std::stringstream ss;
                    ss << "Separation violation between leader " << leader << " (t=" << times[i]
                       << ") and follower " << follower << " (t=" << times[j] << "): actual="
                       << actual_sep << " < required=" << req_sep;
                    *error_msg = ss.str();
                }
                return false;
            }
        }
    }

    return true;
}

} // namespace runway
