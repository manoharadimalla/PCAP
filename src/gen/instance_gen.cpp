#include "runway/instance_io.hpp"
#include "runway/rng.hpp"
#include <vector>
#include <cmath>
#include <algorithm>

namespace runway {

AircraftData generate_instance(
    int n,
    float rho,
    uint64_t seed,
    float mix_arr,
    float wake_h,
    float wake_m,
    float wake_l,
    float prio_normal,
    float prio_high,
    float prio_emergency,
    const SeparationMatrix* custom_sep
) {
    SeparationMatrix sep = custom_sep ? *custom_sep : get_default_separation_matrix();
    XorShift64Star rng(seed);

    // Compute class probabilities
    float class_prob[6];
    float wake_probs[3] = {wake_h, wake_m, wake_l};
    for (int op = 0; op < 2; ++op) {
        float op_p = (op == OP_ARR) ? mix_arr : (1.0f - mix_arr);
        for (int w = 0; w < 3; ++w) {
            class_prob[op * 3 + w] = op_p * wake_probs[w];
        }
    }

    // Weighted mean separation
    float mean_sep = 0.0f;
    for (int a = 0; a < 6; ++a) {
        for (int b = 0; b < 6; ++b) {
            mean_sep += class_prob[a] * class_prob[b] * sep.get(a, b);
        }
    }
    if (mean_sep <= 0.0f) mean_sep = 60.0f;
    if (rho <= 0.0f) rho = 1.0f;
    float mean_gap = mean_sep / rho;

    std::vector<Aircraft> planes(n);
    float cur_time = 0.0f;

    for (int i = 0; i < n; ++i) {
        planes[i].id = i;

        // Operation
        planes[i].op = (rng.next_float() < mix_arr) ? OP_ARR : OP_DEP;

        // Wake
        float r_wake = rng.next_float();
        if (r_wake < wake_h) {
            planes[i].wake = WAKE_HEAVY;
        } else if (r_wake < wake_h + wake_m) {
            planes[i].wake = WAKE_MEDIUM;
        } else {
            planes[i].wake = WAKE_LIGHT;
        }
        planes[i].cls = make_class(planes[i].op, planes[i].wake);

        // Priority
        float r_prio = rng.next_float();
        if (r_prio < prio_normal) {
            planes[i].prio = PRIO_NORMAL;
            planes[i].weight = 1.0f;
        } else if (r_prio < prio_normal + prio_high) {
            planes[i].prio = PRIO_HIGH;
            planes[i].weight = 3.0f;
        } else {
            planes[i].prio = PRIO_EMERGENCY;
            planes[i].weight = 20.0f;
        }

        // Exponential inter-arrival gap
        float u = std::max(1e-7f, rng.next_float());
        float gap = -mean_gap * std::log(1.0f - u);
        cur_time += gap;
        planes[i].target = cur_time;
    }

    // Sort by target time to produce realistic sequence of arrival demands
    std::sort(planes.begin(), planes.end(), [](const Aircraft& a, const Aircraft& b) {
        return a.target < b.target;
    });

    // Re-index ids and assign ready & latest times
    float min_ready = 1e9f;
    for (int i = 0; i < n; ++i) {
        planes[i].id = i;
        if (planes[i].op == OP_ARR) {
            float advance = rng.next_float() * 120.0f; // U(0, 120)
            planes[i].ready = planes[i].target - advance;
            planes[i].latest = planes[i].target + 600.0f;
        } else {
            planes[i].ready = planes[i].target;
            planes[i].latest = planes[i].target + 900.0f;
        }
        if (planes[i].ready < min_ready) {
            min_ready = planes[i].ready;
        }
    }

    // Shift so min(ready) >= 0
    float shift = (min_ready < 0.0f) ? -min_ready : 0.0f;
    AircraftData data;
    for (int i = 0; i < n; ++i) {
        planes[i].ready += shift;
        planes[i].target += shift;
        planes[i].latest += shift;
        data.add(planes[i]);
    }

    return data;
}

} // namespace runway
