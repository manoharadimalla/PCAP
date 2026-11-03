#include "runway/instance_io.hpp"
#include "runway/rng.hpp"
#include <cmath>
#include <algorithm>

namespace runway {

void generate_noise_cpu(
    NoiseData& noise,
    const AircraftData& data,
    int S,
    uint64_t seed,
    float p_close
) {
    int n = data.n;
    noise.resize(n, S);

    if (n == 0 || S == 0) return;

    float t_min = 1e9f;
    float t_max = -1e9f;
    for (int i = 0; i < n; ++i) {
        if (data.target[i] < t_min) t_min = data.target[i];
        if (data.target[i] > t_max) t_max = data.target[i];
    }
    if (t_max <= t_min) t_max = t_min + 1200.0f;

    XorShift64Star rng(seed);

    for (int s = 0; s < S; ++s) {
        // Runway closure window
        float r_c = rng.next_float();
        if (r_c < p_close) {
            float dur = 300.0f + rng.next_float() * 600.0f; // U(300, 900)
            float max_start = t_max - dur;
            float start = (max_start > t_min) ? (t_min + rng.next_float() * (max_start - t_min)) : t_min;
            noise.closeA[s] = start;
            noise.closeB[s] = start + dur;
        } else {
            noise.closeA[s] = 0.0f;
            noise.closeB[s] = 0.0f;
        }

        // Delay for each aircraft in scenario s
        for (int a = 0; a < n; ++a) {
            float mean_d = (data.op[a] == OP_ARR) ? 30.0f : 60.0f;
            float u = std::max(1e-7f, rng.next_float());
            float d = -mean_d * std::log(1.0f - u);
            noise.delay[a * S + s] = d;
        }
    }
}

} // namespace runway
