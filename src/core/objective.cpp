#include "runway/instance_io.hpp"
#include "runway/sim_core.hpp"
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>

namespace runway {

EvalResult evaluate_schedule_cpu(
    const AircraftData& data,
    const SeparationMatrix& sep,
    const std::vector<int>& perm,
    const NoiseData& noise,
    float m_penalty
) {
    int n = data.n;
    int S = noise.S;
    EvalResult res;
    if (n == 0 || S == 0) return res;

    std::vector<float> costs(S, 0.0f);
    double sum_cost = 0.0;
    long long total_late_aircraft = 0;

    // Track emergency aircraft
    std::vector<int> emerg_indices;
    for (int i = 0; i < n; ++i) {
        if (data.prio[i] == PRIO_EMERGENCY) {
            emerg_indices.push_back(i);
        }
    }
    double emerg_delay_sum = 0.0;
    long long emerg_eval_count = 0;

    std::vector<float> realized_times(n, 0.0f);

    for (int s = 0; s < S; ++s) {
        SimMetrics m = simulate_one_with_times(
            perm.data(), n, s, S,
            data.ready.data(), data.target.data(), data.latest.data(),
            data.weight.data(), data.cls.data(), sep.data,
            noise.delay.data(),
            noise.closeA[s], noise.closeB[s],
            realized_times.data(), m_penalty
        );

        costs[s] = m.cost;
        sum_cost += m.cost;
        total_late_aircraft += m.misses;

        for (int e_ac : emerg_indices) {
            for (int k = 0; k < n; ++k) {
                if (perm[k] == e_ac) {
                    float d = std::max(0.0f, realized_times[k] - data.target[e_ac]);
                    emerg_delay_sum += d;
                    emerg_eval_count++;
                    break;
                }
            }
        }
    }

    res.mean_cost = static_cast<float>(sum_cost / S);
    res.J = res.mean_cost;
    res.late_pct = static_cast<float>(static_cast<double>(total_late_aircraft) / (static_cast<double>(S) * n));

    if (emerg_eval_count > 0) {
        res.emergency_mean_delay = static_cast<float>(emerg_delay_sum / emerg_eval_count);
    }

    res.per_scenario_costs = costs;

    // p95 cost
    std::sort(costs.begin(), costs.end());
    int idx95 = static_cast<int>(std::floor(0.95 * (S - 1)));
    res.p95 = costs[idx95];

    return res;
}

double evaluate_schedule_cpu_double(
    const AircraftData& data,
    const SeparationMatrix& sep,
    const std::vector<int>& perm,
    const NoiseData& noise,
    float m_penalty
) {
    int n = data.n;
    int S = noise.S;
    if (n == 0 || S == 0) return 0.0;

    double sum_cost = 0.0;

    for (int s = 0; s < S; ++s) {
        double lastT[6] = {-1e9, -1e9, -1e9, -1e9, -1e9, -1e9};
        double cost = 0.0;
        double closeA = static_cast<double>(noise.closeA[s]);
        double closeB = static_cast<double>(noise.closeB[s]);

        for (int k = 0; k < n; ++k) {
            int a = perm[k];
            double t = static_cast<double>(data.ready[a]) + static_cast<double>(noise.delay[a * S + s]);
            int cls_a = data.cls[a];

            double sep_bound = -1e9;
            for (int c = 0; c < 6; ++c) {
                double candidate = lastT[c] + static_cast<double>(sep.get(c, cls_a));
                if (candidate > sep_bound) sep_bound = candidate;
            }
            if (sep_bound > t) t = sep_bound;

            if (t >= closeA && t < closeB) {
                t = closeB;
            }
            lastT[cls_a] = t;

            double d = t - static_cast<double>(data.target[a]);
            if (d > 0.0) cost += static_cast<double>(data.weight[a]) * d;
            if (t > static_cast<double>(data.latest[a])) {
                cost += static_cast<double>(m_penalty) * static_cast<double>(data.weight[a]) * (t - static_cast<double>(data.latest[a]));
            }
        }
        sum_cost += cost;
    }

    return sum_cost / S;
}

} // namespace runway
