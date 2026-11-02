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
    float lambda_std,
    float m_penalty
) {
    int n = data.n;
    int S = noise.S;
    EvalResult res;
    if (n == 0 || S == 0) return res;

    std::vector<float> costs(S, 0.0f);
    int miss_scenarios = 0;
    double sum_cost = 0.0;
    double sum_sq_cost = 0.0;

    // Track emergency aircraft
    std::vector<int> emerg_indices;
    for (int i = 0; i < n; ++i) {
        if (data.prio[i] == PRIO_EMERGENCY) {
            emerg_indices.push_back(i);
        }
    }
    double emerg_delay_sum = 0.0;
    float emerg_delay_max = 0.0f;
    long long emerg_eval_count = 0;

    // Deterministic plan times for plan deviation
    std::vector<float> plan_times(n, 0.0f);
    std::vector<float> zero_delay(n, 0.0f);
    simulate_one_with_times(
        perm.data(), n, 0, 1,
        data.ready.data(), data.target.data(), data.latest.data(),
        data.weight.data(), data.cls.data(), sep.data,
        zero_delay.data(), 1.0f, 0.0f, 0.0f,
        plan_times.data(), sep.triangle_ok, m_penalty
    );

    double total_abs_dev = 0.0;
    std::vector<float> realized_times(n, 0.0f);

    for (int s = 0; s < S; ++s) {
        SimMetrics m = simulate_one_with_times(
            perm.data(), n, s, S,
            data.ready.data(), data.target.data(), data.latest.data(),
            data.weight.data(), data.cls.data(), sep.data,
            noise.delay.data(), noise.mult[s],
            noise.closeA[s], noise.closeB[s],
            realized_times.data(), sep.triangle_ok, m_penalty
        );

        costs[s] = m.cost;
        sum_cost += m.cost;
        sum_sq_cost += static_cast<double>(m.cost) * m.cost;
        if (m.misses > 0) {
            miss_scenarios++;
        }

        for (int k = 0; k < n; ++k) {
            total_abs_dev += std::abs(realized_times[k] - plan_times[k]);
        }

        for (int e_ac : emerg_indices) {
            // Find its time
            for (int k = 0; k < n; ++k) {
                if (perm[k] == e_ac) {
                    float d = std::max(0.0f, realized_times[k] - data.target[e_ac]);
                    emerg_delay_sum += d;
                    if (d > emerg_delay_max) emerg_delay_max = d;
                    emerg_eval_count++;
                    break;
                }
            }
        }
    }

    res.mean_cost = static_cast<float>(sum_cost / S);
    double variance = (sum_sq_cost / S) - (static_cast<double>(res.mean_cost) * res.mean_cost);
    res.std_cost = static_cast<float>(variance > 0.0 ? std::sqrt(variance) : 0.0);
    res.J = res.mean_cost + lambda_std * res.std_cost;
    res.miss_prob = static_cast<float>(miss_scenarios) / S;
    res.plan_deviation = static_cast<float>(total_abs_dev / (static_cast<double>(S) * n));

    if (emerg_eval_count > 0) {
        res.emergency_mean_delay = static_cast<float>(emerg_delay_sum / emerg_eval_count);
        res.emergency_max_delay = emerg_delay_max;
    }

    // p95 and CVaR95
    std::sort(costs.begin(), costs.end());
    int idx95 = static_cast<int>(std::floor(0.95 * (S - 1)));
    res.p95 = costs[idx95];

    double cvar_sum = 0.0;
    int cvar_count = 0;
    for (int i = idx95; i < S; ++i) {
        cvar_sum += costs[i];
        cvar_count++;
    }
    res.cvar95 = (cvar_count > 0) ? static_cast<float>(cvar_sum / cvar_count) : res.p95;

    return res;
}

double evaluate_schedule_cpu_double(
    const AircraftData& data,
    const SeparationMatrix& sep,
    const std::vector<int>& perm,
    const NoiseData& noise,
    float lambda_std,
    float m_penalty
) {
    int n = data.n;
    int S = noise.S;
    if (n == 0 || S == 0) return 0.0;

    double sum_cost = 0.0;
    double sum_sq_cost = 0.0;

    for (int s = 0; s < S; ++s) {
        double lastT[6] = {-1e9, -1e9, -1e9, -1e9, -1e9, -1e9};
        double cost = 0.0;
        double mult_s = static_cast<double>(noise.mult[s]);
        double closeA = static_cast<double>(noise.closeA[s]);
        double closeB = static_cast<double>(noise.closeB[s]);

        for (int k = 0; k < n; ++k) {
            int a = perm[k];
            double t = static_cast<double>(data.ready[a]) + static_cast<double>(noise.delay[a * S + s]);
            int cls_a = data.cls[a];

            double sep_bound = -1e9;
            for (int c = 0; c < 6; ++c) {
                double candidate = lastT[c] + mult_s * static_cast<double>(sep.get(c, cls_a));
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
        sum_sq_cost += cost * cost;
    }

    double mean = sum_cost / S;
    double var = (sum_sq_cost / S) - (mean * mean);
    double std_dev = (var > 0.0) ? std::sqrt(var) : 0.0;
    return mean + static_cast<double>(lambda_std) * std_dev;
}

} // namespace runway
