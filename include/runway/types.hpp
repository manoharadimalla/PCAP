#pragma once

#include <vector>
#include <string>
#include <cstdint>
#include <cmath>

namespace runway {

enum Operation : int {
    OP_ARR = 0,
    OP_DEP = 1
};

enum Wake : int {
    WAKE_HEAVY = 0,
    WAKE_MEDIUM = 1,
    WAKE_LIGHT = 2
};

enum Priority : int {
    PRIO_NORMAL = 0,
    PRIO_HIGH = 1,
    PRIO_EMERGENCY = 2
};

inline int make_class(int op, int wake) {
    return op * 3 + wake;
}

struct Aircraft {
    int id;
    int op;
    int wake;
    int cls;
    float ready;
    float target;
    float latest;
    int prio;
    float weight;
};

// Structure of Arrays representation
struct AircraftData {
    int n = 0;
    std::vector<int> id;
    std::vector<int> op;
    std::vector<int> wake;
    std::vector<int> cls;
    std::vector<float> ready;
    std::vector<float> target;
    std::vector<float> latest;
    std::vector<int> prio;
    std::vector<float> weight;

    void clear() {
        n = 0;
        id.clear();
        op.clear();
        wake.clear();
        cls.clear();
        ready.clear();
        target.clear();
        latest.clear();
        prio.clear();
        weight.clear();
    }

    void add(const Aircraft& ac) {
        id.push_back(ac.id);
        op.push_back(ac.op);
        wake.push_back(ac.wake);
        cls.push_back(ac.cls);
        ready.push_back(ac.ready);
        target.push_back(ac.target);
        latest.push_back(ac.latest);
        prio.push_back(ac.prio);
        weight.push_back(ac.weight);
        n = static_cast<int>(ready.size());
    }
};

struct SeparationMatrix {
    static constexpr int NUM_CLASSES = 6;
    float data[NUM_CLASSES * NUM_CLASSES] = {0};
    bool triangle_ok = true;

    float get(int leader, int follower) const {
        return data[leader * NUM_CLASSES + follower];
    }

    void set(int leader, int follower, float val) {
        data[leader * NUM_CLASSES + follower] = val;
    }

    void check_triangle_inequality() {
        triangle_ok = true;
        for (int a = 0; a < NUM_CLASSES; ++a) {
            for (int b = 0; b < NUM_CLASSES; ++b) {
                for (int c = 0; c < NUM_CLASSES; ++c) {
                    if (get(a, c) > get(a, b) + get(b, c) + 1e-4f) {
                        triangle_ok = false;
                        return;
                    }
                }
            }
        }
    }
};

struct NoiseData {
    int S = 0; // Number of scenarios
    int n = 0; // Number of aircraft
    // Layout: aircraft-major -> delay[a * S + s]
    std::vector<float> delay;
    std::vector<float> mult;
    std::vector<float> closeA;
    std::vector<float> closeB;

    void resize(int num_aircraft, int num_scenarios) {
        n = num_aircraft;
        S = num_scenarios;
        delay.assign(n * S, 0.0f);
        mult.assign(S, 1.0f);
        closeA.assign(S, 0.0f);
        closeB.assign(S, 0.0f);
    }
};

struct SimMetrics {
    float cost = 0.0f;
    int misses = 0;
};

struct ObjectiveWeights {
    float lambda_std = 1.0f;
    float m_penalty = 5.0f;
};

struct EvalResult {
    float J = 0.0f;
    float mean_cost = 0.0f;
    float std_cost = 0.0f;
    float p95 = 0.0f;
    float cvar95 = 0.0f;
    float miss_prob = 0.0f;
    float emergency_mean_delay = 0.0f;
    float emergency_max_delay = 0.0f;
    float plan_deviation = 0.0f;
    std::vector<float> per_scenario_costs;
};

struct ConvergencePoint {
    int epoch = 0;
    float best_J = 0.0f;
    double elapsed_s = 0.0;
};

struct SAResult {
    std::vector<int> best_perm;
    float in_sample_J = 0.0f;
    EvalResult out_of_sample_eval;
    double kernel_time_ms = 0.0;
    double total_time_ms = 0.0;
    double sim_per_sec = 0.0;
    int epochs = 0;
    int iters_per_epoch = 0;
    int chains = 0;
    int scenarios = 0;
    std::vector<ConvergencePoint> convergence;
};

} // namespace runway
