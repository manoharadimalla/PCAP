#pragma once

#include "types.hpp"
#include "rng.hpp"

#if defined(__CUDACC__)
#define RW_HD __host__ __device__
#define RW_INLINE __forceinline__
#else
#define RW_HD
#define RW_INLINE inline
#endif

namespace runway {

// Exact decoder simulation of one scenario
RW_HD RW_INLINE SimMetrics simulate_one_exact(
    const int* __restrict__ perm,
    int n,
    int s,
    int S,
    const float* __restrict__ ready,
    const float* __restrict__ target,
    const float* __restrict__ latest,
    const float* __restrict__ weight,
    const int* __restrict__ cls,
    const float* __restrict__ sep_matrix, // 6x6
    const float* __restrict__ delay,      // layout: [a * S + s]
    float mult_s,
    float closeA_s,
    float closeB_s,
    float m_penalty = 5.0f
) {
    float lastT[6] = {-1e9f, -1e9f, -1e9f, -1e9f, -1e9f, -1e9f};
    SimMetrics m;
    m.cost = 0.0f;
    m.misses = 0;

    for (int k = 0; k < n; ++k) {
        int a = perm[k];
        float t = ready[a] + delay[a * S + s];
        int cls_a = cls[a];

        float sep_bound = -1e9f;
        #if defined(__CUDA_ARCH__)
        #pragma unroll
        #endif
        for (int c = 0; c < 6; ++c) {
            float candidate = lastT[c] + mult_s * sep_matrix[c * 6 + cls_a];
            if (candidate > sep_bound) {
                sep_bound = candidate;
            }
        }
        if (sep_bound > t) {
            t = sep_bound;
        }

        // Runway closure window [closeA, closeB)
        // Predicated select
        t = (t >= closeA_s && t < closeB_s) ? closeB_s : t;

        lastT[cls_a] = t;

        float d = t - target[a];
        if (d > 0.0f) {
            m.cost += weight[a] * d;
        }
        if (t > latest[a]) {
            m.cost += m_penalty * weight[a] * (t - latest[a]);
            m.misses += 1;
        }
    }
    return m;
}

// Fast-path decoder (valid when triangle inequality holds)
RW_HD RW_INLINE SimMetrics simulate_one_fast(
    const int* __restrict__ perm,
    int n,
    int s,
    int S,
    const float* __restrict__ ready,
    const float* __restrict__ target,
    const float* __restrict__ latest,
    const float* __restrict__ weight,
    const int* __restrict__ cls,
    const float* __restrict__ sep_matrix, // 6x6
    const float* __restrict__ delay,      // layout: [a * S + s]
    float mult_s,
    float closeA_s,
    float closeB_s,
    float m_penalty = 5.0f
) {
    SimMetrics m;
    m.cost = 0.0f;
    m.misses = 0;
    float prev_t = -1e9f;
    int prev_cls = -1;

    for (int k = 0; k < n; ++k) {
        int a = perm[k];
        float t = ready[a] + delay[a * S + s];
        int cls_a = cls[a];

        if (prev_cls >= 0) {
            float candidate = prev_t + mult_s * sep_matrix[prev_cls * 6 + cls_a];
            if (candidate > t) {
                t = candidate;
            }
        }

        // Runway closure window [closeA, closeB)
        t = (t >= closeA_s && t < closeB_s) ? closeB_s : t;

        prev_t = t;
        prev_cls = cls_a;

        float d = t - target[a];
        if (d > 0.0f) {
            m.cost += weight[a] * d;
        }
        if (t > latest[a]) {
            m.cost += m_penalty * weight[a] * (t - latest[a]);
            m.misses += 1;
        }
    }
    return m;
}

// Generic single simulation dispatcher
RW_HD RW_INLINE SimMetrics simulate_one(
    const int* __restrict__ perm,
    int n,
    int s,
    int S,
    const float* __restrict__ ready,
    const float* __restrict__ target,
    const float* __restrict__ latest,
    const float* __restrict__ weight,
    const int* __restrict__ cls,
    const float* __restrict__ sep_matrix,
    const float* __restrict__ delay,
    float mult_s,
    float closeA_s,
    float closeB_s,
    bool triangle_ok = false,
    float m_penalty = 5.0f
) {
    if (triangle_ok) {
        return simulate_one_fast(
            perm, n, s, S, ready, target, latest, weight, cls,
            sep_matrix, delay, mult_s, closeA_s, closeB_s, m_penalty
        );
    } else {
        return simulate_one_exact(
            perm, n, s, S, ready, target, latest, weight, cls,
            sep_matrix, delay, mult_s, closeA_s, closeB_s, m_penalty
        );
    }
}

// Full simulation recording realized runway times
RW_HD inline SimMetrics simulate_one_with_times(
    const int* perm,
    int n,
    int s,
    int S,
    const float* ready,
    const float* target,
    const float* latest,
    const float* weight,
    const int* cls,
    const float* sep_matrix,
    const float* delay,
    float mult_s,
    float closeA_s,
    float closeB_s,
    float* out_times,
    bool triangle_ok = false,
    float m_penalty = 5.0f
) {
    float lastT[6] = {-1e9f, -1e9f, -1e9f, -1e9f, -1e9f, -1e9f};
    SimMetrics m;
    m.cost = 0.0f;
    m.misses = 0;
    float prev_t = -1e9f;
    int prev_cls = -1;

    for (int k = 0; k < n; ++k) {
        int a = perm[k];
        float t = ready[a] + delay[a * S + s];
        int cls_a = cls[a];

        if (triangle_ok) {
            if (prev_cls >= 0) {
                float candidate = prev_t + mult_s * sep_matrix[prev_cls * 6 + cls_a];
                if (candidate > t) t = candidate;
            }
        } else {
            float sep_bound = -1e9f;
            for (int c = 0; c < 6; ++c) {
                float candidate = lastT[c] + mult_s * sep_matrix[c * 6 + cls_a];
                if (candidate > sep_bound) sep_bound = candidate;
            }
            if (sep_bound > t) t = sep_bound;
        }

        t = (t >= closeA_s && t < closeB_s) ? closeB_s : t;

        lastT[cls_a] = t;
        prev_t = t;
        prev_cls = cls_a;
        if (out_times) {
            out_times[k] = t;
        }

        float d = t - target[a];
        if (d > 0.0f) m.cost += weight[a] * d;
        if (t > latest[a]) {
            m.cost += m_penalty * weight[a] * (t - latest[a]);
            m.misses += 1;
        }
    }
    return m;
}

// Propose a neighborhood move: Swap (40%), Insert (40%), Adjacent-Swap (20%)
RW_HD RW_INLINE void propose_move(
    const int* __restrict__ src,
    int* __restrict__ dst,
    int n,
    XorShift64Star& rng
) {
    #if defined(__CUDA_ARCH__)
    #pragma unroll 4
    #endif
    for (int i = 0; i < n; ++i) {
        dst[i] = src[i];
    }
    if (n < 2) return;

    float r = rng.next_float();
    if (n == 2 || r < 0.40f) {
        // Swap
        int i = rng.next_int(n);
        int j = rng.next_int(n - 1);
        if (j >= i) j += 1;
        int tmp = dst[i];
        dst[i] = dst[j];
        dst[j] = tmp;
    } else if (r < 0.80f) {
        // Insert
        int i = rng.next_int(n);
        int j = rng.next_int(n - 1);
        if (j >= i) j += 1;
        int val = dst[i];
        if (i < j) {
            for (int k = i; k < j; ++k) {
                dst[k] = dst[k + 1];
            }
        } else {
            for (int k = i; k > j; --k) {
                dst[k] = dst[k - 1];
            }
        }
        dst[j] = val;
    } else {
        // Adjacent-swap
        int i = rng.next_int(n - 1);
        int tmp = dst[i];
        dst[i] = dst[i + 1];
        dst[i + 1] = tmp;
    }
}

} // namespace runway
