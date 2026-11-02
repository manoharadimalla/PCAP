#pragma once

#include <cstdint>

#if defined(__CUDACC__)
#define RW_HD __host__ __device__
#define RW_INLINE __forceinline__
#else
#define RW_HD
#define RW_INLINE inline
#endif

namespace runway {

// SplitMix64 generator for high-quality state initialization
struct SplitMix64 {
    uint64_t state;

    RW_HD explicit SplitMix64(uint64_t seed) : state(seed) {}

    RW_HD uint64_t next() {
        uint64_t z = (state += 0x9e3779b97f4a7c15ULL);
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        return z ^ (z >> 31);
    }
};

// XorShift64Star generator: extremely lightweight, fast, and branch-free
struct XorShift64Star {
    uint64_t state;

    RW_HD explicit XorShift64Star(uint64_t seed = 88172645463325252ULL) {
        set_seed(seed);
    }

    RW_HD void set_seed(uint64_t seed) {
        SplitMix64 sm(seed == 0 ? 0xdeadbeefULL : seed);
        state = sm.next();
        if (state == 0) state = 0x853c49e6748fea9bULL;
    }

    RW_HD RW_INLINE uint64_t next_u64() {
        uint64_t x = state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        state = x;
        return x * 0x2545F4914F6CDD1DULL;
    }

    RW_HD RW_INLINE uint32_t next_u32() {
        return static_cast<uint32_t>(next_u64() >> 32);
    }

    // Uniform float in [0, 1)
    RW_HD RW_INLINE float next_float() {
        return (next_u32() >> 8) * (1.0f / 16777216.0f);
    }

    // Uniform integer in [0, bound - 1]
    RW_HD RW_INLINE int next_int(int bound) {
        if (bound <= 1) return 0;
        uint32_t x = next_u32();
        return static_cast<int>((static_cast<uint64_t>(x) * static_cast<uint64_t>(bound)) >> 32);
    }

    // Uniform integer in [low, high] inclusive
    RW_HD RW_INLINE int next_int_range(int low, int high) {
        if (high <= low) return low;
        return low + next_int(high - low + 1);
    }
};

} // namespace runway
