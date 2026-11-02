#pragma once

#include "types.hpp"
#include <string>
#include <vector>

namespace runway {

bool load_separation_matrix(const std::string& path, SeparationMatrix& mat);
bool save_separation_matrix(const std::string& path, const SeparationMatrix& mat);
SeparationMatrix get_default_separation_matrix();

bool load_instance(const std::string& path, AircraftData& data);
bool save_instance(const std::string& path, const AircraftData& data);

bool load_schedule(const std::string& path, std::vector<int>& perm);
bool save_schedule(const std::string& path, const std::vector<int>& perm);

bool validate_schedule(
    const AircraftData& data,
    const SeparationMatrix& sep,
    const std::vector<int>& perm,
    const NoiseData& noise,
    int scenario_idx,
    std::string* error_msg = nullptr
);

EvalResult evaluate_schedule_cpu(
    const AircraftData& data,
    const SeparationMatrix& sep,
    const std::vector<int>& perm,
    const NoiseData& noise,
    float lambda_std = 1.0f,
    float m_penalty = 5.0f
);

// High-precision double evaluator for testing R10/verification
double evaluate_schedule_cpu_double(
    const AircraftData& data,
    const SeparationMatrix& sep,
    const std::vector<int>& perm,
    const NoiseData& noise,
    float lambda_std = 1.0f,
    float m_penalty = 5.0f
);

// Instance generator
AircraftData generate_instance(
    int n,
    float rho,
    uint64_t seed,
    float mix_arr = 0.5f,
    float wake_h = 0.2f,
    float wake_m = 0.6f,
    float wake_l = 0.2f,
    float prio_normal = 0.80f,
    float prio_high = 0.15f,
    float prio_emergency = 0.05f,
    const SeparationMatrix* custom_sep = nullptr
);

// Noise generator
void generate_noise_cpu(
    NoiseData& noise,
    const AircraftData& data,
    int S,
    uint64_t seed,
    float p_close = 0.20f
);

// Baselines
std::vector<int> solve_fcfs(const AircraftData& data);

SAResult solve_random_search(
    const AircraftData& data,
    const SeparationMatrix& sep,
    int S,
    int num_samples,
    uint64_t seed,
    float lambda_std = 1.0f,
    float m_penalty = 5.0f
);

SAResult solve_brute_force(
    const AircraftData& data,
    const SeparationMatrix& sep,
    const NoiseData& noise,
    float lambda_std = 1.0f,
    float m_penalty = 5.0f
);

// Solvers
SAResult solve_sa_serial(
    const AircraftData& data,
    const SeparationMatrix& sep,
    int C,
    int S,
    int E,
    int I,
    uint64_t seed,
    float lambda_std = 1.0f,
    float m_penalty = 5.0f
);

#if defined(RUNWAY_HAS_OPENMP)
SAResult solve_sa_omp(
    const AircraftData& data,
    const SeparationMatrix& sep,
    int C,
    int S,
    int E,
    int I,
    uint64_t seed,
    const std::string& mode = "chains", // "chains", "scenarios", "hybrid"
    int num_threads = 0,
    float lambda_std = 1.0f,
    float m_penalty = 5.0f
);
#endif

#if defined(RUNWAY_HAS_CUDA)
SAResult solve_sa_cuda(
    const AircraftData& data,
    const SeparationMatrix& sep,
    int C,
    int S,
    int E,
    int I,
    uint64_t seed,
    const std::string& kernel_name = "k3", // "k1", "k2", "k3"
    int block_size = 256,
    float lambda_std = 1.0f,
    float m_penalty = 5.0f
);
#endif

} // namespace runway
