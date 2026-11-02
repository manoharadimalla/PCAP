#include "runway/instance_io.hpp"
#include "runway/sim_core.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <chrono>
#include <iomanip>
#include <unordered_set>
#include <filesystem>
#include <cctype>
#include <algorithm>

#if defined(RUNWAY_HAS_OPENMP)
#include <omp.h>
#endif

using namespace runway;

static std::string int_array_to_json(const std::vector<int>& vec) {
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < vec.size(); ++i) {
        if (i > 0) oss << ", ";
        oss << vec[i];
    }
    oss << "]";
    return oss.str();
}

static std::string float_array_to_json(const std::vector<float>& vec, size_t limit = (size_t)-1) {
    std::ostringstream oss;
    oss << "[";
    size_t count = std::min(limit, vec.size());
    for (size_t i = 0; i < count; ++i) {
        if (i > 0) oss << ", ";
        oss << vec[i];
    }
    oss << "]";
    return oss.str();
}

static std::string convergence_to_json(const std::vector<ConvergencePoint>& pts) {
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < pts.size(); ++i) {
        if (i > 0) oss << ", ";
        oss << "{\"epoch\": " << pts[i].epoch
            << ", \"best_J\": " << pts[i].best_J
            << ", \"elapsed_s\": " << pts[i].elapsed_s << "}";
    }
    oss << "]";
    return oss.str();
}

static std::string eval_metrics_to_json(const EvalResult& r) {
    std::ostringstream oss;
    oss << "{\"J\": " << r.J
        << ", \"mean_cost\": " << r.mean_cost
        << ", \"std_cost\": " << r.std_cost
        << ", \"p95\": " << r.p95
        << ", \"cvar95\": " << r.cvar95
        << ", \"miss_prob\": " << r.miss_prob
        << ", \"emergency_mean_delay\": " << r.emergency_mean_delay
        << ", \"emergency_max_delay\": " << r.emergency_max_delay
        << ", \"plan_dev\": " << r.plan_deviation << "}";
    return oss.str();
}

void print_usage() {
    std::cout << "RunwayMC — GPU-Accelerated Robust Runway Scheduling under Uncertainty\n"
              << "\n"
              << "Mandatory Disclaimer:\n"
              << "  This is an academic simulation with simplified separation standards.\n"
              << "  It is not validated for, and must not be used in, real air-traffic operations.\n\n"
              << "Commands:\n"
              << "  runway info\n"
              << "  runway validate --instance <file> [--separation <file.csv>]\n"
              << "  runway replay   --instance <file> --perm <file>|fcfs --scenario-index <int> --seed <int>\n"
              << "                  [--separation <file.csv>] [--penalty <float>] [--p-close <float>]\n"
              << "  runway gen      --n <int> --rho <float> --seed <int> --out <file>\n"
              << "  runway eval     --instance <file> --perm fcfs|<file> --scenarios <int> --seed <int>\n"
              << "  runway solve    --instance <file> --impl serial|omp|cuda|random|brute\n"
              << "                  --chains <int> --scenarios <int> --iters <int> --epochs <int>\n"
              << "                  [--lambda <float>] [--penalty <float>] [--seed <int>]\n"
              << "                  [--threads <int>] [--omp-mode chains|scenarios|hybrid]\n"
              << "                  [--block <int>] [--kernel k1|k2|k3] [--out <json/csv>]\n"
              << "                  [--progress] [--dump-costs <int>] [--separation <file.csv>]\n"
              << "                  [--s-final <int>]\n"
              << "  runway final    --instance <file> --perm <file> --scenarios <int> --seed <int>\n";
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        print_usage();
        return 1;
    }

    std::string cmd = argv[1];

    if (cmd == "info") {
        bool cuda_avail = false;
        std::string gpu_name = "N/A";
        int gpu_sms = 0;
#if defined(RUNWAY_HAS_CUDA)
        cuda_avail = true;
#endif

        int omp_max_threads = 1;
#if defined(RUNWAY_HAS_OPENMP)
        omp_max_threads = omp_get_max_threads();
#endif

        std::cout << "{\"version\": \"1.0.0\", \"cuda_available\": " << (cuda_avail ? "true" : "false")
                  << ", \"gpu_name\": \"" << gpu_name << "\", \"gpu_sms\": " << gpu_sms
                  << ", \"omp_max_threads\": " << omp_max_threads << "}\n";
        return 0;
    } else if (cmd == "validate") {
        std::string inst_file;
        std::string sep_file;

        for (int i = 2; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "--instance" && i + 1 < argc) inst_file = argv[++i];
            else if (arg == "--separation" && i + 1 < argc) sep_file = argv[++i];
        }

        struct ValidationError {
            int row;
            std::string column;
            std::string message;
        };

        std::vector<ValidationError> errors;
        std::vector<std::string> warnings;
        bool triangle_ok = true;

        if (!sep_file.empty()) {
            SeparationMatrix sep;
            if (!load_separation_matrix(sep_file, sep)) {
                errors.push_back({0, "separation", "Failed to load separation matrix from " + sep_file});
                triangle_ok = false;
            } else {
                triangle_ok = sep.triangle_ok;
                if (!triangle_ok) {
                    warnings.push_back("Separation matrix violates triangle inequality");
                }
            }
        }

        if (inst_file.empty()) {
            errors.push_back({0, "instance", "Missing --instance argument"});
        } else {
            std::ifstream fin(inst_file);
            if (!fin.is_open()) {
                errors.push_back({0, "instance", "Cannot open instance file: " + inst_file});
            } else {
                std::string line;
                if (!std::getline(fin, line)) {
                    errors.push_back({0, "instance", "Instance file is empty"});
                } else {
                    int data_row = 0;
                    std::unordered_set<int> seen_ids;
                    int valid_aircraft_count = 0;

                    while (std::getline(fin, line)) {
                        if (line.empty() || line[0] == '#') continue;

                        std::stringstream ss(line);
                        std::string token;
                        std::vector<std::string> parts;
                        while (std::getline(ss, token, ',')) {
                            size_t first = token.find_first_not_of(" \t\r\n");
                            size_t last = token.find_last_not_of(" \t\r\n");
                            if (first == std::string::npos) parts.push_back("");
                            else parts.push_back(token.substr(first, last - first + 1));
                        }

                        if (parts.size() < 8) {
                            errors.push_back({data_row, "row", "Expected at least 8 columns, got " + std::to_string(parts.size())});
                            data_row++;
                            continue;
                        }

                        // 1. id
                        int ac_id = 0;
                        bool id_ok = false;
                        try {
                            ac_id = std::stoi(parts[0]);
                            id_ok = true;
                        } catch (...) {
                            errors.push_back({data_row, "id", "Invalid integer for id: " + parts[0]});
                        }
                        if (id_ok) {
                            if (seen_ids.count(ac_id)) {
                                errors.push_back({data_row, "id", "Duplicate aircraft ID: " + std::to_string(ac_id)});
                            } else {
                                seen_ids.insert(ac_id);
                            }
                        }

                        // 2. op (0, 1, ARR, DEP)
                        std::string op_str = parts[1];
                        std::string op_upper = op_str;
                        for (char& c : op_upper) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
                        if (op_upper != "0" && op_upper != "1" && op_upper != "ARR" && op_upper != "DEP") {
                            errors.push_back({data_row, "op", "Invalid op value '" + op_str + "': must be 0, 1, ARR, or DEP"});
                        }

                        // 3. wake (0, 1, 2, H, M, L, Heavy, Medium, Light)
                        std::string wake_str = parts[2];
                        std::string wake_upper = wake_str;
                        for (char& c : wake_upper) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
                        if (wake_upper != "0" && wake_upper != "1" && wake_upper != "2" &&
                            wake_upper != "H" && wake_upper != "M" && wake_upper != "L" &&
                            wake_upper != "HEAVY" && wake_upper != "MEDIUM" && wake_upper != "LIGHT") {
                            errors.push_back({data_row, "wake", "Invalid wake value '" + wake_str + "': must be 0, 1, 2, H, M, or L"});
                        }

                        // 4. ready, target, latest
                        float ready = 0.0f, target = 0.0f, latest = 0.0f;
                        bool ready_ok = false, target_ok = false, latest_ok = false;
                        try {
                            ready = std::stof(parts[3]);
                            ready_ok = true;
                        } catch (...) {
                            errors.push_back({data_row, "ready", "Invalid float for ready: " + parts[3]});
                        }
                        try {
                            target = std::stof(parts[4]);
                            target_ok = true;
                        } catch (...) {
                            errors.push_back({data_row, "target", "Invalid float for target: " + parts[4]});
                        }
                        try {
                            latest = std::stof(parts[5]);
                            latest_ok = true;
                        } catch (...) {
                            errors.push_back({data_row, "latest", "Invalid float for latest: " + parts[5]});
                        }

                        if (ready_ok && target_ok && ready > target) {
                            errors.push_back({data_row, "ready", "ready (" + parts[3] + ") must be <= target (" + parts[4] + ")"});
                        }
                        if (target_ok && latest_ok && target > latest) {
                            errors.push_back({data_row, "target", "target (" + parts[4] + ") must be <= latest (" + parts[5] + ")"});
                        }
                        if (ready_ok && latest_ok && ready > latest && target <= latest) {
                            errors.push_back({data_row, "ready", "ready (" + parts[3] + ") must be <= latest (" + parts[5] + ")"});
                        }

                        // 5. priority (0, 1, 2, Normal, High, Emergency)
                        std::string prio_str = parts[6];
                        std::string prio_upper = prio_str;
                        for (char& c : prio_upper) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
                        if (prio_upper != "0" && prio_upper != "1" && prio_upper != "2" &&
                            prio_upper != "NORMAL" && prio_upper != "HIGH" && prio_upper != "EMERGENCY") {
                            errors.push_back({data_row, "priority", "Invalid priority value '" + prio_str + "': must be 0, 1, 2, Normal, High, or Emergency"});
                        }

                        // 6. weight (> 0)
                        float weight = 0.0f;
                        bool weight_ok = false;
                        try {
                            weight = std::stof(parts[7]);
                            weight_ok = true;
                        } catch (...) {
                            errors.push_back({data_row, "weight", "Invalid float for weight: " + parts[7]});
                        }
                        if (weight_ok && weight <= 0.0f) {
                            errors.push_back({data_row, "weight", "Weight must be > 0 (got " + parts[7] + ")"});
                        }

                        valid_aircraft_count++;
                        data_row++;
                    }

                    if (valid_aircraft_count < 1) {
                        errors.push_back({0, "instance", "Instance must contain at least 1 aircraft (n >= 1)"});
                    }
                }
            }
        }

        bool ok = errors.empty();
        std::cout << "{\"ok\": " << (ok ? "true" : "false")
                  << ", \"errors\": [";
        for (size_t i = 0; i < errors.size(); ++i) {
            if (i > 0) std::cout << ", ";
            std::cout << "{\"row\": " << errors[i].row
                      << ", \"column\": \"" << errors[i].column << "\""
                      << ", \"message\": \"" << errors[i].message << "\"}";
        }
        std::cout << "], \"warnings\": [";
        for (size_t i = 0; i < warnings.size(); ++i) {
            if (i > 0) std::cout << ", ";
            std::cout << "\"" << warnings[i] << "\"";
        }
        std::cout << "], \"triangle_ok\": " << (triangle_ok ? "true" : "false") << "}\n";

        return ok ? 0 : 1;
    } else if (cmd == "replay") {
        std::string inst_file;
        std::string perm_arg = "fcfs";
        int scenario_idx = 0;
        uint64_t seed = 1;
        std::string sep_file;
        float m_penalty = 5.0f;
        float p_close = 0.20f;

        for (int i = 2; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "--instance" && i + 1 < argc) inst_file = argv[++i];
            else if (arg == "--perm" && i + 1 < argc) perm_arg = argv[++i];
            else if (arg == "--scenario-index" && i + 1 < argc) scenario_idx = std::stoi(argv[++i]);
            else if (arg == "--seed" && i + 1 < argc) seed = std::stoull(argv[++i]);
            else if (arg == "--separation" && i + 1 < argc) sep_file = argv[++i];
            else if (arg == "--penalty" && i + 1 < argc) m_penalty = std::stof(argv[++i]);
            else if (arg == "--p-close" && i + 1 < argc) p_close = std::stof(argv[++i]);
        }

        AircraftData data;
        if (!load_instance(inst_file, data)) return 1;

        SeparationMatrix sep;
        if (!sep_file.empty()) {
            if (!load_separation_matrix(sep_file, sep)) {
                std::cerr << "Failed to load separation matrix from " << sep_file << "\n";
                return 1;
            }
        } else {
            sep = get_default_separation_matrix();
        }

        std::vector<int> perm;
        if (perm_arg == "fcfs") {
            perm = solve_fcfs(data);
        } else {
            if (!load_schedule(perm_arg, perm)) {
                std::cerr << "Failed to load schedule from " << perm_arg << "\n";
                return 1;
            }
        }

        if (scenario_idx < 0) {
            std::cerr << "Invalid scenario index: " << scenario_idx << "\n";
            return 1;
        }

        int S = scenario_idx + 1;
        NoiseData noise;
        generate_noise_cpu(noise, data, S, seed, p_close);

        std::vector<float> realized_times(data.n, 0.0f);
        simulate_one_with_times(
            perm.data(), data.n, scenario_idx, noise.S,
            data.ready.data(), data.target.data(), data.latest.data(),
            data.weight.data(), data.cls.data(), sep.data,
            noise.delay.data(), noise.mult[scenario_idx],
            noise.closeA[scenario_idx], noise.closeB[scenario_idx],
            realized_times.data(), sep.triangle_ok, m_penalty
        );

        std::cout << "{\n"
                  << "  \"scenario\": " << scenario_idx << ",\n"
                  << "  \"weather_mult\": " << noise.mult[scenario_idx] << ",\n"
                  << "  \"closure\": [" << noise.closeA[scenario_idx] << ", " << noise.closeB[scenario_idx] << "],\n"
                  << "  \"aircraft\": [\n";
        for (int k = 0; k < data.n; ++k) {
            int a = perm[k];
            float ready_real = data.ready[a] + noise.delay[a * noise.S + scenario_idx];
            float t = realized_times[k];
            float delay = std::max(0.0f, t - data.target[a]);
            std::cout << "    {\"id\": " << data.id[a]
                      << ", \"ready_real\": " << ready_real
                      << ", \"time\": " << t
                      << ", \"delay\": " << delay << "}"
                      << (k + 1 < data.n ? ",\n" : "\n");
        }
        std::cout << "  ]\n}\n";
        return 0;
    } else if (cmd == "gen") {
        int n = 50;
        float rho = 1.2f;
        uint64_t seed = 1;
        float mix_arr = 0.5f;
        std::string out_path = "instance.csv";

        for (int i = 2; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "--n" && i + 1 < argc) n = std::stoi(argv[++i]);
            else if (arg == "--rho" && i + 1 < argc) rho = std::stof(argv[++i]);
            else if (arg == "--seed" && i + 1 < argc) seed = std::stoull(argv[++i]);
            else if (arg == "--mix-arr" && i + 1 < argc) mix_arr = std::stof(argv[++i]);
            else if (arg == "--out" && i + 1 < argc) out_path = argv[++i];
        }

        AircraftData data = generate_instance(n, rho, seed, mix_arr);
        if (save_instance(out_path, data)) {
            std::cout << "Successfully generated " << n << " aircraft (rho=" << rho
                      << ") to " << out_path << "\n";
            return 0;
        } else {
            std::cerr << "Failed to save instance to " << out_path << "\n";
            return 1;
        }
    } else if (cmd == "eval") {
        std::string inst_file;
        std::string perm_arg = "fcfs";
        int scenarios = 1024;
        uint64_t seed = 1;
        float lambda_std = 1.0f;
        float m_penalty = 5.0f;

        for (int i = 2; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "--instance" && i + 1 < argc) inst_file = argv[++i];
            else if (arg == "--perm" && i + 1 < argc) perm_arg = argv[++i];
            else if (arg == "--scenarios" && i + 1 < argc) scenarios = std::stoi(argv[++i]);
            else if (arg == "--seed" && i + 1 < argc) seed = std::stoull(argv[++i]);
            else if (arg == "--lambda" && i + 1 < argc) lambda_std = std::stof(argv[++i]);
            else if (arg == "--penalty" && i + 1 < argc) m_penalty = std::stof(argv[++i]);
        }

        AircraftData data;
        if (!load_instance(inst_file, data)) return 1;
        SeparationMatrix sep = get_default_separation_matrix();

        std::vector<int> perm;
        if (perm_arg == "fcfs") {
            perm = solve_fcfs(data);
        } else {
            if (!load_schedule(perm_arg, perm)) {
                std::cerr << "Failed to load schedule from " << perm_arg << "\n";
                return 1;
            }
        }

        NoiseData noise;
        generate_noise_cpu(noise, data, scenarios, seed);

        EvalResult res = evaluate_schedule_cpu(data, sep, perm, noise, lambda_std, m_penalty);
        std::cout << std::fixed << std::setprecision(4)
                  << "J: " << res.J << "\n"
                  << "mean_cost: " << res.mean_cost << "\n"
                  << "std_cost: " << res.std_cost << "\n"
                  << "p95: " << res.p95 << "\n"
                  << "cvar95: " << res.cvar95 << "\n"
                  << "miss_prob: " << res.miss_prob << "\n"
                  << "plan_dev: " << res.plan_deviation << "\n";
        return 0;
    } else if (cmd == "solve") {
        auto solve_start_time = std::chrono::high_resolution_clock::now();

        std::string inst_file;
        std::string impl = "serial";
        int chains = 64;
        int scenarios = 1024;
        int iters = 2000;
        int epochs = 20;
        float lambda_std = 1.0f;
        float m_penalty = 5.0f;
        uint64_t seed = 1;
        int threads = 0;
        std::string omp_mode = "chains";
        [[maybe_unused]] int block_size = 256;
        [[maybe_unused]] std::string kernel = "k3";
        std::string out_file;
        bool progress = false;
        int dump_costs = 0;
        std::string sep_file;
        int s_final = 20000;

        for (int i = 2; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "--instance" && i + 1 < argc) inst_file = argv[++i];
            else if (arg == "--impl" && i + 1 < argc) impl = argv[++i];
            else if (arg == "--chains" && i + 1 < argc) chains = std::stoi(argv[++i]);
            else if (arg == "--scenarios" && i + 1 < argc) scenarios = std::stoi(argv[++i]);
            else if (arg == "--iters" && i + 1 < argc) iters = std::stoi(argv[++i]);
            else if (arg == "--epochs" && i + 1 < argc) epochs = std::stoi(argv[++i]);
            else if (arg == "--lambda" && i + 1 < argc) lambda_std = std::stof(argv[++i]);
            else if (arg == "--penalty" && i + 1 < argc) m_penalty = std::stof(argv[++i]);
            else if (arg == "--seed" && i + 1 < argc) seed = std::stoull(argv[++i]);
            else if (arg == "--threads" && i + 1 < argc) threads = std::stoi(argv[++i]);
            else if (arg == "--omp-mode" && i + 1 < argc) omp_mode = argv[++i];
            else if (arg == "--block" && i + 1 < argc) block_size = std::stoi(argv[++i]);
            else if (arg == "--kernel" && i + 1 < argc) kernel = argv[++i];
            else if (arg == "--out" && i + 1 < argc) out_file = argv[++i];
            else if (arg == "--progress") progress = true;
            else if (arg == "--dump-costs" && i + 1 < argc) dump_costs = std::stoi(argv[++i]);
            else if (arg == "--separation" && i + 1 < argc) sep_file = argv[++i];
            else if (arg == "--s-final" && i + 1 < argc) s_final = std::stoi(argv[++i]);
        }

        AircraftData data;
        if (!load_instance(inst_file, data)) return 1;

        SeparationMatrix sep;
        if (!sep_file.empty()) {
            if (!load_separation_matrix(sep_file, sep)) {
                std::cerr << "Failed to load separation matrix from " << sep_file << "\n";
                return 1;
            }
        } else {
            sep = get_default_separation_matrix();
        }

        SAResult res;
        if (impl == "serial") {
            res = solve_sa_serial(data, sep, chains, scenarios, epochs, iters, seed, lambda_std, m_penalty, progress);
        } else if (impl == "omp") {
#if defined(RUNWAY_HAS_OPENMP)
            res = solve_sa_omp(data, sep, chains, scenarios, epochs, iters, seed, omp_mode, threads, lambda_std, m_penalty, progress);
#else
            std::cerr << "OpenMP not compiled in this build.\n";
            return 1;
#endif
        } else if (impl == "cuda") {
#if defined(RUNWAY_HAS_CUDA)
            res = solve_sa_cuda(data, sep, chains, scenarios, epochs, iters, seed, kernel, block_size, lambda_std, m_penalty);
#else
            std::cerr << "CUDA not compiled in this build.\n";
            return 1;
#endif
        } else if (impl == "random") {
            int total_evals = chains * epochs * iters;
            res = solve_random_search(data, sep, scenarios, total_evals, seed, lambda_std, m_penalty);
        } else if (impl == "brute") {
            if (data.n > 10) {
                std::cerr << "Brute force only supported for n <= 10 (got n=" << data.n << ").\n";
                return 1;
            }
            NoiseData noise;
            generate_noise_cpu(noise, data, scenarios, seed);
            res = solve_brute_force(data, sep, noise, lambda_std, m_penalty);
        } else {
            std::cerr << "Unknown impl: " << impl << "\n";
            return 1;
        }

        // Out-of-sample evaluation
        NoiseData oos_noise;
        generate_noise_cpu(oos_noise, data, s_final, 99999);
        res.out_of_sample_eval = evaluate_schedule_cpu(data, sep, res.best_perm, oos_noise, lambda_std, m_penalty);

        // FCFS permutation and evaluation
        std::vector<int> perm_fcfs = solve_fcfs(data);
        EvalResult fcfs_eval = evaluate_schedule_cpu(data, sep, perm_fcfs, oos_noise, lambda_std, m_penalty);

        // Deterministic planned times (zero noise)
        std::vector<float> planned_times_fcfs(data.n, 0.0f);
        std::vector<float> zero_delay(data.n, 0.0f);
        simulate_one_with_times(
            perm_fcfs.data(), data.n, 0, 1,
            data.ready.data(), data.target.data(), data.latest.data(),
            data.weight.data(), data.cls.data(), sep.data,
            zero_delay.data(), 1.0f, 0.0f, 0.0f,
            planned_times_fcfs.data(), sep.triangle_ok, m_penalty
        );

        std::vector<float> planned_times_best(data.n, 0.0f);
        simulate_one_with_times(
            res.best_perm.data(), data.n, 0, 1,
            data.ready.data(), data.target.data(), data.latest.data(),
            data.weight.data(), data.cls.data(), sep.data,
            zero_delay.data(), 1.0f, 0.0f, 0.0f,
            planned_times_best.data(), sep.triangle_ok, m_penalty
        );

        auto solve_end_time = std::chrono::high_resolution_clock::now();
        double end_to_end_ms = std::chrono::duration<double, std::milli>(solve_end_time - solve_start_time).count();

        int omp_max_threads = 1;
#if defined(RUNWAY_HAS_OPENMP)
        omp_max_threads = omp_get_max_threads();
#endif
        bool cuda_avail = false;
#if defined(RUNWAY_HAS_CUDA)
        cuda_avail = true;
#endif

        std::cout << std::fixed << std::setprecision(4)
                  << "Solve completed (" << impl << "):\n"
                  << "  J (in-sample):  " << res.in_sample_J << "\n"
                  << "  J (out-of-sample): " << res.out_of_sample_eval.J << "\n"
                  << "  Mean Cost (OOS):" << res.out_of_sample_eval.mean_cost << "\n"
                  << "  Std Cost (OOS): " << res.out_of_sample_eval.std_cost << "\n"
                  << "  P95 Cost (OOS): " << res.out_of_sample_eval.p95 << "\n"
                  << "  CVaR95 (OOS):   " << res.out_of_sample_eval.cvar95 << "\n"
                  << "  Miss Prob (OOS):" << res.out_of_sample_eval.miss_prob << "\n"
                  << "  Plan Dev (OOS): " << res.out_of_sample_eval.plan_deviation << "\n"
                  << "  Total Time (ms):" << res.total_time_ms << "\n"
                  << "  Sims/sec:       " << res.sim_per_sec << "\n";

        if (!out_file.empty()) {
            std::filesystem::path p(out_file);
            if (p.has_parent_path()) {
                std::filesystem::create_directories(p.parent_path());
            }
            std::ofstream fout(out_file);
            if (fout.is_open()) {
                size_t dc = dump_costs > 0 ? static_cast<size_t>(dump_costs) : 0;
                fout << "{\n"
                     << "  \"impl\": \"" << impl << "\",\n"
                     << "  \"seed\": " << seed << ",\n"
                     << "  \"n\": " << data.n << ",\n"
                     << "  \"params\": {\"chains\": " << chains << ", \"scenarios\": " << scenarios
                     << ", \"iters\": " << iters << ", \"epochs\": " << epochs
                     << ", \"lambda\": " << lambda_std << ", \"penalty\": " << m_penalty << "},\n"
                     << "  \"perm_fcfs\": " << int_array_to_json(perm_fcfs) << ",\n"
                     << "  \"perm_best\": " << int_array_to_json(res.best_perm) << ",\n"
                     << "  \"planned_times_fcfs\": " << float_array_to_json(planned_times_fcfs) << ",\n"
                     << "  \"planned_times_best\": " << float_array_to_json(planned_times_best) << ",\n"
                     << "  \"metrics\": {\n"
                     << "    \"fcfs\": " << eval_metrics_to_json(fcfs_eval) << ",\n"
                     << "    \"best\": " << eval_metrics_to_json(res.out_of_sample_eval) << "\n"
                     << "  },\n"
                     << "  \"costs_fcfs\": " << float_array_to_json(fcfs_eval.per_scenario_costs, dc) << ",\n"
                     << "  \"costs_best\": " << float_array_to_json(res.out_of_sample_eval.per_scenario_costs, dc) << ",\n"
                     << "  \"convergence\": " << convergence_to_json(res.convergence) << ",\n"
                     << "  \"timings\": {\"kernel_ms\": " << res.kernel_time_ms
                     << ", \"end_to_end_ms\": " << end_to_end_ms
                     << ", \"sim_per_s\": " << res.sim_per_sec << "},\n"
                     << "  \"hardware\": {\"omp_max_threads\": " << omp_max_threads
                     << ", \"cuda_available\": " << (cuda_avail ? "true" : "false") << "}\n"
                     << "}\n";
            } else {
                std::cerr << "Failed to open output file: " << out_file << "\n";
            }
        }
        return 0;
    } else if (cmd == "final") {
        std::string inst_file;
        std::string perm_file;
        int scenarios = 20000;
        uint64_t seed = 999;
        float lambda_std = 1.0f;
        float m_penalty = 5.0f;

        for (int i = 2; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "--instance" && i + 1 < argc) inst_file = argv[++i];
            else if (arg == "--perm" && i + 1 < argc) perm_file = argv[++i];
            else if (arg == "--scenarios" && i + 1 < argc) scenarios = std::stoi(argv[++i]);
            else if (arg == "--seed" && i + 1 < argc) seed = std::stoull(argv[++i]);
            else if (arg == "--lambda" && i + 1 < argc) lambda_std = std::stof(argv[++i]);
            else if (arg == "--penalty" && i + 1 < argc) m_penalty = std::stof(argv[++i]);
        }

        AircraftData data;
        if (!load_instance(inst_file, data)) return 1;
        SeparationMatrix sep = get_default_separation_matrix();

        std::vector<int> perm;
        if (!load_schedule(perm_file, perm)) {
            std::cerr << "Failed to load schedule: " << perm_file << "\n";
            return 1;
        }

        NoiseData noise;
        generate_noise_cpu(noise, data, scenarios, seed);

        EvalResult res = evaluate_schedule_cpu(data, sep, perm, noise, lambda_std, m_penalty);
        std::cout << std::fixed << std::setprecision(4)
                  << "Out-of-sample final evaluation (" << scenarios << " scenarios):\n"
                  << "  J:              " << res.J << "\n"
                  << "  Mean Cost:      " << res.mean_cost << "\n"
                  << "  Std Cost:       " << res.std_cost << "\n"
                  << "  P95 Cost:       " << res.p95 << "\n"
                  << "  CVaR95:         " << res.cvar95 << "\n"
                  << "  Miss Prob:      " << res.miss_prob << "\n"
                  << "  Plan Dev:       " << res.plan_deviation << "\n"
                  << "  Emergency Mean: " << res.emergency_mean_delay << "\n"
                  << "  Emergency Max:  " << res.emergency_max_delay << "\n";
        return 0;
    } else {
        std::cerr << "Unknown command: " << cmd << "\n";
        print_usage();
        return 1;
    }
}
