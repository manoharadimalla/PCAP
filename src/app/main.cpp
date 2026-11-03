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

static std::string eval_metrics_to_json(const EvalResult& r) {
    std::ostringstream oss;
    oss << "{\"mean_cost\": " << r.mean_cost
        << ", \"p95\": " << r.p95
        << ", \"late_pct\": " << r.late_pct
        << ", \"emergency_mean_delay\": " << r.emergency_mean_delay << "}";
    return oss.str();
}

void print_usage() {
    std::cout << "RunwayMC — Robust Runway Scheduling under Uncertainty\n"
              << "\n"
              << "Mandatory Disclaimer:\n"
              << "  This is an academic simulation with simplified separation standards.\n"
              << "  It is not validated for, and must not be used in, real air-traffic operations.\n\n"
              << "Commands:\n"
              << "  runway info\n"
              << "  runway validate --instance <file> [--separation <file.csv>]\n"
              << "  runway gen      --n <int> --rho <float> --seed <int> --out <file>\n"
              << "  runway eval     --instance <file> --perm fcfs|<file> --scenarios <int> --seed <int>\n"
              << "  runway solve    --instance <file> --impl serial|omp|cuda|random|brute\n"
              << "                  --chains <int> --scenarios <int> --iters <int>\n"
              << "                  [--seed <int>] [--threads <int>] [--omp-mode chains|scenarios]\n"
              << "                  [--out <json>] [--progress] [--dump-costs <int>]\n"
              << "                  [--separation <file.csv>] [--s-final <int>]\n"
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
                    int row = 0;
                    std::unordered_set<int> seen_ids;
                    while (std::getline(fin, line)) {
                        row++;
                        if (line.empty()) continue;
                        std::stringstream ss(line);
                        std::string sid, sop, swake, sready, starget, slatest, sprio, sweight;

                        if (!std::getline(ss, sid, ',') || !std::getline(ss, sop, ',') ||
                            !std::getline(ss, swake, ',') || !std::getline(ss, sready, ',') ||
                            !std::getline(ss, starget, ',') || !std::getline(ss, slatest, ',') ||
                            !std::getline(ss, sprio, ',') || !std::getline(ss, sweight, ',')) {
                            errors.push_back({row, "format", "Missing required CSV columns in row " + std::to_string(row)});
                            continue;
                        }

                        int id = 0;
                        try {
                            id = std::stoi(sid);
                            if (seen_ids.count(id)) {
                                errors.push_back({row, "id", "Duplicate aircraft ID " + std::to_string(id)});
                            }
                            seen_ids.insert(id);
                        } catch (...) {
                            errors.push_back({row, "id", "Invalid non-integer aircraft ID: " + sid});
                        }

                        float ready = 0.0f, target = 0.0f, latest = 0.0f, weight = 1.0f;
                        try { ready = std::stof(sready); } catch (...) { errors.push_back({row, "ready", "Invalid float: " + sready}); }
                        try { target = std::stof(starget); } catch (...) { errors.push_back({row, "target", "Invalid float: " + starget}); }
                        try { latest = std::stof(slatest); } catch (...) { errors.push_back({row, "latest", "Invalid float: " + slatest}); }
                        try { weight = std::stof(sweight); } catch (...) { errors.push_back({row, "weight", "Invalid float: " + sweight}); }

                        if (ready > target + 1e-4f) {
                            errors.push_back({row, "target", "ready (" + sready + ") > target (" + starget + ")"});
                        }
                        if (target > latest + 1e-4f) {
                            errors.push_back({row, "latest", "target (" + starget + ") > latest (" + slatest + ")"});
                        }
                        if (weight <= 0.0f) {
                            errors.push_back({row, "weight", "weight must be > 0, got " + sweight});
                        }
                    }
                    if (row == 0) {
                        errors.push_back({0, "instance", "Instance has 0 aircraft records (n >= 1 required)"});
                    }
                }
            }
        }

        std::cout << "{\"ok\": " << (errors.empty() ? "true" : "false") << ", \"errors\": [";
        for (size_t i = 0; i < errors.size(); ++i) {
            if (i > 0) std::cout << ", ";
            std::cout << "{\"row\": " << errors[i].row
                      << ", \"column\": \"" << errors[i].column
                      << "\", \"message\": \"" << errors[i].message << "\"}";
        }
        std::cout << "], \"warnings\": [";
        for (size_t i = 0; i < warnings.size(); ++i) {
            if (i > 0) std::cout << ", ";
            std::cout << "\"" << warnings[i] << "\"";
        }
        std::cout << "], \"triangle_ok\": " << (triangle_ok ? "true" : "false") << "}\n";
        return errors.empty() ? 0 : 1;
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

        for (int i = 2; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "--instance" && i + 1 < argc) inst_file = argv[++i];
            else if (arg == "--perm" && i + 1 < argc) perm_arg = argv[++i];
            else if (arg == "--scenarios" && i + 1 < argc) scenarios = std::stoi(argv[++i]);
            else if (arg == "--seed" && i + 1 < argc) seed = std::stoull(argv[++i]);
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

        EvalResult res = evaluate_schedule_cpu(data, sep, perm, noise);
        std::cout << std::fixed << std::setprecision(4)
                  << "mean_cost: " << res.mean_cost << "\n"
                  << "p95: " << res.p95 << "\n"
                  << "late_pct: " << res.late_pct << "\n"
                  << "emergency_mean_delay: " << res.emergency_mean_delay << "\n";
        return 0;
    } else if (cmd == "solve") {
        auto solve_start_time = std::chrono::high_resolution_clock::now();

        std::string inst_file;
        std::string impl = "serial";
        int chains = 16;
        int scenarios = 1024;
        int iters = 1000;
        uint64_t seed = 1;
        int threads = 0;
        std::string omp_mode = "chains";
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
            else if (arg == "--seed" && i + 1 < argc) seed = std::stoull(argv[++i]);
            else if (arg == "--threads" && i + 1 < argc) threads = std::stoi(argv[++i]);
            else if (arg == "--omp-mode" && i + 1 < argc) omp_mode = argv[++i];
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
            res = solve_sa_serial(data, sep, chains, scenarios, iters, seed, 5.0f, progress);
        } else if (impl == "omp") {
#if defined(RUNWAY_HAS_OPENMP)
            res = solve_sa_omp(data, sep, chains, scenarios, iters, seed, omp_mode, threads, 5.0f, progress);
#else
            std::cerr << "OpenMP not compiled in this build.\n";
            return 1;
#endif
        } else if (impl == "random") {
            int total_evals = chains * iters;
            res = solve_random_search(data, sep, scenarios, total_evals, seed, 5.0f);
        } else if (impl == "brute") {
            if (data.n > 9) {
                std::cerr << "Brute force only supported for n <= 9 (got n=" << data.n << ").\n";
                return 1;
            }
            NoiseData noise;
            generate_noise_cpu(noise, data, scenarios, seed);
            res = solve_brute_force(data, sep, noise, 5.0f);
        } else {
            std::cerr << "Unknown impl: " << impl << "\n";
            return 1;
        }

        // Test evaluation on fresh 20,000 scenarios
        NoiseData test_noise;
        generate_noise_cpu(test_noise, data, s_final, 99999);
        res.out_of_sample_eval = evaluate_schedule_cpu(data, sep, res.best_perm, test_noise);

        // Baseline FCFS schedule evaluated on identical test days
        std::vector<int> perm_fcfs = solve_fcfs(data);
        EvalResult fcfs_eval = evaluate_schedule_cpu(data, sep, perm_fcfs, test_noise);

        // Planned times (zero noise)
        std::vector<float> planned_times_fcfs(data.n, 0.0f);
        std::vector<float> zero_delay(data.n, 0.0f);
        simulate_one_with_times(
            perm_fcfs.data(), data.n, 0, 1,
            data.ready.data(), data.target.data(), data.latest.data(),
            data.weight.data(), data.cls.data(), sep.data,
            zero_delay.data(), 0.0f, 0.0f,
            planned_times_fcfs.data()
        );

        std::vector<float> planned_times_best(data.n, 0.0f);
        simulate_one_with_times(
            res.best_perm.data(), data.n, 0, 1,
            data.ready.data(), data.target.data(), data.latest.data(),
            data.weight.data(), data.cls.data(), sep.data,
            zero_delay.data(), 0.0f, 0.0f,
            planned_times_best.data()
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
                  << "  Train Mean Cost:  " << res.in_sample_J << "\n"
                  << "  Test Mean Cost:   " << res.out_of_sample_eval.mean_cost << "\n"
                  << "  Test P95 Cost:    " << res.out_of_sample_eval.p95 << "\n"
                  << "  Test Late Pct:    " << res.out_of_sample_eval.late_pct << "\n"
                  << "  Test Emerg Delay: " << res.out_of_sample_eval.emergency_mean_delay << "\n"
                  << "  Total Time (ms):  " << res.total_time_ms << "\n"
                  << "  Sims/sec:         " << res.sim_per_sec << "\n";

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
                     << ", \"iters\": " << iters << "},\n"
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

        for (int i = 2; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "--instance" && i + 1 < argc) inst_file = argv[++i];
            else if (arg == "--perm" && i + 1 < argc) perm_file = argv[++i];
            else if (arg == "--scenarios" && i + 1 < argc) scenarios = std::stoi(argv[++i]);
            else if (arg == "--seed" && i + 1 < argc) seed = std::stoull(argv[++i]);
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

        EvalResult res = evaluate_schedule_cpu(data, sep, perm, noise);
        std::cout << std::fixed << std::setprecision(4)
                  << "Test Evaluation (" << scenarios << " test days):\n"
                  << "  Mean Cost:      " << res.mean_cost << "\n"
                  << "  P95 Cost:       " << res.p95 << "\n"
                  << "  Late Pct:       " << res.late_pct << "\n"
                  << "  Emergency Mean: " << res.emergency_mean_delay << "\n";
        return 0;
    } else {
        std::cerr << "Unknown command: " << cmd << "\n";
        print_usage();
        return 1;
    }
}
