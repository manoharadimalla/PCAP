#include "runway/instance_io.hpp"
#include "runway/sim_core.hpp"
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <chrono>
#include <iomanip>

using namespace runway;

void print_usage() {
    std::cout << "RunwayMC — GPU-Accelerated Robust Runway Scheduling under Uncertainty\n"
              << "\n"
              << "Mandatory Disclaimer:\n"
              << "  This is an academic simulation with simplified separation standards.\n"
              << "  It is not validated for, and must not be used in, real air-traffic operations.\n\n"
              << "Commands:\n"
              << "  runway gen   --n <int> --rho <float> --seed <int> --out <file>\n"
              << "  runway eval  --instance <file> --perm fcfs|<file> --scenarios <int> --seed <int>\n"
              << "  runway solve --instance <file> --impl serial|omp|cuda|random|brute\n"
              << "               --chains <int> --scenarios <int> --iters <int> --epochs <int>\n"
              << "               [--lambda <float>] [--penalty <float>] [--seed <int>]\n"
              << "               [--threads <int>] [--omp-mode chains|scenarios|hybrid]\n"
              << "               [--block <int>] [--kernel k1|k2|k3] [--out <json/csv>]\n"
              << "  runway final --instance <file> --perm <file> --scenarios <int> --seed <int>\n";
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        print_usage();
        return 1;
    }

    std::string cmd = argv[1];

    if (cmd == "gen") {
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
        int block_size = 256;
        std::string kernel = "k3";
        std::string out_file;

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
        }

        AircraftData data;
        if (!load_instance(inst_file, data)) return 1;
        SeparationMatrix sep = get_default_separation_matrix();

        SAResult res;
        if (impl == "serial") {
            res = solve_sa_serial(data, sep, chains, scenarios, epochs, iters, seed, lambda_std, m_penalty);
        } else if (impl == "omp") {
#if defined(RUNWAY_HAS_OPENMP)
            res = solve_sa_omp(data, sep, chains, scenarios, epochs, iters, seed, omp_mode, threads, lambda_std, m_penalty);
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
        generate_noise_cpu(oos_noise, data, 20000, 99999);
        res.out_of_sample_eval = evaluate_schedule_cpu(data, sep, res.best_perm, oos_noise, lambda_std, m_penalty);

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
            std::ofstream fout(out_file);
            if (fout.is_open()) {
                fout << "{\n"
                     << "  \"impl\": \"" << impl << "\",\n"
                     << "  \"chains\": " << chains << ",\n"
                     << "  \"scenarios\": " << scenarios << ",\n"
                     << "  \"iters\": " << iters << ",\n"
                     << "  \"epochs\": " << epochs << ",\n"
                     << "  \"seed\": " << seed << ",\n"
                     << "  \"J_in_sample\": " << res.in_sample_J << ",\n"
                     << "  \"J_out_of_sample\": " << res.out_of_sample_eval.J << ",\n"
                     << "  \"mean_cost\": " << res.out_of_sample_eval.mean_cost << ",\n"
                     << "  \"std_cost\": " << res.out_of_sample_eval.std_cost << ",\n"
                     << "  \"p95\": " << res.out_of_sample_eval.p95 << ",\n"
                     << "  \"cvar95\": " << res.out_of_sample_eval.cvar95 << ",\n"
                     << "  \"miss_prob\": " << res.out_of_sample_eval.miss_prob << ",\n"
                     << "  \"plan_dev\": " << res.out_of_sample_eval.plan_deviation << ",\n"
                     << "  \"total_time_ms\": " << res.total_time_ms << ",\n"
                     << "  \"sim_per_sec\": " << res.sim_per_sec << "\n"
                     << "}\n";
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
