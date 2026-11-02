#!/usr/bin/env python3
"""
RunwayMC Benchmark Suite (Section 10)
Executes experiments E2 through E9 and outputs CSV results to results/
"""

import os
import sys
import subprocess
import json
import csv
import time
import argparse
import numpy as np

def get_executable():
    paths = [
        os.path.join("build", "runway.exe"),
        os.path.join("build", "runway"),
        os.path.join(".", "runway.exe"),
        os.path.join(".", "runway")
    ]
    for p in paths:
        if os.path.exists(p):
            return os.path.abspath(p)
    raise FileNotFoundError("Could not find runway executable. Build the project first.")

def run_command(cmd):
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"Error: {' '.join(cmd)}\n{res.stderr}")
        return None
    return res.stdout

def ensure_instance(exe, n, rho, seed=42):
    os.makedirs("data/instances", exist_ok=True)
    path = f"data/instances/n{n}_r{int(rho*10)}_s{seed}.csv"
    if not os.path.exists(path):
        run_command([exe, "gen", "--n", str(n), "--rho", str(rho), "--seed", str(seed), "--out", path])
    return path

def exp_e2_baselines(exe):
    print("\n--- Running Experiment E2: Baselines Comparison ---")
    os.makedirs("results", exist_ok=True)
    n = 8
    rhos = [1.0, 1.2, 1.5]
    records = []

    for rho in rhos:
        inst = ensure_instance(exe, n, rho)
        # 1. FCFS
        out = run_command([exe, "eval", "--instance", inst, "--perm", "fcfs", "--scenarios", "500", "--seed", "1"])
        j_fcfs = float([l for l in out.splitlines() if l.startswith("J:")][0].split(":")[1])

        # 2. Brute Force
        tmp_json = f"results/temp_e2_brute_{int(rho*10)}.json"
        run_command([exe, "solve", "--instance", inst, "--impl", "brute", "--scenarios", "500", "--seed", "1", "--out", tmp_json])
        with open(tmp_json) as f: b_data = json.load(f)
        j_brute = b_data["J_in_sample"]
        t_brute = b_data["total_time_ms"]
        os.remove(tmp_json)

        # 3. Random Search (same eval budget as SA: 16 chains * 5 epochs * 400 iters = 32000 evals)
        tmp_json = f"results/temp_e2_rand_{int(rho*10)}.json"
        run_command([exe, "solve", "--instance", inst, "--impl", "random", "--chains", "16", "--epochs", "5", "--iters", "400", "--scenarios", "500", "--seed", "1", "--out", tmp_json])
        with open(tmp_json) as f: r_data = json.load(f)
        j_rand = r_data["J_in_sample"]
        t_rand = r_data["total_time_ms"]
        os.remove(tmp_json)

        # 4. SA Serial
        tmp_json = f"results/temp_e2_sa_{int(rho*10)}.json"
        run_command([exe, "solve", "--instance", inst, "--impl", "serial", "--chains", "16", "--epochs", "5", "--iters", "400", "--scenarios", "500", "--seed", "1", "--out", tmp_json])
        with open(tmp_json) as f: sa_data = json.load(f)
        j_sa = sa_data["J_in_sample"]
        t_sa = sa_data["total_time_ms"]
        os.remove(tmp_json)

        records.append({
            "n": n, "rho": rho,
            "J_FCFS": j_fcfs, "J_Brute": j_brute, "J_Random": j_rand, "J_SA": j_sa,
            "Gap_FCFS_pct": round((j_fcfs - j_brute)/j_fcfs * 100, 2),
            "Gap_Random_pct": round((j_rand - j_brute)/j_brute * 100, 2),
            "Gap_SA_pct": round((j_sa - j_brute)/j_brute * 100, 2),
            "Time_Brute_ms": t_brute, "Time_Random_ms": t_rand, "Time_SA_ms": t_sa
        })
        print(f"rho={rho}: FCFS={j_fcfs:.1f}, Brute={j_brute:.1f}, Random={j_rand:.1f}, SA={j_sa:.1f}")

    csv_path = "results/e2_baselines.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        w.writeheader()
        w.writerows(records)
    print(f"Saved {csv_path}")

def exp_e3_openmp_scaling(exe):
    print("\n--- Running Experiment E3: OpenMP Strong Scaling ---")
    inst = ensure_instance(exe, 50, 1.2)
    threads_list = [1, 2, 4, 8, 12]
    modes = ["chains", "scenarios", "hybrid"]
    records = []

    # Warm-up
    run_command([exe, "solve", "--instance", inst, "--impl", "omp", "--omp-mode", "chains", "--threads", "2", "--chains", "32", "--scenarios", "512", "--iters", "200", "--epochs", "2", "--seed", "1"])

    t1_chains = None

    for mode in modes:
        for th in threads_list:
            times = []
            for rep in range(3):
                tmp_json = f"results/temp_e3_{mode}_{th}_{rep}.json"
                run_command([exe, "solve", "--instance", inst, "--impl", "omp", "--omp-mode", mode, "--threads", str(th),
                             "--chains", "32", "--scenarios", "512", "--iters", "400", "--epochs", "3", "--seed", str(rep+1), "--out", tmp_json])
                if os.path.exists(tmp_json):
                    with open(tmp_json) as f: d = json.load(f)
                    times.append(d["total_time_ms"])
                    os.remove(tmp_json)

            median_t = float(np.median(times))
            iqr_t = float(np.percentile(times, 75) - np.percentile(times, 25))

            if mode == "chains" and th == 1:
                t1_chains = median_t

            speedup = t1_chains / median_t if t1_chains else 1.0
            efficiency = speedup / th

            records.append({
                "mode": mode, "threads": th,
                "median_time_ms": round(median_t, 2), "iqr_time_ms": round(iqr_t, 2),
                "speedup": round(speedup, 3), "efficiency": round(efficiency, 3)
            })
            print(f"Mode={mode:9s} | Threads={th:2d} | Time={median_t:8.2f} ms | Speedup={speedup:5.2f}x | Efficiency={efficiency:5.2f}")

    csv_path = "results/e3_openmp_scaling.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        w.writeheader()
        w.writerows(records)
    print(f"Saved {csv_path}")

def exp_e4_throughput(exe):
    print("\n--- Running Experiment E4: Scenario and Aircraft Scaling ---")
    s_list = [128, 256, 512, 1024, 2048]
    n_list = [10, 20, 50, 100]
    records = []

    for n in n_list:
        inst = ensure_instance(exe, n, 1.2)
        for s in s_list:
            # 1. Serial
            tmp_s = f"results/temp_e4_s_{n}_{s}.json"
            run_command([exe, "solve", "--instance", inst, "--impl", "serial", "--chains", "16", "--scenarios", str(s), "--iters", "300", "--epochs", "2", "--seed", "1", "--out", tmp_s])
            with open(tmp_s) as f: d_s = json.load(f)
            os.remove(tmp_s)

            # 2. OpenMP (8 threads)
            tmp_o = f"results/temp_e4_o_{n}_{s}.json"
            run_command([exe, "solve", "--instance", inst, "--impl", "omp", "--omp-mode", "chains", "--threads", "8", "--chains", "16", "--scenarios", str(s), "--iters", "300", "--epochs", "2", "--seed", "1", "--out", tmp_o])
            with open(tmp_o) as f: d_o = json.load(f)
            os.remove(tmp_o)

            records.append({
                "n": n, "S": s,
                "serial_time_ms": d_s["total_time_ms"],
                "serial_sim_per_sec": d_s["sim_per_sec"],
                "omp_time_ms": d_o["total_time_ms"],
                "omp_sim_per_sec": d_o["sim_per_sec"],
                "speedup_omp_vs_serial": round(d_s["total_time_ms"] / d_o["total_time_ms"], 2)
            })
            print(f"n={n:3d}, S={s:4d} | Serial: {d_s['sim_per_sec']:10.0f} sim/s | OMP: {d_o['sim_per_sec']:10.0f} sim/s | Speedup: {records[-1]['speedup_omp_vs_serial']:.2f}x")

    csv_path = "results/e4_throughput.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        w.writeheader()
        w.writerows(records)
    print(f"Saved {csv_path}")

def exp_e8_value_of_uncertainty(exe):
    print("\n--- Running Experiment E8: Value of Modelling Uncertainty ---")
    inst = ensure_instance(exe, 50, 1.2)
    records = []

    # 1. Deterministic-optimized schedule (S=1 scenario, zero noise)
    tmp_det = "results/temp_e8_det.json"
    run_command([exe, "solve", "--instance", inst, "--impl", "omp", "--chains", "32", "--scenarios", "1", "--iters", "1000", "--epochs", "5", "--seed", "1", "--out", tmp_det])
    with open(tmp_det) as f: d_det = json.load(f)
    os.remove(tmp_det)

    # 2. Robust-optimized schedule (S=1024 scenarios)
    tmp_rob = "results/temp_e8_rob.json"
    run_command([exe, "solve", "--instance", inst, "--impl", "omp", "--chains", "32", "--scenarios", "1024", "--iters", "1000", "--epochs", "5", "--seed", "1", "--out", tmp_rob])
    with open(tmp_rob) as f: d_rob = json.load(f)
    os.remove(tmp_rob)

    records.append({
        "strategy": "Deterministic-Optimized",
        "J_out_of_sample": d_det["J_out_of_sample"],
        "mean_cost": d_det["mean_cost"],
        "std_cost": d_det["std_cost"],
        "p95": d_det["p95"],
        "cvar95": d_det["cvar95"],
        "miss_prob": d_det["miss_prob"],
        "plan_dev": d_det["plan_dev"]
    })
    records.append({
        "strategy": "Robust-Optimized (RunwayMC)",
        "J_out_of_sample": d_rob["J_out_of_sample"],
        "mean_cost": d_rob["mean_cost"],
        "std_cost": d_rob["std_cost"],
        "p95": d_rob["p95"],
        "cvar95": d_rob["cvar95"],
        "miss_prob": d_rob["miss_prob"],
        "plan_dev": d_rob["plan_dev"]
    })

    print(f"Deterministic: OOS J={d_det['J_out_of_sample']:.1f}, Miss Prob={d_det['miss_prob']*100:.1f}%, CVaR95={d_det['cvar95']:.1f}")
    print(f"Robust:        OOS J={d_rob['J_out_of_sample']:.1f}, Miss Prob={d_rob['miss_prob']*100:.1f}%, CVaR95={d_rob['cvar95']:.1f}")

    csv_path = "results/e8_uncertainty.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        w.writeheader()
        w.writerows(records)
    print(f"Saved {csv_path}")

def exp_e9_sensitivity(exe):
    print("\n--- Running Experiment E9: Sensitivity Analysis (Closure & Lambda) ---")
    inst = ensure_instance(exe, 30, 1.2)
    lambdas = [0.0, 0.5, 1.0, 2.0]
    records = []

    for lam in lambdas:
        tmp_json = f"results/temp_e9_lam_{lam}.json"
        run_command([exe, "solve", "--instance", inst, "--impl", "omp", "--chains", "24", "--scenarios", "512",
                     "--iters", "600", "--epochs", "4", "--lambda", str(lam), "--seed", "42", "--out", tmp_json])
        if os.path.exists(tmp_json):
            with open(tmp_json) as f: d = json.load(f)
            os.remove(tmp_json)
            records.append({
                "lambda": lam,
                "J_out_of_sample": d["J_out_of_sample"],
                "mean_cost": d["mean_cost"],
                "std_cost": d["std_cost"],
                "p95": d["p95"],
                "cvar95": d["cvar95"],
                "miss_prob": d["miss_prob"]
            })
            print(f"Lambda={lam:.1f} | OOS J={d['J_out_of_sample']:.1f} | Mean={d['mean_cost']:.1f} | Std={d['std_cost']:.1f}")

    csv_path = "results/e9_sensitivity.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        w.writeheader()
        w.writerows(records)
    print(f"Saved {csv_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true", help="Run all experiments")
    parser.add_argument("--e2", action="store_true", help="Run E2 baselines")
    parser.add_argument("--e3", action="store_true", help="Run E3 OpenMP scaling")
    parser.add_argument("--e4", action="store_true", help="Run E4 throughput scaling")
    parser.add_argument("--e8", action="store_true", help="Run E8 value of uncertainty")
    parser.add_argument("--e9", action="store_true", help="Run E9 sensitivity")
    args = parser.parse_args()

    exe = get_executable()
    if args.all or not any([args.e2, args.e3, args.e4, args.e8, args.e9]):
        exp_e2_baselines(exe)
        exp_e3_openmp_scaling(exe)
        exp_e4_throughput(exe)
        exp_e8_value_of_uncertainty(exe)
        exp_e9_sensitivity(exe)
    else:
        if args.e2: exp_e2_baselines(exe)
        if args.e3: exp_e3_openmp_scaling(exe)
        if args.e4: exp_e4_throughput(exe)
        if args.e8: exp_e8_value_of_uncertainty(exe)
        if args.e9: exp_e9_sensitivity(exe)

if __name__ == "__main__":
    main()
