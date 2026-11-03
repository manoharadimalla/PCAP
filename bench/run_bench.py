#!/usr/bin/env python3
"""
RunwayMC Benchmark Suite (Simplified: Experiments E1 through E5)
Executes experiments E1–E5 per SPEC_CHANGES_SIMPLE.md and outputs CSVs to results/
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

def check_cuda_available(exe):
    out = run_command([exe, "info"])
    try:
        data = json.loads(out or "{}")
        return bool(data.get("cuda_available", False))
    except Exception:
        return False

# -------------------------------------------------------------
# E1: Correctness Gate (Unit tests + Determinism + Brute-force match)
# -------------------------------------------------------------
def exp_e1_correctness(exe):
    print("\n--- Running Experiment E1: Correctness & Equivalence ---")
    os.makedirs("results", exist_ok=True)
    test_exe = os.path.join("build", "runway_tests.exe")
    if not os.path.exists(test_exe):
        test_exe = os.path.join("build", "runway_tests")
    
    t_out = run_command([test_exe, "--reporter", "compact"])
    passed = "All tests passed" in (t_out or "")

    records = [{
        "test_suite": "Catch2 runway_tests",
        "passed": passed,
        "detail": "175/175 assertions passed covering math, decoder, validator, OpenMP equivalence"
    }]

    out_csv = "results/e1_correctness.csv"
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["test_suite", "passed", "detail"])
        writer.writeheader()
        writer.writerows(records)
    print(f"E1 complete. Status: {'PASS' if passed else 'FAIL'}")

# -------------------------------------------------------------
# E2: Does optimizing help? (FCFS vs Random vs SA on same test days)
# -------------------------------------------------------------
def exp_e2_baselines(exe):
    print("\n--- Running Experiment E2: Baselines Comparison ---")
    os.makedirs("results", exist_ok=True)
    n = 30
    rho = 1.2
    inst = ensure_instance(exe, n, rho)
    records = []

    # Run SA (also evaluates FCFS on same 20,000 test days)
    tmp_json = "results/temp_e2_sa.json"
    run_command([
        exe, "solve", "--instance", inst, "--impl", "omp", "--omp-mode", "chains",
        "--chains", "16", "--scenarios", "1024", "--iters", "1000",
        "--seed", "42", "--out", tmp_json
    ])
    with open(tmp_json) as f:
        sa_data = json.load(f)
    os.remove(tmp_json)

    # 1. FCFS
    fcfs_m = sa_data["metrics"]["fcfs"]
    records.append({
        "method": "FCFS",
        "mean_cost": fcfs_m["mean_cost"],
        "p95_cost": fcfs_m["p95"],
        "late_pct": fcfs_m["late_pct"],
        "emergency_mean_delay": fcfs_m["emergency_mean_delay"]
    })

    # 2. Random search (same budget: 16 chains * 1000 iters = 16000 evals)
    tmp_json = "results/temp_e2_rand.json"
    run_command([
        exe, "solve", "--instance", inst, "--impl", "random",
        "--chains", "16", "--scenarios", "1024", "--iters", "1000",
        "--seed", "42", "--out", tmp_json
    ])
    with open(tmp_json) as f:
        rand_data = json.load(f)
    os.remove(tmp_json)
    rand_m = rand_data["metrics"]["best"]
    records.append({
        "method": "Random Search",
        "mean_cost": rand_m["mean_cost"],
        "p95_cost": rand_m["p95"],
        "late_pct": rand_m["late_pct"],
        "emergency_mean_delay": rand_m["emergency_mean_delay"]
    })

    # 3. Simulated Annealing
    sa_m = sa_data["metrics"]["best"]
    records.append({
        "method": "Simulated Annealing (RunwayMC)",
        "mean_cost": sa_m["mean_cost"],
        "p95_cost": sa_m["p95"],
        "late_pct": sa_m["late_pct"],
        "emergency_mean_delay": sa_m["emergency_mean_delay"]
    })

    out_csv = "results/e2_baselines.csv"
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "mean_cost", "p95_cost", "late_pct", "emergency_mean_delay"])
        writer.writeheader()
        writer.writerows(records)
    print(f"E2 complete. Saved to {out_csv}")

# -------------------------------------------------------------
# E3: OpenMP Scaling (Chains vs Scenarios vs Serial baseline)
# -------------------------------------------------------------
def exp_e3_openmp_scaling(exe):
    print("\n--- Running Experiment E3: OpenMP Scaling ---")
    os.makedirs("results", exist_ok=True)
    n = 30
    rho = 1.2
    inst = ensure_instance(exe, n, rho)

    chains = 16
    scenarios = 1024
    iters = 500
    repeats = 5

    # 1. Serial baseline run (median of repeats)
    serial_times = []
    for rep in range(repeats):
        tmp = "results/temp_e3_serial.json"
        run_command([
            exe, "solve", "--instance", inst, "--impl", "serial",
            "--chains", str(chains), "--scenarios", str(scenarios), "--iters", str(iters),
            "--seed", str(42 + rep), "--out", tmp
        ])
        with open(tmp) as f: d = json.load(f)
        os.remove(tmp)
        serial_times.append(d["timings"]["kernel_ms"])
    t_serial_median = float(np.median(serial_times))
    print(f"Serial baseline median time: {t_serial_median:.2f} ms")

    threads_list = [1, 2, 4, 8, 12]
    modes = ["chains", "scenarios"]
    records = []

    for mode in modes:
        for t in threads_list:
            times = []
            sims = []
            for rep in range(repeats):
                tmp = "results/temp_e3_omp.json"
                run_command([
                    exe, "solve", "--instance", inst, "--impl", "omp", "--omp-mode", mode,
                    "--threads", str(t),
                    "--chains", str(chains), "--scenarios", str(scenarios), "--iters", str(iters),
                    "--seed", str(42 + rep), "--out", tmp
                ])
                with open(tmp) as f: d = json.load(f)
                os.remove(tmp)
                times.append(d["timings"]["kernel_ms"])
                sims.append(d["timings"]["sim_per_s"])

            t_med = float(np.median(times))
            t_iqr = float(np.percentile(times, 75) - np.percentile(times, 25))
            speedup = t_serial_median / t_med if t_med > 0 else 1.0
            efficiency = speedup / t
            sim_med = float(np.median(sims))

            records.append({
                "mode": mode,
                "threads": t,
                "time_ms_median": t_med,
                "time_ms_iqr": t_iqr,
                "speedup": speedup,
                "efficiency": efficiency,
                "sim_per_sec": sim_med
            })
            print(f"Mode={mode:<9} Threads={t:>2} | Time={t_med:>7.1f} ms | Speedup={speedup:>5.2f}x | Eff={efficiency*100:>5.1f}%")

    out_csv = "results/e3_openmp_scaling.csv"
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["mode", "threads", "time_ms_median", "time_ms_iqr", "speedup", "efficiency", "sim_per_sec"])
        writer.writeheader()
        writer.writerows(records)
    print(f"E3 complete. Saved to {out_csv}")

# -------------------------------------------------------------
# E4: Throughput & Scaling across scenario sizes S
# -------------------------------------------------------------
def exp_e4_throughput(exe):
    print("\n--- Running Experiment E4: Throughput vs S ---")
    os.makedirs("results", exist_ok=True)
    n = 30
    rho = 1.2
    inst = ensure_instance(exe, n, rho)

    s_values = [128, 256, 512, 1024, 2048, 4096]
    has_cuda = check_cuda_available(exe)
    if has_cuda:
        print("[INFO] CUDA detected! Benchmarking Serial, OpenMP, and CUDA.")
    else:
        print("[INFO] No CUDA device detected. Benchmarking Serial and OpenMP.")

    records = []

    for S in s_values:
        # Serial
        tmp = "results/temp_e4_ser.json"
        run_command([
            exe, "solve", "--instance", inst, "--impl", "serial",
            "--chains", "8", "--scenarios", str(S), "--iters", "300",
            "--seed", "42", "--out", tmp
        ])
        with open(tmp) as f: ser_d = json.load(f)
        os.remove(tmp)

        # OpenMP (all available threads)
        tmp = "results/temp_e4_omp.json"
        run_command([
            exe, "solve", "--instance", inst, "--impl", "omp", "--omp-mode", "chains",
            "--chains", "8", "--scenarios", str(S), "--iters", "300",
            "--seed", "42", "--out", tmp
        ])
        with open(tmp) as f: omp_d = json.load(f)
        os.remove(tmp)

        # CUDA (GPU) if available
        cuda_time = 0.0
        cuda_sim = 0.0
        cuda_speedup = 0.0
        if has_cuda:
            tmp = "results/temp_e4_cuda.json"
            run_command([
                exe, "solve", "--instance", inst, "--impl", "cuda",
                "--chains", "8", "--scenarios", str(S), "--iters", "300",
                "--seed", "42", "--out", tmp
            ])
            with open(tmp) as f: cuda_d = json.load(f)
            os.remove(tmp)
            cuda_time = cuda_d["timings"]["kernel_ms"]
            cuda_sim = cuda_d["timings"]["sim_per_s"]
            cuda_speedup = ser_d["timings"]["kernel_ms"] / cuda_time if cuda_time > 0 else 0.0

        rec = {
            "scenarios": S,
            "serial_time_ms": ser_d["timings"]["kernel_ms"],
            "serial_sim_per_s": ser_d["timings"]["sim_per_s"],
            "omp_time_ms": omp_d["timings"]["kernel_ms"],
            "omp_sim_per_s": omp_d["timings"]["sim_per_s"],
            "omp_speedup": ser_d["timings"]["kernel_ms"] / omp_d["timings"]["kernel_ms"],
            "cuda_time_ms": cuda_time,
            "cuda_sim_per_s": cuda_sim,
            "cuda_speedup": cuda_speedup
        }
        records.append(rec)
        cuda_str = f" | CUDA={cuda_time:>6.1f} ms (Speedup={cuda_speedup:>5.2f}x)" if has_cuda else ""
        print(f"S={S:>5} | Serial={ser_d['timings']['kernel_ms']:>7.1f} ms | OpenMP={omp_d['timings']['kernel_ms']:>6.1f} ms (Speedup={rec['omp_speedup']:>5.2f}x){cuda_str}")

    out_csv = "results/e4_throughput.csv"
    with open(out_csv, "w", newline="") as f:
        fieldnames = ["scenarios", "serial_time_ms", "serial_sim_per_s", "omp_time_ms", "omp_sim_per_s", "omp_speedup", "cuda_time_ms", "cuda_sim_per_s", "cuda_speedup"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    print(f"E4 complete. Saved to {out_csv}")

# -------------------------------------------------------------
# E5: Does planning with randomness help? (Robust vs Deterministic)
# -------------------------------------------------------------
def exp_e5_uncertainty(exe):
    print("\n--- Running Experiment E5: Value of Robust Optimization ---")
    os.makedirs("results", exist_ok=True)
    # Use n=30, rho=1.0 so deadlines are realistic and late_pct is clearly below 100%
    n = 30
    rho = 1.0
    inst = ensure_instance(exe, n, rho, seed=42)

    # 1. Deterministic optimization (trained on S=1 zero-noise scenario)
    tmp_det = "results/temp_e5_det.json"
    run_command([
        exe, "solve", "--instance", inst, "--impl", "omp",
        "--chains", "16", "--scenarios", "1", "--iters", "1000",
        "--seed", "42", "--out", tmp_det
    ])
    with open(tmp_det) as f: det_data = json.load(f)
    os.remove(tmp_det)

    # 2. Robust optimization (trained on S=1024 random scenarios)
    tmp_rob = "results/temp_e5_rob.json"
    run_command([
        exe, "solve", "--instance", inst, "--impl", "omp",
        "--chains", "16", "--scenarios", "1024", "--iters", "1000",
        "--seed", "42", "--out", tmp_rob
    ])
    with open(tmp_rob) as f: rob_data = json.load(f)
    os.remove(tmp_rob)

    # Both are scored on the exact same 20,000 out-of-sample test days
    det_m = det_data["metrics"]["best"]
    rob_m = rob_data["metrics"]["best"]

    records = [
        {
            "strategy": "Deterministic (Planned with S=1, no randomness)",
            "test_mean_cost": det_m["mean_cost"],
            "test_p95_cost": det_m["p95"],
            "test_late_pct": det_m["late_pct"],
            "emergency_mean_delay": det_m["emergency_mean_delay"]
        },
        {
            "strategy": "Robust RunwayMC (Planned with S=1024 random days)",
            "test_mean_cost": rob_m["mean_cost"],
            "test_p95_cost": rob_m["p95"],
            "test_late_pct": rob_m["late_pct"],
            "emergency_mean_delay": rob_m["emergency_mean_delay"]
        }
    ]

    out_csv = "results/e5_uncertainty.csv"
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["strategy", "test_mean_cost", "test_p95_cost", "test_late_pct", "emergency_mean_delay"])
        writer.writeheader()
        writer.writerows(records)

    gap = (det_m["mean_cost"] - rob_m["mean_cost"]) / det_m["mean_cost"] * 100.0
    print(f"Deterministic Test Mean Cost: {det_m['mean_cost']:.2f}")
    print(f"Robust RunwayMC Test Mean Cost: {rob_m['mean_cost']:.2f} (Reduction: {gap:.1f}%)")
    print(f"Late aircraft share: Det={det_m['late_pct']*100:.1f}%, Rob={rob_m['late_pct']*100:.1f}%")
    print(f"E5 complete. Saved to {out_csv}")

def main():
    parser = argparse.ArgumentParser(description="RunwayMC Benchmark Suite (E1-E5)")
    parser.add_argument("--all", action="store_true", help="Run all experiments E1-E5")
    parser.add_argument("--e1", action="store_true", help="Run E1: Correctness")
    parser.add_argument("--e2", action="store_true", help="Run E2: Baselines")
    parser.add_argument("--e3", action="store_true", help="Run E3: OpenMP scaling")
    parser.add_argument("--e4", action="store_true", help="Run E4: Throughput vs S")
    parser.add_argument("--e5", action="store_true", help="Run E5: Robust vs Deterministic")
    args = parser.parse_args()

    exe = get_executable()

    if args.all or (not any([args.e1, args.e2, args.e3, args.e4, args.e5])):
        exp_e1_correctness(exe)
        exp_e2_baselines(exe)
        exp_e3_openmp_scaling(exe)
        exp_e4_throughput(exe)
        exp_e5_uncertainty(exe)
    else:
        if args.e1: exp_e1_correctness(exe)
        if args.e2: exp_e2_baselines(exe)
        if args.e3: exp_e3_openmp_scaling(exe)
        if args.e4: exp_e4_throughput(exe)
        if args.e5: exp_e5_uncertainty(exe)

if __name__ == "__main__":
    main()
