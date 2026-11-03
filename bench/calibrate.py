#!/usr/bin/env python3
"""
RunwayMC Calibration Script (Section 6 & 8.2)
Confirms that FCFS is at least 10% worse than the best schedule found.
Outputs results/calibration.csv.
"""

import os
import sys
import subprocess
import csv
import json

def get_executable():
    paths = [
        os.path.join("build", "runway.exe"),
        os.path.join("build", "runway"),
        os.path.join(".", "runway.exe"),
        os.path.join(".", "runway")
    ]
    for p in paths:
        if os.path.exists(p):
            return p
    raise FileNotFoundError("Could not find runway executable. Build the project first.")

def run_command(cmd):
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"Error running command: {' '.join(cmd)}\n{res.stderr}")
        sys.exit(1)
    return res.stdout

def main():
    exe = get_executable()
    os.makedirs("data/instances", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    n_values = [20, 50]
    rho_values = [1.0, 1.2, 1.5]
    seed = 42

    records = []
    print("=" * 60)
    print("RunwayMC Calibration Gate Verification")
    print("=" * 60)

    for n in n_values:
        for rho in rho_values:
            inst_path = f"data/instances/n{n}_r{int(rho*10)}_s{seed}.csv"
            # 1. Generate instance
            gen_cmd = [exe, "gen", "--n", str(n), "--rho", str(rho), "--seed", str(seed), "--out", inst_path]
            run_command(gen_cmd)

            # 2. Run solver with 32 chains, 2500 iterations
            res_json = f"results/calib_temp_n{n}_r{int(rho*10)}.json"
            solve_cmd = [
                exe, "solve", "--instance", inst_path,
                "--impl", "omp", "--omp-mode", "chains",
                "--chains", "32", "--scenarios", "512",
                "--iters", "2500",
                "--seed", str(seed), "--out", res_json
            ]
            run_command(solve_cmd)

            with open(res_json, "r") as f:
                res_data = json.load(f)

            cost_fcfs = res_data["metrics"]["fcfs"]["mean_cost"]
            cost_best = res_data["metrics"]["best"]["mean_cost"]
            if os.path.exists(res_json):
                os.remove(res_json)

            gap = (cost_fcfs - cost_best) / cost_fcfs if cost_fcfs > 0 else 0.0
            passed = gap >= 0.10

            records.append({
                "n": n,
                "rho": rho,
                "cost_fcfs": cost_fcfs,
                "cost_best": cost_best,
                "gap_pct": gap * 100.0,
                "gate_passed": passed
            })

            status = "PASS" if passed else "FAIL"
            print(f"n={n:>2}, rho={rho:>3.1f} | FCFS={cost_fcfs:>10.2f}, Best={cost_best:>10.2f} | Gap={gap*100:>6.2f}% | [{status}]")

    csv_path = "results/calibration.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["n", "rho", "cost_fcfs", "cost_best", "gap_pct", "gate_passed"])
        writer.writeheader()
        writer.writerows(records)

    all_passed = all(r["gate_passed"] for r in records)
    print("=" * 60)
    print(f"Calibration Summary: {'ALL PASSED (Gate >= 10% satisfied)' if all_passed else 'SOME FAILED'}")
    print(f"Saved results to: {csv_path}")

if __name__ == "__main__":
    main()
