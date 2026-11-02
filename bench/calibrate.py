#!/usr/bin/env python3
"""
RunwayMC Calibration Script (Section 8.2)
Verifies that all experiment instances satisfy the calibration gate:
gap = (J_FCFS - J_best) / J_FCFS >= 10% (0.10)
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
    print("RunwayMC Calibration Gate Verification (Section 8.2)")
    print("=" * 60)

    for n in n_values:
        for rho in rho_values:
            inst_path = f"data/instances/n{n}_r{int(rho*10)}_s{seed}.csv"
            # 1. Generate instance
            gen_cmd = [exe, "gen", "--n", str(n), "--rho", str(rho), "--seed", str(seed), "--out", inst_path]
            run_command(gen_cmd)

            # 2. Evaluate FCFS
            eval_out = run_command([exe, "eval", "--instance", inst_path, "--perm", "fcfs", "--scenarios", "1024", "--seed", str(seed)])
            j_fcfs = None
            for line in eval_out.splitlines():
                if line.startswith("J:"):
                    j_fcfs = float(line.split(":")[1].strip())
                    break
            if j_fcfs is None:
                raise ValueError(f"Failed to parse J from eval: {eval_out}")

            # 3. Run solver to find J_best
            res_json = f"results/calib_temp_n{n}_r{int(rho*10)}.json"
            solve_cmd = [
                exe, "solve", "--instance", inst_path,
                "--impl", "omp", "--omp-mode", "chains",
                "--chains", "32", "--scenarios", "512",
                "--iters", "1000", "--epochs", "8",
                "--seed", str(seed), "--out", res_json
            ]
            run_command(solve_cmd)

            with open(res_json, "r") as f:
                res_data = json.load(f)
            j_best = res_data["J_in_sample"]
            if os.path.exists(res_json):
                os.remove(res_json)

            gap = (j_fcfs - j_best) / j_fcfs if j_fcfs > 0 else 0.0
            passed = gap >= 0.10

            records.append({
                "n": n,
                "rho": rho,
                "instance": inst_path,
                "J_FCFS": round(j_fcfs, 4),
                "J_best": round(j_best, 4),
                "gap_percent": round(gap * 100.0, 2),
                "gate_passed": passed
            })

            status = "PASSED" if passed else "FAILED"
            print(f"n={n:3d}, rho={rho:.1f} | J_FCFS={j_fcfs:10.2f} | J_best={j_best:10.2f} | Gap={gap*100:6.2f}% | Gate: {status}")

    calib_csv = "results/calibration.csv"
    with open(calib_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["n", "rho", "instance", "J_FCFS", "J_best", "gap_percent", "gate_passed"])
        writer.writeheader()
        writer.writerows(records)

    print("=" * 60)
    all_passed = all(r["gate_passed"] for r in records)
    print(f"Calibration table written to {calib_csv}")
    if all_passed:
        print("ALL CALIBRATION GATES PASSED (gap >= 10%).")
    else:
        print("WARNING: Some instances did not achieve gap >= 10%.")
    print("=" * 60)

if __name__ == "__main__":
    main()
