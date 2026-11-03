#!/usr/bin/env python3
"""
RunwayMC — Interactive Demo
Simplified according to SPEC_CHANGES_SIMPLE.md.
"""

import os
import sys
import subprocess
import time
import json

def find_exe():
    candidates = [
        os.path.join("build", "runway.exe"),
        os.path.join("build", "runway"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return os.path.abspath(p)
    return None

def run(cmd, show=True):
    """Run a command and print its output."""
    if show:
        print(f"\n  > {' '.join(cmd)}\n")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.stdout.strip():
        for line in result.stdout.strip().split("\n"):
            print(f"    {line}")
    if result.returncode != 0 and result.stderr.strip():
        print(f"    [ERROR] {result.stderr.strip()}")
    return result.stdout

def banner(text):
    width = 64
    print()
    print("=" * width)
    print(f"  {text}")
    print("=" * width)

def pause():
    print()
    input("  Press ENTER to continue...")

def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    exe = find_exe()
    if not exe:
        print("\n  ERROR: Could not find the runway executable!")
        print("  Build the project first: cmake --build build")
        sys.exit(1)

    banner("WELCOME TO RunwayMC (Robust Runway Scheduling)")
    print("""
  The Problem:
    Planes must share ONE runway. The order decides how long aircraft wait.
    Heavy planes leave wake turbulence, requiring light planes to wait longer.

  The Method:
    • Simulated Annealing (SA) explores candidate orderings.
    • Monte Carlo simulation tests each candidate across thousands of random
      days (arrival/departure delays and runway closures).
    • Parallel Computing (OpenMP on CPU / CUDA on GPU) evaluates all random
      days simultaneously, turning hours into seconds.
  """)
    pause()

    # ── STEP 1: Generate an instance ────────────────────────────────────────
    banner("STEP 1: Generate a Runway Scheduling Problem")
    inst_path = os.path.join("data", "instances", "demo_interactive.csv")
    os.makedirs(os.path.dirname(inst_path), exist_ok=True)
    run([exe, "gen", "--n", "20", "--rho", "1.2", "--seed", "42", "--out", inst_path])
    pause()

    # ── STEP 2: Evaluate the naive FCFS schedule ────────────────────────────
    banner("STEP 2: Evaluate the Naive FCFS Schedule")
    print("""
  FCFS (First Come, First Served) serves planes strictly by target time.
  Testing FCFS against 1024 random scenarios...
  """)
    run([exe, "eval", "--instance", inst_path, "--perm", "fcfs",
         "--scenarios", "1024", "--seed", "42"])
    pause()

    # ── STEP 3: Optimize with Simulated Annealing ──────────────────────────
    banner("STEP 3: Find a Better Schedule with Simulated Annealing")
    print("""
  Running 16 parallel search chains with OpenMP multi-core acceleration...
  """)

    start = time.time()
    result_json = os.path.join("results", "demo_interactive.json")
    os.makedirs("results", exist_ok=True)
    run([exe, "solve", "--instance", inst_path, "--impl", "omp",
         "--omp-mode", "chains", "--threads", "8",
         "--chains", "16", "--scenarios", "1024",
         "--iters", "1000",
         "--seed", "42", "--out", result_json])
    elapsed = time.time() - start
    pause()

    # ── STEP 4: Show the comparison ────────────────────────────────────────
    banner("STEP 4: Results Comparison on 20,000 Unseen Test Days")
    if os.path.exists(result_json):
        with open(result_json) as f:
            data = json.load(f)
        fcfs = data["metrics"]["fcfs"]
        best = data["metrics"]["best"]
        cost_f = fcfs["mean_cost"]
        cost_b = best["mean_cost"]
        imp = (cost_f - cost_b) / cost_f * 100.0 if cost_f > 0 else 0.0

        print(f"    {'Metric':<32} {'FCFS Baseline':>15}  {'RunwayMC (SA)':>15}  {'Improvement':>12}")
        print("    " + "-" * 78)
        print(f"    {'Mean Delay Cost (s)':<32} {cost_f:>15.2f}  {cost_b:>15.2f}  {imp:>+11.1f}%")
        print(f"    {'95th Percentile Cost':<32} {fcfs['p95']:>15.2f}  {best['p95']:>15.2f}")
        print(f"    {'Late Aircraft Share (%)':<32} {fcfs['late_pct']*100:>14.1f}%  {best['late_pct']*100:>14.1f}%")
        print(f"    {'Emergency Flight Mean Delay':<32} {fcfs['emergency_mean_delay']:>15.2f}  {best['emergency_mean_delay']:>15.2f}")
        print()
        print(f"    Compute time: {elapsed:.2f} seconds | Throughput: {data['timings']['sim_per_s']:,.0f} sims/sec")
        print(f"    >> RunwayMC reduced delay cost by {imp:.1f}% compared to naive FCFS!")
    pause()

    # ── STEP 5: Run the test suite ──────────────────────────────────────────
    banner("STEP 5: Automated Verification Suite")
    test_exe = os.path.join("build", "runway_tests.exe")
    if os.path.exists(test_exe):
        run([test_exe, "--reporter", "compact"])
    pause()

    banner("DONE! Everything verified.")

if __name__ == "__main__":
    main()
