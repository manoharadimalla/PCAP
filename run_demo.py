#!/usr/bin/env python3
"""
RunwayMC — Interactive Demo
Just run this script and follow the prompts!

Usage:
    python run_demo.py
"""

import os
import sys
import subprocess
import time
import json

# ── Find the runway executable ──────────────────────────────────────────────
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
        print("  You need to build the project first. Run these commands:\n")
        print('    cmake -B build -G "Ninja" -DUSE_CUDA=OFF -DUSE_OPENMP=ON')
        print("    cmake --build build\n")
        print("  (Make sure cmake, ninja, and g++ are on your PATH)")
        sys.exit(1)

    banner("WELCOME TO RunwayMC")
    print("""
  This project solves the RUNWAY SCHEDULING problem:
  
  Imagine you're an air traffic controller with ONE runway.
  Planes are arriving and departing — each has a target time,
  a wake turbulence class (Heavy/Medium/Light), and a priority.
  
  You need to decide the ORDER in which planes use the runway
  so that total delay is minimized, even when random things
  happen (weather, delays, runway closures).
  
  RunwayMC uses:
    • Simulated Annealing (SA) — a smart search algorithm
    • Monte Carlo simulation — test each schedule against
      thousands of random "what-if" scenarios
    • OpenMP parallelism — use all your CPU cores at once
  """)
    pause()

    # ── STEP 1: Generate an instance ────────────────────────────────────────
    banner("STEP 1: Generate a Runway Scheduling Problem")
    print("""
  We'll create a problem with 20 aircraft (mix of arrivals and
  departures), with congestion level rho=1.2 (meaning demand
  exceeds capacity — planes have to wait!).
  """)

    inst_path = os.path.join("data", "instances", "demo_interactive.csv")
    os.makedirs(os.path.dirname(inst_path), exist_ok=True)
    run([exe, "gen", "--n", "20", "--rho", "1.2", "--seed", "42", "--out", inst_path])

    print("\n  Here's what the instance looks like (first few aircraft):\n")
    with open(inst_path, "r") as f:
        lines = f.readlines()
    print(f"    {'ID':>3}  {'Type':>4}  {'Wake':>4}  {'Ready':>8}  {'Target':>8}  {'Deadline':>8}  {'Priority':>9}  {'Weight':>6}")
    print("    " + "-" * 70)
    for line in lines[1:8]:  # Show first 7 aircraft
        parts = line.strip().split(",")
        if len(parts) >= 8:
            print(f"    {parts[0]:>3}  {parts[1]:>4}  {parts[2]:>4}  {parts[3]:>8}  {parts[4]:>8}  {parts[5]:>8}  {parts[6]:>9}  {parts[7]:>6}")
    print(f"    ... and {len(lines) - 8} more aircraft")
    pause()

    # ── STEP 2: Evaluate the naive FCFS schedule ────────────────────────────
    banner("STEP 2: Evaluate the Naive FCFS Schedule")
    print("""
  FCFS = "First Come, First Served" — just serve planes
  in the order they're scheduled. This is the simplest
  approach but often terrible because it ignores wake
  turbulence classes (a Heavy plane followed by a Light
  plane needs a HUGE 196-second gap!).
  
  We'll test this schedule against 1024 random scenarios...
  """)

    fcfs_out = run([exe, "eval", "--instance", inst_path, "--perm", "fcfs",
                    "--scenarios", "1024", "--seed", "42"])
    
    fcfs_j = None
    for line in fcfs_out.split("\n"):
        if "J:" in line:
            try:
                fcfs_j = float(line.split(":")[1].strip())
            except:
                pass
    pause()

    # ── STEP 3: Optimize with Simulated Annealing ──────────────────────────
    banner("STEP 3: Find a Better Schedule with Simulated Annealing")
    print("""
  Now we'll use SA (Simulated Annealing) to search for a
  MUCH better ordering. SA works by:
  
    1. Start from the FCFS schedule
    2. Make small changes (swap two planes, move one around)
    3. Evaluate each change against 1024 random scenarios
    4. Accept good changes, sometimes accept bad ones (to
       escape local optima)
    5. Gradually get pickier (cool the "temperature")
  
  We'll run 16 independent search chains in parallel using
  OpenMP (your CPU has multiple cores!).
  
  Running now... (this takes about 10-20 seconds)
  """)

    start = time.time()
    result_json = os.path.join("results", "demo_interactive.json")
    os.makedirs("results", exist_ok=True)
    run([exe, "solve", "--instance", inst_path, "--impl", "omp",
         "--omp-mode", "chains", "--threads", "8",
         "--chains", "16", "--scenarios", "1024",
         "--iters", "1000", "--epochs", "5",
         "--seed", "42", "--out", result_json])
    elapsed = time.time() - start

    pause()

    # ── STEP 4: Show the comparison ────────────────────────────────────────
    banner("STEP 4: Results Comparison")

    sa_j = None
    sa_mean = None
    sa_p95 = None
    sa_cvar = None
    sa_miss = None
    sa_sims = None
    if os.path.exists(result_json):
        with open(result_json) as f:
            data = json.load(f)
        sa_j = data.get("J_out_of_sample")
        sa_mean = data.get("mean_cost")
        sa_p95 = data.get("p95")
        sa_cvar = data.get("cvar95")
        sa_miss = data.get("miss_prob")
        sa_sims = data.get("sim_per_sec")

    print("""
  Here's how our optimized schedule compares to the naive FCFS:
  """)

    if fcfs_j and sa_j:
        improvement = (1.0 - sa_j / fcfs_j) * 100
        print(f"    {'Metric':<35} {'FCFS':>12}  {'RunwayMC SA':>12}  {'Change':>10}")
        print("    " + "-" * 75)
        print(f"    {'Robust Objective J(pi)':<35} {fcfs_j:>12,.1f}  {sa_j:>12,.1f}  {improvement:>+9.1f}%")
        if sa_mean:
            print(f"    {'Mean Delay Cost':<35} {'-':>12}  {sa_mean:>12,.1f}")
        if sa_p95:
            print(f"    {'95th Percentile (worst 5%)':<35} {'-':>12}  {sa_p95:>12,.1f}")
        if sa_cvar:
            print(f"    {'CVaR95 (tail catastrophic risk)':<35} {'-':>12}  {sa_cvar:>12,.1f}")
        if sa_miss is not None:
            print(f"    {'Deadline Miss Probability':<35} {'-':>12}  {sa_miss*100:>11.1f}%")
        print()
        print(f"    Compute time:  {elapsed:.1f} seconds")
        if sa_sims:
            print(f"    Throughput:    {sa_sims:,.0f} scenario-simulations/sec")
        print()
        print(f"    ** RunwayMC reduced delay cost by {improvement:.1f}% compared to FCFS! **")
    else:
        print("    (Could not parse results — check the output above)")

    pause()

    # ── STEP 5: Run the test suite ──────────────────────────────────────────
    banner("STEP 5: Run the Automated Test Suite")
    print("""
  The project includes a full Catch2 test suite that verifies:
  • Mathematical correctness (worked examples from the spec)
  • Runway closure edge cases
  • Schedule validation (wake separation for ALL pairs)
  • Move operator correctness (1,000,000 fuzzing iterations)
  • OpenMP equivalence to serial computation
  • SA vs brute-force optimality on small instances
  
  Running tests now...
  """)

    test_exe = os.path.join("build", "runway_tests.exe")
    if not os.path.exists(test_exe):
        test_exe = os.path.join("build", "runway_tests")
    if os.path.exists(test_exe):
        run([test_exe, "--reporter", "compact"])
    else:
        print("    Test executable not found. Run: cmake --build build")

    pause()

    # ── Done! ───────────────────────────────────────────────────────────────
    banner("DONE!")
    print("""
  That's RunwayMC! Here's a summary of the project:
  
  FILES YOU MIGHT WANT TO LOOK AT:
    • SPEC.md              — The full project specification
    • report/REPORT.md     — The research report with results
    • docs/DECISIONS.md    — Design decisions and phase log
    • results/figures/     — Generated charts (open the PNGs!)
    • results/*.csv        — Raw benchmark data
  
  COMMANDS YOU CAN RUN YOURSELF:
    build\\runway.exe gen   --n 100 --rho 1.5 --seed 1 --out my_instance.csv
    build\\runway.exe eval  --instance my_instance.csv --perm fcfs --scenarios 2048
    build\\runway.exe solve --instance my_instance.csv --impl omp --chains 32 --scenarios 1024 --iters 2000 --epochs 10 --seed 1
  
  FOR GPU (CUDA) EXECUTION:
    Open scripts/colab_run.ipynb in Google Colab with a GPU runtime.
  
  Happy scheduling! ✈️
  """)

if __name__ == "__main__":
    main()
