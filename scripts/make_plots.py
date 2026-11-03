#!/usr/bin/env python3
"""
RunwayMC Plotting Script
Generates figures for experiments E2, E3, E4, E5 to results/figures/
"""

import os
import csv
import matplotlib.pyplot as plt
import numpy as np

os.makedirs("results/figures", exist_ok=True)
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

# -------------------------------------------------------------
# Figure E2: Baselines
# -------------------------------------------------------------
if os.path.exists("results/e2_baselines.csv"):
    methods = []
    mean_costs = []
    p95_costs = []
    with open("results/e2_baselines.csv") as f:
        reader = csv.DictReader(f)
        for r in reader:
            methods.append(r["method"])
            mean_costs.append(float(r["mean_cost"]))
            p95_costs.append(float(r["p95_cost"]))

    x = np.arange(len(methods))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 4.5))
    rects1 = ax.bar(x - width/2, mean_costs, width, label="Mean Cost", color="#2b5c8f")
    rects2 = ax.bar(x + width/2, p95_costs, width, label="P95 Tail Cost", color="#d95f02")

    ax.set_ylabel("Cost (seconds delay penalty)")
    ax.set_title("E2: Schedule Quality Comparison on 20,000 Test Days (n=30)", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=10, ha="right")
    ax.legend()
    fig.tight_layout()
    fig.savefig("results/figures/e2_baselines.png", dpi=300)
    fig.savefig("results/figures/e2_baselines.pdf")
    plt.close(fig)
    print("Generated E2 figure")

# -------------------------------------------------------------
# Figure E3: OpenMP Scaling
# -------------------------------------------------------------
if os.path.exists("results/e3_openmp_scaling.csv"):
    data = {"chains": ([], []), "scenarios": ([], [])}
    with open("results/e3_openmp_scaling.csv") as f:
        reader = csv.DictReader(f)
        for r in reader:
            mode = r["mode"]
            data[mode][0].append(int(r["threads"]))
            data[mode][1].append(float(r["speedup"]))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(data["chains"][0], data["chains"][1], "o-", label="OpenMP Chains (Coarse-grained)", color="#2b5c8f", lw=2)
    ax.plot(data["scenarios"][0], data["scenarios"][1], "s--", label="OpenMP Scenarios (Fine-grained)", color="#d95f02", lw=2)
    ax.plot(data["chains"][0], data["chains"][0], "k:", alpha=0.5, label="Linear Ideal")

    ax.set_xlabel("CPU Threads")
    ax.set_ylabel("Speedup vs. Serial Baseline")
    ax.set_title("E3: OpenMP Multi-Core Scaling on Intel Core i5-12450H", fontweight="bold")
    ax.set_xticks(data["chains"][0])
    ax.legend()
    fig.tight_layout()
    fig.savefig("results/figures/e3_openmp_scaling.png", dpi=300)
    fig.savefig("results/figures/e3_openmp_scaling.pdf")
    plt.close(fig)
    print("Generated E3 figure")

# -------------------------------------------------------------
# Figure E4: Throughput vs S
# -------------------------------------------------------------
if os.path.exists("results/e4_throughput.csv"):
    s_vals, ser_times, omp_times = [], [], []
    with open("results/e4_throughput.csv") as f:
        reader = csv.DictReader(f)
        for r in reader:
            s_vals.append(int(r["scenarios"]))
            ser_times.append(float(r["serial_time_ms"]))
            omp_times.append(float(r["omp_time_ms"]))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(s_vals, ser_times, "o-", label="Serial Reference", color="#e7298a", lw=2)
    ax.plot(s_vals, omp_times, "s-", label="OpenMP (12 threads)", color="#2b5c8f", lw=2)

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Scenarios (S)")
    ax.set_ylabel("Execution Time (ms, log scale)")
    ax.set_title("E4: Execution Time Scaling with Scenario Count S", fontweight="bold")
    ax.legend()
    fig.tight_layout()
    fig.savefig("results/figures/e4_throughput.png", dpi=300)
    fig.savefig("results/figures/e4_throughput.pdf")
    plt.close(fig)
    print("Generated E4 figure")

# -------------------------------------------------------------
# Figure E5: Robust vs Deterministic
# -------------------------------------------------------------
if os.path.exists("results/e5_uncertainty.csv"):
    strategies = []
    mean_c, p95_c = [], []
    with open("results/e5_uncertainty.csv") as f:
        reader = csv.DictReader(f)
        for r in reader:
            strategies.append("Deterministic\n(no randomness)" if "Deterministic" in r["strategy"] else "Robust RunwayMC\n(S=1024 training)")
            mean_c.append(float(r["test_mean_cost"]))
            p95_c.append(float(r["test_p95_cost"]))

    x = np.arange(len(strategies))
    width = 0.35

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.bar(x - width/2, mean_c, width, label="Test Mean Cost", color="#7570b3")
    ax.bar(x + width/2, p95_c, width, label="Test P95 Cost", color="#1b9e77")

    ax.set_ylabel("Cost on 20,000 Unseen Test Days")
    ax.set_title("E5: Value of Planning with Randomness", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(strategies)
    ax.legend()
    fig.tight_layout()
    fig.savefig("results/figures/e5_uncertainty.png", dpi=300)
    fig.savefig("results/figures/e5_uncertainty.pdf")
    plt.close(fig)
    print("Generated E5 figure")

print("All benchmark figures successfully generated!")
