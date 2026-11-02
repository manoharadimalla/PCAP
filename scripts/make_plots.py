#!/usr/bin/env python3
"""
RunwayMC Plot Generation Script (Section 10)
Reads CSV files from results/ and generates plots to results/figures/
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def setup_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 14,
        "grid.color": "#e0e0e0",
        "grid.linestyle": "--",
        "grid.linewidth": 0.7,
        "axes.grid": True,
        "figure.autolayout": True
    })

def plot_e2(out_dir="results/figures"):
    csv_file = "results/e2_baselines.csv"
    if not os.path.exists(csv_file): return
    df = pd.read_csv(csv_file)

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(df["rho"]))
    w = 0.2

    ax.bar(x - 1.5*w, df["J_FCFS"], width=w, label="FCFS", color="#4e79a7")
    ax.bar(x - 0.5*w, df["J_Random"], width=w, label="Random Search", color="#f28e2b")
    ax.bar(x + 0.5*w, df["J_SA"], width=w, label="Simulated Annealing (Ours)", color="#59a14f")
    ax.bar(x + 1.5*w, df["J_Brute"], width=w, label="Brute Force Optimum", color="#e15759")

    ax.set_xticks(x)
    ax.set_xticklabels([f"rho = {r}" for r in df["rho"]])
    ax.set_ylabel("Expected Robust Cost J(pi)")
    ax.set_title("E2: Baseline Comparison on n=8, S=500")
    ax.legend()

    for ext in ["png", "pdf"]:
        fig.savefig(os.path.join(out_dir, f"e2_baselines.{ext}"), dpi=300)
    plt.close(fig)
    print("Generated E2 baselines plot")

def plot_e3(out_dir="results/figures"):
    csv_file = "results/e3_openmp_scaling.csv"
    if not os.path.exists(csv_file): return
    df = pd.read_csv(csv_file)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    modes = df["mode"].unique()
    colors = {"chains": "#1f77b4", "scenarios": "#ff7f0e", "hybrid": "#2ca02c"}

    max_th = df["threads"].max()
    ax1.plot([1, max_th], [1, max_th], "k--", label="Ideal Linear", alpha=0.7)

    for m in modes:
        sub = df[df["mode"] == m]
        ax1.plot(sub["threads"], sub["speedup"], marker="o", label=f"O: {m}", color=colors.get(m, "blue"))
        ax2.plot(sub["threads"], sub["efficiency"], marker="s", label=f"O: {m}", color=colors.get(m, "blue"))

    ax1.set_xlabel("OpenMP Threads")
    ax1.set_ylabel("Speedup vs 1-Thread Chains")
    ax1.set_title("OpenMP Strong Scaling (Speedup)")
    ax1.legend()

    ax2.set_xlabel("OpenMP Threads")
    ax2.set_ylabel("Parallel Efficiency (Speedup / Threads)")
    ax2.set_title("OpenMP Parallel Efficiency")
    ax2.set_ylim(0, 1.1)
    ax2.legend()

    for ext in ["png", "pdf"]:
        fig.savefig(os.path.join(out_dir, f"e3_openmp_scaling.{ext}"), dpi=300)
    plt.close(fig)
    print("Generated E3 OpenMP scaling plot")

def plot_e4(out_dir="results/figures"):
    csv_file = "results/e4_throughput.csv"
    if not os.path.exists(csv_file): return
    df = pd.read_csv(csv_file)

    fig, ax = plt.subplots(figsize=(8, 5))
    for n in df["n"].unique():
        sub = df[df["n"] == n]
        ax.plot(sub["S"], sub["omp_sim_per_sec"], marker="^", label=f"n = {n} (OpenMP)")

    ax.set_xlabel("Number of Scenarios (S)")
    ax.set_ylabel("Throughput (Scenario Simulations / sec)")
    ax.set_title("E4: Parallel Evaluation Throughput Scaling")
    ax.set_xscale("log", base=2)
    ax.legend()

    for ext in ["png", "pdf"]:
        fig.savefig(os.path.join(out_dir, f"e4_throughput.{ext}"), dpi=300)
    plt.close(fig)
    print("Generated E4 throughput plot")

def plot_e8(out_dir="results/figures"):
    csv_file = "results/e8_uncertainty.csv"
    if not os.path.exists(csv_file): return
    df = pd.read_csv(csv_file)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    strat = df["strategy"]
    colors = ["#d95f02", "#1b9e77"]

    # Bar chart of OOS J and CVaR95
    x = np.arange(len(strat))
    w = 0.35
    ax1.bar(x - w/2, df["J_out_of_sample"], width=w, label="Out-of-Sample J", color=colors[0])
    ax1.bar(x + w/2, df["cvar95"], width=w, label="CVaR95 (Tail Risk)", color=colors[1])
    ax1.set_xticks(x)
    ax1.set_xticklabels(["Deterministic", "Robust (Ours)"])
    ax1.set_ylabel("Expected Cost")
    ax1.set_title("Cost & Tail Risk Under Uncertainty")
    ax1.legend()

    # Miss Probability
    ax2.bar(strat, df["miss_prob"] * 100, color=["#e7298a", "#7570b3"], width=0.4)
    ax2.set_ylabel("Deadline Miss Probability (%)")
    ax2.set_title("Deadline Miss Risk Under Operational Disturbances")
    ax2.set_xticklabels(["Deterministic", "Robust (Ours)"])

    for ext in ["png", "pdf"]:
        fig.savefig(os.path.join(out_dir, f"e8_uncertainty.{ext}"), dpi=300)
    plt.close(fig)
    print("Generated E8 uncertainty plot")

def main():
    setup_style()
    out_dir = "results/figures"
    os.makedirs(out_dir, exist_ok=True)
    plot_e2(out_dir)
    plot_e3(out_dir)
    plot_e4(out_dir)
    plot_e8(out_dir)
    print(f"All available plots generated in {out_dir}")

if __name__ == "__main__":
    main()
