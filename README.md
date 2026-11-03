# RunwayMC: Robust Runway Scheduling under Uncertainty

> **Academic Simulation Disclaimer:** This is an academic research simulation using simplified wake vortex separation standards. It is not validated for, and must not be used in, real air-traffic operations.

RunwayMC is a parallel Monte Carlo simulation and optimization engine for robust single-runway aircraft sequencing under uncertainty (exponential flight delays and random runway closures).

---

## Highlights & Features
- **Parallel Computing Core:** Three unified implementations sharing a single exact simulation engine: **Serial** (1 thread baseline), **OpenMP** (multi-core CPU), and **CUDA** (massively parallel GPU acceleration).
- **GUI-to-Cloud Workflow:** Create and visualize aircraft schedules in the native Windows desktop GUI, export to CSV, and run large-scale CUDA solves on NVIDIA GPUs via Google Colab.
- **Robust vs. Deterministic:** Achieves over **44% delay cost reduction** on 20,000 unseen test days compared to deterministic planning.
- **Automated Benchmarks & Visualizations:** Automated harness for experiments E1 through E5 generating publication-quality speedup and scaling curves comparing Serial, OpenMP, and CUDA.
- **Tested & Verified:** Validated with Catch2 unit tests, determinism checks, and numerical CPU $\leftrightarrow$ GPU equivalence.

---

## Quick Start

### 1. Run the Desktop GUI (Windows)
Double-click **`RUN_GUI.bat`** or run:
```powershell
python -m gui
```
*Features:* Real-time flight table editor, interactive solve launcher, Gantt timeline charts, delay distributions, and CSV import/export.

### 2. Run on Google Colab with NVIDIA GPU (CUDA)
Open **`scripts/colab_run.ipynb`** in Google Colab:
1. Set runtime to **T4 GPU** (`Runtime -> Change runtime type -> T4 GPU`).
2. Run cells in order to clone the repo, compile with CUDA + OpenMP, solve interactive flight batches, and render all benchmark charts.

### 3. Build & Test from Source (Local / Command-Line)
Requirements: C++17 compiler (GCC / Clang / MSVC), CMake $\ge$ 3.20.
```bash
# Build without CUDA (CPU + OpenMP)
cmake -B build -DCMAKE_BUILD_TYPE=Release -DUSE_CUDA=OFF -DUSE_OPENMP=ON
cmake --build build -j$(nproc)

# Run Catch2 tests
ctest --test-dir build --output-on-failure
```

---

## Benchmark Experiments (E1 – E5)

Run the full automated benchmark suite:
```bash
python3 bench/run_bench.py --all
python3 scripts/make_plots.py
```
This produces all result CSVs and generated plots in `results/figures/`:
* **`e2_baselines.png`:** Compares FCFS vs. Random Search vs. RunwayMC SA. Proves optimization cuts delays by 48%.
* **`e3_openmp_scaling.png`:** Evaluates multi-core CPU scaling across thread counts (Chains vs. Scenarios).
* **`e4_throughput.png` & `e4_cuda_speedup.png`:** Scaling across scenario volumes ($S \in [128, 4096]$) comparing **Serial vs. OpenMP vs. CUDA**.
* **`e5_uncertainty.png`:** Proves that planning under uncertainty reduces test-day delay costs by 44.3%.

---

## Project Structure & Documentation
- **`docs/EXPLAIN.md`:** ⭐ Comprehensive **Viva & Presentation Guide** with architecture diagrams, graph explanations, CPU vs GPU comparisons, and likely exam questions.
- **`report/REPORT.md`:** Full academic research paper and evaluation report.
- **`docs/SPEC.md`:** Technical specification of the exact simulation decoder and mathematical formulation.
- **`scripts/colab_run.ipynb`:** Self-contained Google Colab demonstration notebook.
