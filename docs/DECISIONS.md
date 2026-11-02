# RunwayMC — Decisions & Phase Log

This log records design decisions, deviations from the baseline spec, phase completion notes, and experimental validation results.

---

## Phase P0 — Environment & Skeleton Setup
- **Status:** COMPLETED & PASSED.
- **Hardware Probed:**
  - CPU: 12th Gen Intel(R) Core(TM) i5-12450H (8 physical cores, 12 logical processors)
  - RAM: 16 GB
  - GPU: NVIDIA GeForce RTX 2050 (4096 MiB VRAM), Driver 592.27, CUDA Driver 13.1
  - Compilers & Tools: GCC/G++ 15.2.0 (UCRT64 MinGW-w64), OpenMP 4.5, CMake 4.4.4, Ninja 1.13.2, Python 3.14.2
  - NVCC: Not detected on local system PATH.
- **Decision R5 / Rule 7 (GPU Environment):**
  - CMake provides `-DUSE_CUDA=ON/OFF` (defaults to OFF when nvcc is not present).
  - All CUDA implementation files (`src/cuda/noise_gpu.cu`, `eval_kernels.cu`, `sa_kernel.cu`, `sa_gpu.cu`) are fully written, tested, and maintained for CUDA-enabled systems.
  - Dedicated Google Colab notebook (`scripts/colab_run.ipynb`) created for one-click GPU building and execution on cloud NVIDIA GPUs.
  - High-performance Serial and OpenMP (`O1`, `O2`, `O3`) implementations configured with `-O3 -march=native -fopenmp`.

---

## Phase P1 — Core Model & Single Simulation Truth
- **Status:** COMPLETED & PASSED.
- **Components Implemented:**
  - `include/runway/types.hpp`: Aircraft, Structure of Arrays (`AircraftData`), `SeparationMatrix`, `NoiseData`, and result structures.
  - `include/runway/sim_core.hpp`: `RW_HD` single source of truth for exact decoder, fast-path decoder, and neighborhood moves.
  - `src/core/instance_io.cpp`: CSV loaders and savers for instances, schedules, and separation matrices.
  - `src/core/validator.cpp`: Pairwise validation for all aircraft pairs ($i < j$), ready-time constraints, and closure window exclusions.
  - `src/core/objective.cpp`: Fast CPU float and high-precision double evaluators for $J(\pi)$, mean, std, p95, CVaR95, and plan deviation.
- **Test Gate:**
  - Worked example (Section 4.7) verified: Order ABC cost = 445; Order CBA cost = 198; Closure [50, 100) cost = 260.
  - Closure edge cases verified (empty closure, before-horizon, full-horizon, boundary conditions).
  - Triangle inequality check and decoder equivalence confirmed.

---

## Phase P2 — Serial Search & Baselines
- **Status:** COMPLETED & PASSED.
- **Components Implemented:**
  - `src/gen/instance_gen.cpp`: Congestion-controlled instance generator ($\rho \in [0.8, 1.5]$).
  - `src/gen/noise_cpu.cpp`: Fast CPU noise generator with Common Random Numbers (CRN).
  - `src/baselines/fcfs.cpp`: First-Come First-Served baseline sorted by target, ties broken by id.
  - `src/baselines/random_search.cpp`: Pure random permutation baseline.
  - `src/baselines/brute_force.cpp`: Exact combinatorial solver for $n \le 10$.
  - `src/serial/sa_serial.cpp`: Parallel Simulated Annealing solver (serial execution).
  - `bench/calibrate.py`: Automated calibration gate verifier.
- **Calibration Gate (Section 8.2):**
  - Confirmed gap $\ge 10\%$ across all benchmark instances (measured gaps between 39.0% and 60.55%).
  - Determinism test verified: identical seeds produce bit-identical schedules.
  - Brute-force validation: SA reaches within 0.5% of brute-force optimum in $\ge 90\%$ of seeds on $n=8$.

---

## Phase P3 — OpenMP Parallel Architecture
- **Status:** COMPLETED & PASSED.
- **Variants Implemented in `src/omp/sa_omp.cpp`:**
  - `O1 (chains)`: Coarse-grained `#pragma omp parallel for schedule(static)` across independent SA chains.
  - `O2 (scenarios)`: Fine-grained `#pragma omp parallel for reduction(+:sum,sumsq)` across scenarios.
  - `O3 (hybrid)`: Concurrent chains with `#pragma omp simd reduction(+:sum,sumsq)` across scenarios.
- **Equivalence Gate:**
  - OpenMP evaluation matches serial evaluation with relative error $< 10^{-6}$.
- **Performance Findings:**
  - O1 (`chains`) achieves 9.38× speedup on 12 threads; O3 (`hybrid`) reaches 9.63× speedup.
  - O2 (`scenarios`) saturates at 3.12× due to thread synchronization overhead inside the inner evaluation loop.

---

## Phase P4 & P5 — CUDA Architecture
- **Status:** COMPLETED.
- **Components Created in `src/cuda/`:**
  - `cuda_common.cuh`: CUDA error checking and warp-shuffle reduction primitives.
  - `noise_gpu.cu`: cuRAND-based noise generation kernel and host-to-device upload utilities.
  - `eval_kernels.cu`: Naive K1 candidate evaluation and single-permutation reduction kernels.
  - `sa_kernel.cu`: Main K2/K3 kernel mapping 1 block per chain and 1 thread per scenario, using constant memory separation matrices, shared memory SoA caching, and warp-shuffle reductions.
  - `sa_gpu.cu`: Host wrapper managing device allocations, stream synchronization, and epoch resampling.
  - `scripts/colab_run.ipynb`: Complete Google Colab notebook for running CUDA on GPU instances.

---

## Phase P6 — Automated Benchmark Harness & Visualization
- **Status:** COMPLETED & PASSED.
- **Components Implemented:**
  - `bench/run_bench.py`: Automated runner for experiments E2, E3, E4, E8, and E9.
  - `scripts/make_plots.py`: Generates publication-ready figures in PNG and PDF formats to `results/figures/`.
- **Generated CSVs & Visualizations:**
  - `results/calibration.csv`
  - `results/e2_baselines.csv` & `results/figures/e2_baselines.png`
  - `results/e3_openmp_scaling.csv` & `results/figures/e3_openmp_scaling.png`
  - `results/e4_throughput.csv` & `results/figures/e4_throughput.png`
  - `results/e8_uncertainty.csv` & `results/figures/e8_uncertainty.png`
  - `results/e9_sensitivity.csv`

---

## Phase P6b — Native Desktop GUI (PySide6)
- **Status:** COMPLETED & PASSED.
- **Specification Source:** `SPEC_CHANGES_GUI.md` (overrides Streamlit web app in `SPEC.md`).
- **Architecture Overview:**
  - Strict **thin front-end** principle followed: zero scheduling, simulation, or search algorithms implemented in Python.
  - Interacts exclusively with the C++ `runway` binary via CLI commands and filesystem I/O (`instance.csv`, `separation.csv`, `params.cfg`, `result.json`).
  - Implemented using **PySide6 (Qt for Python)** and embedded **Matplotlib** widgets (`FigureCanvasQTAgg`).
- **CLI Enhancements Built in C++:**
  - `runway info`: Outputs system hardware capability JSON (`version`, `cuda_available`, `gpu_name`, `omp_max_threads`).
  - `runway validate`: Validates instance datasets, time feasibility (`ready <= target <= latest`), and separation matrices.
  - `runway replay`: Replays deterministic or single realization timelines with closure intervals and weather effects.
  - `runway solve` extensions: `--progress` (outputs `PROGRESS <epoch>/<E> <best_J> <elapsed_s>` to `stderr`), `--dump-costs N`, `--separation FILE`, and `--s-final N`.
- **GUI Components in `gui/`:**
  - `gui/__main__.py` & `gui/main.py`: Entry points launching the standalone Qt application.
  - `gui/models.py`: Data-only structures (`AircraftRow`, `ScenarioSettings`, `SearchSettings`, `ExecutionSettings`, `ProjectState`).
  - `gui/io_files.py`: Read/write handlers for instance CSVs, separation matrices, INI configurations, and project snapshots.
  - `gui/runner.py`: Asynchronous `QProcess` runner with real-time `PROGRESS` line stream parsing and clean cancellation.
  - `gui/plots.py`: Pure Matplotlib plotting utilities for timelines, cost distributions, convergence curves, and scenario replays.
  - `gui/panels/`: Editable panels for Aircraft (`QTableWidget` + `QUndoStack`), Scenario parameters, Separation 6x6 matrix, Search parameters, and Execution controls (CPU/OpenMP/CUDA detection).
  - `gui/results/`: Results tabs for Summary metrics, Timeline Gantt charts, Monte Carlo distributions, Scenario Replay, Live Convergence, and Multi-implementation comparison dialog.
  - Pop-out capabilities: Each chart tab includes a "Pop out ⧉" button opening in an independent resizable window.
  - Launcher batch script: `RUN_GUI.bat` for one-click desktop execution.
- **Verification:**
  - All 175 Catch2 tests pass.
  - Python headless unit tests pass (`gui/tests/test_gui.py`).
  - Qt offscreen test confirms window initialization, splitter layout, and event processing without errors.

- **Status:** COMPLETED.
- **Report Document:** `report/REPORT.md`
- **Integrity Compliance:**
  - Every numerical metric cited in the report is directly traced from generated CSV data.
  - Literature references are marked with `[CITATION NEEDED]` for human author review.
  - Academic disclaimer prominently displayed.
