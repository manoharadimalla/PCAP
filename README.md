# RunwayMC: Robust Runway Scheduling under Uncertainty

> **Academic Simulation Disclaimer:** This is an academic research simulation using simplified wake vortex separation standards. It is not validated for, and must not be used in, real air-traffic operations.

RunwayMC is a parallel Monte Carlo simulation and optimization engine for robust single-runway aircraft sequencing under uncertainty (exponential flight delays and random runway closures).

---

## Highlights & Features
- **Parallel Computing Core:** Multi-core OpenMP parallelization on CPU and massively parallel CUDA architecture for NVIDIA GPUs.
- **Robust vs. Deterministic:** Achieves over **44% delay cost reduction** on unseen test days compared to deterministic planning.
- **Interactive Native GUI:** Native PySide6 desktop pop-up application with Gantt charts, Monte Carlo distributions, and real-time solving.
- **Tested & Verified:** Validated with Catch2 unit tests, determinism checks, and brute-force equivalence on small problem sizes.

---

## Quick Start

### 1. Run the Desktop GUI
Simply double-click **`RUN_GUI.bat`** or run:
```powershell
python -m gui
```

### 2. Run the Interactive Terminal Demo
Double-click **`RUN_DEMO.bat`** or run:
```powershell
python run_demo.py
```

### 3. Build from Source
Requirements: C++17 compiler (GCC / Clang / MSVC), CMake $\ge$ 3.20, Ninja.
```powershell
cmake -B build -G "Ninja" -DCMAKE_BUILD_TYPE=Release -DUSE_CUDA=OFF -DUSE_OPENMP=ON
cmake --build build -j8
```

### 4. Run Automated Tests
```powershell
.\build\runway_tests.exe
```

---

## Project Structure & Documentation
- **`docs/PROJECT.md`:** Concise technical summary of the model, algorithm, and experiments.
- **`docs/EXPLAIN.md`:** Comprehensive viva and presentation guide with 12 Q&As.
- **`report/REPORT.md`:** Full academic research report with experimental figures.
- **`results/`:** Generated benchmark CSV files and figures for experiments E1–E5.
