# RunwayMC — GPU-Accelerated Robust Runway Scheduling under Uncertainty

> **Mandatory Disclaimer:**
> *This is an academic simulation with simplified separation standards. It is not validated for, and must not be used in, real air-traffic operations.*

---

## Overview

RunwayMC addresses the single-runway aircraft scheduling problem under operational uncertainty (flight arrival delays, weather-induced separation multipliers, and random runway closure windows). It implements parallel Simulated Annealing (SA) combined with Monte Carlo scenario evaluation to find sequence permutations that minimize expected weighted delay and deadline penalties.

The project provides three implementations sharing a single unified simulation core (`sim_core.hpp`):
1. **Serial CPU:** High-efficiency baseline compiled with `-O3 -march=native`.
2. **OpenMP Multithreading:**
   - `O1 (chains)`: Coarse-grained parallelization across SA chains.
   - `O2 (scenarios)`: Fine-grained parallelization across Monte Carlo scenarios with parallel reduction.
   - `O3 (hybrid)`: Parallel SA chains combined with SIMD-vectorized scenario evaluation.
3. **CUDA GPU Acceleration:**
   - Block-per-chain, thread-per-scenario execution (`K2` and `K3`).
   - Constant-memory separation matrix, shared-memory aircraft SoA buffers, warp-shuffle parallel reductions, and pre-generated coalesced noise arrays via cuRAND.

---

## Repository Structure

```
runway-mc/
├── CMakeLists.txt             # Options: -DUSE_CUDA=ON/OFF, -DUSE_OPENMP=ON/OFF
├── README.md                  # Overview, build/run guides, disclaimer
├── docs/
│   ├── SPEC.md                # System specification & mathematical model
│   ├── DECISIONS.md           # Architectural logs and phase decisions
│   └── HARDWARE.md            # Target environment & hardware specifications
├── config/
│   └── separation_default.csv # Wake turbulence separation matrix (6x6)
├── include/runway/
│   ├── types.hpp              # Aircraft, scenario, separation, and config structures
│   ├── sim_core.hpp           # __host__ __device__ simulation core (Single Source of Truth)
│   ├── instance_io.hpp        # CSV instance loading and validation
│   └── rng.hpp                # Fast host & device PRNG (SplitMix64 / XorShift)
├── src/
│   ├── core/                  # instance_io, validator, objective
│   ├── gen/                   # instance generator, CPU noise generator
│   ├── baselines/             # FCFS, random search, brute force
│   ├── serial/                # Serial SA solver
│   ├── omp/                   # OpenMP SA solver (O1, O2, O3)
│   ├── cuda/                  # CUDA kernels (K1, K2, K3) and GPU solver
│   └── app/                   # Main CLI driver (runway)
├── tests/                     # Unit & integration test suite (Catch2)
├── bench/                     # Benchmark harness and environment probe
├── scripts/                   # Plot generation and Google Colab notebook
├── data/instances/            # Benchmark problem instances
└── results/                   # Benchmark CSVs and generated figures
```

---

## Building RunwayMC

### Prerequisites
- CMake ≥ 3.18
- C++17 compliant compiler (GCC/G++ ≥ 9, Clang ≥ 10, or MSVC 2019+)
- OpenMP support (for multi-threaded CPU execution)
- NVIDIA CUDA Toolkit ≥ 11.0 (optional, required for GPU acceleration)
- Python 3.8+ with `numpy`, `pandas`, `scipy`, `matplotlib`, `pyyaml` (for benchmarking and plotting)

### Build Instructions

#### CPU + OpenMP Build:
```bash
cmake -B build -DUSE_CUDA=OFF -DUSE_OPENMP=ON
cmake --build build --config Release
```

#### CUDA + OpenMP Build (on CUDA-equipped host or Google Colab):
```bash
cmake -B build -DUSE_CUDA=ON -DUSE_OPENMP=ON
cmake --build build --config Release
```

### Running Tests
```bash
ctest --test-dir build --output-on-failure
```

---

## Usage (CLI Reference)

RunwayMC provides the unified `runway` command-line executable:

### 1. Generate Problem Instances
```bash
./build/runway gen --n 50 --rho 1.2 --seed 1 --out data/instances/n50_r12_s1.csv
```

### 2. Evaluate a Schedule
```bash
./build/runway eval --instance data/instances/n50_r12_s1.csv --perm fcfs --scenarios 1024 --seed 1
```

### 3. Solve via Simulated Annealing
```bash
# Serial solver:
./build/runway solve --instance data/instances/n50_r12_s1.csv --impl serial --chains 64 --scenarios 1024 --iters 2000 --epochs 20 --seed 1

# OpenMP solver (O1 chains mode):
./build/runway solve --instance data/instances/n50_r12_s1.csv --impl omp --omp-mode chains --threads 8 --chains 64 --scenarios 1024 --iters 2000 --epochs 20 --seed 1

# CUDA GPU solver:
./build/runway solve --instance data/instances/n50_r12_s1.csv --impl cuda --kernel k3 --block 256 --chains 256 --scenarios 1024 --iters 2000 --epochs 20 --seed 1
```

### 4. Out-of-Sample Final Evaluation
```bash
./build/runway final --instance data/instances/n50_r12_s1.csv --perm results/best_perm.csv --scenarios 20000 --seed 999
```

---

## Google Colab Execution

For environments without a local CUDA installation, use `scripts/colab_run.ipynb` to build and run the full CUDA test suite and benchmarks on a free GPU runtime.
