# RunwayMC — Project Specification & Architecture

> **Academic Disclaimer:** This is an academic research simulation using simplified separation standards. It is not validated for, and must not be used in, real air-traffic operations.

---

## 1. The Project in Plain Words

1. **The Problem:** We have a list of aircraft that must use **one runway**. The **order** decides how long aircraft wait. Heavy planes leave wake vortex turbulence, meaning following aircraft must wait longer.
2. **The Evaluation:** To judge an order, we **simulate many random days** (random airborne/ground delays, and sometimes a random runway closure) and compute the **average weighted waiting time**.
3. **The Search:** A search program (**Simulated Annealing**) tries swapping or moving aircraft positions, keeping better orders while occasionally accepting worse ones early on to avoid getting trapped in local optima.
4. **The Parallel Acceleration (Core PCAP focus):** Simulating thousands of random days is the computational bottleneck. Because each random day is completely independent, we **run them in parallel**: **Serial Reference → OpenMP (Multi-Core CPU) → CUDA (NVIDIA GPU)**.
5. **The Verification:** We evaluate final schedules on **20,000 unseen test days** that the search never observed, proving that robust optimization beats naive and deterministic scheduling without overfitting.

```
Aircraft List ──► Propose New Order ──► Simulate S Random Days ──► Average Delay Cost
      ▲                                       │   (OpenMP / CUDA Parallelism)
      └─────────── Keep if Better ◄───────────┘
```

---

## 2. Mathematical Model & Constants

- **Aircraft Parameters:** $n$ aircraft, operation (ARR/DEP), wake class (Heavy, Medium, Light), ready time $r_i$, target time $T_i$, latest deadline $L_i$, priority weight $w_i$.
- **Fixed Constants:**
  - Priority Weights: Normal = 1, High = 3, Emergency = 20.
  - Deadline Penalty: $M = 5\times$ delay cost past deadline $L_i$.
  - Random Delays: Exponential distribution; Arrival mean = 30 s, Departure mean = 60 s.
  - Runway Closure: Probability $p = 0.20$; duration $U(300, 900)$ seconds.
  - Wake Vortex Separation: $6 \times 6$ standard separation matrix (`config/separation_default.csv`).
- **Single Source of Truth Decoder (`include/runway/sim_core.hpp`):**
  Exact decoder tracking `lastT[cls]` (the last runway time used by each of the 6 aircraft classes).

---

## 3. Algorithm & Train/Test Split

1. **Simulated Annealing (SA):**
   - Auto-calibrates initial temperature $T_0$ such that ~50% of worse moves are accepted initially.
   - Neighborhood moves: Swap two aircraft (50%) or Move one aircraft to another position (50%).
   - Geometric cooling: $T \leftarrow T \times \alpha$.
2. **Train vs. Test Rigor:**
   - **Training Days:** $S$ scenarios (default 1,024), shared by all search chains.
   - **Test Days:** 20,000 independent fresh scenarios generated once with seed 99999 to score final schedules.

---

## 4. Parallel Computing Architecture

| Implementation | Level of Parallelism | Role |
| :--- | :--- | :--- |
| **Serial Reference** | 1 CPU core | Ground truth baseline for speedup calculations |
| **OpenMP Chains (Coarse)** | Multi-core CPU threads | Independent Markov search chains run concurrently |
| **OpenMP Scenarios (Fine)** | Multi-core CPU threads | Scenarios of each schedule evaluated with `#pragma omp parallel for reduction` |
| **CUDA Main Engine** | NVIDIA GPU Streaming Multiprocessors | 1 block per chain, 1 thread per scenario, constant memory separation matrix, warp-shuffle sum |

---

## 5. Five Benchmark Experiments (E1–E5)

1. **E1 (Correctness):** Catch2 test suite passes all assertions; serial and OpenMP produce identical costs within $10^{-6}$.
2. **E2 (Baselines):** Evaluates FCFS vs. Random Search vs. Simulated Annealing on identical 20,000 test days.
3. **E3 (OpenMP Scaling):** Measures speedup and efficiency for 1 to 12 threads against the serial reference baseline.
4. **E4 (Throughput vs. S):** Evaluates runtime scaling as scenario count $S$ grows from 128 to 4,096.
5. **E5 (Value of Randomness):** Demonstrates that planning with uncertainty cuts test-day delay costs by >40% and lowers deadline misses compared to deterministic schedules.

---

## 6. How to Build and Run

### Building
```powershell
cmake -B build -G "Ninja" -DCMAKE_BUILD_TYPE=Release -DUSE_CUDA=OFF -DUSE_OPENMP=ON
cmake --build build -j8
```

### Running Tests
```powershell
.\build\runway_tests.exe
```

### Running the Desktop GUI
Double-click `RUN_GUI.bat` or run `python -m gui`.

### Running the Interactive Demo
Double-click `RUN_DEMO.bat` or run `python run_demo.py`.
