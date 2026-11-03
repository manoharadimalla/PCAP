# RunwayMC — Viva & Presentation Explanation Kit

This document is your **exam and presentation companion**. It gives plain-language answers to every question an examiner might ask.

---

## 1. The 60-Second Pitch

> "RunwayMC is a parallel simulation and optimization engine for airport runway scheduling. Air traffic controllers must sequence landings and takeoffs to minimize delay while respecting safety separation distances between heavy and light planes. But real airports are unpredictable—weather causes random delays, and runways occasionally close for inspections. 
> 
> Evaluating a candidate schedule across thousands of Monte Carlo scenarios is computationally demanding. We parallelized this simulation using **OpenMP for multi-core CPUs** and designed a **CUDA GPU architecture** that simulates thousands of independent random days simultaneously. On our 12-thread CPU, OpenMP delivers over **7× speedup**, and schedules planned under uncertainty cut delay costs by **over 44%** on unseen test days compared to deterministic planning."

---

## 2. Architecture & Pipeline Diagrams

### Overall Optimization Pipeline
```
┌─────────────────┐       ┌────────────────────────┐       ┌──────────────────────┐
│  Aircraft List  │ ────► │  Simulated Annealing   │ ────► │  Monte Carlo Days    │
│  (Target times, │       │  (Propose Swap/Insert) │       │  (1,024 scenarios)   │
│   wake classes) │       └────────────────────────┘       └──────────┬───────────┘
└─────────────────┘                    ▲                              │
                                       │    Accept / Reject Move      ▼
                                       └─────────────────────── [Parallel Sum]
                                                                Average Delay Cost
```

### CUDA / GPU Mapping
```
┌────────────────────────────────────────────────────────┐
│ GPU Grid                                               │
│ ┌──────────────────────┐      ┌──────────────────────┐ │
│ │ CUDA Block 0         │      │ CUDA Block 1         │ │
│ │ (Search Chain 0)     │      │ (Search Chain 1)     │ │
│ │                      │      │                      │ │
│ │ Thread 0: Scenario 0 │      │ Thread 0: Scenario 0 │ │
│ │ Thread 1: Scenario 1 │ ...  │ Thread 1: Scenario 1 │ │
│ │ Thread S: Scenario S │      │ Thread S: Scenario S │ │
│ │ ───────────────────  │      │ ───────────────────  │ │
│ │ Warp Reduction (Sum) │      │ Warp Reduction (Sum) │ │
│ └──────────────────────┘      └──────────────────────┘ │
└────────────────────────────────────────────────────────┘
```

---

## 3. Glossary of Terms

1. **Scenario:** One simulated realization of random aircraft delays and runway closures.
2. **Chain:** An independent Markov search process in Simulated Annealing exploring different schedule sequences.
3. **Simulated Annealing (SA):** A probabilistic heuristic that accepts better solutions, and occasionally worse ones at higher temperatures, to escape local minima.
4. **Monte Carlo:** A statistical method that models uncertainty by evaluating thousands of repeated random samplings.
5. **FCFS (First-Come, First-Served):** The naive baseline order sorting planes strictly by scheduled target time, ignoring wake turbulence classes.
6. **Wake Separation:** The mandatory safety gap required between aircraft. Heavy planes leave powerful vortexes, requiring light planes behind them to wait up to 196 seconds.
7. **Closure Window:** A time interval during which the runway is unavailable due to emergencies or inspections.
8. **P95 Cost:** The 95th percentile cost—the worst delay cost experienced on the worst 5% of simulated days.
9. **Train / Test Days:** The search only ever observes the training scenarios; the final schedule is evaluated on 20,000 completely unseen test days to prevent overfitting.
10. **Speedup:** Execution time on 1 core divided by execution time on $P$ parallel threads ($T_1 / T_P$).
11. **Efficiency:** Speedup divided by thread count ($S_P / P$).
12. **Reduction:** Combining thousands of individual scenario cost numbers into a single sum using parallel tree or warp operations.
13. **Block:** A group of CUDA threads that execute on a single Streaming Multiprocessor and share high-speed shared memory.
14. **Thread:** The individual unit of execution calculating one scenario on CPU or GPU.
15. **Constant Memory:** Dedicated read-only GPU memory with hardware broadcast cache, ideal for lookup tables like the $6\times 6$ separation matrix.

---

## 4. What Each File Does

| File | Purpose |
| :--- | :--- |
| `include/runway/sim_core.hpp` | **Single source of truth:** Exact decoder simulating runway physics, separation, and closure. |
| `include/runway/types.hpp` | Data structures (Aircraft, Structure-of-Arrays, Separation Matrix, Noise, Results). |
| `src/core/objective.cpp` | Computes mean cost, P95 tail cost, and late percentage across scenarios. |
| `src/core/validator.cpp` | Checks schedule validity (all-pairs wake separation and closure non-interference). |
| `src/gen/noise_cpu.cpp` | Generates exponential delays and random closure windows using Common Random Numbers. |
| `src/serial/sa_serial.cpp` | Serial reference Simulated Annealing solver (1 CPU thread). |
| `src/omp/sa_omp.cpp` | **OpenMP parallel solver:** Coarse-grained `chains` mode and fine-grained `scenarios` mode. |
| `src/cuda/sa_kernel.cu` | **CUDA GPU kernel:** 1 block per chain, 1 thread per scenario, warp-shuffle reductions. |
| `src/app/main.cpp` | Command-line interface (`gen`, `eval`, `solve`, `validate`, `info`). |
| `bench/run_bench.py` | Benchmark harness running experiments E1 through E5 and logging CSVs. |
| `gui/main_window.py` | Native PySide6 pop-up GUI front-end with interactive table, Gantt charts, and distribution plots. |
| `docs/PROJECT.md` | Core technical specification and project documentation. |

---

## 5. Twelve Likely Viva Questions & Answers

#### Q1: What problem are you solving and why is it hard?
**Answer:** We are solving robust aircraft runway scheduling for a single runway under uncertainty. It is an NP-hard permutation problem because wake turbulence safety standards require variable separation depending on aircraft classes (up to 196 seconds), meaning order drastically affects total delay.

#### Q2: Why Simulated Annealing and not an exact brute-force search?
**Answer:** The search space for $n$ aircraft is $n!$ permutations. For $n=30$, $30! \approx 2.65 \times 10^{32}$, which would take billions of years to evaluate. Simulated Annealing finds near-optimal schedules in a few seconds.

#### Q3: What is a scenario and why do you need thousands of them?
**Answer:** A scenario is one random day with specific aircraft arrival/departure delays and runway closures. Because delays have high variance and closure windows are probabilistic, testing only 5 or 10 days would produce noisy, brittle schedules; thousands are needed for statistical confidence.

#### Q4: What exactly runs in parallel?
**Answer:** In coarse-grained OpenMP, multiple independent Markov search chains run concurrently on separate CPU cores. In fine-grained OpenMP and CUDA, the evaluation of thousands of independent scenarios for each proposed schedule is executed in parallel across threads.

#### Q5: Why does coarse-grained OpenMP (chains) scale better than fine-grained (scenarios)?
**Answer:** Coarse-grained OpenMP assigns an entire search chain to each thread, requiring zero synchronization during an iteration. Fine-grained scenario parallelization requires thread fork-join overhead and a reduction barrier at every single candidate evaluation step, creating thread synchronization bottlenecks.

#### Q6: Why could a GPU be slower for small scenario counts ($S$)?
**Answer:** GPU kernel launches and PCIe data transfers carry fixed hardware overheads. If $S$ is small (e.g., $S=128$), the GPU cannot saturate its thousands of cores, and kernel overheads dominate. The GPU wins when $S \ge 1024$.

#### Q7: How did you map the problem to CUDA?
**Answer:** We mapped **1 CUDA block per search chain** and **1 CUDA thread per Monte Carlo scenario**. We placed the $6\times 6$ separation matrix in fast constant memory (`__constant__`), cached the aircraft sequence in block-level shared memory, and aggregated scenario costs using register warp shuffles.

#### Q8: How do you know the parallel versions give the same answer as serial?
**Answer:** We test equivalence using identical seeds and Common Random Numbers (CRN). Catch2 unit test `test_omp.cpp` verifies that OpenMP evaluation matches the serial reference evaluation within a relative error tolerance of $10^{-6}$.

#### Q9: Why test on days the search never saw?
**Answer:** If you score a schedule only on the training days used during optimization, you suffer from "in-sample optimization bias" (overfitting). Testing on 20,000 independent unseen test days guarantees that the schedule performs reliably in real-world conditions.

#### Q10: What is your baseline and how much better is your schedule?
**Answer:** Our baseline is standard First-Come, First-Served (FCFS). In Experiment E2, RunwayMC reduces delay costs from over 52,000 seconds down to ~27,000 seconds—a **48% reduction**—while cutting P95 tail risk by more than half.

#### Q11: Does planning with randomness really help (E5)?
**Answer:** Yes. In Experiment E5, a schedule optimized under deterministic assumptions experienced a test-day cost of 27,046 seconds and an 18.5% deadline miss rate. The robust schedule planned under uncertainty reduced the cost to 15,063 seconds (**44.3% improvement**) and dropped the late aircraft share to only **5.8%**.

#### Q12: What are the limitations and what would you do next?
**Answer:** Our model assumes a single runway, simplified FAA separation categories, and independent exponential delays. Future extensions could include multi-runway dependent arrival/departure operations, surface taxiway routing constraints, and real airport flight radar data.

---

## 6. What We Did NOT Do (Scope Boundaries)
- We did **not** implement real FAA/Eurocontrol air traffic routing; separation standards are academic benchmarks.
- We simulate a **single runway**, not multi-runway or intersection operations.
- The desktop GUI is a **thin front-end** for demonstration—it contains zero scheduling or simulation logic.

---

## 7. End-to-End Workflow: GUI Export to CUDA (Cloud/Colab)

A key highlight of RunwayMC is its **cross-platform unified architecture**: you can build or inspect an aircraft schedule visually on Windows, export it, and run massive GPU-accelerated Monte Carlo optimization on an NVIDIA GPU (e.g. in Google Colab).

```
┌───────────────────────────┐         ┌──────────────────────────┐         ┌───────────────────────────┐
│     Windows Native GUI    │         │      CSV File Format     │         │   Google Colab (CUDA GPU) │
│  (Aircraft Table Editor)  │ ──────► │  `demo_interactive.csv`  │ ──────► │   Unified C++ Engine      │
│  [Click "Export CSV"]     │         │  (id, op, wake, ready..) │         │   `--impl cuda` (Tesla T4)│
└───────────────────────────┘         └──────────────────────────┘         └───────────────────────────┘
```

### Step-by-Step Instructions:
1. **Design / Edit Flights in the GUI:**
   - Launch the GUI on your PC with `RUN_GUI.bat`.
   - In the **Aircraft Instance** panel, adjust flight ready times, deadlines, wake turbulence categories (`H`, `M`, `L`), and operation types (`ARR` vs `DEP`).
2. **Export to CSV:**
   - In the toolbar above the table, click **"Export CSV"**.
   - Save the file as `my_flights.csv`.
3. **Run on CUDA via Google Colab:**
   - Open your Colab notebook (`scripts/colab_run.ipynb`).
   - Drag and drop `my_flights.csv` into the Colab file browser on the left (or place it in `data/instances/`).
   - In Step 5 of the notebook, set:
     ```python
     INSTANCE = 'my_flights.csv'
     ```
   - Execute the cell. The unified C++ engine compiles the exact same math, executes GPU kernels (`solve_sa_cuda`), and displays the speedup comparison!

---

## 8. What Each Benchmark Graph Represents

When you run `python3 bench/run_bench.py --all && python3 scripts/make_plots.py`, five high-resolution publication-quality figures are generated in `results/figures/`:

### 1. Figure E2: Baselines Comparison (`e2_baselines.png`)
* **What it shows:** Bar chart comparing **Mean Cost** and **P95 Tail Cost** across three strategies:
  1. **FCFS (First-Come, First-Served):** Standard arrival order (highest delay, ~52,000 s).
  2. **Random Search:** Pure random permutation sampling with identical evaluation budget (~39,000 s).
  3. **RunwayMC (Simulated Annealing):** Optimized schedule (~27,000 s).
* **The Takeaway:** Proves that metaheuristic search cuts flight delays by **48%** and cuts catastrophic tail risk (P95) by half compared to naive sequencing.

### 2. Figure E3: OpenMP Multi-Core Scaling (`e3_openmp_scaling.png`)
* **What it shows:** Speedup vs. thread count ($T \in [1, 12]$) comparing:
  - **OpenMP Chains (Coarse-grained):** Near-linear scaling (7.30× speedup on 8 cores).
  - **OpenMP Scenarios (Fine-grained):** Saturates at ~2.61× speedup.
  - **Ideal Linear Line ($y = x$):** Theoretical upper bound.
* **The Takeaway (Amdahl's Law in Action):** Demonstrates parallel granularity. Coarse-grained partitioning avoids thread synchronization, whereas fine-grained scenario partitioning incurs fork-join overhead and reduction barriers at every candidate proposal.

### 3. Figure E4: Execution Time Scaling with $S$ (`e4_throughput.png`)
* **What it shows:** Log-log plot of execution time (ms) as the scenario volume grows from $S=128$ to $S=4,096$:
  - **Serial (1 CPU thread):** Steepest slope (slowest).
  - **OpenMP (Multi-Core CPU):** Significantly faster across moderate scenario counts.
  - **CUDA (NVIDIA GPU):** Flattest curve; dominates at high scenario volumes.
* **The Takeaway:** Confirms computational complexity $O(C \cdot I \cdot n \cdot S)$ and shows how parallel architectures scale with problem volume.

### 4. Figure E4b: Parallel Speedup vs. Serial Baseline (`e4_cuda_speedup.png`)
* **What it shows:** Relative speedup multiplier ($T_{\text{serial}} / T_{\text{parallel}}$) across increasing scenario counts $S$:
  - OpenMP delivers a steady **5× to 7× speedup**.
  - CUDA speedup increases dramatically with $S$ (from ~4× at $S=128$ to **over 20×+ at $S \ge 2,048$**).
* **The Takeaway (The Hardware Crossover Point):** Proves that GPUs require massive data parallelism to overcome launch and data transfer overheads.

### 5. Figure E5: Value of Robust Optimization (`e5_uncertainty.png`)
* **What it shows:** Bar chart comparing two schedules scored on **20,000 unseen test days**:
  - **Deterministic Planning ($S=1$, planned assuming zero delays):** Suffers severely on real days (27,046 s cost, 18.5% late flights).
  - **Robust RunwayMC ($S=1024$, planned under uncertainty):** Survives operational disruptions (15,063 s cost, only 5.8% late flights).
* **The Takeaway:** Proves that planning under uncertainty reduces realized delay costs by **44.3%** and cuts deadline violations by more than two-thirds.

---

## 9. Comparing OpenMP (CPU) vs. CUDA (GPU) — Architectural Deep Dive

| Dimension | OpenMP (Multi-Core CPU) | CUDA (Massively Parallel GPU) |
| :--- | :--- | :--- |
| **Hardware Paradigm** | **MIMD** (Multiple Instruction, Multiple Data) | **SIMT** (Single Instruction, Multiple Threads) |
| **Compute Units** | Few heavy cores (e.g., 8–16 cores with hyperthreading) | Thousands of lightweight cores (e.g., 2,560 cores on Tesla T4) |
| **Clock & Cache** | High clock speed (4.0+ GHz), large multi-megabyte L3 caches | Lower clock speed (~1.5 GHz), high-bandwidth shared memory |
| **Mapping Strategy** | Coarse-grained: 1 thread per search chain | Hierarchical: 1 block per chain, 1 thread per scenario |
| **Parallel Reduction** | OpenMP tree reduction: `#pragma omp parallel for reduction(+:sum)` | Register warp shuffles (`__shfl_down_sync`) across 32-lane warps |
| **Memory Strategy** | Standard host RAM with hardware prefetching | Constant Memory (`__constant__`) for 6x6 separation matrix broadcast |
| **Best Workload** | Low scenario counts ($S \le 256$) or interactive GUI response | Massive Monte Carlo counts ($S \ge 1024$) and large flight batches |
| **Launch Overhead** | Nanoseconds (thread pool already active) | Microseconds (kernel launch latency + host-device transfers) |

### Why OpenMP Wins on Small $S$ and CUDA Dominates on Large $S$:
1. **Low $S$ ($S \le 256$):** A GPU needs tens of thousands of active threads to hide memory latency. With small $S$, the GPU Streaming Multiprocessors (SMs) are under-utilized, so the high single-thread clock speed and zero-latency cache of the CPU gives OpenMP the edge.
2. **High $S$ ($S \ge 1024$):** Monte Carlo simulation is an **embarrassingly parallel** compute problem. The thousands of GPU cores execute independent scenarios simultaneously, while hardware warp shuffles aggregate scenario costs in registers without touching global RAM, allowing CUDA to pull ahead with **massive speedups**.
