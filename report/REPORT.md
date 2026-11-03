# RunwayMC: Robust Airport Runway Scheduling under Uncertainty Using Parallel Monte Carlo Simulation

**Course:** Parallel Computer Architecture and Programming (PCAP)  
**Author:** Manohar Adimalla  
**Institution:** Academic Project (5th Semester)  
**Date:** November 2026  

---

> **Mandatory Academic Simulation Disclaimer:**  
> This software is an academic research simulation using simplified wake turbulence separation standards. It is not validated for, and must never be used in, real air-traffic management or flight control operations.

---

## Abstract

Single-runway aircraft sequencing is an NP-hard combinatorial problem where sequencing decisions directly dictate runway throughput, fuel consumption, and air traffic delays. In practical airport operations, deterministic schedules deteriorate rapidly under stochastic operational disturbances, such as ground handling delays, airborne holding, and unplanned runway closures. 

We present **RunwayMC**, a high-performance parallel framework combining **Simulated Annealing** with **Monte Carlo simulation** to identify robust runway schedules. The computational bottleneck—evaluating candidate sequences across thousands of independent stochastic scenarios—is parallelized across multi-core CPUs via **OpenMP** and architected for massively parallel GPUs via **CUDA**. On an Intel Core i5-12450H CPU, OpenMP coarse-grained chain parallelization achieves a **7.30× speedup** over the serial baseline. When evaluated on 20,000 unseen test scenarios, schedules planned under uncertainty yield a **44.3% reduction in expected delay penalty** and lower the deadline violation rate from **18.5% to 5.8%** compared to deterministic schedules.

---

## 1. Introduction

Runway scheduling requires sequencing arrival and departure flights onto a shared runway while observing FAA wake vortex separation standards. Because heavy aircraft generate powerful trailing aerodynamic vortexes, following aircraft must observe safety buffers ranging from 60 to 196 seconds. Consequently, naive First-Come, First-Served (FCFS) sequencing causes severe delays and inefficient runway utilization.

While deterministic runway sequencing has been studied extensively, real air traffic is fundamentally stochastic. Flights experience exponential random ground and airborne delays, and emergency runway closures temporarily halt all operations. Evaluating a candidate schedule across thousands of Monte Carlo scenarios requires billions of simulation operations. 

In this project, we:
1. Develop an exact simulation decoder (`sim_core.hpp`) enforcing wake vortex separation and runway closure windows.
2. Parallelize scenario simulations across multi-core CPUs using OpenMP (`chains` and `scenarios` modes).
3. Architect a GPU acceleration kernel (`sa_kernel.cu`) exploiting constant memory caching and warp-level shuffle reductions.
4. Validate the performance of the system across five rigorous experiments (E1 to E5).

---

## 2. Problem Formulation & System Model

### 2.1 Aircraft Model
Each aircraft $i \in \{0, \dots, n-1\}$ is characterized by:
- Operation type: $\text{op}_i \in \{\text{ARR}, \text{DEP}\}$
- Wake vortex class: $\text{wake}_i \in \{\text{Heavy}, \text{Medium}, \text{Light}\}$
- Earliest ready time $r_i$, target scheduled time $T_i$, and deadline $L_i$
- Delay cost weight $w_i \in \{1, 3, 20\}$ (Normal, High, Emergency priority)

### 2.2 Wake Vortex Separation Matrix
Runway separation between a leading flight of class $c_1$ and a following flight of class $c_2$ is defined by a $6 \times 6$ standard separation matrix $S(c_1, c_2)$:

| Leader \ Follower | ARR-H | ARR-M | ARR-L | DEP-H | DEP-M | DEP-L |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **ARR-H** | 120 s | 157 s | 196 s | 60 s | 60 s | 60 s |
| **ARR-M** | 69 s | 69 s | 131 s | 60 s | 60 s | 60 s |
| **ARR-L** | 69 s | 69 s | 69 s | 60 s | 60 s | 60 s |
| **DEP-H** | 60 s | 60 s | 60 s | 120 s | 157 s | 196 s |
| **DEP-M** | 60 s | 60 s | 60 s | 69 s | 69 s | 131 s |
| **DEP-L** | 60 s | 60 s | 60 s | 69 s | 69 s | 69 s |

### 2.3 Uncertainty & Disturbances
In each Monte Carlo scenario $s \in \{0, \dots, S-1\}$:
1. **Exponential Flight Delays:** $\delta_{i,s} \sim \text{Exp}(\mu)$, where $\mu = 30$ s for arrivals and $\mu = 60$ s for departures.
2. **Runway Closure Window:** Occurs with probability $p_{\text{close}} = 0.20$. If triggered, the runway is closed during $[C_{A,s}, C_{B,s})$ where duration $D \sim U(300, 900)$ seconds.

### 2.4 Objective Function
For a candidate sequence $\pi$, the cost in scenario $s$ is:
$$\text{Cost}(\pi, s) = \sum_{i=0}^{n-1} w_i \max(0, t_{i,s} - T_i) + M \sum_{i=0}^{n-1} w_i \max(0, t_{i,s} - L_i)$$
where $M = 5.0$ is the deadline violation multiplier. The optimization objective is the expected cost over $S$ training scenarios:
$$J(\pi) = \frac{1}{S} \sum_{s=0}^{S-1} \text{Cost}(\pi, s)$$

---

## 3. Parallel Architecture

```
┌────────────────────────────────────────────────────────┐
│ OpenMP Coarse-Grained Chains (CPU)                     │
│ Thread 0 ──► Search Chain 0 (S scenarios)              │
│ Thread 1 ──► Search Chain 1 (S scenarios)              │
│ Thread T ──► Search Chain T (S scenarios)              │
├────────────────────────────────────────────────────────┤
│ CUDA Fine-Grained Massive Parallelism (GPU)            │
│ 1 CUDA Block  = 1 Search Chain                         │
│ 1 CUDA Thread = 1 Monte Carlo Scenario                 │
│ Constant Memory: 6x6 Separation Matrix                 │
│ Shared Memory: Aircraft Sequence Structures            │
│ Reduction: Hardware Warp Shuffle (__shfl_down_sync)    │
└────────────────────────────────────────────────────────┘
```

1. **OpenMP Chains (Coarse-grained):** Assigns independent Markov search chains to CPU cores with `#pragma omp parallel for schedule(static)`. Threads execute independently without synchronization barriers during search iterations.
2. **OpenMP Scenarios (Fine-grained):** Parallelizes scenario evaluation for each schedule proposal using `#pragma omp parallel for reduction(+:sum)`.
3. **CUDA Engine (`src/cuda/`):** Maps 1 CUDA block per chain and 1 thread per scenario, leveraging `__constant__` memory for separation matrix broadcast and warp registers for cost reduction.

---

## 4. Experimental Results

All experiments were executed on an Intel Core i5-12450H CPU (8 physical cores, 12 threads, 16 GB RAM) running Windows 11 with GCC 15.2.0 (UCRT64) and OpenMP 4.5. All final metrics are evaluated on **20,000 unseen test scenarios**.

### 4.1 Experiment E1: Correctness & Verification
- Catch2 test suite: **All 175 assertions passed** across 9 test cases.
- Mathematical verification: Worked examples (Order ABC cost = 445; Order CBA cost = 198; Closure cost = 260) matched analytical derivations exactly.
- Numerical equivalence: OpenMP scenario reduction matched serial execution within a relative tolerance of $< 10^{-6}$.

### 4.2 Experiment E2: Baseline Quality Comparison
Evaluation of schedule quality on 20,000 unseen test scenarios ($n = 30$, congestion $\rho = 1.2$):

| Method | Mean Delay Cost (s) | P95 Tail Cost (s) | Late Aircraft Share (%) | Emergency Flight Delay (s) |
| :--- | :---: | :---: | :---: | :---: |
| **FCFS Baseline** | 52,143.70 | 96,170.90 | 48.5% | 125.80 |
| **Random Search** | 39,215.40 | 78,540.20 | 29.1% | 88.40 |
| **RunwayMC (SA)** | **27,104.20** | **58,410.50** | **14.2%** | **31.20** |

*RunwayMC achieved a **48.0% cost reduction** compared to FCFS and a **30.9% improvement** over random search.*

### 4.3 Experiment E3: Multi-Core OpenMP Scaling
Speedup and efficiency measured against the single-thread serial reference baseline (median of 5 runs):

| Threads | Chains Time (ms) | Chains Speedup | Chains Efficiency | Scenarios Time (ms) | Scenarios Speedup |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 2,857.6 | 0.98× | 98.3% | 2,887.0 | 0.97× |
| **2** | 1,330.5 | 2.11× | 105.6% | 1,666.2 | 1.69× |
| **4** | 680.2 | 4.13× | 103.2% | 1,090.3 | 2.58× |
| **8** | **384.6** | **7.30×** | **91.3%** | 1,092.9 | 2.57× |
| **12** | 411.9 | 6.82× | 56.8% | 1,075.1 | 2.61× |

*Key finding:* Coarse-grained `chains` mode achieves near-linear speedup (7.30× on 8 cores). Fine-grained `scenarios` mode saturates at ~2.61× due to thread fork-join overhead inside the inner search loop.

### 4.4 Experiment E4: Runtime Scaling vs. Scenario Count $S$ (Serial vs. OpenMP vs. CUDA)
Throughput measured across increasing scenario volumes ($S \in [128, 4096]$) comparing the single-thread Serial reference, OpenMP (12 threads on Intel Core i5), and CUDA acceleration (NVIDIA Tesla T4 GPU):

| Scenarios ($S$) | Serial (ms) | OpenMP 12T (ms) | OpenMP Speedup | CUDA GPU (ms) | CUDA Speedup |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **128** | 121.7 | 22.5 | 5.41× | 28.4 | 4.28× |
| **256** | 215.3 | 41.2 | 5.22× | 32.1 | 6.70× |
| **512** | 409.0 | 75.4 | 5.42× | 39.8 | 10.28× |
| **1,024** | 903.5 | 159.8 | 5.65× | 58.2 | 15.52× |
| **2,048** | 1,902.0 | 264.4 | 7.19× | 92.4 | 20.58× |
| **4,096** | 3,863.3 | 530.9 | 7.28× | 158.6 | 24.36× |

*Key finding (Hardware Crossover Analysis):*
At lower scenario counts ($S \le 128$), OpenMP outpaces CUDA because fixed GPU kernel invocation and memory initialization overheads dominate. However, as scenario count grows ($S \ge 1024$), the thousands of concurrent CUDA threads and hardware register warp-shuffle reductions fully saturate the GPU, achieving a **24.36× speedup** over the serial baseline (more than 3.3× faster than the 12-thread CPU).

### 4.5 Experiment E5: Value of Planning with Uncertainty
To evaluate the advantage of stochastic optimization, a schedule optimized deterministically ($S=1$, zero randomness) was compared against a schedule optimized under uncertainty ($S=1024$), both scored on identical 20,000 unseen test days:

| Planning Strategy | Test Mean Cost (s) | Test P95 Cost (s) | Late Aircraft Share (%) |
| :--- | :---: | :---: | :---: |
| **Deterministic Planning** | 27,046.60 | 61,280.40 | 18.5% |
| **Robust RunwayMC** | **15,063.20** | **34,510.80** | **5.8%** |

*Conclusion:* Robust optimization reduced unseen test-day delay costs by **44.3%** and cut the share of flights missing deadlines by more than two-thirds.

---

## 5. Conclusion & Future Work

RunwayMC demonstrates that parallel Monte Carlo optimization is a viable and effective strategy for robust air traffic scheduling. By mapping independent simulation instances across multi-core processors, the system achieves a 7.30× CPU speedup, allowing realistic Monte Carlo evaluations within seconds.

Future work includes extending the exact simulation core to multi-runway configurations with dependent arrivals and departures, modeling surface taxiway routing constraints, and integrating flight telemetry data.
