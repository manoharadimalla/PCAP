# RunwayMC — GPU-Accelerated Robust Runway Scheduling under Uncertainty
### Project Manual & Build Specification (for a coding agent, e.g. Antigravity)

> **How to read this document.** It is a complete build specification. Follow the phases in order (Section 11). Do not start a phase until the previous phase's **gate** passes. Everything marked **MUST** is mandatory. Everything marked **STRETCH** is optional and only allowed after Phase 7 is complete.

---

## 1. Project summary

**Title:** RunwayMC: GPU-accelerated Monte Carlo evaluation for robust single-runway scheduling.

**Problem.** Given a set of aircraft (arrivals and departures) with ready times, target times, deadlines, wake-turbulence class and priority, find the **order** in which they use one runway so that **expected weighted delay is minimized under random disturbances** (delays, weather, runway closures), while always respecting minimum separation.

**Why it is a parallel-computing project.** A schedule's quality is estimated by simulating it under thousands of random scenarios. A search algorithm needs to evaluate huge numbers of candidate schedules. The workload has two independent levels of parallelism:
- **Level 1 — scenarios:** the S Monte Carlo scenarios of one candidate are independent.
- **Level 2 — chains/candidates:** many independent search chains run at once.

**Research question.**
> How much does GPU-accelerated Monte Carlo evaluation improve the speed and the robustness of metaheuristic runway scheduling compared with serial and OpenMP implementations?

**Deliverables.**
1. C++17/CUDA code with three implementations: **serial**, **OpenMP**, **CUDA**, sharing one simulation core.
2. Instance generator, baselines (FCFS, random search, brute-force optimum for small n), validator.
3. Automated benchmark harness producing CSVs and plots.
4. A written report (outline in Section 13).

---

## 2. Risk register → design decisions already made

| # | Risk | Decision built into this spec |
|---|------|-------------------------------|
| R1 | "Generate millions of random schedules" is a weak algorithm (n! search space). | Use **parallel Simulated Annealing (SA)** to *search*, and Monte Carlo to *evaluate*. Pure random search is kept **only as a baseline** to show why it is insufficient. |
| R2 | GPU design is tricky (divergence, memory, RNG). | One **CUDA block = one SA chain**, one **thread = one scenario**. Schedule lives in shared memory. Noise is **pre-generated** (cuRAND) in a coalesced layout, so the hot kernel contains no RNG calls and almost no divergence. Naive kernel kept for ablation. |
| R3 | Claiming safety-critical realism. | Project is a **research simulation**. Simplified separation rules, clearly labelled illustrative. Disclaimer required in README and report. |
| R4 | Scope creep (weather, taxiways, multi-runway). | **Single runway MVP.** Disturbances = scenario-level weather separation multiplier + one random runway closure window per scenario (both are cheap, near-branch-free additions to the same timeline simulation). Everything else is STRETCH, max one item. |
| R5 | GPU not available. | Phase 0 detects hardware. CUDA is an **optional CMake target**. CPU-only build must always work. A Google Colab notebook is a required fallback for GPU runs. |
| R6 | Weak baseline (FCFS already near-optimal → boring results). | **Congestion-controlled instance generator** + a **calibration gate**: test instances must show FCFS is ≥10% worse than the best found. |
| R7 | Unfair/inflated speedup claims. | Fairness rules in Section 10 (equal work, optimized CPU baseline, transfers included, warm-up, repeated runs). A crossover point where CPU wins is an *expected finding*, not a failure. |
| R8 | Overfitting to the sampled scenarios (winner's curse). | Scenarios are **resampled each epoch**; the final schedule is evaluated **out-of-sample** on a fresh, larger scenario set. |
| R9 | CPU and GPU implementing different logic. | **One header** (`sim_core.hpp`) with `__host__ __device__` simulation code compiled by both. Equivalence tests mandatory. |
| R10 | Float non-determinism. | Fixed seeds, tolerance-based comparisons (Section 9). |
| R11 | Agent fabricates numbers or citations. | Rules in Section 12: every number in the report must come from a generated CSV; no invented citations. |
| R12 | Running out of time. | Cut lines in Section 11. |

---

## 3. Scope

**In scope (MUST):**
- One runway, mixed arrivals and departures.
- Wake classes Heavy/Medium/Light; priorities Normal/High/Emergency (as weights).
- Random per-aircraft delay + scenario-level weather separation multiplier + random runway closure window.
- Serial, OpenMP, CUDA implementations of the same SA + Monte Carlo algorithm.
- Baselines: FCFS, random search, brute force (n ≤ 9).
- Out-of-sample robustness evaluation.

**Out of scope:** taxiways, apron, gates, airspace routing, real-time operation, real ATC procedures, certification of any kind.

**STRETCH (pick at most one, only after Phase 7):**
- (a) Second runway (assignment gene + per-runway separation).
- (b) *(moved into the core model, see Section 4.4; letter kept so other references stay stable)*
- (c) CVaR objective instead of mean+λ·std.
- (d) Constrained Position Shifting (max shift K from FCFS order).
- (e) OR-Tools CP-SAT baseline for deterministic version.
- (f) Multi-stream / pinned-memory overlap on GPU.
- (g) Beasley OR-Library benchmark loader (Section 8.3).

**Mandatory disclaimer** (README + report): *"This is an academic simulation with simplified separation standards. It is not validated for, and must not be used in, real air-traffic operations."*

---

## 4. Problem model

### 4.1 Aircraft
Each aircraft `i` (0-indexed, `n` total) has:

| Field | Meaning |
|-------|---------|
| `op` | ARR or DEP |
| `wake` | H, M, or L |
| `cls` | class index = `op*3 + wake` (0..5). Used to index the separation matrix |
| `ready` | earliest feasible time (seconds from t=0) the aircraft can use the runway |
| `target` | scheduled/ETA time; delay is measured against it |
| `latest` | soft deadline |
| `weight` | Normal=1, High=3, Emergency=20 (configurable) |

### 4.2 Separation
`sep[a][b]` = minimum seconds between runway use by **leader** class `a` and **follower** class `b`. Default (illustrative, configurable in `config/separation_default.csv`):

- ARR→ARR wake table (seconds), rows = leader, cols = follower:

|   | H | M | L |
|---|---|---|---|
| H | 96 | 157 | 196 |
| M | 60 | 69 | 131 |
| L | 60 | 69 | 82 |

- DEP→DEP: `max(60, round(0.6 * ARR_table))`
- ARR→DEP: 60 s. DEP→ARR: 75 s.

**Important — triangle inequality.** With mixed operations the matrix can violate `sep[a][c] <= sep[a][b] + sep[b][c]`. If it does, checking only the immediately preceding aircraft is **wrong** (e.g. ARR-H, DEP, ARR-L at 0, 60, 135 breaks the 196 s H→L rule). Therefore:

- At load time compute `triangle_ok`.
- **Exact decoder (default, always correct):** keep `lastT[c]` = last runway time used by class `c` (init `-1e9`), and use `t = max(ready', max_c(lastT[c] + m*sep[c][cls_a]))`.
- **Fast path:** if `triangle_ok`, use only the previous aircraft.
- A test MUST confirm both paths give identical results when `triangle_ok` is true.

### 4.3 Schedule (decision variable)
A permutation `π` of `0..n-1`: the order of runway use. Times are **derived** by the decoder (the runway-use time is as early as constraints allow).

### 4.4 One scenario `s`
- `m_s` — separation multiplier (weather): 1.0 w.p. 0.70, 1.15 w.p. 0.20, 1.30 w.p. 0.10 (configurable).
- `d_{s,i}` — extra ready delay: ARR ~ Exp(mean 30 s), DEP ~ Exp(mean 60 s) (configurable; `0` = deterministic).
- `[a_s, b_s)` — **runway closure window**: with probability `p_close` (default 0.20) the runway is closed for one interval. Duration ~ U(300, 900) s; start ~ U(`t_min`, `t_max − duration`), where `t_min`/`t_max` are the smallest/largest `target` in the instance. No operation may **start** inside the interval. If the scenario has no closure, set `a_s = b_s` (an empty interval, so the check in 4.5 is a no-op and needs no special case). At most one closure per scenario.
- Realized ready time `ready'_i = ready_i + d_{s,i}`.

### 4.5 Simulation of one (schedule, scenario) pair
```
tPrev = none; cost = 0; misses = 0
for k in 0..n-1:
    a = π[k]
    t = ready[a] + d[s][a]
    t = max(t, max over classes c of ( lastT[c] + m_s * sep[c][cls[a]] ))   # or adjacent-only fast path
    t = (t >= closeA[s] && t < closeB[s]) ? closeB[s] : t                    # runway closure: wait until it reopens
    lastT[cls[a]] = t
    delay = max(0, t - target[a])
    cost += weight[a] * delay
    if t > latest[a]: cost += M_PENALTY * weight[a] * (t - latest[a]);  misses += 1
```
`M_PENALTY` default 5 (a deadline overrun costs 5× a normal delay second). The order `π` is fixed (planned order is held); only the times adapt. This mirrors "hold the sequence, absorb the disturbance." The closure step is applied **after** the separation `max`: separation is a *minimum*, so pushing a time later can never violate it, and with one closure per scenario a single push is exact. Write it as a predicated select (no real branch).

### 4.6 Objective
For a candidate `π` evaluated on `S` scenarios with costs `c_1..c_S`:

```
J(π) = mean(c) + λ · std(c)         (λ default 1.0)
```
Report additionally (for final schedules only, computed on CPU): `p95(c)`, `CVaR95(c)`, `P(deadline miss)`, mean/max delay of Emergency flights, and **plan deviation** (mean |realized − deterministic planned time|).

### 4.7 Worked example (MUST be a unit test)
Three arrivals A(H), B(M), C(L), all `ready=0`, `target=0`, `weight=1`, no noise, `m=1`, default matrix.
- Order A,B,C → times 0, 157, 288 → cost **445**.
- Order C,B,A → times 0, 69, 129 → cost **198**.
- **Closure example:** same aircraft, order C,B,A, closure `[50, 100)`. C at 0; B would be 69 → inside closure → pushed to 100; A = max(0+60 (L→H), 100+60 (M→H)) = 160 → cost 0+100+160 = **260**. With an empty closure (`a=b`) the cost is 198 again.

---

## 5. Algorithm

### 5.1 Parallel Simulated Annealing with Monte Carlo evaluation

```
Input: instance, S scenarios, C chains, epochs E, iterations per epoch I
Init: each chain starts from FCFS (chain 0) or FCFS + random perturbations (others)
for epoch in 1..E:
    generate fresh noise array (S scenarios)   # same noise shared by all chains in this epoch (common random numbers)
    parallel for chain c in 1..C:
        J_cur = evaluate(π_c)                  # re-evaluate on the NEW noise
        for it in 1..I:
            π' = move(π_c)                      # swap / insert / adjacent-swap (40/40/20 %)
            J' = evaluate(π')                   # <-- the S-scenario Monte Carlo (Level-1 parallelism)
            accept if J' < J_cur or rand < exp((J_cur - J')/T)
            T *= alpha
        record chain best
    synchronise: re-score the C chain-bests on a FRESH scenario set,
                 pick global best, overwrite the worst 25% of chains with it
Final: evaluate global best on OUT-OF-SAMPLE set (S_final = 20000 default)
```
- **Temperature:** auto-calibrate `T0` so ~50% of worsening moves are accepted at start (sample 200 random moves); `T_end = 1e-3 * T0`; geometric cooling over all `E*I` iterations.
- **Common random numbers (CRN):** all chains in one epoch use the same noise, which reduces comparison variance.
- **Winner's curse control:** selection of the global best uses fresh scenarios, never the ones the chain optimized on.

### 5.2 Baselines (MUST)
1. **FCFS:** sort by `target`.
2. **Random search:** sample `N` random permutations, keep best (same total evaluation budget as SA). Demonstrates the weakness of pure sampling.
3. **Brute force:** all permutations for `n ≤ 9` (also `n=10` allowed) under the same robust objective with a fixed scenario set. This gives the true optimum for validation.
4. **Deterministic-optimized schedule:** run SA with zero noise (`S=1`, `d=0`, `m=1`), then evaluate it under the stochastic out-of-sample set. Comparing it to the robust schedule shows the *value of modelling uncertainty* (experiment E8).

---

## 6. Architecture & repository layout

```
runway-mc/
├── CMakeLists.txt             # options: -DUSE_CUDA=ON/OFF, -DUSE_OPENMP=ON/OFF
├── README.md                  # build, run, disclaimer
├── docs/
│   ├── SPEC.md                # this file
│   ├── DECISIONS.md           # log every deviation from this spec + why
│   └── HARDWARE.md            # auto-generated machine info
├── config/separation_default.csv
├── include/runway/
│   ├── types.hpp              # Instance, Scenario params, Result structs
│   ├── sim_core.hpp           # __host__ __device__ simulate_one(); SINGLE SOURCE OF TRUTH
│   ├── instance_io.hpp
│   └── rng.hpp                # small PCG/xorshift for moves (host+device)
├── src/
│   ├── core/    instance_io.cpp, validator.cpp, objective.cpp
│   ├── gen/     instance_gen.cpp, noise_cpu.cpp
│   ├── baselines/ fcfs.cpp, random_search.cpp, brute_force.cpp
│   ├── serial/  sa_serial.cpp
│   ├── omp/     sa_omp.cpp
│   ├── cuda/    noise_gpu.cu, eval_kernels.cu, sa_kernel.cu, sa_gpu.cu
│   └── app/     main.cpp      # CLI
├── tests/                     # Catch2 or GoogleTest (via FetchContent)
├── bench/
│   ├── configs/*.yaml
│   ├── run_bench.py           # runs experiments, writes results/*.csv
│   └── env_info.py            # records CPU/GPU/compiler/git hash
├── scripts/
│   ├── make_plots.py
│   └── colab_run.ipynb        # clone → build → run on Colab GPU
├── data/instances/            # generated instances
├── results/                   # CSVs + figures (git-tracked small ones)
└── report/                    # report draft, figures
```

**Language/tooling:** C++17, CUDA 11+ (`-arch=native` or detected `sm_XX`), OpenMP, CMake ≥ 3.18, Python 3 (numpy, pandas, matplotlib) for scripts. No exotic dependencies.

**Rule:** `sim_core.hpp` defines `RW_HD` = `__host__ __device__` when compiled by nvcc, empty otherwise. `simulate_one(...)` and the move operators are written once there.

---

## 7. Implementation specifications

### 7.1 Data layout (all implementations)
- **Structure of Arrays** for aircraft: `ready[], target[], latest[], weight[], cls[]`.
- **Noise array layout is aircraft-major:** `noise[a * S + s]`. Threads/SIMD lanes working on neighbouring scenarios `s` read neighbouring addresses (coalesced on GPU, vectorizable on CPU).
- Scenario weather multipliers: `mult[s]`. Closure windows: `closeA[s]`, `closeB[s]` (empty interval `a=b` when no closure). These are three small per-scenario arrays, read coalesced like `noise`.
- Precision: **float** for the hot path on all platforms; CPU reference evaluator in **double** for validation.

### 7.2 Serial CPU (`sa_serial.cpp`)
Straight implementation of Section 5.1: one chain after another, loop over scenarios. It must be **cleanly written and compiled with `-O3 -march=native`**. It is the baseline of record. No intentionally slow code.

### 7.3 OpenMP (`sa_omp.cpp`) — three variants (selectable `--omp-mode`)
- **O1 `chains`:** `#pragma omp parallel for schedule(static)` over chains; scenarios serial inside. Coarse-grained.
- **O2 `scenarios`:** chains serial; the scenario loop inside `evaluate()` is parallelized with a reduction (`reduction(+:sum,sumsq)`). Fine-grained (expect overhead for small S; report it).
- **O3 `hybrid`:** chains across threads + `#pragma omp simd` over scenarios.
Each chain owns its RNG (seeded `seed ^ chain_id` via splitmix). Support `OMP_NUM_THREADS`, and document thread-pinning (`OMP_PROC_BIND=close`, `OMP_PLACES=cores`).

### 7.4 CUDA — kernels (`src/cuda/`)

**Noise generation (`noise_gpu.cu`)**: cuRAND host API (e.g. Philox/XORWOW generator) → uniform array `U[n*S]` → small transform kernel: `d = -mean[op[a]] * logf(U)`; also generate `mult[s]`, and the closure arrays `closeA[s]`, `closeB[s]` (Bernoulli(`p_close`), duration, start) from uniforms. Seeded per epoch (`seed + epoch`). A test flag `--noise-from-host` uploads CPU-generated noise instead, for equivalence tests.

**K1 — naive baseline (`thread-per-candidate`)**: one thread evaluates one full candidate across all S scenarios in a serial loop; permutation in local memory. Kept for the ablation (this is the "thread per schedule" design from the original idea). Expect it to be poor with few candidates.

**K2 — main kernel (`block-per-chain`, `thread-per-scenario`)**
```
grid  = C blocks (one SA chain per block)
block = 128 or 256 threads (tunable)
shared: perm[n], proposal[n], aircraft arrays (ready,target,latest,weight,cls), reduction buffers, T, J_cur
init:   cooperative load of aircraft arrays + start perm into shared memory
loop it = 0..I-1:
    thread 0: propose move on proposal[] (from shared copy of perm), using per-chain RNG state
    __syncthreads()
    each thread: for s = tid; s < S; s += blockDim.x:
          (cost_s, miss_s) = simulate_one(proposal, s)      # reads noise[a*S+s] -> coalesced
          local_sum += cost_s; local_sumsq += cost_s*cost_s
    block reduction: warp shuffle (__shfl_down_sync) → shared → thread 0
    thread 0: mean, std → J'; Metropolis accept/reject; update perm; cool T; update chain best
    __syncthreads()
write chain best + final perm + RNG state to global memory
```
- `S` need not be a multiple of the block size (the strided loop handles it; a test covers this).
- Divergence is low because every thread executes the same fixed `n`-step loop; only `max()`/ penalty branches differ (write them branch-free with `fmaxf`/predication where possible). The closure check is a predicated select, so it adds no real divergence; scenarios with and without a closure execute the same instructions.
- Per-chain RNG state kept in a global array, loaded at kernel start, stored at end.

**K3 — optimized K2** (each step is an ablation row, measured separately):
1. aircraft arrays in shared memory (vs `__ldg` global);
2. `__restrict__`/`__ldg` on noise and separation matrix; separation matrix in `__constant__` (it is 6×6);
3. warp-shuffle reduction instead of shared-memory tree reduction;
4. loop unrolling of the class loop (`#pragma unroll`, K=6);
5. block size sweep (64/128/256/512) and `__launch_bounds__`;
6. STRETCH-f: multi-stream overlap of noise generation for epoch e+1 with search of epoch e.

**Epoch synchronization**: after each kernel launch, a small kernel/host step re-scores chain-bests on a fresh scenario set, picks the best and overwrites the worst 25% of chains.

**Error handling:** every CUDA call wrapped in a `CUDA_CHECK` macro. Run `compute-sanitizer` in Phase 4/5 gates.

**Timing:** `cudaEvent_t` for kernels; wall-clock `std::chrono` for end-to-end including H2D/D2H copies and noise generation. Report both: *kernel-only* and *end-to-end*.

### 7.5 Throughput metric
Primary throughput: **scenario-simulations per second** = `(total evaluations × S) / time`. Secondary: aircraft-steps/s = that × n.

---

## 8. Instances

### 8.1 Generator (`runway gen`)
Parameters: `--n`, `--rho` (congestion), `--seed`, `--mix-arr` (default 0.5), `--wake-mix` (H/M/L default 0.2/0.6/0.2), `--prio-mix` (Normal/High/Emergency default 0.80/0.15/0.05).

- `mean_sep` = average of `sep` weighted by the class mix.
- Targets: cumulative sum of gaps with mean gap `mean_sep / rho`, gaps ~ Exp (random arrival stream).
- `rho` = demand/capacity: **0.8** light, **1.0** busy, **1.2** congested, **1.5** heavy. Use **≥1.0 for experiments**.
- `ready = target - U(0,120)` for ARR (can speed up slightly), `ready = target` for DEP.
- `latest = target + 600` (ARR), `target + 900` (DEP).
- Shift all times so that `min(ready) ≥ 0`.
- Output CSV: `id,op,wake,ready,target,latest,priority,weight`.

### 8.2 Calibration gate (R6)
For each experiment instance, a script `bench/calibrate.py` must report `gap = (J_FCFS − J_best) / J_FCFS`. Experiment instances MUST satisfy `gap ≥ 10%`. If not, increase `rho` or noise and regenerate. Store the table in `results/calibration.csv`.

### 8.3 Benchmark instances — STRETCH-g (Beasley OR-Library)
The Aircraft Landing Problem instances `airland1..13` from the OR-Library (J.E. Beasley) can be used as structural benchmarks (≈10–500 aircraft, general separation matrices, single-runway variant available). File format per instance (verify by reading the files): number of planes and freeze time, then per plane: appearance time, earliest, target, latest, penalty cost before target, penalty cost after target, followed by that plane's separation times to all other planes.

Caveats the agent MUST respect:
- Their separation matrix is general (n×n). Map each aircraft to its own class (`K = n`); the decoder's `lastT[]` path then costs O(n) per step, so use the adjacent-only fast path only if `triangle_ok`.
- Our decoder uses the earliest feasible time, which is optimal only when early landing is free. **Published optima assume early penalties**, so they are not directly comparable. Set the early penalty to 0 for the comparison and compare against *our own* brute force / best-found, **not** against published values. Do not hard-code published optimal values from memory.
- If network access to the OR-Library is unavailable, ask the user to place the files in `data/beasley/`.

---

## 9. Testing (MUST pass before each phase gate)

| Test | Phase |
|------|-------|
| Worked example in 4.7 gives 445 and 198 exactly, and the closure example gives 260 | 1 |
| Closure edge cases: empty closure (`a=b`) equals no-closure result; closure entirely before all `ready` times has no effect; closure covering the whole horizon pushes every aircraft to `b`; closure start equal to a computed time pushes it (interval is `[a,b)`, so `t=b` is not pushed) | 1 |
| Validator: for any schedule produced by any method, ALL pairs (not just adjacent) satisfy `t_b − t_a ≥ m·sep[cls_a][cls_b]` for the realized scenario, **and no operation starts inside that scenario's closure window** | 1 |
| Moves (swap/insert/adjacent) always yield a valid permutation (fuzz 1e6 moves) | 2 |
| FCFS is sorted by target; ties broken by id | 1 |
| Exact decoder == fast-path decoder when `triangle_ok` | 1 |
| Brute-force optimum vs SA on n=8, S=500: SA reaches optimum (or within 0.5%) in ≥ 90% of 20 seeds | 2 |
| Serial determinism: same seed → bit-identical result | 2 |
| OpenMP result equals serial *evaluation* for fixed permutations + fixed noise (relative error < 1e-6, double) | 3 |
| **CPU↔GPU equivalence:** for 1000 random permutations and fixed host-supplied noise (including closure windows), `J_gpu` vs `J_cpu_float` relative error < 1e-4, vs `J_cpu_double` < 1e-3 | 4 |
| `S` not a multiple of block size (e.g. S=1000, 257) gives the same J as CPU | 4 |
| `compute-sanitizer --tool memcheck` and `racecheck` clean | 4, 5 |
| Statistical equivalence: serial, OpenMP, CUDA, 30 seeds each → final out-of-sample J not significantly different (Welch t-test p>0.01) or differences explained in DECISIONS.md | 5 |

---

## 10. Experiments (the heart of the report)

**Fairness rules (R7) — MUST:**
1. All implementations run the **same algorithm with the same total work** (chains × iterations × epochs × S).
2. CPU baseline compiled `-O3 -march=native -ffast-math` is *off* unless validated; document flags.
3. GPU timings include noise generation and memcpy ("end-to-end"). Kernel-only timing reported separately.
4. One untimed **warm-up** run (CUDA context, caches). Then **≥10 repetitions**; report **median and IQR**; fixed seeds list.
5. Record hardware (`bench/env_info.py`): CPU model/cores, GPU model/SMs/memory, driver, CUDA, compiler versions, git commit, flags. Save beside every CSV.
6. Report **speedup = T_serial / T_impl** and **parallel efficiency = speedup / #threads** for CPU.

| ID | Experiment | Output |
|----|-----------|--------|
| E1 | Correctness: tests in Section 9 | pass/fail log |
| E2 | Baseline comparison: FCFS vs random search vs SA(serial) vs brute force (n≤9), across rho ∈ {1.0,1.2,1.5} | table of J, gap |
| E3 | OpenMP strong scaling: threads 1,2,4,…,max for O1/O2/O3 at fixed problem (n=50, S=1024, C=64) | speedup/efficiency plot, Amdahl fit |
| E4 | Scaling with S ∈ {128,256,512,1024,4096,16384} and n ∈ {10,20,50,100,200} for serial/OMP/CUDA | throughput (sim/s) plots; **find CPU↔GPU crossover** |
| E5 | Scaling with chains C ∈ {1,4,16,64,256,1024} (GPU occupancy / saturation) | throughput plot |
| E6 | CUDA ablation: K1 → K2 → each K3 optimization; block-size sweep | stacked speedup bar chart; Nsight metrics if available (occupancy, branch efficiency, achieved memory throughput) |
| E7 | Quality vs time: best out-of-sample J over wall-clock time for each implementation (same instance) | convergence curves; **time-to-target** (time to reach within 2% of best-found) |
| E8 | Value of uncertainty: deterministic-optimized vs robust-optimized schedule, evaluated out-of-sample (mean, p95, CVaR95, miss prob) | table + box/violin plot |
| E9 | Sensitivity: noise level (mean delay ×0.5, ×1, ×2), λ ∈ {0,0.5,1,2}, weather probability, **closure probability `p_close` ∈ {0, 0.1, 0.2, 0.4}** (also report mean cost in closure vs non-closure scenarios) | tables |
| E10 | (STRETCH-g) Beasley-derived instances | table |

**Plot style:** every figure labelled with units, error bars/IQR, instance parameters in the caption. Save PNG and PDF to `results/figures/`.

**Analysis to include (not just numbers):** where time goes (profiling), why the GPU wins/loses at small S, effect of the sequential dependency inside one simulation (depth `n`), memory vs compute boundness (simple roofline reasoning), Amdahl's law (serial fraction from epoch synchronisation), Gustafson-style scaled speedup.

---

## 11. Phased build plan with gates

| Phase | Work | Gate (must pass to continue) |
|-------|------|------------------------------|
| **P0** Environment & skeleton | Detect `nvidia-smi`, `nvcc --version`, OpenMP, cmake. Create repo layout, CMake with `USE_CUDA` option, test framework, `env_info.py`, README with disclaimer. If no NVIDIA GPU: build CPU-only and create `scripts/colab_run.ipynb`. | `cmake` + `ctest` runs (even with 0 tests); HARDWARE.md generated |
| **P1** Core model | `types.hpp`, `sim_core.hpp`, separation loader + triangle check, instance generator, FCFS, validator, CLI `gen`/`eval` | Worked-example tests (incl. closure example and edge cases), validator tests, decoder-equivalence test pass |
| **P2** Serial search | Noise generator (CPU, incl. closure windows), evaluator, brute force, random search, serial SA, calibration script | SA ≈ brute-force optimum on n=8; **calibration gate (gap ≥10%)** satisfied on the planned instances; determinism test |
| **P3** OpenMP | O1, O2, O3 | Evaluation-equality test; scaling run produces sensible CSV |
| **P4** CUDA baseline | `noise_gpu.cu`, K1, K2, epoch sync, CLI `--impl cuda` | CPU↔GPU equivalence tests; sanitizer clean |
| **P5** CUDA optimization | K3 steps one at a time, with a timing CSV per step | Statistical equivalence test; each optimization measured (even if no gain: record it) |
| **P6** Benchmark harness | `run_bench.py`, configs, all experiments E2–E9, `make_plots.py` | All CSVs + figures regenerate with a single command (`make bench`) |
| **P7** Analysis & report | Fill report template (Section 13) from actual CSVs | Report builds; every number traceable to a CSV |
| **P8** STRETCH | At most one item from Section 3 | Doesn't break any earlier gate |

### Suggested 6-week timeline
- Week 1: P0–P1. Week 2: P2. Week 3: P3 + start P4. Week 4: P4–P5. Week 5: P6. Week 6: P7 + buffer.

### Cut lines (if behind schedule — drop in this order)
1. STRETCH items. 2. E9 and E10. 3. OpenMP variants O2/O3 (keep O1). 4. K3 optimizations beyond shared-memory + warp-shuffle + block-size sweep. **Never cut:** serial, OpenMP O1, CUDA K2, baselines, E2–E4, E7, E8, correctness tests.

---

## 12. Rules for the coding agent

1. **Work phase by phase.** At the end of each phase: run all tests, commit with message `P<k>: <summary>`, and write a short status note in `docs/DECISIONS.md`.
2. **Do not change the model or algorithm silently.** Any deviation → entry in `docs/DECISIONS.md` (what, why, impact).
3. **No fabricated results.** Never type performance numbers by hand into any document. All numbers come from CSVs produced by scripts in this repo. If a run fails or is slow, report that honestly; negative results are acceptable.
4. **No invented citations.** In the report, mark every reference as `[CITATION NEEDED]` unless the agent has actually opened the source. The human will verify references.
5. **Reproducibility:** every CLI run takes `--seed`; every result row stores seed, git hash, flags, hardware.
6. **Keep the hot path clean:** no heap allocation, no printf, no unnecessary branches inside `simulate_one`.
7. **If a required tool is missing** (no GPU, no nvcc, no network): stop that branch, state what is missing, continue with the CPU-only path, and tell the user exactly what to run elsewhere (Colab notebook).
8. **Ask the human** only for: GPU access, ambiguity in grading requirements, or approval to use a STRETCH item. Otherwise decide, document, proceed.
9. Prefer clear code to clever code; comment non-obvious GPU decisions (why shared memory, why layout, why block size).

---

## 13. Report outline (target 10–15 pages)

1. **Introduction** — runway scheduling, uncertainty, why parallel; research question; contributions.
2. **Problem formulation** — Section 4 (aircraft, separation, scenarios, objective).
3. **Related work** — Aircraft Landing Problem and runway scheduling; robust/stochastic scheduling; Monte Carlo and GPU parallelism. *Starting points to locate and verify (do not cite until the source has been opened and read):* Beasley, Krishnamoorthy, Sharaiha & Abramson, "Scheduling aircraft landings — the static case" (Transportation Science, 2000); Bennell, Mesgarpour & Potts, "Airport runway scheduling" (4OR, 2011); Balakrishnan & Chandran, "Algorithms for scheduling runway operations under constrained position shifting" (Operations Research, 2010). Additional sources on Monte Carlo/GPU air-traffic optimization to be found via search by the human author.
4. **Method** — SA + Monte Carlo, CRN, resampling, out-of-sample evaluation.
5. **Parallel implementations** — serial, OpenMP variants, CUDA kernels (with diagrams of the block/thread mapping and memory layout).
6. **Experimental setup** — hardware, instances, calibration, fairness rules.
7. **Results** — E1–E9, figures and tables.
8. **Discussion** — Amdahl/Gustafson, divergence, memory behaviour, crossover point, quality vs time, value of uncertainty modelling.
9. **Limitations** — simplified separation, single runway, simulated (not real) data, disclaimer.
10. **Conclusion & future work** — STRETCH items.

---

## 14. Definition of Done (checklist)

- [ ] `cmake -B build -DUSE_CUDA=ON && cmake --build build && ctest --test-dir build` passes (and CPU-only build passes).
- [ ] Serial, OpenMP, CUDA implementations produce statistically equivalent solution quality.
- [ ] Baselines FCFS, random search, brute force implemented and compared (E2).
- [ ] Calibration table shows every experiment instance has FCFS gap ≥ 10%.
- [ ] `make bench` (or `python bench/run_bench.py --all`) regenerates all CSVs and figures.
- [ ] E3, E4, E6, E7, E8 completed with plots, median/IQR, hardware info.
- [ ] CPU↔GPU crossover point identified and discussed.
- [ ] README with build/run instructions and the disclaimer.
- [ ] `docs/DECISIONS.md` complete; no hand-typed performance numbers anywhere.
- [ ] Report draft complete with `[CITATION NEEDED]` markers resolved by the human author.

---

## Appendix A — Default parameters

| Parameter | Default |
|-----------|---------|
| n (experiments) | 20, 50, 100, 200 |
| rho | 1.2 (also 1.0, 1.5) |
| S (search) | 1024 |
| S_final (out-of-sample) | 20000 |
| Chains C | 64 (GPU: 256) |
| Epochs E | 20 |
| Iterations per epoch I | 2000 |
| λ (std weight) | 1.0 |
| M_PENALTY | 5 |
| Cooling | geometric, `T_end = 1e-3·T0`, T0 auto-calibrated (50% worsening acceptance) |
| Move mix | swap 40%, insert 40%, adjacent-swap 20% |
| Noise | ARR Exp(30 s), DEP Exp(60 s) |
| Weather | m = 1.0 / 1.15 / 1.30 with p = 0.70 / 0.20 / 0.10 |
| Runway closure | `p_close` = 0.20; duration U(300, 900) s; start U(`t_min`, `t_max − duration`); one window per scenario |
| CUDA block size | 256 (sweep 64–512) |
| Seeds | 1..30 for statistics; 1..10 for timing |

## Appendix B — CLI reference

```
runway gen    --n 50 --rho 1.2 --seed 1 --out data/instances/n50_r12_s1.csv
runway eval   --instance FILE --perm fcfs|FILE --scenarios 1024 --seed 1
runway solve  --instance FILE --impl serial|omp|cuda|random|brute \
              --chains 64 --scenarios 1024 --iters 2000 --epochs 20 \
              --lambda 1.0 --seed 1 --threads 8 --omp-mode chains|scenarios|hybrid \
              --block 256 --kernel k1|k2|k3 --out results/run.json
runway final  --instance FILE --perm FILE --scenarios 20000 --seed 999   # out-of-sample evaluation
```
Each `solve` run writes a JSON/CSV row: implementation, parameters, seed, J(in-sample), J(out-of-sample), mean, std, p95, CVaR95, miss prob, kernel time, end-to-end time, sim/s, git hash.

## Appendix C — Kickoff prompt to paste into the agent

> Read `docs/SPEC.md` completely. You are building the RunwayMC project exactly as specified. Start with Phase P0. After each phase: run the tests, commit, update `docs/DECISIONS.md`, and then summarize what passed before moving to the next phase. Follow Section 12 strictly: no hand-typed performance numbers, no invented citations, and report any missing tool (GPU, nvcc, network) instead of working around it silently. If there is no NVIDIA GPU on this machine, build the CPU-only path and prepare the Colab notebook for GPU experiments. Begin now with P0 and tell me what hardware you detected.
