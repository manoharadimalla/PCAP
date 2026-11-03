# RunwayMC — CHANGE NOTE: "Make it simple and explainable"
### Addendum to `SPEC.md` and `SPEC_CHANGES_GUI.md` — contains ONLY the changes

> **Precedence rule for the agent:** where this note conflicts with `SPEC.md` or `SPEC_CHANGES_GUI.md`, **this note wins**. Anything not mentioned here stays as it is.
>
> **Goal:** the human author must be able to explain the whole project, every file and every result, in a viva. Fewer ideas, fewer knobs, fewer experiments. **Nothing may be added in this change note's scope; this is a reduction.**

---

## 1. The project in plain words (this is the story every document must tell)

1. We have a list of aircraft that must use **one runway**. The **order** decides how long aircraft wait.
2. To judge an order we **simulate many random days** (random delays, sometimes a runway closure) and take the **average weighted waiting time**.
3. A search program (**simulated annealing**) tries swapping/moving aircraft and keeps better orders. Early on it also accepts some worse ones so it doesn't get stuck.
4. The simulation of thousands of random days is the slow part, and each day is independent. So we **run it in parallel**: **serial → OpenMP (CPU threads) → CUDA (GPU)**.
5. We check that (a) the search beats the simple first-come-first-served order, (b) all versions give the same answer, (c) parallel versions are faster, and (d) planning with randomness beats planning without it, **tested on random days the search never saw**.

```
aircraft list ─► try a new order ─► simulate S random days ─► average delay = score
      ▲                                   │   (this is the parallel part)
      └────────── keep it if better ◄─────┘
```

**GPU mapping in one sentence:** one GPU block = one search chain; each thread in the block simulates a different random day; the block adds up the results.

---

## 2. What changes at a glance

| Area | Before | After |
|---|---|---|
| Disturbances | delays + weather multiplier + closure | **delays + closure** (weather removed) |
| Objective | mean + λ·std | **mean cost** (λ removed) |
| Reported metrics | mean, std, p95, CVaR95, miss probability, emergency delay, plan deviation | **mean, p95, % aircraft past deadline, emergency mean delay** |
| Search | epochs, fresh noise each epoch, share-best-with-worst-25%, manual temperature options, 3 move types with mix slider | **one run per chain, no epochs, no sharing**, auto temperature, fixed moves |
| Validation | in-sample, fresh-scenario selection, out-of-sample | **train days → test days** (two sets only) |
| Decoder | exact decoder + adjacent-only fast path + triangle check | **exact decoder only** |
| OpenMP | 3 modes (chains, scenarios, hybrid-SIMD) | **2 modes (chains, scenarios)** |
| CUDA | K1 naive, K2, K3 with ablation steps, block-size sweep | **2 kernels: naive vs main**, one fixed block size |
| Experiments | E1–E10 | **5 experiments (E1–E5)** |
| GUI | 6 tabs, separation editor, project save/open, replay, compare, convergence | **3 tabs, small settings panel** |
| Docs | SPEC (34 KB) + 2 change notes + DECISIONS + report | **README + `docs/PROJECT.md` + `docs/EXPLAIN.md` + report**; old specs archived |

---

## 3. Model changes

1. **Remove the weather multiplier.** Delete `weather_mult`, `weather_prob` and the `mult[s]` array everywhere (types, noise generators CPU+GPU, kernels, params file, GUI, report). Separation is always the plain matrix value.
2. **Keep** per-aircraft random delay (Exp, means ARR 30 s / DEP 60 s) and the **one random closure window** per scenario (probability 0.20, duration 300–900 s).
3. **Cost of one random day** stays: `Σ weight × delay` where `delay = max(0, time − target)`, plus the deadline penalty (**constant 5×, no longer a user setting**). Priority weights stay fixed at 1 / 3 / 20 (hidden constants).
4. **Objective = average cost over the S training days.** Remove λ and the std term from the search, CLI (`--lambda`), params file, kernels and GUI.
5. **Metrics, computed on the test days only:**
   - `mean_cost`
   - `p95_cost`
   - `late_pct` = average share of aircraft that finish after their deadline (**replaces miss probability**, which saturated at 1.0 in earlier experiments)
   - `emergency_mean_delay`
   Remove `std`, `cvar95`, `plan_dev`, `miss_prob` from code, JSON, CSVs, GUI and report.
6. **Decoder:** keep only the **exact** decoder (remember the last runway time used by each of the 6 aircraft classes). Delete the adjacent-only fast path and its equivalence test. One decoder = one thing to explain.
7. **Keep** the 6×6 separation file (`config/separation_default.csv`) and the worked-example tests (445, 198, closure 260). Update any test that depended on weather/λ.

---

## 4. Algorithm changes

1. **Simulated annealing, one run per chain.** Remove: epochs, per-epoch noise resampling, "overwrite worst 25% chains with the best", `--epochs`, manual `--t0/--t-end`, `--move-mix`. Keep: auto-calibrated start temperature (≈50% of worse moves accepted at start), geometric cooling, **two moves: swap two aircraft, or move one aircraft to another position**. (Remove the adjacent-swap move.)
2. **Train / test split.**
   - Training days: `S` scenarios (default 1024), generated once per run, shared by all chains. The search only ever sees these.
   - Test days: **20,000 fresh scenarios**, used once at the end for every reported number. The best chain is chosen by training score; its test score is reported.
   - Report the train-vs-test gap honestly (it shows overfitting to the training days).
3. **Baselines (keep, unchanged):** FCFS (sort by target time), random search (same evaluation budget as SA), brute force for n ≤ 9.
4. **Total work is defined as** `chains × iterations × S × n`. The same number is used for serial, OpenMP and CUDA.
5. **Remaining user-visible parameters** (CLI + GUI): instance, implementation, threads, chains, scenarios `S`, iterations, seed, delay means, closure probability. Everything else is a fixed constant documented in `docs/PROJECT.md`.

---

## 5. Parallel implementations

| Keep | Remove |
|---|---|
| Serial reference | — |
| OpenMP **chains** mode (parallel across chains) | OpenMP **hybrid/SIMD** mode |
| OpenMP **scenarios** mode (parallel across days, with reduction) | — |
| CUDA **main kernel**: one block per chain, one thread per scenario, shared-memory aircraft data, constant-memory separation matrix, warp-shuffle sum, cuRAND noise pre-generated once | K3 as a separate variant, block-size sweep, multi-stream ideas |
| CUDA **naive kernel**: one thread evaluates one whole candidate (kept only for the "naive vs good design" comparison, **only if it already compiles and passes the equivalence test; otherwise delete it**) | — |

Merge K2/K3 into one kernel named `main`. Remove `--kernel k3`, `--omp-mode hybrid`, `--block` (fix 256). `src/omp/sa_omp.cpp` and `src/cuda/*` shrink accordingly; comments in each kernel must explain **why** (one sentence each) in plain language.

**Explain-ability rule:** the report and `EXPLAIN.md` explain the two OpenMP modes as "coarse-grained (a whole chain per thread) vs fine-grained (the days of one chain split among threads)", and the GPU design as in Section 1.

---

## 6. Experiments (replace E1–E10 with these five)

All use the fairness rules from `SPEC.md` §10: same total work, warm-up, ≥10 repetitions with median and IQR, hardware info saved with each CSV. Every number in any document must come from these CSVs.

| ID | Question | What is measured | Output |
|----|----------|------------------|--------|
| **E1** | Is it correct? | Unit tests pass; GPU and CPU give the same cost for the same schedules and the same noise (relative error < 1e-4); SA matches brute force on n=8 | pass/fail table |
| **E2** | Does optimizing help? | FCFS vs random search vs SA: `mean_cost` and `p95_cost` **on the same test days for all three** | table + bar chart |
| **E3** | How do CPU threads scale? | OpenMP chains and scenarios modes, threads 1…max, speedup and efficiency vs the **serial implementation** | line chart |
| **E4** | When does the GPU win? | Serial vs OpenMP vs CUDA time as `S` grows (128 … 16384) at fixed n (e.g. 50): shows the crossover point | line chart (+ end-to-end vs kernel-only time) |
| **E5** | Does planning with randomness help? | Schedule optimized with **no randomness** vs schedule optimized **with** randomness, both scored on the same test days (`mean_cost`, `p95_cost`, `late_pct`) | table + bar chart |

**Removed:** E6 (kernel ablation), E7 (quality-vs-time), E9 (sensitivity), E10 (Beasley). Remove their scripts, CSVs and figures from the repo and report (they stay in git history).

**Instance for E2–E5:** n = 30 to 50 aircraft, congestion ρ = 1.2. Before running, `bench/calibrate.py` must confirm FCFS is at least 10% worse than the best schedule found. **E5 must use an instance and deadlines where `late_pct` is clearly below 100%** (earlier runs saturated).

---

## 7. Fixes required before the experiments are rerun (known problems in the current repo)

1. **CUDA has never been compiled or run.** Before any simplification: build with `-DUSE_CUDA=ON` (local CUDA Toolkit + Visual Studio Build Tools C++ workload on Windows, or `scripts/colab_run.ipynb`), run the CUDA tests, and save a GPU run. If it cannot be done, state this in the report; do not claim GPU results.
2. **Old E2 compared different scenario sets** (SA appeared better than the brute-force optimum, which is impossible on equal terms). New E2 scores every method's final schedule on the **same test days**.
3. **Old E3 showed efficiency above 1.0** at 2 and 4 threads, which points to a noisy or unfair 1-thread baseline. New E3 uses the real serial implementation as the baseline, repeated runs, and reports IQR.
4. **Old E8/E9 had miss probability ≈ 1.0**, so it said nothing. Replaced by `late_pct` and a less congested instance (see Section 6).
5. **Wording:** `docs/DECISIONS.md` currently says CUDA code is "tested" and the report title says "GPU-Accelerated". Change both to match what has actually been run. The title may keep "GPU" only if E4 contains real CUDA measurements.

---

## 8. GUI simplification (replaces most of `SPEC_CHANGES_GUI.md`)

**Keep:** the PySide6 desktop window **and its two launchers: `RUN_GUI.bat` (opens the pop-up window) and `RUN_DEMO.bat` / `run_demo.py` (terminal demo), both updated to use only CLI flags that still exist**; the aircraft table (add, delete, duplicate, undo/redo, import/export CSV, generate random); Run and Cancel; Save PNG on charts.

**Settings panel (only these controls):** implementation (serial / OpenMP / CUDA), threads, chains, scenarios `S`, iterations, seed, arrival and departure delay means, closure probability. Presets: **Quick demo** and **Standard**. CUDA disabled with tooltip if no GPU.

**Result tabs (only three):**
1. **Summary** — table FCFS vs optimized: `mean_cost`, `p95_cost`, `late_pct`, `emergency_mean_delay`, % improvement; plus run time and simulations/second.
2. **Timeline** — runway chart, FCFS vs optimized.
3. **Distribution** — histogram of test-day costs, FCFS vs optimized.

**Remove:** Scenario Replay tab, Convergence tab, Compare dialog, benchmark viewer, separation-matrix editor panel (the file stays editable by hand), project save/open, preferences dialog, weather and λ controls, "Pop out". Remove `runway replay` and `--dump-costs` only if nothing else uses them (the Distribution tab still needs the test-day costs, so keep `--dump-costs`).

The progress bar may be indeterminate ("Running…") because epochs are gone.

GUI rules from the earlier note still apply: thin front-end, no algorithms in Python, numbers only from program output.

---

## 9. Repository and documentation cleanup

1. Tag the current commit `before-simplify` and make sure all tests pass.
2. Move `SPEC.md`, `SPEC_CHANGES_GUI.md` and `SPEC_CHANGES_SIMPLE.md` to `docs/archive/`. Do not delete them.
3. Create **`docs/PROJECT.md`** (max 3 pages): the story (Section 1), the model in 10 lines, the fixed constants table, the algorithm in 10 lines, the five experiments, how to build and run, limitations, disclaimer.
4. Replace `docs/DECISIONS.md` with a short `CHANGELOG.md` (what changed and why, one line each).
5. Shrink `README.md` to: what it is, build, run (CLI and GUI), where results are, disclaimer.
6. Update `report/REPORT.md` to match the simplified model, five experiments, and real data. Keep `[CITATION NEEDED]` markers; do not invent citations. Remove every sentence that refers to removed features.
7. Remove dead code, flags, tests, scripts, CSVs and figures belonging to removed features. **Delete, do not comment out.** Run all tests after each removal step.
8. Result files keep their `env_info` hardware record.
9. Launchers (`RUN_GUI.bat`, `RUN_DEMO.bat`) must not hard-code machine-specific paths such as `C:\Users\Admin\...`; they should use tools from `PATH`, with one clearly marked editable line at the top for overrides, so the demo also works on another computer.

---

## 10. The explanation kit (the agent MUST generate `docs/EXPLAIN.md`)

Contents, all in plain language for a student who did not write the code:

1. **60-second pitch** (adapt Section 1).
2. **One diagram** of the pipeline and one of the GPU mapping.
3. **Glossary** (max 15 terms): scenario, chain, simulated annealing, Monte Carlo, FCFS, wake separation, closure window, p95, train/test days, speedup, efficiency, crossover, thread, block, reduction.
4. **"What each file does"** table: max 12 rows.
5. **Twelve likely viva questions with 2–3 sentence answers.** Numbers in answers must be read from the CSVs by a script, never typed by hand. Questions to include:
   1. What problem are you solving and why is it hard?
   2. Why simulated annealing and not trying all orders?
   3. What is a scenario and why do you need thousands?
   4. What exactly runs in parallel?
   5. Why do whole chains scale better than splitting the days in OpenMP?
   6. Why can the GPU be slower for small `S`?
   7. How did you map the problem to CUDA?
   8. How do you know the GPU gives the same answer as the CPU?
   9. Why test on days the search never saw?
   10. What is your baseline and how much better are you?
   11. Does planning with randomness really help (E5)?
   12. What are the limitations and what would you do next?
6. **"What we did NOT do"**: no real ATC rules, one runway only, simplified separation, simulated data, no weather, no taxiways, the GUI is only a demo front-end.

---

## 11. Phases and gates for this cleanup

| Phase | Work | Gate |
|---|---|---|
| **S0** | Tag `before-simplify`; run all tests | Tests pass |
| **S1** | CUDA build and run (Section 7.1) | CUDA tests pass and one GPU solve completes, **or** a written note explains why not |
| **S2** | Model and algorithm reduction (Sections 3–5) | All remaining tests pass; worked examples 445 / 198 / 260 still pass; serial/OpenMP/CUDA agree on cost for fixed schedules |
| **S3** | Fix and rerun E1–E5 (Sections 6–7) | CSVs and figures regenerate with one command; E5 `late_pct` below 100% |
| **S4** | GUI reduction (Section 8) | GUI runs the default example on every available implementation and shows the three tabs; `RUN_GUI.bat` opens the window and `RUN_DEMO.bat` completes without errors |
| **S5** | Docs, report, `EXPLAIN.md` (Sections 9–10) | Every number in the docs traces to a CSV; no mention of removed features |

**Cut line if time is short:** do S1, S2, S3, S5. For S4, only hide the removed GUI parts instead of deleting their code.

Estimated effort: 4–5 days.

---

## 12. Agent rules for this change

1. Reduce only. Do not add features, options, or experiments.
2. After every deletion step run the full test suite; if a test fails because it tested a removed feature, remove that test; otherwise fix the code.
3. No hand-typed performance numbers; no invented citations; report anything that could not be run.
4. Log each removal as one line in `CHANGELOG.md`.

## 13. Kickoff prompt to paste into the agent

> Read `SPEC_CHANGES_SIMPLE.md`. It overrides `SPEC.md` and `SPEC_CHANGES_GUI.md`. Start with S0 and S1 (tag the repo, then get CUDA compiled and run, or tell me exactly why it cannot be). After that, do S2 to S5 in order, running all tests after every step and committing per phase. The goal is a smaller project whose every part I can explain: reduce only, add nothing. Begin with S0 and report what you find.
