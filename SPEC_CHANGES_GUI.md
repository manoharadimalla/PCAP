# RunwayMC — CHANGE NOTE: Desktop Pop-up Window GUI
### Addendum to `SPEC.md` — contains ONLY the changes

> **Precedence rule for the agent:** where this note conflicts with `SPEC.md`, **this note wins**. Everything in `SPEC.md` not mentioned here stays exactly as it is.

---

## 1. Summary of the change

| | Before (`SPEC.md` §7.6.2) | After (this note) |
|---|---|---|
| GUI type | Streamlit app in a web browser | **Native desktop application**: double-click/launch → a **pop-up window** opens |
| Toolkit | Streamlit | **PySide6 (Qt)** + **matplotlib** embedded charts |
| Editing | Basic table + sidebar | **Edit everything**: aircraft, scenario/weather/closure settings, separation matrix, priority weights, search and execution settings |
| Results | Page tabs | Result tabs in the same window **plus "Pop out" buttons** that open any chart in its own window |
| Launch | `streamlit run gui/app.py` | `python -m gui` (also `python gui/main.py`) |
| Extras | — | Save/Load project, Export results, Undo/Redo in the aircraft table, live convergence plot |

**Unchanged principle (MUST):** the GUI is a **thin front-end**. It writes input files, runs the `runway` binary, reads the output files, and draws them. **No scheduling or simulation logic in Python.**

---

## 2. Edits to `SPEC.md` sections

| Section in `SPEC.md` | Action |
|---|---|
| §1 Deliverables, item 4 | **Replace** "local GUI (Streamlit)" with "**desktop GUI (PySide6 pop-up window)** to edit inputs, run any implementation, and view results (this note)". |
| §3 Scope (in scope) | **Replace** "thin local GUI" with "thin desktop GUI". Out-of-scope list unchanged. |
| §6 Repository layout, `gui/` | **Replace** the `gui/` block with the one in §3 below. |
| §7.6.2 GUI (Streamlit …) | **Delete** and **replace** with §4–§8 of this note. |
| §7.6.1 Input files | **Unchanged**, plus the new `runway validate` command (§9 below). |
| §7.6.3 GUI rules | **Keep** all rules; **add** the extra rules in §8 below. |
| §9 Testing, GUI smoke-test row | **Replace** with the test list in §10 below. |
| §11 Phase P6b | **Replace** with the P6b row in §11 below. |
| §11 Cut lines | "GUI tabs 4–5" becomes "GUI Compare dialog and Benchmark viewer (§5.3)". Input validation, main window, editors and results tabs remain in "Never cut". |
| §12 Rule 10 | **Replace** "Streamlit" wording with: "The GUI is a thin PySide6 front-end: no scheduling/simulation logic in Python, no hand-typed numbers, and it must work with the CPU-only build." |
| §14 Definition of Done, GUI item | **Replace** with the checklist in §12 below. |
| Appendix B CLI | **Add** `runway validate` and the result/replay JSON schemas (§9). |
| Appendix C kickoff prompt | **Append** the sentence in §13 below. |

---

## 3. New repository layout for `gui/`

```
gui/
├── __main__.py            # enables `python -m gui`
├── main.py                # creates QApplication + MainWindow
├── main_window.py         # menus, docks, tabs, status bar, Run/Cancel
├── panels/
│   ├── aircraft_panel.py  # editable aircraft table
│   ├── scenario_panel.py  # noise, weather, closure, penalty
│   ├── separation_panel.py# 6x6 matrix editor
│   ├── search_panel.py    # chains, epochs, iterations, lambda, seed, move mix
│   └── execution_panel.py # implementation, threads, kernel, block size
├── results/
│   ├── summary_tab.py
│   ├── timeline_tab.py    # runway Gantt
│   ├── distribution_tab.py
│   ├── replay_tab.py
│   ├── convergence_tab.py
│   └── compare_dialog.py
├── models.py              # data classes for GUI state <-> files (no algorithms)
├── runner.py              # QProcess wrapper: start, parse PROGRESS lines, cancel
├── io_files.py            # write instance.csv / params.cfg / separation.csv; read result JSON
├── plots.py               # pure matplotlib functions (shared with notebook + make_plots.py)
├── requirements.txt       # PySide6, matplotlib, numpy, pandas
└── tests/                 # pytest + pytest-qt
```

If PySide6 cannot be installed on the user's machine, fall back to Tkinter with the same panel structure and tell the user (record in `docs/DECISIONS.md`).

---

## 4. Main window design

Launched with `python -m gui`. A normal desktop window (resizable, remembers size/position).

```
┌─ RunwayMC ───────────────────────────────────────────────────── – □ ✕ ┐
│ File   Edit   Run   View   Help                                        │
├───────────────────────────┬────────────────────────────────────────────┤
│ INPUTS  (left dock)       │ RESULTS  (tabs)                            │
│ ▾ Aircraft                │ [Summary][Timeline][Distribution][Replay]  │
│   [table + toolbar]       │ [Convergence][Compare]                     │
│ ▸ Scenario settings       │                                            │
│ ▸ Separation matrix       │   (charts / tables render here;            │
│ ▸ Search settings         │    each has a "Pop out ⧉" button)          │
│ ▸ Execution               │                                            │
├───────────────────────────┴────────────────────────────────────────────┤
│ [▶ Run] [■ Cancel]  ███████░░░ 70%   status text         Log ▾         │
└────────────────────────────────────────────────────────────────────────┘
```

- Left dock sections are collapsible group boxes; the splitter between inputs and results is draggable.
- Menus:
  - **File:** New project, Open project…, Save project, Save as…, Import instance CSV…, Export instance CSV…, Export results…, Quit.
  - **Edit:** Undo, Redo, Add aircraft, Delete selected, Duplicate selected, Generate random instance…, Restore defaults.
  - **Run:** Run, Cancel, Compare implementations…
  - **View:** Pop out current chart, Show log, Light/Dark theme.
  - **Help:** About (shows the disclaimer from `SPEC.md` §3), Hardware info (from `runway info`).
- Status bar: current instance size n, selected implementation, GPU detected yes/no.
- **About dialog and first-launch dialog** show the mandatory disclaimer.

---

## 5. What the user can edit (all inputs)

### 5.1 Input panels

**Aircraft panel (table, editable cells)**

| Column | Editor |
|---|---|
| ID | text (unique) |
| Type | dropdown ARR / DEP |
| Wake | dropdown H / M / L |
| Earliest (ready) | time `HH:MM[:SS]` |
| Scheduled (target) | time `HH:MM[:SS]` |
| Deadline (latest) | time `HH:MM[:SS]` |
| Priority | dropdown Normal / High / Emergency |
| Weight | number, auto-filled from priority weights, overridable |

Toolbar: Add, Delete, Duplicate, Move up/down, Undo/Redo, Import CSV, Export CSV, **Generate random…** (opens a small dialog: n, congestion `rho`, seed, arrival share, wake mix, priority mix). Pre-filled on first launch with the 4-aircraft example from the project proposal. Invalid cells are highlighted red with a tooltip (message from `runway validate`, never from GUI logic). Time display can be toggled between clock time and seconds.

**Scenario settings panel**

| Setting | Control | Default |
|---|---|---|
| Arrival delay mean (s) | spin box | 30 |
| Departure delay mean (s) | spin box | 60 |
| Weather multipliers (3) | 3 spin boxes | 1.0 / 1.15 / 1.30 |
| Weather probabilities (3) | 3 spin boxes (must sum to 1; shows warning) | 0.70 / 0.20 / 0.10 |
| Closure probability `p_close` | slider + spin box | 0.20 |
| Closure duration min / max (s) | 2 spin boxes | 300 / 900 |
| Deadline-miss penalty `M_PENALTY` | spin box | 5 |
| Priority weights Normal / High / Emergency | 3 spin boxes | 1 / 3 / 20 |

**Separation matrix panel** — editable 6×6 grid (rows = leader, cols = follower; class order ARR-H, ARR-M, ARR-L, DEP-H, DEP-M, DEP-L), buttons: *Reset to defaults*, *Import CSV*, *Export CSV*. A status label shows "triangle inequality OK / violated (exact decoder will be used)" taken from `runway validate`.

**Search settings panel** — chains C, epochs E, iterations per epoch I, scenarios per evaluation S, final out-of-sample scenarios `S_final`, λ, seed, move mix (swap/insert/adjacent sliders summing to 100%), temperature: auto-calibrate (default) or manual T0/T_end.

**Execution panel** — implementation (serial / OpenMP / CUDA), OpenMP mode (chains / scenarios / hybrid), threads, CUDA kernel variant (K1/K2/K3), block size. CUDA options are **disabled with a tooltip** if `runway info` reports no GPU. Presets: **Quick demo** (a few seconds), **Standard**, **Heavy**.

### 5.2 Results tabs (all drawn from the output JSON)

| Tab | Content |
|---|---|
| **Summary** | Table FCFS vs optimized: mean cost, std, p95, CVaR95, deadline-miss probability, Emergency mean/max delay, plan deviation, % improvement; run timings (kernel-only, end-to-end), simulations/s; hardware line. |
| **Timeline** | Runway Gantt chart: FCFS vs optimized planned times; bars coloured by wake class; priority marker; target and deadline ticks; hover tooltip with aircraft details; toggle clock time/seconds. |
| **Distribution** | Histogram (with p95 line) of out-of-sample scenario costs, FCFS vs optimized; optional box plot. |
| **Replay** | Spin box/slider for scenario index; Gantt of the realized times in that scenario with the closure window shaded and the weather multiplier shown (data from `runway replay`). |
| **Convergence** | Best J versus epoch/time, drawn live while the run progresses (from `PROGRESS` lines). |
| **Compare** | (dialog, §5.3) |

Every chart has: **Pop out ⧉** (opens that chart in its own resizable window), **Save PNG/PDF**, **Export data CSV**.

### 5.3 Compare implementations dialog and benchmark viewer
- **Compare dialog** (Run → Compare implementations…): runs serial / OpenMP / CUDA with identical current settings, shows a table (time, simulations/s, speedup, final J) and a bar chart. Must display: *"Single quick runs, not a benchmark. Use the benchmark harness for report numbers."*
- **Benchmark viewer** (View → Benchmark results…): read-only window listing `results/*.csv` and `results/figures/*` and the hardware info file.

*(These two are the first GUI parts to cut if time is short.)*

---

## 6. Run behaviour

1. **Run** → GUI creates `runs/<timestamp>/`, writes `instance.csv`, `params.cfg`, `separation.csv`, and `gui_state.json` there.
2. GUI calls `runway validate`; if errors exist, highlight the offending cells, show the messages, and **do not start**.
3. GUI starts the solver in a background process (`QProcess`):
   `runway solve --instance … --params … --impl … --progress --dump-costs 20000 --out runs/<ts>/result.json` (+ the settings from the panels).
4. The window stays responsive. Progress bar and convergence plot update from lines `PROGRESS <epoch>/<E> <best_J>`. **Cancel** terminates the process cleanly.
5. On success: read `result.json`, then call `runway replay` on demand for the Replay tab; fill all result tabs.
6. On failure: show the program's stderr in the Log panel and a message box. Never fail silently.
7. **Save project** stores the whole run folder layout (inputs + `gui_state.json`); **Open project** restores every panel exactly.
8. **Export results** writes the result JSON, a metrics CSV, and the figures to a chosen folder.

---

## 7. File formats the GUI uses (all produced/consumed by the C++ program)

**`result.json` (MUST be produced by `runway solve`):**
```
{
  "impl": "...", "seed": 1, "params": {...}, "n": 50,
  "perm_fcfs": [...], "perm_best": [...],
  "planned_times_fcfs": [...], "planned_times_best": [...],     # seconds, indexed by aircraft id order
  "metrics": { "fcfs": {...}, "best": {...} },                   # mean, std, p95, cvar95, miss_prob, emerg_mean, emerg_max, plan_dev
  "costs_fcfs": [...], "costs_best": [...],                      # up to --dump-costs values (out-of-sample)
  "convergence": [ {"epoch": 1, "best_J": 0.0, "elapsed_s": 0.0}, ... ],
  "timings": { "kernel_ms": 0.0, "end_to_end_ms": 0.0, "sim_per_s": 0.0 },
  "hardware": {...}, "git_hash": "..."
}
```
(The values above are placeholders showing the schema only.) FCFS is always evaluated on the same out-of-sample scenarios as the optimized schedule.

**`runway replay` output:** `{ "scenario": 17, "weather_mult": 1.15, "closure": [a, b], "aircraft": [ {"id": "A1", "ready_real": 0.0, "time": 0.0, "delay": 0.0}, ... ] }`.

**`runway validate` output:** `{ "ok": true|false, "errors": [ {"row": 3, "column": "latest", "message": "..."} ], "warnings": [...], "triangle_ok": true|false }`. Accepts `--instance`, `--params`, `--separation`.

---

## 8. Additional GUI rules (added to `SPEC.md` §7.6.3)

1. No algorithm in Python: the GUI never sorts, schedules, simulates, or computes separation validity. Unit conversion (seconds ↔ clock time) and percentage differences are the only allowed computations.
2. All displayed numbers come from `result.json`, `replay` output, or `results/*.csv`.
3. The GUI must start and work on a **CPU-only** build (CUDA controls disabled).
4. Long operations never block the UI thread.
5. Every setting in §5.1 must round-trip: edit → Run → `params.cfg` → program uses it → Save/Open project restores it.
6. Works on Windows, Linux and macOS; no hard-coded paths; the `runway` binary location is configurable (Edit → Preferences, default `build/runway`).
7. On a machine without a display (e.g. Colab), use `gui/plots.py` from the notebook instead.

---

## 9. CLI additions needed (add to Appendix B)

```
runway validate --instance FILE [--params FILE.cfg] [--separation FILE.csv]   # JSON as in §7
runway solve    ... --progress --dump-costs 20000 --out FILE.json             # result.json as in §7
runway replay   --instance FILE --perm FILE --scenario-index 17 --params FILE.cfg --seed 1
runway info                                                                  # JSON: version, cuda available, GPU name/SMs, OpenMP max threads
```
`runway solve` must also accept the extra flags the panels expose: `--omp-mode`, `--threads`, `--kernel`, `--block`, `--move-mix a,b,c`, `--t0 auto|X`, `--t-end X`, `--s-final N`.

---

## 10. Tests that replace the old GUI smoke test (Phase P6b)

Run headless with `QT_QPA_PLATFORM=offscreen` and `pytest-qt`:

| Test |
|---|
| Window opens, default 4-aircraft example is loaded, no exceptions. |
| Editing a cell, adding/deleting/duplicating rows, undo/redo work and update the written `instance.csv`. |
| Invalid cell (duplicate id, `target > latest`, bad time text) is highlighted with the message from `runway validate`; Run stays disabled. |
| Weather probabilities not summing to 1 show a warning and block Run. |
| Quick-demo run completes; the Summary table values equal the numbers in `result.json` exactly. |
| GUI run result equals a direct CLI run with the same seed. |
| Cancel stops the process within 2 s and leaves the UI usable. |
| CUDA controls disabled when `runway info` reports no GPU. |
| Save project → close → Open project restores every panel value. |
| CSV import and table entry produce byte-identical instance files for the same data. |
| "Pop out" opens a separate window for each chart type; Save PNG creates a file. |

---

## 11. Replacement for the Phase P6b row (`SPEC.md` §11)

| Phase | Work | Gate |
|---|---|---|
| **P6b** Desktop GUI | CLI additions (§9); `gui/` main window, five input panels, Summary/Timeline/Distribution/Replay/Convergence tabs, Run/Cancel, Save/Open project, Export, pop-out charts; then (cuttable) Compare dialog and Benchmark viewer | GUI tests in §10 pass; user can edit every setting in §5.1, run each available implementation, and see all result tabs |

Estimated effort: about 3–4 days, done in week 5 alongside P6.

---

## 12. Replacement for the Definition-of-Done GUI item

- [ ] `python -m gui` opens a pop-up desktop window; the user can edit aircraft, scenario/weather/closure settings, separation matrix, priority weights, search and execution settings; run serial/OpenMP/CUDA; see Summary, Timeline, Distribution, Replay and Convergence results; pop out charts; save/open projects; export results. GUI tests pass.

---

## 13. Sentence to append to the Appendix C kickoff prompt

> The GUI is a **desktop PySide6 pop-up window** as defined in `SPEC_CHANGES_GUI.md`, which overrides the Streamlit GUI in `SPEC.md`. Build it in phase P6b as a thin front-end over the `runway` binary, with no algorithms in Python.
