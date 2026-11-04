"""
Matplotlib plotting functions for RunwayMC GUI.

Pure plotting routines with no Qt dependencies. Can be embedded into
FigureCanvasQTAgg in Qt tabs or used headlessly in scripts and notebooks.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import numpy as np


def _format_time_val(seconds: float, time_format: str) -> str:
    if time_format == "clock":
        total_s = int(abs(seconds))
        h = total_s // 3600
        m = (total_s % 3600) // 60
        s = total_s % 60
        sign = "-" if seconds < 0 else ""
        return f"{sign}{h:02d}:{m:02d}:{s:02d}"
    return f"{seconds:.0f}s"


WAKE_COLORS = {
    "H": "#e74c3c",  # Red for Heavy
    "M": "#3498db",  # Blue for Medium
    "L": "#2ecc71",  # Green for Light
}


def plot_timeline(
    ax: Any,
    planned_fcfs: List[float],
    planned_best: List[float],
    aircraft_data: List[Dict[str, Any]],
    time_format: str = "seconds",
) -> None:
    """
    Plot Gantt timeline comparing FCFS planned times vs Optimized planned times.
    """
    ax.clear()
    n = len(aircraft_data)
    if n == 0 or (not planned_fcfs and not planned_best):
        ax.text(0.5, 0.5, "No schedule data to display", ha="center", va="center", transform=ax.transAxes)
        return

    bar_height = 0.35
    duration = 45.0  # Nominal runway occupancy display width

    y_fcfs = 1.0
    y_best = 0.0

    all_times: List[float] = []

    # Map aircraft by id or order
    for idx, ac in enumerate(aircraft_data):
        wake = str(ac.get("wake", "M")).upper()
        color = WAKE_COLORS.get(wake, "#95a5a6")
        aid = ac.get("id", idx)
        try:
            ac_idx = int(aid)
        except (ValueError, TypeError):
            ac_idx = idx

        prio = str(ac.get("priority", "Normal")).capitalize()
        target = float(ac.get("target", 0.0))
        latest = float(ac.get("latest", 0.0))

        # FCFS bar
        if ac_idx < len(planned_fcfs):
            t_fcfs = planned_fcfs[ac_idx]
            all_times.extend([t_fcfs, target, latest])
            ax.broken_barh(
                [(t_fcfs, duration)],
                (y_fcfs - bar_height / 2, bar_height),
                facecolors=color,
                edgecolors="#2c3e50",
                linewidth=1,
                alpha=0.85,
            )
            ax.text(
                t_fcfs + duration / 2,
                y_fcfs,
                f"A{aid}",
                ha="center",
                va="center",
                fontsize=8,
                color="white",
                fontweight="bold",
            )
            # Target tick
            ax.plot([target, target], [y_fcfs + 0.2, y_fcfs + 0.3], color="black", lw=1.5)
            # Deadline mark
            ax.plot([latest, latest], [y_fcfs - 0.3, y_fcfs - 0.2], color="red", lw=1.5)

            if prio == "Emergency":
                ax.plot(t_fcfs + duration / 2, y_fcfs + 0.25, marker="^", color="red", markersize=7)
            elif prio == "High":
                ax.plot(t_fcfs + duration / 2, y_fcfs + 0.25, marker="d", color="gold", markersize=6)

        # Best bar
        if ac_idx < len(planned_best):
            t_best = planned_best[ac_idx]
            all_times.extend([t_best, target, latest])
            ax.broken_barh(
                [(t_best, duration)],
                (y_best - bar_height / 2, bar_height),
                facecolors=color,
                edgecolors="#2c3e50",
                linewidth=1,
                alpha=0.85,
            )
            ax.text(
                t_best + duration / 2,
                y_best,
                f"A{aid}",
                ha="center",
                va="center",
                fontsize=8,
                color="white",
                fontweight="bold",
            )
            # Target tick
            ax.plot([target, target], [y_best + 0.2, y_best + 0.3], color="black", lw=1.5)
            # Deadline mark
            ax.plot([latest, latest], [y_best - 0.3, y_best - 0.2], color="red", lw=1.5)

            if prio == "Emergency":
                ax.plot(t_best + duration / 2, y_best + 0.25, marker="^", color="red", markersize=7)
            elif prio == "High":
                ax.plot(t_best + duration / 2, y_best + 0.25, marker="d", color="gold", markersize=6)

    ax.set_yticks([y_best, y_fcfs])
    ax.set_yticklabels(["Optimized", "FCFS"], fontweight="bold", fontsize=10)
    ax.set_ylim(-0.6, 1.6)

    if all_times:
        min_x = max(0.0, min(all_times) - 60.0)
        max_x = max(all_times) + 120.0
        ax.set_xlim(min_x, max_x)

    ax.set_xlabel(f"Runway Time ({'HH:MM:SS' if time_format == 'clock' else 'seconds'})", fontsize=10)
    ax.set_title("Planned Runway Schedule: FCFS vs Optimized", fontsize=11, fontweight="bold", pad=10)
    ax.grid(True, axis="x", linestyle="--", alpha=0.5)

    # Custom legend
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_elements = [
        Patch(facecolor=WAKE_COLORS["H"], label="Heavy (H)"),
        Patch(facecolor=WAKE_COLORS["M"], label="Medium (M)"),
        Patch(facecolor=WAKE_COLORS["L"], label="Light (L)"),
        Line2D([0], [0], color="black", lw=1.5, label="Target Time (tick)"),
        Line2D([0], [0], color="red", lw=1.5, label="Deadline (tick)"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="red", markersize=7, label="Emergency"),
        Line2D([0], [0], marker="d", color="w", markerfacecolor="gold", markersize=6, label="High Priority"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=8, ncol=2)


def plot_distribution(
    ax: Any,
    costs_fcfs: List[float],
    costs_best: List[float],
    show_box: bool = False,
) -> None:
    """
    Plot overlaid cost distributions for FCFS vs Optimized schedules.
    """
    ax.clear()
    if not costs_fcfs and not costs_best:
        ax.text(0.5, 0.5, "No cost distribution data available\n(Run with --dump-costs to generate)",
                ha="center", va="center", transform=ax.transAxes)
        return

    arr_fcfs = np.array(costs_fcfs) if costs_fcfs else np.array([])
    arr_best = np.array(costs_best) if costs_best else np.array([])

    if show_box:
        box_data = []
        labels = []
        if len(arr_fcfs) > 0:
            box_data.append(arr_fcfs)
            labels.append("FCFS")
        if len(arr_best) > 0:
            box_data.append(arr_best)
            labels.append("Optimized")
        if box_data:
            ax.boxplot(box_data, labels=labels, patch_artist=True,
                       boxprops=dict(facecolor="#3498db", alpha=0.6))
            ax.set_ylabel("Scenario Cost", fontsize=10)
            ax.set_title("Cost Distribution Boxplot", fontsize=11, fontweight="bold")
            ax.grid(True, linestyle="--", alpha=0.5)
        return

    # Determine common binning range
    all_vals = []
    if len(arr_fcfs) > 0:
        all_vals.append(arr_fcfs)
    if len(arr_best) > 0:
        all_vals.append(arr_best)
    concat_vals = np.concatenate(all_vals)
    bins = np.linspace(np.min(concat_vals), np.max(concat_vals), 50)

    if len(arr_fcfs) > 0:
        p95_fcfs = float(np.percentile(arr_fcfs, 95))
        ax.hist(arr_fcfs, bins=bins, alpha=0.5, color="#e67e22", label=f"FCFS (Mean={np.mean(arr_fcfs):.1f})", density=True)
        ax.axvline(p95_fcfs, color="#d35400", linestyle="--", linewidth=2, label=f"FCFS P95 ({p95_fcfs:.1f})")

    if len(arr_best) > 0:
        p95_best = float(np.percentile(arr_best, 95))
        ax.hist(arr_best, bins=bins, alpha=0.6, color="#2980b9", label=f"Optimized (Mean={np.mean(arr_best):.1f})", density=True)
        ax.axvline(p95_best, color="#1b4f72", linestyle="--", linewidth=2, label=f"Optimized P95 ({p95_best:.1f})")

    ax.set_xlabel("Scenario Cost", fontsize=10)
    ax.set_ylabel("Probability Density", fontsize=10)
    ax.set_title("Out-of-Sample Cost Distribution: FCFS vs Optimized", fontsize=11, fontweight="bold", pad=10)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.5)


def plot_convergence(ax: Any, convergence_data: List[Dict[str, Any]]) -> None:
    """
    Plot best J vs epoch / time.
    """
    ax.clear()
    if not convergence_data:
        ax.text(0.5, 0.5, "No convergence history yet", ha="center", va="center", transform=ax.transAxes)
        return

    epochs = [d.get("epoch", idx + 1) for idx, d in enumerate(convergence_data)]
    best_js = [float(d.get("best_J", 0.0)) for d in convergence_data]

    ax.plot(epochs, best_js, marker="o", color="#2980b9", linewidth=2.5, markersize=5, label="Best J (In-Sample)")

    if best_js:
        min_j = min(best_js)
        min_idx = best_js.index(min_j)
        ax.plot(epochs[min_idx], min_j, marker="*", color="#e74c3c", markersize=12, label=f"Min J = {min_j:.2f}")
        ax.annotate(
            f"{min_j:.2f}",
            (epochs[min_idx], min_j),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontweight="bold",
            color="#c0392b",
        )

    ax.set_xlabel("Epoch", fontsize=10)
    ax.set_ylabel("Objective Function J", fontsize=10)
    ax.set_title("Simulated Annealing Convergence Across Epochs", fontsize=11, fontweight="bold", pad=10)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", fontsize=9)


def plot_replay(
    ax: Any,
    replay_data: Dict[str, Any],
    aircraft_data: Optional[List[Dict[str, Any]]] = None,
    time_format: str = "seconds",
) -> None:
    """
    Plot realized timeline for a single scenario with runway closure shading.
    """
    ax.clear()
    if not replay_data:
        ax.text(0.5, 0.5, "Select a scenario and click Replay", ha="center", va="center", transform=ax.transAxes)
        return

    scen_idx = replay_data.get("scenario", 0)
    w_mult = replay_data.get("weather_mult", 1.0)
    closure = replay_data.get("closure", [0.0, 0.0])
    ac_list = replay_data.get("aircraft", [])

    if not ac_list:
        ax.text(0.5, 0.5, f"Scenario {scen_idx}: No aircraft realized data", ha="center", va="center", transform=ax.transAxes)
        return

    # Shaded closure region
    if len(closure) >= 2 and closure[1] > closure[0]:
        c_start, c_end = closure[0], closure[1]
        ax.axvspan(
            c_start, c_end,
            color="#e74c3c", alpha=0.25, hatch="//",
            label=f"Runway Closure [{c_start:.0f}s - {c_end:.0f}s]"
        )

    # Wake lookup from aircraft_data if available
    wake_map = {}
    if aircraft_data:
        for a in aircraft_data:
            wake_map[str(a.get("id"))] = a.get("wake", "M")

    y_pos = 0
    all_x = []
    if len(closure) >= 2 and closure[1] > closure[0]:
        all_x.extend(closure)

    for ac in ac_list:
        aid = str(ac.get("id", y_pos))
        ready_real = float(ac.get("ready_real", 0.0))
        t_service = float(ac.get("time", ready_real))
        delay = float(ac.get("delay", max(0.0, t_service - ready_real)))
        wake = wake_map.get(aid, "M")
        color = WAKE_COLORS.get(wake, "#3498db")

        all_x.extend([ready_real, t_service, t_service + 30.0])

        # Bar for service
        ax.broken_barh(
            [(t_service, 30.0)],
            (y_pos - 0.35, 0.7),
            facecolors=color,
            edgecolors="#2c3e50",
            linewidth=1,
        )
        # Delay connector line if delayed
        if t_service > ready_real:
            ax.plot([ready_real, t_service], [y_pos, y_pos], color="#7f8c8d", linestyle=":", lw=1.5)
            ax.plot(ready_real, y_pos, marker="|", color="#7f8c8d", markersize=8)

        label_txt = f"A{aid} (+{delay:.0f}s)" if delay > 0 else f"A{aid}"
        ax.text(t_service + 15.0, y_pos, label_txt, ha="center", va="center", fontsize=7.5, color="white", fontweight="bold")
        y_pos += 1

    ax.set_yticks(range(len(ac_list)))
    ax.set_yticklabels([f"Op {i}" for i in range(len(ac_list))], fontsize=8)
    ax.set_ylim(-0.8, len(ac_list) - 0.2)
    if all_x:
        ax.set_xlim(max(0.0, min(all_x) - 30.0), max(all_x) + 60.0)

    ax.set_xlabel(f"Timeline ({'HH:MM:SS' if time_format == 'clock' else 'seconds'})", fontsize=10)
    ax.set_title(f"Scenario #{scen_idx} Replay | Weather Multiplier: {w_mult:.2f}x", fontsize=11, fontweight="bold", pad=10)
    ax.grid(True, axis="x", linestyle="--", alpha=0.5)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, loc="upper right", fontsize=8)
