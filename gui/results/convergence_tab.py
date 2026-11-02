"""
Convergence Tab for RunwayMC GUI.

Embeds the objective function J convergence plot across SA epochs,
supporting both real-time live updates during execution and post-run
replotting from result.json convergence records.
"""

from __future__ import annotations

import csv
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from PySide6.QtWidgets import QVBoxLayout, QWidget

from gui.plots import plot_convergence
from gui.results.base_tab import BaseResultTab


class ConvergenceTab(BaseResultTab):
    """
    Result tab displaying objective function J vs search progress.
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__("Convergence", parent)
        self.live_data: List[Dict[str, Any]] = []
        self._setup_figure()

    def _setup_figure(self) -> None:
        self.fig = Figure(figsize=(8, 4.5), dpi=100)
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.ax = self.fig.add_subplot(111)
        self._main_layout.addWidget(self.canvas)
        self.fig.tight_layout()

    def reset_live_history(self) -> None:
        """Clear live convergence points at the start of a new run."""
        self.live_data.clear()
        self.ax.clear()
        self.ax.text(0.5, 0.5, "Starting optimization run...", ha="center", va="center", transform=self.ax.transAxes)
        self.canvas.draw()

    def on_live_progress(self, epoch: int, total_epochs: int, best_j: float, elapsed_s: float) -> None:
        """Slot called whenever the runner emits a PROGRESS line."""
        self.live_data.append({
            "epoch": epoch,
            "total_epochs": total_epochs,
            "best_J": best_j,
            "elapsed_s": elapsed_s,
        })
        plot_convergence(self.ax, self.live_data)
        self.fig.tight_layout()
        self.canvas.draw()

    def update_view(self) -> None:
        """Called upon run completion to plot authoritative convergence from result.json."""
        if not self.current_result:
            return

        convergence = self.current_result.get("convergence", [])
        if convergence:
            self.live_data = list(convergence)
            plot_convergence(self.ax, self.live_data)
        elif self.live_data:
            plot_convergence(self.ax, self.live_data)

        self.fig.tight_layout()
        self.canvas.draw()

    def save_figure(self, filepath: str) -> None:
        self.fig.savefig(filepath, dpi=300, bbox_inches="tight")

    def export_csv(self, filepath: str) -> None:
        data = self.live_data
        if not data and self.current_result:
            data = self.current_result.get("convergence", [])

        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "best_J", "elapsed_s"])
            for d in data:
                writer.writerow([
                    d.get("epoch"), d.get("best_J"), d.get("elapsed_s", "")
                ])

    def create_popout_widget(self) -> QWidget:
        clone = ConvergenceTab(self)
        clone.live_data = list(self.live_data)
        plot_convergence(clone.ax, clone.live_data)
        clone.fig.tight_layout()
        clone.canvas.draw()
        return clone
