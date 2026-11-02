"""
Distribution Tab for RunwayMC GUI.

Embeds histograms and boxplots of out-of-sample Monte Carlo scenario costs,
comparing the risk profile (tail risk, P95) of FCFS vs Optimized schedules.
"""

from __future__ import annotations

import csv
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from PySide6.QtWidgets import QCheckBox, QPushButton, QVBoxLayout, QWidget

from gui.plots import plot_distribution
from gui.results.base_tab import BaseResultTab


class DistributionTab(BaseResultTab):
    """
    Result tab displaying cost probability distributions and risk percentiles.
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__("Distribution", parent)
        self.show_box_plot: bool = False

        self.chk_boxplot = QCheckBox("Show Box Plot")
        self.chk_boxplot.setToolTip("Toggle between histogram density and box plot representation")
        self.chk_boxplot.toggled.connect(self._on_boxplot_toggled)
        self.toolbar.addWidget(self.chk_boxplot)

        self._setup_figure()

    def _setup_figure(self) -> None:
        self.fig = Figure(figsize=(8, 4.5), dpi=100)
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.ax = self.fig.add_subplot(111)
        self._main_layout.addWidget(self.canvas)
        self.fig.tight_layout()

    def _on_boxplot_toggled(self, checked: bool) -> None:
        self.show_box_plot = checked
        self.update_view()

    def update_view(self) -> None:
        if not self.current_result:
            return

        res = self.current_result
        costs_fcfs = res.get("costs_fcfs", [])
        costs_best = res.get("costs_best", [])

        plot_distribution(self.ax, costs_fcfs, costs_best, show_box=self.show_box_plot)
        self.fig.tight_layout()
        self.canvas.draw()

    def save_figure(self, filepath: str) -> None:
        self.fig.savefig(filepath, dpi=300, bbox_inches="tight")

    def export_csv(self, filepath: str) -> None:
        if not self.current_result:
            return

        res = self.current_result
        costs_fcfs = res.get("costs_fcfs", [])
        costs_best = res.get("costs_best", [])

        max_len = max(len(costs_fcfs), len(costs_best))
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["scenario_index", "cost_fcfs", "cost_best"])
            for i in range(max_len):
                c_f = costs_fcfs[i] if i < len(costs_fcfs) else ""
                c_b = costs_best[i] if i < len(costs_best) else ""
                writer.writerow([i, c_f, c_b])

    def create_popout_widget(self) -> QWidget:
        clone = DistributionTab(self)
        if self.current_result:
            clone.set_data(self.current_result, self.current_state)
            clone.chk_boxplot.setChecked(self.show_box_plot)
        return clone
