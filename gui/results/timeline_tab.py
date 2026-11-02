"""
Timeline Tab for RunwayMC GUI.

Embeds a Gantt chart comparing planned runway occupancy times between
FCFS baseline and the SA-optimized schedule, with wake classes, priorities,
and deadline indicators.
"""

from __future__ import annotations

import csv
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QPushButton, QVBoxLayout, QWidget

from gui.plots import plot_timeline
from gui.results.base_tab import BaseResultTab


class TimelineTab(BaseResultTab):
    """
    Result tab displaying the Gantt timeline of scheduled flights.
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__("Timeline", parent)
        self.time_display_clock: bool = False

        # Add toggle time button to tab toolbar
        self.btn_toggle_time = QPushButton("Time: Sec")
        self.btn_toggle_time.setToolTip("Toggle time display between Seconds and Clock HH:MM:SS")
        self.btn_toggle_time.clicked.connect(self._toggle_time_format)
        self.toolbar.addWidget(self.btn_toggle_time)

        self._setup_figure()

    def _setup_figure(self) -> None:
        self.fig = Figure(figsize=(8, 4.5), dpi=100)
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.ax = self.fig.add_subplot(111)
        self._main_layout.addWidget(self.canvas)
        self.fig.tight_layout()

    def _toggle_time_format(self) -> None:
        self.time_display_clock = not self.time_display_clock
        self.btn_toggle_time.setText("Time: Clock" if self.time_display_clock else "Time: Sec")
        self.update_view()

    def update_view(self) -> None:
        if not self.current_result or not self.current_state:
            return

        res = self.current_result
        state = self.current_state

        planned_fcfs = res.get("planned_times_fcfs", [])
        planned_best = res.get("planned_times_best", [])

        # Aircraft info from state
        aircraft_data = [ac.to_dict() for ac in state.aircraft_list]

        time_fmt = "clock" if self.time_display_clock else "seconds"
        plot_timeline(self.ax, planned_fcfs, planned_best, aircraft_data, time_format=time_fmt)
        self.fig.tight_layout()
        self.canvas.draw()

    def save_figure(self, filepath: str) -> None:
        self.fig.savefig(filepath, dpi=300, bbox_inches="tight")

    def export_csv(self, filepath: str) -> None:
        if not self.current_result or not self.current_state:
            return

        res = self.current_result
        state = self.current_state
        planned_fcfs = res.get("planned_times_fcfs", [])
        planned_best = res.get("planned_times_best", [])

        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "op", "wake", "ready", "target", "latest", "priority", "weight", "planned_fcfs", "planned_best"])
            for idx, ac in enumerate(state.aircraft_list):
                t_fcfs = planned_fcfs[idx] if idx < len(planned_fcfs) else ""
                t_best = planned_best[idx] if idx < len(planned_best) else ""
                writer.writerow([
                    ac.id, ac.op, ac.wake, ac.ready, ac.target, ac.latest, ac.priority, ac.weight, t_fcfs, t_best
                ])

    def create_popout_widget(self) -> QWidget:
        clone = TimelineTab(self)
        if self.current_result and self.current_state:
            clone.set_data(self.current_result, self.current_state)
        return clone
