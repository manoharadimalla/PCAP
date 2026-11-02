"""
Replay Tab for RunwayMC GUI.

Embeds scenario-specific Gantt chart visualizing realized flight times under
sampled delay noise, runway closure windows, and weather multipliers,
obtained by calling `runway replay`.
"""

from __future__ import annotations

import csv
import json
import os
import tempfile
from typing import Any, Callable, Dict, List, Optional

import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from gui.plots import plot_replay
from gui.results.base_tab import BaseResultTab
from gui.runner import run_command_sync


class ReplayTab(BaseResultTab):
    """
    Result tab for inspecting realized scenario outcomes.
    """

    def __init__(
        self,
        get_binary_path_cb: Optional[Callable[[], Optional[str]]] = None,
        get_run_context_cb: Optional[Callable[[], Dict[str, str]]] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__("Replay", parent)
        self.get_binary_path = get_binary_path_cb or (lambda: None)
        self.get_run_context = get_run_context_cb or (lambda: {})
        self.current_replay_data: Dict[str, Any] = {}
        self.time_display_clock: bool = False

        self._setup_scenario_controls()
        self._setup_figure()

    def _setup_scenario_controls(self) -> None:
        self.toolbar.addSeparator()

        self.toolbar.addWidget(QLabel("Scenario Index:"))
        self.spin_scenario = QSpinBox()
        self.spin_scenario.setRange(0, 99999)
        self.spin_scenario.setValue(0)
        self.spin_scenario.valueChanged.connect(self._on_scenario_changed)
        self.toolbar.addWidget(self.spin_scenario)

        self.btn_replay = QPushButton("Replay Scenario")
        self.btn_replay.clicked.connect(self.load_replay_for_current_scenario)
        self.toolbar.addWidget(self.btn_replay)

        self.btn_toggle_time = QPushButton("Time: Sec")
        self.btn_toggle_time.clicked.connect(self._toggle_time_format)
        self.toolbar.addWidget(self.btn_toggle_time)

    def _setup_figure(self) -> None:
        self.fig = Figure(figsize=(8, 4.5), dpi=100)
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.ax = self.fig.add_subplot(111)
        self._main_layout.addWidget(self.canvas)
        self.fig.tight_layout()

    def _toggle_time_format(self) -> None:
        self.time_display_clock = not self.time_display_clock
        self.btn_toggle_time.setText("Time: Clock" if self.time_display_clock else "Time: Sec")
        self._render_chart()

    def _on_scenario_changed(self) -> None:
        self.load_replay_for_current_scenario()

    def update_view(self) -> None:
        if self.current_result:
            s_final = self.current_result.get("params", {}).get("s_final", 20000)
            self.spin_scenario.setMaximum(max(0, s_final - 1))
            self.load_replay_for_current_scenario()

    def load_replay_for_current_scenario(self) -> None:
        scenario_idx = self.spin_scenario.value()
        ctx = self.get_run_context()
        binary = self.get_binary_path()

        inst_path = ctx.get("instance_path")
        params_path = ctx.get("params_path")
        perm_path = ctx.get("perm_best_path")
        seed = ctx.get("seed", 42)

        if not binary or not os.path.isfile(binary) or not inst_path or not os.path.isfile(inst_path):
            # Display fallback placeholder or message
            self.current_replay_data = {
                "scenario": scenario_idx,
                "weather_mult": 1.0,
                "closure": [0.0, 0.0],
                "aircraft": [],
            }
            self._render_chart()
            return

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_out = tmp.name

        try:
            args = [
                "replay",
                "--instance", inst_path,
                "--scenario-index", str(scenario_idx),
                "--seed", str(seed),
            ]
            if perm_path and os.path.isfile(perm_path):
                args.extend(["--perm", perm_path])
            if params_path and os.path.isfile(params_path):
                args.extend(["--params", params_path])
            args.extend(["--out", tmp_out])

            code, stdout, stderr = run_command_sync(binary, args, timeout_ms=5000)

            # Check if JSON written to tmp_out or stdout
            data = None
            if os.path.isfile(tmp_out) and os.path.getsize(tmp_out) > 0:
                with open(tmp_out, "r", encoding="utf-8") as f:
                    data = json.load(f)
            elif stdout.strip():
                try:
                    data = json.loads(stdout.strip())
                except Exception:
                    pass

            if data and isinstance(data, dict):
                self.current_replay_data = data
            else:
                self.current_replay_data = {
                    "scenario": scenario_idx,
                    "weather_mult": 1.0,
                    "closure": [0.0, 0.0],
                    "aircraft": [],
                }
        except Exception:
            self.current_replay_data = {
                "scenario": scenario_idx,
                "weather_mult": 1.0,
                "closure": [0.0, 0.0],
                "aircraft": [],
            }
        finally:
            if os.path.exists(tmp_out):
                os.remove(tmp_out)

        self._render_chart()

    def _render_chart(self) -> None:
        aircraft_data = [ac.to_dict() for ac in self.current_state.aircraft_list] if self.current_state else []
        time_fmt = "clock" if self.time_display_clock else "seconds"
        plot_replay(self.ax, self.current_replay_data, aircraft_data=aircraft_data, time_format=time_fmt)
        self.fig.tight_layout()
        self.canvas.draw()

    def save_figure(self, filepath: str) -> None:
        self.fig.savefig(filepath, dpi=300, bbox_inches="tight")

    def export_csv(self, filepath: str) -> None:
        if not self.current_replay_data:
            return
        ac_list = self.current_replay_data.get("aircraft", [])
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["scenario", "id", "ready_real", "time", "delay"])
            scen = self.current_replay_data.get("scenario", 0)
            for ac in ac_list:
                writer.writerow([
                    scen, ac.get("id"), ac.get("ready_real"), ac.get("time"), ac.get("delay")
                ])

    def create_popout_widget(self) -> QWidget:
        clone = ReplayTab(self.get_binary_path, self.get_run_context, self)
        clone.current_replay_data = self.current_replay_data
        clone.current_state = self.current_state
        clone.spin_scenario.setValue(self.spin_scenario.value())
        clone._render_chart()
        return clone
