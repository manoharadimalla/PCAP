"""
Summary Tab for RunwayMC GUI.

Displays side-by-side performance comparison between FCFS baseline
and Optimized schedule, execution timing, simulation throughput, and hardware info.
Rule: All numbers come directly from result.json (or results/*.csv) — NEVER computed in Python,
except percentage differences.
"""

from __future__ import annotations

import csv
from typing import Any, Dict, Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from gui.results.base_tab import BaseResultTab


class SummaryTab(BaseResultTab):
    """
    Summary tab showing key optimization metrics and run statistics.
    """

    METRICS = [
        ("Objective J", "J", True),
        ("Mean Cost ($/delay)", "mean_cost", True),
        ("Std Dev Cost", "std_cost", True),
        ("95th Percentile Cost (P95)", "p95", True),
        ("CVaR95 (Tail Risk)", "cvar95", True),
        ("Deadline Miss Probability", "miss_prob", True),
        ("Emergency Mean Delay (s)", "emerg_mean", True),
        ("Emergency Max Delay (s)", "emerg_max", True),
        ("Plan Deviation (s)", "plan_dev", False),
    ]

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__("Summary", parent)
        # Disable save PNG/PDF for tabular view
        self.btn_save_png.setVisible(False)
        self.btn_save_pdf.setVisible(False)
        self._setup_content()

    def _setup_content(self) -> None:
        content_widget = QWidget()
        layout = QVBoxLayout(content_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        # Metrics Table
        self.table = QTableWidget(len(self.METRICS), 4)
        self.table.setHorizontalHeaderLabels(["Metric", "FCFS Baseline", "Optimized", "Improvement (%)"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        self.table.setAlternatingRowColors(True)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)

        for r, (label, _, _) in enumerate(self.METRICS):
            item = QTableWidgetItem(label)
            item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
            self.table.setItem(r, 0, item)
            for c in range(1, 4):
                val_item = QTableWidgetItem("—")
                val_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.table.setItem(r, c, val_item)

        layout.addWidget(self.table)

        # Performance & Hardware Cards
        info_group = QGroupBox("Execution & Hardware Information")
        info_layout = QHBoxLayout(info_group)

        self.lbl_time = QLabel("Run Time: —")
        self.lbl_throughput = QLabel("Simulations/sec: —")
        self.lbl_hardware = QLabel("Hardware: —")

        self.lbl_time.setStyleSheet("font-weight: bold; color: #2c3e50;")
        self.lbl_throughput.setStyleSheet("font-weight: bold; color: #27ae60;")
        self.lbl_hardware.setStyleSheet("color: #7f8c8d;")

        info_layout.addWidget(self.lbl_time)
        info_layout.addWidget(self.lbl_throughput)
        info_layout.addWidget(self.lbl_hardware)

        layout.addWidget(info_group)
        self._main_layout.addWidget(content_widget)

    def update_view(self) -> None:
        if not self.current_result:
            return

        res = self.current_result
        metrics_dict = res.get("metrics", {})
        m_fcfs = metrics_dict.get("fcfs", {})
        m_best = metrics_dict.get("best", {})

        # If flat schema:
        if not m_best:
            m_best = {
                "J": res.get("J_out_of_sample", res.get("J_in_sample", "—")),
                "mean_cost": res.get("mean_cost", "—"),
                "std_cost": res.get("std_cost", "—"),
                "p95": res.get("p95", "—"),
                "cvar95": res.get("cvar95", "—"),
                "miss_prob": res.get("miss_prob", "—"),
                "emerg_mean": res.get("emerg_mean", "—"),
                "emerg_max": res.get("emerg_max", "—"),
                "plan_dev": res.get("plan_dev", "—"),
            }

        for r, (label, key, is_lower_better) in enumerate(self.METRICS):
            v_fcfs = m_fcfs.get(key, None)
            v_best = m_best.get(key, None)

            str_fcfs = f"{float(v_fcfs):.4f}" if isinstance(v_fcfs, (int, float)) else "—"
            str_best = f"{float(v_best):.4f}" if isinstance(v_best, (int, float)) else "—"

            imp_str = "—"
            if isinstance(v_fcfs, (int, float)) and isinstance(v_best, (int, float)) and abs(v_fcfs) > 1e-7:
                # Allowed unit/percentage difference
                diff_pct = ((v_fcfs - v_best) / abs(v_fcfs)) * 100.0 if is_lower_better else ((v_best - v_fcfs) / abs(v_fcfs)) * 100.0
                imp_str = f"{diff_pct:+.1f}%"

            item_fcfs = self.table.item(r, 1)
            item_best = self.table.item(r, 2)
            item_imp = self.table.item(r, 3)

            if item_fcfs:
                item_fcfs.setText(str_fcfs)
            if item_best:
                item_best.setText(str_best)
            if item_imp:
                item_imp.setText(imp_str)
                if imp_str.startswith("+"):
                    item_imp.setForeground(Qt.GlobalColor.darkGreen)
                elif imp_str.startswith("-"):
                    item_imp.setForeground(Qt.GlobalColor.red)

        # Timings
        timings = res.get("timings", {})
        total_time_ms = timings.get("end_to_end_ms", res.get("total_time_ms", 0.0))
        sim_per_s = timings.get("sim_per_s", res.get("sim_per_sec", 0.0))

        if total_time_ms > 0:
            if total_time_ms >= 1000:
                self.lbl_time.setText(f"Run Time: {total_time_ms / 1000.0:.2f} s")
            else:
                self.lbl_time.setText(f"Run Time: {total_time_ms:.1f} ms")

        if sim_per_s > 0:
            self.lbl_throughput.setText(f"Throughput: {sim_per_s:,.0f} sims/sec")

        hw = res.get("hardware", {})
        hw_text = f"Backend: {res.get('impl', 'N/A').upper()}"
        if isinstance(hw, dict) and hw.get("cpu_model"):
            hw_text += f" | CPU: {hw.get('cpu_model')}"
        if isinstance(hw, dict) and hw.get("gpu_name"):
            hw_text += f" | GPU: {hw.get('gpu_name')}"
        self.lbl_hardware.setText(hw_text)

    def export_csv(self, filepath: str) -> None:
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Metric", "FCFS", "Optimized", "Improvement_Pct"])
            for r in range(self.table.rowCount()):
                writer.writerow([
                    self.table.item(r, 0).text() if self.table.item(r, 0) else "",
                    self.table.item(r, 1).text() if self.table.item(r, 1) else "",
                    self.table.item(r, 2).text() if self.table.item(r, 2) else "",
                    self.table.item(r, 3).text() if self.table.item(r, 3) else "",
                ])

    def create_popout_widget(self) -> QWidget:
        # Create a copy of the summary widget
        clone = SummaryTab(self)
        if self.current_result:
            clone.set_data(self.current_result, self.current_state)
        return clone
