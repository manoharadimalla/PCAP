"""
Summary Tab for RunwayMC GUI (Simplified Model).

Displays side-by-side performance comparison between FCFS baseline
and Optimized schedule on test days:
- mean_cost
- p95_cost
- late_pct
- emergency_mean_delay
- run time and throughput
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
        ("Mean Cost (avg delay seconds)", "mean_cost"),
        ("95th Percentile Cost (P95)", "p95"),
        ("Late Aircraft Share (% past deadline)", "late_pct"),
        ("Emergency Flight Mean Delay (s)", "emergency_mean_delay"),
    ]

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__("Summary", parent)
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

        for r, (label, _) in enumerate(self.METRICS):
            item = QTableWidgetItem(label)
            item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
            self.table.setItem(r, 0, item)
            for c in range(1, 4):
                val_item = QTableWidgetItem("—")
                val_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.table.setItem(r, c, val_item)

        layout.addWidget(self.table)

        # Runtime & Throughput Box
        box_timing = QGroupBox("Execution & Throughput (Single Source of Truth)")
        t_layout = QHBoxLayout(box_timing)
        self.lbl_runtime = QLabel("Time: —")
        self.lbl_sims = QLabel("Throughput: —")
        self.lbl_hw = QLabel("Hardware: —")
        t_layout.addWidget(self.lbl_runtime)
        t_layout.addWidget(self.lbl_sims)
        t_layout.addWidget(self.lbl_hw)
        layout.addWidget(box_timing)

        self._main_layout.addWidget(content_widget)

    def set_data(self, result: Dict[str, Any], project_state: Any) -> None:
        metrics = result.get("metrics", {})
        fcfs = metrics.get("fcfs", {})
        best = metrics.get("best", {})

        for r, (_, key) in enumerate(self.METRICS):
            v_fcfs = fcfs.get(key)
            v_best = best.get(key)

            if v_fcfs is not None:
                if key == "late_pct":
                    self.table.item(r, 1).setText(f"{float(v_fcfs)*100.0:.1f}%")
                else:
                    self.table.item(r, 1).setText(f"{float(v_fcfs):,.2f}")

            if v_best is not None:
                if key == "late_pct":
                    self.table.item(r, 2).setText(f"{float(v_best)*100.0:.1f}%")
                else:
                    self.table.item(r, 2).setText(f"{float(v_best):,.2f}")

            if v_fcfs is not None and v_best is not None and float(v_fcfs) > 1e-6:
                diff = (float(v_fcfs) - float(v_best)) / float(v_fcfs) * 100.0
                sign = "+" if diff > 0 else ""
                self.table.item(r, 3).setText(f"{sign}{diff:.1f}%")

        timings = result.get("timings", {})
        k_ms = timings.get("kernel_ms", 0.0)
        sim_s = timings.get("sim_per_s", 0.0)
        self.lbl_runtime.setText(f"Compute Time: {k_ms:.1f} ms")
        self.lbl_sims.setText(f"Throughput: {sim_s:,.0f} sims/sec")

        impl = str(result.get("impl", "serial")).upper()
        hw = result.get("hardware", {})
        threads_used = hw.get("threads_used", 1)
        if impl == "OMP":
            hw_str = f"Hardware: OpenMP ({threads_used} threads)"
        elif impl == "CUDA":
            hw_str = "Hardware: NVIDIA GPU (CUDA)"
        else:
            hw_str = "Hardware: Serial CPU (1 thread)"
        self.lbl_hw.setText(hw_str)

    def _export_csv(self, file_path: str) -> None:
        with open(file_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Metric", "FCFS Baseline", "Optimized", "Improvement (%)"])
            for r in range(self.table.rowCount()):
                row_data = [self.table.item(r, c).text() for c in range(4)]
                writer.writerow(row_data)
