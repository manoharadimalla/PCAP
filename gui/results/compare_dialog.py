"""
Compare Implementations Dialog for RunwayMC GUI.

Executes serial, OpenMP, and CUDA solvers with identical problem instance
and parameters to provide a side-by-side performance and speedup comparison.
Displays the mandatory disclaimer:
"Single quick runs, not a benchmark. Use the benchmark harness for report numbers."
"""

from __future__ import annotations

import os
import tempfile
from typing import Any, Callable, Dict, List, Optional

import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QProgressBar,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from gui.io_files import read_result_json, write_instance_csv, write_params_cfg, write_separation_csv
from gui.models import ProjectState
from gui.runner import RunwayRunner


class CompareDialog(QDialog):
    """
    Comparison runner dialog testing Serial vs OpenMP vs CUDA.
    """

    MANDATORY_DISCLAIMER = (
        "Single quick runs, not a benchmark. Use the benchmark harness for report numbers."
    )

    def __init__(
        self,
        state: ProjectState,
        binary_path: str,
        cuda_available: bool = False,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.state = state
        self.binary_path = binary_path
        self.cuda_available = cuda_available

        self.setWindowTitle("Compare Implementations — RunwayMC")
        self.resize(850, 600)
        self.runner = RunwayRunner(self)
        self.runner.progress.connect(self._on_runner_progress)
        self.runner.finished.connect(self._on_runner_finished)
        self.runner.error.connect(self._on_runner_error)
        self.runner.cancelled.connect(self._on_runner_cancelled)

        self._pending_impls: List[str] = []
        self._current_impl_index: int = 0
        self._results: Dict[str, Dict[str, Any]] = {}
        self._temp_dir: Optional[str] = None

        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        # Disclaimer banner
        lbl_disclaimer = QLabel(f"⚠ {self.MANDATORY_DISCLAIMER}")
        lbl_disclaimer.setStyleSheet("background-color: #fff3cd; color: #856404; padding: 6px; font-weight: bold; border-radius: 4px;")
        layout.addWidget(lbl_disclaimer)

        # Selection checkboxes
        cfg_group = QGroupBox("Select Implementations to Compare")
        cfg_layout = QHBoxLayout(cfg_group)

        self.chk_serial = QCheckBox("Serial CPU")
        self.chk_serial.setChecked(True)
        self.chk_omp = QCheckBox("OpenMP Parallel")
        self.chk_omp.setChecked(True)
        self.chk_cuda = QCheckBox("CUDA GPU")
        self.chk_cuda.setChecked(self.cuda_available)
        self.chk_cuda.setEnabled(self.cuda_available)

        cfg_layout.addWidget(self.chk_serial)
        cfg_layout.addWidget(self.chk_omp)
        cfg_layout.addWidget(self.chk_cuda)
        layout.addWidget(cfg_group)

        # Comparison Table
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["Implementation", "Total Time (s)", "Throughput (sim/s)", "Speedup vs Serial", "Final J"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setAlternatingRowColors(True)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        layout.addWidget(self.table)

        # Bar chart
        self.fig = Figure(figsize=(6, 3), dpi=90)
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.ax = self.fig.add_subplot(111)
        layout.addWidget(self.canvas)

        # Progress bar & controls
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

        self.lbl_status = QLabel("Ready to run comparison.")
        self.lbl_status.setStyleSheet("color: #7f8c8d;")
        layout.addWidget(self.lbl_status)

        btn_layout = QHBoxLayout()
        self.btn_start = QPushButton("Start Comparison")
        self.btn_start.clicked.connect(self.start_comparison)

        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(self.cancel_comparison)

        self.btn_close = QPushButton("Close")
        self.btn_close.clicked.connect(self.accept)

        btn_layout.addWidget(self.btn_start)
        btn_layout.addWidget(self.btn_cancel)
        btn_layout.addStretch()
        btn_layout.addWidget(self.btn_close)
        layout.addLayout(btn_layout)

    def start_comparison(self) -> None:
        self._pending_impls = []
        if self.chk_serial.isChecked():
            self._pending_impls.append("serial")
        if self.chk_omp.isChecked():
            self._pending_impls.append("omp")
        if self.chk_cuda.isChecked() and self.cuda_available:
            self._pending_impls.append("cuda")

        if not self._pending_impls:
            self.lbl_status.setText("Please select at least one implementation.")
            return

        self._results.clear()
        self.table.setRowCount(0)
        self.ax.clear()
        self.canvas.draw()

        self.btn_start.setEnabled(False)
        self.btn_cancel.setEnabled(True)

        self._temp_dir = tempfile.mkdtemp(prefix="runway_cmp_")
        inst_path = os.path.join(self._temp_dir, "instance.csv")
        sep_path = os.path.join(self._temp_dir, "separation.csv")
        params_path = os.path.join(self._temp_dir, "params.cfg")

        write_instance_csv(inst_path, self.state.aircraft_list)
        write_separation_csv(sep_path, self.state.separation_matrix)
        write_params_cfg(params_path, self.state.scenario_settings, self.state.search_settings)

        self._current_impl_index = 0
        self._run_next_impl()

    def _run_next_impl(self) -> None:
        if self._current_impl_index >= len(self._pending_impls):
            self._on_all_finished()
            return

        impl = self._pending_impls[self._current_impl_index]
        self.lbl_status.setText(f"Running {impl.upper()} ({self._current_impl_index + 1}/{len(self._pending_impls)})...")
        self.progress_bar.setValue(int((self._current_impl_index / len(self._pending_impls)) * 100))

        inst_path = os.path.join(self._temp_dir, "instance.csv")
        params_path = os.path.join(self._temp_dir, "params.cfg")
        result_json_path = os.path.join(self._temp_dir, f"result_{impl}.json")

        search = self.state.search_settings
        exec_cfg = self.state.execution_settings

        args = [
            "solve",
            "--instance", inst_path,
            "--params", params_path,
            "--impl", impl,
            "--chains", str(search.chains),
            "--scenarios", str(search.scenarios),
            "--iters", str(search.iters),
            "--epochs", str(search.epochs),
            "--lambda", str(search.lambda_std),
            "--seed", str(search.seed),
            "--s-final", str(search.s_final),
            "--dump-costs", str(search.s_final),
            "--out", result_json_path,
            "--progress",
        ]
        if impl == "omp":
            args.extend(["--omp-mode", exec_cfg.omp_mode])
            if exec_cfg.threads > 0:
                args.extend(["--threads", str(exec_cfg.threads)])
        elif impl == "cuda":
            args.extend(["--kernel", exec_cfg.kernel, "--block", str(exec_cfg.block_size)])

        self.runner.start(self.binary_path, args, result_json_path=result_json_path)

    def _on_runner_progress(self, epoch: int, total_epochs: int, best_j: float, elapsed_s: float) -> None:
        base_pct = (self._current_impl_index / len(self._pending_impls)) * 100
        step_pct = (1.0 / len(self._pending_impls)) * (epoch / max(1, total_epochs)) * 100
        self.progress_bar.setValue(int(base_pct + step_pct))

    def _on_runner_finished(self, result_dict: Dict[str, Any]) -> None:
        impl = self._pending_impls[self._current_impl_index]
        self._results[impl] = result_dict
        self._current_impl_index += 1
        self._run_next_impl()

    def _on_runner_error(self, err: str) -> None:
        impl = self._pending_impls[self._current_impl_index]
        self.lbl_status.setText(f"Error running {impl.upper()}: {err}")
        self._results[impl] = {"error": err, "total_time_ms": 0, "sim_per_sec": 0, "J_out_of_sample": 0}
        self._current_impl_index += 1
        self._run_next_impl()

    def _on_runner_cancelled(self) -> None:
        self.lbl_status.setText("Comparison cancelled by user.")
        self.btn_start.setEnabled(True)
        self.btn_cancel.setEnabled(False)

    def cancel_comparison(self) -> None:
        if self.runner.is_running():
            self.runner.cancel()

    def _on_all_finished(self) -> None:
        self.progress_bar.setValue(100)
        self.lbl_status.setText("Comparison finished.")
        self.btn_start.setEnabled(True)
        self.btn_cancel.setEnabled(False)

        # Populate Table
        self.table.setRowCount(len(self._results))
        serial_time = None
        if "serial" in self._results:
            serial_time = self._results["serial"].get("timings", {}).get("end_to_end_ms", self._results["serial"].get("total_time_ms", 0.0))

        impl_names = []
        times_s = []

        for r, (impl, data) in enumerate(self._results.items()):
            timings = data.get("timings", {})
            t_ms = timings.get("end_to_end_ms", data.get("total_time_ms", 0.0))
            t_s = t_ms / 1000.0 if t_ms else 0.0
            throughput = timings.get("sim_per_s", data.get("sim_per_sec", 0.0))

            metrics = data.get("metrics", {})
            j_val = metrics.get("best", {}).get("J", data.get("J_out_of_sample", 0.0))

            speedup_str = "1.00x"
            if serial_time and serial_time > 0 and t_ms > 0:
                speedup_str = f"{serial_time / t_ms:.2f}x"

            self.table.setItem(r, 0, QTableWidgetItem(impl.upper()))
            self.table.setItem(r, 1, QTableWidgetItem(f"{t_s:.2f} s"))
            self.table.setItem(r, 2, QTableWidgetItem(f"{throughput:,.0f}"))
            self.table.setItem(r, 3, QTableWidgetItem(speedup_str))
            self.table.setItem(r, 4, QTableWidgetItem(f"{float(j_val):.2f}" if j_val else "—"))

            impl_names.append(impl.upper())
            times_s.append(t_s)

        # Update bar chart
        self.ax.clear()
        if impl_names:
            bars = self.ax.bar(impl_names, times_s, color=["#7f8c8d", "#2980b9", "#27ae60"][:len(impl_names)])
            self.ax.set_ylabel("Execution Time (s)")
            self.ax.set_title("Execution Time Comparison (Lower is Better)")
            self.ax.grid(True, axis="y", linestyle="--", alpha=0.5)
            for bar in bars:
                height = bar.get_height()
                self.ax.annotate(f"{height:.2f}s", xy=(bar.get_x() + bar.get_width() / 2, height),
                                 xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9)
        self.fig.tight_layout()
        self.canvas.draw()
