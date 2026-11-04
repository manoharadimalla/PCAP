"""
Main Application Window for RunwayMC GUI.

Assembles input dock panels, result visualization tabs, background QProcess execution,
undo/redo integration, and configuration persistence via QSettings.
"""

from __future__ import annotations

import datetime
import json
import os
import shutil
from typing import Any, Dict, Optional

from PySide6.QtCore import QSettings, Qt, Signal, Slot
from PySide6.QtGui import QAction, QColor, QFont, QIcon, QKeySequence, QUndoStack
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSplitter,
    QStatusBar,
    QTabWidget,
    QToolBar,
    QToolBox,
    QVBoxLayout,
    QWidget,
)

from gui.io_files import (
    find_runway_binary,
    load_project_state,
    read_instance_csv,
    read_result_json,
    save_project_state,
    write_instance_csv,
    write_params_cfg,
    write_separation_csv,
)
from gui.models import (
    ExecutionSettings,
    ProjectState,
    ScenarioSettings,
    SearchSettings,
    get_default_aircraft_list,
    get_default_separation_matrix,
)
from gui.panels.aircraft_panel import AircraftPanel
from gui.panels.execution_panel import ExecutionPanel
from gui.panels.scenario_panel import ScenarioPanel
from gui.panels.search_panel import SearchPanel
from gui.panels.separation_panel import SeparationPanel
from gui.results.compare_dialog import CompareDialog
from gui.results.convergence_tab import ConvergenceTab
from gui.results.distribution_tab import DistributionTab
from gui.results.replay_tab import ReplayTab
from gui.results.summary_tab import SummaryTab
from gui.results.timeline_tab import TimelineTab
from gui.runner import RunwayRunner, detect_hardware_info, validate_instance_via_cli


MANDATORY_DISCLAIMER = (
    "RunwayMC — GPU-Accelerated Robust Runway Scheduling under Uncertainty\n\n"
    "Mandatory Disclaimer:\n"
    "This is an academic simulation with simplified separation standards.\n"
    "It is not validated for, and must not be used in, real air-traffic operations."
)


class BenchmarkViewerDialog(QDialog):
    """Simple read-only viewer for results/*.csv and figures."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Benchmark Results Viewer — RunwayMC")
        self.resize(700, 500)
        layout = QVBoxLayout(self)

        lbl = QLabel("Existing Benchmark Results (results/):")
        lbl.setStyleSheet("font-weight: bold;")
        layout.addWidget(lbl)

        self.text_area = QPlainTextEdit()
        self.text_area.setReadOnly(True)
        self.text_area.setFont(QFont("Consolas", 10))
        layout.addWidget(self.text_area)

        btn_close = QPushButton("Close")
        btn_close.clicked.connect(self.accept)
        layout.addWidget(btn_close)

        self._populate()

    def _populate(self) -> None:
        proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        results_dir = os.path.join(proj_root, "results")
        lines = []
        if os.path.isdir(results_dir):
            for root, dirs, files in os.walk(results_dir):
                for f in files:
                    rel_path = os.path.relpath(os.path.join(root, f), proj_root)
                    size = os.path.getsize(os.path.join(root, f))
                    lines.append(f"{rel_path:<45} ({size:,} bytes)")
        if not lines:
            lines.append("No benchmark result files found in results/ directory.")
        self.text_area.setPlainText("\n".join(lines))


class PreferencesDialog(QDialog):
    """Preferences dialog to configure runway executable location."""

    def __init__(self, current_path: Optional[str], parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Preferences — Runway Binary")
        self.resize(550, 150)
        layout = QVBoxLayout(self)

        lbl = QLabel("Runway C++ Executable Path:")
        layout.addWidget(lbl)

        h_layout = QHBoxLayout()
        self.edit_path = QPlainTextEdit(current_path or "")
        self.edit_path.setMaximumHeight(35)
        btn_browse = QPushButton("Browse...")
        btn_browse.clicked.connect(self._browse)
        h_layout.addWidget(self.edit_path)
        h_layout.addWidget(btn_browse)
        layout.addLayout(h_layout)

        btn_box = QHBoxLayout()
        btn_detect = QPushButton("Auto-Detect")
        btn_detect.clicked.connect(self._detect)
        btn_ok = QPushButton("OK")
        btn_ok.clicked.connect(self.accept)
        btn_cancel = QPushButton("Cancel")
        btn_cancel.clicked.connect(self.reject)

        btn_box.addWidget(btn_detect)
        btn_box.addStretch()
        btn_box.addWidget(btn_ok)
        btn_box.addWidget(btn_cancel)
        layout.addLayout(btn_box)

    def _browse(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Locate Runway Executable", "", "Executables (*.exe *runway*);;All Files (*)")
        if path:
            self.edit_path.setPlainText(path)

    def _detect(self) -> None:
        detected = find_runway_binary()
        if detected:
            self.edit_path.setPlainText(detected)
        else:
            QMessageBox.information(self, "Detection", "Runway executable not automatically found in standard locations.")

    def get_path(self) -> str:
        return self.edit_path.toPlainText().strip()


class MainWindow(QMainWindow):
    """
    Main Application Window.
    """

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("RunwayMC — Robust Runway Scheduling")
        self.resize(1200, 800)

        self.settings = QSettings("RunwayMC", "RunwayMC")
        self.undo_stack = QUndoStack(self)
        self.runner = RunwayRunner(self)

        self.binary_path: Optional[str] = self._load_configured_binary()
        self.hardware_info: Dict[str, Any] = {}
        self.current_state = ProjectState()
        self.current_run_context: Dict[str, str] = {}
        self._is_dark_theme: bool = False

        self._setup_ui()
        self._setup_menus()
        self._setup_status_bar()
        self._connect_signals()
        self._refresh_hardware()
        self._restore_geometry()

    def _load_configured_binary(self) -> Optional[str]:
        cfg_path = self.settings.value("runway_binary", "")
        return find_runway_binary(cfg_path if cfg_path else None)

    def _setup_ui(self) -> None:
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(4, 4, 4, 4)
        main_layout.setSpacing(4)

        # Main horizontal splitter
        self.splitter = QSplitter(Qt.Orientation.Horizontal)

        # ── LEFT DOCK: Accordion ToolBox of Input Panels ──────────────────────
        self.left_dock = QWidget()
        left_layout = QVBoxLayout(self.left_dock)
        left_layout.setContentsMargins(0, 0, 0, 0)

        self.toolbox = QToolBox()
        self.toolbox.setStyleSheet("QToolBox::tab { font-weight: bold; }")

        # 1. Aircraft Panel
        self.panel_aircraft = AircraftPanel(
            undo_stack=self.undo_stack,
            get_binary_path_cb=lambda: self.binary_path,
            parent=self,
        )
        self.toolbox.addItem(self.panel_aircraft, "▾ Aircraft Instance")

        # 2. Scenario Settings Panel
        self.panel_scenario = ScenarioPanel(self)
        self.toolbox.addItem(self.panel_scenario, "▸ Scenario & Uncertainty")

        # 3. Separation Matrix Panel
        self.panel_separation = SeparationPanel(self)
        self.toolbox.addItem(self.panel_separation, "▸ Separation Standards")

        # 4. Search Parameters Panel
        self.panel_search = SearchPanel(self)
        self.toolbox.addItem(self.panel_search, "▸ Search Settings")

        # 5. Execution Parameters Panel
        self.panel_execution = ExecutionPanel(self)
        self.toolbox.addItem(self.panel_execution, "▸ Execution & Hardware")

        left_layout.addWidget(self.toolbox)
        self.splitter.addWidget(self.left_dock)

        # ── RIGHT AREA: Results Tab Widget ───────────────────────────────────
        self.tabs_results = QTabWidget()

        self.tab_summary = SummaryTab(self)
        self.tab_timeline = TimelineTab(self)
        self.tab_distribution = DistributionTab(self)

        self.tabs_results.addTab(self.tab_summary, "Summary")
        self.tabs_results.addTab(self.tab_timeline, "Timeline (Gantt)")
        self.tabs_results.addTab(self.tab_distribution, "Cost Distribution")

        self.splitter.addWidget(self.tabs_results)
        self.splitter.setSizes([450, 750])
        main_layout.addWidget(self.splitter, stretch=1)

        # ── BOTTOM BAR: Run / Cancel / Progress / Status / Log Toggle ────────
        bottom_bar = QFrame()
        bottom_bar.setFrameShape(QFrame.Shape.StyledPanel)
        b_layout = QHBoxLayout(bottom_bar)
        b_layout.setContentsMargins(6, 4, 6, 4)

        self.btn_run = QPushButton("▶ Run Solve")
        self.btn_run.setStyleSheet("font-weight: bold; background-color: #27ae60; color: white; padding: 6px 14px; border-radius: 4px;")
        self.btn_run.clicked.connect(self.run_solve)

        self.btn_cancel = QPushButton("■ Cancel")
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.setStyleSheet("font-weight: bold; background-color: #c0392b; color: white; padding: 6px 14px; border-radius: 4px;")
        self.btn_cancel.clicked.connect(self.cancel_solve)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFixedWidth(160)

        self.lbl_run_status = QLabel("Ready")
        self.lbl_run_status.setStyleSheet("font-weight: bold; color: #2c3e50;")

        self.btn_toggle_log = QPushButton("Log ▾")
        self.btn_toggle_log.setCheckable(True)
        self.btn_toggle_log.clicked.connect(self._toggle_log_drawer)

        b_layout.addWidget(self.btn_run)
        b_layout.addWidget(self.btn_cancel)
        b_layout.addWidget(self.progress_bar)
        b_layout.addWidget(self.lbl_run_status, stretch=1)
        b_layout.addWidget(self.btn_toggle_log)

        main_layout.addWidget(bottom_bar)

        # ── LOG CONSOLE DRAWER ────────────────────────────────────────────────
        self.log_drawer = QPlainTextEdit()
        self.log_drawer.setReadOnly(True)
        self.log_drawer.setMaximumHeight(160)
        self.log_drawer.setFont(QFont("Consolas", 9))
        self.log_drawer.setStyleSheet("background-color: #1e1e1e; color: #d4d4d4;")
        self.log_drawer.setVisible(False)
        main_layout.addWidget(self.log_drawer)

    def _setup_menus(self) -> None:
        mb = self.menuBar()

        # ── File Menu
        menu_file = mb.addMenu("&File")

        act_new = menu_file.addAction("&New Project")
        act_new.setShortcut(QKeySequence.StandardKey.New)
        act_new.triggered.connect(self.new_project)

        act_open = menu_file.addAction("&Open Project...")
        act_open.setShortcut(QKeySequence.StandardKey.Open)
        act_open.triggered.connect(self.open_project_dialog)

        act_save = menu_file.addAction("&Save Project")
        act_save.setShortcut(QKeySequence.StandardKey.Save)
        act_save.triggered.connect(self.save_project)

        act_save_as = menu_file.addAction("Save Project &As...")
        act_save_as.setShortcut(QKeySequence.StandardKey.SaveAs)
        act_save_as.triggered.connect(self.save_project_as)

        menu_file.addSeparator()

        act_imp_csv = menu_file.addAction("Import Instance CSV...")
        act_imp_csv.triggered.connect(self.panel_aircraft.import_csv)

        act_exp_csv = menu_file.addAction("Export Instance CSV...")
        act_exp_csv.triggered.connect(self.panel_aircraft.export_csv)

        act_exp_results = menu_file.addAction("Export Results...")
        act_exp_results.triggered.connect(self.export_all_results)

        menu_file.addSeparator()

        act_quit = menu_file.addAction("&Quit")
        act_quit.setShortcut(QKeySequence.StandardKey.Quit)
        act_quit.triggered.connect(self.close)

        # ── Edit Menu
        menu_edit = mb.addMenu("&Edit")
        menu_edit.addAction(self.panel_aircraft.act_undo)
        menu_edit.addAction(self.panel_aircraft.act_redo)
        menu_edit.addSeparator()
        menu_edit.addAction(self.panel_aircraft.act_add)
        menu_edit.addAction(self.panel_aircraft.act_del)
        menu_edit.addAction(self.panel_aircraft.act_dup)
        menu_edit.addSeparator()
        menu_edit.addAction(self.panel_aircraft.act_gen)

        act_restore = menu_edit.addAction("Restore Defaults")
        act_restore.triggered.connect(self.restore_defaults)

        menu_edit.addSeparator()
        act_pref = menu_edit.addAction("Preferences...")
        act_pref.triggered.connect(self.open_preferences)

        # ── Run Menu
        menu_run = mb.addMenu("&Run")

        act_run = menu_run.addAction("&Run")
        act_run.setShortcut(QKeySequence("F5"))
        act_run.triggered.connect(self.run_solve)

        act_cancel = menu_run.addAction("&Cancel")
        act_cancel.setShortcut(QKeySequence("Esc"))
        act_cancel.triggered.connect(self.cancel_solve)

        menu_run.addSeparator()

        act_compare = menu_run.addAction("Compare Implementations...")
        act_compare.triggered.connect(self.open_compare_dialog)

        # ── View Menu
        menu_view = mb.addMenu("&View")

        act_popout = menu_view.addAction("Pop out Current Chart")
        act_popout.triggered.connect(self._popout_current_tab)

        act_log = menu_view.addAction("Toggle Log Console")
        act_log.triggered.connect(lambda: self.btn_toggle_log.click())

        act_theme = menu_view.addAction("Toggle Light/Dark Theme")
        act_theme.triggered.connect(self.toggle_theme)

        menu_view.addSeparator()
        act_bench = menu_view.addAction("Benchmark Results Viewer...")
        act_bench.triggered.connect(lambda: BenchmarkViewerDialog(self).exec())

        # ── Help Menu
        menu_help = mb.addMenu("&Help")

        act_about = menu_help.addAction("&About RunwayMC")
        act_about.triggered.connect(self.show_about_dialog)

        act_hw = menu_help.addAction("&Hardware Info")
        act_hw.triggered.connect(self.show_hardware_info)

    def _setup_status_bar(self) -> None:
        self.status_bar = QStatusBar(self)
        self.setStatusBar(self.status_bar)

        self.sb_n = QLabel("n = 4")
        self.sb_impl = QLabel("Impl: Serial")
        self.sb_gpu = QLabel("GPU: None")
        self.sb_binary = QLabel("Binary: OK")

        self.status_bar.addWidget(self.sb_n)
        self.status_bar.addWidget(QLabel(" | "))
        self.status_bar.addWidget(self.sb_impl)
        self.status_bar.addWidget(QLabel(" | "))
        self.status_bar.addWidget(self.sb_gpu)
        self.status_bar.addPermanentWidget(self.sb_binary)

    def _connect_signals(self) -> None:
        # Aircraft count updates
        self.panel_aircraft.data_changed.connect(self._on_inputs_changed)
        self.panel_scenario.scenario_changed.connect(self._on_inputs_changed)
        self.panel_separation.matrix_changed.connect(self._on_inputs_changed)
        self.panel_search.search_changed.connect(self._on_inputs_changed)
        self.panel_execution.execution_changed.connect(self._on_inputs_changed)

        # Priority weights linking
        self.panel_scenario.priority_weights_changed.connect(self.panel_aircraft.set_priority_weights)

        # Preset selection
        self.panel_execution.preset_selected.connect(self._on_preset_applied)

        # Runner signals
        self.runner.progress.connect(self._on_runner_progress)
        self.runner.finished.connect(self._on_runner_finished)
        self.runner.error.connect(self._on_runner_error)
        self.runner.cancelled.connect(self._on_runner_cancelled)
        self.runner.log_line.connect(self._on_runner_log)

    def _refresh_hardware(self) -> None:
        self.hardware_info = detect_hardware_info(self.binary_path)
        self.panel_execution.update_hardware_capabilities(self.hardware_info)

        cuda_ok = self.hardware_info.get("cuda_available", False)
        gpu_name = self.hardware_info.get("gpu_name", "None")
        self.sb_gpu.setText(f"GPU: {gpu_name if cuda_ok else 'None'}")

        if self.binary_path and os.path.isfile(self.binary_path):
            self.sb_binary.setText(f"Runway: {os.path.basename(self.binary_path)}")
            self.sb_binary.setStyleSheet("color: green;")
        else:
            self.sb_binary.setText("Runway: Not Found")
            self.sb_binary.setStyleSheet("color: red;")

    def _on_inputs_changed(self) -> None:
        aircraft_list = self.panel_aircraft.get_aircraft_list()
        self.sb_n.setText(f"n = {len(aircraft_list)}")
        exec_cfg = self.panel_execution.get_settings()
        self.sb_impl.setText(f"Impl: {exec_cfg.impl.upper()}")

    def _on_preset_applied(self, preset_dict: Dict[str, Any]) -> None:
        search_cfg = self.panel_search.get_settings()
        search_cfg.chains = preset_dict.get("chains", search_cfg.chains)
        search_cfg.epochs = preset_dict.get("epochs", search_cfg.epochs)
        search_cfg.iters = preset_dict.get("iters", search_cfg.iters)
        search_cfg.scenarios = preset_dict.get("scenarios", search_cfg.scenarios)
        self.panel_search.set_settings(search_cfg)
        self._append_log(f"[Preset] Applied preset: {preset_dict}")

    def _toggle_log_drawer(self) -> None:
        is_visible = self.btn_toggle_log.isChecked()
        self.log_drawer.setVisible(is_visible)
        self.btn_toggle_log.setText("Log ▲" if is_visible else "Log ▾")

    def _append_log(self, text: str) -> None:
        self.log_drawer.appendPlainText(text)

    def _on_runner_log(self, text: str) -> None:
        self._append_log(text)

    def _popout_current_tab(self) -> None:
        curr_widget = self.tabs_results.currentWidget()
        if hasattr(curr_widget, "pop_out"):
            curr_widget.pop_out()

    def run_solve(self) -> None:
        """Main execution trigger on Run button click."""
        if self.runner.is_running():
            return

        if not self.binary_path or not os.path.isfile(self.binary_path):
            QMessageBox.critical(
                self,
                "Binary Not Found",
                "The runway CLI executable was not found.\n"
                "Please configure its path in Edit -> Preferences.",
            )
            self.open_preferences()
            return

        # Check weather probabilities
        if not self.panel_scenario.is_weather_probs_valid():
            QMessageBox.warning(
                self,
                "Invalid Weather Probabilities",
                "Weather probabilities must sum to 1.0 before launching the simulation.",
            )
            self.toolbox.setCurrentIndex(1)
            return

        # Prepare run directory: runs/<timestamp>/
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        run_dir = os.path.join(proj_root, "runs", timestamp)
        os.makedirs(run_dir, exist_ok=True)

        inst_path = os.path.join(run_dir, "instance.csv")
        sep_path = os.path.join(run_dir, "separation.csv")
        params_path = os.path.join(run_dir, "params.cfg")
        state_path = os.path.join(run_dir, "gui_state.json")
        result_json_path = os.path.join(run_dir, "result.json")

        aircraft_list = self.panel_aircraft.get_aircraft_list()
        scenario_settings = self.panel_scenario.get_settings()
        separation_matrix = self.panel_separation.get_matrix()
        search_settings = self.panel_search.get_settings()
        execution_settings = self.panel_execution.get_settings()

        write_instance_csv(inst_path, aircraft_list)
        write_separation_csv(sep_path, separation_matrix)
        write_params_cfg(params_path, scenario_settings, search_settings)

        self.current_state = ProjectState(
            aircraft_list=aircraft_list,
            scenario_settings=scenario_settings,
            search_settings=search_settings,
            execution_settings=execution_settings,
            separation_matrix=separation_matrix,
        )
        save_project_state(state_path, self.current_state)

        # Validate via CLI
        val_result = validate_instance_via_cli(self.binary_path, inst_path, params_path, sep_path)
        if not val_result.get("ok", True):
            errors = val_result.get("errors", [])
            self.panel_aircraft.highlight_errors(errors)
            self.toolbox.setCurrentIndex(0)
            err_msg_lines = [e.get("message", "Error") for e in errors[:5]]
            QMessageBox.critical(
                self,
                "Validation Failed",
                f"Cannot launch solve because instance validation failed with {len(errors)} error(s):\n\n"
                + "\n".join(f"• {m}" for m in err_msg_lines),
            )
            return

        # Clear error highlights
        self.panel_aircraft.highlight_errors([])

        # Store context
        self.current_run_context = {
            "instance_path": inst_path,
            "params_path": params_path,
            "separation_path": sep_path,
            "result_json_path": result_json_path,
            "seed": str(search_settings.seed),
        }

        # Build CLI arguments
        args = [
            "solve",
            "--instance", inst_path,
            "--impl", execution_settings.impl,
            "--chains", str(search_settings.chains),
            "--scenarios", str(search_settings.scenarios),
            "--iters", str(search_settings.iters),
            "--seed", str(search_settings.seed),
            "--s-final", str(search_settings.s_final),
            "--dump-costs", str(search_settings.s_final),
            "--separation", sep_path,
            "--m-penalty", str(scenario_settings.m_penalty),
            "--out", result_json_path,
            "--progress",
        ]

        if execution_settings.impl == "omp":
            args.extend(["--omp-mode", execution_settings.omp_mode])
            if execution_settings.threads > 0:
                args.extend(["--threads", str(execution_settings.threads)])

        # Update UI state
        self.btn_run.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.progress_bar.setRange(0, 0) # Indeterminate running indicator
        self.lbl_run_status.setText("Solving in parallel...")
        self.tabs_results.setCurrentWidget(self.tab_summary)

        self._append_log(f"\n[Run] Starting solve: {execution_settings.impl.upper()} (n={len(aircraft_list)})")
        self.runner.start(self.binary_path, args, result_json_path=result_json_path)

    def cancel_solve(self) -> None:
        if self.runner.is_running():
            self.runner.cancel()

    @Slot(int, int, float, float)
    def _on_runner_progress(self, epoch: int, total_epochs: int, best_j: float, elapsed_s: float) -> None:
        self.lbl_run_status.setText(f"Running (Best cost: {best_j:.2f})...")

    @Slot(object)
    def _on_runner_finished(self, result_dict: Dict[str, Any]) -> None:
        self.btn_run.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(100)
        self.lbl_run_status.setText("Run completed successfully.")

        self.current_state.last_result = result_dict
        # Update all 3 result tabs
        self.tab_summary.set_data(result_dict, self.current_state)
        self.tab_timeline.set_data(result_dict, self.current_state)
        self.tab_distribution.set_data(result_dict, self.current_state)
        self.tab_convergence.set_data(result_dict, self.current_state)

        # Switch to summary view
        self.tabs_results.setCurrentWidget(self.tab_summary)
        self._append_log("[Run] Optimization completed.")

    @Slot(str)
    def _on_runner_error(self, err_msg: str) -> None:
        self.btn_run.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.lbl_run_status.setText("Error during solve.")
        self._append_log(f"[Error] {err_msg}")
        QMessageBox.critical(self, "Solver Execution Error", f"An error occurred while running runway:\n\n{err_msg}")

    @Slot()
    def _on_runner_cancelled(self) -> None:
        self.btn_run.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.lbl_run_status.setText("Run cancelled by user.")
        self._append_log("[Run] Cancelled by user.")

    def new_project(self) -> None:
        self.panel_aircraft.set_aircraft_list(get_default_aircraft_list())
        self.panel_scenario.set_settings(ScenarioSettings())
        self.panel_separation.set_matrix(get_default_separation_matrix())
        self.panel_search.set_settings(SearchSettings())
        self.panel_execution.set_settings(ExecutionSettings())
        self.lbl_run_status.setText("New project loaded.")

    def open_project_dialog(self) -> None:
        filename, _ = QFileDialog.getOpenFileName(
            self, "Open Project State", "", "JSON State Files (*gui_state.json *.json);;All Files (*)"
        )
        if not filename:
            return
        try:
            state = load_project_state(filename)
            self.load_project(state)
            QMessageBox.information(self, "Project Loaded", f"Project successfully loaded from:\n{filename}")
        except Exception as e:
            QMessageBox.critical(self, "Open Error", f"Failed to load project: {e}")

    def load_project(self, state: ProjectState) -> None:
        self.current_state = state
        self.panel_aircraft.set_aircraft_list(state.aircraft_list)
        self.panel_scenario.set_settings(state.scenario_settings)
        self.panel_separation.set_matrix(state.separation_matrix)
        self.panel_search.set_settings(state.search_settings)
        self.panel_execution.set_settings(state.execution_settings)
        if state.last_result:
            self._on_runner_finished(state.last_result)

    def save_project(self) -> None:
        filename = self.settings.value("last_project_file", "")
        if not filename:
            self.save_project_as()
            return
        self._save_to_path(filename)

    def save_project_as(self) -> None:
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save Project State", "gui_state.json", "JSON State Files (*.json)"
        )
        if filename:
            self._save_to_path(filename)
            self.settings.setValue("last_project_file", filename)

    def _save_to_path(self, filepath: str) -> None:
        try:
            state = ProjectState(
                aircraft_list=self.panel_aircraft.get_aircraft_list(),
                scenario_settings=self.panel_scenario.get_settings(),
                search_settings=self.panel_search.get_settings(),
                execution_settings=self.panel_execution.get_settings(),
                separation_matrix=self.panel_separation.get_matrix(),
                last_result=self.current_state.last_result,
            )
            save_project_state(filepath, state)
            QMessageBox.information(self, "Project Saved", f"Project state saved to:\n{filepath}")
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Failed to save project state: {e}")

    def restore_defaults(self) -> None:
        reply = QMessageBox.question(
            self,
            "Restore Defaults",
            "Are you sure you want to restore all panels to factory default settings?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self.new_project()

    def export_all_results(self) -> None:
        if not self.current_state.last_result:
            QMessageBox.warning(self, "No Results", "No solve results available to export. Run an optimization first.")
            return

        folder = QFileDialog.getExistingDirectory(self, "Select Export Directory")
        if not folder:
            return

        try:
            # 1. Result JSON
            with open(os.path.join(folder, "result.json"), "w", encoding="utf-8") as f:
                json.dump(self.current_state.last_result, f, indent=2)

            # 2. Summary CSV
            self.tab_summary.export_csv(os.path.join(folder, "metrics_summary.csv"))

            # 3. Timeline CSV & Figure
            self.tab_timeline.export_csv(os.path.join(folder, "timeline_schedule.csv"))
            self.tab_timeline.save_figure(os.path.join(folder, "timeline.png"))

            # 4. Distribution CSV & Figure
            self.tab_distribution.export_csv(os.path.join(folder, "costs_distribution.csv"))
            self.tab_distribution.save_figure(os.path.join(folder, "distribution.png"))

            # 5. Convergence CSV & Figure
            self.tab_convergence.export_csv(os.path.join(folder, "convergence.csv"))
            self.tab_convergence.save_figure(os.path.join(folder, "convergence.png"))

            QMessageBox.information(self, "Export Complete", f"All results and figures exported successfully to:\n{folder}")
        except Exception as e:
            QMessageBox.critical(self, "Export Failed", f"Failed to export results: {e}")

    def open_preferences(self) -> None:
        dlg = PreferencesDialog(self.binary_path, self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            new_path = dlg.get_path()
            if new_path:
                self.binary_path = new_path
                self.settings.setValue("runway_binary", new_path)
            self._refresh_hardware()

    def open_compare_dialog(self) -> None:
        if not self.binary_path or not os.path.isfile(self.binary_path):
            QMessageBox.warning(self, "Binary Required", "Runway executable is required to run implementation comparison.")
            return

        state = ProjectState(
            aircraft_list=self.panel_aircraft.get_aircraft_list(),
            scenario_settings=self.panel_scenario.get_settings(),
            search_settings=self.panel_search.get_settings(),
            execution_settings=self.panel_execution.get_settings(),
            separation_matrix=self.panel_separation.get_matrix(),
        )
        cuda_ok = self.hardware_info.get("cuda_available", False)
        dlg = CompareDialog(state, self.binary_path, cuda_available=cuda_ok, parent=self)
        dlg.exec()

    def toggle_theme(self) -> None:
        self._is_dark_theme = not self._is_dark_theme
        if self._is_dark_theme:
            dark_qss = """
            QMainWindow, QDialog, QWidget { background-color: #2b2b2b; color: #f0f0f0; }
            QTableWidget, QPlainTextEdit { background-color: #1e1e1e; color: #f0f0f0; gridline-color: #3e3e3e; }
            QHeaderView::section { background-color: #333; color: white; }
            QGroupBox { border: 1px solid #555; margin-top: 6px; padding-top: 10px; font-weight: bold; }
            QToolBox::tab { background-color: #3a3a3a; color: white; }
            QPushButton { background-color: #3c3f41; border: 1px solid #555; padding: 4px 8px; color: white; }
            """
            self.setStyleSheet(dark_qss)
        else:
            self.setStyleSheet("")

    def show_about_dialog(self) -> None:
        QMessageBox.about(self, "About RunwayMC", MANDATORY_DISCLAIMER)

    def show_hardware_info(self) -> None:
        self._refresh_hardware()
        msg = (
            f"Runway Version: {self.hardware_info.get('version', 'Unknown')}\n\n"
            f"OpenMP Multi-Threading: {'Available' if self.hardware_info.get('omp_available') else 'Not compiled'}\n"
            f"Max OpenMP Threads: {self.hardware_info.get('omp_max_threads', 'N/A')}\n\n"
            f"CUDA Acceleration: {'Available' if self.hardware_info.get('cuda_available') else 'Not detected / CPU-only build'}\n"
            f"GPU Device: {self.hardware_info.get('gpu_name', 'None')}\n"
            f"GPU Streaming Multiprocessors: {self.hardware_info.get('gpu_sms', 0)}\n\n"
            f"Executable Path: {self.binary_path or 'Not configured'}"
        )
        QMessageBox.information(self, "Hardware & Capability Info", msg)

    def closeEvent(self, event) -> None:
        self._save_geometry()
        if self.runner.is_running():
            self.runner.cancel()
        event.accept()

    def _restore_geometry(self) -> None:
        geo = self.settings.value("geometry")
        if geo:
            self.restoreGeometry(geo)
        splitter_state = self.settings.value("splitter")
        if splitter_state:
            self.splitter.restoreState(splitter_state)

    def _save_geometry(self) -> None:
        self.settings.setValue("geometry", self.saveGeometry())
        self.settings.setValue("splitter", self.splitter.saveState())
