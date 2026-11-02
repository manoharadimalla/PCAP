"""
Aircraft Table Panel for RunwayMC GUI.

Provides an editable table for managing aircraft instances, toolbar actions
(add, delete, duplicate, move, undo/redo, import/export, generate random),
and validation error highlighting.
"""

from __future__ import annotations

import copy
import os
import tempfile
from typing import Any, Callable, Dict, List, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QAction, QColor, QIcon, QKeySequence, QUndoCommand, QUndoStack
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from gui.io_files import clock_to_sec, read_instance_csv, sec_to_clock, write_instance_csv
from gui.models import AircraftRow, get_default_aircraft_list
from gui.runner import run_command_sync


class TableStateCommand(QUndoCommand):
    """Undo command that captures the entire list of aircraft before and after a change."""

    def __init__(
        self,
        panel: "AircraftPanel",
        old_state: List[AircraftRow],
        new_state: List[AircraftRow],
        description: str = "Edit Aircraft",
    ) -> None:
        super().__init__(description)
        self.panel = panel
        self.old_state = copy.deepcopy(old_state)
        self.new_state = copy.deepcopy(new_state)

    def undo(self) -> None:
        self.panel.set_aircraft_list(self.old_state, push_undo=False)

    def redo(self) -> None:
        self.panel.set_aircraft_list(self.new_state, push_undo=False)


class GenerateRandomDialog(QDialog):
    """Dialog for generating a random problem instance using runway gen."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Generate Random Instance")
        self.setModal(True)
        self.resize(320, 220)

        layout = QVBoxLayout(self)
        form = QFormLayout()

        self.spin_n = QSpinBox()
        self.spin_n.setRange(2, 500)
        self.spin_n.setValue(20)

        self.spin_rho = QDoubleSpinBox()
        self.spin_rho.setRange(0.5, 3.0)
        self.spin_rho.setSingleStep(0.1)
        self.spin_rho.setValue(1.2)

        self.spin_seed = QSpinBox()
        self.spin_seed.setRange(0, 9999999)
        self.spin_seed.setValue(42)

        self.spin_mix_arr = QDoubleSpinBox()
        self.spin_mix_arr.setRange(0.0, 1.0)
        self.spin_mix_arr.setSingleStep(0.05)
        self.spin_mix_arr.setValue(0.50)

        form.addRow("Aircraft count (n):", self.spin_n)
        form.addRow("Congestion ratio (rho):", self.spin_rho)
        form.addRow("Random Seed:", self.spin_seed)
        form.addRow("Arrival Share:", self.spin_mix_arr)

        layout.addLayout(form)

        btn_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        btn_box.accepted.connect(self.accept)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)

    def get_params(self) -> dict[str, Any]:
        return {
            "n": self.spin_n.value(),
            "rho": self.spin_rho.value(),
            "seed": self.spin_seed.value(),
            "mix_arr": self.spin_mix_arr.value(),
        }


class AircraftPanel(QWidget):
    """
    Panel containing the editable aircraft table and management controls.
    """
    data_changed = Signal()

    COL_ID = 0
    COL_TYPE = 1
    COL_WAKE = 2
    COL_READY = 3
    COL_TARGET = 4
    COL_LATEST = 5
    COL_PRIORITY = 6
    COL_WEIGHT = 7
    COLUMNS = ["ID", "Type", "Wake", "Earliest (Ready)", "Scheduled (Target)", "Deadline (Latest)", "Priority", "Weight"]

    def __init__(
        self,
        undo_stack: Optional[QUndoStack] = None,
        get_binary_path_cb: Optional[Callable[[], Optional[str]]] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.undo_stack = undo_stack or QUndoStack(self)
        self.get_binary_path = get_binary_path_cb or (lambda: None)
        self.time_display_clock: bool = False
        self._block_signals: bool = False
        self._priority_weights = {"Normal": 1.0, "High": 3.0, "Emergency": 20.0}

        self._setup_ui()
        self.set_aircraft_list(get_default_aircraft_list(), push_undo=False)

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        # Toolbar
        self.toolbar = QToolBar("Aircraft Actions")
        self.toolbar.setIconSize(self.toolbar.iconSize())

        self.act_add = self.toolbar.addAction("+ Add")
        self.act_add.setToolTip("Add new aircraft")
        self.act_add.triggered.connect(self.add_aircraft)

        self.act_del = self.toolbar.addAction("- Delete")
        self.act_del.setToolTip("Delete selected aircraft")
        self.act_del.triggered.connect(self.delete_selected)

        self.act_dup = self.toolbar.addAction("Duplicate")
        self.act_dup.setToolTip("Duplicate selected aircraft")
        self.act_dup.triggered.connect(self.duplicate_selected)

        self.toolbar.addSeparator()

        self.act_up = self.toolbar.addAction("▲ Up")
        self.act_up.setToolTip("Move selected aircraft up")
        self.act_up.triggered.connect(self.move_up)

        self.act_down = self.toolbar.addAction("▼ Down")
        self.act_down.setToolTip("Move selected aircraft down")
        self.act_down.triggered.connect(self.move_down)

        self.toolbar.addSeparator()

        self.act_undo = self.undo_stack.createUndoAction(self, "Undo")
        self.act_undo.setShortcut(QKeySequence.StandardKey.Undo)
        self.toolbar.addAction(self.act_undo)

        self.act_redo = self.undo_stack.createRedoAction(self, "Redo")
        self.act_redo.setShortcut(QKeySequence.StandardKey.Redo)
        self.toolbar.addAction(self.act_redo)

        self.toolbar.addSeparator()

        self.act_import = self.toolbar.addAction("Import CSV")
        self.act_import.triggered.connect(self.import_csv)

        self.act_export = self.toolbar.addAction("Export CSV")
        self.act_export.triggered.connect(self.export_csv)

        self.act_gen = self.toolbar.addAction("Generate Random...")
        self.act_gen.triggered.connect(self.open_generate_dialog)

        self.toolbar.addSeparator()

        self.act_toggle_time = self.toolbar.addAction("Time: Sec")
        self.act_toggle_time.setToolTip("Toggle time display between Seconds and HH:MM:SS")
        self.act_toggle_time.triggered.connect(self.toggle_time_display)

        layout.addWidget(self.toolbar)

        # Table Widget
        self.table = QTableWidget(0, len(self.COLUMNS))
        self.table.setHorizontalHeaderLabels(self.COLUMNS)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setAlternatingRowColors(True)

        self.table.itemChanged.connect(self._on_table_item_changed)
        layout.addWidget(self.table)

        # Summary footer label
        self.lbl_count = QLabel("Total: 0 aircraft")
        self.lbl_count.setStyleSheet("color: #7f8c8d; font-size: 11px;")
        layout.addWidget(self.lbl_count)

    def set_priority_weights(self, normal: float, high: float, emergency: float) -> None:
        """Update default weights mapping from Scenario panel."""
        self._priority_weights = {
            "Normal": normal,
            "High": high,
            "Emergency": emergency,
        }

    def toggle_time_display(self) -> None:
        self.time_display_clock = not self.time_display_clock
        self.act_toggle_time.setText("Time: Clock" if self.time_display_clock else "Time: Sec")
        # Refresh current table display values
        current_rows = self.get_aircraft_list()
        self.set_aircraft_list(current_rows, push_undo=False)

    def get_aircraft_list(self) -> List[AircraftRow]:
        """Extract aircraft list from table."""
        rows: List[AircraftRow] = []
        for r in range(self.table.rowCount()):
            # ID
            id_item = self.table.item(r, self.COL_ID)
            try:
                aid = int(id_item.text()) if id_item and id_item.text() else r
            except ValueError:
                aid = r

            # Type
            op_widget = self.table.cellWidget(r, self.COL_TYPE)
            op = op_widget.currentText() if isinstance(op_widget, QComboBox) else "ARR"

            # Wake
            wake_widget = self.table.cellWidget(r, self.COL_WAKE)
            wake = wake_widget.currentText() if isinstance(wake_widget, QComboBox) else "M"

            # Ready
            ready_item = self.table.item(r, self.COL_READY)
            try:
                ready = clock_to_sec(ready_item.text()) if ready_item else 0.0
            except ValueError:
                ready = 0.0

            # Target
            target_item = self.table.item(r, self.COL_TARGET)
            try:
                target = clock_to_sec(target_item.text()) if target_item else 0.0
            except ValueError:
                target = ready

            # Latest
            latest_item = self.table.item(r, self.COL_LATEST)
            try:
                latest = clock_to_sec(latest_item.text()) if latest_item else target + 600.0
            except ValueError:
                latest = target + 600.0

            # Priority
            prio_widget = self.table.cellWidget(r, self.COL_PRIORITY)
            prio = prio_widget.currentText() if isinstance(prio_widget, QComboBox) else "Normal"

            # Weight
            weight_item = self.table.item(r, self.COL_WEIGHT)
            try:
                weight = float(weight_item.text()) if weight_item and weight_item.text() else 1.0
            except ValueError:
                weight = 1.0

            rows.append(
                AircraftRow(
                    id=aid,
                    op=op,
                    wake=wake,
                    ready=ready,
                    target=target,
                    latest=latest,
                    priority=prio,
                    weight=weight,
                )
            )
        return rows

    def set_aircraft_list(self, rows: List[AircraftRow], push_undo: bool = True) -> None:
        """Populate the table with given aircraft rows."""
        if push_undo:
            old_rows = self.get_aircraft_list()
            self.undo_stack.push(TableStateCommand(self, old_rows, rows, "Set Aircraft List"))
            return

        self._block_signals = True
        self.table.setRowCount(0)
        self.table.setRowCount(len(rows))

        for r, ac in enumerate(rows):
            # ID
            id_item = QTableWidgetItem(str(ac.id))
            id_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(r, self.COL_ID, id_item)

            # Type combo
            combo_op = QComboBox()
            combo_op.addItems(["ARR", "DEP"])
            combo_op.setCurrentText(ac.op)
            combo_op.currentTextChanged.connect(self._on_cell_changed_manual)
            self.table.setCellWidget(r, self.COL_TYPE, combo_op)

            # Wake combo
            combo_wake = QComboBox()
            combo_wake.addItems(["H", "M", "L"])
            combo_wake.setCurrentText(ac.wake)
            combo_wake.currentTextChanged.connect(self._on_cell_changed_manual)
            self.table.setCellWidget(r, self.COL_WAKE, combo_wake)

            # Ready
            ready_str = sec_to_clock(ac.ready) if self.time_display_clock else f"{ac.ready:.2f}"
            self.table.setItem(r, self.COL_READY, QTableWidgetItem(ready_str))

            # Target
            target_str = sec_to_clock(ac.target) if self.time_display_clock else f"{ac.target:.2f}"
            self.table.setItem(r, self.COL_TARGET, QTableWidgetItem(target_str))

            # Latest
            latest_str = sec_to_clock(ac.latest) if self.time_display_clock else f"{ac.latest:.2f}"
            self.table.setItem(r, self.COL_LATEST, QTableWidgetItem(latest_str))

            # Priority combo
            combo_prio = QComboBox()
            combo_prio.addItems(["Normal", "High", "Emergency"])
            combo_prio.setCurrentText(ac.priority)
            combo_prio.currentTextChanged.connect(lambda txt, row_idx=r: self._on_priority_changed(row_idx, txt))
            self.table.setCellWidget(r, self.COL_PRIORITY, combo_prio)

            # Weight
            self.table.setItem(r, self.COL_WEIGHT, QTableWidgetItem(f"{ac.weight:.2f}"))

        self._block_signals = False
        self.lbl_count.setText(f"Total: {len(rows)} aircraft")
        self.data_changed.emit()

    def _on_priority_changed(self, row: int, priority_text: str) -> None:
        if self._block_signals:
            return
        weight_val = self._priority_weights.get(priority_text, 1.0)
        item = self.table.item(row, self.COL_WEIGHT)
        if item:
            item.setText(f"{weight_val:.2f}")
        self._on_cell_changed_manual()

    def _on_cell_changed_manual(self) -> None:
        if not self._block_signals:
            self.data_changed.emit()

    def _on_table_item_changed(self, item: QTableWidgetItem) -> None:
        if not self._block_signals:
            self.data_changed.emit()

    def add_aircraft(self) -> None:
        current_rows = self.get_aircraft_list()
        new_id = max([ac.id for ac in current_rows], default=-1) + 1
        new_ready = max([ac.ready for ac in current_rows], default=0.0) + 60.0
        new_ac = AircraftRow(
            id=new_id,
            op="ARR",
            wake="M",
            ready=new_ready,
            target=new_ready + 60.0,
            latest=new_ready + 660.0,
            priority="Normal",
            weight=self._priority_weights.get("Normal", 1.0),
        )
        new_rows = current_rows + [new_ac]
        self.undo_stack.push(TableStateCommand(self, current_rows, new_rows, "Add Aircraft"))

    def delete_selected(self) -> None:
        selected_rows = sorted(set(idx.row() for idx in self.table.selectedIndexes()), reverse=True)
        if not selected_rows:
            return
        current_rows = self.get_aircraft_list()
        new_rows = [ac for idx, ac in enumerate(current_rows) if idx not in selected_rows]
        self.undo_stack.push(TableStateCommand(self, current_rows, new_rows, "Delete Aircraft"))

    def duplicate_selected(self) -> None:
        selected_rows = sorted(set(idx.row() for idx in self.table.selectedIndexes()))
        if not selected_rows:
            return
        current_rows = self.get_aircraft_list()
        next_id = max([ac.id for ac in current_rows], default=-1) + 1

        dups = []
        for idx in selected_rows:
            if idx < len(current_rows):
                ac = copy.deepcopy(current_rows[idx])
                ac.id = next_id
                next_id += 1
                dups.append(ac)

        new_rows = current_rows + dups
        self.undo_stack.push(TableStateCommand(self, current_rows, new_rows, "Duplicate Aircraft"))

    def move_up(self) -> None:
        selected_rows = sorted(set(idx.row() for idx in self.table.selectedIndexes()))
        if not selected_rows or selected_rows[0] == 0:
            return
        current_rows = self.get_aircraft_list()
        new_rows = copy.deepcopy(current_rows)
        for r in selected_rows:
            new_rows[r - 1], new_rows[r] = new_rows[r], new_rows[r - 1]
        self.undo_stack.push(TableStateCommand(self, current_rows, new_rows, "Move Up"))
        self.table.selectRow(selected_rows[0] - 1)

    def move_down(self) -> None:
        selected_rows = sorted(set(idx.row() for idx in self.table.selectedIndexes()), reverse=True)
        if not selected_rows or selected_rows[0] >= self.table.rowCount() - 1:
            return
        current_rows = self.get_aircraft_list()
        new_rows = copy.deepcopy(current_rows)
        for r in selected_rows:
            new_rows[r + 1], new_rows[r] = new_rows[r], new_rows[r + 1]
        self.undo_stack.push(TableStateCommand(self, current_rows, new_rows, "Move Down"))
        self.table.selectRow(selected_rows[0] + 1)

    def import_csv(self) -> None:
        filename, _ = QFileDialog.getOpenFileName(self, "Import Instance CSV", "", "CSV Files (*.csv);;All Files (*)")
        if not filename:
            return
        try:
            loaded_rows = read_instance_csv(filename)
            if loaded_rows:
                current_rows = self.get_aircraft_list()
                self.undo_stack.push(TableStateCommand(self, current_rows, loaded_rows, f"Import CSV {os.path.basename(filename)}"))
            else:
                QMessageBox.warning(self, "Import Error", "No aircraft records found in the specified file.")
        except Exception as e:
            QMessageBox.critical(self, "Import Failed", f"Could not import CSV: {e}")

    def export_csv(self) -> None:
        filename, _ = QFileDialog.getSaveFileName(self, "Export Instance CSV", "instance.csv", "CSV Files (*.csv);;All Files (*)")
        if not filename:
            return
        try:
            rows = self.get_aircraft_list()
            write_instance_csv(filename, rows)
            QMessageBox.information(self, "Export Successful", f"Saved {len(rows)} aircraft to:\n{filename}")
        except Exception as e:
            QMessageBox.critical(self, "Export Failed", f"Could not write CSV: {e}")

    def open_generate_dialog(self) -> None:
        dlg = GenerateRandomDialog(self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        params = dlg.get_params()
        binary = self.get_binary_path()

        if not binary or not os.path.isfile(binary):
            QMessageBox.warning(
                self,
                "Binary Not Found",
                "The runway executable is required to generate random instances according to spec.\n"
                "Please configure the path in Edit -> Preferences.",
            )
            return

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            args = [
                "gen",
                "--n", str(params["n"]),
                "--rho", str(params["rho"]),
                "--seed", str(params["seed"]),
                "--mix-arr", str(params["mix_arr"]),
                "--out", tmp_path,
            ]
            code, stdout, stderr = run_command_sync(binary, args, timeout_ms=8000)
            if code == 0 and os.path.isfile(tmp_path):
                new_rows = read_instance_csv(tmp_path)
                current_rows = self.get_aircraft_list()
                self.undo_stack.push(TableStateCommand(self, current_rows, new_rows, f"Generate Random (n={params['n']})"))
            else:
                QMessageBox.critical(self, "Generation Failed", f"runway gen exited with error:\n{stderr or stdout}")
        except Exception as e:
            QMessageBox.critical(self, "Generation Error", f"Failed to run generator: {e}")
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def highlight_errors(self, errors: List[Dict[str, Any]]) -> None:
        """
        Highlight invalid cells red with tooltip messages from validation.
        """
        # Clear all previous red highlights
        for r in range(self.table.rowCount()):
            for c in range(self.table.columnCount()):
                item = self.table.item(r, c)
                if item:
                    item.setBackground(QColor(Qt.GlobalColor.white))
                    item.setToolTip("")

        col_map = {
            "id": self.COL_ID,
            "op": self.COL_TYPE,
            "wake": self.COL_WAKE,
            "ready": self.COL_READY,
            "target": self.COL_TARGET,
            "latest": self.COL_LATEST,
            "priority": self.COL_PRIORITY,
            "weight": self.COL_WEIGHT,
        }

        error_color = QColor(255, 204, 204)  # Light red

        for err in errors:
            row_idx = err.get("row", -1)
            col_name = err.get("column", "").lower()
            msg = err.get("message", "Validation error")

            if 0 <= row_idx < self.table.rowCount():
                if col_name in col_map:
                    target_col = col_map[col_name]
                    item = self.table.item(row_idx, target_col)
                    if item:
                        item.setBackground(error_color)
                        item.setToolTip(msg)
                else:
                    # Highlight whole row if generic
                    for c in range(self.table.columnCount()):
                        item = self.table.item(row_idx, c)
                        if item:
                            item.setBackground(error_color)
                            item.setToolTip(msg)
