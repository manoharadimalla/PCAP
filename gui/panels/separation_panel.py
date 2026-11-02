"""
Separation Matrix Panel for RunwayMC GUI.

Provides a 6x6 editable grid corresponding to the aircraft wake vortex
and operation separation requirements, with import/export and
triangle-inequality verification.
"""

from __future__ import annotations

import os
from typing import List, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from gui.io_files import read_separation_csv, write_separation_csv
from gui.models import SEPARATION_CLASSES, get_default_separation_matrix


class SeparationPanel(QWidget):
    """
    Panel for displaying and editing the 6x6 separation matrix.
    """
    matrix_changed = Signal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._block_signals: bool = False
        self._setup_ui()
        self.set_matrix(get_default_separation_matrix())

    def _setup_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)

        group = QGroupBox("Wake Separation Standards (seconds)")
        layout = QVBoxLayout(group)

        # Header description
        lbl_info = QLabel("Rows: Leading Aircraft | Columns: Following Aircraft")
        lbl_info.setStyleSheet("color: #7f8c8d; font-size: 11px;")
        layout.addWidget(lbl_info)

        # 6x6 Table
        self.table = QTableWidget(6, 6)
        self.table.setHorizontalHeaderLabels(SEPARATION_CLASSES)
        self.table.setVerticalHeaderLabels(SEPARATION_CLASSES)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)

        self.table.itemChanged.connect(self._on_item_changed)
        layout.addWidget(self.table)

        # Status label for Triangle Inequality
        self.lbl_triangle = QLabel("Triangle inequality: OK")
        self.lbl_triangle.setStyleSheet("color: green; font-weight: bold; font-size: 11px;")
        layout.addWidget(self.lbl_triangle)

        # Action Buttons
        btn_layout = QHBoxLayout()
        self.btn_reset = QPushButton("Reset to Defaults")
        self.btn_reset.clicked.connect(self.reset_defaults)

        self.btn_import = QPushButton("Import CSV...")
        self.btn_import.clicked.connect(self.import_csv)

        self.btn_export = QPushButton("Export CSV...")
        self.btn_export.clicked.connect(self.export_csv)

        btn_layout.addWidget(self.btn_reset)
        btn_layout.addWidget(self.btn_import)
        btn_layout.addWidget(self.btn_export)
        layout.addLayout(btn_layout)

        main_layout.addWidget(group)

    def _on_item_changed(self, item: QTableWidgetItem) -> None:
        if self._block_signals:
            return
        self._verify_triangle_inequality()
        self.matrix_changed.emit()

    def _verify_triangle_inequality(self) -> bool:
        """
        Check whether sep[i][k] <= sep[i][j] + sep[j][k] for all i,j,k.
        Display status in label.
        """
        mat = self.get_matrix()
        ok = True
        for i in range(6):
            for j in range(6):
                for k in range(6):
                    if mat[i][k] > mat[i][j] + mat[j][k] + 1e-4:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                break

        if ok:
            self.lbl_triangle.setText("Triangle inequality: OK (fast decoder active)")
            self.lbl_triangle.setStyleSheet("color: #27ae60; font-weight: bold;")
        else:
            self.lbl_triangle.setText("Triangle inequality: VIOLATED (exact decoder will be used)")
            self.lbl_triangle.setStyleSheet("color: #d35400; font-weight: bold;")

        return ok

    def get_matrix(self) -> List[List[float]]:
        mat: List[List[float]] = []
        for r in range(6):
            row: List[float] = []
            for c in range(6):
                item = self.table.item(r, c)
                try:
                    val = float(item.text()) if item and item.text() else 60.0
                except ValueError:
                    val = 60.0
                row.append(val)
            mat.append(row)
        return mat

    def set_matrix(self, matrix: List[List[float]]) -> None:
        self._block_signals = True
        for r in range(min(6, len(matrix))):
            for c in range(min(6, len(matrix[r]))):
                val = matrix[r][c]
                item = QTableWidgetItem(f"{val:.1f}")
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.table.setItem(r, c, item)
        self._block_signals = False
        self._verify_triangle_inequality()
        self.matrix_changed.emit()

    def reset_defaults(self) -> None:
        self.set_matrix(get_default_separation_matrix())

    def import_csv(self) -> None:
        filename, _ = QFileDialog.getOpenFileName(self, "Import Separation CSV", "", "CSV Files (*.csv);;All Files (*)")
        if not filename:
            return
        try:
            mat = read_separation_csv(filename)
            self.set_matrix(mat)
            QMessageBox.information(self, "Import Successful", f"Loaded separation matrix from:\n{filename}")
        except Exception as e:
            QMessageBox.critical(self, "Import Failed", f"Could not read separation file: {e}")

    def export_csv(self) -> None:
        filename, _ = QFileDialog.getSaveFileName(self, "Export Separation CSV", "separation.csv", "CSV Files (*.csv);;All Files (*)")
        if not filename:
            return
        try:
            mat = self.get_matrix()
            write_separation_csv(filename, mat)
            QMessageBox.information(self, "Export Successful", f"Saved separation matrix to:\n{filename}")
        except Exception as e:
            QMessageBox.critical(self, "Export Failed", f"Could not save separation file: {e}")
