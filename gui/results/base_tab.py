"""
Base class for result tabs in RunwayMC GUI.

Provides standard actions across all result viewers:
- "Pop out ⧉" button to open the visualization in a dedicated resizable window.
- "Save PNG" and "Save PDF" buttons for report graphics.
- "Export CSV" button to extract underlying numerical data.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QMessageBox,
    QPushButton,
    QToolBar,
    QVBoxLayout,
    QWidget,
)


class PopoutWindow(QDialog):
    """Container window for popped-out chart views."""

    def __init__(self, title: str, content_widget: QWidget, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"RunwayMC — {title} (Pop-out)")
        self.resize(900, 600)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.addWidget(content_widget)


class BaseResultTab(QWidget):
    """
    Common base widget for result tabs with export and pop-out features.
    """

    def __init__(self, title: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.tab_title = title
        self.current_result: Optional[Dict[str, Any]] = None
        self.current_state: Optional[Any] = None

        self._main_layout = QVBoxLayout(self)
        self._main_layout.setContentsMargins(6, 6, 6, 6)
        self._main_layout.setSpacing(4)

        # Toolbar
        self.toolbar = QToolBar(f"{title} Toolbar")
        self.toolbar.setIconSize(self.toolbar.iconSize())

        self.btn_popout = QPushButton("Pop out ⧉")
        self.btn_popout.setToolTip("Open this chart in a separate pop-up window")
        self.btn_popout.clicked.connect(self.pop_out)
        self.toolbar.addWidget(self.btn_popout)

        self.toolbar.addSeparator()

        self.btn_save_png = QPushButton("Save PNG")
        self.btn_save_png.clicked.connect(self._save_png_dialog)
        self.toolbar.addWidget(self.btn_save_png)

        self.btn_save_pdf = QPushButton("Save PDF")
        self.btn_save_pdf.clicked.connect(self._save_pdf_dialog)
        self.toolbar.addWidget(self.btn_save_pdf)

        self.btn_export_csv = QPushButton("Export CSV")
        self.btn_export_csv.clicked.connect(self._export_csv_dialog)
        self.toolbar.addWidget(self.btn_export_csv)

        self._main_layout.addWidget(self.toolbar)

    def set_data(self, result: Dict[str, Any], state: Any) -> None:
        """Update tab with solver result and project state."""
        self.current_result = result
        self.current_state = state
        self.update_view()

    def update_view(self) -> None:
        """To be implemented by subclasses to update rendering."""
        pass

    def create_popout_widget(self) -> QWidget:
        """Create a widget instance to show inside the popped-out window."""
        raise NotImplementedError("Subclasses must implement create_popout_widget")

    def pop_out(self) -> None:
        try:
            widget = self.create_popout_widget()
            pop = PopoutWindow(self.tab_title, widget, self)
            pop.show()
        except Exception as e:
            QMessageBox.critical(self, "Pop out Error", f"Failed to pop out chart: {e}")

    def _save_png_dialog(self) -> None:
        filename, _ = QFileDialog.getSaveFileName(
            self, f"Save {self.tab_title} PNG", f"{self.tab_title.lower().replace(' ', '_')}.png", "PNG Images (*.png)"
        )
        if filename:
            try:
                self.save_figure(filename)
                QMessageBox.information(self, "Saved", f"Chart saved to:\n{filename}")
            except Exception as e:
                QMessageBox.critical(self, "Save Error", f"Failed to save image: {e}")

    def _save_pdf_dialog(self) -> None:
        filename, _ = QFileDialog.getSaveFileName(
            self, f"Save {self.tab_title} PDF", f"{self.tab_title.lower().replace(' ', '_')}.pdf", "PDF Documents (*.pdf)"
        )
        if filename:
            try:
                self.save_figure(filename)
                QMessageBox.information(self, "Saved", f"Document saved to:\n{filename}")
            except Exception as e:
                QMessageBox.critical(self, "Save Error", f"Failed to save PDF: {e}")

    def _export_csv_dialog(self) -> None:
        filename, _ = QFileDialog.getSaveFileName(
            self, f"Export {self.tab_title} CSV", f"{self.tab_title.lower().replace(' ', '_')}.csv", "CSV Files (*.csv)"
        )
        if filename:
            try:
                self.export_csv(filename)
                QMessageBox.information(self, "Exported", f"Data exported to:\n{filename}")
            except Exception as e:
                QMessageBox.critical(self, "Export Error", f"Failed to export CSV: {e}")

    def save_figure(self, filepath: str) -> None:
        """Override in subclasses that host matplotlib figures."""
        raise NotImplementedError("This tab does not support image saving.")

    def export_csv(self, filepath: str) -> None:
        """Override in subclasses to write out data."""
        raise NotImplementedError("This tab does not support CSV export.")
