"""
Execution Panel for RunwayMC GUI.

Configures execution backends (Serial, OpenMP, CUDA), multi-threading options,
CUDA kernel selection, and provides workload presets (Quick Demo, Standard, Heavy).
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from gui.models import ExecutionSettings


class ExecutionPanel(QWidget):
    """
    Panel for choosing execution engine and parallel compute parameters.
    """
    execution_changed = Signal()
    preset_selected = Signal(dict)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._block_signals: bool = False
        self._cuda_available: bool = False
        self._setup_ui()
        self.set_settings(ExecutionSettings())

    def _setup_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)

        group = QGroupBox("Execution & Hardware Settings")
        form = QFormLayout(group)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)

        # 1. Implementation
        impl_layout = QHBoxLayout()
        self.rb_serial = QRadioButton("Serial")
        self.rb_omp = QRadioButton("OpenMP")
        self.rb_cuda = QRadioButton("CUDA (GPU)")
        self.rb_serial.setChecked(True)

        self.rb_serial.toggled.connect(self._on_impl_toggled)
        self.rb_omp.toggled.connect(self._on_impl_toggled)
        self.rb_cuda.toggled.connect(self._on_impl_toggled)

        impl_layout.addWidget(self.rb_serial)
        impl_layout.addWidget(self.rb_omp)
        impl_layout.addWidget(self.rb_cuda)
        form.addRow("Backend:", impl_layout)

        # 2. OpenMP mode
        self.combo_omp_mode = QComboBox()
        self.combo_omp_mode.addItems(["chains", "scenarios", "hybrid"])
        self.combo_omp_mode.currentTextChanged.connect(self._on_changed)
        form.addRow("OpenMP Mode:", self.combo_omp_mode)

        # 3. Threads
        self.spin_threads = QSpinBox()
        self.spin_threads.setRange(0, 256)
        self.spin_threads.setValue(0)
        self.spin_threads.setSpecialValueText("0 (Auto / All Cores)")
        self.spin_threads.valueChanged.connect(self._on_changed)
        form.addRow("Threads:", self.spin_threads)

        # 4. CUDA Kernel
        self.combo_kernel = QComboBox()
        self.combo_kernel.addItems(["k3", "k2", "k1"])
        self.combo_kernel.currentTextChanged.connect(self._on_changed)
        form.addRow("CUDA Kernel:", self.combo_kernel)

        # 5. CUDA Block Size
        self.spin_block = QSpinBox()
        self.spin_block.setRange(32, 1024)
        self.spin_block.setSingleStep(32)
        self.spin_block.setValue(256)
        self.spin_block.valueChanged.connect(self._on_changed)
        form.addRow("Block Size:", self.spin_block)

        # 6. Presets
        presets_layout = QHBoxLayout()
        self.btn_quick = QPushButton("Quick Demo")
        self.btn_quick.setToolTip("Quick run: C=4, E=2, I=100, S=256")
        self.btn_quick.clicked.connect(self._apply_quick_preset)

        self.btn_standard = QPushButton("Standard")
        self.btn_standard.setToolTip("Default: C=16, E=5, I=1000, S=1024")
        self.btn_standard.clicked.connect(self._apply_standard_preset)

        self.btn_heavy = QPushButton("Heavy")
        self.btn_heavy.setToolTip("High accuracy: C=64, E=20, I=5000, S=4096")
        self.btn_heavy.clicked.connect(self._apply_heavy_preset)

        presets_layout.addWidget(self.btn_quick)
        presets_layout.addWidget(self.btn_standard)
        presets_layout.addWidget(self.btn_heavy)
        form.addRow("Presets:", presets_layout)

        main_layout.addWidget(group)
        self._update_widget_states()

    def update_hardware_capabilities(self, info: Dict[str, Any]) -> None:
        """Update controls based on hardware capabilities query."""
        self._cuda_available = bool(info.get("cuda_available", False))
        max_threads = int(info.get("omp_max_threads", os.cpu_count() or 4))
        self.spin_threads.setMaximum(max(64, max_threads * 2))

        if not self._cuda_available:
            self.rb_cuda.setEnabled(False)
            self.rb_cuda.setToolTip("CUDA is not available on this system (no NVIDIA GPU or nvcc build detected).")
            if self.rb_cuda.isChecked():
                self.rb_omp.setChecked(True)
        else:
            gpu_name = info.get("gpu_name", "NVIDIA GPU")
            self.rb_cuda.setEnabled(True)
            self.rb_cuda.setToolTip(f"CUDA enabled: {gpu_name}")

        self._update_widget_states()

    def _on_impl_toggled(self) -> None:
        self._update_widget_states()
        self._on_changed()

    def _update_widget_states(self) -> None:
        is_omp = self.rb_omp.isChecked()
        is_cuda = self.rb_cuda.isChecked() and self._cuda_available

        self.combo_omp_mode.setEnabled(is_omp)
        self.spin_threads.setEnabled(is_omp)

        self.combo_kernel.setEnabled(is_cuda)
        self.spin_block.setEnabled(is_cuda)

    def _on_changed(self) -> None:
        if not self._block_signals:
            self.execution_changed.emit()

    def _apply_quick_preset(self) -> None:
        self.preset_selected.emit({"chains": 4, "epochs": 2, "iters": 100, "scenarios": 256})

    def _apply_standard_preset(self) -> None:
        self.preset_selected.emit({"chains": 16, "epochs": 5, "iters": 1000, "scenarios": 1024})

    def _apply_heavy_preset(self) -> None:
        self.preset_selected.emit({"chains": 64, "epochs": 20, "iters": 5000, "scenarios": 4096})

    def get_settings(self) -> ExecutionSettings:
        impl = "serial"
        if self.rb_omp.isChecked():
            impl = "omp"
        elif self.rb_cuda.isChecked() and self._cuda_available:
            impl = "cuda"

        return ExecutionSettings(
            impl=impl,
            omp_mode=self.combo_omp_mode.currentText(),
            threads=self.spin_threads.value(),
            kernel=self.combo_kernel.currentText(),
            block_size=self.spin_block.value(),
        )

    def set_settings(self, s: ExecutionSettings) -> None:
        self._block_signals = True
        if s.impl == "omp":
            self.rb_omp.setChecked(True)
        elif s.impl == "cuda" and self._cuda_available:
            self.rb_cuda.setChecked(True)
        else:
            self.rb_serial.setChecked(True)

        self.combo_omp_mode.setCurrentText(s.omp_mode)
        self.spin_threads.setValue(s.threads)
        self.combo_kernel.setCurrentText(s.kernel)
        self.spin_block.setValue(s.block_size)

        self._block_signals = False
        self._update_widget_states()
