"""
Search Settings Panel for RunwayMC GUI.

Configures Simulated Annealing Monte Carlo search hyperparameters:
chains, epochs, iterations per epoch, in-sample & out-of-sample scenarios,
robust objective lambda weight, RNG seed, neighborhood move mix, and temperature cooling.
"""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QRadioButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from gui.models import SearchSettings


class SearchPanel(QWidget):
    """
    Panel for configuring Simulated Annealing hyperparameters.
    """
    search_changed = Signal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._block_signals: bool = False
        self._setup_ui()
        self.set_settings(SearchSettings())

    def _setup_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)

        group = QGroupBox("Search & Optimization Parameters")
        form = QFormLayout(group)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)

        # 1. Chains
        self.spin_chains = QSpinBox()
        self.spin_chains.setRange(1, 1024)
        self.spin_chains.setValue(16)
        self.spin_chains.valueChanged.connect(self._on_changed)
        form.addRow("Chains (C):", self.spin_chains)

        # 2. Epochs
        self.spin_epochs = QSpinBox()
        self.spin_epochs.setRange(1, 200)
        self.spin_epochs.setValue(5)
        self.spin_epochs.valueChanged.connect(self._on_changed)
        form.addRow("Epochs (E):", self.spin_epochs)

        # 3. Iterations
        self.spin_iters = QSpinBox()
        self.spin_iters.setRange(10, 500000)
        self.spin_iters.setSingleStep(100)
        self.spin_iters.setValue(1000)
        self.spin_iters.valueChanged.connect(self._on_changed)
        form.addRow("Iterations / Epoch (I):", self.spin_iters)

        # 4. In-sample Scenarios
        self.spin_scenarios = QSpinBox()
        self.spin_scenarios.setRange(1, 65536)
        self.spin_scenarios.setSingleStep(64)
        self.spin_scenarios.setValue(1024)
        self.spin_scenarios.valueChanged.connect(self._on_changed)
        form.addRow("In-Sample Scenarios (S):", self.spin_scenarios)

        # 5. Out-of-sample Scenarios
        self.spin_s_final = QSpinBox()
        self.spin_s_final.setRange(100, 100000)
        self.spin_s_final.setSingleStep(1000)
        self.spin_s_final.setValue(20000)
        self.spin_s_final.valueChanged.connect(self._on_changed)
        form.addRow("Out-of-Sample (S_final):", self.spin_s_final)

        # 6. Lambda std weight
        self.spin_lambda = QDoubleSpinBox()
        self.spin_lambda.setRange(0.0, 10.0)
        self.spin_lambda.setSingleStep(0.1)
        self.spin_lambda.setValue(1.0)
        self.spin_lambda.valueChanged.connect(self._on_changed)
        form.addRow("Std Dev Weight (λ):", self.spin_lambda)

        # 7. Seed
        self.spin_seed = QSpinBox()
        self.spin_seed.setRange(0, 2147483647)
        self.spin_seed.setValue(42)
        self.spin_seed.valueChanged.connect(self._on_changed)
        form.addRow("Random Seed:", self.spin_seed)

        # 8. Move mix (Swap %, Insert %, Adjacent %)
        mix_layout = QVBoxLayout()

        row_swap = QHBoxLayout()
        self.spin_swap = QSpinBox()
        self.spin_swap.setRange(0, 100)
        self.spin_swap.setValue(40)
        self.spin_swap.setSuffix("%")
        self.spin_swap.valueChanged.connect(self._on_mix_changed)
        row_swap.addWidget(QLabel("Swap:"))
        row_swap.addWidget(self.spin_swap)

        row_insert = QHBoxLayout()
        self.spin_insert = QSpinBox()
        self.spin_insert.setRange(0, 100)
        self.spin_insert.setValue(40)
        self.spin_insert.setSuffix("%")
        self.spin_insert.valueChanged.connect(self._on_mix_changed)
        row_insert.addWidget(QLabel("Insert:"))
        row_insert.addWidget(self.spin_insert)

        row_adj = QHBoxLayout()
        self.spin_adj = QSpinBox()
        self.spin_adj.setRange(0, 100)
        self.spin_adj.setValue(20)
        self.spin_adj.setSuffix("%")
        self.spin_adj.valueChanged.connect(self._on_mix_changed)
        row_adj.addWidget(QLabel("Adjacent:"))
        row_adj.addWidget(self.spin_adj)

        mix_layout.addLayout(row_swap)
        mix_layout.addLayout(row_insert)
        mix_layout.addLayout(row_adj)

        self.lbl_mix_sum = QLabel("Move Mix Sum: 100%")
        self.lbl_mix_sum.setStyleSheet("color: #27ae60; font-size: 11px;")
        mix_layout.addWidget(self.lbl_mix_sum)
        form.addRow("Move Mix:", mix_layout)

        # 9. Temperature
        temp_layout = QVBoxLayout()
        self.rb_temp_auto = QRadioButton("Auto-calibrate T0 (50% accept)")
        self.rb_temp_manual = QRadioButton("Manual Temperature")
        self.rb_temp_auto.setChecked(True)

        self.temp_group = QButtonGroup(self)
        self.temp_group.addButton(self.rb_temp_auto)
        self.temp_group.addButton(self.rb_temp_manual)
        self.temp_group.buttonToggled.connect(self._on_temp_mode_toggled)

        temp_layout.addWidget(self.rb_temp_auto)
        temp_layout.addWidget(self.rb_temp_manual)

        manual_layout = QHBoxLayout()
        self.spin_t0 = QDoubleSpinBox()
        self.spin_t0.setRange(0.01, 10000.0)
        self.spin_t0.setValue(100.0)
        self.spin_t0.setEnabled(False)
        self.spin_t0.valueChanged.connect(self._on_changed)

        self.spin_tend = QDoubleSpinBox()
        self.spin_tend.setRange(0.0001, 1000.0)
        self.spin_tend.setDecimals(4)
        self.spin_tend.setValue(0.1)
        self.spin_tend.setEnabled(False)
        self.spin_tend.valueChanged.connect(self._on_changed)

        manual_layout.addWidget(QLabel("T0:"))
        manual_layout.addWidget(self.spin_t0)
        manual_layout.addWidget(QLabel("T_end:"))
        manual_layout.addWidget(self.spin_tend)
        temp_layout.addLayout(manual_layout)

        form.addRow("Cooling Schedule:", temp_layout)

        main_layout.addWidget(group)

    def _on_mix_changed(self) -> None:
        total = self.spin_swap.value() + self.spin_insert.value() + self.spin_adj.value()
        if total != 100:
            self.lbl_mix_sum.setText(f"⚠ Move Mix Sum: {total}% (should be 100%)")
            self.lbl_mix_sum.setStyleSheet("color: #d35400; font-size: 11px; font-weight: bold;")
        else:
            self.lbl_mix_sum.setText("Move Mix Sum: 100% OK")
            self.lbl_mix_sum.setStyleSheet("color: #27ae60; font-size: 11px;")
        self._on_changed()

    def _on_temp_mode_toggled(self) -> None:
        is_manual = self.rb_temp_manual.isChecked()
        self.spin_t0.setEnabled(is_manual)
        self.spin_tend.setEnabled(is_manual)
        self._on_changed()

    def _on_changed(self) -> None:
        if not self._block_signals:
            self.search_changed.emit()

    def get_settings(self) -> SearchSettings:
        return SearchSettings(
            chains=self.spin_chains.value(),
            epochs=self.spin_epochs.value(),
            iters=self.spin_iters.value(),
            scenarios=self.spin_scenarios.value(),
            s_final=self.spin_s_final.value(),
            lambda_std=self.spin_lambda.value(),
            seed=self.spin_seed.value(),
            swap_pct=float(self.spin_swap.value()),
            insert_pct=float(self.spin_insert.value()),
            adjacent_pct=float(self.spin_adj.value()),
            t0_auto=self.rb_temp_auto.isChecked(),
            t0_manual=self.spin_t0.value(),
            t_end=self.spin_tend.value(),
        )

    def set_settings(self, s: SearchSettings) -> None:
        self._block_signals = True
        self.spin_chains.setValue(s.chains)
        self.spin_epochs.setValue(s.epochs)
        self.spin_iters.setValue(s.iters)
        self.spin_scenarios.setValue(s.scenarios)
        self.spin_s_final.setValue(s.s_final)
        self.spin_lambda.setValue(s.lambda_std)
        self.spin_seed.setValue(s.seed)

        self.spin_swap.setValue(int(round(s.swap_pct)))
        self.spin_insert.setValue(int(round(s.insert_pct)))
        self.spin_adj.setValue(int(round(s.adjacent_pct)))

        if s.t0_auto:
            self.rb_temp_auto.setChecked(True)
            self.spin_t0.setEnabled(False)
            self.spin_tend.setEnabled(False)
        else:
            self.rb_temp_manual.setChecked(True)
            self.spin_t0.setEnabled(True)
            self.spin_tend.setEnabled(True)

        self.spin_t0.setValue(s.t0_manual)
        self.spin_tend.setValue(s.t_end)

        self._block_signals = False
        self._on_mix_changed()
