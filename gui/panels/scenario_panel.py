"""
Scenario Settings Panel for RunwayMC GUI.

Configures operational uncertainty parameters: arrival/departure exponential delay means,
weather state multipliers & probabilities, runway closure probabilities & durations,
deadline-miss penalty, and priority weights.
"""

from __future__ import annotations

from typing import List, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from gui.models import ScenarioSettings


class ScenarioPanel(QWidget):
    """
    Panel for configuring scenario simulation parameters.
    """
    scenario_changed = Signal()
    priority_weights_changed = Signal(float, float, float)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._block_signals: bool = False
        self._setup_ui()
        self.set_settings(ScenarioSettings())

    def _setup_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)

        group = QGroupBox("Scenario & Uncertainty Parameters")
        form = QFormLayout(group)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)

        # 1. Delays
        self.spin_arr_delay = QDoubleSpinBox()
        self.spin_arr_delay.setRange(0.0, 3600.0)
        self.spin_arr_delay.setSingleStep(5.0)
        self.spin_arr_delay.setSuffix(" s")
        self.spin_arr_delay.setValue(30.0)
        self.spin_arr_delay.valueChanged.connect(self._on_changed)
        form.addRow("Arrival Delay Mean:", self.spin_arr_delay)

        self.spin_dep_delay = QDoubleSpinBox()
        self.spin_dep_delay.setRange(0.0, 3600.0)
        self.spin_dep_delay.setSingleStep(5.0)
        self.spin_dep_delay.setSuffix(" s")
        self.spin_dep_delay.setValue(60.0)
        self.spin_dep_delay.valueChanged.connect(self._on_changed)
        form.addRow("Departure Delay Mean:", self.spin_dep_delay)

        # 2. Weather multipliers (Good / Moderate / Severe)
        w_mult_layout = QHBoxLayout()
        self.spin_w_mult0 = QDoubleSpinBox()
        self.spin_w_mult0.setRange(0.5, 3.0)
        self.spin_w_mult0.setSingleStep(0.05)
        self.spin_w_mult0.setValue(1.0)
        self.spin_w_mult0.valueChanged.connect(self._on_changed)

        self.spin_w_mult1 = QDoubleSpinBox()
        self.spin_w_mult1.setRange(0.5, 3.0)
        self.spin_w_mult1.setSingleStep(0.05)
        self.spin_w_mult1.setValue(1.15)
        self.spin_w_mult1.valueChanged.connect(self._on_changed)

        self.spin_w_mult2 = QDoubleSpinBox()
        self.spin_w_mult2.setRange(0.5, 3.0)
        self.spin_w_mult2.setSingleStep(0.05)
        self.spin_w_mult2.setValue(1.30)
        self.spin_w_mult2.valueChanged.connect(self._on_changed)

        w_mult_layout.addWidget(QLabel("W0:"))
        w_mult_layout.addWidget(self.spin_w_mult0)
        w_mult_layout.addWidget(QLabel("W1:"))
        w_mult_layout.addWidget(self.spin_w_mult1)
        w_mult_layout.addWidget(QLabel("W2:"))
        w_mult_layout.addWidget(self.spin_w_mult2)
        form.addRow("Weather Multipliers:", w_mult_layout)

        # 3. Weather probabilities
        w_prob_layout = QHBoxLayout()
        self.spin_w_prob0 = QDoubleSpinBox()
        self.spin_w_prob0.setRange(0.0, 1.0)
        self.spin_w_prob0.setSingleStep(0.05)
        self.spin_w_prob0.setValue(0.70)
        self.spin_w_prob0.valueChanged.connect(self._on_weather_prob_changed)

        self.spin_w_prob1 = QDoubleSpinBox()
        self.spin_w_prob1.setRange(0.0, 1.0)
        self.spin_w_prob1.setSingleStep(0.05)
        self.spin_w_prob1.setValue(0.20)
        self.spin_w_prob1.valueChanged.connect(self._on_weather_prob_changed)

        self.spin_w_prob2 = QDoubleSpinBox()
        self.spin_w_prob2.setRange(0.0, 1.0)
        self.spin_w_prob2.setSingleStep(0.05)
        self.spin_w_prob2.setValue(0.10)
        self.spin_w_prob2.valueChanged.connect(self._on_weather_prob_changed)

        w_prob_layout.addWidget(QLabel("P0:"))
        w_prob_layout.addWidget(self.spin_w_prob0)
        w_prob_layout.addWidget(QLabel("P1:"))
        w_prob_layout.addWidget(self.spin_w_prob1)
        w_prob_layout.addWidget(QLabel("P2:"))
        w_prob_layout.addWidget(self.spin_w_prob2)
        form.addRow("Weather Probs:", w_prob_layout)

        # Warning label if sum != 1.0
        self.lbl_prob_warning = QLabel("")
        self.lbl_prob_warning.setStyleSheet("color: red; font-weight: bold; font-size: 11px;")
        form.addRow("", self.lbl_prob_warning)

        # 4. Runway closure probability
        close_p_layout = QHBoxLayout()
        self.slider_p_close = QSlider(Qt.Orientation.Horizontal)
        self.slider_p_close.setRange(0, 100)
        self.slider_p_close.setValue(20)

        self.spin_p_close = QDoubleSpinBox()
        self.spin_p_close.setRange(0.0, 1.0)
        self.spin_p_close.setSingleStep(0.05)
        self.spin_p_close.setValue(0.20)

        self.slider_p_close.valueChanged.connect(lambda v: self._sync_close_slider(v))
        self.spin_p_close.valueChanged.connect(lambda v: self._sync_close_spin(v))

        close_p_layout.addWidget(self.slider_p_close)
        close_p_layout.addWidget(self.spin_p_close)
        form.addRow("Closure Prob (p_close):", close_p_layout)

        # 5. Closure durations
        close_dur_layout = QHBoxLayout()
        self.spin_close_min = QDoubleSpinBox()
        self.spin_close_min.setRange(0.0, 7200.0)
        self.spin_close_min.setSingleStep(30.0)
        self.spin_close_min.setSuffix(" s")
        self.spin_close_min.setValue(300.0)
        self.spin_close_min.valueChanged.connect(self._on_changed)

        self.spin_close_max = QDoubleSpinBox()
        self.spin_close_max.setRange(0.0, 7200.0)
        self.spin_close_max.setSingleStep(30.0)
        self.spin_close_max.setSuffix(" s")
        self.spin_close_max.setValue(900.0)
        self.spin_close_max.valueChanged.connect(self._on_changed)

        close_dur_layout.addWidget(QLabel("Min:"))
        close_dur_layout.addWidget(self.spin_close_min)
        close_dur_layout.addWidget(QLabel("Max:"))
        close_dur_layout.addWidget(self.spin_close_max)
        form.addRow("Closure Duration:", close_dur_layout)

        # 6. Penalty M
        self.spin_m_penalty = QDoubleSpinBox()
        self.spin_m_penalty.setRange(1.0, 100.0)
        self.spin_m_penalty.setSingleStep(1.0)
        self.spin_m_penalty.setValue(5.0)
        self.spin_m_penalty.valueChanged.connect(self._on_changed)
        form.addRow("Miss Penalty (M):", self.spin_m_penalty)

        # 7. Priority weights
        prio_layout = QHBoxLayout()
        self.spin_w_norm = QDoubleSpinBox()
        self.spin_w_norm.setRange(0.1, 100.0)
        self.spin_w_norm.setValue(1.0)
        self.spin_w_norm.valueChanged.connect(self._on_prio_changed)

        self.spin_w_high = QDoubleSpinBox()
        self.spin_w_high.setRange(0.1, 100.0)
        self.spin_w_high.setValue(3.0)
        self.spin_w_high.valueChanged.connect(self._on_prio_changed)

        self.spin_w_emerg = QDoubleSpinBox()
        self.spin_w_emerg.setRange(0.1, 500.0)
        self.spin_w_emerg.setValue(20.0)
        self.spin_w_emerg.valueChanged.connect(self._on_prio_changed)

        prio_layout.addWidget(QLabel("Normal:"))
        prio_layout.addWidget(self.spin_w_norm)
        prio_layout.addWidget(QLabel("High:"))
        prio_layout.addWidget(self.spin_w_high)
        prio_layout.addWidget(QLabel("Emerg:"))
        prio_layout.addWidget(self.spin_w_emerg)
        form.addRow("Priority Weights:", prio_layout)

        main_layout.addWidget(group)

    def _sync_close_slider(self, val: int) -> None:
        if self._block_signals:
            return
        self._block_signals = True
        self.spin_p_close.setValue(val / 100.0)
        self._block_signals = False
        self._on_changed()

    def _sync_close_spin(self, val: float) -> None:
        if self._block_signals:
            return
        self._block_signals = True
        self.slider_p_close.setValue(int(round(val * 100.0)))
        self._block_signals = False
        self._on_changed()

    def _on_weather_prob_changed(self) -> None:
        prob_sum = self.spin_w_prob0.value() + self.spin_w_prob1.value() + self.spin_w_prob2.value()
        if abs(prob_sum - 1.0) > 0.001:
            self.lbl_prob_warning.setText(f"⚠ Warning: Probs sum to {prob_sum:.2f} (must equal 1.0)")
        else:
            self.lbl_prob_warning.setText("")
        self._on_changed()

    def _on_prio_changed(self) -> None:
        if not self._block_signals:
            self.priority_weights_changed.emit(
                self.spin_w_norm.value(),
                self.spin_w_high.value(),
                self.spin_w_emerg.value(),
            )
            self._on_changed()

    def _on_changed(self) -> None:
        if not self._block_signals:
            self.scenario_changed.emit()

    def is_weather_probs_valid(self) -> bool:
        prob_sum = self.spin_w_prob0.value() + self.spin_w_prob1.value() + self.spin_w_prob2.value()
        return abs(prob_sum - 1.0) <= 0.001

    def get_settings(self) -> ScenarioSettings:
        return ScenarioSettings(
            arr_delay_mean=self.spin_arr_delay.value(),
            dep_delay_mean=self.spin_dep_delay.value(),
            weather_mults=[
                self.spin_w_mult0.value(),
                self.spin_w_mult1.value(),
                self.spin_w_mult2.value(),
            ],
            weather_probs=[
                self.spin_w_prob0.value(),
                self.spin_w_prob1.value(),
                self.spin_w_prob2.value(),
            ],
            p_close=self.spin_p_close.value(),
            closure_min=self.spin_close_min.value(),
            closure_max=self.spin_close_max.value(),
            m_penalty=self.spin_m_penalty.value(),
            weight_normal=self.spin_w_norm.value(),
            weight_high=self.spin_w_high.value(),
            weight_emergency=self.spin_w_emerg.value(),
        )

    def set_settings(self, s: ScenarioSettings) -> None:
        self._block_signals = True
        self.spin_arr_delay.setValue(s.arr_delay_mean)
        self.spin_dep_delay.setValue(s.dep_delay_mean)

        m = s.weather_mults if len(s.weather_mults) == 3 else [1.0, 1.15, 1.30]
        self.spin_w_mult0.setValue(m[0])
        self.spin_w_mult1.setValue(m[1])
        self.spin_w_mult2.setValue(m[2])

        p = s.weather_probs if len(s.weather_probs) == 3 else [0.70, 0.20, 0.10]
        self.spin_w_prob0.setValue(p[0])
        self.spin_w_prob1.setValue(p[1])
        self.spin_w_prob2.setValue(p[2])

        self.spin_p_close.setValue(s.p_close)
        self.slider_p_close.setValue(int(round(s.p_close * 100.0)))
        self.spin_close_min.setValue(s.closure_min)
        self.spin_close_max.setValue(s.closure_max)
        self.spin_m_penalty.setValue(s.m_penalty)

        self.spin_w_norm.setValue(s.weight_normal)
        self.spin_w_high.setValue(s.weight_high)
        self.spin_w_emerg.setValue(s.weight_emergency)

        self._block_signals = False
        self._on_weather_prob_changed()
