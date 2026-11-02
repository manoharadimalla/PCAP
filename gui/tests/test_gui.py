"""
Headless unit and integration tests for RunwayMC GUI.
"""

from __future__ import annotations

import os
import tempfile
import unittest

from gui.models import (
    AircraftRow,
    ExecutionSettings,
    ProjectState,
    ScenarioSettings,
    SearchSettings,
    get_default_aircraft_list,
    get_default_separation_matrix,
)
from gui.io_files import (
    clock_to_sec,
    sec_to_clock,
    read_instance_csv,
    write_instance_csv,
    read_separation_csv,
    write_separation_csv,
    read_params_cfg,
    write_params_cfg,
    save_project_state,
    load_project_state,
)


class TestGuiCore(unittest.TestCase):
    def test_time_conversion(self):
        self.assertEqual(sec_to_clock(0), "00:00:00")
        self.assertEqual(sec_to_clock(111), "00:01:51")
        self.assertEqual(sec_to_clock(3665), "01:01:05")

        self.assertEqual(clock_to_sec("00:00:00"), 0.0)
        self.assertEqual(clock_to_sec("00:01:51"), 111.0)
        self.assertEqual(clock_to_sec("01:01:05"), 3665.0)
        self.assertEqual(clock_to_sec("120"), 120.0)

    def test_aircraft_roundtrip(self):
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
            path = tmp.name

        try:
            defaults = get_default_aircraft_list()
            self.assertEqual(len(defaults), 4)
            write_instance_csv(path, defaults)
            loaded = read_instance_csv(path)
            self.assertEqual(len(loaded), 4)
            self.assertEqual(loaded[0].id, 0)
            self.assertEqual(loaded[0].op, "ARR")
            self.assertEqual(loaded[0].wake, "M")
            self.assertAlmostEqual(loaded[0].target, 111.0, places=3)
            self.assertEqual(loaded[2].priority, "High")
            self.assertEqual(loaded[2].weight, 3.0)
        finally:
            if os.path.exists(path):
                os.remove(path)

    def test_separation_roundtrip(self):
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
            path = tmp.name

        try:
            mat = get_default_separation_matrix()
            self.assertEqual(len(mat), 6)
            self.assertEqual(len(mat[0]), 6)
            write_separation_csv(path, mat)
            loaded = read_separation_csv(path)
            self.assertEqual(len(loaded), 6)
            self.assertEqual(loaded[0][0], 120.0)
            self.assertEqual(loaded[0][1], 157.0)
        finally:
            if os.path.exists(path):
                os.remove(path)

    def test_project_state_roundtrip(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            path = tmp.name

        try:
            state = ProjectState()
            save_project_state(path, state)
            loaded = load_project_state(path)
            self.assertEqual(len(loaded.aircraft_list), len(state.aircraft_list))
            self.assertEqual(loaded.scenario_settings.p_close, state.scenario_settings.p_close)
            self.assertEqual(loaded.search_settings.chains, state.search_settings.chains)
            self.assertEqual(loaded.execution_settings.impl, state.execution_settings.impl)
        finally:
            if os.path.exists(path):
                os.remove(path)


if __name__ == "__main__":
    unittest.main()

