"""
File I/O operations for RunwayMC GUI.

Handles reading and writing input files (instance.csv, separation.csv, params.cfg),
results (result.json, replay JSON), and project state files.
"""

from __future__ import annotations

import configparser
import json
import os
import shutil
from typing import Any, List, Optional

from gui.models import (
    AircraftRow,
    ExecutionSettings,
    ProjectState,
    ScenarioSettings,
    SearchSettings,
    get_default_aircraft_list,
    get_default_separation_matrix,
)


def sec_to_clock(seconds: float, include_subseconds: bool = False) -> str:
    """Convert seconds to clock format HH:MM:SS or HH:MM:SS.s."""
    if seconds < 0:
        sign = "-"
        seconds = abs(seconds)
    else:
        sign = ""

    total_secs = int(seconds)
    hours = total_secs // 3600
    minutes = (total_secs % 3600) // 60
    secs = total_secs % 60

    if include_subseconds:
        frac = seconds - total_secs
        if frac > 0.001:
            return f"{sign}{hours:02d}:{minutes:02d}:{secs:02d}.{int(round(frac * 10)):01d}"
    return f"{sign}{hours:02d}:{minutes:02d}:{secs:02d}"


def clock_to_sec(time_str: str) -> float:
    """Parse time string (HH:MM:SS, MM:SS, or decimal seconds) to float seconds."""
    s = time_str.strip()
    if not s:
        return 0.0

    # If it is purely a number, return float
    try:
        return float(s)
    except ValueError:
        pass

    parts = s.split(":")
    try:
        if len(parts) == 3:
            h = float(parts[0])
            m = float(parts[1])
            sec = float(parts[2])
            sign = -1.0 if s.startswith("-") else 1.0
            return sign * (abs(h) * 3600.0 + abs(m) * 60.0 + abs(sec))
        elif len(parts) == 2:
            m = float(parts[0])
            sec = float(parts[1])
            sign = -1.0 if s.startswith("-") else 1.0
            return sign * (abs(m) * 60.0 + abs(sec))
    except ValueError:
        raise ValueError(f"Invalid time format: '{time_str}'. Expected HH:MM:SS or seconds.")

    raise ValueError(f"Invalid time format: '{time_str}'. Expected HH:MM:SS or seconds.")


def write_instance_csv(filepath: str, aircraft_list: List[AircraftRow]) -> None:
    """Write aircraft rows to instance.csv format."""
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("id,op,wake,ready,target,latest,priority,weight\n")
        for ac in aircraft_list:
            f.write(
                f"{ac.id},{ac.op},{ac.wake},"
                f"{ac.ready:.4f},{ac.target:.4f},{ac.latest:.4f},"
                f"{ac.priority},{ac.weight:.4f}\n"
            )


def read_instance_csv(filepath: str) -> List[AircraftRow]:
    """Read aircraft rows from an instance.csv file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Instance file not found: {filepath}")

    aircraft_list: List[AircraftRow] = []
    with open(filepath, "r", encoding="utf-8") as f:
        header_read = False
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if not header_read:
                header_read = True
                if "id" in line.lower() and "wake" in line.lower():
                    continue

            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 8:
                aircraft_list.append(
                    AircraftRow(
                        id=int(parts[0]),
                        op=parts[1].upper(),
                        wake=parts[2].upper(),
                        ready=float(parts[3]),
                        target=float(parts[4]),
                        latest=float(parts[5]),
                        priority=parts[6].capitalize(),
                        weight=float(parts[7]),
                    )
                )
    return aircraft_list


def write_separation_csv(filepath: str, matrix: List[List[float]]) -> None:
    """Write 6x6 separation matrix to CSV file."""
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("# RunwayMC Separation Matrix (6x6)\n")
        f.write("# Rows: Leader (ARR-H, ARR-M, ARR-L, DEP-H, DEP-M, DEP-L)\n")
        f.write("# Cols: Follower (ARR-H, ARR-M, ARR-L, DEP-H, DEP-M, DEP-L)\n")
        for row in matrix:
            f.write(",".join(f"{val:.1f}" for val in row) + "\n")


def read_separation_csv(filepath: str) -> List[List[float]]:
    """Read 6x6 separation matrix from CSV file."""
    if not os.path.exists(filepath):
        return get_default_separation_matrix()

    matrix: List[List[float]] = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 6:
                matrix.append([float(x) for x in parts[:6]])
            if len(matrix) == 6:
                break

    if len(matrix) != 6 or any(len(r) != 6 for r in matrix):
        return get_default_separation_matrix()
    return matrix


def write_params_cfg(
    filepath: str,
    scenario: ScenarioSettings,
    search: Optional[SearchSettings] = None,
) -> None:
    """Write scenario (and optional search) settings to INI-style params.cfg."""
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    config = configparser.ConfigParser()
    config["scenario"] = {
        "arr_delay_mean": str(scenario.arr_delay_mean),
        "dep_delay_mean": str(scenario.dep_delay_mean),
        "weather_mult_0": str(scenario.weather_mults[0] if len(scenario.weather_mults) > 0 else 1.0),
        "weather_mult_1": str(scenario.weather_mults[1] if len(scenario.weather_mults) > 1 else 1.15),
        "weather_mult_2": str(scenario.weather_mults[2] if len(scenario.weather_mults) > 2 else 1.30),
        "weather_prob_0": str(scenario.weather_probs[0] if len(scenario.weather_probs) > 0 else 0.70),
        "weather_prob_1": str(scenario.weather_probs[1] if len(scenario.weather_probs) > 1 else 0.20),
        "weather_prob_2": str(scenario.weather_probs[2] if len(scenario.weather_probs) > 2 else 0.10),
        "p_close": str(scenario.p_close),
        "closure_min": str(scenario.closure_min),
        "closure_max": str(scenario.closure_max),
        "m_penalty": str(scenario.m_penalty),
        "weight_normal": str(scenario.weight_normal),
        "weight_high": str(scenario.weight_high),
        "weight_emergency": str(scenario.weight_emergency),
    }
    if search is not None:
        config["search"] = {
            "chains": str(search.chains),
            "epochs": str(search.epochs),
            "iters": str(search.iters),
            "scenarios": str(search.scenarios),
            "s_final": str(search.s_final),
            "lambda_std": str(search.lambda_std),
            "seed": str(search.seed),
            "swap_pct": str(search.swap_pct),
            "insert_pct": str(search.insert_pct),
            "adjacent_pct": str(search.adjacent_pct),
            "t0_auto": str(search.t0_auto),
            "t0_manual": str(search.t0_manual),
            "t_end": str(search.t_end),
        }

    with open(filepath, "w", encoding="utf-8") as f:
        config.write(f)


def read_params_cfg(filepath: str) -> ScenarioSettings:
    """Read scenario settings from params.cfg."""
    if not os.path.exists(filepath):
        return ScenarioSettings()

    config = configparser.ConfigParser()
    config.read(filepath, encoding="utf-8")

    sec = config["scenario"] if "scenario" in config else {}

    mults = [
        float(sec.get("weather_mult_0", "1.0")),
        float(sec.get("weather_mult_1", "1.15")),
        float(sec.get("weather_mult_2", "1.30")),
    ]
    probs = [
        float(sec.get("weather_prob_0", "0.70")),
        float(sec.get("weather_prob_1", "0.20")),
        float(sec.get("weather_prob_2", "0.10")),
    ]

    return ScenarioSettings(
        arr_delay_mean=float(sec.get("arr_delay_mean", "30.0")),
        dep_delay_mean=float(sec.get("dep_delay_mean", "60.0")),
        weather_mults=mults,
        weather_probs=probs,
        p_close=float(sec.get("p_close", "0.20")),
        closure_min=float(sec.get("closure_min", "300.0")),
        closure_max=float(sec.get("closure_max", "900.0")),
        m_penalty=float(sec.get("m_penalty", "5.0")),
        weight_normal=float(sec.get("weight_normal", "1.0")),
        weight_high=float(sec.get("weight_high", "3.0")),
        weight_emergency=float(sec.get("weight_emergency", "20.0")),
    )


def read_result_json(filepath: str) -> dict[str, Any]:
    """Read result.json produced by runway CLI."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Result file not found: {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def read_replay_json(filepath: str) -> dict[str, Any]:
    """Read replay JSON produced by runway replay."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Replay file not found: {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def save_project_state(filepath: str, state: ProjectState) -> None:
    """Save full project state to a JSON file (gui_state.json)."""
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(state.to_dict(), f, indent=2)


def load_project_state(filepath: str) -> ProjectState:
    """Load full project state from a JSON file (gui_state.json)."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Project state file not found: {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    return ProjectState.from_dict(data)


def find_runway_binary(configured_path: Optional[str] = None) -> Optional[str]:
    """
    Search for the runway executable.
    Checks:
      1. Configured path if provided and exists
      2. Standard build directories relative to current working directory or project root
      3. PATH environment variable
    """
    if configured_path and os.path.isfile(configured_path):
        return os.path.abspath(configured_path)

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    candidates = [
        os.path.join(project_root, "build", "runway.exe"),
        os.path.join(project_root, "build", "runway"),
        os.path.join(project_root, "build", "Release", "runway.exe"),
        os.path.join(project_root, "build", "Debug", "runway.exe"),
        os.path.join("build", "runway.exe"),
        os.path.join("build", "runway"),
    ]

    for p in candidates:
        if os.path.isfile(p):
            return os.path.abspath(p)

    which_runway = shutil.which("runway") or shutil.which("runway.exe")
    if which_runway:
        return os.path.abspath(which_runway)

    return None
