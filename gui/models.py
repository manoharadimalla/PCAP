"""
Data models for RunwayMC GUI.

This module defines pure data classes for GUI state and file representations.
Rule: The GUI is a thin front-end. No scheduling, simulation, or algorithm
logic in Python.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, List, Optional


@dataclass
class AircraftRow:
    """Represents a single aircraft in the scheduling instance."""
    id: int
    op: str          # "ARR" or "DEP"
    wake: str        # "H", "M", or "L"
    ready: float     # Ready / earliest release time in seconds
    target: float    # Scheduled / target runway time in seconds
    latest: float    # Deadline time in seconds
    priority: str    # "Normal", "High", or "Emergency"
    weight: float    # Cost penalty multiplier

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AircraftRow:
        return cls(
            id=int(data["id"]),
            op=str(data["op"]).strip().upper(),
            wake=str(data["wake"]).strip().upper(),
            ready=float(data["ready"]),
            target=float(data["target"]),
            latest=float(data["latest"]),
            priority=str(data["priority"]).strip().capitalize(),
            weight=float(data["weight"]),
        )


@dataclass
class ScenarioSettings:
    """Settings governing uncertainty, delays, weather, and penalties."""
    arr_delay_mean: float = 30.0
    dep_delay_mean: float = 60.0
    weather_mults: List[float] = field(default_factory=lambda: [1.0, 1.15, 1.30])
    weather_probs: List[float] = field(default_factory=lambda: [0.70, 0.20, 0.10])
    p_close: float = 0.20
    closure_min: float = 300.0
    closure_max: float = 900.0
    m_penalty: float = 5.0
    weight_normal: float = 1.0
    weight_high: float = 3.0
    weight_emergency: float = 20.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ScenarioSettings:
        return cls(
            arr_delay_mean=float(data.get("arr_delay_mean", 30.0)),
            dep_delay_mean=float(data.get("dep_delay_mean", 60.0)),
            weather_mults=[float(x) for x in data.get("weather_mults", [1.0, 1.15, 1.30])],
            weather_probs=[float(x) for x in data.get("weather_probs", [0.70, 0.20, 0.10])],
            p_close=float(data.get("p_close", 0.20)),
            closure_min=float(data.get("closure_min", 300.0)),
            closure_max=float(data.get("closure_max", 900.0)),
            m_penalty=float(data.get("m_penalty", 5.0)),
            weight_normal=float(data.get("weight_normal", 1.0)),
            weight_high=float(data.get("weight_high", 3.0)),
            weight_emergency=float(data.get("weight_emergency", 20.0)),
        )


@dataclass
class SearchSettings:
    """Settings for the Simulated Annealing Monte Carlo search."""
    chains: int = 16
    epochs: int = 5
    iters: int = 1000
    scenarios: int = 1024
    s_final: int = 20000
    lambda_std: float = 1.0
    seed: int = 42
    swap_pct: float = 40.0
    insert_pct: float = 40.0
    adjacent_pct: float = 20.0
    t0_auto: bool = True
    t0_manual: float = 100.0
    t_end: float = 0.1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SearchSettings:
        return cls(
            chains=int(data.get("chains", 16)),
            epochs=int(data.get("epochs", 5)),
            iters=int(data.get("iters", 1000)),
            scenarios=int(data.get("scenarios", 1024)),
            s_final=int(data.get("s_final", 20000)),
            lambda_std=float(data.get("lambda_std", 1.0)),
            seed=int(data.get("seed", 42)),
            swap_pct=float(data.get("swap_pct", 40.0)),
            insert_pct=float(data.get("insert_pct", 40.0)),
            adjacent_pct=float(data.get("adjacent_pct", 20.0)),
            t0_auto=bool(data.get("t0_auto", True)),
            t0_manual=float(data.get("t0_manual", 100.0)),
            t_end=float(data.get("t_end", 0.1)),
        )


@dataclass
class ExecutionSettings:
    """Settings for compute backend and hardware-specific parameters."""
    impl: str = "serial"          # "serial", "omp", "cuda"
    omp_mode: str = "chains"      # "chains", "scenarios", "hybrid"
    threads: int = 0              # 0 = automatic / OpenMP default
    kernel: str = "k3"            # "k1", "k2", "k3"
    block_size: int = 256

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExecutionSettings:
        return cls(
            impl=str(data.get("impl", "serial")),
            omp_mode=str(data.get("omp_mode", "chains")),
            threads=int(data.get("threads", 0)),
            kernel=str(data.get("kernel", "k3")),
            block_size=int(data.get("block_size", 256)),
        )


def get_default_aircraft_list() -> List[AircraftRow]:
    """Pre-filled 4-aircraft instance from the project proposal."""
    return [
        AircraftRow(id=0, op="ARR", wake="M", ready=0.0, target=111.0, latest=711.0, priority="Normal", weight=1.0),
        AircraftRow(id=1, op="DEP", wake="M", ready=209.0, target=209.0, latest=1109.0, priority="Normal", weight=1.0),
        AircraftRow(id=2, op="ARR", wake="H", ready=177.0, target=284.0, latest=884.0, priority="High", weight=3.0),
        AircraftRow(id=3, op="DEP", wake="L", ready=426.0, target=426.0, latest=1326.0, priority="Normal", weight=1.0),
    ]


def get_default_separation_matrix() -> List[List[float]]:
    """Default 6x6 separation matrix in seconds (Leader x Follower)."""
    return [
        [120.0, 157.0, 196.0,  60.0,  60.0,  60.0],
        [ 69.0,  69.0, 131.0,  60.0,  60.0,  60.0],
        [ 69.0,  69.0,  69.0,  60.0,  60.0,  60.0],
        [ 60.0,  60.0,  60.0, 120.0, 157.0, 196.0],
        [ 60.0,  60.0,  60.0,  69.0,  69.0, 131.0],
        [ 60.0,  60.0,  60.0,  69.0,  69.0,  69.0],
    ]


SEPARATION_CLASSES = [
    "ARR-H", "ARR-M", "ARR-L",
    "DEP-H", "DEP-M", "DEP-L",
]


@dataclass
class ProjectState:
    """Complete project state saved to/restored from gui_state.json."""
    aircraft_list: List[AircraftRow] = field(default_factory=get_default_aircraft_list)
    scenario_settings: ScenarioSettings = field(default_factory=ScenarioSettings)
    search_settings: SearchSettings = field(default_factory=SearchSettings)
    execution_settings: ExecutionSettings = field(default_factory=ExecutionSettings)
    separation_matrix: List[List[float]] = field(default_factory=get_default_separation_matrix)
    last_result: Optional[dict[str, Any]] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "aircraft_list": [ac.to_dict() for ac in self.aircraft_list],
            "scenario_settings": self.scenario_settings.to_dict(),
            "search_settings": self.search_settings.to_dict(),
            "execution_settings": self.execution_settings.to_dict(),
            "separation_matrix": self.separation_matrix,
            "last_result": self.last_result,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProjectState:
        aircraft_raw = data.get("aircraft_list", [])
        aircraft_list = [AircraftRow.from_dict(x) for x in aircraft_raw] if aircraft_raw else get_default_aircraft_list()
        scenario_settings = ScenarioSettings.from_dict(data.get("scenario_settings", {}))
        search_settings = SearchSettings.from_dict(data.get("search_settings", {}))
        execution_settings = ExecutionSettings.from_dict(data.get("execution_settings", {}))
        separation_matrix = data.get("separation_matrix", get_default_separation_matrix())
        last_result = data.get("last_result", None)

        return cls(
            aircraft_list=aircraft_list,
            scenario_settings=scenario_settings,
            search_settings=search_settings,
            execution_settings=execution_settings,
            separation_matrix=separation_matrix,
            last_result=last_result,
        )
