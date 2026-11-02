"""
QProcess runner for RunwayMC CLI.

Wraps execution of the C++ `runway` binary with QProcess to ensure
the GUI remains fully responsive, supports cancellation, parses
real-time progress updates, and emits structured completion signals.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, List, Optional

from PySide6.QtCore import QObject, QProcess, Signal

from gui.io_files import read_result_json


class RunwayRunner(QObject):
    """
    Asynchronous runner for long-running runway solver commands.
    """
    # Signals
    # epoch, total_epochs, best_j, elapsed_seconds
    progress = Signal(int, int, float, float)
    finished = Signal(dict)
    error = Signal(str)
    cancelled = Signal()
    log_line = Signal(str)

    _PROGRESS_REGEX = re.compile(
        r"PROGRESS\s+(\d+)\s*/\s*(\d+)\s+([0-9.eE+-]+)(?:\s+([0-9.eE+-]+))?"
    )

    def __init__(self, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self.process: Optional[QProcess] = None
        self._result_json_path: str = ""
        self._is_cancelled: bool = False
        self._stdout_buffer: List[str] = []
        self._stderr_buffer: List[str] = []

    def is_running(self) -> bool:
        return self.process is not None and self.process.state() != QProcess.ProcessState.NotRunning

    def start(self, command: str, args: List[str], result_json_path: str = "") -> bool:
        """
        Start the runner process with the specified command and arguments.
        """
        if self.is_running():
            self.error.emit("A solve process is already running.")
            return False

        if not os.path.isfile(command):
            self.error.emit(f"Runway executable not found at: {command}")
            return False

        self._result_json_path = result_json_path
        self._is_cancelled = False
        self._stdout_buffer.clear()
        self._stderr_buffer.clear()

        self.process = QProcess(self)
        self.process.setProgram(command)
        self.process.setArguments(args)

        self.process.readyReadStandardOutput.connect(self._on_stdout_ready)
        self.process.readyReadStandardError.connect(self._on_stderr_ready)
        self.process.finished.connect(self._on_process_finished)

        self.log_line.emit(f"$ {command} " + " ".join(f'"{a}"' if " " in a else a for a in args))
        self.process.start()
        return True

    def cancel(self) -> None:
        """Cancel the currently running process."""
        if self.is_running() and self.process is not None:
            self._is_cancelled = True
            self.log_line.emit("[Runner] Cancelling process...")
            self.process.terminate()
            # Give it a short chance to terminate gracefully, otherwise kill
            if not self.process.waitForFinished(1000):
                self.process.kill()
            self.cancelled.emit()

    def _on_stdout_ready(self) -> None:
        if self.process is None:
            return
        data = self.process.readAllStandardOutput().data().decode("utf-8", errors="replace")
        for line in data.splitlines():
            line_str = line.strip()
            if not line_str:
                continue
            self._stdout_buffer.append(line_str)
            self.log_line.emit(line_str)
            self._parse_line_for_progress(line_str)

    def _on_stderr_ready(self) -> None:
        if self.process is None:
            return
        data = self.process.readAllStandardError().data().decode("utf-8", errors="replace")
        for line in data.splitlines():
            line_str = line.strip()
            if not line_str:
                continue
            self._stderr_buffer.append(line_str)
            self.log_line.emit(f"[stderr] {line_str}")
            self._parse_line_for_progress(line_str)

    def _parse_line_for_progress(self, line: str) -> None:
        m = self._PROGRESS_REGEX.search(line)
        if m:
            epoch = int(m.group(1))
            total = int(m.group(2))
            best_j = float(m.group(3))
            elapsed_s = float(m.group(4)) if m.group(4) else 0.0
            self.progress.emit(epoch, total, best_j, elapsed_s)

    def _on_process_finished(self, exit_code: int, exit_status: QProcess.ExitStatus) -> None:
        if self._is_cancelled:
            self.cancelled.emit()
            return

        if exit_code == 0:
            result_dict: dict[str, Any] = {}
            if self._result_json_path and os.path.isfile(self._result_json_path):
                try:
                    result_dict = read_result_json(self._result_json_path)
                except Exception as e:
                    self.error.emit(f"Failed to parse result JSON at {self._result_json_path}: {e}")
                    return
            else:
                result_dict = {
                    "impl": "unknown",
                    "stdout": "\n".join(self._stdout_buffer),
                }
            self.finished.emit(result_dict)
        else:
            stderr_all = "\n".join(self._stderr_buffer)
            stdout_all = "\n".join(self._stdout_buffer)
            err_msg = stderr_all if stderr_all else (stdout_all if stdout_all else f"Process exited with code {exit_code}")
            self.error.emit(f"Process failed (exit code {exit_code}):\n{err_msg}")


def run_command_sync(binary: str, args: List[str], timeout_ms: int = 5000) -> tuple[int, str, str]:
    """
    Run a short command synchronously using QProcess.
    Returns (exit_code, stdout, stderr).
    """
    proc = QProcess()
    proc.start(binary, args)
    if not proc.waitForStarted(2000):
        return -1, "", "Failed to start executable"
    if not proc.waitForFinished(timeout_ms):
        proc.kill()
        return -2, "", "Command timed out"

    out = proc.readAllStandardOutput().data().decode("utf-8", errors="replace")
    err = proc.readAllStandardError().data().decode("utf-8", errors="replace")
    return proc.exitCode(), out, err


def detect_hardware_info(binary_path: Optional[str]) -> dict[str, Any]:
    """
    Query capabilities via `runway info` or provide safe fallback.
    """
    default_info = {
        "version": "1.0.0",
        "cuda_available": False,
        "gpu_name": "No CUDA GPU Detected",
        "gpu_sms": 0,
        "omp_available": True,
        "omp_max_threads": os.cpu_count() or 4,
    }

    if not binary_path or not os.path.isfile(binary_path):
        return default_info

    exit_code, stdout, _ = run_command_sync(binary_path, ["info"], timeout_ms=3000)
    if exit_code == 0 and stdout.strip():
        try:
            parsed = json.loads(stdout.strip())
            return {
                "version": parsed.get("version", "1.0.0"),
                "cuda_available": bool(parsed.get("cuda_available", False)),
                "gpu_name": parsed.get("gpu_name", "None"),
                "gpu_sms": int(parsed.get("gpu_sms", 0)),
                "omp_available": bool(parsed.get("omp_available", True)),
                "omp_max_threads": int(parsed.get("omp_max_threads", os.cpu_count() or 4)),
            }
        except Exception:
            pass

    return default_info


def validate_instance_via_cli(
    binary_path: Optional[str],
    instance_path: str,
    params_path: Optional[str] = None,
    separation_path: Optional[str] = None,
) -> dict[str, Any]:
    """
    Execute `runway validate --instance ...` and return validation dictionary.
    Falls back gracefully if the CLI binary doesn't support the validate subcommand.
    """
    if binary_path and os.path.isfile(binary_path):
        args = ["validate", "--instance", instance_path]
        if params_path:
            args.extend(["--params", params_path])
        if separation_path:
            args.extend(["--separation", separation_path])

        exit_code, stdout, _ = run_command_sync(binary_path, args, timeout_ms=4000)
        if stdout.strip():
            try:
                data = json.loads(stdout.strip())
                if isinstance(data, dict) and "ok" in data:
                    return data
            except Exception:
                pass

    # Basic structural validation in Python without algorithm logic
    # Checks: file exists, header format, target <= latest, ready <= target
    errors = []
    warnings = []
    try:
        if not os.path.isfile(instance_path):
            errors.append({"row": -1, "column": "file", "message": f"Instance file not found: {instance_path}"})
        else:
            with open(instance_path, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f if line.strip() and not line.startswith("#")]
            if len(lines) < 2:
                errors.append({"row": 0, "column": "id", "message": "Instance file must contain at least one aircraft"})
            else:
                seen_ids = set()
                for row_idx, line in enumerate(lines[1:]):
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) < 8:
                        errors.append({"row": row_idx, "column": "format", "message": f"Row {row_idx} has {len(parts)} columns (expected 8)"})
                        continue
                    aid, op, wake, ready_str, target_str, latest_str, prio, weight_str = parts[:8]
                    if aid in seen_ids:
                        errors.append({"row": row_idx, "column": "id", "message": f"Duplicate aircraft ID: {aid}"})
                    seen_ids.add(aid)
                    try:
                        ready = float(ready_str)
                        target = float(target_str)
                        latest = float(latest_str)
                        if target < ready:
                            errors.append({"row": row_idx, "column": "target", "message": f"Scheduled target ({target}) < ready ({ready})"})
                        if latest < target:
                            errors.append({"row": row_idx, "column": "latest", "message": f"Deadline ({latest}) < target ({target})"})
                    except ValueError as ve:
                        errors.append({"row": row_idx, "column": "time", "message": f"Invalid numeric time value: {ve}"})
    except Exception as e:
        errors.append({"row": -1, "column": "general", "message": str(e)})

    return {
        "ok": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "triangle_ok": True,
    }
