#!/usr/bin/env python3
"""
Hardware and environment probe for RunwayMC.
Records CPU, GPU, compiler, OS, Python, and Git environment.
"""

import os
import sys
import platform
import subprocess
import json
import shutil

def run_cmd(cmd_list):
    try:
        res = subprocess.run(cmd_list, capture_output=True, text=True, check=True)
        return res.stdout.strip()
    except Exception as e:
        return None

def get_git_info():
    git_hash = run_cmd(["git", "rev-parse", "--short", "HEAD"])
    branch = run_cmd(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    dirty = run_cmd(["git", "status", "--porcelain"])
    return {
        "hash": git_hash or "unknown",
        "branch": branch or "unknown",
        "is_dirty": bool(dirty)
    }

def get_cpu_info():
    info = {
        "processor": platform.processor(),
        "architecture": platform.machine(),
        "logical_cores": os.cpu_count()
    }
    if platform.system() == "Windows":
        wmic_name = run_cmd(["powershell", "-Command", "(Get-CimInstance Win32_Processor).Name"])
        if wmic_name:
            info["model"] = wmic_name
        cores = run_cmd(["powershell", "-Command", "(Get-CimInstance Win32_Processor).NumberOfCores"])
        if cores:
            info["physical_cores"] = cores
    return info

def get_gpu_info():
    smi = shutil.which("nvidia-smi")
    if not smi:
        return {"available": False, "reason": "nvidia-smi not found"}
    out = run_cmd(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader,nounits"])
    if not out:
        return {"available": False, "reason": "nvidia-smi query failed"}
    gpus = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3:
            gpus.append({
                "name": parts[0],
                "driver_version": parts[1],
                "memory_total_mib": parts[2]
            })
    nvcc = shutil.which("nvcc")
    nvcc_ver = run_cmd(["nvcc", "--version"]) if nvcc else None
    return {
        "available": len(gpus) > 0,
        "nvcc_installed": bool(nvcc),
        "nvcc_version": nvcc_ver,
        "gpus": gpus
    }

def get_compiler_info():
    compilers = {}
    for c in ["g++", "gcc", "clang++", "cl", "nvcc", "cmake", "ninja"]:
        path = shutil.which(c)
        if path:
            compilers[c] = {"path": path}
            if c in ["g++", "gcc", "clang++"]:
                ver = run_cmd([c, "-dumpversion"]) or run_cmd([c, "--version"])
                compilers[c]["version"] = ver.splitlines()[0] if ver else "unknown"
            elif c == "cmake":
                ver = run_cmd([c, "--version"])
                compilers[c]["version"] = ver.splitlines()[0] if ver else "unknown"
            elif c == "ninja":
                ver = run_cmd([c, "--version"])
                compilers[c]["version"] = ver.splitlines()[0] if ver else "unknown"
    return compilers

def collect_all():
    return {
        "os": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version()
        },
        "python": {
            "version": platform.python_version(),
            "executable": sys.executable
        },
        "cpu": get_cpu_info(),
        "gpu": get_gpu_info(),
        "compilers": get_compiler_info(),
        "git": get_git_info()
    }

def generate_hardware_markdown(env_data, out_path="docs/HARDWARE.md"):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    lines = [
        "# Hardware and System Environment",
        "",
        f"**OS:** {env_data['os']['system']} {env_data['os']['release']} (Version {env_data['os']['version']})",
        f"**Python:** {env_data['python']['version']} ({env_data['python']['executable']})",
        "",
        "## CPU Details",
        f"- **Model:** {env_data['cpu'].get('model', env_data['cpu'].get('processor', 'Unknown'))}",
        f"- **Physical Cores:** {env_data['cpu'].get('physical_cores', 'Unknown')}",
        f"- **Logical Cores:** {env_data['cpu'].get('logical_cores', 'Unknown')}",
        f"- **Architecture:** {env_data['cpu'].get('architecture', 'Unknown')}",
        "",
        "## GPU Details",
    ]
    gpu = env_data["gpu"]
    if gpu.get("available") and gpu.get("gpus"):
        for i, g in enumerate(gpu["gpus"]):
            lines.append(f"### GPU {i}: {g['name']}")
            lines.append(f"- **Driver Version:** {g['driver_version']}")
            lines.append(f"- **Total VRAM:** {g['memory_total_mib']} MiB")
        lines.append(f"- **NVCC Compiler Available:** {'Yes' if gpu.get('nvcc_installed') else 'No'}")
        if not gpu.get("nvcc_installed"):
            lines.append("  *(Note: NVIDIA GPU hardware and driver detected, but CUDA Toolkit nvcc is not on PATH. Local builds run CPU/OpenMP; GPU builds can be executed on Google Colab or with CUDA Toolkit.)*")
    else:
        lines.append("- No NVIDIA GPU detected.")
    lines.append("")
    lines.append("## Compilers & Tooling")
    for name, info in env_data.get("compilers", {}).items():
        ver = info.get("version", "installed")
        lines.append(f"- **{name}:** `{info['path']}` ({ver})")
    lines.append("")
    lines.append("## Git Status")
    lines.append(f"- **Commit:** `{env_data['git']['hash']}`")
    lines.append(f"- **Branch:** `{env_data['git']['branch']}`")
    lines.append(f"- **Dirty:** {env_data['git']['is_dirty']}")
    lines.append("")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

if __name__ == "__main__":
    env_data = collect_all()
    out_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(out_dir, ".."))
    hw_md_path = os.path.join(root_dir, "docs", "HARDWARE.md")
    generate_hardware_markdown(env_data, hw_md_path)
    json_path = os.path.join(root_dir, "docs", "env_info.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(env_data, f, indent=2)
    print(f"Generated {hw_md_path} and {json_path}")
