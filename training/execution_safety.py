"""Execution-only CPU safety controls for long unattended training jobs.

These controls deliberately do not alter PPO configuration, seeds, budgets, or
artifacts.  They constrain only the operating-system process that executes a
sealed request and are therefore suitable for append-only evidence runs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any

import psutil


ROOT = Path(__file__).resolve().parents[1]
PROFILE_ENV = "TRANSFERGRID_EXECUTION_SAFETY_PROFILE"
CPU_SAFE_PROFILE = "cpu_safe_v1"
CPU_COOL_PROFILE = "cpu_cool_v2"


@dataclass(frozen=True)
class ExecutionSafetyProfile:
    profile_id: str = CPU_SAFE_PROFILE
    logical_cpu_limit: int = 4
    priority: str = "below_normal"
    maximum_cpu_temperature_c: float = 90.0
    launch_temperature_c: float = 85.0
    temperature_poll_seconds: int = 60
    cooldown_seconds: int = 180
    consecutive_hot_readings_to_stop: int = 2

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


DEFAULT_PROFILE = ExecutionSafetyProfile()
COOL_PROFILE = ExecutionSafetyProfile(
    profile_id=CPU_COOL_PROFILE,
    logical_cpu_limit=2,
    priority="idle",
    maximum_cpu_temperature_c=90.0,
    launch_temperature_c=80.0,
    temperature_poll_seconds=30,
    cooldown_seconds=600,
    consecutive_hot_readings_to_stop=2,
)


def get_execution_safety_profile(profile_id: str) -> ExecutionSafetyProfile:
    if profile_id == CPU_SAFE_PROFILE:
        return DEFAULT_PROFILE
    if profile_id == CPU_COOL_PROFILE:
        return COOL_PROFILE
    raise ValueError(f"Unsupported execution safety profile: {profile_id}")


def _selected_affinity(current: list[int], limit: int) -> list[int]:
    if limit < 1:
        raise ValueError("logical_cpu_limit must be positive")
    # Preserve the process's already authorized CPU set while selecting only a
    # small trailing subset.  On the project's hybrid Intel workstation these
    # are the lower-power logical CPUs exposed to the process; on conventional
    # systems the choice remains a deterministic bounded subset.
    return current[-min(limit, len(current)) :]


def apply_process_limits(pid: int, profile: ExecutionSafetyProfile = DEFAULT_PROFILE) -> dict[str, Any]:
    """Apply deterministic OS scheduling limits to one process."""

    process = psutil.Process(pid)
    before_affinity = process.cpu_affinity()
    applied_affinity = _selected_affinity(before_affinity, profile.logical_cpu_limit)
    process.cpu_affinity(applied_affinity)
    if os.name == "nt":
        priority = (
            psutil.IDLE_PRIORITY_CLASS
            if profile.priority == "idle"
            else psutil.BELOW_NORMAL_PRIORITY_CLASS
        )
    else:
        priority = 19 if profile.priority == "idle" else 10
    process.nice(priority)
    return {
        "profile": profile.to_dict(),
        "pid": pid,
        "affinity_before": before_affinity,
        "affinity_applied": process.cpu_affinity(),
        "priority_applied": str(process.nice()),
        "applied_at_unix": time.time(),
    }


def apply_profile_from_environment() -> dict[str, Any] | None:
    """Apply the named profile to the current runner before environments spawn."""

    requested = os.environ.get(PROFILE_ENV)
    if not requested:
        return None
    return apply_process_limits(os.getpid(), get_execution_safety_profile(requested))


def read_cpu_temperature() -> dict[str, Any]:
    """Read a CPU sensor when available; never substitute an ACPI-zone guess."""

    script = ROOT / "scripts" / "read_cpu_temperature.ps1"
    powershell = Path(os.environ.get("WINDIR", r"C:\Windows")) / "System32" / "WindowsPowerShell" / "v1.0" / "powershell.exe"
    if os.name != "nt" or not script.is_file() or not powershell.is_file():
        return {"status": "unavailable", "max_cpu_package_c": None, "message": "CPU sensor reader unavailable."}
    try:
        completed = subprocess.run(
            [str(powershell), "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script)],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
        lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
        if not lines:
            return {"status": "unavailable", "max_cpu_package_c": None, "message": completed.stderr.strip() or "No sensor output."}
        payload = json.loads(lines[-1])
        return payload if isinstance(payload, dict) else {"status": "unavailable", "max_cpu_package_c": None}
    except Exception as error:  # Sensor failure must not corrupt a training attempt.
        return {"status": "unavailable", "max_cpu_package_c": None, "message": str(error)}


def wait_for_launch_temperature(profile: ExecutionSafetyProfile = DEFAULT_PROFILE) -> dict[str, Any]:
    """Wait while a real sensor reports above the launch threshold."""

    while True:
        reading = read_cpu_temperature()
        value = reading.get("max_cpu_package_c")
        if not isinstance(value, (int, float)) or float(value) <= profile.launch_temperature_c:
            return reading
        time.sleep(profile.temperature_poll_seconds)
