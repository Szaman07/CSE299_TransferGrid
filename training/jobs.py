"""Minimal subprocess boundary between Streamlit and PPO training."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from training.checkpoints import validate_training_run_id
from training.trainer import TrainingConfig

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_JOB_ROOT = PROJECT_ROOT / "logs" / "training_v0"

JobStatus = Literal["running", "succeeded", "failed"]


@dataclass(frozen=True, slots=True)
class TrainingJob:
    """Serializable metadata needed to observe one background training run."""

    pid: int
    request_path: Path
    result_path: Path
    log_path: Path
    started_at: str
    run_id: str

    def to_dict(self) -> dict[str, Any]:
        """Return session-safe primitives only."""
        return {
            "pid": self.pid,
            "request_path": str(self.request_path),
            "result_path": str(self.result_path),
            "log_path": str(self.log_path),
            "started_at": self.started_at,
            "run_id": self.run_id,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> TrainingJob:
        """Restore job metadata from Streamlit session state."""
        request_path = Path(payload["request_path"])
        legacy_run_id = request_path.name.removesuffix(".request.json")
        return cls(
            pid=int(payload["pid"]),
            request_path=request_path,
            result_path=Path(payload["result_path"]),
            log_path=Path(payload["log_path"]),
            started_at=str(payload["started_at"]),
            run_id=validate_training_run_id(
                str(payload.get("run_id", legacy_run_id))
            ),
        )


@dataclass(frozen=True, slots=True)
class TrainingJobState:
    """Current state read from a background runner's result file."""

    status: JobStatus
    summary: dict[str, Any] | None = None
    error: str | None = None


def start_training_job(
    config: TrainingConfig,
    *,
    job_root: Path = DEFAULT_JOB_ROOT,
) -> TrainingJob:
    """Launch the canonical trainer in a background Python process."""
    root = Path(job_root)
    root.mkdir(parents=True, exist_ok=True)
    job_id = validate_training_run_id(config.run_id or "")
    request_path = root / f"{job_id}.request.json"
    result_path = root / f"{job_id}.result.json"
    log_path = root / f"{job_id}.log"
    for path in (request_path, result_path, log_path):
        if path.exists():
            raise FileExistsError(f"Training job artifact already exists: {path}")
    with request_path.open("x", encoding="utf-8") as request_file:
        request_file.write(
            json.dumps(config.to_dict(), indent=2, sort_keys=True) + "\n"
        )

    command = [
        sys.executable,
        "-m",
        "training.runner",
        "--request",
        str(request_path),
        "--result",
        str(result_path),
    ]
    creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    with log_path.open("x", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            creationflags=creation_flags,
        )
    return TrainingJob(
        pid=process.pid,
        request_path=request_path,
        result_path=result_path,
        log_path=log_path,
        started_at=datetime.now(timezone.utc).isoformat(),
        run_id=job_id,
    )


def read_training_job(job: TrainingJob) -> TrainingJobState:
    """Read a runner result, or report that the job is still in progress."""
    if not job.result_path.is_file():
        if _process_is_running(job.pid):
            return TrainingJobState(status="running")
        return TrainingJobState(
            status="failed",
            error=(
                f"Training process {job.pid} exited without publishing a result "
                f"for run {job.run_id}. See {job.log_path}."
            ),
        )

    try:
        payload = json.loads(job.result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return TrainingJobState(
            status="failed",
            error=f"Could not read training result: {error}",
        )
    if not isinstance(payload, dict):
        return TrainingJobState(
            status="failed",
            error="Training result must contain a JSON object.",
        )

    try:
        request_run_id = _read_request_run_id(job.request_path)
    except ValueError as error:
        return TrainingJobState(
            status="failed",
            error=f"Could not validate training request identity: {error}",
        )
    if request_run_id is not None and request_run_id != job.run_id:
        return TrainingJobState(
            status="failed",
            error=(
                f"Training request run ID {request_run_id!r} does not match "
                f"job {job.run_id!r}."
            ),
        )
    identity_required = request_run_id is not None

    status = payload.get("status")
    result_run_id = payload.get("run_id")
    if (
        identity_required and result_run_id != job.run_id
    ) or (
        result_run_id is not None and result_run_id != job.run_id
    ):
        return TrainingJobState(
            status="failed",
            error=(
                f"Training result run ID {result_run_id!r} does not match "
                f"job {job.run_id!r}."
            ),
        )
    summary = payload.get("summary")
    if status == "succeeded" and isinstance(summary, dict):
        summary_run_id = summary.get("run_id")
        if (
            identity_required and summary_run_id != job.run_id
        ) or (
            summary_run_id is not None and summary_run_id != job.run_id
        ):
            return TrainingJobState(
                status="failed",
                error=(
                    f"Training summary run ID {summary_run_id!r} does not match "
                    f"job {job.run_id!r}."
                ),
            )
        return TrainingJobState(status="succeeded", summary=summary)
    return TrainingJobState(
        status="failed",
        error=str(payload.get("error", "Training runner returned an invalid result.")),
    )


def _read_request_run_id(request_path: Path) -> str | None:
    """Return a declared request identity, preserving legacy requests without one."""
    if not request_path.is_file():
        return None
    try:
        payload = json.loads(request_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"request JSON is unreadable: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError("request must contain a JSON object.")
    if "run_id" not in payload:
        return None
    run_id = payload["run_id"]
    return validate_training_run_id(run_id)


def _process_is_running(pid: int) -> bool:
    """Return whether a local process is still alive without retaining a handle."""
    if pid <= 0:
        return False
    try:
        import psutil

        if not psutil.pid_exists(pid):
            return False
        proc = psutil.Process(pid)
        if not proc.is_running() or proc.status() == psutil.STATUS_ZOMBIE:
            return False
        return "python" in proc.name().lower()
    except Exception:
        pass
    if os.name == "nt":
        return _windows_process_is_running(pid)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True



def _windows_process_is_running(pid: int) -> bool:
    """Query a Windows process exit code through a short-lived native handle."""
    import ctypes
    from ctypes import wintypes

    process_query_limited_information = 0x1000
    still_active = 259
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.argtypes = (
        wintypes.DWORD,
        wintypes.BOOL,
        wintypes.DWORD,
    )
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.GetExitCodeProcess.argtypes = (
        wintypes.HANDLE,
        ctypes.POINTER(wintypes.DWORD),
    )
    kernel32.GetExitCodeProcess.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
    kernel32.CloseHandle.restype = wintypes.BOOL
    handle = kernel32.OpenProcess(
        process_query_limited_information,
        False,
        pid,
    )
    if not handle:
        return ctypes.get_last_error() == 5
    try:
        exit_code = wintypes.DWORD()
        if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
            return True
        return exit_code.value == still_active
    finally:
        kernel32.CloseHandle(handle)
