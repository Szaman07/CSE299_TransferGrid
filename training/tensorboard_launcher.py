"""Minimal local TensorBoard process management for the Streamlit UI."""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import time
import webbrowser
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from environments.catalog import task_display_name
from training.initialization_methods import TRANSFER_INITIALIZATION_METHODS

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TENSORBOARD_HOST = "127.0.0.1"
DEFAULT_TENSORBOARD_PORT = 6006
DEFAULT_TENSORBOARD_LOG_DIRECTORY = PROJECT_ROOT / "logs" / "tensorboard"
DEFAULT_COHORT_LOG_DIRECTORY = PROJECT_ROOT / "reports"
DEFAULT_EXPERIMENT_ROOT = PROJECT_ROOT / "experiments"
DEFAULT_STATE_PATH = PROJECT_ROOT / "logs" / "tensorboard-launcher.json"
DEFAULT_PROCESS_LOG_PATH = PROJECT_ROOT / "logs" / "tensorboard-launcher.log"
DYNAMIC_LOG_ROUTING_MODE = "recursive-multi-root-v3"


@dataclass(frozen=True, slots=True)
class TensorBoardConfig:
    """Validated launcher settings read from the local process configuration."""

    host: str = DEFAULT_TENSORBOARD_HOST
    port: int = DEFAULT_TENSORBOARD_PORT
    log_directory: Path = DEFAULT_TENSORBOARD_LOG_DIRECTORY
    experiment_root: Path = DEFAULT_EXPERIMENT_ROOT
    state_path: Path = DEFAULT_STATE_PATH
    process_log_path: Path = DEFAULT_PROCESS_LOG_PATH

    def __post_init__(self) -> None:
        host = self.host.strip()
        if not host:
            raise ValueError("TensorBoard host cannot be empty.")
        if type(self.port) is not int or not 1 <= self.port <= 65535:
            raise ValueError("TensorBoard port must be an integer from 1 to 65535.")
        object.__setattr__(self, "host", host)
        for name in (
            "log_directory",
            "experiment_root",
            "state_path",
            "process_log_path",
        ):
            path = Path(getattr(self, name))
            if not path.is_absolute():
                path = PROJECT_ROOT / path
            object.__setattr__(self, name, path.resolve())

    @classmethod
    def from_environment(cls) -> TensorBoardConfig:
        """Read optional launcher overrides from TransferGrid environment variables."""
        raw_port = os.environ.get(
            "TRANSFERGRID_TENSORBOARD_PORT",
            str(DEFAULT_TENSORBOARD_PORT),
        )
        try:
            port = int(raw_port)
        except ValueError as error:
            raise ValueError(
                "TRANSFERGRID_TENSORBOARD_PORT must be an integer."
            ) from error
        return cls(
            host=os.environ.get(
                "TRANSFERGRID_TENSORBOARD_HOST",
                DEFAULT_TENSORBOARD_HOST,
            ),
            port=port,
            log_directory=Path(
                os.environ.get(
                    "TRANSFERGRID_TENSORBOARD_LOG_DIR",
                    DEFAULT_TENSORBOARD_LOG_DIRECTORY,
                )
            ),
            experiment_root=Path(
                os.environ.get(
                    "TRANSFERGRID_EXPERIMENT_ROOT",
                    DEFAULT_EXPERIMENT_ROOT,
                )
            ),
        )

    @property
    def browser_url(self) -> str:
        """Return the browser-safe TensorBoard server root."""
        browser_host = "127.0.0.1" if self.host in {"0.0.0.0", "::"} else self.host
        return f"http://{browser_host}:{self.port}"

    @property
    def dashboard_url(self) -> str:
        """Open the populated Scalars dashboard instead of the empty Time Series tab."""
        return f"{self.browser_url}/#scalars"

    @property
    def log_directories(self) -> tuple[Path, ...]:
        """Return stable roots that may contain TensorBoard event files."""
        roots = [self.log_directory]
        if self.log_directory == DEFAULT_TENSORBOARD_LOG_DIRECTORY.resolve():
            roots.append(DEFAULT_COHORT_LOG_DIRECTORY.resolve())
        return tuple(dict.fromkeys(roots))


@dataclass(frozen=True, slots=True)
class TensorBoardStatus:
    """Serializable status for the configured local TensorBoard endpoint."""

    running: bool
    owned: bool
    config: TensorBoardConfig
    pid: int | None = None
    started_at: str | None = None

    @property
    def label(self) -> str:
        return "Running" if self.running else "Not Running"


@dataclass(frozen=True, slots=True)
class TensorBoardRun:
    """One event directory and its manifest-derived presentation label."""

    label: str
    path: Path
    run_id: str | None = None


def discover_tensorboard_runs(
    config: TensorBoardConfig | None = None,
) -> tuple[TensorBoardRun, ...]:
    """Return all event directories with readable, collision-safe labels."""
    config = config or TensorBoardConfig.from_environment()
    manifest_runs = _manifest_run_labels(config)
    event_directories = {
        event_file.parent.resolve()
        for root in config.log_directories if root.is_dir()
        for event_file in root.rglob("events.out.tfevents.*")
    }
    runs: list[TensorBoardRun] = []
    for path in sorted(event_directories, key=lambda item: str(item).lower()):
        manifest_run = manifest_runs.get(path)
        if manifest_run is not None:
            runs.append(manifest_run)
            continue
        runs.append(
            TensorBoardRun(
                label=_fallback_run_label(path, config),
                path=path,
            )
        )
    return tuple(runs)


def get_tensorboard_status(
    config: TensorBoardConfig | None = None,
) -> TensorBoardStatus:
    """Refresh status from durable ownership metadata and the configured port."""
    config = config or TensorBoardConfig.from_environment()
    state = _read_state(config.state_path)
    if state is not None:
        pid = state.get("pid")
        if type(pid) is int and _process_is_running(pid):
            return TensorBoardStatus(
                running=True,
                owned=True,
                config=config,
                pid=pid,
                started_at=_optional_string(state.get("started_at")),
            )
        config.state_path.unlink(missing_ok=True)

    if _port_is_open(config.host, config.port):
        return TensorBoardStatus(running=True, owned=False, config=config)
    return TensorBoardStatus(running=False, owned=False, config=config)


def launch_tensorboard(
    config: TensorBoardConfig | None = None,
    *,
    readiness_timeout: float = 10.0,
) -> TensorBoardStatus:
    """Launch one owned TensorBoard process or return the detected instance."""
    config = config or TensorBoardConfig.from_environment()
    status = get_tensorboard_status(config)
    if status.running:
        return status

    config.log_directory.mkdir(parents=True, exist_ok=True)
    config.experiment_root.mkdir(parents=True, exist_ok=True)
    config.state_path.parent.mkdir(parents=True, exist_ok=True)
    launch_lock = config.state_path.with_suffix(config.state_path.suffix + ".launching")
    try:
        with launch_lock.open("x", encoding="utf-8") as lock_file:
            lock_file.write(f"{os.getpid()}\n")
    except FileExistsError as error:
        raise RuntimeError("A TensorBoard launch is already in progress.") from error

    process: subprocess.Popen[Any] | None = None
    try:
        status = get_tensorboard_status(config)
        if status.running:
            return status
        command = [
            sys.executable,
            "-m",
            "tensorboard.main",
            "--host",
            config.host,
            "--port",
            str(config.port),
        ]
        # A per-run logdir_spec is a launch-time snapshot: TensorBoard cannot
        # discover directories created after the process starts. Pointing it at
        # the stable root preserves every legacy run and lets its normal reload
        # loop discover scratch and transfer runs created later.
        runs = discover_tensorboard_runs(config)
        command.extend([
            *_tensorboard_log_arguments(config),
            "--reload_interval",
            "30" if len(config.log_directories) > 1 else "1",
        ])
        creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        with config.process_log_path.open("a", encoding="utf-8") as process_log:
            process = subprocess.Popen(
                command,
                cwd=PROJECT_ROOT,
                stdout=process_log,
                stderr=subprocess.STDOUT,
                creationflags=creation_flags,
            )
        started_at = datetime.now(timezone.utc).isoformat()
        _write_state(
            config.state_path,
            {
                "pid": process.pid,
                "started_at": started_at,
                "host": config.host,
                "port": config.port,
                "log_directory": str(config.log_directory),
                "log_directories": [str(path) for path in config.log_directories],
                "experiment_root": str(config.experiment_root),
                "run_alias_count": len(runs),
                "routing_mode": DYNAMIC_LOG_ROUTING_MODE,
            },
        )
        deadline = time.monotonic() + readiness_timeout
        while time.monotonic() < deadline:
            if _port_is_open(config.host, config.port):
                return TensorBoardStatus(
                    running=True,
                    owned=True,
                    config=config,
                    pid=process.pid,
                    started_at=started_at,
                )
            if process.poll() is not None:
                raise RuntimeError(
                    "TensorBoard exited during launch. "
                    f"See {config.process_log_path}."
                )
            time.sleep(0.1)
        raise RuntimeError(
            f"TensorBoard did not become ready within {readiness_timeout:g} seconds."
        )
    except Exception:
        if process is not None and process.poll() is None:
            process.terminate()
        config.state_path.unlink(missing_ok=True)
        raise
    finally:
        launch_lock.unlink(missing_ok=True)


def stop_tensorboard(
    config: TensorBoardConfig | None = None,
    *,
    timeout: float = 5.0,
) -> TensorBoardStatus:
    """Stop the owned server without terminating an externally managed process."""
    config = config or TensorBoardConfig.from_environment()
    status = get_tensorboard_status(config)
    if not status.running:
        return status
    if not status.owned or status.pid is None:
        raise PermissionError(
            "The configured port is in use by a process TransferGrid does not own."
        )

    _terminate_process(status.pid)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        process_running = _process_is_running(status.pid)
        port_open = _port_is_open(config.host, config.port)
        if not process_running and not port_open:
            break
        time.sleep(0.05)
    if _process_is_running(status.pid):
        raise RuntimeError(f"TensorBoard process {status.pid} did not stop.")
    if _port_is_open(config.host, config.port):
        raise RuntimeError(
            f"TensorBoard stopped but port {config.port} was not released."
        )
    config.state_path.unlink(missing_ok=True)
    return get_tensorboard_status(config)


def open_tensorboard_browser(
    config: TensorBoardConfig | None = None,
    *,
    opener: Callable[[str], bool] | None = None,
) -> str:
    """Open the configured running endpoint in the default browser."""
    config = config or TensorBoardConfig.from_environment()
    status = get_tensorboard_status(config)
    if not status.running:
        raise RuntimeError("TensorBoard is not running.")
    url = config.dashboard_url
    opened = (opener or webbrowser.open)(url)
    if opened is False:
        raise RuntimeError(
            "The operating system did not accept the browser-open request."
        )
    return url


def _read_state(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        path.unlink(missing_ok=True)
        return None
    return payload if isinstance(payload, dict) else None


def _write_state(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _port_is_open(host: str, port: int) -> bool:
    probe_host = "127.0.0.1" if host in {"0.0.0.0", "::"} else host
    try:
        with socket.create_connection((probe_host, port), timeout=0.2):
            return True
    except OSError:
        return False


def _process_is_running(pid: int) -> bool:
    if pid <= 0:
        return False
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
        if not kernel32.GetExitCodeProcess(
            handle,
            ctypes.byref(exit_code),
        ):
            return True
        return exit_code.value == still_active
    finally:
        kernel32.CloseHandle(handle)


def _terminate_process(pid: int) -> None:
    if os.name != "nt":
        os.kill(pid, signal.SIGTERM)
        return

    import ctypes
    from ctypes import wintypes

    process_terminate = 0x0001
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.argtypes = (
        wintypes.DWORD,
        wintypes.BOOL,
        wintypes.DWORD,
    )
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.TerminateProcess.argtypes = (wintypes.HANDLE, wintypes.UINT)
    kernel32.TerminateProcess.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
    kernel32.CloseHandle.restype = wintypes.BOOL
    handle = kernel32.OpenProcess(process_terminate, False, pid)
    if not handle:
        raise RuntimeError(f"Could not open TensorBoard process {pid}.")
    try:
        if not kernel32.TerminateProcess(handle, 0):
            raise RuntimeError(f"Could not terminate TensorBoard process {pid}.")
    finally:
        kernel32.CloseHandle(handle)


def _optional_string(value: Any) -> str | None:
    return value if isinstance(value, str) else None


def _manifest_run_labels(
    config: TensorBoardConfig,
) -> dict[Path, TensorBoardRun]:
    runs: dict[Path, TensorBoardRun] = {}
    manifest_paths = list(config.experiment_root.glob("*/manifest.json"))
    if config.experiment_root == DEFAULT_EXPERIMENT_ROOT.resolve():
        manifest_paths.extend(DEFAULT_COHORT_LOG_DIRECTORY.rglob("manifest.json"))
    for manifest_path in sorted(set(manifest_paths)):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            configuration_path = Path(manifest["configuration"]["json"])
            configuration_payload = json.loads(
                configuration_path.read_text(encoding="utf-8")
            )
            training_config = configuration_payload["configuration"]
            run_id = str(manifest["run_id"])
            tensorboard_path = Path(manifest["tensorboard_path"]).resolve()
            if not any(
                tensorboard_path.is_relative_to(root)
                for root in config.log_directories
            ):
                continue
            label = _format_run_label(training_config, run_id)
        except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
            continue
        runs[tensorboard_path] = TensorBoardRun(
            label=label,
            path=tensorboard_path,
            run_id=run_id,
        )
    return runs


def _tensorboard_log_arguments(config: TensorBoardConfig) -> list[str]:
    roots = tuple(path for path in config.log_directories if path.is_dir())
    if not roots:
        roots = (config.log_directory,)
    if len(roots) == 1:
        return ["--logdir", str(roots[0])]
    aliases = ("workspace", "cohorts")
    spec = ",".join(
        f"{aliases[index] if index < len(aliases) else f'root{index + 1}'}:{path.as_posix()}"
        for index, path in enumerate(roots)
    )
    return ["--logdir_spec", spec]


def _fallback_run_label(path: Path, config: TensorBoardConfig) -> str:
    for root in config.log_directories:
        if path.is_relative_to(root):
            relative = path.relative_to(root)
            prefix = "COHORT" if root == DEFAULT_COHORT_LOG_DIRECTORY.resolve() else "LEGACY"
            return _safe_run_label(f"{prefix} | {relative}")
    return _safe_run_label(f"LEGACY | {path.name}")


def _format_run_label(configuration: dict[str, Any], run_id: str) -> str:
    environment = configuration["environment"]
    grid_size = int(environment["grid_size"])
    timesteps = int(configuration["total_timesteps"])
    seed = int(configuration["seed"])
    initialization = configuration.get("initialization")
    initialization = initialization if isinstance(initialization, dict) else {}
    target_task_id = environment.get("task_id")
    source_task_id = initialization.get("source_task_id")
    is_transfer = (
        initialization.get("method") in TRANSFER_INITIALIZATION_METHODS
    )

    if is_transfer:
        role_fields = ["TRANSFER"]
        relation = _format_transfer_relation(source_task_id, target_task_id)
        if relation is not None:
            role_fields.append(relation)
    else:
        role_fields = ["NORMAL"]
        target_label = _optional_task_label(target_task_id)
        if target_label is not None:
            role_fields.append(target_label)

    fields = role_fields + [
        f"Grid {grid_size}",
        _format_timestep_count(timesteps),
        f"Seed {seed}",
    ]
    if configuration.get("training_mode") == "fixed_layout":
        fields.append(f"Fixed {int(configuration['layout_seed'])}")
    fields.append(run_id[-8:])
    return _safe_run_label(" | ".join(fields))


def _format_transfer_relation(
    source_task_id: Any,
    target_task_id: Any,
) -> str | None:
    source_label = _optional_task_label(source_task_id)
    target_label = _optional_task_label(target_task_id)
    if source_label is not None and target_label is not None:
        return f"{source_label} -> {target_label}"
    if target_label is not None:
        return f"-> {target_label}"
    if source_label is not None:
        return f"{source_label} -> target"
    return None


def _optional_task_label(task_id: Any) -> str | None:
    if not isinstance(task_id, str) or not task_id.strip():
        return None
    try:
        return task_display_name(task_id)
    except ValueError:
        return task_id.strip()


def _format_timestep_count(value: int) -> str:
    if value >= 1_000_000 and value % 1_000_000 == 0:
        return f"{value // 1_000_000}M"
    if value >= 1_000 and value % 1_000 == 0:
        return f"{value // 1_000}K"
    return f"{value:,}"


def _safe_run_label(value: str) -> str:
    return value.replace(",", " ").replace(":", " ").strip()
