"""Non-blocking UI-process warmup for the expensive flagship layout support."""

from __future__ import annotations

from threading import Event, Lock, Timer

from environments.lava_door_key_v2 import valid_lava_door_key_v2_layouts

FLAGSHIP_TASK_ID = "lava_door_key-v2"
WARMUP_DELAY_SECONDS = 1.0

_LOCK = Lock()
_READY = Event()
_STARTED = False
_ERROR: BaseException | None = None


def start_environment_support_warmup() -> None:
    """Build immutable flagship support in a daemon thread once per UI process."""
    global _STARTED
    with _LOCK:
        if _STARTED:
            return
        _STARTED = True
        timer = Timer(WARMUP_DELAY_SECONDS, _warm_flagship_support)
        timer.name = "transfergrid-layout-warmup"
        timer.daemon = True
        timer.start()


def wait_for_environment_support(task_id: object) -> None:
    """Avoid duplicate support construction when a flagship action starts early."""
    if task_id != FLAGSHIP_TASK_ID:
        return
    start_environment_support_warmup()
    _READY.wait()
    if _ERROR is not None:
        raise RuntimeError("Flagship environment support warmup failed.") from _ERROR


def _warm_flagship_support() -> None:
    global _ERROR
    try:
        valid_lava_door_key_v2_layouts()
    except BaseException as error:  # surface background initialization failures
        _ERROR = error
    finally:
        _READY.set()
