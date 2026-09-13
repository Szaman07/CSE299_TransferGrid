"""Command-line process entry point used by the Train workspace."""

from __future__ import annotations

import argparse
import json
import os
import traceback
from pathlib import Path
from typing import Any

from training.artifacts import write_text_exclusive
from training.execution_safety import apply_profile_from_environment
from training.trainer import TrainingConfig, train_model


def run_training_request(request_path: Path, result_path: Path) -> int:
    """Execute one request while allowing exactly one runner to own its result."""
    result_path.parent.mkdir(parents=True, exist_ok=True)
    invocation_lock = result_path.with_suffix(result_path.suffix + ".running")
    try:
        with invocation_lock.open("x", encoding="utf-8") as lock_file:
            lock_file.write(f"{os.getpid()}\n")
    except FileExistsError as error:
        raise FileExistsError(
            f"Training request already has an active runner: {result_path}"
        ) from error
    try:
        if result_path.exists():
            raise FileExistsError(f"Training result already exists: {result_path}")
        return _execute_training_request(request_path, result_path)
    finally:
        invocation_lock.unlink(missing_ok=True)


def _execute_training_request(request_path: Path, result_path: Path) -> int:
    """Execute one serialized request and publish one immutable result."""
    run_id: str | None = None
    try:
        request = json.loads(request_path.read_text(encoding="utf-8"))
        if not isinstance(request, dict):
            raise ValueError("Training request must contain a JSON object.")
        declared_run_id = request.get("run_id")
        if isinstance(declared_run_id, str):
            run_id = declared_run_id
        config = TrainingConfig.from_dict(request)
        run_id = config.run_id
        print(
            f"[INFO] PPO training run {run_id} started for "
            f"{config.total_timesteps} timesteps."
        )
        summary = train_model(config)
        print(f"[SUCCESS] Checkpoint saved to {summary.checkpoint_path}.")
        payload: dict[str, Any] = {
            "status": "succeeded",
            "run_id": run_id,
            "summary": summary.to_dict(),
        }
        exit_code = 0
    except Exception as error:
        traceback.print_exc()
        payload = {"status": "failed", "run_id": run_id, "error": str(error)}
        exit_code = 1

    write_text_exclusive(
        result_path,
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
    )
    return exit_code


def main() -> int:
    """Parse runner arguments and execute the requested training run."""
    safety = apply_profile_from_environment()
    if safety is not None:
        print(f"[INFO] Execution safety profile applied: {json.dumps(safety, sort_keys=True)}")
    parser = argparse.ArgumentParser(description="Run one TransferGrid PPO job.")
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    arguments = parser.parse_args()
    return run_training_request(arguments.request, arguments.result)


if __name__ == "__main__":
    raise SystemExit(main())
