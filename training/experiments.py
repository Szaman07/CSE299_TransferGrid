"""Immutable, run-centered experiment artifacts for downstream consumers."""

from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from training.artifacts import write_text_exclusive
from training.checkpoints import validate_training_run_id

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPERIMENT_ROOT = PROJECT_ROOT / "experiments"
EXPERIMENT_SCHEMA_VERSION = 1
GOVERNED_EXPERIMENT_SCHEMA_VERSION = 2


@dataclass(frozen=True, slots=True)
class ExperimentPaths:
    root: Path
    configuration_json: Path
    configuration_csv: Path
    training_summary_json: Path
    training_summary_csv: Path
    artifact_table_csv: Path
    manifest_json: Path
    monitor_directory: Path
    evaluation_directory: Path


def build_experiment_paths(
    run_id: str,
    experiment_root: str | Path = DEFAULT_EXPERIMENT_ROOT,
) -> ExperimentPaths:
    """Derive every run-centered path from the one canonical run ID."""
    run_id = validate_training_run_id(run_id)
    root = Path(experiment_root) / run_id
    return ExperimentPaths(
        root=root,
        configuration_json=root / "configuration.json",
        configuration_csv=root / "configuration.csv",
        training_summary_json=root / "training_summary.json",
        training_summary_csv=root / "training_summary.csv",
        artifact_table_csv=root / "artifacts.csv",
        manifest_json=root / "manifest.json",
        monitor_directory=root / "monitor",
        evaluation_directory=root / "evaluations",
    )


def reserve_experiment_directory(paths: ExperimentPaths) -> None:
    """Exclusively claim the public experiment directory."""
    paths.root.parent.mkdir(parents=True, exist_ok=True)
    paths.root.mkdir(exist_ok=False)
    paths.monitor_directory.mkdir()


def save_experiment_configuration(
    paths: ExperimentPaths,
    *,
    run_id: str,
    created_at: str,
    configuration: Mapping[str, Any],
    runtime_context: Mapping[str, Any],
    source_revision: Mapping[str, Any],
    schema_version: int = EXPERIMENT_SCHEMA_VERSION,
) -> None:
    """Publish immutable JSON and one-row CSV configuration snapshots."""
    payload = {
        "schema_version": schema_version,
        "artifact_type": "transfergrid-experiment-configuration",
        "run_id": run_id,
        "created_at": created_at,
        "configuration": dict(configuration),
        "runtime_context": dict(runtime_context),
        "source_revision": dict(source_revision),
    }
    write_text_exclusive(
        paths.configuration_json,
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
    )
    environment = configuration["environment"]
    environment_row = (
        {
            "task_id": environment["task_id"],
            "task_contract_digest": environment["task_contract_digest"],
            "grid_size": environment["grid_size"],
        }
        if "task_id" in environment
        else {
            "capability": environment["capability"],
            "grid_size": environment["grid_size"],
            "difficulty": environment["difficulty"],
        }
    )
    save_csv_rows(
        paths.configuration_csv,
        (
            {
                "run_id": run_id,
                "created_at": created_at,
                **environment_row,
                "training_mode": configuration["training_mode"],
                "training_seed": configuration["seed"],
                "layout_seed": configuration["layout_seed"],
                "requested_timesteps": configuration["total_timesteps"],
                "learning_rate": configuration["learning_rate"],
            },
        ),
    )


def save_completed_experiment(
    paths: ExperimentPaths,
    *,
    run_id: str,
    summary: Mapping[str, Any],
    manifest: Mapping[str, Any],
    artifacts: Iterable[Mapping[str, Any]],
    schema_version: int = EXPERIMENT_SCHEMA_VERSION,
) -> None:
    """Publish terminal summary/tables, then the manifest as commit marker."""
    paths.evaluation_directory.mkdir(exist_ok=False)
    summary_payload = {
        "schema_version": schema_version,
        "artifact_type": "transfergrid-experiment-training-summary",
        "run_id": run_id,
        **dict(summary),
    }
    write_text_exclusive(
        paths.training_summary_json,
        json.dumps(summary_payload, indent=2, sort_keys=True) + "\n",
    )
    save_csv_rows(paths.training_summary_csv, (_flatten_summary(summary_payload),))
    save_csv_rows(paths.artifact_table_csv, artifacts)
    manifest_payload = {
        "schema_version": schema_version,
        "artifact_type": "transfergrid-experiment-manifest",
        "run_id": run_id,
        **dict(manifest),
    }
    write_text_exclusive(
        paths.manifest_json,
        json.dumps(manifest_payload, indent=2, sort_keys=True) + "\n",
    )


def save_csv_rows(
    destination: Path,
    rows: Iterable[Mapping[str, Any]],
) -> Path:
    """Publish a rectangular UTF-8 CSV table without overwriting."""
    materialized = [dict(row) for row in rows]
    if not materialized:
        raise ValueError("CSV artifact requires at least one row.")
    fieldnames = tuple(materialized[0])
    if any(tuple(row) != fieldnames for row in materialized):
        raise ValueError("CSV artifact rows must share one ordered schema.")
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(materialized)
    return write_text_exclusive(destination, stream.getvalue())


def _flatten_summary(payload: Mapping[str, Any]) -> dict[str, Any]:
    metrics = payload["training_metrics"]
    return {
        "run_id": payload["run_id"],
        "status": payload["status"],
        "started_at": payload["started_at"],
        "finished_at": payload["finished_at"],
        "requested_timesteps": payload["requested_timesteps"],
        "completed_timesteps": payload["completed_timesteps"],
        "elapsed_seconds": payload["elapsed_seconds"],
        "episode_count": metrics["episode_count"],
        "success_count": metrics["success_count"],
        "failure_count": metrics["failure_count"],
        "timeout_count": metrics["timeout_count"],
        "success_rate": metrics["success_rate"],
        "checkpoint_path": payload["checkpoint_path"],
        "checkpoint_sha256": payload["checkpoint_sha256"],
        "tensorboard_path": payload["tensorboard_path"],
    }
