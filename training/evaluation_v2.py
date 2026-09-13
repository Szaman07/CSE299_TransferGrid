"""Task-scoped deterministic Evaluation V2 for governed benchmarks."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean
from typing import Any, Literal

import numpy as np

from environments.catalog import require_released_task
from environments.factory import EnvironmentSpec, make_environment
from utils.wrappers import stable_minigrid_layout_signature
from training.artifacts import write_text_exclusive
from training.benchmark_contracts import DEVELOPMENT_SEEDS, VALIDATION_SEEDS
from training.checkpoints import checkpoint_sha256, load_checkpoint_metadata, load_model
from training.evaluation_identity import (
    build_evaluation_v2_comparison_condition,
    evaluation_v2_comparison_condition_digest,
)

EVALUATION_V2_PROTOCOL = "evaluation-v2"
EVALUATION_V2_REPORT_SCHEMA = 3
EVALUATION_V2_DISTRIBUTION_VERSION = 1
EVALUATION_V2_REPORT_ROOT = (
    Path(__file__).resolve().parents[1] / "reports" / "evaluations_v2"
)
INTERPRETATION_BOUNDARY = (
    "This deterministic report supports claims only about the exact governed task, "
    "checkpoint, contract digest, and ordered seed bank recorded here. It is not "
    "transfer, final-test, or statistical-significance evidence."
)

EvidenceRole = Literal["development", "validation"]


@dataclass(frozen=True, slots=True)
class EvaluationV2Config:
    checkpoint_path: Path
    role: EvidenceRole = "development"

    def __post_init__(self) -> None:
        object.__setattr__(self, "checkpoint_path", Path(self.checkpoint_path))
        if self.role not in ("development", "validation"):
            raise ValueError("Evaluation V2 role must be development or validation.")


def _distribution(task_id: str, role: EvidenceRole) -> tuple[str, tuple[int, ...]]:
    return (
        (f"{task_id}-development", DEVELOPMENT_SEEDS)
        if role == "development"
        else (f"{task_id}-validation", VALIDATION_SEEDS)
    )


def _seed_digest(seeds: tuple[int, ...]) -> str:
    canonical = json.dumps(list(seeds), separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def _layout_digest(environment: Any, direction: int) -> str:
    payload = {
        "layout": stable_minigrid_layout_signature(environment),
        "direction": direction,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def run_evaluation_v2(config: EvaluationV2Config) -> dict[str, Any]:
    """Run the canonical governed seed bank and return a complete V2 report."""
    checkpoint_path = config.checkpoint_path
    metadata = load_checkpoint_metadata(checkpoint_path)
    if metadata.get("schema_version") != 3:
        raise ValueError("Evaluation V2 requires governed checkpoint metadata schema 3.")
    environment_payload = metadata.get("environment")
    if not isinstance(environment_payload, dict):
        raise ValueError("Checkpoint metadata has no governed environment payload.")
    specification = EnvironmentSpec.from_dict(environment_payload)
    definition = require_released_task(specification.task_id)
    if metadata.get("task_contract_digest") != specification.task_contract_digest:
        raise ValueError("Checkpoint/task contract mismatch.")
    expected_hash = metadata.get("checkpoint_sha256")
    actual_hash = checkpoint_sha256(checkpoint_path)
    if expected_hash != actual_hash:
        raise ValueError("Checkpoint provenance is corrupt or incomplete.")

    distribution_id, seeds = _distribution(specification.task_id, config.role)
    environment = make_environment(specification, observation_mode="flat")
    maximum_steps = int(environment.unwrapped.max_steps)
    records: list[dict[str, Any]] = []
    try:
        model = load_model(checkpoint_path, environment=environment)
        for seed in seeds:
            try:
                observation, _ = environment.reset(seed=seed)
                direction = int(environment.unwrapped.agent_dir)
                digest = _layout_digest(environment, direction)
                total_reward = 0.0
                actions: list[int] = []
                terminated = truncated = False
                final_info: dict[str, Any] = {}
                while not (terminated or truncated):
                    action, _ = model.predict(observation, deterministic=True)
                    action_value = int(np.asarray(action).item())
                    actions.append(action_value)
                    observation, reward, terminated, truncated, info = environment.step(
                        action_value
                    )
                    total_reward += float(reward)
                    final_info = dict(info)
                records.append({
                    "seed": seed,
                    "layout_digest": digest,
                    "total_reward": total_reward,
                    "episode_length": len(actions),
                    "success": bool(final_info.get("success", False)),
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    "failure_reason": final_info.get("failure_reason"),
                    "actions": actions,
                    "completed": True,
                    "evaluator_error": None,
                })
            except Exception as error:
                records.append({
                    "seed": seed,
                    "layout_digest": None,
                    "total_reward": None,
                    "episode_length": None,
                    "success": None,
                    "terminated": None,
                    "truncated": None,
                    "failure_reason": None,
                    "actions": None,
                    "completed": False,
                    "evaluator_error": f"{type(error).__name__}: {error}",
                })
    finally:
        environment.close()

    complete = [record for record in records if record["completed"]]
    successes = sum(record["success"] is True for record in complete)
    created_at = datetime.now(timezone.utc).isoformat()
    seed_digest = _seed_digest(seeds)
    report_id = (
        f"evaluation-v2_{config.role}_"
        f"{created_at.replace(':', '').replace('-', '').replace('+00:00', 'Z')}_"
        f"{actual_hash[:12]}_{seed_digest[:8]}"
    )
    comparison_condition = build_evaluation_v2_comparison_condition(
        protocol_version=EVALUATION_V2_PROTOCOL,
        task_id=specification.task_id,
        task_contract_digest=specification.task_contract_digest,
        grid_size=specification.grid_size,
        maximum_steps=maximum_steps,
        observation_contract_id=definition.observation_contract_id,
        action_contract_id=definition.action_contract_id,
        evidence_role=config.role,
        distribution_id=distribution_id,
        distribution_version=EVALUATION_V2_DISTRIBUTION_VERSION,
        ordered_seed_digest=seed_digest,
        deterministic_actions=True,
    )
    comparison_condition_digest = evaluation_v2_comparison_condition_digest(
        comparison_condition
    )
    return {
        "schema_version": EVALUATION_V2_REPORT_SCHEMA,
        "artifact_type": "transfergrid-evaluation-report",
        "protocol_version": EVALUATION_V2_PROTOCOL,
        "report_id": report_id,
        "created_at": created_at,
        "provenance": {
            "checkpoint": {
                "path": str(checkpoint_path),
                "sha256": actual_hash,
                "training_metadata": metadata,
            },
            "training_run": {
                "run_id": metadata.get("run_id"),
                "manifest_path": (metadata.get("experiment") or {}).get("manifest_path"),
                "identity_source": "checkpoint-metadata-schema-3",
            },
        },
        "evaluation_identity": {
            "task_id": specification.task_id,
            "task_contract_digest": specification.task_contract_digest,
            "grid_size": specification.grid_size,
            "checkpoint_sha256": actual_hash,
            "evidence_role": config.role,
            "distribution_id": distribution_id,
            "distribution_version": EVALUATION_V2_DISTRIBUTION_VERSION,
            "ordered_seed_digest": seed_digest,
            "deterministic_actions": True,
            "maximum_steps": maximum_steps,
            "observation_contract_id": definition.observation_contract_id,
            "action_contract_id": definition.action_contract_id,
            "comparison_condition_digest": comparison_condition_digest,
        },
        "evaluation_conditions": {
            "environment": specification.to_dict(),
            "distribution": {
                "name": distribution_id,
                "role": config.role,
                "version": EVALUATION_V2_DISTRIBUTION_VERSION,
                "seeds": list(seeds),
                "episode_count": len(seeds),
                "condition_digest": seed_digest,
            },
            "deterministic_actions": True,
        },
        "episode_records": records,
        "aggregate_metrics": {
            "requested_episode_count": len(seeds),
            "completed_episode_count": len(complete),
            "error_count": len(records) - len(complete),
            "completion_rate": len(complete) / len(seeds),
            "report_complete": len(complete) == len(seeds),
            "success_count": successes,
            "success_rate": successes / len(complete) if complete else None,
            "mean_reward": fmean(record["total_reward"] for record in complete) if complete else None,
            "mean_episode_length": fmean(record["episode_length"] for record in complete) if complete else None,
            "terminated_count": sum(record["terminated"] is True for record in complete),
            "truncated_count": sum(record["truncated"] is True for record in complete),
        },
        "interpretation_boundary": INTERPRETATION_BOUNDARY,
    }


def save_evaluation_v2_report(
    report: dict[str, Any],
    report_root: str | Path = EVALUATION_V2_REPORT_ROOT,
) -> Path:
    """Persist one immutable report beneath its task namespace."""
    identity = report["evaluation_identity"]
    destination = (
        Path(report_root)
        / str(identity["task_id"])
        / f"{report['report_id']}.json"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    return write_text_exclusive(
        destination, json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
