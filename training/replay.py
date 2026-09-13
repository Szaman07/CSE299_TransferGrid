"""Deterministic, report-driven Episode Replay V1 services."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from environments.factory import EnvironmentSpec, make_environment
from training.artifact_catalog import (
    TRANSFER_FINAL_CHECKPOINT,
    TRANSFER_INITIALIZATION_CHECKPOINT,
    classify_checkpoint_metadata,
)
from training.checkpoints import checkpoint_sha256, load_model
from training.evaluation import DEFAULT_EVALUATION_REPORT_ROOT
from training.evaluation_v2 import EVALUATION_V2_REPORT_ROOT
from training.portable_paths import resolve_recorded_path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _resolve_checkpoint_path(value: object) -> Path:
    """Resolve live absolute paths or relocate legacy repository paths."""
    candidate = Path(str(value))
    if candidate.is_absolute() and candidate.is_file():
        return candidate.resolve()
    return resolve_recorded_path(str(value), PROJECT_ROOT)


@dataclass(frozen=True, slots=True)
class ReplayEpisodeReference:
    index: int
    seed: int
    success: bool | None
    total_reward: float | None
    episode_length: int | None
    completed: bool
    actions: tuple[int, ...] | None


@dataclass(frozen=True, slots=True)
class ReplayReportReference:
    path: Path
    report_id: str
    created_at: str
    experiment_run_id: str | None
    checkpoint_path: Path
    grid_size: int | None
    training_seed: int | None
    training_mode: str
    requested_timesteps: int | None
    distribution_name: str
    success_count: int | None
    completed_episode_count: int | None
    success_rate: float | None
    episodes: tuple[ReplayEpisodeReference, ...]
    task_id: str
    task_contract_digest: str | None
    checkpoint_sha256: str | None
    checkpoint_kind: str

    @property
    def experiment_key(self) -> str:
        """Group reports only when they evaluate the exact same policy."""
        run = self.experiment_run_id or "legacy"
        return f"{run}|{self.checkpoint_path.resolve()}"

    @property
    def evaluation_family(self) -> str:
        """Separate scratch/legacy policies from transfer-study policies."""
        return (
            "transfer"
            if self.checkpoint_kind in (
                TRANSFER_FINAL_CHECKPOINT,
                TRANSFER_INITIALIZATION_CHECKPOINT,
            )
            else "normal"
        )

    @property
    def checkpoint_stage(self) -> str:
        """Return final versus pre-target-training transfer stage."""
        return (
            "initialization"
            if self.checkpoint_kind == TRANSFER_INITIALIZATION_CHECKPOINT
            else "final"
        )

    @property
    def deterministic_result_key(self) -> tuple[object, ...]:
        """Collapse only identical deterministic episode outcomes."""
        return (
            self.checkpoint_sha256 or str(self.checkpoint_path.resolve()),
            self.distribution_name,
            tuple(
                (
                    episode.seed,
                    episode.success,
                    episode.total_reward,
                    episode.episode_length,
                    episode.completed,
                    episode.actions,
                )
                for episode in self.episodes
            ),
        )


@dataclass(frozen=True, slots=True)
class ReplayRequest:
    report_path: Path
    episode_index: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "report_path", Path(self.report_path))
        if type(self.episode_index) is not int or self.episode_index < 0:
            raise ValueError("episode_index must be a non-negative integer.")


@dataclass(frozen=True, slots=True)
class ReplayStep:
    index: int
    frame: np.ndarray
    action: int | None
    action_name: str
    immediate_reward: float
    cumulative_reward: float
    terminated: bool
    truncated: bool
    success: bool
    failure_reason: str | None


@dataclass(frozen=True, slots=True)
class ReplayTrace:
    experiment_run_id: str | None
    report_id: str
    report_path: Path
    checkpoint_path: Path
    checkpoint_sha256: str
    grid_size: int | None
    training_seed: int | None
    requested_timesteps: int | None
    distribution_name: str
    episode_index: int
    seed: int
    expected_success: bool
    expected_reward: float
    expected_episode_length: int
    training_mode: str
    layout_seed: int | None
    steps: tuple[ReplayStep, ...]
    actions_match_evaluation: bool | None
    outcome_matches_evaluation: bool
    task_id: str
    expected_failure_reason: str | None
    checkpoint_kind: str


def discover_replay_reports(
    report_root: str | Path = DEFAULT_EVALUATION_REPORT_ROOT,
) -> tuple[ReplayReportReference, ...]:
    """Return valid Evaluation V1 reports containing replayable episodes."""
    roots = [Path(report_root)]
    if Path(report_root) == DEFAULT_EVALUATION_REPORT_ROOT:
        roots.append(EVALUATION_V2_REPORT_ROOT)
    references: list[ReplayReportReference] = []
    paths = [
        path
        for root in roots if root.is_dir()
        for path in root.rglob("*.json")
    ]
    for path in sorted(paths, reverse=True):
        try:
            reference = load_replay_report_reference(path)
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
        if reference.checkpoint_path.is_file() and any(
            episode.completed for episode in reference.episodes
        ):
            references.append(reference)
    references.sort(key=lambda reference: reference.created_at, reverse=True)
    return tuple(references)


def load_replay_report_reference(path: str | Path) -> ReplayReportReference:
    """Load lightweight report metadata without loading PPO or an environment."""
    report_path = Path(path).resolve()
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    provenance = _required_mapping(payload, "provenance")
    checkpoint = _required_mapping(provenance, "checkpoint")
    conditions = _required_mapping(payload, "evaluation_conditions")
    distribution = _required_mapping(conditions, "distribution")
    deterministic = conditions.get("deterministic_actions")
    if deterministic is not True:
        raise ValueError("Replay V1 requires deterministic Evaluation V1 actions.")
    records = payload.get("episode_records")
    if not isinstance(records, list):
        raise ValueError("Evaluation report is missing episode_records.")
    training_run = provenance.get("training_run")
    run_id = (
        training_run.get("run_id")
        if isinstance(training_run, dict)
        and isinstance(training_run.get("run_id"), str)
        else None
    )
    training_metadata = checkpoint.get("training_metadata")
    metadata = training_metadata if isinstance(training_metadata, dict) else {}
    environment_metadata = metadata.get("environment")
    environment = (
        environment_metadata if isinstance(environment_metadata, dict) else {}
    )
    condition_environment = conditions.get("environment")
    if isinstance(condition_environment, dict):
        environment = condition_environment
    aggregate_payload = payload.get("aggregate_metrics")
    aggregates = aggregate_payload if isinstance(aggregate_payload, dict) else {}
    return ReplayReportReference(
        path=report_path,
        report_id=str(payload["report_id"]),
        created_at=str(payload.get("created_at", "")),
        experiment_run_id=run_id,
        checkpoint_path=_resolve_checkpoint_path(checkpoint["path"]),
        grid_size=_optional_int(environment.get("grid_size")),
        training_seed=_optional_int(metadata.get("seed")),
        training_mode=str(metadata.get("training_mode", "unknown")),
        requested_timesteps=_optional_int(metadata.get("requested_timesteps")),
        distribution_name=str(distribution["name"]),
        success_count=_optional_int(aggregates.get("success_count")),
        completed_episode_count=_optional_int(
            aggregates.get("completed_episode_count")
        ),
        success_rate=_optional_float(aggregates.get("success_rate")),
        episodes=tuple(
            ReplayEpisodeReference(
                index=index,
                seed=int(record["seed"]),
                success=record.get("success"),
                total_reward=record.get("total_reward"),
                episode_length=record.get("episode_length"),
                completed=record.get("completed") is True,
                actions=(
                    tuple(int(action) for action in record["actions"])
                    if isinstance(record.get("actions"), list)
                    else None
                ),
            )
            for index, record in enumerate(records)
            if isinstance(record, dict)
        ),
        task_id=str(
            environment.get("task_id", "legacy/object_interaction-v0")
        ),
        task_contract_digest=(
            str(environment["task_contract_digest"])
            if isinstance(environment.get("task_contract_digest"), str)
            else None
        ),
        checkpoint_sha256=(
            str(checkpoint["sha256"])
            if isinstance(checkpoint.get("sha256"), str)
            else None
        ),
        checkpoint_kind=classify_checkpoint_metadata(metadata),
    )


def replay_evaluated_episode(request: ReplayRequest) -> ReplayTrace:
    """Regenerate one evaluated episode and capture one frame per state."""
    payload = json.loads(request.report_path.read_text(encoding="utf-8"))
    provenance = _required_mapping(payload, "provenance")
    checkpoint = _required_mapping(provenance, "checkpoint")
    conditions = _required_mapping(payload, "evaluation_conditions")
    environment_payload = _required_mapping(conditions, "environment")
    records = payload.get("episode_records")
    if not isinstance(records, list) or request.episode_index >= len(records):
        raise ValueError("Selected episode does not exist in the evaluation report.")
    record = records[request.episode_index]
    if not isinstance(record, dict) or record.get("completed") is not True:
        raise ValueError("Replay requires a completed evaluation episode.")

    checkpoint_path = _resolve_checkpoint_path(checkpoint["path"])
    expected_hash = checkpoint.get("sha256")
    actual_hash = checkpoint_sha256(checkpoint_path)
    if isinstance(expected_hash, str) and actual_hash != expected_hash:
        raise ValueError("Evaluation checkpoint SHA-256 no longer matches the report.")

    specification = EnvironmentSpec.from_dict(environment_payload)
    environment = make_environment(
        specification,
        render_mode="rgb_array",
        observation_mode="flat",
    )
    try:
        model = load_model(checkpoint_path, environment=environment)
        observation, _ = environment.reset(seed=int(record["seed"]))
        steps = [
            ReplayStep(
                index=0,
                frame=_capture_frame(environment),
                action=None,
                action_name="RESET",
                immediate_reward=0.0,
                cumulative_reward=0.0,
                terminated=False,
                truncated=False,
                success=False,
                failure_reason=None,
            )
        ]
        actions: list[int] = []
        cumulative_reward = 0.0
        terminated = False
        truncated = False
        while not (terminated or truncated):
            action, _ = model.predict(observation, deterministic=True)
            action_value = int(np.asarray(action).item())
            actions.append(action_value)
            observation, reward, terminated, truncated, info = environment.step(
                action_value
            )
            cumulative_reward += float(reward)
            steps.append(
                ReplayStep(
                    index=len(steps),
                    frame=_capture_frame(environment),
                    action=action_value,
                    action_name=_action_name(environment, action_value),
                    immediate_reward=float(reward),
                    cumulative_reward=cumulative_reward,
                    terminated=bool(terminated),
                    truncated=bool(truncated),
                    success=bool(info.get("success", False)),
                    failure_reason=(
                        str(info["failure_reason"])
                        if isinstance(info.get("failure_reason"), str)
                        else None
                    ),
                )
            )
    finally:
        environment.close()

    expected_actions = record.get("actions")
    actions_match = (
        tuple(actions) == tuple(int(action) for action in expected_actions)
        if isinstance(expected_actions, list)
        else None
    )
    outcome_matches = (
        len(actions) == int(record["episode_length"])
        and np.isclose(cumulative_reward, float(record["total_reward"]))
        and steps[-1].success is bool(record["success"])
        and steps[-1].terminated is bool(record["terminated"])
        and steps[-1].truncated is bool(record["truncated"])
        and steps[-1].failure_reason == record.get("failure_reason")
    )
    if actions_match is False or not outcome_matches:
        raise RuntimeError("Replay diverged from its stored evaluation episode record.")

    training_metadata = checkpoint.get("training_metadata")
    run_id = None
    if isinstance(provenance.get("training_run"), dict):
        run_id = provenance["training_run"].get("run_id")
    distribution = _required_mapping(conditions, "distribution")
    return ReplayTrace(
        experiment_run_id=run_id if isinstance(run_id, str) else None,
        report_id=str(payload["report_id"]),
        report_path=request.report_path.resolve(),
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=actual_hash,
        grid_size=_optional_int(environment_payload.get("grid_size")),
        training_seed=(
            _optional_int(training_metadata.get("seed"))
            if isinstance(training_metadata, dict)
            else None
        ),
        requested_timesteps=(
            _optional_int(training_metadata.get("requested_timesteps"))
            if isinstance(training_metadata, dict)
            else None
        ),
        distribution_name=str(distribution["name"]),
        episode_index=request.episode_index,
        seed=int(record["seed"]),
        expected_success=bool(record["success"]),
        expected_reward=float(record["total_reward"]),
        expected_episode_length=int(record["episode_length"]),
        training_mode=(
            str(training_metadata.get("training_mode", "unknown"))
            if isinstance(training_metadata, dict)
            else "unknown"
        ),
        layout_seed=(
            training_metadata.get("layout_seed")
            if isinstance(training_metadata, dict)
            else None
        ),
        steps=tuple(steps),
        actions_match_evaluation=actions_match,
        outcome_matches_evaluation=outcome_matches,
        task_id=specification.resolved_task_id,
        expected_failure_reason=(
            str(record["failure_reason"])
            if isinstance(record.get("failure_reason"), str)
            else None
        ),
        checkpoint_kind=classify_checkpoint_metadata(
            training_metadata
            if isinstance(training_metadata, dict)
            else {}
        ),
    )


def _capture_frame(environment: Any) -> np.ndarray:
    frame = environment.render()
    if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[-1] != 3:
        raise RuntimeError("Replay environment did not produce an RGB frame.")
    copied = np.array(frame, copy=True)
    copied.flags.writeable = False
    return copied


def _action_name(environment: Any, action: int) -> str:
    action_enum = getattr(getattr(environment, "unwrapped", environment), "actions", None)
    try:
        return str(action_enum(action).name).upper()
    except (TypeError, ValueError, AttributeError):
        return f"ACTION {action}"


def _required_mapping(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Evaluation report is missing {key}.")
    return value


def _optional_int(value: object) -> int | None:
    return value if type(value) is int else None


def _optional_float(value: object) -> float | None:
    return float(value) if type(value) in (int, float) else None
