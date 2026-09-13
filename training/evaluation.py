"""Evaluation services for TransferGrid checkpoints.

``evaluate_model`` preserves the original rendered, single-episode API used by
Training v0.  Evaluation V1 builds on the same factory and checkpoint services
to produce a reproducible, multi-episode evidence report.
"""

from __future__ import annotations

import hashlib
import json
import platform
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from statistics import fmean, pstdev
from types import MappingProxyType
from typing import Any, NamedTuple

import numpy as np

from environments.factory import EnvironmentSpec, make_environment
from environments.generation import DEFAULT_SEED, MAX_SEED
from training.artifacts import publish_new_file, write_text_exclusive
from training.artifact_catalog import scan_checkpoint_artifacts
from training.checkpoints import (
    DEFAULT_CHECKPOINT_ROOT,
    checkpoint_sha256 as calculate_checkpoint_sha256,
    load_checkpoint_metadata,
    load_model,
)
from training.provenance import capture_source_revision

EVALUATION_PROTOCOL_VERSION = "evaluation-v1.2"
DEFAULT_EVALUATION_DISTRIBUTION_NAME = "object-interaction-v1-development"
DEFAULT_EVALUATION_DISTRIBUTION_ROLE = "development"
DEFAULT_EVALUATION_SEEDS: tuple[int, ...] = tuple(range(1001, 1011))
VALIDATION_EVALUATION_DISTRIBUTION_NAME = "object-interaction-v1-validation"
VALIDATION_EVALUATION_SEEDS: tuple[int, ...] = tuple(range(1011, 1051))
FINAL_TEST_EVALUATION_DISTRIBUTION_NAME = "object-interaction-v1-final-test"
FINAL_TEST_EVALUATION_SEEDS: tuple[int, ...] = tuple(range(50001, 50051))
MAX_EVALUATION_EPISODE_COUNT = 100
DEFAULT_EVALUATION_REPORT_ROOT = (
    Path(__file__).resolve().parents[1] / "reports" / "evaluations"
)
REPRODUCIBILITY_STATEMENT = (
    "This report fixes the checkpoint content, checkpoint metadata, environment "
    "contract, distribution seeds, and deterministic action mode. Equivalent "
    "runtime and dependency versions are required to reproduce its outcomes."
)
INTERPRETATION_BOUNDARY = (
    "This report supports a claim only about this checkpoint's deterministic "
    "performance on the declared TransferGrid evaluation distribution. It does "
    "not establish general intelligence, transfer performance, statistical "
    "significance, or performance on undisclosed layouts."
)


class EvaluationDistributionDefinition(NamedTuple):
    """Immutable governance metadata for one canonical seed distribution."""

    role: str
    purpose: str
    owner: str
    seeds: tuple[int, ...]


CANONICAL_EVALUATION_DISTRIBUTIONS = MappingProxyType(
    {
        DEFAULT_EVALUATION_DISTRIBUTION_NAME: EvaluationDistributionDefinition(
            role=DEFAULT_EVALUATION_DISTRIBUTION_ROLE,
            purpose=(
                "Reusable engineering diagnostics; results may guide implementation "
                "and must not be presented as final held-out evidence."
            ),
            owner="TransferGrid development workflow",
            seeds=DEFAULT_EVALUATION_SEEDS,
        ),
        VALIDATION_EVALUATION_DISTRIBUTION_NAME: EvaluationDistributionDefinition(
            role="validation",
            purpose=(
                "Reusable checkpoint comparison and reporting evidence; this "
                "historically inspected set is not a final held-out test."
            ),
            owner="TransferGrid evaluation maintainer",
            seeds=VALIDATION_EVALUATION_SEEDS,
        ),
        FINAL_TEST_EVALUATION_DISTRIBUTION_NAME: EvaluationDistributionDefinition(
            role="final-test",
            purpose=(
                "One-time generalization evidence for a checkpoint frozen before "
                "these seeds are inspected."
            ),
            owner="TransferGrid project owner",
            seeds=FINAL_TEST_EVALUATION_SEEDS,
        ),
    }
)


@dataclass(frozen=True, slots=True)
class EvaluationConfig:
    """Configuration for the original deterministic one-episode evaluation."""

    checkpoint_path: Path
    seed: int = DEFAULT_SEED
    environment: EnvironmentSpec | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "checkpoint_path",
            _normalize_checkpoint_path(self.checkpoint_path),
        )
        _validate_seed(self.seed)


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    """Structured output from one deterministic, rendered evaluation episode."""

    checkpoint_path: Path
    environment: EnvironmentSpec
    seed: int
    total_reward: float
    episode_length: int
    terminated: bool
    truncated: bool
    success: bool
    final_frame: np.ndarray

    def to_dict(self) -> dict[str, Any]:
        """Return lightweight result metadata while excluding frame pixels."""
        return {
            "checkpoint_path": str(self.checkpoint_path),
            "environment": self.environment.to_dict(),
            "seed": self.seed,
            "total_reward": self.total_reward,
            "episode_length": self.episode_length,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "success": self.success,
            "final_frame_shape": list(self.final_frame.shape),
        }


@dataclass(frozen=True, slots=True)
class EvaluationDistribution:
    """Frozen procedural layout conditions for one Evaluation V1 execution."""

    name: str = DEFAULT_EVALUATION_DISTRIBUTION_NAME
    seeds: tuple[int, ...] = DEFAULT_EVALUATION_SEEDS
    role: str = DEFAULT_EVALUATION_DISTRIBUTION_ROLE
    purpose: str = ""
    owner: str = ""

    def __post_init__(self) -> None:
        normalized_name = self.name.strip()
        normalized_role = self.role.strip()
        normalized_purpose = self.purpose.strip()
        normalized_owner = self.owner.strip()
        normalized_seeds = tuple(self.seeds)
        if not normalized_name:
            raise ValueError("evaluation distribution name must not be empty.")
        if not normalized_role:
            raise ValueError("evaluation distribution role must not be empty.")
        if not normalized_seeds:
            raise ValueError("evaluation distribution must declare at least one seed.")
        if len(normalized_seeds) > MAX_EVALUATION_EPISODE_COUNT:
            raise ValueError(
                "evaluation distribution cannot exceed "
                f"{MAX_EVALUATION_EPISODE_COUNT} seeds."
            )
        for seed in normalized_seeds:
            _validate_seed(seed)
        if len(set(normalized_seeds)) != len(normalized_seeds):
            raise ValueError("evaluation distribution seeds must be unique.")
        definition = CANONICAL_EVALUATION_DISTRIBUTIONS.get(normalized_name)
        if definition is not None:
            normalized_purpose = normalized_purpose or definition.purpose
            normalized_owner = normalized_owner or definition.owner
            actual = (
                normalized_role,
                normalized_purpose,
                normalized_owner,
                normalized_seeds,
            )
            if actual != tuple(definition):
                raise ValueError(
                    f"{normalized_name!r} is reserved for its canonical role, "
                    "purpose, owner, and seed list. Use a custom distribution "
                    "name for different conditions."
                )
        else:
            if normalized_role in {"validation", "final-test"}:
                raise ValueError(
                    f"role {normalized_role!r} is reserved for a canonical "
                    "evaluation distribution."
                )
            normalized_purpose = normalized_purpose or (
                "Ad hoc development evaluation; not final-test evidence."
            )
            normalized_owner = normalized_owner or "User-declared workflow"
        object.__setattr__(self, "name", normalized_name)
        object.__setattr__(self, "role", normalized_role)
        object.__setattr__(self, "purpose", normalized_purpose)
        object.__setattr__(self, "owner", normalized_owner)
        object.__setattr__(self, "seeds", normalized_seeds)

    @property
    def condition_digest(self) -> str:
        """Return a stable identity for the complete ordered seed conditions."""
        conditions = {
            "name": self.name,
            "role": self.role,
            "purpose": self.purpose,
            "owner": self.owner,
            "seeds": list(self.seeds),
        }
        canonical = json.dumps(conditions, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        """Return the declared, JSON-compatible evaluation conditions."""
        return {
            "name": self.name,
            "role": self.role,
            "purpose": self.purpose,
            "owner": self.owner,
            "seeds": list(self.seeds),
            "episode_count": len(self.seeds),
            "condition_digest": self.condition_digest,
        }


@dataclass(frozen=True, slots=True)
class EvaluationProtocolConfig:
    """Validated configuration for one complete Evaluation Protocol V1 run."""

    checkpoint_path: Path
    distribution: EvaluationDistribution = field(default_factory=EvaluationDistribution)
    deterministic_actions: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "checkpoint_path", _normalize_checkpoint_path(self.checkpoint_path))
        if self.deterministic_actions is not True:
            raise ValueError(
                "Evaluation Protocol V1 requires deterministic policy actions."
            )


@dataclass(frozen=True, slots=True)
class EpisodeRecord:
    """Immutable evidence retained for one declared evaluation seed."""

    seed: int
    layout_signature: tuple[tuple[str, tuple[int, int]], ...] | None
    total_reward: float | None
    episode_length: int | None
    success: bool | None
    terminated: bool | None
    truncated: bool | None
    error: str | None = None
    initial_agent_direction: int | None = None
    actions: tuple[int, ...] | None = None

    @property
    def completed(self) -> bool:
        """Whether the episode reached a complete, valid outcome."""
        required_outcomes = (
            self.total_reward,
            self.episode_length,
            self.success,
            self.terminated,
            self.truncated,
        )
        return self.error is None and all(
            value is not None for value in required_outcomes
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the report-safe representation of one episode outcome."""
        layout = None
        if self.layout_signature is not None:
            layout = [
                {"object": name, "position": list(position)}
                for name, position in self.layout_signature
            ]
        return {
            "seed": self.seed,
            "layout_signature": layout,
            "total_reward": self.total_reward,
            "episode_length": self.episode_length,
            "success": self.success,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "initial_agent_direction": self.initial_agent_direction,
            "actions": list(self.actions) if self.actions is not None else None,
            "completed": self.completed,
            "error": self.error,
        }


@dataclass(frozen=True, slots=True)
class AggregateMetrics:
    """Protocol-defined summaries derived only from Episode Records."""

    requested_episode_count: int
    completed_episode_count: int
    error_count: int
    completion_rate: float | None
    report_complete: bool
    success_count: int
    success_rate: float | None
    mean_reward: float | None
    reward_standard_deviation: float | None
    mean_episode_length: float | None
    mean_successful_episode_length: float | None
    terminated_count: int
    truncated_count: int

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible metric values without recomputing them."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    """Canonical persisted artifact produced by one Evaluation Protocol V1 run."""

    report_id: str
    protocol_version: str
    created_at: str
    checkpoint_path: Path
    checkpoint_sha256: str
    checkpoint_metadata: dict[str, Any]
    environment: EnvironmentSpec
    distribution: EvaluationDistribution
    deterministic_actions: bool
    episode_records: tuple[EpisodeRecord, ...]
    aggregate_metrics: AggregateMetrics
    runtime_context: dict[str, Any]
    experiment_run_id: str | None = None
    experiment_manifest_path: Path | None = None
    reproducibility_statement: str = REPRODUCIBILITY_STATEMENT
    interpretation_boundary: str = INTERPRETATION_BOUNDARY

    def to_dict(self) -> dict[str, Any]:
        """Serialize the full protocol artifact in its conceptual sections."""
        return {
            "report_id": self.report_id,
            "protocol_version": self.protocol_version,
            "created_at": self.created_at,
            "provenance": {
                "training_run": {
                    "run_id": self.experiment_run_id,
                    "manifest_path": (
                        str(self.experiment_manifest_path)
                        if self.experiment_manifest_path is not None
                        else None
                    ),
                    "identity_source": (
                        "checkpoint-metadata"
                        if self.experiment_run_id is not None
                        else "legacy-unavailable"
                    ),
                },
                "checkpoint": {
                    "path": str(self.checkpoint_path),
                    "sha256": self.checkpoint_sha256,
                    "training_metadata": self.checkpoint_metadata,
                },
                "runtime_context": self.runtime_context,
            },
            "evaluation_conditions": {
                "environment": self.environment.to_dict(),
                "distribution": self.distribution.to_dict(),
                "deterministic_actions": self.deterministic_actions,
            },
            "episode_records": [record.to_dict() for record in self.episode_records],
            "aggregate_metrics": self.aggregate_metrics.to_dict(),
            "reproducibility_statement": self.reproducibility_statement,
            "interpretation_boundary": self.interpretation_boundary,
        }

    def to_json(self) -> str:
        """Return the canonical, human-readable JSON report representation."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


@dataclass(frozen=True, slots=True)
class _EpisodeOutcome:
    """Internal episode data shared by the legacy and protocol evaluators."""

    total_reward: float
    episode_length: int
    terminated: bool
    truncated: bool
    success: bool
    layout_signature: tuple[tuple[str, tuple[int, int]], ...] | None
    initial_agent_direction: int | None
    actions: tuple[int, ...]
    final_frame: np.ndarray | None


def evaluate_model(config: EvaluationConfig) -> EvaluationResult:
    """Load a checkpoint and run the original one deterministic rendered episode."""
    checkpoint_environment_spec = _load_environment_spec(config.checkpoint_path)
    environment_spec = config.environment or checkpoint_environment_spec
    environment = make_environment(
        environment_spec,
        render_mode="rgb_array",
        observation_mode="flat",
    )
    try:
        model = load_model(config.checkpoint_path, environment=environment)
        outcome = _execute_episode(
            model,
            environment,
            seed=config.seed,
            capture_frame=True,
            require_environment_success=False,
        )
        if outcome.final_frame is None:
            raise RuntimeError("Evaluation did not produce a final RGB frame.")
        return EvaluationResult(
            checkpoint_path=config.checkpoint_path,
            environment=environment_spec,
            seed=config.seed,
            total_reward=outcome.total_reward,
            episode_length=outcome.episode_length,
            terminated=outcome.terminated,
            truncated=outcome.truncated,
            success=outcome.success,
            final_frame=outcome.final_frame,
        )
    finally:
        environment.close()


def run_evaluation_protocol(config: EvaluationProtocolConfig) -> EvaluationReport:
    """Execute every declared seed and return the complete Evaluation V1 report.

    The environment and PPO model are created once for the run, while every
    declared seed receives its own reset-to-terminal episode and record. An
    individual episode failure becomes an explicit record rather than being
    silently excluded from the report.
    """
    checkpoint_metadata = load_checkpoint_metadata(config.checkpoint_path)
    experiment_run_id, experiment_manifest_path = _experiment_identity(
        checkpoint_metadata
    )
    environment_spec = _load_environment_spec(config.checkpoint_path)
    started_at = datetime.now(timezone.utc)
    checkpoint_sha256 = calculate_checkpoint_sha256(config.checkpoint_path)
    environment = make_environment(environment_spec, observation_mode="flat")
    try:
        model = load_model(config.checkpoint_path, environment=environment)
        records = tuple(
            _run_protocol_episode(
                model,
                environment,
                seed=seed,
                deterministic_actions=config.deterministic_actions,
            )
            for seed in config.distribution.seeds
        )
    finally:
        environment.close()

    aggregate_metrics = aggregate_episode_records(records)
    created_at = started_at.isoformat()
    report_id = _build_report_id(
        created_at=created_at,
        checkpoint_sha256=checkpoint_sha256,
        distribution=config.distribution,
    )
    return EvaluationReport(
        report_id=report_id,
        protocol_version=EVALUATION_PROTOCOL_VERSION,
        created_at=created_at,
        checkpoint_path=config.checkpoint_path.resolve(),
        checkpoint_sha256=checkpoint_sha256,
        checkpoint_metadata=checkpoint_metadata,
        environment=environment_spec,
        distribution=config.distribution,
        deterministic_actions=config.deterministic_actions,
        episode_records=records,
        aggregate_metrics=aggregate_metrics,
        runtime_context=_runtime_context(),
        experiment_run_id=experiment_run_id,
        experiment_manifest_path=experiment_manifest_path,
    )


def aggregate_episode_records(
    records: tuple[EpisodeRecord, ...] | list[EpisodeRecord],
) -> AggregateMetrics:
    """Derive every Evaluation V1 aggregate from retained Episode Records."""
    completed_records = [record for record in records if record.completed]
    rewards = [
        record.total_reward
        for record in completed_records
        if record.total_reward is not None
    ]
    lengths = [
        record.episode_length
        for record in completed_records
        if record.episode_length is not None
    ]
    successful_lengths = [
        record.episode_length
        for record in completed_records
        if record.success is True and record.episode_length is not None
    ]
    success_count = sum(record.success is True for record in completed_records)
    completed_count = len(completed_records)
    return AggregateMetrics(
        requested_episode_count=len(records),
        completed_episode_count=completed_count,
        error_count=len(records) - completed_count,
        completion_rate=(completed_count / len(records)) if records else None,
        report_complete=bool(records) and completed_count == len(records),
        success_count=success_count,
        success_rate=(success_count / completed_count) if completed_count else None,
        mean_reward=fmean(rewards) if rewards else None,
        reward_standard_deviation=pstdev(rewards) if rewards else None,
        mean_episode_length=fmean(lengths) if lengths else None,
        mean_successful_episode_length=(
            fmean(successful_lengths) if successful_lengths else None
        ),
        terminated_count=sum(
            record.terminated is True for record in completed_records
        ),
        truncated_count=sum(
            record.truncated is True for record in completed_records
        ),
    )


def save_evaluation_report(
    report: EvaluationReport,
    *,
    report_root: str | Path = DEFAULT_EVALUATION_REPORT_ROOT,
) -> Path:
    """Persist one immutable report under the local evaluation artifact root."""
    destination = (
        Path(report_root)
        / report.environment.capability
        / f"{report.report_id}.json"
    )
    published = write_text_exclusive(destination, report.to_json())
    if report.experiment_manifest_path is None:
        return published
    experiment_destination = (
        report.experiment_manifest_path.parent
        / "evaluations"
        / published.name
    )
    experiment_destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        publish_new_file(published, experiment_destination)
    except Exception:
        published.unlink(missing_ok=True)
        raise
    return published


def discover_evaluation_checkpoints(
    checkpoint_root: str | Path = DEFAULT_CHECKPOINT_ROOT,
) -> tuple[Path, ...]:
    """Return checkpoints that have valid metadata required by Evaluation V1."""
    scan = scan_checkpoint_artifacts(checkpoint_root)
    return tuple(
        reference.checkpoint_path
        for reference in scan.included
    )


def parse_evaluation_seeds(seed_text: str) -> tuple[int, ...]:
    """Parse a comma- or whitespace-separated seed list at the backend boundary."""
    values = [value for value in re.split(r"[\s,]+", seed_text.strip()) if value]
    if not values:
        raise ValueError("Provide at least one evaluation seed.")
    try:
        return tuple(int(value) for value in values)
    except ValueError as error:
        raise ValueError(
            "Evaluation seeds must be comma- or space-separated integers."
        ) from error


def get_canonical_evaluation_distribution(name: str) -> EvaluationDistribution:
    """Return one immutable canonical distribution by its exact name."""
    normalized_name = name.strip()
    definition = CANONICAL_EVALUATION_DISTRIBUTIONS.get(normalized_name)
    if definition is None:
        raise ValueError(f"Unknown canonical evaluation distribution: {name!r}.")
    return EvaluationDistribution(
        name=normalized_name,
        role=definition.role,
        purpose=definition.purpose,
        owner=definition.owner,
        seeds=definition.seeds,
    )


def resolve_evaluation_distribution(
    name: str,
    seeds: tuple[int, ...],
) -> EvaluationDistribution:
    """Resolve canonical names strictly while preserving ad hoc development runs."""
    normalized_name = name.strip()
    if normalized_name in CANONICAL_EVALUATION_DISTRIBUTIONS:
        canonical = get_canonical_evaluation_distribution(normalized_name)
        if tuple(seeds) != canonical.seeds:
            raise ValueError(
                f"{normalized_name!r} has an immutable seed list. Use its "
                "documented seeds or choose a custom distribution name."
            )
        return canonical
    return EvaluationDistribution(name=normalized_name, seeds=seeds)


def format_evaluation_seeds(
    seeds: tuple[int, ...] = DEFAULT_EVALUATION_SEEDS,
) -> str:
    """Return the compact seed text used by presentation defaults."""
    return ", ".join(str(seed) for seed in seeds)


def _run_protocol_episode(
    model: Any,
    environment: Any,
    *,
    seed: int,
    deterministic_actions: bool,
) -> EpisodeRecord:
    """Convert one protocol episode into evidence, including explicit failures."""
    try:
        outcome = _execute_episode(
            model,
            environment,
            seed=seed,
            capture_frame=False,
            require_environment_success=True,
            deterministic_actions=deterministic_actions,
        )
    except Exception as error:
        try:
            failed_layout_signature = _layout_signature(environment)
        except Exception:
            failed_layout_signature = None
        return EpisodeRecord(
            seed=seed,
            layout_signature=failed_layout_signature,
            total_reward=None,
            episode_length=None,
            success=None,
            terminated=None,
            truncated=None,
            initial_agent_direction=None,
            error=f"{type(error).__name__}: {error}",
        )
    return EpisodeRecord(
        seed=seed,
        layout_signature=outcome.layout_signature,
        total_reward=outcome.total_reward,
        episode_length=outcome.episode_length,
        success=outcome.success,
        terminated=outcome.terminated,
        truncated=outcome.truncated,
        initial_agent_direction=outcome.initial_agent_direction,
        actions=outcome.actions,
    )


def _execute_episode(
    model: Any,
    environment: Any,
    *,
    seed: int,
    capture_frame: bool,
    require_environment_success: bool,
    deterministic_actions: bool = True,
) -> _EpisodeOutcome:
    """Run one reset-to-terminal episode through an already-owned environment."""
    observation, _ = environment.reset(seed=seed)
    layout_signature = _layout_signature(environment)
    initial_agent_direction = _initial_agent_direction(environment)
    total_reward = 0.0
    episode_length = 0
    terminated = False
    truncated = False
    final_info: dict[str, Any] = {}
    actions: list[int] = []

    while not (terminated or truncated):
        action, _ = model.predict(observation, deterministic=deterministic_actions)
        action_value = int(np.asarray(action).item())
        actions.append(action_value)
        observation, reward, terminated, truncated, step_info = environment.step(
            action_value
        )
        final_info = dict(step_info)
        total_reward += float(reward)
        episode_length += 1

    if "success" not in final_info:
        if require_environment_success:
            raise RuntimeError(
                "Environment did not provide the required info['success'] outcome."
            )
        success = bool(terminated and total_reward > 0.0)
    else:
        success = bool(final_info["success"])

    final_frame: np.ndarray | None = None
    if capture_frame:
        frame = environment.render()
        if (
            not isinstance(frame, np.ndarray)
            or frame.ndim != 3
            or frame.shape[-1] != 3
        ):
            raise RuntimeError("Evaluation did not produce a valid RGB frame.")
        final_frame = np.array(frame, copy=True)
    return _EpisodeOutcome(
        total_reward=total_reward,
        episode_length=episode_length,
        terminated=bool(terminated),
        truncated=bool(truncated),
        success=success,
        layout_signature=layout_signature,
        initial_agent_direction=initial_agent_direction,
        actions=tuple(actions),
        final_frame=final_frame,
    )


def _layout_signature(
    environment: Any,
) -> tuple[tuple[str, tuple[int, int]], ...] | None:
    """Read the optional procedural layout identifier without coupling to a class."""
    unwrapped = getattr(environment, "unwrapped", environment)
    method = getattr(unwrapped, "layout_signature", None)
    if not callable(method):
        return None
    raw_signature = method()
    return tuple(
        (str(name), (int(position[0]), int(position[1])))
        for name, position in raw_signature
    )


def _initial_agent_direction(environment: Any) -> int | None:
    """Read the initial direction separately from the spatial layout signature."""
    unwrapped = getattr(environment, "unwrapped", environment)
    direction = getattr(unwrapped, "agent_dir", None)
    if isinstance(direction, (int, np.integer)):
        return int(direction)
    return None


def _load_environment_spec(checkpoint_path: Path) -> EnvironmentSpec:
    """Reconstruct the environment contract stored beside a checkpoint."""
    metadata = load_checkpoint_metadata(checkpoint_path)
    environment_payload = metadata.get("environment")
    if not isinstance(environment_payload, dict):
        raise ValueError("Checkpoint metadata is missing its environment object.")
    return EnvironmentSpec.from_dict(environment_payload)


def _experiment_identity(
    checkpoint_metadata: dict[str, Any],
) -> tuple[str | None, Path | None]:
    """Read explicit run linkage without guessing identity for legacy artifacts."""
    run_id = checkpoint_metadata.get("run_id")
    experiment = checkpoint_metadata.get("experiment")
    if not isinstance(run_id, str) or not isinstance(experiment, dict):
        return None, None
    manifest_path = experiment.get("manifest_path")
    if not isinstance(manifest_path, str) or not manifest_path:
        return run_id, None
    return run_id, Path(manifest_path)


def _validate_seed(seed: int) -> None:
    """Validate generator seeds consistently across both evaluation APIs."""
    if type(seed) is not int or not 0 <= seed <= MAX_SEED:
        raise ValueError(f"seed must be an integer from 0 to {MAX_SEED}.")


def _normalize_checkpoint_path(checkpoint_path: str | Path) -> Path:
    """Match Stable-Baselines3's .zip checkpoint convention."""
    path = Path(checkpoint_path)
    return path if path.suffix.lower() == ".zip" else path.with_suffix(".zip")


def _build_report_id(
    *,
    created_at: str,
    checkpoint_sha256: str,
    distribution: EvaluationDistribution,
) -> str:
    """Build a human-readable, collision-resistant immutable report identity."""
    timestamp = (
        created_at.replace("+00:00", "Z").replace(":", "").replace("-", "")
    )
    return (
        f"{EVALUATION_PROTOCOL_VERSION}_{_safe_report_label(distribution.role)}"
        f"_{timestamp}_{checkpoint_sha256[:12]}"
        f"_{distribution.condition_digest[:8]}"
    )


def _safe_report_label(value: str) -> str:
    """Return a filesystem-safe label for a report's declared role."""
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-") or "unspecified"


def _runtime_context() -> dict[str, Any]:
    """Capture the compact software context needed to interpret reproducibility."""
    package_versions = {
        package: _package_version(package)
        for package in (
            "gymnasium",
            "minigrid",
            "numpy",
            "stable-baselines3",
            "torch",
        )
    }
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        **package_versions,
        "source_revision": capture_source_revision(),
    }


def _package_version(package: str) -> str:
    """Return an installed package version without making reports fail on lookup."""
    try:
        return version(package)
    except PackageNotFoundError:
        return "unavailable"
