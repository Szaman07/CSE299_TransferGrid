"""Canonical PPO training service for TransferGrid."""

from __future__ import annotations

import platform
import json
import math
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from importlib import metadata as package_metadata
from pathlib import Path
from time import perf_counter
from typing import Any, Literal, Mapping
from uuid import uuid4

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv

from environments.factory import EnvironmentSpec, make_environment
from environments.generation import DEFAULT_SEED, MAX_SEED
from environments.catalog import require_released_task
from training.callbacks import ExperimentLoggingCallback
from training.artifacts import write_text_exclusive
from training.checkpoints import (
    DEFAULT_CHECKPOINT_ROOT,
    build_checkpoint_path,
    checkpoint_metadata_path,
    checkpoint_sha256,
    save_checkpoint_metadata,
    save_model,
    validate_training_run_id,
)
from training.experiments import (
    GOVERNED_EXPERIMENT_SCHEMA_VERSION,
    DEFAULT_EXPERIMENT_ROOT,
    ExperimentPaths,
    build_experiment_paths,
    reserve_experiment_directory,
    save_completed_experiment,
    save_experiment_configuration,
)
from training.provenance import capture_source_revision
from training.development import (
    DEVELOPMENT_EVALUATION_INTERVAL,
    DevelopmentEvaluationCallback,
)
from training.representation_snapshots import RepresentationSnapshotCallback
from training.initialization import (
    FROZEN_ACTOR_PROBE,
    FullPolicyInitialization,
    Initialization,
    ScratchInitialization,
    SelectivePolicyInitialization,
    TransferCompatibilityError,
    copy_full_policy,
    copy_selective_policy,
    checkpoint_compatible_optimizer,
    is_transfer_initialization,
    initialization_from_dict,
    policy_tensor_digest,
    preflight_full_policy,
    preflight_selective_policy,
    CompatibilityReport,
)
from training.transfer_governance import (
    DEFAULT_EVALUATION_V2_REPORT_ROOT,
    SourceCompetenceError,
    TransferPreflightReport,
    assess_source_competence,
    build_transfer_preflight_report,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TENSORBOARD_LOG_ROOT = PROJECT_ROOT / "logs" / "tensorboard"

DEFAULT_TIMESTEPS = 256
MIN_TIMESTEPS = 64
DEFAULT_LEARNING_RATE = 3e-4
PPO_ROLLOUT_STEPS = 128
PPO_ENTROPY_COEF = 0.01
PPO_BATCH_SIZE = 256
PPO_EPOCHS = 10
PPO_ADAM_EPS = 1e-5
PPO_GAMMA = 0.99
N_ENVS = 8
PROCEDURAL_TRAINING_MODE = "procedural"
FIXED_LAYOUT_TRAINING_MODE = "fixed_layout"
TrainingMode = Literal[PROCEDURAL_TRAINING_MODE, FIXED_LAYOUT_TRAINING_MODE]


def _records_development_curve(config: TrainingConfig) -> bool:
    """Return whether this run belongs to the governed learning-curve path."""
    if is_transfer_initialization(config.initialization):
        return True
    if (
        config.training_mode != PROCEDURAL_TRAINING_MODE
        or config.total_timesteps < DEVELOPMENT_EVALUATION_INTERVAL
    ):
        return False
    try:
        require_released_task(config.environment.task_id)
    except (TypeError, ValueError):
        return False
    return True


def create_training_run_id() -> str:
    """Return a collision-resistant, filename-safe training run identifier."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{timestamp}_{uuid4().hex[:8]}"


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Small, serializable configuration for a Training v0 run."""

    environment: EnvironmentSpec = field(default_factory=EnvironmentSpec)
    total_timesteps: int = DEFAULT_TIMESTEPS
    learning_rate: float = DEFAULT_LEARNING_RATE
    n_envs: int = N_ENVS
    n_epochs: int = PPO_EPOCHS
    adam_eps: float = PPO_ADAM_EPS
    gamma: float = PPO_GAMMA
    ent_coef: float = PPO_ENTROPY_COEF
    seed: int = DEFAULT_SEED
    training_mode: TrainingMode = PROCEDURAL_TRAINING_MODE
    layout_seed: int | None = None
    checkpoint_root: Path = DEFAULT_CHECKPOINT_ROOT
    tensorboard_root: Path = DEFAULT_TENSORBOARD_LOG_ROOT
    experiment_root: Path = DEFAULT_EXPERIMENT_ROOT
    run_id: str | None = None
    initialization: Initialization = field(
        default_factory=ScratchInitialization
    )
    representation_snapshot_fractions: tuple[float, ...] = ()
    study_binding: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if (
            type(self.total_timesteps) is not int
            or self.total_timesteps < MIN_TIMESTEPS
        ):
            raise ValueError(
                f"total_timesteps must be an integer of at least {MIN_TIMESTEPS}."
            )
        if type(self.learning_rate) not in (int, float) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be greater than zero.")
        if type(self.n_envs) is not int or self.n_envs < 1:
            raise ValueError("n_envs must be a positive integer.")
        if type(self.n_epochs) is not int or self.n_epochs < 1:
            raise ValueError("n_epochs must be a positive integer.")
        if type(self.adam_eps) not in (int, float) or self.adam_eps <= 0:
            raise ValueError("adam_eps must be greater than zero.")
        if type(self.gamma) not in (int, float) or not 0 < self.gamma <= 1:
            raise ValueError("gamma must be greater than zero and at most one.")
        if type(self.ent_coef) not in (int, float) or self.ent_coef < 0:
            raise ValueError("ent_coef must be zero or greater.")
        if type(self.seed) is not int or not 0 <= self.seed <= MAX_SEED:
            raise ValueError(f"seed must be an integer from 0 to {MAX_SEED}.")
        if self.training_mode not in (
            PROCEDURAL_TRAINING_MODE,
            FIXED_LAYOUT_TRAINING_MODE,
        ):
            raise ValueError(f"Unsupported training mode: {self.training_mode!r}.")
        if self.training_mode == FIXED_LAYOUT_TRAINING_MODE:
            if (
                type(self.layout_seed) is not int
                or not 0 <= self.layout_seed <= MAX_SEED
            ):
                raise ValueError(
                    f"layout_seed must be an integer from 0 to {MAX_SEED} "
                    "for fixed-layout training."
                )
        elif self.layout_seed is not None:
            raise ValueError("layout_seed is only valid for fixed-layout training.")
        run_id = (
            create_training_run_id()
            if self.run_id is None
            else self.run_id
        )
        object.__setattr__(self, "learning_rate", float(self.learning_rate))
        object.__setattr__(self, "adam_eps", float(self.adam_eps))
        object.__setattr__(self, "gamma", float(self.gamma))
        object.__setattr__(self, "ent_coef", float(self.ent_coef))
        object.__setattr__(self, "checkpoint_root", Path(self.checkpoint_root))
        object.__setattr__(self, "tensorboard_root", Path(self.tensorboard_root))
        object.__setattr__(self, "experiment_root", Path(self.experiment_root))
        object.__setattr__(self, "run_id", validate_training_run_id(run_id))
        if not isinstance(
            self.initialization,
            (
                ScratchInitialization,
                FullPolicyInitialization,
                SelectivePolicyInitialization,
            ),
        ):
            raise ValueError("initialization must be a supported contract.")
        fractions = tuple(float(value) for value in self.representation_snapshot_fractions)
        if fractions and (
            fractions != tuple(sorted(set(fractions)))
            or not all(math.isfinite(value) for value in fractions)
            or fractions[0] < 0.0
            or fractions[-1] > 1.0
        ):
            raise ValueError(
                "representation_snapshot_fractions must be sorted, unique values "
                "between 0 and 1."
            )
        rollout_size = PPO_ROLLOUT_STEPS * self.n_envs
        if fractions and self.total_timesteps % rollout_size:
            raise ValueError(
                "Snapshot-instrumented total_timesteps must align with a "
                f"complete PPO rollout ({rollout_size} steps)."
            )
        for fraction in fractions:
            step = int(round(fraction * self.total_timesteps))
            if step not in (0, self.total_timesteps) and step % rollout_size:
                raise ValueError(
                    "Representation snapshot steps must align with completed "
                    f"PPO rollouts ({rollout_size} steps); got {step}."
                )
        object.__setattr__(self, "representation_snapshot_fractions", fractions)
        if self.study_binding is not None:
            if not isinstance(self.study_binding, Mapping):
                raise ValueError("study_binding must be a mapping or None.")
            binding = dict(self.study_binding)
            try:
                json.dumps(binding, sort_keys=True)
            except (TypeError, ValueError) as error:
                raise ValueError("study_binding must be JSON serializable.") from error
            object.__setattr__(self, "study_binding", binding)

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible values for subprocess and metadata boundaries."""
        payload = {
            "environment": self.environment.to_dict(),
            "total_timesteps": self.total_timesteps,
            "learning_rate": self.learning_rate,
            "n_envs": self.n_envs,
            "n_epochs": self.n_epochs,
            "adam_eps": self.adam_eps,
            "gamma": self.gamma,
            "ent_coef": self.ent_coef,
            "seed": self.seed,
            "training_mode": self.training_mode,
            "layout_seed": self.layout_seed,
            "checkpoint_root": str(self.checkpoint_root),
            "tensorboard_root": str(self.tensorboard_root),
            "experiment_root": str(self.experiment_root),
            "run_id": self.run_id,
            "initialization": self.initialization.to_dict(),
            "representation_snapshot_fractions": list(
                self.representation_snapshot_fractions
            ),
        }
        if self.study_binding is not None:
            payload["study_binding"] = dict(self.study_binding)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> TrainingConfig:
        """Build a validated configuration from a JSON-compatible mapping."""
        environment_payload = payload.get("environment", {})
        if not isinstance(environment_payload, dict):
            raise ValueError("environment must be a JSON object.")
        return cls(
            environment=EnvironmentSpec.from_dict(environment_payload),
            total_timesteps=payload.get("total_timesteps", DEFAULT_TIMESTEPS),
            learning_rate=payload.get("learning_rate", DEFAULT_LEARNING_RATE),
            n_envs=payload.get("n_envs", N_ENVS),
            n_epochs=payload.get("n_epochs", PPO_EPOCHS),
            adam_eps=payload.get("adam_eps", PPO_ADAM_EPS),
            gamma=payload.get("gamma", PPO_GAMMA),
            ent_coef=payload.get("ent_coef", PPO_ENTROPY_COEF),
            seed=payload.get("seed", DEFAULT_SEED),
            training_mode=payload.get("training_mode", PROCEDURAL_TRAINING_MODE),
            layout_seed=payload.get("layout_seed"),
            checkpoint_root=Path(
                payload.get("checkpoint_root", str(DEFAULT_CHECKPOINT_ROOT))
            ),
            tensorboard_root=Path(
                payload.get(
                    "tensorboard_root",
                    str(DEFAULT_TENSORBOARD_LOG_ROOT),
                )
            ),
            experiment_root=Path(
                payload.get("experiment_root", str(DEFAULT_EXPERIMENT_ROOT))
            ),
            run_id=payload.get("run_id"),
            initialization=initialization_from_dict(
                payload.get("initialization", {"method": "scratch"})
            ),
            representation_snapshot_fractions=tuple(
                payload.get("representation_snapshot_fractions", ())
            ),
            study_binding=payload.get("study_binding"),
        )


@dataclass(frozen=True, slots=True)
class TrainingSummary:
    """Lightweight outcome metadata returned after a training run."""

    checkpoint_path: Path
    metadata_path: Path
    requested_timesteps: int
    completed_timesteps: int
    learning_rate: float
    n_envs: int
    n_epochs: int
    adam_eps: float
    gamma: float
    ent_coef: float
    seed: int
    environment: EnvironmentSpec
    training_mode: TrainingMode
    layout_seed: int | None
    run_id: str
    tensorboard_path: Path
    experiment_path: Path
    experiment_manifest_path: Path
    configuration_path: Path
    monitor_path: Path
    training_metrics: dict[str, Any]
    elapsed_seconds: float
    initialization: dict[str, Any]
    initialization_checkpoint_path: Path | None
    initialization_checkpoint_sha256: str | None
    development_curve_path: Path | None
    evaluation_overhead_seconds: float
    representation_snapshot_manifest_path: Path | None

    def to_dict(self) -> dict[str, Any]:
        """Return metadata containing no model or live environment instance."""
        return {
            "checkpoint_path": str(self.checkpoint_path),
            "metadata_path": str(self.metadata_path),
            "requested_timesteps": self.requested_timesteps,
            "completed_timesteps": self.completed_timesteps,
            "learning_rate": self.learning_rate,
            "n_envs": self.n_envs,
            "n_epochs": self.n_epochs,
            "adam_eps": self.adam_eps,
            "gamma": self.gamma,
            "ent_coef": self.ent_coef,
            "seed": self.seed,
            "environment": self.environment.to_dict(),
            "training_mode": self.training_mode,
            "layout_seed": self.layout_seed,
            "run_id": self.run_id,
            "tensorboard_path": str(self.tensorboard_path),
            "experiment_path": str(self.experiment_path),
            "experiment_manifest_path": str(self.experiment_manifest_path),
            "configuration_path": str(self.configuration_path),
            "monitor_path": str(self.monitor_path),
            "training_metrics": dict(self.training_metrics),
            "elapsed_seconds": self.elapsed_seconds,
            "initialization": dict(self.initialization),
            "initialization_checkpoint_path": (
                str(self.initialization_checkpoint_path)
                if self.initialization_checkpoint_path is not None else None
            ),
            "initialization_checkpoint_sha256": (
                self.initialization_checkpoint_sha256
            ),
            "development_curve_path": (
                str(self.development_curve_path)
                if self.development_curve_path is not None else None
            ),
            "evaluation_overhead_seconds": self.evaluation_overhead_seconds,
            "representation_snapshot_manifest_path": (
                str(self.representation_snapshot_manifest_path)
                if self.representation_snapshot_manifest_path is not None
                else None
            ),
        }


def train_model(config: TrainingConfig) -> TrainingSummary:
    """Train one PPO model through the canonical environment factory and save it."""
    started_at = datetime.now(timezone.utc)
    started_timer = perf_counter()
    runtime_context = _runtime_context()
    source_revision = capture_source_revision()
    run_id = validate_training_run_id(config.run_id or "")
    study_metadata: dict[str, Any] = {}
    if config.study_binding is not None:
        artifact_role = config.study_binding.get("artifact_role")
        if not isinstance(artifact_role, str) or not artifact_role:
            raise ValueError(
                "A bound study run must declare a non-empty artifact_role."
            )
        study_metadata = {
            "study_binding": dict(config.study_binding),
            "artifact_role": artifact_role,
        }
    transfer_preflight: TransferPreflightReport | None = None
    if is_transfer_initialization(config.initialization):
        # Fail before reserving or publishing any run artifact. Policy copying
        # performs the technical checks again after reservation to close the
        # source-mutation window.
        transfer_preflight = check_transfer_preflight(config)
        if not transfer_preflight.technical_compatibility.compatible:
            raise TransferCompatibilityError(
                transfer_preflight.technical_compatibility
            )
        if not transfer_preflight.source_competence.competent:
            raise SourceCompetenceError(
                transfer_preflight.source_competence
            )
    checkpoint_path = build_checkpoint_path(
        config.environment,
        seed=config.seed,
        total_timesteps=config.total_timesteps,
        learning_rate=config.learning_rate,
        training_mode=config.training_mode,
        layout_seed=config.layout_seed,
        run_id=run_id,
        checkpoint_root=config.checkpoint_root,
    )
    metadata_path = checkpoint_metadata_path(checkpoint_path)
    tensorboard_path = config.tensorboard_root / f"{run_id}_1"
    experiment_paths = build_experiment_paths(run_id, config.experiment_root)
    _reserve_training_run(
        checkpoint_path,
        tensorboard_root=config.tensorboard_root,
        run_id=run_id,
        tensorboard_path=tensorboard_path,
        experiment_paths=experiment_paths,
    )
    artifact_schema = (
        GOVERNED_EXPERIMENT_SCHEMA_VERSION
        if config.environment.task_id is not None
        else 1
    )
    save_experiment_configuration(
        experiment_paths,
        run_id=run_id,
        created_at=started_at.isoformat(),
        configuration=config.to_dict(),
        runtime_context=runtime_context,
        source_revision=source_revision,
        schema_version=artifact_schema,
    )

    def _env_factory():
        return make_environment(
            config.environment,
            observation_mode="flat",
            fixed_layout_seed=(
                config.layout_seed
                if config.training_mode == FIXED_LAYOUT_TRAINING_MODE
                else None
            ),
        )

    environment = make_vec_env(
        _env_factory,
        n_envs=config.n_envs,
        vec_env_cls=DummyVecEnv,
        monitor_dir=str(experiment_paths.monitor_directory),
        monitor_kwargs={"info_keywords": ("success", "timeout")},
    )
    try:
        model = _create_ppo(environment, config)
        initialization_checkpoint_path: Path | None = None
        initialization_metadata_path: Path | None = None
        initialization_checkpoint_hash: str | None = None
        development_curve_path: Path | None = None
        development_callback: DevelopmentEvaluationCallback | None = None
        representation_callback: RepresentationSnapshotCallback | None = None
        representation_snapshot_manifest_path: Path | None = None
        initialization_provenance: dict[str, Any] = (
            config.initialization.to_dict()
        )
        if is_transfer_initialization(config.initialization):
            if transfer_preflight is None:
                raise RuntimeError("Transfer readiness was not established.")
            target_definition = require_released_task(
                config.environment.task_id
            )
            if isinstance(config.initialization, FullPolicyInitialization):
                copy_result = copy_full_policy(
                    model,
                    config.initialization,
                    target_observation_contract_id=(
                        target_definition.observation_contract_id
                    ),
                    target_action_contract_id=(
                        target_definition.action_contract_id
                    ),
                )
            else:
                copy_result = copy_selective_policy(
                    model,
                    config.initialization,
                    target_observation_contract_id=(
                        target_definition.observation_contract_id
                    ),
                    target_action_contract_id=(
                        target_definition.action_contract_id
                    ),
                )
            if (
                copy_result.compatibility.digest
                != transfer_preflight.technical_compatibility.digest
            ):
                raise RuntimeError(
                    "Transfer compatibility changed after artifact reservation."
                )
            initialization_checkpoint_path = checkpoint_path.with_name(
                f"{checkpoint_path.stem}_initialization0.zip"
            )
            with checkpoint_compatible_optimizer(model):
                initialization_checkpoint_path = save_model(
                    model, initialization_checkpoint_path
                )
            initialization_checkpoint_hash = checkpoint_sha256(
                initialization_checkpoint_path
            )
            initialization_provenance = {
                **config.initialization.to_dict(),
                "compatibility": copy_result.compatibility.to_dict(),
                "compatibility_digest": copy_result.compatibility.digest,
                "technical_compatibility": (
                    copy_result.compatibility.to_dict()
                ),
                "technical_compatibility_digest": (
                    copy_result.compatibility.digest
                ),
                "source_competence": (
                    transfer_preflight.source_competence.to_dict()
                ),
                "source_competence_digest": (
                    transfer_preflight.source_competence.digest
                ),
                "study_relation": config.initialization.study_relation,
                "transfer_preflight": transfer_preflight.to_dict(),
                "transfer_preflight_binding_digest": (
                    transfer_preflight.binding_digest
                ),
                "copy_result": copy_result.to_dict(),
                "target_ppo_seed": config.seed,
                "initialization_checkpoint_path": str(
                    initialization_checkpoint_path
                ),
                "initialization_checkpoint_sha256": (
                    initialization_checkpoint_hash
                ),
            }
            initialization_metadata_path = save_checkpoint_metadata(
                initialization_checkpoint_path,
                {
                    "schema_version": 3,
                    "artifact_role": "target-initialization-0-percent",
                    "run_id": run_id,
                    "algorithm": "PPO",
                    "environment": config.environment.to_dict(),
                    "task_contract_digest": (
                        config.environment.task_contract_digest
                    ),
                    "observation_mode": "flat",
                    "seed": config.seed,
                    "training_mode": config.training_mode,
                    "layout_seed": config.layout_seed,
                    "requested_timesteps": config.total_timesteps,
                    "completed_timesteps": 0,
                    "checkpoint_path": str(initialization_checkpoint_path),
                    "checkpoint_sha256": initialization_checkpoint_hash,
                    "initialization": initialization_provenance,
                    "study_binding": (
                        dict(config.study_binding)
                        if config.study_binding is not None else None
                    ),
                    "runtime_context": runtime_context,
                    "source_revision": source_revision,
                },
            )
        if _records_development_curve(config):
            development_callback = DevelopmentEvaluationCallback(
                config.environment
            )
            development_callback.evaluate_point(0, model)
        logging_callback = ExperimentLoggingCallback(
            {
                "run_id": run_id,
                "capability": config.environment.resolved_task_id,
                "environment": config.environment.to_dict(),
                "training_mode": config.training_mode,
                "training_seed": config.seed,
                "layout_seed": config.layout_seed,
                "requested_timesteps": config.total_timesteps,
                "learning_rate": config.learning_rate,
                "n_envs": config.n_envs,
                "n_epochs": config.n_epochs,
                "adam_eps": config.adam_eps,
                "gamma": config.gamma,
                "ent_coef": config.ent_coef,
                 "initialization": initialization_provenance,
                 "study_binding": (
                     dict(config.study_binding)
                     if config.study_binding is not None else None
                 ),
                 "action_labels": _environment_action_labels(environment),
                "checkpoint_path": str(checkpoint_path),
                "experiment_manifest_path": str(experiment_paths.manifest_json),
            }
        )
        callback_items: list[Any] = [logging_callback]
        if development_callback is not None:
            callback_items.append(development_callback)
        if config.representation_snapshot_fractions:
            representation_callback = RepresentationSnapshotCallback(
                experiment_paths.root / "representation_snapshots",
                run_id=run_id,
                total_timesteps=config.total_timesteps,
                fractions=config.representation_snapshot_fractions,
            )
            callback_items.append(representation_callback)
        callbacks = (
            CallbackList(callback_items) if len(callback_items) > 1
            else logging_callback
        )
        model.learn(
            total_timesteps=config.total_timesteps,
            callback=callbacks,
            tb_log_name=run_id,
        )
        if (
            isinstance(config.initialization, SelectivePolicyInitialization)
            and config.initialization.method == FROZEN_ACTOR_PROBE
        ):
            frozen_keys = tuple(
                initialization_provenance["copy_result"]["frozen_keys"]
            )
            before_digest = initialization_provenance["copy_result"][
                "frozen_group_sha256"
            ]
            after_digest = policy_tensor_digest(model, frozen_keys)
            if after_digest != before_digest:
                raise RuntimeError(
                    "Frozen actor-body tensors changed during C3 training."
                )
            initialization_provenance = {
                **initialization_provenance,
                "frozen_group_post_training_sha256": after_digest,
                "frozen_group_unchanged": True,
            }
        if representation_callback is not None:
            representation_callback.capture_final()
            representation_snapshot_manifest_path = (
                experiment_paths.root / "representation_snapshots.json"
            )
            representation_callback.save_manifest(
                representation_snapshot_manifest_path,
                context={
                    "task_id": config.environment.task_id,
                    "task_contract_digest": config.environment.task_contract_digest,
                    "training_seed": config.seed,
                    "initialization": initialization_provenance,
                },
            )
        if development_callback is not None:
            # A callback reached at the last collected transition runs before
            # PPO's final optimizer update. Replace that boundary measurement
            # so the exact-final point describes the saved final policy.
            if (
                development_callback.points
                and development_callback.points[-1]["step"]
                == model.num_timesteps
            ):
                replaced = development_callback.points.pop()
                development_callback.evaluation_seconds -= float(
                    replaced["evaluation_seconds"]
                )
            development_callback.evaluate_point(model.num_timesteps)
            development_curve_path = (
                experiment_paths.root / "development_curve.json"
            )
            write_text_exclusive(
                development_curve_path,
                json.dumps({
                    "schema_version": 1,
                    "artifact_type": "transfergrid-development-curve",
                    "run_id": run_id,
                    "task_id": config.environment.task_id,
                    "task_contract_digest": (
                        config.environment.task_contract_digest
                    ),
                    "deterministic_actions": True,
                    "separate_evaluation_environments": True,
                    "rng_state_restored": True,
                    "evaluation_interval": development_callback.interval,
                    "points": development_callback.points,
                    "evaluation_seconds": (
                        development_callback.evaluation_seconds
                    ),
                }, indent=2, sort_keys=True) + "\n",
            )
        with checkpoint_compatible_optimizer(model):
            checkpoint_path = save_model(model, checkpoint_path)
        elapsed_seconds = perf_counter() - started_timer
        finished_at = datetime.now(timezone.utc)
        training_metrics = logging_callback.metrics().to_dict()
        try:
            checkpoint_hash = checkpoint_sha256(checkpoint_path)
            metadata_path = save_checkpoint_metadata(
                checkpoint_path,
                {
                    "schema_version": 3 if config.environment.task_id else 2,
                    "run_id": run_id,
                    "algorithm": "PPO",
                    "environment": config.environment.to_dict(),
                    "task_contract_digest": config.environment.task_contract_digest,
                    "observation_mode": "flat",
                    "seed": config.seed,
                    "training_mode": config.training_mode,
                    "layout_seed": config.layout_seed,
                    "requested_timesteps": config.total_timesteps,
                    "completed_timesteps": model.num_timesteps,
                    "learning_rate": config.learning_rate,
                    "ppo_rollout_steps": PPO_ROLLOUT_STEPS,
                    "ppo_batch_size": int(model.batch_size),
                    "ppo_entropy_coef": config.ent_coef,
                    "ppo_epochs": config.n_epochs,
                    "ppo_adam_eps": config.adam_eps,
                    "ppo_gamma": config.gamma,
                    "n_envs": config.n_envs,
                    "effective_ppo": _effective_ppo_settings(model),
                    "initialization": initialization_provenance,
                    "development_evaluation": {
                        "curve_path": (
                            str(development_curve_path)
                            if development_curve_path is not None else None
                        ),
                        "evaluation_seconds": (
                            development_callback.evaluation_seconds
                            if development_callback is not None else 0.0
                        ),
                    },
                    "representation_snapshots": (
                        str(representation_snapshot_manifest_path)
                        if representation_snapshot_manifest_path is not None
                        else None
                    ),
                    "started_at": started_at.isoformat(),
                    "finished_at": finished_at.isoformat(),
                    "elapsed_seconds": round(elapsed_seconds, 6),
                    "checkpoint_path": str(checkpoint_path),
                    "checkpoint_sha256": checkpoint_hash,
                    "tensorboard_path": str(tensorboard_path),
                    "monitor_path": str(experiment_paths.monitor_directory),
                    "experiment": {
                        "root": str(experiment_paths.root),
                        "manifest_path": str(experiment_paths.manifest_json),
                        "configuration_path": str(
                            experiment_paths.configuration_json
                        ),
                        "training_summary_path": str(
                            experiment_paths.training_summary_json
                        ),
                        "artifact_table_path": str(
                            experiment_paths.artifact_table_csv
                        ),
                        "evaluation_directory": str(
                            experiment_paths.evaluation_directory
                        ),
                    },
                    "training_metrics": training_metrics,
                    "runtime_context": runtime_context,
                    "source_revision": source_revision,
                    **study_metadata,
                },
            )
        except Exception:
            checkpoint_path.unlink(missing_ok=True)
            raise
        summary = TrainingSummary(
            checkpoint_path=checkpoint_path,
            metadata_path=metadata_path,
            requested_timesteps=config.total_timesteps,
            completed_timesteps=model.num_timesteps,
            learning_rate=config.learning_rate,
            n_envs=config.n_envs,
            n_epochs=config.n_epochs,
            adam_eps=config.adam_eps,
            gamma=config.gamma,
            ent_coef=config.ent_coef,
            seed=config.seed,
            environment=config.environment,
            training_mode=config.training_mode,
            layout_seed=config.layout_seed,
            run_id=run_id,
            tensorboard_path=tensorboard_path,
            experiment_path=experiment_paths.root,
            experiment_manifest_path=experiment_paths.manifest_json,
            configuration_path=experiment_paths.configuration_json,
            monitor_path=experiment_paths.monitor_directory,
            training_metrics=training_metrics,
            elapsed_seconds=elapsed_seconds,
            initialization=initialization_provenance,
            initialization_checkpoint_path=initialization_checkpoint_path,
            initialization_checkpoint_sha256=initialization_checkpoint_hash,
            development_curve_path=development_curve_path,
            evaluation_overhead_seconds=(
                development_callback.evaluation_seconds
                if development_callback is not None else 0.0
            ),
            representation_snapshot_manifest_path=(
                representation_snapshot_manifest_path
            ),
        )
        save_completed_experiment(
            experiment_paths,
            run_id=run_id,
            summary={
                "status": "succeeded",
                "started_at": started_at.isoformat(),
                "finished_at": finished_at.isoformat(),
                **summary.to_dict(),
                "checkpoint_sha256": checkpoint_hash,
            },
            manifest={
                "status": "succeeded",
                "created_at": started_at.isoformat(),
                "completed_at": finished_at.isoformat(),
                "configuration": {
                    "json": str(experiment_paths.configuration_json),
                    "csv": str(experiment_paths.configuration_csv),
                },
                "training_summary": {
                    "json": str(experiment_paths.training_summary_json),
                    "csv": str(experiment_paths.training_summary_csv),
                },
                "artifact_table": str(experiment_paths.artifact_table_csv),
                "checkpoint": {
                    "path": str(checkpoint_path),
                    "metadata_path": str(metadata_path),
                    "sha256": checkpoint_hash,
                },
                "initialization": initialization_provenance,
                "initialization_checkpoint": (
                    {
                        "path": str(initialization_checkpoint_path),
                        "metadata_path": str(initialization_metadata_path),
                        "sha256": initialization_checkpoint_hash,
                    }
                    if initialization_checkpoint_path is not None else None
                ),
                "development_curve": (
                    str(development_curve_path)
                    if development_curve_path is not None else None
                ),
                "representation_snapshots": (
                    str(representation_snapshot_manifest_path)
                    if representation_snapshot_manifest_path is not None
                    else None
                ),
                "tensorboard_path": str(tensorboard_path),
                "monitor_path": str(experiment_paths.monitor_directory),
                "evaluation": {
                    "directory": str(experiment_paths.evaluation_directory),
                    "linkage": (
                        "Evaluation V1 reports reference this run_id and "
                        "checkpoint SHA-256."
                    ),
                },
                "runtime_context": runtime_context,
                "source_revision": source_revision,
                **study_metadata,
            },
            artifacts=_experiment_artifact_rows(
                run_id=run_id,
                experiment_paths=experiment_paths,
                checkpoint_path=checkpoint_path,
                metadata_path=metadata_path,
                tensorboard_path=tensorboard_path,
                initialization_checkpoint_path=(
                    initialization_checkpoint_path
                ),
                initialization_metadata_path=initialization_metadata_path,
                development_curve_path=development_curve_path,
                representation_snapshot_manifest_path=(
                    representation_snapshot_manifest_path
                ),
            ),
            schema_version=artifact_schema,
        )
        return summary
    finally:
        environment.close()


def check_transfer_compatibility(config: TrainingConfig) -> CompatibilityReport:
    """Run strict preflight without reserving artifacts or starting training."""
    if not is_transfer_initialization(config.initialization):
        raise ValueError("Transfer preflight requires a source-policy initialization.")
    target_definition = require_released_task(config.environment.task_id)

    def environment_factory():
        return make_environment(
            config.environment,
            observation_mode="flat",
            fixed_layout_seed=(
                config.layout_seed
                if config.training_mode == FIXED_LAYOUT_TRAINING_MODE
                else None
            ),
        )

    environment = make_vec_env(
        environment_factory,
        n_envs=config.n_envs,
        vec_env_cls=DummyVecEnv,
    )
    try:
        model = _create_ppo(environment, config)
        if isinstance(config.initialization, FullPolicyInitialization):
            return preflight_full_policy(
                config.initialization,
                model,
                target_observation_contract_id=(
                    target_definition.observation_contract_id
                ),
                target_action_contract_id=target_definition.action_contract_id,
            )
        return preflight_selective_policy(
            config.initialization,
            model,
            target_observation_contract_id=(
                target_definition.observation_contract_id
            ),
            target_action_contract_id=target_definition.action_contract_id,
        )
    finally:
        environment.close()


def check_transfer_preflight(
    config: TrainingConfig,
    *,
    report_root: str | Path = DEFAULT_EVALUATION_V2_REPORT_ROOT,
) -> TransferPreflightReport:
    """Return request-bound technical and source-competence decisions."""
    if not is_transfer_initialization(config.initialization):
        raise ValueError("Transfer preflight requires a source-policy initialization.")
    target_definition = require_released_task(config.environment.task_id)
    technical = check_transfer_compatibility(config)
    competence = assess_source_competence(
        config.initialization.source_checkpoint_path,
        report_root=report_root,
    )
    return build_transfer_preflight_report(
        initialization=config.initialization,
        technical_compatibility=technical,
        source_competence=competence,
        target_task_id=target_definition.task_id,
        target_task_contract_digest=(
            target_definition.contract_digest or ""
        ),
        request_payload=_transfer_request_identity(config),
    )


def _transfer_request_identity(config: TrainingConfig) -> dict[str, Any]:
    """Return semantic job fields used to invalidate a cached preflight."""
    payload = config.to_dict()
    for key in (
        "checkpoint_root",
        "tensorboard_root",
        "experiment_root",
        "run_id",
    ):
        payload.pop(key, None)
    return payload


def _create_ppo(environment: gym.Env, config: TrainingConfig) -> PPO:
    """Create the PPO baseline aligned with reference rl-starter-files."""
    return PPO(
        "MlpPolicy",
        environment,
        learning_rate=config.learning_rate,
        n_steps=PPO_ROLLOUT_STEPS,
        batch_size=min(PPO_BATCH_SIZE, PPO_ROLLOUT_STEPS * config.n_envs),
        n_epochs=config.n_epochs,
        ent_coef=config.ent_coef,
        gamma=config.gamma,
        seed=config.seed,
        device="cpu",
        verbose=0,
        tensorboard_log=str(config.tensorboard_root),
        policy_kwargs={"optimizer_kwargs": {"eps": config.adam_eps}},
    )


def _refuse_artifact_collisions(*paths: Path) -> None:
    """Reject reuse of a run identity before any expensive work begins."""
    for path in paths:
        if path.exists():
            raise FileExistsError(f"Training run artifact already exists: {path}")


def _reserve_training_run(
    checkpoint_path: Path,
    *,
    tensorboard_root: Path,
    run_id: str,
    tensorboard_path: Path,
    experiment_paths: ExperimentPaths,
) -> Path:
    """Atomically reserve one run identity across checkpoint and log roots."""
    run_id = validate_training_run_id(run_id)
    metadata_path = checkpoint_metadata_path(checkpoint_path)
    _refuse_artifact_collisions(
        checkpoint_path,
        metadata_path,
        checkpoint_path.parent,
        tensorboard_path,
        experiment_paths.root,
    )

    reservation_root = Path(tensorboard_root) / ".run-reservations"
    reservation_root.mkdir(parents=True, exist_ok=True)
    reservation_path = reservation_root / f"{run_id}.reserved"
    try:
        with reservation_path.open("x", encoding="utf-8") as reservation_file:
            reservation_file.write(f"{run_id}\n")
    except FileExistsError as error:
        raise FileExistsError(
            f"Training run ID is already reserved: {run_id}"
        ) from error

    try:
        checkpoint_path.parent.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_path.parent.mkdir(exist_ok=False)
        reserve_experiment_directory(experiment_paths)
    except Exception:
        experiment_paths.monitor_directory.rmdir() if (
            experiment_paths.monitor_directory.is_dir()
            and not any(experiment_paths.monitor_directory.iterdir())
        ) else None
        experiment_paths.root.rmdir() if (
            experiment_paths.root.is_dir()
            and not any(experiment_paths.root.iterdir())
        ) else None
        checkpoint_path.parent.rmdir() if (
            checkpoint_path.parent.is_dir()
            and not any(checkpoint_path.parent.iterdir())
        ) else None
        reservation_path.unlink(missing_ok=True)
        raise
    return reservation_path


def _effective_ppo_settings(model: PPO) -> dict[str, Any]:
    """Return the effective PPO settings required to interpret a training run."""
    clip_range_vf = (
        None
        if model.clip_range_vf is None
        else float(model.clip_range_vf(1.0))
    )
    return {
        "policy": model.policy_class.__name__,
        "learning_rate": float(model.learning_rate),
        "n_steps": int(model.n_steps),
        "batch_size": int(model.batch_size),
        "n_epochs": int(model.n_epochs),
        "gamma": float(model.gamma),
        "gae_lambda": float(model.gae_lambda),
        "clip_range": float(model.clip_range(1.0)),
        "clip_range_vf": clip_range_vf,
        "normalize_advantage": bool(model.normalize_advantage),
        "entropy_coefficient": float(model.ent_coef),
        "value_function_coefficient": float(model.vf_coef),
        "max_gradient_norm": float(model.max_grad_norm),
        "use_sde": bool(model.use_sde),
        "sde_sample_frequency": int(model.sde_sample_freq),
        "target_kl": model.target_kl,
        "optimizer_epsilon": float(model.policy.optimizer.defaults["eps"]),
        "n_envs": int(model.n_envs),
        "device": str(model.device),
    }


def _experiment_artifact_rows(
    *,
    run_id: str,
    experiment_paths: ExperimentPaths,
    checkpoint_path: Path,
    metadata_path: Path,
    tensorboard_path: Path,
    initialization_checkpoint_path: Path | None = None,
    initialization_metadata_path: Path | None = None,
    development_curve_path: Path | None = None,
    representation_snapshot_manifest_path: Path | None = None,
) -> tuple[dict[str, str], ...]:
    """Return the stable machine-readable artifact inventory."""
    optional = []
    if initialization_checkpoint_path is not None:
        optional.extend((
            (
                "initialization_checkpoint",
                initialization_checkpoint_path,
                "sb3-zip",
                "training.checkpoints",
            ),
            (
                "initialization_checkpoint_metadata",
                initialization_metadata_path,
                "json",
                "training.checkpoints",
            ),
        ))
    if development_curve_path is not None:
        optional.append((
            "development_curve",
            development_curve_path,
            "json",
            "training.development",
        ))
    if representation_snapshot_manifest_path is not None:
        optional.append((
            "representation_snapshots",
            representation_snapshot_manifest_path,
            "json+torch-state-dict",
            "training.representation_snapshots",
        ))
    return tuple(
        {
            "run_id": run_id,
            "artifact_type": artifact_type,
            "path": str(path),
            "format": artifact_format,
            "owner": owner,
        }
        for artifact_type, path, artifact_format, owner in (
            (
                "configuration_json",
                experiment_paths.configuration_json,
                "json",
                "training.experiments",
            ),
            (
                "configuration_csv",
                experiment_paths.configuration_csv,
                "csv",
                "training.experiments",
            ),
            ("checkpoint", checkpoint_path, "sb3-zip", "training.checkpoints"),
            (
                "checkpoint_metadata",
                metadata_path,
                "json",
                "training.checkpoints",
            ),
            ("tensorboard", tensorboard_path, "event-protobuf", "stable-baselines3"),
            (
                "monitor",
                experiment_paths.monitor_directory,
                "monitor-csv",
                "stable-baselines3",
            ),
            (
                "training_summary_json",
                experiment_paths.training_summary_json,
                "json",
                "training.experiments",
            ),
            (
                "training_summary_csv",
                experiment_paths.training_summary_csv,
                "csv",
                "training.experiments",
            ),
            (
                "artifact_table",
                experiment_paths.artifact_table_csv,
                "csv",
                "training.experiments",
            ),
            (
                "experiment_manifest",
                experiment_paths.manifest_json,
                "json",
                "training.experiments",
            ),
            (
                "evaluation_directory",
                experiment_paths.evaluation_directory,
                "evaluation-v1-json",
                "training.evaluation",
            ),
            *optional,
        )
    )


def _environment_action_labels(environment: DummyVecEnv) -> dict[int, str]:
    """Read action names from the environment without hardcoding MiniGrid."""
    try:
        action_enum = environment.get_attr("actions")[0]
        return {int(member): str(member.name) for member in action_enum}
    except (AttributeError, IndexError, TypeError):
        return {
            index: "action"
            for index in range(int(getattr(environment.action_space, "n", 0)))
        }


def _runtime_context() -> dict[str, Any]:
    """Capture training-time runtime versions that affect reproducibility."""
    distributions = (
        "gymnasium",
        "minigrid",
        "numpy",
        "stable-baselines3",
        "tensorboard",
        "torch",
    )
    return {
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "packages": {
            name: _package_version(name)
            for name in distributions
        },
    }


def _package_version(distribution: str) -> str:
    """Return an installed distribution version without making it mandatory."""
    try:
        return package_metadata.version(distribution)
    except package_metadata.PackageNotFoundError:
        return "unavailable"
