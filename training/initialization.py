"""Explicit scratch, full-policy, and selective-policy contracts."""

from __future__ import annotations

import hashlib
import json
import random
import re
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Mapping

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO

from environments.catalog import require_released_task
from training.checkpoints import (
    DEFAULT_CHECKPOINT_ROOT,
    checkpoint_sha256,
    load_checkpoint_metadata,
    load_model,
)
from training.initialization_methods import (
    ACTOR_BODY_FRESH_HEAD_COPIED_CRITIC,
    ACTOR_BODY_FINETUNE,
    FROZEN_ACTOR_PROBE,
    FULL_ACTOR_FRESH_CRITIC,
    SELECTIVE_INITIALIZATION_METHODS,
)

SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
OBSERVATION_CONTRACT_ID = "minigrid-egocentric-flat148-v1"
ACTION_CONTRACT_ID = "minigrid-discrete7-v1"
ACTION_ORDER = ("left", "right", "forward", "pickup", "drop", "toggle", "done")
POLICY_COPY_VERSION = "full-policy-state-dict-v1"
SELECTIVE_POLICY_COPY_VERSION = "selective-policy-state-dict-v1"
STUDY_RELATIONS = frozenset(
    {"primary", "control", "warm-start", "exploratory"}
)
StudyRelation = Literal[
    "primary", "control", "warm-start", "exploratory"
]


@dataclass(frozen=True, slots=True)
class ScratchInitialization:
    method: str = "scratch"

    def to_dict(self) -> dict[str, Any]:
        return {"method": self.method}


@dataclass(frozen=True, slots=True)
class FullPolicyInitialization:
    source_checkpoint_path: Path
    expected_source_sha256: str
    source_run_id: str
    source_task_id: str
    source_task_contract_digest: str
    observation_contract_id: str
    action_contract_id: str
    policy_architecture_contract: Mapping[str, Any]
    study_relation: StudyRelation = "exploratory"
    method: str = "full_policy"

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "source_checkpoint_path", Path(self.source_checkpoint_path)
        )
        if SHA256_PATTERN.fullmatch(self.expected_source_sha256) is None:
            raise ValueError("expected_source_sha256 must be a lowercase SHA-256.")
        for name in (
            "source_run_id", "source_task_id", "source_task_contract_digest",
            "observation_contract_id", "action_contract_id",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string.")
        if SHA256_PATTERN.fullmatch(self.source_task_contract_digest) is None:
            raise ValueError(
                "source_task_contract_digest must be a lowercase SHA-256."
            )
        object.__setattr__(
            self,
            "policy_architecture_contract",
            dict(self.policy_architecture_contract),
        )
        if self.study_relation not in STUDY_RELATIONS:
            raise ValueError(
                "study_relation must be primary, control, warm-start, or "
                "exploratory."
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "source_checkpoint_path": str(self.source_checkpoint_path),
            "expected_source_sha256": self.expected_source_sha256,
            "source_run_id": self.source_run_id,
            "source_task_id": self.source_task_id,
            "source_task_contract_digest": self.source_task_contract_digest,
            "observation_contract_id": self.observation_contract_id,
            "action_contract_id": self.action_contract_id,
            "policy_architecture_contract": dict(
                self.policy_architecture_contract
            ),
            "study_relation": self.study_relation,
        }


@dataclass(frozen=True, slots=True)
class SelectivePolicyInitialization:
    """One explicit component-selective source-policy initialization contract."""

    source_checkpoint_path: Path
    expected_source_sha256: str
    source_run_id: str
    source_task_id: str
    source_task_contract_digest: str
    observation_contract_id: str
    action_contract_id: str
    policy_architecture_contract: Mapping[str, Any]
    method: str
    study_relation: StudyRelation = "exploratory"

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "source_checkpoint_path", Path(self.source_checkpoint_path)
        )
        if self.method not in SELECTIVE_INITIALIZATION_METHODS:
            raise ValueError(
                "Selective method must be actor_body_finetune_v1, "
                "frozen_actor_probe_v1, full_actor_fresh_critic_v1, or "
                "actor_body_fresh_head_copied_critic_v1."
            )
        if SHA256_PATTERN.fullmatch(self.expected_source_sha256) is None:
            raise ValueError("expected_source_sha256 must be a lowercase SHA-256.")
        for name in (
            "source_run_id", "source_task_id", "source_task_contract_digest",
            "observation_contract_id", "action_contract_id",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string.")
        if SHA256_PATTERN.fullmatch(self.source_task_contract_digest) is None:
            raise ValueError(
                "source_task_contract_digest must be a lowercase SHA-256."
            )
        object.__setattr__(
            self,
            "policy_architecture_contract",
            dict(self.policy_architecture_contract),
        )
        if self.study_relation not in STUDY_RELATIONS:
            raise ValueError(
                "study_relation must be primary, control, warm-start, or "
                "exploratory."
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "source_checkpoint_path": str(self.source_checkpoint_path),
            "expected_source_sha256": self.expected_source_sha256,
            "source_run_id": self.source_run_id,
            "source_task_id": self.source_task_id,
            "source_task_contract_digest": self.source_task_contract_digest,
            "observation_contract_id": self.observation_contract_id,
            "action_contract_id": self.action_contract_id,
            "policy_architecture_contract": dict(
                self.policy_architecture_contract
            ),
            "study_relation": self.study_relation,
        }


SourcePolicyInitialization = (
    FullPolicyInitialization | SelectivePolicyInitialization
)
Initialization = ScratchInitialization | SourcePolicyInitialization


def is_transfer_initialization(value: object) -> bool:
    return isinstance(
        value,
        (FullPolicyInitialization, SelectivePolicyInitialization),
    )


@dataclass(frozen=True, slots=True)
class CompatibilityCheck:
    code: str
    passed: bool
    detail: str


@dataclass(frozen=True, slots=True)
class CompatibilityReport:
    compatible: bool
    checks: tuple[CompatibilityCheck, ...]
    digest: str
    source_checkpoint_sha256: str | None
    source_policy_contract: Mapping[str, Any] | None
    target_policy_contract: Mapping[str, Any]

    @property
    def failure_codes(self) -> tuple[str, ...]:
        return tuple(check.code for check in self.checks if not check.passed)

    def to_dict(self) -> dict[str, Any]:
        return {
            "compatible": self.compatible,
            "checks": [asdict(check) for check in self.checks],
            "failure_codes": list(self.failure_codes),
            "digest": self.digest,
            "source_checkpoint_sha256": self.source_checkpoint_sha256,
            "source_policy_contract": (
                dict(self.source_policy_contract)
                if self.source_policy_contract is not None else None
            ),
            "target_policy_contract": dict(self.target_policy_contract),
        }


class TransferCompatibilityError(ValueError):
    def __init__(self, report: CompatibilityReport) -> None:
        self.report = report
        super().__init__(
            "Transfer compatibility failed: "
            + ", ".join(report.failure_codes)
        )


@dataclass(frozen=True, slots=True)
class PolicyCopyResult:
    compatibility: CompatibilityReport
    copied_key_count: int
    copied_tensor_count: int
    copied_tensors_equal: bool
    optimizer_fresh: bool
    counters_fresh: bool
    rollout_buffer_fresh: bool
    copy_method_version: str = POLICY_COPY_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "compatibility": self.compatibility.to_dict(),
            "copied_key_count": self.copied_key_count,
            "copied_tensor_count": self.copied_tensor_count,
            "copied_tensors_equal": self.copied_tensors_equal,
            "optimizer_fresh": self.optimizer_fresh,
            "counters_fresh": self.counters_fresh,
            "rollout_buffer_fresh": self.rollout_buffer_fresh,
            "copy_method_version": self.copy_method_version,
        }


@dataclass(frozen=True, slots=True)
class SelectivePolicyCopyResult:
    compatibility: CompatibilityReport
    method: str
    copied_keys: tuple[str, ...]
    fresh_keys: tuple[str, ...]
    frozen_keys: tuple[str, ...]
    copied_tensors_equal: bool
    fresh_group_differs_from_source: bool
    optimizer_fresh: bool
    optimizer_excludes_frozen: bool
    counters_fresh: bool
    rollout_buffer_fresh: bool
    copied_group_sha256: str
    fresh_group_sha256: str
    source_fresh_group_sha256: str
    frozen_group_sha256: str | None
    copy_method_version: str = SELECTIVE_POLICY_COPY_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "compatibility": self.compatibility.to_dict(),
            "method": self.method,
            "copied_key_count": len(self.copied_keys),
            "copied_tensor_count": len(self.copied_keys),
            "copied_keys": list(self.copied_keys),
            "fresh_key_count": len(self.fresh_keys),
            "fresh_keys": list(self.fresh_keys),
            "frozen_key_count": len(self.frozen_keys),
            "frozen_keys": list(self.frozen_keys),
            "copied_tensors_equal": self.copied_tensors_equal,
            "fresh_group_differs_from_source": (
                self.fresh_group_differs_from_source
            ),
            "optimizer_fresh": self.optimizer_fresh,
            "optimizer_excludes_frozen": self.optimizer_excludes_frozen,
            "counters_fresh": self.counters_fresh,
            "rollout_buffer_fresh": self.rollout_buffer_fresh,
            "copied_group_sha256": self.copied_group_sha256,
            "fresh_group_sha256": self.fresh_group_sha256,
            "source_fresh_group_sha256": self.source_fresh_group_sha256,
            "frozen_group_sha256": self.frozen_group_sha256,
            "copy_method_version": self.copy_method_version,
        }


def initialization_from_dict(payload: object) -> Initialization:
    if payload is None:
        return ScratchInitialization()
    if not isinstance(payload, Mapping):
        raise ValueError("initialization must be a mapping.")
    values = dict(payload)
    method = values.pop("method", None)
    if method == "scratch":
        if values:
            raise ValueError("Scratch initialization has unexpected fields.")
        return ScratchInitialization()
    if method == "full_policy":
        return FullPolicyInitialization(**values)
    if method in SELECTIVE_INITIALIZATION_METHODS:
        return SelectivePolicyInitialization(method=method, **values)
    raise ValueError(f"Unsupported initialization method: {method!r}.")


def policy_architecture_contract(model: PPO) -> dict[str, Any]:
    policy = model.policy
    state = policy.state_dict()
    payload = {
        "algorithm": type(model).__name__,
        "policy_class": _qualified_name(type(policy)),
        "feature_extractor": _qualified_name(type(policy.features_extractor)),
        "net_arch": policy.net_arch,
        "activation": _qualified_name(policy.activation_fn),
        "distribution": _qualified_name(type(policy.action_dist)),
        "state_dict": {
            key: {"shape": list(tensor.shape), "dtype": str(tensor.dtype)}
            for key, tensor in state.items()
        },
    }
    payload["digest"] = _digest(payload)
    return payload


def build_full_policy_initialization(
    checkpoint_path: str | Path,
    *,
    study_relation: StudyRelation = "exploratory",
) -> FullPolicyInitialization:
    path = Path(checkpoint_path).resolve()
    metadata = load_checkpoint_metadata(path)
    environment_payload = metadata.get("environment")
    if not isinstance(environment_payload, Mapping):
        raise ValueError("Source metadata is missing its environment contract.")
    task_id = environment_payload.get("task_id")
    if not isinstance(task_id, str):
        raise ValueError("Full-policy transfer requires a governed source task.")
    definition = require_released_task(task_id)
    model = load_model(path)
    return FullPolicyInitialization(
        source_checkpoint_path=path,
        expected_source_sha256=checkpoint_sha256(path),
        source_run_id=str(metadata.get("run_id", "")),
        source_task_id=task_id,
        source_task_contract_digest=str(metadata.get("task_contract_digest", "")),
        observation_contract_id=definition.observation_contract_id,
        action_contract_id=definition.action_contract_id,
        policy_architecture_contract=policy_architecture_contract(model),
        study_relation=study_relation,
    )


def build_selective_policy_initialization(
    checkpoint_path: str | Path,
    *,
    method: str,
    study_relation: StudyRelation = "exploratory",
) -> SelectivePolicyInitialization:
    """Build one governed C2/C3/C2a/C2b request from source metadata."""
    full = build_full_policy_initialization(
        checkpoint_path,
        study_relation=study_relation,
    )
    return SelectivePolicyInitialization(
        source_checkpoint_path=full.source_checkpoint_path,
        expected_source_sha256=full.expected_source_sha256,
        source_run_id=full.source_run_id,
        source_task_id=full.source_task_id,
        source_task_contract_digest=full.source_task_contract_digest,
        observation_contract_id=full.observation_contract_id,
        action_contract_id=full.action_contract_id,
        policy_architecture_contract=full.policy_architecture_contract,
        method=method,
        study_relation=full.study_relation,
    )


def discover_full_policy_sources(
    *,
    source_task_id: str = "door_key-v1",
    checkpoint_root: str | Path = DEFAULT_CHECKPOINT_ROOT,
) -> tuple[Path, ...]:
    """Return competent canonical governed checkpoints for one source task."""
    # The richer governance module imports the initialization contracts, so
    # keep this compatibility wrapper lazy and preserve its historical API.
    from training.transfer_governance import discover_governed_transfer_sources

    discovery = discover_governed_transfer_sources(
        source_task_id=source_task_id,
        checkpoint_root=checkpoint_root,
    )
    return tuple(source.checkpoint_path for source in discovery.included)


def preflight_full_policy(
    initialization: FullPolicyInitialization,
    target_model: PPO,
    *,
    target_observation_contract_id: str,
    target_action_contract_id: str,
    target_action_order: tuple[str, ...] = ACTION_ORDER,
) -> CompatibilityReport:
    """Validate compatibility without advancing global training RNG streams."""
    with _preserve_global_rng_state():
        return _preflight_full_policy(
            initialization,
            target_model,
            target_observation_contract_id=target_observation_contract_id,
            target_action_contract_id=target_action_contract_id,
            target_action_order=target_action_order,
        )


def preflight_selective_policy(
    initialization: SelectivePolicyInitialization,
    target_model: PPO,
    *,
    target_observation_contract_id: str,
    target_action_contract_id: str,
    target_action_order: tuple[str, ...] = ACTION_ORDER,
) -> CompatibilityReport:
    """Apply the full source/space/architecture preflight to selective copies."""
    with _preserve_global_rng_state():
        return _preflight_full_policy(
            initialization,
            target_model,
            target_observation_contract_id=target_observation_contract_id,
            target_action_contract_id=target_action_contract_id,
            target_action_order=target_action_order,
        )


def _preflight_full_policy(
    initialization: SourcePolicyInitialization,
    target_model: PPO,
    *,
    target_observation_contract_id: str,
    target_action_contract_id: str,
    target_action_order: tuple[str, ...] = ACTION_ORDER,
) -> CompatibilityReport:
    """Validate source identity and exact target-policy compatibility."""
    checks: list[CompatibilityCheck] = []

    def record(code: str, passed: bool, detail: str) -> None:
        checks.append(CompatibilityCheck(code, bool(passed), detail))

    path = initialization.source_checkpoint_path
    exists = path.is_file()
    record("SOURCE_CHECKPOINT_EXISTS", exists, str(path))
    metadata: dict[str, Any] = {}
    actual_hash: str | None = None
    source_model: PPO | None = None
    source_contract: dict[str, Any] | None = None
    if exists:
        try:
            metadata = load_checkpoint_metadata(path)
            record(
                "SOURCE_METADATA_SCHEMA",
                metadata.get("schema_version") == 3,
                f"schema={metadata.get('schema_version')!r}",
            )
        except Exception as error:
            record("SOURCE_METADATA_SCHEMA", False, str(error))
        try:
            actual_hash = checkpoint_sha256(path)
            record(
                "SOURCE_SHA_MATCH",
                actual_hash == initialization.expected_source_sha256,
                (
                    f"expected={initialization.expected_source_sha256} "
                    f"actual={actual_hash}"
                ),
            )
        except Exception as error:
            record("SOURCE_SHA_MATCH", False, str(error))
    else:
        record("SOURCE_METADATA_SCHEMA", False, "checkpoint missing")
        record("SOURCE_SHA_MATCH", False, "checkpoint missing")

    record(
        "SOURCE_RUN_MATCH",
        metadata.get("run_id") == initialization.source_run_id,
        f"expected={initialization.source_run_id!r} actual={metadata.get('run_id')!r}",
    )
    environment_payload = metadata.get("environment")
    source_task = (
        environment_payload.get("task_id")
        if isinstance(environment_payload, Mapping) else None
    )
    record(
        "SOURCE_TASK_MATCH",
        source_task == initialization.source_task_id,
        f"expected={initialization.source_task_id!r} actual={source_task!r}",
    )
    record(
        "SOURCE_CONTRACT_MATCH",
        metadata.get("task_contract_digest")
        == initialization.source_task_contract_digest,
        (
            f"expected={initialization.source_task_contract_digest!r} "
            f"actual={metadata.get('task_contract_digest')!r}"
        ),
    )
    governed_complete, governed_detail = _governed_source_complete(
        metadata, initialization, actual_hash
    )
    record("SOURCE_GOVERNED_COMPLETE", governed_complete, governed_detail)

    try:
        source_definition = require_released_task(initialization.source_task_id)
        record(
            "SOURCE_RELEASED_CONTRACT",
            initialization.source_task_contract_digest
            == metadata.get("task_contract_digest")
            == source_definition.contract_digest,
            (
                f"declared={initialization.source_task_contract_digest!r} "
                f"metadata={metadata.get('task_contract_digest')!r} "
                f"released={source_definition.contract_digest!r}"
            ),
        )
        record(
            "OBSERVATION_CONTRACT_ID",
            initialization.observation_contract_id
            == source_definition.observation_contract_id
            == target_observation_contract_id,
            (
                f"source={initialization.observation_contract_id!r} "
                f"target={target_observation_contract_id!r}"
            ),
        )
        record(
            "ACTION_CONTRACT_ID",
            initialization.action_contract_id
            == source_definition.action_contract_id
            == target_action_contract_id,
            (
                f"source={initialization.action_contract_id!r} "
                f"target={target_action_contract_id!r}"
            ),
        )
    except Exception as error:
        record("SOURCE_RELEASED_CONTRACT", False, str(error))
        record("OBSERVATION_CONTRACT_ID", False, str(error))
        record("ACTION_CONTRACT_ID", False, str(error))

    target_obs = _observation_signature(target_model.observation_space)
    target_action = _action_signature(
        target_model.action_space, ordering=target_action_order
    )
    if exists:
        try:
            source_model = load_model(path)
            source_obs = _observation_signature(source_model.observation_space)
            source_action = _action_signature(source_model.action_space)
            record(
                "OBSERVATION_SPACE_EXACT",
                source_obs == target_obs == {
                    "space_type": "Box", "shape": [148], "dtype": "uint8",
                    "low": 0, "high": 255,
                },
                f"source={source_obs} target={target_obs}",
            )
            record(
                "ACTION_SPACE_EXACT",
                source_action == target_action == {
                    "space_type": "Discrete", "count": 7, "start": 0,
                    "ordering": list(ACTION_ORDER),
                },
                f"source={source_action} target={target_action}",
            )
            source_contract = policy_architecture_contract(source_model)
            target_contract = policy_architecture_contract(target_model)
            record(
                "SOURCE_POLICY_DECLARATION",
                dict(initialization.policy_architecture_contract) == source_contract,
                "serialized source policy contract matches checkpoint",
            )
            record(
                "POLICY_ARCHITECTURE_EXACT",
                source_contract == target_contract,
                (
                    f"source_digest={source_contract['digest']} "
                    f"target_digest={target_contract['digest']}"
                ),
            )
        except Exception as error:
            target_contract = policy_architecture_contract(target_model)
            record("OBSERVATION_SPACE_EXACT", False, str(error))
            record("ACTION_SPACE_EXACT", False, str(error))
            record("SOURCE_POLICY_DECLARATION", False, str(error))
            record("POLICY_ARCHITECTURE_EXACT", False, str(error))
    else:
        target_contract = policy_architecture_contract(target_model)
        for code in (
            "OBSERVATION_SPACE_EXACT", "ACTION_SPACE_EXACT",
            "SOURCE_POLICY_DECLARATION", "POLICY_ARCHITECTURE_EXACT",
        ):
            record(code, False, "checkpoint missing")

    payload = {
        "checks": [asdict(check) for check in checks],
        "source_checkpoint_sha256": actual_hash,
        "source_policy_contract": source_contract,
        "target_policy_contract": target_contract,
    }
    return CompatibilityReport(
        compatible=all(check.passed for check in checks),
        checks=tuple(checks),
        digest=_digest(payload),
        source_checkpoint_sha256=actual_hash,
        source_policy_contract=source_contract,
        target_policy_contract=target_contract,
    )


def copy_full_policy(
    target_model: PPO,
    initialization: FullPolicyInitialization,
    *,
    target_observation_contract_id: str,
    target_action_contract_id: str,
    target_action_order: tuple[str, ...] = ACTION_ORDER,
) -> PolicyCopyResult:
    """Copy policy tensors while preserving the fresh target RNG schedule."""
    with _preserve_global_rng_state():
        return _copy_full_policy(
            target_model,
            initialization,
            target_observation_contract_id=target_observation_contract_id,
            target_action_contract_id=target_action_contract_id,
            target_action_order=target_action_order,
        )


def _copy_full_policy(
    target_model: PPO,
    initialization: FullPolicyInitialization,
    *,
    target_observation_contract_id: str,
    target_action_contract_id: str,
    target_action_order: tuple[str, ...] = ACTION_ORDER,
) -> PolicyCopyResult:
    report = preflight_full_policy(
        initialization,
        target_model,
        target_observation_contract_id=target_observation_contract_id,
        target_action_contract_id=target_action_contract_id,
        target_action_order=target_action_order,
    )
    if not report.compatible:
        raise TransferCompatibilityError(report)
    source_model = load_model(initialization.source_checkpoint_path)
    source_state = {
        key: tensor.detach().cpu().clone()
        for key, tensor in source_model.policy.state_dict().items()
    }
    if target_model.policy.optimizer.state:
        raise RuntimeError("Fresh target optimizer unexpectedly has state.")
    target_model.policy.load_state_dict(source_state, strict=True)
    target_state = target_model.policy.state_dict()
    equal = (
        tuple(source_state) == tuple(target_state)
        and all(
            torch.equal(source_state[key], target_state[key].detach().cpu())
            for key in source_state
        )
    )
    optimizer_fresh = not target_model.policy.optimizer.state
    counters_fresh = (
        target_model.num_timesteps == 0
        and target_model._episode_num == 0
        and target_model._n_updates == 0
    )
    buffer = target_model.rollout_buffer
    rollout_fresh = buffer.pos == 0 and not buffer.full
    if not (equal and optimizer_fresh and counters_fresh and rollout_fresh):
        raise RuntimeError("Full-policy copy violated fresh-target invariants.")
    return PolicyCopyResult(
        compatibility=report,
        copied_key_count=len(source_state),
        copied_tensor_count=len(source_state),
        copied_tensors_equal=equal,
        optimizer_fresh=optimizer_fresh,
        counters_fresh=counters_fresh,
        rollout_buffer_fresh=rollout_fresh,
    )


def _governed_source_complete(
    metadata: Mapping[str, Any],
    initialization: SourcePolicyInitialization,
    actual_hash: str | None,
) -> tuple[bool, str]:
    return governed_source_complete(
        metadata,
        checkpoint_path=initialization.source_checkpoint_path,
        checkpoint_sha256_value=actual_hash,
        source_run_id=initialization.source_run_id,
        source_task_id=initialization.source_task_id,
        source_task_contract_digest=(
            initialization.source_task_contract_digest
        ),
    )


def copy_selective_policy(
    target_model: PPO,
    initialization: SelectivePolicyInitialization,
    *,
    target_observation_contract_id: str,
    target_action_contract_id: str,
    target_action_order: tuple[str, ...] = ACTION_ORDER,
) -> SelectivePolicyCopyResult:
    """Copy the predeclared groups and keep all target training state fresh."""
    with _preserve_global_rng_state():
        report = preflight_selective_policy(
            initialization,
            target_model,
            target_observation_contract_id=target_observation_contract_id,
            target_action_contract_id=target_action_contract_id,
            target_action_order=target_action_order,
        )
        if not report.compatible:
            raise TransferCompatibilityError(report)
        if target_model.policy.optimizer.state:
            raise RuntimeError("Fresh target optimizer unexpectedly has state.")

        source_model = load_model(initialization.source_checkpoint_path)
        source_state = {
            key: tensor.detach().cpu().clone()
            for key, tensor in source_model.policy.state_dict().items()
        }
        target_state = {
            key: tensor.detach().cpu().clone()
            for key, tensor in target_model.policy.state_dict().items()
        }
        copied_keys = tuple(
            key for key in target_state
            if _copy_key_for_method(key, initialization.method)
        )
        fresh_keys = tuple(key for key in target_state if key not in copied_keys)
        if not copied_keys or not fresh_keys:
            raise RuntimeError("Selective copy must have copied and fresh groups.")
        merged = dict(target_state)
        for key in copied_keys:
            merged[key] = source_state[key]
        target_model.policy.load_state_dict(merged, strict=True)

        frozen_keys: tuple[str, ...] = ()
        if initialization.method == FROZEN_ACTOR_PROBE:
            frozen_keys = tuple(
                key for key in copied_keys
                if key.startswith("mlp_extractor.policy_net.")
            )
            frozen = set(frozen_keys)
            for name, parameter in target_model.policy.named_parameters():
                if name in frozen:
                    parameter.requires_grad_(False)
            _rebuild_optimizer_with_trainable_parameters(target_model)

        final_state = target_model.policy.state_dict()
        copied_equal = all(
            torch.equal(source_state[key], final_state[key].detach().cpu())
            for key in copied_keys
        )
        fresh_digest = policy_tensor_digest(target_model, fresh_keys)
        source_fresh_digest = _state_tensor_digest(source_state, fresh_keys)
        fresh_differs = fresh_digest != source_fresh_digest
        frozen_digest = (
            policy_tensor_digest(target_model, frozen_keys)
            if frozen_keys else None
        )
        optimizer_parameter_ids = {
            id(parameter)
            for group in target_model.policy.optimizer.param_groups
            for parameter in group["params"]
        }
        frozen_parameters = {
            id(parameter)
            for name, parameter in target_model.policy.named_parameters()
            if name in set(frozen_keys)
        }
        optimizer_excludes_frozen = not (
            optimizer_parameter_ids & frozen_parameters
        )
        optimizer_fresh = not target_model.policy.optimizer.state
        counters_fresh = (
            target_model.num_timesteps == 0
            and target_model._episode_num == 0
            and target_model._n_updates == 0
        )
        buffer = target_model.rollout_buffer
        rollout_fresh = buffer.pos == 0 and not buffer.full
        if not all((
            copied_equal,
            fresh_differs,
            optimizer_fresh,
            optimizer_excludes_frozen,
            counters_fresh,
            rollout_fresh,
        )):
            raise RuntimeError("Selective copy violated fresh-target invariants.")
        return SelectivePolicyCopyResult(
            compatibility=report,
            method=initialization.method,
            copied_keys=copied_keys,
            fresh_keys=fresh_keys,
            frozen_keys=frozen_keys,
            copied_tensors_equal=copied_equal,
            fresh_group_differs_from_source=fresh_differs,
            optimizer_fresh=optimizer_fresh,
            optimizer_excludes_frozen=optimizer_excludes_frozen,
            counters_fresh=counters_fresh,
            rollout_buffer_fresh=rollout_fresh,
            copied_group_sha256=policy_tensor_digest(target_model, copied_keys),
            fresh_group_sha256=fresh_digest,
            source_fresh_group_sha256=source_fresh_digest,
            frozen_group_sha256=frozen_digest,
        )


def policy_tensor_digest(model: PPO, keys: tuple[str, ...]) -> str:
    """Hash an ordered policy tensor group by name, shape, dtype, and bytes."""
    state = {
        key: tensor.detach().cpu()
        for key, tensor in model.policy.state_dict().items()
    }
    return _state_tensor_digest(state, keys)


def _state_tensor_digest(
    state: Mapping[str, torch.Tensor],
    keys: tuple[str, ...],
) -> str:
    digest = hashlib.sha256()
    for key in keys:
        tensor = state[key].detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(json.dumps(list(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _copy_key_for_method(key: str, method: str) -> bool:
    if method in (ACTOR_BODY_FINETUNE, FROZEN_ACTOR_PROBE):
        return key.startswith("mlp_extractor.policy_net.")
    if method == FULL_ACTOR_FRESH_CRITIC:
        return (
            key.startswith("mlp_extractor.policy_net.")
            or key.startswith("action_net.")
        )
    if method == ACTOR_BODY_FRESH_HEAD_COPIED_CRITIC:
        return (
            key.startswith("mlp_extractor.policy_net.")
            or key.startswith("mlp_extractor.value_net.")
            or key.startswith("value_net.")
        )
    raise ValueError(f"Unsupported selective method: {method!r}.")


def _rebuild_optimizer_with_trainable_parameters(model: PPO) -> None:
    trainable = [
        parameter
        for parameter in model.policy.parameters()
        if parameter.requires_grad
    ]
    if not trainable:
        raise RuntimeError("Selective policy has no trainable parameters.")
    learning_rate = model.lr_schedule(model._current_progress_remaining)
    model.policy.optimizer = model.policy.optimizer_class(
        trainable,
        lr=learning_rate,
        **model.policy.optimizer_kwargs,
    )


@contextmanager
def checkpoint_compatible_optimizer(model: PPO):
    """Temporarily expose all policy parameters for SB3-loadable checkpoints.

    C3 deliberately excludes frozen tensors from its training optimizer.  SB3
    reconstructs an all-parameter optimizer before loading, so serializing the
    reduced parameter group would make an otherwise valid evaluation checkpoint
    unloadable.  This context preserves state for trainable tensors, adds frozen
    tensors with empty state only while saving, then restores the real optimizer.
    """
    optimizer = model.policy.optimizer
    optimized_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    all_parameters = list(model.policy.parameters())
    if optimized_ids == {id(parameter) for parameter in all_parameters}:
        yield
        return

    learning_rate = model.lr_schedule(model._current_progress_remaining)
    compatible = model.policy.optimizer_class(
        all_parameters,
        lr=learning_rate,
        **model.policy.optimizer_kwargs,
    )
    if len(optimizer.param_groups) != 1 or len(compatible.param_groups) != 1:
        raise RuntimeError("Selective checkpoint expects one optimizer group.")
    for key, value in optimizer.param_groups[0].items():
        if key != "params":
            compatible.param_groups[0][key] = deepcopy(value)
    for parameter, state in optimizer.state.items():
        compatible.state[parameter] = deepcopy(state)
    model.policy.optimizer = compatible
    try:
        yield
    finally:
        model.policy.optimizer = optimizer


def governed_source_complete(
    metadata: Mapping[str, Any],
    *,
    checkpoint_path: str | Path,
    checkpoint_sha256_value: str | None,
    source_run_id: str,
    source_task_id: str,
    source_task_contract_digest: str,
) -> tuple[bool, str]:
    """Validate the canonical schema-3 checkpoint/experiment closure."""
    source_path = Path(checkpoint_path).resolve()
    environment = metadata.get("environment")
    if not isinstance(environment, Mapping):
        return False, "source metadata environment missing"
    try:
        definition = require_released_task(source_task_id)
    except Exception as error:
        return False, f"source task is not released: {error}"
    if (
        metadata.get("schema_version") != 3
        or metadata.get("run_id") != source_run_id
        or environment.get("task_id") != source_task_id
        or environment.get("task_contract_digest")
        != source_task_contract_digest
        or metadata.get("task_contract_digest")
        != source_task_contract_digest
        or definition.contract_digest != source_task_contract_digest
        or metadata.get("checkpoint_sha256") != checkpoint_sha256_value
    ):
        return False, "source metadata identity is not governed and current"
    declared_checkpoint = metadata.get("checkpoint_path")
    if (
        not isinstance(declared_checkpoint, str)
        or Path(declared_checkpoint).resolve() != source_path
    ):
        return False, "source metadata checkpoint path mismatch"

    experiment = metadata.get("experiment")
    manifest_path = (
        Path(experiment.get("manifest_path"))
        if isinstance(experiment, Mapping)
        and isinstance(experiment.get("manifest_path"), str)
        else None
    )
    if manifest_path is None or not manifest_path.is_file():
        return False, "source experiment manifest missing"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as error:
        return False, f"source manifest unreadable: {error}"
    if not isinstance(manifest, Mapping):
        return False, "source manifest must contain a mapping"
    checkpoint = manifest.get("checkpoint")
    canonical_checkpoint = (
        manifest.get("schema_version") == 2
        and manifest.get("artifact_type")
        == "transfergrid-experiment-manifest"
        and manifest.get("status") == "succeeded"
        and manifest.get("run_id") == source_run_id
        and isinstance(checkpoint, Mapping)
        and checkpoint.get("sha256") == checkpoint_sha256_value
        and Path(str(checkpoint.get("path"))).resolve()
        == source_path
    )
    if not canonical_checkpoint:
        return False, "source is not the canonical final manifest checkpoint"

    configuration = manifest.get("configuration")
    configuration_path = (
        Path(configuration.get("json"))
        if isinstance(configuration, Mapping)
        and isinstance(configuration.get("json"), str)
        else None
    )
    if configuration_path is None:
        return False, "source experiment configuration missing"
    if not configuration_path.is_absolute():
        configuration_path = manifest_path.parent / configuration_path
    if not configuration_path.is_file():
        return False, "source experiment configuration missing"
    try:
        configuration_document = json.loads(
            configuration_path.read_text(encoding="utf-8")
        )
    except Exception as error:
        return False, f"source configuration unreadable: {error}"
    if not isinstance(configuration_document, Mapping):
        return False, "source configuration must contain a mapping"
    training_configuration = configuration_document.get("configuration")
    configured_environment = (
        training_configuration.get("environment")
        if isinstance(training_configuration, Mapping)
        else None
    )
    configured = (
        configuration_document.get("schema_version") == 2
        and configuration_document.get("artifact_type")
        == "transfergrid-experiment-configuration"
        and configuration_document.get("run_id") == source_run_id
        and isinstance(configured_environment, Mapping)
        and configured_environment.get("task_id") == source_task_id
        and configured_environment.get("task_contract_digest")
        == source_task_contract_digest
    )
    if not configured:
        return False, "source experiment configuration identity mismatch"
    return True, str(manifest_path.resolve())


def _observation_signature(space: gym.Space) -> dict[str, Any]:
    if not isinstance(space, gym.spaces.Box):
        return {"space_type": type(space).__name__}
    low = np.asarray(space.low)
    high = np.asarray(space.high)
    return {
        "space_type": "Box",
        "shape": list(space.shape),
        "dtype": str(space.dtype),
        "low": int(low.min()) if np.all(low == low.flat[0]) else low.tolist(),
        "high": int(high.max()) if np.all(high == high.flat[0]) else high.tolist(),
    }


def _action_signature(
    space: gym.Space,
    *,
    ordering: tuple[str, ...] = ACTION_ORDER,
) -> dict[str, Any]:
    if not isinstance(space, gym.spaces.Discrete):
        return {"space_type": type(space).__name__}
    return {
        "space_type": "Discrete",
        "count": int(space.n),
        "start": int(space.start),
        "ordering": list(ordering),
    }


def _qualified_name(value: type) -> str:
    return f"{value.__module__}.{value.__qualname__}"


def _digest(payload: Mapping[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


@contextmanager
def _preserve_global_rng_state():
    """Restore Python, NumPy, and CPU Torch RNG after source inspection."""
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)
