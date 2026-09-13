"""Typed, read-only normalization for current and legacy evidence artifacts."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from environments.catalog import require_released_task
from environments.factory import EnvironmentSpec
from training.checkpoints import (
    DEFAULT_CHECKPOINT_ROOT,
    checkpoint_metadata_path,
    load_checkpoint_metadata,
)
from training.initialization_methods import TRANSFER_INITIALIZATION_METHODS
from training.evaluation_identity import (
    build_evaluation_v2_comparison_condition,
    evaluation_v2_comparison_condition_digest,
)

CHECKPOINT_METADATA_SCHEMAS = frozenset({1, 2, 3})
EXPERIMENT_SCHEMAS = frozenset({1, 2})
LEGACY_EVALUATION_PROTOCOLS = frozenset({"evaluation-v1", "evaluation-v1.2"})
SUPPORTED_EVALUATION_PROTOCOLS = LEGACY_EVALUATION_PROTOCOLS | {"evaluation-v2"}
EVALUATION_V2_REPORT_SCHEMAS = frozenset({2, 3})
AUTOMATIC_EVALUATION_V1_2 = "evaluation-v1.2"
AUTOMATIC_EVALUATION_V2 = "evaluation-v2"
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
NORMAL_CHECKPOINT = "normal-final"
TRANSFER_FINAL_CHECKPOINT = "transfer-final"
TRANSFER_INITIALIZATION_CHECKPOINT = "transfer-initialization"


@dataclass(frozen=True, slots=True)
class NormalizedCheckpointMetadata:
    """Identity-bearing subset shared by supported checkpoint schemas."""

    schema_version: int
    environment: EnvironmentSpec
    task_id: str
    checkpoint_sha256: str | None
    run_id: str | None
    original_task_id: str
    legacy_source: bool
    checkpoint_kind: str


@dataclass(frozen=True, slots=True)
class NormalizedExperimentManifest:
    """Identity-bearing subset of an experiment schema-1 manifest."""

    schema_version: int
    run_id: str
    environment: EnvironmentSpec
    task_id: str
    checkpoint_path: Path
    checkpoint_sha256: str | None
    original_task_id: str
    legacy_source: bool


@dataclass(frozen=True, slots=True)
class NormalizedEvaluationEvidence:
    """Protocol-preserving subset of supported evaluation evidence."""

    protocol_version: str
    report_id: str
    environment: EnvironmentSpec
    task_id: str
    checkpoint_path: Path
    checkpoint_sha256: str | None
    deterministic_actions: bool
    replay_eligible: bool
    original_task_id: str
    legacy_source: bool
    report_schema_version: int | None = None
    comparison_condition_digest: str | None = None


def _normalized_task_identity(environment: EnvironmentSpec) -> tuple[str, str, bool]:
    """Normalize proven-equivalent legacy DoorKey without rewriting source data."""
    original = environment.resolved_task_id
    if environment.task_id is None:
        return "door_key-v1", original, True
    return original, original, False


@dataclass(frozen=True, slots=True)
class CheckpointArtifactReference:
    """One governed checkpoint accepted by the read-only catalog."""

    checkpoint_path: Path
    metadata_path: Path
    metadata: NormalizedCheckpointMetadata


@dataclass(frozen=True, slots=True)
class ArtifactExclusion:
    """One excluded artifact and its stable machine-readable reason."""

    path: Path
    reason: str
    detail: str


@dataclass(frozen=True, slots=True)
class CheckpointCatalogScan:
    """Complete included/excluded result for one checkpoint-root scan."""

    included: tuple[CheckpointArtifactReference, ...]
    excluded: tuple[ArtifactExclusion, ...]


@dataclass(frozen=True, slots=True)
class EvaluationCheckpointRoute:
    """One metadata-selected Evaluation protocol route."""

    checkpoint_path: Path
    metadata_path: Path
    metadata: NormalizedCheckpointMetadata
    protocol_version: str

    @property
    def evidence_badge(self) -> str:
        """Return concise user-facing evidence language for this route."""
        return (
            "GOVERNED V2"
            if self.protocol_version == AUTOMATIC_EVALUATION_V2
            else "HISTORICAL V1.2"
        )

    @property
    def evaluation_family(self) -> str:
        """Separate ordinary training evidence from transfer-study evidence."""
        return (
            "transfer"
            if self.metadata.checkpoint_kind in (
                TRANSFER_FINAL_CHECKPOINT,
                TRANSFER_INITIALIZATION_CHECKPOINT,
            )
            else "normal"
        )

    @property
    def checkpoint_stage(self) -> str:
        """Return the policy stage represented by this exact checkpoint."""
        return (
            "initialization"
            if self.metadata.checkpoint_kind
            == TRANSFER_INITIALIZATION_CHECKPOINT
            else "final"
        )


def classify_checkpoint_metadata(metadata: Mapping[str, Any]) -> str:
    """Classify one checkpoint without rewriting historical metadata."""
    if metadata.get("artifact_role") == "target-initialization-0-percent":
        return TRANSFER_INITIALIZATION_CHECKPOINT
    initialization = metadata.get("initialization")
    if (
        isinstance(initialization, Mapping)
        and initialization.get("method") in TRANSFER_INITIALIZATION_METHODS
    ):
        return TRANSFER_FINAL_CHECKPOINT
    return NORMAL_CHECKPOINT


def normalize_checkpoint_metadata_payload(
    payload: Mapping[str, Any],
) -> NormalizedCheckpointMetadata:
    """Normalize checkpoint metadata without reading or rewriting its model."""
    values = _mapping(payload, "checkpoint metadata")
    schema_version = _supported_int(
        values.get("schema_version"),
        CHECKPOINT_METADATA_SCHEMAS,
        "checkpoint metadata schema_version",
    )
    environment = EnvironmentSpec.from_dict(
        _required_mapping(values, "environment", "checkpoint metadata")
    )
    checkpoint_hash = values.get("checkpoint_sha256")
    if schema_version in (2, 3):
        checkpoint_hash = _sha256(
            checkpoint_hash,
            "schema-2 checkpoint metadata checkpoint_sha256",
        )
    elif checkpoint_hash is not None:
        checkpoint_hash = _sha256(
            checkpoint_hash,
            "checkpoint metadata checkpoint_sha256",
        )
    run_id = values.get("run_id")
    if run_id is not None and (not isinstance(run_id, str) or not run_id):
        raise ValueError("checkpoint metadata run_id must be a non-empty string.")
    task_id, original_task_id, legacy_source = _normalized_task_identity(environment)
    return NormalizedCheckpointMetadata(
        schema_version=schema_version,
        environment=environment,
        task_id=task_id,
        checkpoint_sha256=checkpoint_hash,
        run_id=run_id,
        original_task_id=original_task_id,
        legacy_source=legacy_source,
        checkpoint_kind=classify_checkpoint_metadata(values),
    )


def load_normalized_checkpoint(
    checkpoint_path: str | Path,
) -> CheckpointArtifactReference:
    """Load one checkpoint sidecar through existing SHA-validating readers."""
    path = Path(checkpoint_path).resolve()
    payload = load_checkpoint_metadata(path)
    return CheckpointArtifactReference(
        checkpoint_path=path,
        metadata_path=checkpoint_metadata_path(path).resolve(),
        metadata=normalize_checkpoint_metadata_payload(payload),
    )


def scan_checkpoint_artifacts(
    checkpoint_root: str | Path,
) -> CheckpointCatalogScan:
    """Scan checkpoint ZIPs while retaining every exclusion reason."""
    root = Path(checkpoint_root)
    if not root.is_dir():
        return CheckpointCatalogScan(included=(), excluded=())
    included: list[CheckpointArtifactReference] = []
    excluded: list[ArtifactExclusion] = []
    for checkpoint_path in sorted(root.rglob("*.zip")):
        metadata_path = checkpoint_metadata_path(checkpoint_path)
        if not metadata_path.is_file():
            excluded.append(
                ArtifactExclusion(
                    path=checkpoint_path.resolve(),
                    reason="missing_checkpoint_metadata",
                    detail=(
                        "Checkpoint has no metadata sidecar and cannot receive "
                        "a governed task identity."
                    ),
                )
            )
            continue
        try:
            included.append(load_normalized_checkpoint(checkpoint_path))
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
            excluded.append(
                ArtifactExclusion(
                    path=checkpoint_path.resolve(),
                    reason="invalid_checkpoint_metadata",
                    detail=str(error),
                )
            )
    return CheckpointCatalogScan(
        included=tuple(included),
        excluded=tuple(excluded),
    )


def evaluation_protocol_for_checkpoint_metadata(
    metadata: NormalizedCheckpointMetadata,
) -> str:
    """Select the only supported protocol for normalized checkpoint metadata."""
    if metadata.schema_version in (1, 2):
        return AUTOMATIC_EVALUATION_V1_2
    if metadata.schema_version == 3 and not metadata.legacy_source:
        require_released_task(metadata.environment.task_id)
        return AUTOMATIC_EVALUATION_V2
    raise ValueError(
        "Checkpoint metadata does not describe a supported automatic "
        "Evaluation V1.2 or governed V2 route."
    )


def route_evaluation_checkpoint(
    reference: CheckpointArtifactReference,
) -> EvaluationCheckpointRoute:
    """Bind one included checkpoint to its metadata-selected protocol."""
    return EvaluationCheckpointRoute(
        checkpoint_path=reference.checkpoint_path,
        metadata_path=reference.metadata_path,
        metadata=reference.metadata,
        protocol_version=evaluation_protocol_for_checkpoint_metadata(
            reference.metadata
        ),
    )


def discover_evaluation_checkpoint_routes(
    checkpoint_root: str | Path = DEFAULT_CHECKPOINT_ROOT,
) -> tuple[EvaluationCheckpointRoute, ...]:
    """Return supported checkpoint-first routes without changing discovery data."""
    routes: list[EvaluationCheckpointRoute] = []
    for reference in scan_checkpoint_artifacts(checkpoint_root).included:
        try:
            routes.append(route_evaluation_checkpoint(reference))
        except ValueError:
            continue
    return tuple(routes)


def normalize_experiment_payloads(
    manifest_payload: Mapping[str, Any],
    configuration_payload: Mapping[str, Any],
) -> NormalizedExperimentManifest:
    """Normalize one schema-1 manifest and its schema-1 configuration."""
    manifest = _mapping(manifest_payload, "experiment manifest")
    configuration_document = _mapping(
        configuration_payload,
        "experiment configuration",
    )
    schema_version = _supported_int(
        manifest.get("schema_version"),
        EXPERIMENT_SCHEMAS,
        "experiment manifest schema_version",
    )
    configuration_schema = _supported_int(
        configuration_document.get("schema_version"),
        EXPERIMENT_SCHEMAS,
        "experiment configuration schema_version",
    )
    if configuration_schema != schema_version:
        raise ValueError(
            "experiment manifest and configuration schema_version do not match."
        )
    if manifest.get("artifact_type") != "transfergrid-experiment-manifest":
        raise ValueError("unsupported experiment manifest artifact_type.")
    if (
        configuration_document.get("artifact_type")
        != "transfergrid-experiment-configuration"
    ):
        raise ValueError("unsupported experiment configuration artifact_type.")
    run_id = manifest.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("experiment manifest run_id must be a non-empty string.")
    if configuration_document.get("run_id") != run_id:
        raise ValueError("experiment manifest and configuration run_id do not match.")
    configuration = _required_mapping(
        configuration_document,
        "configuration",
        "experiment configuration",
    )
    environment = EnvironmentSpec.from_dict(
        _required_mapping(configuration, "environment", "experiment configuration")
    )
    checkpoint = _required_mapping(manifest, "checkpoint", "experiment manifest")
    checkpoint_path = checkpoint.get("path")
    if not isinstance(checkpoint_path, str) or not checkpoint_path:
        raise ValueError("experiment checkpoint path must be a non-empty string.")
    checkpoint_hash = checkpoint.get("sha256")
    if checkpoint_hash is not None:
        checkpoint_hash = _sha256(
            checkpoint_hash,
            "experiment checkpoint sha256",
        )
    task_id, original_task_id, legacy_source = _normalized_task_identity(environment)
    return NormalizedExperimentManifest(
        schema_version=schema_version,
        run_id=run_id,
        environment=environment,
        task_id=task_id,
        checkpoint_path=Path(checkpoint_path),
        checkpoint_sha256=checkpoint_hash,
        original_task_id=original_task_id,
        legacy_source=legacy_source,
    )


def load_normalized_experiment_manifest(
    manifest_path: str | Path,
) -> NormalizedExperimentManifest:
    """Load one manifest and its linked configuration without mutation."""
    path = Path(manifest_path).resolve()
    manifest = _load_json_mapping(path, "experiment manifest")
    configuration_link = _required_mapping(
        manifest,
        "configuration",
        "experiment manifest",
    ).get("json")
    if not isinstance(configuration_link, str) or not configuration_link:
        raise ValueError(
            "experiment manifest configuration.json must be a non-empty path."
        )
    configuration_path = Path(configuration_link)
    if not configuration_path.is_absolute():
        configuration_path = path.parent / configuration_path
    configuration = _load_json_mapping(
        configuration_path,
        "experiment configuration",
    )
    return normalize_experiment_payloads(manifest, configuration)


def normalize_evaluation_report_payload(
    payload: Mapping[str, Any],
) -> NormalizedEvaluationEvidence:
    """Normalize supported evidence while retaining its original protocol."""
    values = _mapping(payload, "evaluation report")
    protocol_version = values.get("protocol_version")
    if protocol_version not in SUPPORTED_EVALUATION_PROTOCOLS:
        raise ValueError(
            "unsupported evaluation protocol_version: "
            f"{protocol_version!r}."
        )
    report_id = values.get("report_id")
    if not isinstance(report_id, str) or not report_id:
        raise ValueError("evaluation report_id must be a non-empty string.")
    conditions = _required_mapping(
        values,
        "evaluation_conditions",
        "evaluation report",
    )
    environment = EnvironmentSpec.from_dict(
        _required_mapping(conditions, "environment", "evaluation conditions")
    )
    report_schema_version: int | None = None
    comparison_condition_digest: str | None = None
    if protocol_version == "evaluation-v2":
        report_schema_version = _supported_int(
            values.get("schema_version"),
            EVALUATION_V2_REPORT_SCHEMAS,
            "Evaluation V2 report schema_version",
        )
        identity = _required_mapping(
            values, "evaluation_identity", "Evaluation V2 report"
        )
        if identity.get("task_id") != environment.task_id:
            raise ValueError("Evaluation V2 task identity mismatch.")
        if identity.get("task_contract_digest") != environment.task_contract_digest:
            raise ValueError("Evaluation V2 task-contract mismatch.")
    deterministic = conditions.get("deterministic_actions")
    if type(deterministic) is not bool:
        raise ValueError(
            "evaluation deterministic_actions must be a boolean."
        )
    if protocol_version == "evaluation-v2":
        comparison_condition_digest = _normalize_v2_comparison_condition_digest(
            report_schema_version=report_schema_version,
            protocol_version=protocol_version,
            environment=environment,
            identity=identity,
            conditions=conditions,
            deterministic_actions=deterministic,
        )
    provenance = _required_mapping(values, "provenance", "evaluation report")
    checkpoint = _required_mapping(provenance, "checkpoint", "evaluation provenance")
    checkpoint_path = checkpoint.get("path")
    if not isinstance(checkpoint_path, str) or not checkpoint_path:
        raise ValueError("evaluation checkpoint path must be a non-empty string.")
    checkpoint_hash = checkpoint.get("sha256")
    if checkpoint_hash is not None:
        checkpoint_hash = _sha256(
            checkpoint_hash,
            "evaluation checkpoint sha256",
        )
    records = values.get("episode_records")
    if not isinstance(records, list):
        raise ValueError("evaluation episode_records must be a list.")
    replay_eligible = deterministic and any(
        _record_is_replay_eligible(record) for record in records
    )
    task_id, original_task_id, legacy_source = _normalized_task_identity(environment)
    return NormalizedEvaluationEvidence(
        protocol_version=protocol_version,
        report_id=report_id,
        environment=environment,
        task_id=task_id,
        checkpoint_path=Path(checkpoint_path),
        checkpoint_sha256=checkpoint_hash,
        deterministic_actions=deterministic,
        replay_eligible=replay_eligible,
        original_task_id=original_task_id,
        legacy_source=legacy_source,
        report_schema_version=report_schema_version,
        comparison_condition_digest=comparison_condition_digest,
    )


def _normalize_v2_comparison_condition_digest(
    *,
    report_schema_version: int | None,
    protocol_version: str,
    environment: EnvironmentSpec,
    identity: Mapping[str, Any],
    conditions: Mapping[str, Any],
    deterministic_actions: bool,
) -> str | None:
    """Read new identities and derive the same identity for complete old V2."""
    if environment.task_id is None or environment.task_contract_digest is None:
        raise ValueError("Evaluation V2 requires a governed released task.")
    definition = require_released_task(environment.task_id)
    distribution = _required_mapping(
        conditions,
        "distribution",
        "Evaluation V2 conditions",
    )
    required_identity_fields = (
        "grid_size",
        "maximum_steps",
        "evidence_role",
        "distribution_id",
        "distribution_version",
        "ordered_seed_digest",
        "deterministic_actions",
    )
    missing = [
        key for key in required_identity_fields
        if identity.get(key) is None
    ]
    if missing:
        if report_schema_version == 3:
            raise ValueError(
                "Evaluation V2 schema 3 identity is missing comparison fields: "
                + ", ".join(missing)
            )
        return None

    observation_contract_id = identity.get("observation_contract_id")
    action_contract_id = identity.get("action_contract_id")
    if report_schema_version == 2:
        observation_contract_id = (
            observation_contract_id or definition.observation_contract_id
        )
        action_contract_id = action_contract_id or definition.action_contract_id
    if (
        observation_contract_id != definition.observation_contract_id
        or action_contract_id != definition.action_contract_id
    ):
        raise ValueError("Evaluation V2 observation/action contract mismatch.")
    if identity.get("grid_size") != environment.grid_size:
        raise ValueError("Evaluation V2 grid identity mismatch.")
    if identity.get("deterministic_actions") is not deterministic_actions:
        raise ValueError("Evaluation V2 deterministic-action identity mismatch.")
    cross_checks = (
        ("evidence_role", "role"),
        ("distribution_id", "name"),
        ("distribution_version", "version"),
    )
    for identity_key, distribution_key in cross_checks:
        distribution_value = distribution.get(distribution_key)
        if (
            distribution_value is not None
            and identity.get(identity_key) != distribution_value
        ):
            raise ValueError(
                "Evaluation V2 distribution identity mismatch for "
                f"{identity_key}."
            )

    ordered_seed_digest = identity.get("ordered_seed_digest")
    seeds = distribution.get("seeds")
    if isinstance(seeds, list):
        if not seeds or any(type(seed) is not int for seed in seeds):
            raise ValueError("Evaluation V2 distribution seeds must be integers.")
        canonical_seeds = json.dumps(seeds, separators=(",", ":"))
        calculated_seed_digest = hashlib.sha256(
            canonical_seeds.encode("utf-8")
        ).hexdigest()
        if ordered_seed_digest != calculated_seed_digest:
            raise ValueError("Evaluation V2 ordered-seed digest mismatch.")
    elif report_schema_version == 3:
        raise ValueError("Evaluation V2 schema 3 requires ordered distribution seeds.")

    comparison_condition = build_evaluation_v2_comparison_condition(
        protocol_version=protocol_version,
        task_id=environment.task_id,
        task_contract_digest=environment.task_contract_digest,
        grid_size=identity["grid_size"],
        maximum_steps=identity["maximum_steps"],
        observation_contract_id=observation_contract_id,
        action_contract_id=action_contract_id,
        evidence_role=identity["evidence_role"],
        distribution_id=identity["distribution_id"],
        distribution_version=identity["distribution_version"],
        ordered_seed_digest=ordered_seed_digest,
        deterministic_actions=identity["deterministic_actions"],
    )
    calculated_digest = evaluation_v2_comparison_condition_digest(
        comparison_condition
    )
    stored_digest = identity.get("comparison_condition_digest")
    if stored_digest is None:
        if report_schema_version == 3:
            raise ValueError(
                "Evaluation V2 schema 3 requires comparison_condition_digest."
            )
        return calculated_digest
    normalized_stored_digest = _sha256(
        stored_digest,
        "Evaluation V2 comparison_condition_digest",
    )
    if normalized_stored_digest != calculated_digest:
        raise ValueError("Evaluation V2 comparison-condition digest mismatch.")
    return normalized_stored_digest


def load_normalized_evaluation_report(
    report_path: str | Path,
) -> NormalizedEvaluationEvidence:
    """Load one Evaluation V1/V1.2 report without rewriting it."""
    path = Path(report_path)
    return normalize_evaluation_report_payload(
        _load_json_mapping(path, "evaluation report")
    )


def _record_is_replay_eligible(record: object) -> bool:
    if not isinstance(record, Mapping) or record.get("completed") is not True:
        return False
    required = (
        "seed",
        "actions",
        "total_reward",
        "episode_length",
        "success",
        "terminated",
        "truncated",
    )
    return all(key in record and record[key] is not None for key in required) and (
        isinstance(record.get("actions"), list)
    )


def _load_json_mapping(path: Path, label: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return _mapping(payload, label)


def _mapping(payload: object, label: str) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} must be a mapping.")
    return dict(payload)


def _required_mapping(
    payload: Mapping[str, Any],
    key: str,
    label: str,
) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} is missing its {key} mapping.")
    return dict(value)


def _supported_int(value: object, allowed: frozenset[int], label: str) -> int:
    if type(value) is not int or value not in allowed:
        raise ValueError(
            f"{label} must be one of {sorted(allowed)}; received {value!r}."
        )
    return value


def _sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase 64-character SHA-256.")
    return value
