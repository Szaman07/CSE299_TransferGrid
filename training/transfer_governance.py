"""Governed source competence and target-generic transfer preflight records."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from environments.catalog import (
    TaskDefinition,
    released_task_definitions,
    require_released_task,
)
from training.artifact_catalog import scan_checkpoint_artifacts
from training.benchmark_contracts import VALIDATION_SEEDS
from training.checkpoints import (
    DEFAULT_CHECKPOINT_ROOT,
    checkpoint_sha256,
    load_checkpoint_metadata,
)
from training.initialization import (
    CompatibilityReport,
    SourcePolicyInitialization,
    governed_source_complete,
)
from training.portable_paths import resolve_recorded_path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVALUATION_V2_REPORT_ROOT = (
    PROJECT_ROOT / "reports" / "evaluations_v2"
)
SOURCE_COMPETENCE_THRESHOLD = 0.80
EVALUATION_V2_PROTOCOL = "evaluation-v2"
EVALUATION_V2_DISTRIBUTION_VERSION = 1


@dataclass(frozen=True, slots=True)
class SourceCompetenceCheck:
    """One source-only scientific eligibility check."""

    code: str
    passed: bool
    detail: str


@dataclass(frozen=True, slots=True)
class SourceCompetenceReport:
    """Validation evidence kept separate from tensor compatibility."""

    competent: bool
    checks: tuple[SourceCompetenceCheck, ...]
    digest: str
    source_checkpoint_path: Path
    source_checkpoint_sha256: str | None
    source_task_id: str | None
    source_task_contract_digest: str | None
    validation_report_path: Path | None
    validation_report_sha256: str | None
    validation_report_id: str | None
    validation_success_count: int | None
    validation_episode_count: int | None
    validation_success_rate: float | None
    required_success_rate: float = SOURCE_COMPETENCE_THRESHOLD

    @property
    def failure_codes(self) -> tuple[str, ...]:
        return tuple(check.code for check in self.checks if not check.passed)

    def to_dict(self) -> dict[str, Any]:
        return {
            "competent": self.competent,
            "checks": [asdict(check) for check in self.checks],
            "failure_codes": list(self.failure_codes),
            "digest": self.digest,
            "source_checkpoint_path": str(self.source_checkpoint_path),
            "source_checkpoint_sha256": self.source_checkpoint_sha256,
            "source_task_id": self.source_task_id,
            "source_task_contract_digest": (
                self.source_task_contract_digest
            ),
            "validation_report_path": (
                str(self.validation_report_path)
                if self.validation_report_path is not None else None
            ),
            "validation_report_sha256": self.validation_report_sha256,
            "validation_report_id": self.validation_report_id,
            "validation_success_count": self.validation_success_count,
            "validation_episode_count": self.validation_episode_count,
            "validation_success_rate": self.validation_success_rate,
            "required_success_rate": self.required_success_rate,
        }


class SourceCompetenceError(ValueError):
    """Raised when a technically usable checkpoint lacks source competence."""

    def __init__(self, report: SourceCompetenceReport) -> None:
        self.report = report
        super().__init__(
            "Source competence failed: " + ", ".join(report.failure_codes)
        )


@dataclass(frozen=True, slots=True)
class GovernedTransferSource:
    """One canonical final checkpoint eligible to initialize a study."""

    checkpoint_path: Path
    metadata_path: Path
    task_id: str
    task_contract_digest: str
    run_id: str
    checkpoint_sha256: str
    competence: SourceCompetenceReport

    def to_dict(self) -> dict[str, Any]:
        return {
            "checkpoint_path": str(self.checkpoint_path),
            "metadata_path": str(self.metadata_path),
            "task_id": self.task_id,
            "task_contract_digest": self.task_contract_digest,
            "run_id": self.run_id,
            "checkpoint_sha256": self.checkpoint_sha256,
            "competence": self.competence.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class TransferSourceExclusion:
    """Stable reason why a checkpoint is not an eligible governed source."""

    checkpoint_path: Path
    reason: str
    detail: str


@dataclass(frozen=True, slots=True)
class GovernedTransferSourceDiscovery:
    """Complete rich discovery result without mutating source artifacts."""

    included: tuple[GovernedTransferSource, ...]
    excluded: tuple[TransferSourceExclusion, ...]


@dataclass(frozen=True, slots=True)
class TransferPreflightReport:
    """One request-bound decision with independent technical/scientific axes."""

    ready: bool
    technical_compatibility: CompatibilityReport
    source_competence: SourceCompetenceReport
    study_relation: str
    source_task_id: str
    target_task_id: str
    target_task_contract_digest: str
    binding_digest: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "ready": self.ready,
            "technical_compatibility": (
                self.technical_compatibility.to_dict()
            ),
            "source_competence": self.source_competence.to_dict(),
            "study_relation": self.study_relation,
            "source_task_id": self.source_task_id,
            "target_task_id": self.target_task_id,
            "target_task_contract_digest": (
                self.target_task_contract_digest
            ),
            "binding_digest": self.binding_digest,
        }


def discover_released_transfer_targets() -> tuple[TaskDefinition, ...]:
    """Return only catalog-released target definitions, never drafts/legacy."""
    return released_task_definitions()


def discover_governed_transfer_sources(
    *,
    source_task_id: str | None = None,
    checkpoint_root: str | Path = DEFAULT_CHECKPOINT_ROOT,
    report_root: str | Path = DEFAULT_EVALUATION_V2_REPORT_ROOT,
) -> GovernedTransferSourceDiscovery:
    """Discover canonical, complete, Validation-competent schema-3 sources."""
    if source_task_id is not None:
        require_released_task(source_task_id)
    scan = scan_checkpoint_artifacts(checkpoint_root)
    included: list[GovernedTransferSource] = []
    excluded = [
        TransferSourceExclusion(item.path, item.reason, item.detail)
        for item in scan.excluded
    ]
    for reference in scan.included:
        normalized = reference.metadata
        if source_task_id is not None and normalized.task_id != source_task_id:
            continue
        if normalized.schema_version != 3 or normalized.legacy_source:
            excluded.append(TransferSourceExclusion(
                reference.checkpoint_path,
                "source_metadata_not_governed",
                "Full-policy studies require non-legacy schema-3 metadata.",
            ))
            continue
        try:
            metadata = load_checkpoint_metadata(reference.checkpoint_path)
            environment = metadata.get("environment")
            if not isinstance(environment, Mapping):
                raise ValueError("source metadata environment is missing")
            task_id = environment.get("task_id")
            run_id = metadata.get("run_id")
            contract_digest = metadata.get("task_contract_digest")
            actual_hash = checkpoint_sha256(reference.checkpoint_path)
            if not all(
                isinstance(value, str) and value
                for value in (task_id, run_id, contract_digest)
            ):
                raise ValueError("source governed identity is incomplete")
            complete, detail = governed_source_complete(
                metadata,
                checkpoint_path=reference.checkpoint_path,
                checkpoint_sha256_value=actual_hash,
                source_run_id=run_id,
                source_task_id=task_id,
                source_task_contract_digest=contract_digest,
            )
            if not complete:
                excluded.append(TransferSourceExclusion(
                    reference.checkpoint_path,
                    "source_not_canonical_complete_final",
                    detail,
                ))
                continue
            competence = assess_source_competence(
                reference.checkpoint_path,
                report_root=report_root,
            )
            if not competence.competent:
                excluded.append(TransferSourceExclusion(
                    reference.checkpoint_path,
                    "source_competence_failed",
                    ", ".join(competence.failure_codes),
                ))
                continue
            included.append(GovernedTransferSource(
                checkpoint_path=reference.checkpoint_path,
                metadata_path=reference.metadata_path,
                task_id=task_id,
                task_contract_digest=contract_digest,
                run_id=run_id,
                checkpoint_sha256=actual_hash,
                competence=competence,
            ))
        except Exception as error:
            excluded.append(TransferSourceExclusion(
                reference.checkpoint_path,
                "source_governance_invalid",
                f"{type(error).__name__}: {error}",
            ))
    return GovernedTransferSourceDiscovery(
        included=tuple(sorted(
            included,
            key=lambda item: (
                item.task_id, item.run_id, str(item.checkpoint_path)
            ),
        )),
        excluded=tuple(excluded),
    )


def assess_source_competence(
    checkpoint_path: str | Path,
    *,
    report_root: str | Path = DEFAULT_EVALUATION_V2_REPORT_ROOT,
) -> SourceCompetenceReport:
    """Validate exact complete deterministic source-task Validation evidence."""
    source_path = Path(checkpoint_path).resolve()
    try:
        metadata = load_checkpoint_metadata(source_path)
        actual_hash = checkpoint_sha256(source_path)
    except Exception as error:
        return _competence_report(
            source_path=source_path,
            source_hash=None,
            task_id=None,
            contract_digest=None,
            checks=(SourceCompetenceCheck(
                "VALIDATION_SOURCE_READABLE",
                False,
                f"{type(error).__name__}: {error}",
            ),),
        )
    environment = metadata.get("environment")
    task_id = (
        environment.get("task_id")
        if isinstance(environment, Mapping) else None
    )
    contract_digest = metadata.get("task_contract_digest")
    if not isinstance(task_id, str) or not isinstance(contract_digest, str):
        return _competence_report(
            source_path=source_path,
            source_hash=actual_hash,
            task_id=task_id if isinstance(task_id, str) else None,
            contract_digest=(
                contract_digest
                if isinstance(contract_digest, str) else None
            ),
            checks=(SourceCompetenceCheck(
                "VALIDATION_SOURCE_IDENTITY",
                False,
                "source task or contract identity is missing",
            ),),
        )

    candidates: list[tuple[Path, Mapping[str, Any]]] = []
    root = Path(report_root)
    if root.is_dir():
        for report_path in sorted(root.rglob("*.json")):
            try:
                payload = json.loads(report_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(payload, Mapping):
                continue
            identity = payload.get("evaluation_identity")
            if (
                payload.get("protocol_version") == EVALUATION_V2_PROTOCOL
                and isinstance(identity, Mapping)
                and identity.get("evidence_role") == "validation"
                and identity.get("checkpoint_sha256") == actual_hash
            ):
                candidates.append((report_path.resolve(), payload))

    if not candidates:
        return _competence_report(
            source_path=source_path,
            source_hash=actual_hash,
            task_id=task_id,
            contract_digest=contract_digest,
            checks=(SourceCompetenceCheck(
                "VALIDATION_REPORT_FOUND",
                False,
                "no exact Evaluation V2 Validation report for checkpoint SHA",
            ),),
        )

    assessed = [
        _assess_validation_report(
            report_path,
            payload,
            source_path=source_path,
            source_hash=actual_hash,
            task_id=task_id,
            contract_digest=contract_digest,
        )
        for report_path, payload in candidates
    ]
    passing = [item for item in assessed if item.competent]
    if not passing:
        return assessed[0]

    signatures = {
        _validation_outcome_signature(payload)
        for (_, payload), result in zip(candidates, assessed, strict=True)
        if all(
            check.passed
            for check in result.checks
            if check.code != "VALIDATION_SUCCESS_THRESHOLD"
        )
    }
    selected = passing[0]
    consistency = SourceCompetenceCheck(
        "VALIDATION_REPORT_CONSISTENT",
        len(signatures) <= 1,
        (
            "all complete deterministic Validation reports agree"
            if len(signatures) <= 1
            else "conflicting complete Validation reports exist"
        ),
    )
    return _competence_report(
        source_path=selected.source_checkpoint_path,
        source_hash=selected.source_checkpoint_sha256,
        task_id=selected.source_task_id,
        contract_digest=selected.source_task_contract_digest,
        checks=selected.checks + (consistency,),
        report_path=selected.validation_report_path,
        report_sha=selected.validation_report_sha256,
        report_id=selected.validation_report_id,
        success_count=selected.validation_success_count,
        episode_count=selected.validation_episode_count,
        success_rate=selected.validation_success_rate,
    )


def build_transfer_preflight_report(
    *,
    initialization: SourcePolicyInitialization,
    technical_compatibility: CompatibilityReport,
    source_competence: SourceCompetenceReport,
    target_task_id: str,
    target_task_contract_digest: str,
    request_payload: Mapping[str, Any],
) -> TransferPreflightReport:
    """Bind one request without mixing study relation into technical identity."""
    binding_payload = {
        "request": dict(request_payload),
        "technical_compatibility_digest": technical_compatibility.digest,
        "source_competence_digest": source_competence.digest,
        "source_task_id": initialization.source_task_id,
        "target_task_id": target_task_id,
        "target_task_contract_digest": target_task_contract_digest,
        "study_relation": initialization.study_relation,
    }
    return TransferPreflightReport(
        ready=(
            technical_compatibility.compatible
            and source_competence.competent
        ),
        technical_compatibility=technical_compatibility,
        source_competence=source_competence,
        study_relation=initialization.study_relation,
        source_task_id=initialization.source_task_id,
        target_task_id=target_task_id,
        target_task_contract_digest=target_task_contract_digest,
        binding_digest=_digest(binding_payload),
    )


def _assess_validation_report(
    report_path: Path,
    report: Mapping[str, Any],
    *,
    source_path: Path,
    source_hash: str,
    task_id: str,
    contract_digest: str,
) -> SourceCompetenceReport:
    checks: list[SourceCompetenceCheck] = [
        SourceCompetenceCheck(
            "VALIDATION_REPORT_FOUND", True, str(report_path)
        )
    ]

    def record(code: str, passed: bool, detail: str) -> None:
        checks.append(SourceCompetenceCheck(code, bool(passed), detail))

    identity = report.get("evaluation_identity")
    conditions = report.get("evaluation_conditions")
    provenance = report.get("provenance")
    aggregate = report.get("aggregate_metrics")
    records = report.get("episode_records")
    distribution = (
        conditions.get("distribution")
        if isinstance(conditions, Mapping) else None
    )
    checkpoint = (
        provenance.get("checkpoint")
        if isinstance(provenance, Mapping) else None
    )
    expected_seeds = tuple(VALIDATION_SEEDS)
    expected_seed_digest = _seed_digest(expected_seeds)
    expected_distribution = f"{task_id}-validation"

    record(
        "VALIDATION_PROTOCOL",
        report.get("protocol_version") == EVALUATION_V2_PROTOCOL
        and report.get("schema_version") in (2, 3),
        (
            f"protocol={report.get('protocol_version')!r} "
            f"schema={report.get('schema_version')!r}"
        ),
    )
    recorded_checkpoint_path = None
    if isinstance(checkpoint, Mapping) and isinstance(checkpoint.get("path"), str):
        declared = Path(str(checkpoint["path"]))
        try:
            recorded_checkpoint_path = (
                declared.resolve()
                if declared.is_absolute() and declared.is_file()
                else resolve_recorded_path(declared, PROJECT_ROOT)
            )
        except (OSError, ValueError):
            recorded_checkpoint_path = None
    identity_passed = (
        isinstance(identity, Mapping)
        and identity.get("task_id") == task_id
        and identity.get("task_contract_digest") == contract_digest
        and identity.get("checkpoint_sha256") == source_hash
        and identity.get("evidence_role") == "validation"
        and isinstance(checkpoint, Mapping)
        and checkpoint.get("sha256") == source_hash
        and recorded_checkpoint_path == source_path
    )
    record(
        "VALIDATION_SOURCE_IDENTITY",
        identity_passed,
        "report task, contract, path, and checkpoint SHA match source",
    )
    distribution_passed = (
        isinstance(identity, Mapping)
        and identity.get("distribution_id") == expected_distribution
        and identity.get("distribution_version")
        == EVALUATION_V2_DISTRIBUTION_VERSION
        and identity.get("ordered_seed_digest") == expected_seed_digest
        and isinstance(distribution, Mapping)
        and distribution.get("name") == expected_distribution
        and distribution.get("role") == "validation"
        and distribution.get("version")
        == EVALUATION_V2_DISTRIBUTION_VERSION
        and tuple(distribution.get("seeds", ())) == expected_seeds
        and distribution.get("episode_count") == len(expected_seeds)
    )
    record(
        "VALIDATION_DISTRIBUTION",
        distribution_passed,
        (
            f"expected={expected_distribution!r} "
            f"seed_digest={expected_seed_digest}"
        ),
    )
    deterministic_passed = (
        isinstance(identity, Mapping)
        and identity.get("deterministic_actions") is True
        and isinstance(conditions, Mapping)
        and conditions.get("deterministic_actions") is True
    )
    record(
        "VALIDATION_DETERMINISTIC",
        deterministic_passed,
        "identity and conditions require deterministic actions",
    )

    record_list = records if isinstance(records, list) else []
    ordered_record_seeds = tuple(
        item.get("seed") if isinstance(item, Mapping) else None
        for item in record_list
    )
    complete_records = (
        len(record_list) == len(expected_seeds)
        and ordered_record_seeds == expected_seeds
        and all(
            isinstance(item, Mapping)
            and item.get("completed") is True
            and item.get("evaluator_error") is None
            and type(item.get("success")) is bool
            for item in record_list
        )
    )
    complete_aggregate = (
        isinstance(aggregate, Mapping)
        and aggregate.get("requested_episode_count") == len(expected_seeds)
        and aggregate.get("completed_episode_count") == len(expected_seeds)
        and aggregate.get("error_count") == 0
        and aggregate.get("report_complete") is True
    )
    record(
        "VALIDATION_COMPLETE",
        complete_records and complete_aggregate,
        (
            f"records={len(record_list)} "
            f"expected={len(expected_seeds)}"
        ),
    )

    success_count = (
        sum(item.get("success") is True for item in record_list)
        if complete_records else None
    )
    success_rate = (
        success_count / len(expected_seeds)
        if success_count is not None else None
    )
    required_count = math.ceil(
        SOURCE_COMPETENCE_THRESHOLD * len(expected_seeds)
    )
    aggregate_consistent = (
        success_count is not None
        and isinstance(aggregate, Mapping)
        and aggregate.get("success_count") == success_count
        and isinstance(aggregate.get("success_rate"), (int, float))
        and abs(float(aggregate.get("success_rate")) - success_rate) < 1e-12
    )
    threshold_passed = (
        aggregate_consistent
        and success_count is not None
        and success_count >= required_count
    )
    record(
        "VALIDATION_SUCCESS_THRESHOLD",
        threshold_passed,
        (
            f"successes={success_count!r}/{len(expected_seeds)} "
            f"required={required_count}/{len(expected_seeds)}"
        ),
    )
    return _competence_report(
        source_path=source_path,
        source_hash=source_hash,
        task_id=task_id,
        contract_digest=contract_digest,
        checks=tuple(checks),
        report_path=report_path,
        report_sha=_file_sha256(report_path),
        report_id=(
            report.get("report_id")
            if isinstance(report.get("report_id"), str) else None
        ),
        success_count=success_count,
        episode_count=len(expected_seeds),
        success_rate=success_rate,
    )


def _validation_outcome_signature(report: Mapping[str, Any]) -> str:
    identity = report.get("evaluation_identity")
    normalized_identity = {
        key: identity.get(key)
        for key in (
            "task_id",
            "task_contract_digest",
            "checkpoint_sha256",
            "evidence_role",
            "grid_size",
            "maximum_steps",
            "deterministic_actions",
            "distribution_id",
            "distribution_version",
            "ordered_seed_digest",
        )
    } if isinstance(identity, Mapping) else None
    return _digest({
        # Evaluation V2 schema 3 added observation/action/condition digests.
        # Those fields strengthen one report's identity but do not change its
        # deterministic episode outcome. Compare the common governed identity
        # so a schema-2 report and its schema-3 re-evaluation can agree.
        "evaluation_identity": normalized_identity,
        "evaluation_conditions": report.get("evaluation_conditions"),
        "episode_records": report.get("episode_records"),
        "aggregate_metrics": report.get("aggregate_metrics"),
    })


def _competence_report(
    *,
    source_path: Path,
    source_hash: str | None,
    task_id: str | None,
    contract_digest: str | None,
    checks: tuple[SourceCompetenceCheck, ...],
    report_path: Path | None = None,
    report_sha: str | None = None,
    report_id: str | None = None,
    success_count: int | None = None,
    episode_count: int | None = None,
    success_rate: float | None = None,
) -> SourceCompetenceReport:
    payload = {
        "checks": [asdict(check) for check in checks],
        "source_checkpoint_path": str(source_path),
        "source_checkpoint_sha256": source_hash,
        "source_task_id": task_id,
        "source_task_contract_digest": contract_digest,
        "validation_report_path": (
            str(report_path) if report_path is not None else None
        ),
        "validation_report_sha256": report_sha,
        "validation_report_id": report_id,
        "validation_success_count": success_count,
        "validation_episode_count": episode_count,
        "validation_success_rate": success_rate,
        "required_success_rate": SOURCE_COMPETENCE_THRESHOLD,
    }
    competent = all(check.passed for check in checks)
    return SourceCompetenceReport(
        competent=competent,
        checks=checks,
        digest=_digest(payload),
        source_checkpoint_path=source_path,
        source_checkpoint_sha256=source_hash,
        source_task_id=task_id,
        source_task_contract_digest=contract_digest,
        validation_report_path=report_path,
        validation_report_sha256=report_sha,
        validation_report_id=report_id,
        validation_success_count=success_count,
        validation_episode_count=episode_count,
        validation_success_rate=success_rate,
    )


def _seed_digest(seeds: tuple[int, ...]) -> str:
    return hashlib.sha256(
        json.dumps(list(seeds), separators=(",", ":")).encode()
    ).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _digest(payload: Mapping[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()
