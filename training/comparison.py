"""Strict, read-only comparison bundles for cognitive-transfer evidence."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from environments.catalog import require_released_task

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_COMPARISON_ROOT = PROJECT_ROOT / "reports" / "comparisons"
COMPARISON_ARTIFACT_TYPE = "transfergrid-comparison-bundle"
COMPARISON_SCHEMA_VERSION = 1


class ComparisonEvidenceError(ValueError):
    """Raised when a comparison bundle is incomplete or scientifically mixed."""


@dataclass(frozen=True, slots=True)
class DevelopmentPoint:
    """One deterministic fixed-step Development measurement."""

    step: int
    success_count: int
    episode_count: int
    success_rate: float
    mean_reward: float
    mean_episode_length: float
    error_count: int


@dataclass(frozen=True, slots=True)
class ComparisonCondition:
    """One scratch or transferred target-learning condition."""

    condition_id: str
    display_name: str
    initialization_method: str
    source_task_id: str | None
    source_display_name: str | None
    status: str
    points: tuple[DevelopmentPoint, ...]
    normalized_development_auc: float
    sustained_t50: int | None
    sustained_t80: int | None
    final_development_success_rate: float
    validation_success_count: int
    validation_episode_count: int
    validation_success_rate: float
    elapsed_seconds: float
    evaluation_seconds: float
    evaluation_overhead_fraction: float
    final_checkpoint_path: Path
    final_checkpoint_sha256: str
    result_path: Path
    result_sha256: str

    @property
    def is_scratch(self) -> bool:
        return self.initialization_method == "scratch"


@dataclass(frozen=True, slots=True)
class TransferEffect:
    """One transferred condition's exact difference from matched scratch."""

    condition_id: str
    development_auc_delta_vs_scratch: float
    final_development_success_delta_vs_scratch: float
    validation_success_delta_vs_scratch: float


@dataclass(frozen=True, slots=True)
class ComparisonBundle:
    """Validated evidence ready for the Compare workspace."""

    path: Path
    study_id: str
    display_name: str
    evidence_status: str
    interpretation_boundary: str
    target_task_id: str
    target_display_name: str
    target_task_contract_digest: str
    grid_size: int
    target_budget: int
    target_seed: int
    replicate_block_id: str
    development_interval: int
    development_seed_digest: str
    development_episode_count: int
    validation_seed_digest: str
    validation_episode_count: int
    conditions: tuple[ComparisonCondition, ...]
    effects: tuple[TransferEffect, ...]
    source_evidence_path: Path
    source_evidence_sha256: str
    protected_manifest_path: Path
    protected_manifest_sha256: str

    @property
    def scratch(self) -> ComparisonCondition:
        return next(condition for condition in self.conditions if condition.is_scratch)


@dataclass(frozen=True, slots=True)
class ComparisonBundleExclusion:
    """One ignored comparison artifact and its fail-closed reason."""

    path: Path
    reason: str


@dataclass(frozen=True, slots=True)
class ComparisonBundleDiscovery:
    """Included and excluded comparison artifacts."""

    included: tuple[ComparisonBundle, ...]
    excluded: tuple[ComparisonBundleExclusion, ...]


def discover_comparison_bundles(
    comparison_root: str | Path = DEFAULT_COMPARISON_ROOT,
) -> ComparisonBundleDiscovery:
    """Discover valid bundles while retaining explicit exclusion reasons."""
    root = Path(comparison_root)
    included: list[ComparisonBundle] = []
    excluded: list[ComparisonBundleExclusion] = []
    if not root.is_dir():
        return ComparisonBundleDiscovery((), ())
    for path in sorted(root.glob("*.json")):
        try:
            included.append(load_comparison_bundle(path))
        except (OSError, ValueError, json.JSONDecodeError) as error:
            excluded.append(ComparisonBundleExclusion(path, str(error)))
    return ComparisonBundleDiscovery(tuple(included), tuple(excluded))


def load_comparison_bundle(path: str | Path) -> ComparisonBundle:
    """Load and fully validate one immutable comparison bundle."""
    bundle_path = Path(path).resolve()
    payload = json.loads(bundle_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ComparisonEvidenceError("Comparison bundle must be a JSON object.")
    if payload.get("artifact_type") != COMPARISON_ARTIFACT_TYPE:
        raise ComparisonEvidenceError("Unsupported comparison artifact type.")
    if payload.get("schema_version") != COMPARISON_SCHEMA_VERSION:
        raise ComparisonEvidenceError("Unsupported comparison schema version.")

    study = _mapping(payload, "study")
    target = _mapping(study, "target")
    development = _mapping(study, "development_evaluation")
    validation = _mapping(study, "validation_evaluation")
    provenance = _mapping(payload, "provenance")
    target_task_id = _string(target, "task_id")
    target_definition = require_released_task(target_task_id)
    target_digest = _sha256_string(target, "task_contract_digest")
    if target_definition.contract_digest != target_digest:
        raise ComparisonEvidenceError(
            "Bundle target digest does not match the released task contract."
        )

    source_path = _project_path(_string(provenance, "source_evidence_path"))
    source_sha = _sha256_string(provenance, "source_evidence_sha256")
    protected_path = _project_path(
        _string(provenance, "protected_manifest_path")
    )
    protected_sha = _sha256_string(
        provenance, "protected_manifest_sha256"
    )
    _require_file_hash(source_path, source_sha, "source evidence")
    _require_file_hash(protected_path, protected_sha, "protected manifest")
    _verify_protected_source(protected_path, source_path, source_sha)

    target_budget = _positive_int(study, "target_budget")
    target_seed = _nonnegative_int(study, "target_seed")
    interval = _positive_int(development, "interval")
    development_episode_count = _positive_int(development, "episode_count")
    validation_episode_count = _positive_int(validation, "episode_count")
    development_digest = _sha256_string(development, "ordered_seed_digest")
    validation_digest = _sha256_string(validation, "ordered_seed_digest")
    expected_steps = tuple(range(0, target_budget + 1, interval))
    if not expected_steps or expected_steps[-1] != target_budget:
        raise ComparisonEvidenceError(
            "Target budget is not an exact Development interval boundary."
        )

    raw_conditions = payload.get("conditions")
    if not isinstance(raw_conditions, list) or len(raw_conditions) < 2:
        raise ComparisonEvidenceError(
            "Comparison requires scratch and at least one transfer condition."
        )
    conditions = tuple(
        _condition_from_dict(
            item,
            expected_steps=expected_steps,
            target_budget=target_budget,
            development_episode_count=development_episode_count,
            validation_episode_count=validation_episode_count,
        )
        for item in raw_conditions
    )
    ids = tuple(condition.condition_id for condition in conditions)
    if len(set(ids)) != len(ids):
        raise ComparisonEvidenceError("Comparison condition IDs must be unique.")
    scratch = tuple(condition for condition in conditions if condition.is_scratch)
    if len(scratch) != 1:
        raise ComparisonEvidenceError(
            "Comparison bundle must contain exactly one scratch condition."
        )

    raw_effects = payload.get("effects")
    if not isinstance(raw_effects, list):
        raise ComparisonEvidenceError("Comparison effects must be a list.")
    effects = tuple(_effect_from_dict(item) for item in raw_effects)
    _verify_effects(conditions, effects)

    return ComparisonBundle(
        path=bundle_path,
        study_id=_string(study, "study_id"),
        display_name=_string(study, "display_name"),
        evidence_status=_string(study, "evidence_status"),
        interpretation_boundary=_string(study, "interpretation_boundary"),
        target_task_id=target_task_id,
        target_display_name=target_definition.display_name,
        target_task_contract_digest=target_digest,
        grid_size=_positive_int(target, "grid_size"),
        target_budget=target_budget,
        target_seed=target_seed,
        replicate_block_id=_string(study, "replicate_block_id"),
        development_interval=interval,
        development_seed_digest=development_digest,
        development_episode_count=development_episode_count,
        validation_seed_digest=validation_digest,
        validation_episode_count=validation_episode_count,
        conditions=conditions,
        effects=effects,
        source_evidence_path=source_path,
        source_evidence_sha256=source_sha,
        protected_manifest_path=protected_path,
        protected_manifest_sha256=protected_sha,
    )


def normalized_development_auc(
    points: tuple[DevelopmentPoint, ...],
    target_budget: int,
) -> float:
    """Compute the predeclared trapezoidal success AUC."""
    area = 0.0
    for left, right in zip(points, points[1:], strict=False):
        width = right.step - left.step
        area += width * (left.success_rate + right.success_rate) / 2
    return area / target_budget


def sustained_threshold_step(
    points: tuple[DevelopmentPoint, ...],
    threshold: float,
) -> int | None:
    """Return the second point in the first consecutive threshold pair."""
    for previous, current in zip(points, points[1:], strict=False):
        if (
            previous.success_rate >= threshold
            and current.success_rate >= threshold
        ):
            return current.step
    return None


def _condition_from_dict(
    value: Any,
    *,
    expected_steps: tuple[int, ...],
    target_budget: int,
    development_episode_count: int,
    validation_episode_count: int,
) -> ComparisonCondition:
    if not isinstance(value, Mapping):
        raise ComparisonEvidenceError("Comparison condition must be an object.")
    initialization = _mapping(value, "initialization")
    metrics = _mapping(value, "metrics")
    runtime = _mapping(value, "runtime")
    artifacts = _mapping(value, "artifacts")
    method = _string(initialization, "method")
    if method not in {"scratch", "full_policy"}:
        raise ComparisonEvidenceError(
            f"Unsupported comparison initialization method {method!r}."
        )
    source_task_id = initialization.get("source_task_id")
    source_display_name: str | None = None
    if method == "scratch":
        if source_task_id is not None:
            raise ComparisonEvidenceError(
                "Scratch condition cannot declare a source task."
            )
    else:
        if not isinstance(source_task_id, str) or not source_task_id:
            raise ComparisonEvidenceError(
                "Transfer condition must declare its source task."
            )
        source_display_name = require_released_task(source_task_id).display_name

    raw_points = value.get("development_points")
    if not isinstance(raw_points, list):
        raise ComparisonEvidenceError("Development points must be a list.")
    points = tuple(
        _point_from_dict(point, development_episode_count)
        for point in raw_points
    )
    if tuple(point.step for point in points) != expected_steps:
        raise ComparisonEvidenceError(
            "Development points do not use the complete common measured grid."
        )

    auc = _rate(metrics, "normalized_development_auc")
    t50 = _optional_nonnegative_int(metrics, "sustained_t50")
    t80 = _optional_nonnegative_int(metrics, "sustained_t80")
    final_development = _rate(metrics, "final_development_success_rate")
    validation_success_count = _nonnegative_int(
        metrics, "validation_success_count"
    )
    serialized_validation_count = _positive_int(
        metrics, "validation_episode_count"
    )
    validation_rate = _rate(metrics, "validation_success_rate")
    if serialized_validation_count != validation_episode_count:
        raise ComparisonEvidenceError(
            "Condition Validation count differs from the study contract."
        )
    if validation_success_count > validation_episode_count:
        raise ComparisonEvidenceError(
            "Validation successes exceed the episode count."
        )
    if not math.isclose(
        validation_rate,
        validation_success_count / validation_episode_count,
        abs_tol=1e-12,
    ):
        raise ComparisonEvidenceError(
            "Validation success rate does not match its exact count."
        )
    expected_auc = normalized_development_auc(points, target_budget)
    expected_t50 = sustained_threshold_step(points, 0.5)
    expected_t80 = sustained_threshold_step(points, 0.8)
    if not math.isclose(auc, expected_auc, abs_tol=1e-12):
        raise ComparisonEvidenceError("Serialized Development AUC is incorrect.")
    if t50 != expected_t50 or t80 != expected_t80:
        raise ComparisonEvidenceError(
            "Serialized sustained threshold time is incorrect."
        )
    if not math.isclose(
        final_development, points[-1].success_rate, abs_tol=1e-12
    ):
        raise ComparisonEvidenceError(
            "Final Development success does not match the last point."
        )

    elapsed_seconds = _nonnegative_number(runtime, "elapsed_seconds")
    evaluation_seconds = _nonnegative_number(runtime, "evaluation_seconds")
    overhead = _rate(runtime, "evaluation_overhead_fraction")
    if elapsed_seconds == 0 or not math.isclose(
        overhead, evaluation_seconds / elapsed_seconds, abs_tol=1e-12
    ):
        raise ComparisonEvidenceError(
            "Evaluation overhead fraction does not match elapsed time."
        )

    final_checkpoint_path = _project_path(
        _string(artifacts, "final_checkpoint_path")
    )
    final_checkpoint_sha256 = _sha256_string(
        artifacts, "final_checkpoint_sha256"
    )
    result_path = _project_path(_string(artifacts, "result_path"))
    result_sha256 = _sha256_string(artifacts, "result_sha256")
    _require_file_hash(
        final_checkpoint_path,
        final_checkpoint_sha256,
        f"{_string(value, 'condition_id')} final checkpoint",
    )
    _require_file_hash(
        result_path,
        result_sha256,
        f"{_string(value, 'condition_id')} result",
    )

    return ComparisonCondition(
        condition_id=_string(value, "condition_id"),
        display_name=_string(value, "display_name"),
        initialization_method=method,
        source_task_id=source_task_id,
        source_display_name=source_display_name,
        status=_string(value, "status"),
        points=points,
        normalized_development_auc=auc,
        sustained_t50=t50,
        sustained_t80=t80,
        final_development_success_rate=final_development,
        validation_success_count=validation_success_count,
        validation_episode_count=validation_episode_count,
        validation_success_rate=validation_rate,
        elapsed_seconds=elapsed_seconds,
        evaluation_seconds=evaluation_seconds,
        evaluation_overhead_fraction=overhead,
        final_checkpoint_path=final_checkpoint_path,
        final_checkpoint_sha256=final_checkpoint_sha256,
        result_path=result_path,
        result_sha256=result_sha256,
    )


def _point_from_dict(
    value: Any,
    expected_episode_count: int,
) -> DevelopmentPoint:
    if not isinstance(value, Mapping):
        raise ComparisonEvidenceError("Development point must be an object.")
    episode_count = _positive_int(value, "episode_count")
    success_count = _nonnegative_int(value, "success_count")
    error_count = _nonnegative_int(value, "error_count")
    success_rate = _rate(value, "success_rate")
    if episode_count != expected_episode_count:
        raise ComparisonEvidenceError(
            "Development point episode count differs from the study contract."
        )
    if success_count > episode_count:
        raise ComparisonEvidenceError(
            "Development successes exceed the episode count."
        )
    if error_count != 0:
        raise ComparisonEvidenceError(
            "Incomplete Development points cannot enter governed AUC."
        )
    if not math.isclose(
        success_rate, success_count / episode_count, abs_tol=1e-12
    ):
        raise ComparisonEvidenceError(
            "Development success rate does not match its exact count."
        )
    return DevelopmentPoint(
        step=_nonnegative_int(value, "step"),
        success_count=success_count,
        episode_count=episode_count,
        success_rate=success_rate,
        mean_reward=_number(value, "mean_reward"),
        mean_episode_length=_nonnegative_number(value, "mean_episode_length"),
        error_count=error_count,
    )


def _effect_from_dict(value: Any) -> TransferEffect:
    if not isinstance(value, Mapping):
        raise ComparisonEvidenceError("Transfer effect must be an object.")
    return TransferEffect(
        condition_id=_string(value, "condition_id"),
        development_auc_delta_vs_scratch=_number(
            value, "development_auc_delta_vs_scratch"
        ),
        final_development_success_delta_vs_scratch=_number(
            value, "final_development_success_delta_vs_scratch"
        ),
        validation_success_delta_vs_scratch=_number(
            value, "validation_success_delta_vs_scratch"
        ),
    )


def _verify_effects(
    conditions: tuple[ComparisonCondition, ...],
    effects: tuple[TransferEffect, ...],
) -> None:
    by_id = {condition.condition_id: condition for condition in conditions}
    scratch = next(condition for condition in conditions if condition.is_scratch)
    expected_ids = {
        condition.condition_id
        for condition in conditions
        if not condition.is_scratch
    }
    if {effect.condition_id for effect in effects} != expected_ids:
        raise ComparisonEvidenceError(
            "Transfer effects must cover every non-scratch condition exactly once."
        )
    for effect in effects:
        condition = by_id[effect.condition_id]
        expected = (
            condition.normalized_development_auc
            - scratch.normalized_development_auc,
            condition.final_development_success_rate
            - scratch.final_development_success_rate,
            condition.validation_success_rate - scratch.validation_success_rate,
        )
        actual = (
            effect.development_auc_delta_vs_scratch,
            effect.final_development_success_delta_vs_scratch,
            effect.validation_success_delta_vs_scratch,
        )
        if any(
            not math.isclose(left, right, abs_tol=1e-12)
            for left, right in zip(expected, actual, strict=True)
        ):
            raise ComparisonEvidenceError(
                f"Transfer effect for {effect.condition_id!r} is incorrect."
            )


def _verify_protected_source(
    protected_manifest_path: Path,
    source_path: Path,
    source_sha256: str,
) -> None:
    payload = json.loads(protected_manifest_path.read_text(encoding="utf-8"))
    files = payload.get("files")
    if not isinstance(files, Mapping):
        raise ComparisonEvidenceError(
            "Protected evidence manifest has no file catalog."
        )
    record = files.get("matched_pilot_summary")
    if not isinstance(record, Mapping):
        raise ComparisonEvidenceError(
            "Protected evidence manifest does not bind the matched pilot."
        )
    protected_source = _project_path(_string(record, "path"))
    protected_sha = _sha256_string(record, "sha256")
    if protected_source != source_path or protected_sha != source_sha256:
        raise ComparisonEvidenceError(
            "Comparison source is not the protected matched-pilot summary."
        )


def _require_file_hash(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise ComparisonEvidenceError(f"{label.title()} is missing: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != expected:
        raise ComparisonEvidenceError(f"{label.title()} SHA-256 mismatch.")


def _project_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        resolved = path.resolve()
    else:
        resolved = (PROJECT_ROOT / path).resolve()
    if not resolved.is_relative_to(PROJECT_ROOT):
        raise ComparisonEvidenceError(
            "Comparison provenance path escapes the project root."
        )
    return resolved


def _mapping(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ComparisonEvidenceError(f"{key} must be an object.")
    return value


def _string(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ComparisonEvidenceError(f"{key} must be a non-empty string.")
    return value


def _sha256_string(payload: Mapping[str, Any], key: str) -> str:
    value = _string(payload, key)
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ComparisonEvidenceError(f"{key} must be a lowercase SHA-256.")
    return value


def _positive_int(payload: Mapping[str, Any], key: str) -> int:
    value = payload.get(key)
    if type(value) is not int or value <= 0:
        raise ComparisonEvidenceError(f"{key} must be a positive integer.")
    return value


def _nonnegative_int(payload: Mapping[str, Any], key: str) -> int:
    value = payload.get(key)
    if type(value) is not int or value < 0:
        raise ComparisonEvidenceError(f"{key} must be a nonnegative integer.")
    return value


def _optional_nonnegative_int(
    payload: Mapping[str, Any],
    key: str,
) -> int | None:
    value = payload.get(key)
    if value is None:
        return None
    if type(value) is not int or value < 0:
        raise ComparisonEvidenceError(
            f"{key} must be null or a nonnegative integer."
        )
    return value


def _number(payload: Mapping[str, Any], key: str) -> float:
    value = payload.get(key)
    if type(value) not in {int, float} or not math.isfinite(float(value)):
        raise ComparisonEvidenceError(f"{key} must be a finite number.")
    return float(value)


def _nonnegative_number(payload: Mapping[str, Any], key: str) -> float:
    value = _number(payload, key)
    if value < 0:
        raise ComparisonEvidenceError(f"{key} must be nonnegative.")
    return value


def _rate(payload: Mapping[str, Any], key: str) -> float:
    value = _number(payload, key)
    if not 0 <= value <= 1:
        raise ComparisonEvidenceError(f"{key} must be between zero and one.")
    return value
