"""Independent, fail-closed verification for the certified transfer matrix.

This module treats the serialized certified table as a set of claims.  Every
returned metric is recomputed from the retained standard experiment trail;
TensorBoard data and historical comparison bundles are never consulted.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from environments.catalog import require_released_task
from training.artifact_catalog import load_normalized_evaluation_report
from training.benchmark_contracts import DEVELOPMENT_SEEDS, VALIDATION_SEEDS
from training.checkpoints import (
    checkpoint_metadata_path,
    checkpoint_sha256,
    load_checkpoint_metadata,
)
from training.source_pool import (
    DEFAULT_SOURCE_POOL_PATH,
    SourcePoolEntry,
    load_source_pool,
)
from training.portable_paths import PortablePathError, resolve_recorded_path
from training.transfer_governance import assess_source_competence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LEGACY_CERTIFIED_DATASET = (
    PROJECT_ROOT
    / "reports"
    / "data_integrity"
    / "certified_n30_v3"
    / "certified_n30.json"
)
EXPANDED_CERTIFIED_DATASET = (
    PROJECT_ROOT / "reports" / "data_integrity" / "certified_n40_v1" / "certified_n40.json"
)
COMBINED_CERTIFIED_DATASET = (
    PROJECT_ROOT / "reports" / "data_integrity" / "certified_combined_n45_v1" / "certified_combined_n45.json"
)
DEFAULT_CERTIFIED_DATASET = LEGACY_CERTIFIED_DATASET
DEVELOPMENT_INTERVAL = 102_400
POOL_ID = "weekly-transfer-source-pool-v1"
SOURCE_SEED = 19


class EvidenceIntegrityError(RuntimeError):
    """Raised when one claimed evidence dependency cannot be proven."""


@dataclass(frozen=True, slots=True)
class VerifiedRun:
    run_id: str
    task_id: str
    seed: int
    budget: int
    initialization_method: str
    source_task_id: str | None
    source_checkpoint_sha256: str | None
    checkpoint_path: Path
    checkpoint_sha256: str
    development_curve_path: Path
    validation_report_paths: tuple[Path, ...]
    normalized_development_auc: float
    sustained_t50: int | None
    sustained_t80: int | None
    final_development_success_rate: float
    validation_success_rate: float
    elapsed_seconds: float
    dependencies: tuple[tuple[Path, str], ...]

    def metrics(self) -> dict[str, Any]:
        return {
            "auc": self.normalized_development_auc,
            "t50": self.sustained_t50,
            "t80": self.sustained_t80,
            "final_dev": self.final_development_success_rate,
            "validation_rate": self.validation_success_rate,
        }


@dataclass(frozen=True, slots=True)
class VerifiedCell:
    pair: int
    seed: int
    source_task_id: str
    target_task_id: str
    budget: int
    transfer: VerifiedRun
    scratch: VerifiedRun
    delta_auc: float


@dataclass(frozen=True, slots=True)
class CertifiedDatasetAudit:
    dataset_path: Path
    dataset_sha256: str
    cells: tuple[VerifiedCell, ...]
    transfer_run_count: int
    scratch_run_count: int
    expected_cell_count: int
    expected_transfer_run_count: int
    expected_scratch_run_count: int
    dependency_hashes: tuple[tuple[Path, str], ...]

    @property
    def fully_certified(self) -> bool:
        return (
            len(self.cells) == self.expected_cell_count
            and self.transfer_run_count == self.expected_transfer_run_count
            and self.scratch_run_count == self.expected_scratch_run_count
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_type": "transfergrid-independent-recertification",
            "schema_version": 1,
            "status": "FULL_PASS" if self.fully_certified else "PARTIAL",
            "dataset_path": str(self.dataset_path),
            "dataset_sha256": self.dataset_sha256,
            "cell_count": len(self.cells),
            "transfer_run_count": self.transfer_run_count,
            "scratch_run_count": self.scratch_run_count,
            "expected_cell_count": self.expected_cell_count,
            "expected_transfer_run_count": self.expected_transfer_run_count,
            "expected_scratch_run_count": self.expected_scratch_run_count,
            "source_pool_key": [POOL_ID, "<task_id>", SOURCE_SEED],
            "tensorboard_used_as_governed_evidence": False,
            "cells": [
                {
                    "pair": cell.pair,
                    "seed": cell.seed,
                    "source_task_id": cell.source_task_id,
                    "target_task_id": cell.target_task_id,
                    "budget": cell.budget,
                    "transfer_run_id": cell.transfer.run_id,
                    "scratch_run_id": cell.scratch.run_id,
                    "transfer_checkpoint_sha256": cell.transfer.checkpoint_sha256,
                    "scratch_checkpoint_sha256": cell.scratch.checkpoint_sha256,
                    "source_checkpoint_sha256": (
                        cell.transfer.source_checkpoint_sha256
                    ),
                    "transfer_auc": cell.transfer.normalized_development_auc,
                    "scratch_auc": cell.scratch.normalized_development_auc,
                    "delta_auc": cell.delta_auc,
                    "transfer_validation_rate": (
                        cell.transfer.validation_success_rate
                    ),
                    "scratch_validation_rate": cell.scratch.validation_success_rate,
                    "status": "PASS",
                }
                for cell in self.cells
            ],
            "dependencies": [
                {
                    "path": str(path),
                    "sha256": digest,
                    "size": path.stat().st_size,
                }
                for path, digest in self.dependency_hashes
            ],
        }


def recertify_certified_n30(
    dataset_path: str | Path = DEFAULT_CERTIFIED_DATASET,
    *,
    source_pool_path: str | Path = DEFAULT_SOURCE_POOL_PATH,
) -> CertifiedDatasetAudit:
    """Recompute every declared cell from its standard experiment dependencies."""
    dataset = _inside_project(Path(dataset_path))
    payload = _json_mapping(dataset, "certified dataset")
    if payload.get("schema_version") != 1:
        raise EvidenceIntegrityError("Unsupported certified dataset schema.")
    raw_rows = payload.get("rows")
    raw_evidence = payload.get("evidence")
    if not isinstance(raw_rows, list) or not isinstance(raw_evidence, list):
        raise EvidenceIntegrityError("Certified dataset requires rows and evidence.")
    expected_counts = payload.get("expected_counts")
    if expected_counts is None:
        expected_cell_count, expected_transfer_count, expected_scratch_count = 30, 30, 12
        expected_keys = {(pair, seed) for pair in range(1, 11) for seed in (19, 20, 21)}
    else:
        counts = _mapping(expected_counts, "expected counts")
        expected_cell_count = _exact_int(counts, "cells")
        expected_transfer_count = _exact_int(counts, "transfer_runs")
        expected_scratch_count = _exact_int(counts, "scratch_runs")
        declared_cells = payload.get("expected_cells")
        if not isinstance(declared_cells, list):
            raise EvidenceIntegrityError("Expanded datasets require expected_cells.")
        expected_keys = {
            (_exact_int(_mapping(cell, "expected cell"), "pair"),
             _exact_int(_mapping(cell, "expected cell"), "seed"))
            for cell in declared_cells
        }
        if len(expected_keys) != expected_cell_count:
            raise EvidenceIntegrityError("Expanded expected_cells are incomplete or duplicated.")
    if len(raw_rows) != expected_cell_count or len(raw_evidence) != expected_cell_count:
        raise EvidenceIntegrityError(
            f"Certified dataset must contain exactly {expected_cell_count} cells."
        )

    source_pool_file = _inside_project(Path(source_pool_path))
    pool = load_source_pool(source_pool_file)
    source_competence_cache: dict[tuple[str, str, int], SourcePoolEntry] = {}
    verified_runs: dict[tuple[str, str | None], VerifiedRun] = {}
    rows_by_key = {
        (_exact_int(row, "Pair"), _exact_int(row, "Seed")): row
        for row in raw_rows
        if isinstance(row, Mapping)
    }
    if len(rows_by_key) != expected_cell_count:
        raise EvidenceIntegrityError("Certified table has duplicate Pair/Seed cells.")

    cells: list[VerifiedCell] = []
    dependencies: dict[Path, str] = {
        dataset: _sha256(dataset),
        source_pool_file: _sha256(source_pool_file),
    }
    for claim in raw_evidence:
        if not isinstance(claim, Mapping):
            raise EvidenceIntegrityError("Evidence cell must be an object.")
        pair = _exact_int(claim, "pair")
        seed = _exact_int(claim, "seed")
        row = rows_by_key.get((pair, seed))
        if row is None:
            raise EvidenceIntegrityError(f"Missing table row for Pair {pair} seed {seed}.")
        source_task = _exact_string(row, "Source")
        target_task = _exact_string(row, "Target")
        budget = _exact_int(row, "Budget")
        transfer_claim = _mapping(claim.get("transfer"), "transfer claim")
        scratch_claim = _mapping(claim.get("scratch"), "scratch claim")

        transfer_key = (_exact_string(transfer_claim, "run_id"), source_task)
        transfer = verified_runs.get(transfer_key)
        if transfer is None:
            transfer = _verify_standard_run(
                transfer_claim,
                expected_task_id=target_task,
                expected_seed=seed,
                expected_budget=budget,
                expected_source_task_id=source_task,
                source_pool=pool,
                source_competence_cache=source_competence_cache,
            )
            verified_runs[transfer_key] = transfer

        scratch_key = (_exact_string(scratch_claim, "run_id"), None)
        scratch = verified_runs.get(scratch_key)
        if scratch is None:
            scratch = _verify_standard_run(
                scratch_claim,
                expected_task_id=target_task,
                expected_seed=seed,
                expected_budget=budget,
                expected_source_task_id=None,
                source_pool=pool,
                source_competence_cache=source_competence_cache,
            )
            verified_runs[scratch_key] = scratch

        _verify_claimed_run_metrics(transfer_claim, transfer)
        _verify_claimed_run_metrics(scratch_claim, scratch)
        delta = transfer.normalized_development_auc - scratch.normalized_development_auc
        _close(_exact_number(row, "Dev AUC"), transfer.normalized_development_auc, "transfer AUC")
        _close(_exact_number(row, "Scratch AUC"), scratch.normalized_development_auc, "scratch AUC")
        _close(_exact_number(row, "Delta AUC"), delta, "delta AUC")
        _close(_exact_number(row, "Final Dev %"), transfer.final_development_success_rate, "final Development")
        _close(_exact_number(row, "Validation %"), transfer.validation_success_rate, "Validation rate")
        if row.get("T50") != transfer.sustained_t50 or row.get("T80") != transfer.sustained_t80:
            raise EvidenceIntegrityError(f"Threshold mismatch for Pair {pair} seed {seed}.")
        if row.get("Source-Checkpoint-SHA256") != transfer.source_checkpoint_sha256:
            raise EvidenceIntegrityError(f"Source SHA mismatch for Pair {pair} seed {seed}.")
        if row.get("Run Identity") != transfer.run_id:
            raise EvidenceIntegrityError(f"Run identity mismatch for Pair {pair} seed {seed}.")
        for path, digest in transfer.dependencies + scratch.dependencies:
            previous = dependencies.setdefault(path, digest)
            if previous != digest:
                raise EvidenceIntegrityError(f"Dependency changed during audit: {path}")
        cells.append(VerifiedCell(
            pair=pair,
            seed=seed,
            source_task_id=source_task,
            target_task_id=target_task,
            budget=budget,
            transfer=transfer,
            scratch=scratch,
            delta_auc=delta,
        ))

    keys = {(cell.pair, cell.seed) for cell in cells}
    if keys != expected_keys:
        raise EvidenceIntegrityError("Evidence does not match the declared cell contract.")
    transfer_ids = {cell.transfer.run_id for cell in cells}
    scratch_ids = {cell.scratch.run_id for cell in cells}
    if len(transfer_ids) != expected_transfer_count or len(scratch_ids) != expected_scratch_count:
        raise EvidenceIntegrityError(
            f"Expected {expected_transfer_count} transfer and {expected_scratch_count} scratch runs; "
            f"found {len(transfer_ids)} and {len(scratch_ids)}."
        )
    return CertifiedDatasetAudit(
        dataset_path=dataset,
        dataset_sha256=dependencies[dataset],
        cells=tuple(sorted(cells, key=lambda item: (item.pair, item.seed))),
        transfer_run_count=len(transfer_ids),
        scratch_run_count=len(scratch_ids),
        expected_cell_count=expected_cell_count,
        expected_transfer_run_count=expected_transfer_count,
        expected_scratch_run_count=expected_scratch_count,
        dependency_hashes=tuple(sorted(dependencies.items(), key=lambda item: str(item[0]))),
    )


def _verify_standard_run(
    claim: Mapping[str, Any],
    *,
    expected_task_id: str,
    expected_seed: int,
    expected_budget: int,
    expected_source_task_id: str | None,
    source_pool: Mapping[tuple[str, str, int], SourcePoolEntry],
    source_competence_cache: dict[tuple[str, str, int], SourcePoolEntry],
) -> VerifiedRun:
    run_id = _exact_string(claim, "run_id")
    root = (PROJECT_ROOT / "experiments" / run_id).resolve()
    if not root.is_relative_to((PROJECT_ROOT / "experiments").resolve()):
        raise EvidenceIntegrityError(f"Experiment path escapes root: {run_id}")
    canonical = require_standard_run_artifacts(run_id)
    for field, path in canonical.items():
        if _inside_project(Path(_exact_string(claim, field))) != path.resolve():
            raise EvidenceIntegrityError(f"Non-canonical {field} for {run_id}.")
        if not path.is_file():
            raise EvidenceIntegrityError(f"Missing {field} for {run_id}: {path}")

    summary = _json_mapping(canonical["training_summary_path"], "training summary")
    manifest = _json_mapping(canonical["manifest_path"], "experiment manifest")
    request = _json_mapping(canonical["runner_request_path"], "runner request")
    result = _json_mapping(canonical["runner_result_path"], "runner result")
    curve = _json_mapping(canonical["development_curve_path"], "Development curve")
    if summary.get("artifact_type") != "transfergrid-experiment-training-summary" or summary.get("schema_version") != 2:
        raise EvidenceIntegrityError(f"Unsupported training summary for {run_id}.")
    if manifest.get("artifact_type") != "transfergrid-experiment-manifest" or manifest.get("schema_version") != 2:
        raise EvidenceIntegrityError(f"Unsupported manifest for {run_id}.")
    if manifest.get("status") != "succeeded" or result.get("status") != "succeeded":
        raise EvidenceIntegrityError(f"Run is not complete: {run_id}")
    result_summary = _mapping(result.get("summary"), "runner result summary")
    for label, value in (
        ("summary", summary.get("run_id")),
        ("manifest", manifest.get("run_id")),
        ("request", request.get("run_id")),
        ("result", result.get("run_id") or result_summary.get("run_id")),
        ("curve", curve.get("run_id")),
    ):
        if value != run_id:
            raise EvidenceIntegrityError(f"{label} run ID mismatch for {run_id}.")

    definition = require_released_task(expected_task_id)
    expected_environment = {
        "task_id": expected_task_id,
        "task_contract_digest": definition.contract_digest,
        "grid_size": 7,
    }
    for label, environment in (
        ("summary", summary.get("environment")),
        ("request", request.get("environment")),
    ):
        env = _mapping(environment, f"{label} environment")
        for key, expected in expected_environment.items():
            if env.get(key) != expected:
                raise EvidenceIntegrityError(f"{label} {key} mismatch for {run_id}.")
    if summary.get("seed") != expected_seed or request.get("seed") != expected_seed:
        raise EvidenceIntegrityError(f"PPO seed mismatch for {run_id}.")
    if summary.get("requested_timesteps") != expected_budget or request.get("total_timesteps") != expected_budget:
        raise EvidenceIntegrityError(f"Training budget mismatch for {run_id}.")
    for key in ("seed", "requested_timesteps", "checkpoint_path"):
        if result_summary.get(key) != summary.get(key):
            raise EvidenceIntegrityError(f"Runner result {key} mismatch for {run_id}.")

    checkpoint_path = _inside_project(Path(_exact_string(summary, "checkpoint_path")))
    if checkpoint_path != _inside_project(Path(_exact_string(claim, "checkpoint_path"))):
        raise EvidenceIntegrityError(f"Checkpoint claim mismatch for {run_id}.")
    checkpoint_hash = _exact_string(summary, "checkpoint_sha256")
    if not checkpoint_path.is_file() or checkpoint_sha256(checkpoint_path) != checkpoint_hash:
        raise EvidenceIntegrityError(f"Checkpoint SHA mismatch for {run_id}.")
    if claim.get("checkpoint_sha256") != checkpoint_hash:
        raise EvidenceIntegrityError(f"Claimed checkpoint SHA mismatch for {run_id}.")
    metadata_path = checkpoint_metadata_path(checkpoint_path).resolve()
    metadata = load_checkpoint_metadata(checkpoint_path)
    if metadata.get("schema_version") != 3 or metadata.get("run_id") != run_id:
        raise EvidenceIntegrityError(f"Checkpoint metadata identity mismatch for {run_id}.")
    if metadata.get("seed") != expected_seed or metadata.get("requested_timesteps") != expected_budget:
        raise EvidenceIntegrityError(f"Checkpoint metadata seed/budget mismatch for {run_id}.")
    metadata_environment = _mapping(metadata.get("environment"), "checkpoint environment")
    if any(metadata_environment.get(key) != value for key, value in expected_environment.items()):
        raise EvidenceIntegrityError(f"Checkpoint task contract mismatch for {run_id}.")
    manifest_checkpoint = _mapping(manifest.get("checkpoint"), "manifest checkpoint")
    if (
        _inside_project(Path(_exact_string(manifest_checkpoint, "path"))) != checkpoint_path
        or manifest_checkpoint.get("sha256") != checkpoint_hash
        or _inside_project(Path(_exact_string(manifest_checkpoint, "metadata_path"))) != metadata_path
    ):
        raise EvidenceIntegrityError(f"Manifest checkpoint binding mismatch for {run_id}.")

    summary_initialization = _mapping(summary.get("initialization"), "summary initialization")
    request_initialization = _mapping(request.get("initialization"), "request initialization")
    source_sha: str | None = None
    if expected_source_task_id is None:
        if summary_initialization.get("method") != "scratch" or request_initialization.get("method") != "scratch":
            raise EvidenceIntegrityError(f"Scratch provenance mismatch for {run_id}.")
        forbidden = ("source_checkpoint_path", "expected_source_sha256", "source_task_id")
        if any(summary_initialization.get(key) or request_initialization.get(key) for key in forbidden):
            raise EvidenceIntegrityError(f"Scratch run declares a source: {run_id}.")
    else:
        if summary_initialization.get("method") != "full_policy" or request_initialization.get("method") != "full_policy":
            raise EvidenceIntegrityError(f"Transfer provenance mismatch for {run_id}.")
        pool_key = (POOL_ID, expected_source_task_id, SOURCE_SEED)
        entry = source_pool.get(pool_key)
        if entry is None:
            raise EvidenceIntegrityError(f"Missing immutable source pool key {pool_key!r}.")
        if pool_key not in source_competence_cache:
            _verify_source_pool_entry(entry)
            source_competence_cache[pool_key] = entry
        source_sha = entry.checkpoint_sha256
        for initialization in (summary_initialization, request_initialization):
            if (
                initialization.get("source_task_id") != entry.task_id
                or initialization.get("source_run_id") != entry.run_id
                or initialization.get("expected_source_sha256") != entry.checkpoint_sha256
                or _inside_project(Path(_exact_string(initialization, "source_checkpoint_path"))) != entry.checkpoint_path
            ):
                raise EvidenceIntegrityError(f"Immutable source-pool mismatch for {run_id}.")
        technical = _mapping(
            summary_initialization.get("technical_compatibility")
            or summary_initialization.get("compatibility"),
            "technical compatibility",
        )
        competence = _mapping(summary_initialization.get("source_competence"), "source competence")
        if technical.get("compatible") is not True or competence.get("competent") is not True:
            raise EvidenceIntegrityError(f"Transfer governance failed for {run_id}.")
        for group, label in ((technical, "compatibility"), (competence, "competence")):
            checks = group.get("checks")
            if not isinstance(checks, list) or any(
                not isinstance(check, Mapping) or check.get("passed") is not True
                for check in checks
            ):
                raise EvidenceIntegrityError(f"Incomplete {label} evidence for {run_id}.")
        if summary_initialization.get("transfer_preflight", {}).get("ready") is not True:
            raise EvidenceIntegrityError(f"Transfer preflight was not ready for {run_id}.")

    development_metrics = _verify_development_curve(
        curve,
        run_id=run_id,
        task_id=expected_task_id,
        task_contract_digest=definition.contract_digest,
        budget=expected_budget,
    )
    validation_paths = tuple(
        _inside_project(Path(value))
        for value in claim.get("validation_report_paths", [])
        if isinstance(value, str)
    )
    if not validation_paths:
        raise EvidenceIntegrityError(f"No Validation evidence claimed for {run_id}.")
    validation_rates = {
        _verify_validation_report(
            path,
            checkpoint_path=checkpoint_path,
            checkpoint_hash=checkpoint_hash,
            task_id=expected_task_id,
            task_contract_digest=definition.contract_digest,
        )
        for path in validation_paths
    }
    if len(validation_rates) != 1:
        raise EvidenceIntegrityError(f"Conflicting Validation reports for {run_id}.")
    validation_rate = next(iter(validation_rates))
    dependencies = [
        (path.resolve(), _sha256(path))
        for path in (
            canonical["training_summary_path"],
            canonical["manifest_path"],
            canonical["runner_request_path"],
            canonical["runner_result_path"],
            canonical["development_curve_path"],
            checkpoint_path,
            metadata_path,
            *validation_paths,
        )
    ]
    return VerifiedRun(
        run_id=run_id,
        task_id=expected_task_id,
        seed=expected_seed,
        budget=expected_budget,
        initialization_method="full_policy" if expected_source_task_id else "scratch",
        source_task_id=expected_source_task_id,
        source_checkpoint_sha256=source_sha,
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=checkpoint_hash,
        development_curve_path=canonical["development_curve_path"].resolve(),
        validation_report_paths=validation_paths,
        normalized_development_auc=development_metrics["auc"],
        sustained_t50=development_metrics["t50"],
        sustained_t80=development_metrics["t80"],
        final_development_success_rate=development_metrics["final"],
        validation_success_rate=validation_rate,
        elapsed_seconds=_exact_number(summary, "elapsed_seconds"),
        dependencies=tuple(dependencies),
    )


def require_standard_run_artifacts(
    run_id: str,
    *,
    project_root: str | Path = PROJECT_ROOT,
) -> dict[str, Path]:
    """Resolve the mandatory standard-run trail and reject any missing file."""
    root = Path(project_root).resolve()
    experiment = root / "experiments" / run_id
    paths = {
        "training_summary_path": experiment / "training_summary.json",
        "manifest_path": experiment / "manifest.json",
        "development_curve_path": experiment / "development_curve.json",
        "runner_request_path": root / "logs" / "training_v0" / f"{run_id}.request.json",
        "runner_result_path": root / "logs" / "training_v0" / f"{run_id}.result.json",
    }
    for field, path in paths.items():
        if not path.is_file():
            raise EvidenceIntegrityError(f"Missing {field} for {run_id}: {path}")
    return paths


def _verify_source_pool_entry(entry: SourcePoolEntry) -> None:
    if entry.key != (POOL_ID, entry.task_id, SOURCE_SEED):
        raise EvidenceIntegrityError(f"Wrong source-pool identity: {entry.key!r}")
    if not entry.checkpoint_path.is_file() or checkpoint_sha256(entry.checkpoint_path) != entry.checkpoint_sha256:
        raise EvidenceIntegrityError(f"Source checkpoint SHA mismatch: {entry.checkpoint_path}")
    metadata = load_checkpoint_metadata(entry.checkpoint_path)
    environment = _mapping(metadata.get("environment"), "source environment")
    definition = require_released_task(entry.task_id)
    if (
        metadata.get("schema_version") != 3
        or metadata.get("run_id") != entry.run_id
        or metadata.get("seed") != SOURCE_SEED
        or metadata.get("checkpoint_sha256") != entry.checkpoint_sha256
        or environment.get("task_id") != entry.task_id
        or environment.get("task_contract_digest") != definition.contract_digest
    ):
        raise EvidenceIntegrityError(f"Source metadata mismatch: {entry.checkpoint_path}")
    competence = assess_source_competence(entry.checkpoint_path)
    if not competence.competent:
        raise EvidenceIntegrityError(
            f"Source competence failed for {entry.task_id}: {competence.failure_codes}"
        )


def _verify_development_curve(
    curve: Mapping[str, Any],
    *,
    run_id: str,
    task_id: str,
    task_contract_digest: str,
    budget: int,
) -> dict[str, Any]:
    if (
        curve.get("artifact_type") != "transfergrid-development-curve"
        or curve.get("schema_version") != 1
        or curve.get("run_id") != run_id
        or curve.get("task_id") != task_id
        or curve.get("task_contract_digest") != task_contract_digest
        or curve.get("evaluation_interval") != DEVELOPMENT_INTERVAL
        or curve.get("deterministic_actions") is not True
        or curve.get("separate_evaluation_environments") is not True
        or curve.get("rng_state_restored") is not True
    ):
        raise EvidenceIntegrityError(f"Development contract mismatch for {run_id}.")
    points = curve.get("points")
    if not isinstance(points, list):
        raise EvidenceIntegrityError(f"Missing Development points for {run_id}.")
    expected_steps = list(range(0, budget + 1, DEVELOPMENT_INTERVAL))
    if [point.get("step") for point in points if isinstance(point, Mapping)] != expected_steps:
        raise EvidenceIntegrityError(f"Incomplete Development step grid for {run_id}.")
    rates: list[float] = []
    for point in points:
        mapping = _mapping(point, "Development point")
        records = mapping.get("records")
        if not isinstance(records, list) or len(records) != len(DEVELOPMENT_SEEDS):
            raise EvidenceIntegrityError(f"Incomplete Development records for {run_id}.")
        if tuple(record.get("seed") for record in records if isinstance(record, Mapping)) != DEVELOPMENT_SEEDS:
            raise EvidenceIntegrityError(f"Wrong Development seed bank for {run_id}.")
        if any(not isinstance(record, Mapping) or type(record.get("success")) is not bool for record in records):
            raise EvidenceIntegrityError(f"Malformed Development record for {run_id}.")
        success_count = sum(record["success"] for record in records)
        rate = success_count / len(DEVELOPMENT_SEEDS)
        if mapping.get("success_count") != success_count:
            raise EvidenceIntegrityError(f"Development count mismatch for {run_id}.")
        _close(_exact_number(mapping, "success_rate"), rate, "Development rate")
        _close(
            _exact_number(mapping, "mean_reward"),
            math.fsum(float(record["total_reward"]) for record in records) / len(records),
            "Development mean reward",
        )
        _close(
            _exact_number(mapping, "mean_episode_length"),
            math.fsum(float(record["episode_length"]) for record in records) / len(records),
            "Development mean episode length",
        )
        rates.append(rate)
    area = math.fsum(
        (rates[index] + rates[index + 1])
        * 0.5
        * DEVELOPMENT_INTERVAL
        for index in range(len(rates) - 1)
    )
    def sustained(threshold: float) -> int | None:
        for index in range(len(rates) - 1):
            if rates[index] >= threshold and rates[index + 1] >= threshold:
                return expected_steps[index + 1]
        return None
    return {
        "auc": area / budget,
        "t50": sustained(0.5),
        "t80": sustained(0.8),
        "final": rates[-1],
    }


def _verify_validation_report(
    path: Path,
    *,
    checkpoint_path: Path,
    checkpoint_hash: str,
    task_id: str,
    task_contract_digest: str,
) -> float:
    if not path.is_file():
        raise EvidenceIntegrityError(f"Missing Validation report: {path}")
    normalized = load_normalized_evaluation_report(path)
    payload = _json_mapping(path, "Validation report")
    identity = _mapping(payload.get("evaluation_identity"), "Validation identity")
    conditions = _mapping(payload.get("evaluation_conditions"), "Validation conditions")
    distribution = _mapping(conditions.get("distribution"), "Validation distribution")
    provenance = _mapping(payload.get("provenance"), "Validation provenance")
    checkpoint = _mapping(provenance.get("checkpoint"), "Validation checkpoint")
    aggregate = _mapping(payload.get("aggregate_metrics"), "Validation aggregate")
    records = payload.get("episode_records")
    if (
        normalized.protocol_version != "evaluation-v2"
        or normalized.task_id != task_id
        or identity.get("evidence_role") != "validation"
        or identity.get("checkpoint_sha256") != checkpoint_hash
        or identity.get("task_contract_digest") != task_contract_digest
        or conditions.get("deterministic_actions") is not True
        or tuple(distribution.get("seeds", ())) != VALIDATION_SEEDS
        or distribution.get("episode_count") != len(VALIDATION_SEEDS)
        or checkpoint.get("sha256") != checkpoint_hash
        or _inside_project(Path(_exact_string(checkpoint, "path"))) != checkpoint_path
        or aggregate.get("report_complete") is not True
        or aggregate.get("error_count") != 0
        or not isinstance(records, list)
        or len(records) != len(VALIDATION_SEEDS)
    ):
        raise EvidenceIntegrityError(f"Validation contract mismatch: {path}")
    if tuple(record.get("seed") for record in records if isinstance(record, Mapping)) != VALIDATION_SEEDS:
        raise EvidenceIntegrityError(f"Wrong Validation seed bank: {path}")
    if any(not isinstance(record, Mapping) or record.get("completed") is not True for record in records):
        raise EvidenceIntegrityError(f"Incomplete Validation record: {path}")
    success_count = sum(record.get("success") is True for record in records)
    if aggregate.get("success_count") != success_count or aggregate.get("completed_episode_count") != len(records):
        raise EvidenceIntegrityError(f"Validation aggregate count mismatch: {path}")
    rate = success_count / len(records)
    _close(_exact_number(aggregate, "success_rate"), rate, "Validation rate")
    return rate


def _verify_claimed_run_metrics(claim: Mapping[str, Any], run: VerifiedRun) -> None:
    for key, actual in run.metrics().items():
        claimed = claim.get(key)
        if actual is None:
            if claimed is not None:
                raise EvidenceIntegrityError(f"Claimed {key} mismatch for {run.run_id}.")
        elif type(actual) is int:
            if claimed != actual:
                raise EvidenceIntegrityError(f"Claimed {key} mismatch for {run.run_id}.")
        else:
            _close(float(claimed), float(actual), f"claimed {key}")
    if claim.get("task_id") != run.task_id or claim.get("seed") != run.seed or claim.get("budget") != run.budget:
        raise EvidenceIntegrityError(f"Claimed run identity mismatch for {run.run_id}.")


def _inside_project(path: Path) -> Path:
    try:
        return resolve_recorded_path(path, PROJECT_ROOT)
    except (PortablePathError, FileNotFoundError) as error:
        raise EvidenceIntegrityError(str(error)) from error


def _json_mapping(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise EvidenceIntegrityError(f"Missing {label}: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise EvidenceIntegrityError(f"Unreadable {label}: {path}: {error}") from error
    return dict(_mapping(payload, label))


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise EvidenceIntegrityError(f"{label} must be an object.")
    return value


def _exact_string(value: Mapping[str, Any], key: str) -> str:
    result = value.get(key)
    if not isinstance(result, str) or not result:
        raise EvidenceIntegrityError(f"{key} must be a non-empty string.")
    return result


def _exact_int(value: Mapping[str, Any], key: str) -> int:
    result = value.get(key)
    if type(result) is not int:
        raise EvidenceIntegrityError(f"{key} must be an integer.")
    return result


def _exact_number(value: Mapping[str, Any], key: str) -> float:
    result = value.get(key)
    if type(result) not in (int, float) or not math.isfinite(float(result)):
        raise EvidenceIntegrityError(f"{key} must be a finite number.")
    return float(result)


def _close(left: float, right: float, label: str) -> None:
    if not math.isclose(left, right, rel_tol=0.0, abs_tol=1e-12):
        raise EvidenceIntegrityError(f"{label} mismatch: {left!r} != {right!r}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
