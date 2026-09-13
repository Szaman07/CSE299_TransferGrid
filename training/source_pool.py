"""Immutable source-pool selection for governed batch transfer studies.

Generic discovery answers which checkpoints are technically eligible today.
A frozen study source pool answers which exact checkpoint a declared condition
must use.  Those are deliberately separate questions.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

from training.checkpoints import checkpoint_sha256, load_checkpoint_metadata
from training.transfer_governance import assess_source_competence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_POOL_PATH = (
    REPOSITORY_ROOT / "configs" / "weekly_transfer_source_pool_v1.json"
)


class SourcePoolIntegrityError(RuntimeError):
    """Raised when a declared study source is missing or has changed."""


@dataclass(frozen=True, slots=True)
class SourcePoolEntry:
    pool_id: str
    task_id: str
    source_seed: int
    run_id: str
    checkpoint_path: Path
    checkpoint_sha256: str
    pool_role: str

    @property
    def key(self) -> tuple[str, str, int]:
        return (self.pool_id, self.task_id, self.source_seed)


def load_source_pool(
    path: str | Path = DEFAULT_SOURCE_POOL_PATH,
) -> dict[tuple[str, str, int], SourcePoolEntry]:
    """Load a frozen source pool and reject ambiguous identities."""
    pool_path = Path(path).resolve()
    payload = json.loads(pool_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or payload.get("schema_version") != 1:
        raise SourcePoolIntegrityError("Unsupported source-pool schema.")
    pool_id = payload.get("pool_id")
    entries = payload.get("entries")
    if not isinstance(pool_id, str) or not pool_id:
        raise SourcePoolIntegrityError("Source pool is missing pool_id.")
    if not isinstance(entries, list):
        raise SourcePoolIntegrityError("Source pool entries must be a list.")

    result: dict[tuple[str, str, int], SourcePoolEntry] = {}
    for raw in entries:
        if not isinstance(raw, Mapping):
            raise SourcePoolIntegrityError("Source pool entry must be a mapping.")
        relative_path = raw.get("checkpoint_path")
        if not isinstance(relative_path, str):
            raise SourcePoolIntegrityError("Source entry is missing checkpoint_path.")
        entry = SourcePoolEntry(
            pool_id=pool_id,
            task_id=str(raw.get("task_id", "")),
            source_seed=int(raw.get("source_seed")),
            run_id=str(raw.get("run_id", "")),
            checkpoint_path=(REPOSITORY_ROOT / relative_path).resolve(),
            checkpoint_sha256=str(raw.get("checkpoint_sha256", "")),
            pool_role=str(raw.get("pool_role", "")),
        )
        if entry.key in result:
            raise SourcePoolIntegrityError(
                f"Duplicate source-pool identity: {entry.key!r}"
            )
        result[entry.key] = entry
    return result


def require_source_pool_entry(
    task_id: str,
    *,
    source_seed: int = 19,
    pool_id: str = "weekly-transfer-source-pool-v1",
    expected_sha256: str | None = None,
    pool_path: str | Path = DEFAULT_SOURCE_POOL_PATH,
    verify_competence: bool = True,
) -> SourcePoolEntry:
    """Return one exact source or fail before a training request is written."""
    entries = load_source_pool(pool_path)
    key = (pool_id, task_id, int(source_seed))
    entry = entries.get(key)
    if entry is None:
        raise SourcePoolIntegrityError(f"No declared source for {key!r}.")
    if expected_sha256 is not None and expected_sha256 != entry.checkpoint_sha256:
        raise SourcePoolIntegrityError(
            "Caller/source-pool SHA mismatch: "
            f"expected={expected_sha256} pool={entry.checkpoint_sha256}"
        )
    if not entry.checkpoint_path.is_file():
        raise SourcePoolIntegrityError(
            f"Declared source checkpoint is missing: {entry.checkpoint_path}"
        )

    actual_sha = checkpoint_sha256(entry.checkpoint_path)
    if actual_sha != entry.checkpoint_sha256:
        raise SourcePoolIntegrityError(
            "Declared source checkpoint SHA mismatch: "
            f"expected={entry.checkpoint_sha256} actual={actual_sha}"
        )
    metadata = load_checkpoint_metadata(entry.checkpoint_path)
    environment = metadata.get("environment")
    actual_task = (
        environment.get("task_id") if isinstance(environment, Mapping) else None
    )
    if metadata.get("run_id") != entry.run_id or actual_task != entry.task_id:
        raise SourcePoolIntegrityError(
            "Declared source metadata identity mismatch: "
            f"expected=({entry.run_id}, {entry.task_id}) "
            f"actual=({metadata.get('run_id')}, {actual_task})"
        )
    if metadata.get("checkpoint_sha256") != entry.checkpoint_sha256:
        raise SourcePoolIntegrityError(
            "Declared source metadata SHA does not match the source pool."
        )
    if verify_competence:
        competence = assess_source_competence(entry.checkpoint_path)
        if not competence.competent:
            raise SourcePoolIntegrityError(
                "Declared source no longer passes competence governance: "
                + ", ".join(competence.failure_codes)
            )
    return entry


def build_declared_full_policy_initialization(
    task_id: str,
    *,
    source_seed: int = 19,
    pool_id: str = "weekly-transfer-source-pool-v1",
    expected_sha256: str | None = None,
    pool_path: str | Path = DEFAULT_SOURCE_POOL_PATH,
):
    """Build initialization only after validating the external pool identity."""
    from training.initialization import build_full_policy_initialization

    entry = require_source_pool_entry(
        task_id,
        source_seed=source_seed,
        pool_id=pool_id,
        expected_sha256=expected_sha256,
        pool_path=pool_path,
    )
    initialization = build_full_policy_initialization(entry.checkpoint_path)
    if initialization.expected_source_sha256 != entry.checkpoint_sha256:
        raise SourcePoolIntegrityError(
            "Initialization SHA differs from the frozen source-pool SHA."
        )
    return initialization


def build_declared_selective_policy_initialization(
    task_id: str,
    *,
    method: str,
    source_seed: int = 19,
    pool_id: str = "weekly-transfer-source-pool-v1",
    expected_sha256: str | None = None,
    pool_path: str | Path = DEFAULT_SOURCE_POOL_PATH,
):
    """Build one selective request through the same immutable source pool."""
    from training.initialization import build_selective_policy_initialization

    entry = require_source_pool_entry(
        task_id,
        source_seed=source_seed,
        pool_id=pool_id,
        expected_sha256=expected_sha256,
        pool_path=pool_path,
    )
    initialization = build_selective_policy_initialization(
        entry.checkpoint_path,
        method=method,
        study_relation="primary",
    )
    if initialization.expected_source_sha256 != entry.checkpoint_sha256:
        raise SourcePoolIntegrityError(
            "Selective initialization SHA differs from the source-pool SHA."
        )
    return initialization
