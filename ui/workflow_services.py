"""Shared presentation services for Compact and Advanced workflows.

This module is deliberately UI-thin: it owns verified discovery, aliases,
compatibility construction, and backend calls that would otherwise be copied
between presentation modes.  It never mutates certified evidence.
"""

from __future__ import annotations

from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping

import streamlit as st

from environments.catalog import (
    interactive_task_definitions,
    released_task_definitions,
    require_released_task,
    task_display_name,
)
from environments.factory import EnvironmentSpec
from training.benchmark_contracts import (
    LAVA_DOOR_KEY_TARGET_TIMESTEPS,
    PILOT_MAX_TARGET_TIMESTEPS,
    SOURCE_TIMESTEPS,
)
from training.checkpoints import load_checkpoint_metadata
from training.evaluation_v2 import (
    EvaluationV2Config,
    run_evaluation_v2,
    save_evaluation_v2_report,
)
from training.initialization import (
    build_full_policy_initialization,
    build_selective_policy_initialization,
)
from training.initialization_methods import (
    ACTOR_BODY_FINETUNE,
    ACTOR_BODY_FRESH_HEAD_COPIED_CRITIC,
    FULL_ACTOR_FRESH_CRITIC,
)
from training.portable_paths import repository_relative, resolve_recorded_path
from training.replay import (
    ReplayRequest,
    ReplayReportReference,
    ReplayTrace,
    discover_replay_reports,
    load_replay_report_reference,
    replay_evaluated_episode,
)
from training.source_pool import load_source_pool
from training.transfer_governance import (
    assess_source_competence,
    discover_governed_transfer_sources,
)
from training.trainer import TrainingConfig, check_transfer_preflight
from ui.display_aliases import (
    artifact_study_label,
    scratch_alias,
    task_code,
    transfer_alias,
)
CERTIFIED_SNAPSHOT_KEY = "compact_certified_snapshot_v2"
LOCAL_SNAPSHOT_KEY = "compact_local_snapshot_v3"
LOCAL_ARTIFACT_INDEX = (
    Path(__file__).resolve().parents[1]
    / "reports"
    / "ui"
    / "compact_artifact_index_v1.json"
)
SOURCE_CATALOG_KEY = "transfer_source_catalog"
_SOURCE_VALIDATION_RATES: dict[str, float] = {}
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_SOURCE_POOL = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "official_minigrid_replication_source_pool_v2.json"
)
SELECTIVE_TRANSFER_CONDITIONS: dict[str, dict[str, str | None]] = {
    "C1": {
        "label": "C1 · Full policy warm start",
        "method": None,
        "summary": "copied actor body, action head, and critic",
    },
    "C2a": {
        "label": "C2a · Fresh critic",
        "method": FULL_ACTOR_FRESH_CRITIC,
        "summary": "copied actor body and action head; fresh critic",
    },
    "C2": {
        "label": "C2 · Fresh action head + critic",
        "method": ACTOR_BODY_FINETUNE,
        "summary": "copied actor body; fresh action head and critic",
    },
    "C2b": {
        "label": "C2b · Fresh action head, copied critic",
        "method": ACTOR_BODY_FRESH_HEAD_COPIED_CRITIC,
        "summary": "copied actor body and critic; fresh action head",
    },
}


def certified_snapshot() -> dict[str, Any]:
    """Verify once per Streamlit session and retain only serializable metadata."""
    cached = st.session_state.get(CERTIFIED_SNAPSHOT_KEY)
    if isinstance(cached, dict):
        return cached

    started = perf_counter()
    model = load_analysis_read_model()
    alias_by_run: dict[str, str] = {}
    pair_by_route: dict[tuple[str, str], int] = {}
    for cell in model.cells:
        alias_by_run[cell.transfer.run_id] = transfer_alias(
            cell.pair,
            cell.source_task_id,
            cell.target_task_id,
            cell.seed,
        )
        alias_by_run[cell.scratch.run_id] = scratch_alias(
            cell.scratch.task_id,
            cell.scratch.seed,
        )
        pair_by_route[(cell.source_task_id, cell.target_task_id)] = cell.pair

    runs: list[dict[str, Any]] = []
    for run in model.runs:
        alias = alias_by_run[run.run_id]
        latest_report = max(
            run.validation_reports,
            key=lambda report: (report.created_at, report.report_id),
        )
        episodes: list[dict[str, Any]] = []
        for episode_index, episode in enumerate(latest_report.episodes):
            if not episode.completed:
                continue
            outcome = "SUCCESS" if episode.success else (
                str(episode.failure_reason).upper()
                if episode.failure_reason else
                "TIMEOUT" if episode.truncated else "FAILED"
            )
            group = (
                "SUCCESS SEEDS" if episode.success else
                "LAVA FAILURE SEEDS" if outcome == "LAVA" else
                "TIMEOUT SEEDS" if episode.truncated else
                "OTHER FAILURE SEEDS"
            )
            episodes.append({
                "alias": alias,
                "label": f"{group} · {episode.seed}",
                "group": group,
                "report_path": str(latest_report.path),
                "task_id": run.task_id,
                "episode_index": episode_index,
                "seed": episode.seed,
                "success": episode.success,
                "outcome": outcome,
            })
        group_order = {
            "SUCCESS SEEDS": 0,
            "LAVA FAILURE SEEDS": 1,
            "TIMEOUT SEEDS": 2,
            "OTHER FAILURE SEEDS": 3,
        }
        episodes.sort(key=lambda item: (group_order[item["group"]], item["seed"]))
        try:
            checkpoint_metadata = load_checkpoint_metadata(run.checkpoint_path)
        except (OSError, TypeError, ValueError):
            checkpoint_metadata = {}
        role = "scratch" if run.initialization_method == "scratch" else "transfer"
        runs.append({
            "alias": alias,
            "run_id": run.run_id,
            "role": role,
            "study_label": artifact_study_label(checkpoint_metadata, role=role),
            "checkpoint_stage": "final",
            "task_id": run.task_id,
            "task_code": task_code(run.task_id),
            "source_task_id": run.source_task_id,
            "seed": run.seed,
            "budget": run.budget,
            "checkpoint_path": str(run.checkpoint_path),
            "checkpoint_sha256": run.checkpoint_sha256,
            "finished_at": run.finished_at,
            "validation_report_path": str(latest_report.path),
            "validation_report_id": latest_report.report_id,
            "validation_success_rate": latest_report.success_rate,
            "episodes": episodes,
        })

    recommended = model.recommended
    latest_run = recommended.latest_completed_checkpoint
    latest_alias = alias_by_run[latest_run.run_id]
    replay_report = recommended.latest_replayable_report
    replay_run = next(
        run for run in runs
        if Path(run["validation_report_path"]).resolve() == replay_report.path.resolve()
    )
    replay_choices = list(replay_run["episodes"])
    recommended_replay = replay_choices[0]
    source = recommended.transfer.source
    source_pool = tuple(
        {
            "pool_id": entry.pool_id,
            "task_id": entry.task_id,
            "source_seed": entry.source_seed,
            "run_id": entry.run_id,
            "checkpoint_path": str(entry.checkpoint_path),
            "checkpoint_sha256": entry.checkpoint_sha256,
        }
        for entry in sorted(
            load_source_pool().values(),
            key=lambda item: (item.task_id, item.source_seed),
        )
    )
    snapshot = {
        "verified_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "verification_seconds": perf_counter() - started,
        "dataset_sha256": model.dataset_sha256,
        "cell_count": len(model.cells),
        "run_count": len(model.runs),
        "dependency_count": len(model.dependencies),
        "runs": runs,
        "replay_choices": replay_choices,
        "pair_by_route": {
            f"{source_id}|{target_id}": pair
            for (source_id, target_id), pair in pair_by_route.items()
        },
        "source_pool": source_pool,
        "recommended_training": {
            "task_id": recommended.training.task_id,
            "grid_size": recommended.training.grid_size,
            "seed": recommended.training.seed,
            "total_timesteps": recommended.training.total_timesteps,
            "training_mode": recommended.training.training_mode,
        },
        "recommended_evaluation_alias": latest_alias,
        "recommended_replay_alias": replay_run["alias"],
        "recommended_replay_label": recommended_replay["label"],
        "recommended_transfer": {
            "source_checkpoint_path": str(source.checkpoint_path),
            "source_checkpoint_sha256": source.checkpoint_sha256,
            "source_task_id": source.task_id,
            "source_seed": source.source_seed,
            "target_task_id": recommended.transfer.target_task_id,
            "target_seed": recommended.transfer.target_seed,
            "target_timesteps": recommended.transfer.target_timesteps,
        },
    }
    st.session_state[CERTIFIED_SNAPSHOT_KEY] = snapshot
    return snapshot


def local_artifact_snapshot(*, force_refresh: bool = False) -> dict[str, Any]:
    """Build a usable snapshot from individually verified local artifacts.

    This is intentionally not a replacement certification for the historical
    aggregate dataset.  Every included checkpoint must pass its metadata/SHA
    binding, and Replay performs the report/checkpoint SHA check again when an
    episode is opened.
    """
    cached = st.session_state.get(LOCAL_SNAPSHOT_KEY)
    if isinstance(cached, dict):
        return cached
    if not force_refresh:
        indexed = _load_local_artifact_index()
        if indexed is not None:
            st.session_state[LOCAL_SNAPSHOT_KEY] = indexed
            return indexed

    runs: list[dict[str, Any]] = []
    seen_hashes: set[str] = set()
    for report in _discover_all_replayable_reports():
        if not report.distribution_name.lower().endswith("validation"):
            continue
        expected_hash = report.checkpoint_sha256
        if not expected_hash or expected_hash in seen_hashes:
            continue
        try:
            metadata = load_checkpoint_metadata(report.checkpoint_path)
            task_alias = task_code(report.task_id)
        except (OSError, TypeError, ValueError):
            continue
        if metadata.get("checkpoint_sha256") != expected_hash:
            continue
        completed = tuple(episode for episode in report.episodes if episode.completed)
        if not completed:
            continue
        seed = report.training_seed if report.training_seed is not None else 0
        stage = "I" if report.checkpoint_kind == "transfer-initialization" else (
            "T" if report.checkpoint_kind == "transfer-final" else "S"
        )
        initialization = metadata.get("initialization")
        source_task_id = (
            initialization.get("source_task_id")
            if isinstance(initialization, dict)
            else None
        )
        role = "transfer" if stage in {"T", "I"} else "scratch"
        alias = f"{task_alias}-{stage}-s{seed}-{expected_hash[:6]}"
        episodes: list[dict[str, Any]] = []
        for episode in completed:
            outcome = "SUCCESS" if episode.success else "FAILED"
            group = "SUCCESS SEEDS" if episode.success else "OTHER FAILURE SEEDS"
            episodes.append({
                "alias": alias,
                "label": f"{group} · {episode.seed}",
                "group": group,
                "report_path": str(report.path),
                "task_id": report.task_id,
                "episode_index": episode.index,
                "seed": episode.seed,
                "success": bool(episode.success),
                "outcome": outcome,
            })
        episodes.sort(key=lambda item: (item["group"] != "SUCCESS SEEDS", item["seed"]))
        runs.append({
            "alias": alias,
            "run_id": report.experiment_run_id or alias,
            "role": role,
            "study_label": artifact_study_label(metadata, role=role),
            "checkpoint_stage": (
                "initialization" if stage == "I" else "final"
            ),
            "task_id": report.task_id,
            "task_code": task_alias,
            "source_task_id": source_task_id,
            "seed": seed,
            "budget": report.requested_timesteps or 0,
            "checkpoint_path": str(report.checkpoint_path),
            "checkpoint_sha256": expected_hash,
            "finished_at": report.created_at,
            "validation_report_path": str(report.path),
            "validation_report_id": report.report_id,
            "validation_success_rate": float(report.success_rate or 0.0),
            "episodes": episodes,
        })
        seen_hashes.add(expected_hash)

    if not runs:
        raise RuntimeError("No independently verified local checkpoint/report pair is available.")
    runs.sort(
        key=lambda run: (
            {"C0": 0, "C1": 1, "C2a": 2, "C2": 3, "C2b": 4}.get(
                str(run["study_label"]).split(" · ")[-1], 5
            ),
            str(run["study_label"]),
            run["task_code"],
            -float(run["validation_success_rate"]),
            run["alias"],
        )
    )
    replay_run = next(run for run in runs if run["episodes"])
    condition_coverage = {
        str(run["study_label"]).split(" · ")[-1]
        for run in runs
        if str(run["study_label"]).split(" · ")[-1]
        in {"C0", "C1", "C2a", "C2", "C2b"}
    }
    snapshot = {
        "scope": "individually-verified-local-artifacts",
        "runs": runs,
        "run_count": len(runs),
        "replay_choices": list(replay_run["episodes"]),
        "recommended_evaluation_alias": runs[0]["alias"],
        "recommended_replay_alias": replay_run["alias"],
        "recommended_replay_label": replay_run["episodes"][0]["label"],
        "condition_coverage": tuple(
            condition for condition in ("C0", "C1", "C2a", "C2", "C2b")
            if condition in condition_coverage
        ),
    }
    st.session_state[LOCAL_SNAPSHOT_KEY] = snapshot
    return snapshot


def write_local_artifact_index(snapshot: Mapping[str, Any]) -> Path:
    """Write a portable UI index; actions still reverify the selected artifact."""
    import json

    payload = dict(snapshot)
    payload["schema_version"] = 1
    payload["scope"] = "indexed-local-artifacts"
    payload["selection_verification"] = "on-demand"
    for run in payload.get("runs", []):
        run["checkpoint_path"] = repository_relative(
            run["checkpoint_path"], PROJECT_ROOT
        )
        run["validation_report_path"] = repository_relative(
            run["validation_report_path"], PROJECT_ROOT
        )
        for episode in run.get("episodes", []):
            episode["report_path"] = repository_relative(
                episode["report_path"], PROJECT_ROOT
            )
    for choice in payload.get("replay_choices", []):
        choice["report_path"] = repository_relative(
            choice["report_path"], PROJECT_ROOT
        )
    LOCAL_ARTIFACT_INDEX.parent.mkdir(parents=True, exist_ok=True)
    LOCAL_ARTIFACT_INDEX.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return LOCAL_ARTIFACT_INDEX


def _load_local_artifact_index() -> dict[str, Any] | None:
    if not LOCAL_ARTIFACT_INDEX.is_file():
        return None
    import json

    try:
        payload = json.loads(LOCAL_ARTIFACT_INDEX.read_text(encoding="utf-8"))
        if payload.get("schema_version") != 1 or not payload.get("runs"):
            return None
        for run in payload["runs"]:
            checkpoint = resolve_recorded_path(run["checkpoint_path"], PROJECT_ROOT)
            report = resolve_recorded_path(
                run["validation_report_path"], PROJECT_ROOT
            )
            run["checkpoint_path"] = str(checkpoint)
            run["validation_report_path"] = str(report)
            for episode in run.get("episodes", []):
                episode["report_path"] = str(resolve_recorded_path(
                    episode["report_path"], PROJECT_ROOT
                ))
        for choice in payload.get("replay_choices", []):
            choice["report_path"] = str(resolve_recorded_path(
                choice["report_path"], PROJECT_ROOT
            ))
        return payload
    except (OSError, KeyError, TypeError, ValueError):
        return None


def _discover_all_replayable_reports() -> tuple[ReplayReportReference, ...]:
    """Include cohort-local V2 reports while preserving replay validation gates."""
    by_path = {
        report.path.resolve(): report for report in discover_replay_reports()
    }
    reports_root = PROJECT_ROOT / "reports"
    if reports_root.is_dir():
        for path in reports_root.rglob("evaluation-v2_*.json"):
            if "evaluations_v2" not in {part.lower() for part in path.parts}:
                continue
            resolved = path.resolve()
            if resolved in by_path:
                continue
            try:
                report = load_replay_report_reference(resolved)
            except (OSError, KeyError, TypeError, ValueError):
                continue
            if report.checkpoint_path.is_file() and any(
                episode.completed for episode in report.episodes
            ):
                by_path[resolved] = report
    return tuple(sorted(
        by_path.values(), key=lambda report: report.created_at, reverse=True
    ))


def governed_source_catalog() -> dict[str, tuple[Path, ...]]:
    """Discover governed sources once per UI session and retain only paths."""
    cached = st.session_state.get(SOURCE_CATALOG_KEY)
    if isinstance(cached, dict):
        return {
            str(task_id): tuple(Path(path) for path in paths)
            for task_id, paths in cached.items()
        }
    released_ids = {
        definition.task_id for definition in released_task_definitions()
    }
    grouped: dict[str, list[Path]] = {}
    discovery = discover_governed_transfer_sources()
    for source in discovery.included:
        if source.task_id in released_ids:
            grouped.setdefault(source.task_id, []).append(source.checkpoint_path)
            rate = source.competence.validation_success_rate
            if rate is not None:
                _SOURCE_VALIDATION_RATES[
                    str(source.checkpoint_path.resolve())
                ] = float(rate)
    if OFFICIAL_SOURCE_POOL.is_file():
        for entry in load_source_pool(OFFICIAL_SOURCE_POOL).values():
            grouped.setdefault(entry.task_id, []).append(entry.checkpoint_path)
    catalog = {
        task_id: tuple(sorted(set(paths), key=lambda path: str(path).lower()))
        for task_id, paths in grouped.items()
        if paths
    }
    st.session_state[SOURCE_CATALOG_KEY] = {
        task_id: tuple(str(path) for path in paths)
        for task_id, paths in catalog.items()
    }
    return catalog


def transfer_pair_catalog(
    source_catalog: Mapping[str, tuple[Path, ...]] | None = None,
) -> tuple[dict[str, str], ...]:
    """Return every available governed source-task/interactive-target pair.

    Pair discovery is intentionally independent of initialization condition:
    C1, C2a, C2, and C2b all use the same source/target menu, then the strict
    compatibility preflight decides whether the selected checkpoint can be
    copied safely.
    """
    sources = source_catalog or governed_source_catalog()
    target_ids = tuple(
        definition.task_id for definition in interactive_task_definitions()
    )
    pairs = []
    for source_task_id in sorted(sources, key=task_display_name):
        for target_task_id in sorted(target_ids, key=task_display_name):
            relation = study_relation(source_task_id, target_task_id)
            pairs.append({
                "id": transfer_pair_id(source_task_id, target_task_id),
                "source_task_id": source_task_id,
                "target_task_id": target_task_id,
                "relation": relation,
                "label": transfer_pair_label(
                    source_task_id,
                    target_task_id,
                    relation=relation,
                ),
            })
    return tuple(pairs)


def transfer_pair_id(source_task_id: str, target_task_id: str) -> str:
    """Return the stable UI identity for one source-task/target-task pair."""
    return f"{source_task_id}|{target_task_id}"


def split_transfer_pair_id(pair_id: str) -> tuple[str, str]:
    """Parse a pair identity and reject malformed widget/session values."""
    parts = str(pair_id).split("|")
    if len(parts) != 2 or not all(parts):
        raise ValueError(f"Invalid transfer pair identity: {pair_id!r}.")
    return parts[0], parts[1]


def transfer_pair_label(
    source_task_id: str,
    target_task_id: str,
    *,
    relation: str | None = None,
) -> str:
    """Format a compact pair label; scientific relation stays in preflight."""
    return f"{task_code(source_task_id)} → {task_code(target_task_id)}"


def preferred_source_checkpoint(
    source_task_id: str,
    *,
    preferred: str | Path | None = None,
    source_catalog: Mapping[str, tuple[Path, ...]] | None = None,
) -> Path:
    """Choose a stable default while allowing explicit checkpoint override."""
    catalog = source_catalog or governed_source_catalog()
    candidates = tuple(catalog.get(source_task_id, ()))
    if not candidates:
        raise ValueError(
            f"No governed checkpoint is available for source task {source_task_id!r}."
        )
    if preferred is not None:
        preferred_path = Path(preferred).resolve()
        for candidate in candidates:
            if candidate.resolve() == preferred_path:
                return candidate

    def rank(path: Path) -> tuple[str, int, str]:
        metadata = load_checkpoint_metadata(path)
        return (
            str(metadata.get("finished_at", "")),
            int(metadata.get("requested_timesteps", 0)),
            str(path).lower(),
        )

    return max(candidates, key=rank)


def source_checkpoint_label(path: Path) -> str:
    """Format governed sources with the same compact task codes as the UI."""
    metadata = load_checkpoint_metadata(path)
    environment = metadata.get("environment")
    task_id = environment.get("task_id") if isinstance(environment, dict) else None
    try:
        source_name = task_code(task_id)
    except ValueError:
        source_name = str(task_id or "Unknown Source")
    steps = int(metadata.get("requested_timesteps", 0))
    rate = _source_validation_rate(str(path.resolve()))
    validation = (
        f" · {float(rate):.0%} prior Validation"
        if isinstance(rate, (int, float)) else ""
    )
    return (
        f"{source_name}{validation} · Seed {metadata.get('seed', '?')} · "
        f"{steps // 1000}k · SHA "
        f"{str(metadata.get('checkpoint_sha256', ''))[:10]}"
    )


@lru_cache(maxsize=256)
def _source_validation_rate(checkpoint_path: str) -> float | None:
    """Return the governed source rate without mutable widget-label state."""
    if checkpoint_path in _SOURCE_VALIDATION_RATES:
        return _SOURCE_VALIDATION_RATES[checkpoint_path]
    try:
        rate = assess_source_competence(
            Path(checkpoint_path)
        ).validation_success_rate
    except (OSError, TypeError, ValueError):
        return None
    if rate is None:
        return None
    _SOURCE_VALIDATION_RATES[checkpoint_path] = float(rate)
    return float(rate)


def target_budget(task_id: str) -> int:
    if task_id == "official-lava-crossing-s9n1-v1":
        return 5_017_600
    if task_id == "lava_door_key-v2":
        return PILOT_MAX_TARGET_TIMESTEPS
    if task_id == "lava_door_key-v1":
        return LAVA_DOOR_KEY_TARGET_TIMESTEPS
    return SOURCE_TIMESTEPS


def study_relation(source_task_id: str, target_task_id: str) -> str:
    if source_task_id == target_task_id:
        return "warm-start"
    if target_task_id == "lava_door_key-v2" and source_task_id in {
        "goal_navigation-v1",
        "key_pickup-v1",
        "key_pickup-v2",
        "door_key-v1",
        "lava_navigation-v1",
    }:
        return "primary"
    if source_task_id == "lava_navigation-v1" and target_task_id == "door_key-v1":
        return "control"
    return "exploratory"


def transfer_selection_key(values: Mapping[str, Any]) -> str:
    return "|".join((
        str(Path(str(values["source_checkpoint_path"])).resolve()),
        str(values["source_task_id"]),
        str(values["target_task_id"]),
        str(values.get("condition", "C1")),
    ))


def transfer_configuration(
    values: Mapping[str, Any],
    initialization: object,
) -> TrainingConfig:
    target = require_released_task(str(values["target_task_id"]))
    condition = str(values.get("condition", "C1"))
    condition_contract = SELECTIVE_TRANSFER_CONDITIONS.get(condition)
    if condition_contract is None:
        raise ValueError(f"Unsupported selective-transfer condition: {condition!r}.")
    return TrainingConfig(
        environment=EnvironmentSpec(
            task_id=target.task_id,
            grid_size=target.default_grid_size,
        ),
        total_timesteps=int(values["total_timesteps"]),
        seed=int(values["seed"]),
        training_mode=str(values["training_mode"]),
        layout_seed=values.get("layout_seed"),
        initialization=initialization,
        study_binding={
            "artifact_role": "interactive-transfer-run",
            "evidence_eligible": False,
            "condition": condition,
            "condition_method": (
                "full_policy"
                if condition_contract["method"] is None
                else condition_contract["method"]
            ),
            "launch_surface": "transfergrid-ui",
        },
    )


def run_transfer_preflight(
    values: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    relation = study_relation(
        str(values["source_task_id"]),
        str(values["target_task_id"]),
    )
    condition = str(values.get("condition", "C1"))
    contract = SELECTIVE_TRANSFER_CONDITIONS.get(condition)
    if contract is None:
        raise ValueError(f"Unsupported selective-transfer condition: {condition!r}.")
    method = contract["method"]
    initialization = (
        build_full_policy_initialization(
            values["source_checkpoint_path"],
            study_relation=relation,
        )
        if method is None
        else build_selective_policy_initialization(
            values["source_checkpoint_path"],
            method=method,
            study_relation=relation,
        )
    )
    report = check_transfer_preflight(
        transfer_configuration(values, initialization)
    )
    return initialization.to_dict(), report.to_dict()


def execute_governed_validation(checkpoint_path: Path) -> tuple[dict, Path]:
    """Run and save the frozen Validation protocol for one governed checkpoint."""
    report = run_evaluation_v2(EvaluationV2Config(checkpoint_path, "validation"))
    return report, save_evaluation_v2_report(report)


@st.cache_resource(show_spinner=False, max_entries=4)
def cached_replay_trace(report_path: str, episode_index: int) -> ReplayTrace:
    """Regenerate and cache a closed deterministic trace, never a live model/env."""
    return replay_evaluated_episode(
        ReplayRequest(report_path=Path(report_path), episode_index=episode_index)
    )
