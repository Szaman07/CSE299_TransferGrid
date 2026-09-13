"""Lightweight display-only aliases shared by both UI modes."""

from __future__ import annotations

from typing import Mapping


TASK_CODES: Mapping[str, str] = {
    "goal_navigation-v1": "GN",
    "key_pickup-v1": "KP",
    "key_pickup-v2": "KP2",
    "door_key-v1": "DK",
    "lava_navigation-v1": "LN",
    "lava_door_key-v1": "LK1",
    "lava_door_key-v2": "LK2",
    "official-empty-random-6x6-v1": "OER6",
    "official-door-key-6x6-v1": "ODK6",
    "official-lava-crossing-s9n1-v1": "OLC9",
}


def task_code(task_id: object) -> str:
    try:
        return TASK_CODES[str(task_id)]
    except KeyError as error:
        raise ValueError(f"Unknown display-alias task identity: {task_id!r}") from error


def artifact_route_label(
    target_task_id: object,
    source_task_id: object | None = None,
) -> str:
    """Keep compact artifact labels explicit about source and target identity."""
    target = task_code(target_task_id)
    if source_task_id is None:
        return f"SCRATCH → {target}"
    return f"{task_code(source_task_id)} → {target}"


def artifact_study_label(
    metadata: Mapping[str, object] | None,
    *,
    role: str | None = None,
) -> str:
    """Return the governed study condition without inventing missing metadata."""
    values = metadata if isinstance(metadata, Mapping) else {}
    binding_value = values.get("study_binding")
    binding = binding_value if isinstance(binding_value, Mapping) else {}
    artifact_role = str(
        binding.get("artifact_role") or values.get("artifact_role") or ""
    ).lower()
    condition = binding.get("condition")
    route = binding.get("route")
    if isinstance(condition, str) and condition:
        condition_label = condition
    else:
        condition_label = "SCRATCH" if role == "scratch" else "TRANSFER"
    if isinstance(route, str) and route and route.lower() not in {"shared", "none"}:
        return f"{route.upper()} · {condition_label}"
    return condition_label


def artifact_selection_label(run: Mapping[str, object]) -> str:
    """Format one Evaluate/Replay checkpoint with study, route, and identity."""
    study = str(run.get("study_label") or run.get("role") or "ARTIFACT")
    route = artifact_route_label(run["task_id"], run.get("source_task_id"))
    stage = "INIT 0%" if run.get("checkpoint_stage") == "initialization" else "FINAL"
    return (
        f"{study} · {route} · {stage} · "
        f"{float(run['validation_success_rate']):.0%} prior Validation · "
        f"seed {run['seed']} · SHA {str(run['checkpoint_sha256'])[:8]}"
    )


def scratch_alias(task_id: object, seed: int) -> str:
    return f"S-{task_code(task_id)}-s{int(seed)}"


def transfer_alias(
    pair: int,
    source_task_id: object,
    target_task_id: object,
    seed: int,
) -> str:
    return (
        f"P{int(pair):02d}-T-{task_code(source_task_id)}-"
        f"{task_code(target_task_id)}-s{int(seed)}"
    )


def compact_count(value: int) -> str:
    if value >= 1_000_000 and value % 1_000_000 == 0:
        return f"{value // 1_000_000}M"
    if value >= 1_000 and value % 1_000 == 0:
        return f"{value // 1_000}k"
    return f"{value:,}"
