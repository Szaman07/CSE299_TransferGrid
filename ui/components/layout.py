"""Shared session-state helpers for launcher pages."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from environments.catalog import LEGACY_OBJECT_INTERACTION_TASK_ID
from environments.factory import (
    DEFAULT_GRID_SIZE,
    MEDIUM_DIFFICULTY,
    OBJECT_INTERACTION_CAPABILITY,
)
from environments.generation import DEFAULT_SEED
from training.evaluation import (
    DEFAULT_EVALUATION_DISTRIBUTION_NAME,
    format_evaluation_seeds,
)
from training.trainer import DEFAULT_TIMESTEPS
from ui.constants import DEFAULT_ACTIVE_VIEW, LEGACY_VIEW_MIGRATION

LEGACY_GENERATION_STATE_KEYS = (
    "last_generated_config",
    "last_rgb_observation",
    "last_observation",
)


def initialize_home_state(workspace_path: Path) -> None:
    """Initialize the state shared by the reusable launcher shell."""
    _clear_legacy_generation_state()
    st.session_state.setdefault("active_view", DEFAULT_ACTIVE_VIEW)
    st.session_state["active_view"] = LEGACY_VIEW_MIGRATION.get(
        st.session_state["active_view"], st.session_state["active_view"]
    )
    st.session_state.setdefault("console_logs", ["TransferGrid ready."])
    st.session_state.setdefault("workspace_path", str(workspace_path))
    st.session_state.setdefault(
        "generate_config",
        {
            "capability": OBJECT_INTERACTION_CAPABILITY,
            "task_id": "door_key-v1",
            "seed": DEFAULT_SEED,
            "grid_size": DEFAULT_GRID_SIZE,
            "difficulty": MEDIUM_DIFFICULTY,
        },
    )
    st.session_state.setdefault("last_generate_request", None)
    st.session_state.setdefault("generation_error", None)
    st.session_state.setdefault(
        "training_config",
        {
            "total_timesteps": DEFAULT_TIMESTEPS,
            "task_id": "door_key-v1",
            "grid_size": DEFAULT_GRID_SIZE,
            "seed": DEFAULT_SEED,
            "training_mode": "procedural",
            "layout_seed": DEFAULT_SEED,
        },
    )
    st.session_state.setdefault("training_job", None)
    st.session_state.setdefault("training_status", "idle")
    st.session_state.setdefault("training_summary", None)
    st.session_state.setdefault("training_error", None)
    st.session_state.setdefault(
        "evaluation_config",
        {
            "checkpoint_path": "",
            "distribution_name": DEFAULT_EVALUATION_DISTRIBUTION_NAME,
            "seed_text": format_evaluation_seeds(),
            "evidence_role": "validation",
            "evaluation_family": "normal",
            "checkpoint_stage": "final",
        },
    )
    st.session_state.setdefault("evaluation_status", "idle")
    st.session_state.setdefault("evaluation_report", None)
    st.session_state.setdefault("evaluation_report_path", None)
    st.session_state.setdefault("evaluation_error", None)
    st.session_state.setdefault("replay_selection", None)
    st.session_state.setdefault("replay_step", 0)
    st.session_state.setdefault("replay_playing", False)
    st.session_state.setdefault("replay_speed", 1)
    st.session_state.setdefault(
        "transfer_config",
        {
            "condition": "C1",
            "source_checkpoint_path": "",
            "source_task_id": "door_key-v1",
            "target_task_id": "lava_door_key-v2",
            "seed": 19,
            "total_timesteps": 3_072_000,
            "training_mode": "procedural",
            "layout_seed": 42,
        },
    )
    st.session_state.setdefault("transfer_initialization", None)
    st.session_state.setdefault("transfer_compatibility", None)
    st.session_state.setdefault("transfer_preflight_selection", None)
    st.session_state.setdefault("transfer_source_catalog", None)
    st.session_state.setdefault("transfer_job", None)
    st.session_state.setdefault("transfer_status", "idle")
    st.session_state.setdefault("transfer_summary", None)
    st.session_state.setdefault("transfer_error", None)
    _migrate_legacy_new_run_selections()


def _migrate_legacy_new_run_selections() -> None:
    """Move stale UI selections to canonical DoorKey without touching artifacts."""
    migrated: list[str] = []
    for state_key in ("generate_config", "training_config"):
        payload = st.session_state.get(state_key)
        if not isinstance(payload, dict):
            continue
        if payload.get("task_id") not in (
            None,
            LEGACY_OBJECT_INTERACTION_TASK_ID,
        ):
            continue
        normalized = dict(payload)
        normalized["task_id"] = "door_key-v1"
        normalized["grid_size"] = DEFAULT_GRID_SIZE
        st.session_state[state_key] = normalized
        migrated.append(state_key)
        if state_key == "generate_config":
            st.session_state["last_generate_request"] = None

    transfer = st.session_state.get("transfer_config")
    if (
        isinstance(transfer, dict)
        and transfer.get("target_task_id") == LEGACY_OBJECT_INTERACTION_TASK_ID
    ):
        normalized_transfer = dict(transfer)
        normalized_transfer["target_task_id"] = "door_key-v1"
        st.session_state["transfer_config"] = normalized_transfer
        st.session_state["transfer_initialization"] = None
        st.session_state["transfer_compatibility"] = None
        migrated.append("transfer_config")

    if migrated:
        st.session_state["new_run_migration_notice"] = (
            "Legacy Object Interaction is retained for historical Evaluate and "
            "Replay evidence. Its old new-run selection was moved to canonical "
            "DoorKey on Grid 7."
        )


def _clear_legacy_generation_state() -> None:
    """Release and remove heavyweight state retained by earlier UI versions."""
    previous_environment = st.session_state.pop(
        "last_generated_environment",
        None,
    )
    if previous_environment is not None:
        previous_environment.close()
    for key in LEGACY_GENERATION_STATE_KEYS:
        st.session_state.pop(key, None)


def append_console_log(message: str) -> None:
    """Append a launcher status message while retaining the latest 40 entries."""
    logs = list(st.session_state["console_logs"])
    logs.append(message)
    st.session_state["console_logs"] = logs[-40:]
