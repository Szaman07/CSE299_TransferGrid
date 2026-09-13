"""Compact scratch-training workflow."""

from __future__ import annotations

from html import escape

import streamlit as st

from environments.catalog import interactive_task_definitions
from environments.factory import EnvironmentSpec
from environments.generation import GeneratedPreview, GenerationRequest, generate_preview
from training.jobs import TrainingJob, read_training_job, start_training_job
from training.trainer import (
    FIXED_LAYOUT_TRAINING_MODE,
    PROCEDURAL_TRAINING_MODE,
    TrainingConfig,
)
from ui.components.layout import append_console_log
from ui.display_aliases import compact_count, scratch_alias, task_code
from ui.runtime_warmup import wait_for_environment_support


@st.cache_data(show_spinner=False, max_entries=24, persist="disk")
def _cached_training_preview(
    task_id: str,
    grid_size: int,
    preview_seed: int,
) -> GeneratedPreview:
    """Load or generate one deterministic preview without retaining an env."""
    wait_for_environment_support(task_id)
    return generate_preview(
        GenerationRequest(
            environment=EnvironmentSpec(task_id=task_id, grid_size=grid_size),
            seed=preview_seed,
        )
    )


def _defaults(snapshot: dict | None) -> dict:
    # Compact Train deliberately starts on the simplest released task.  A
    # certified-analysis recommendation must not silently change the UI default.
    recommended = {
        "task_id": "door_key-v1",
        "grid_size": 7,
        "seed": 19,
        "total_timesteps": 1_024_000,
        "training_mode": PROCEDURAL_TRAINING_MODE,
    }
    return {
        "task_id": recommended["task_id"],
        "grid_size": int(recommended["grid_size"]),
        "seed": int(recommended["seed"]),
        "total_timesteps": int(recommended["total_timesteps"]),
        "training_mode": recommended["training_mode"],
        "layout_seed": 42,
    }


def render_compact_train(snapshot: dict | None = None) -> None:
    existing = st.session_state.get("compact_train_config")
    if (
        isinstance(existing, dict)
        and existing.get("task_id") == "lava_door_key-v2"
        and not st.session_state.get("compact_train_doorkey_default_v1")
    ):
        # One-time migration from the former flagship default. Explicit choices
        # made after this migration remain untouched for the rest of the session.
        st.session_state["compact_train_config"] = _defaults(snapshot)
        st.session_state["compact_train_doorkey_default_v1"] = True
    st.session_state.setdefault("compact_train_config", _defaults(snapshot))
    saved = dict(st.session_state["compact_train_config"])
    running = st.session_state["training_status"] == "running"
    st.markdown("## TRAIN")
    st.caption("Create one scratch PPO run using the certified project defaults.")

    task_id = str(saved["task_id"])
    total_timesteps = int(saved["total_timesteps"])
    seed = int(saved["seed"])
    training_mode = str(saved["training_mode"])
    layout_seed = int(saved.get("layout_seed") or 42)
    definitions = interactive_task_definitions()
    definitions_by_id = {
        definition.task_id: definition for definition in definitions
    }
    task_ids = tuple(definition.task_id for definition in definitions)
    task_id = st.selectbox(
        "TASK",
        task_ids,
        index=task_ids.index(task_id) if task_id in task_ids else 0,
        format_func=task_code,
        disabled=running,
        key="compact_train_task",
    )
    definition = definitions_by_id[task_id]
    if task_id != saved.get("task_id"):
        st.session_state.pop("compact_train_budget", None)
        total_timesteps = (
            5_017_600
            if task_id == "official-lava-crossing-s9n1-v1"
            else 1_024_000
        )
    with st.expander("Advanced options", expanded=False):
        total_timesteps = int(st.number_input(
            "BUDGET", min_value=64, step=102_400, value=total_timesteps,
            disabled=running, key="compact_train_budget",
        ))
        seed = int(st.number_input(
            "PPO SEED", min_value=0, max_value=2_147_483_647,
            value=seed, step=1, disabled=running, key="compact_train_seed",
        ))
        training_mode = st.selectbox(
            "TRAINING MODE", (PROCEDURAL_TRAINING_MODE, FIXED_LAYOUT_TRAINING_MODE),
            index=1 if training_mode == FIXED_LAYOUT_TRAINING_MODE else 0,
            format_func=lambda value: "Fixed Layout" if value == FIXED_LAYOUT_TRAINING_MODE else "Procedural",
            disabled=running, key="compact_train_mode",
        )
        layout_seed = int(st.number_input(
            "LAYOUT SEED", min_value=0, max_value=2_147_483_647,
            value=layout_seed, step=1,
            disabled=running or training_mode == PROCEDURAL_TRAINING_MODE,
            key="compact_train_layout_seed",
        ))

    values = {
        "task_id": task_id,
        "grid_size": definition.default_grid_size,
        "seed": seed,
        "total_timesteps": total_timesteps,
        "training_mode": training_mode,
        "layout_seed": layout_seed if training_mode == FIXED_LAYOUT_TRAINING_MODE else None,
    }
    st.session_state["compact_train_config"] = values
    alias = scratch_alias(task_id, seed)
    st.markdown(
        '<div class="tg-meta">'
        f'<span>RUN: {alias}</span>'
        f'<span>TASK: {task_code(task_id)}</span>'
        f'<span>GRID: {definition.default_grid_size}</span>'
        f'<span>BUDGET: {compact_count(total_timesteps)}</span>'
        f'<span>MODE: {escape(training_mode.upper())}</span>'
        '</div>', unsafe_allow_html=True,
    )
    _render_training_preview(values)
    if st.button(
        "START TRAINING", key="compact_train_start", type="primary",
        width="stretch", disabled=running,
    ):
        _start(values)
    if st.session_state["training_status"] == "running":
        _monitor()
    else:
        _render_state(alias)


def _render_training_preview(values: dict) -> None:
    """Render the layout represented by the current Compact Train choices."""
    fixed = values["training_mode"] == FIXED_LAYOUT_TRAINING_MODE
    preview_seed = int(
        values["layout_seed"] if fixed else values["seed"]
    )
    try:
        preview = _cached_training_preview(
            str(values["task_id"]),
            int(values["grid_size"]),
            preview_seed,
        )
    except Exception as error:
        st.warning(f"TRAINING LAYOUT PREVIEW UNAVAILABLE · {error}")
        return

    description = (
        f"FIXED LAYOUT · exact episode layout seed {preview_seed}"
        if fixed
        else (
            f"PROCEDURAL EXAMPLE · initial seed {preview_seed}; "
            "later episodes generate new layouts"
        )
    )
    st.caption(
        f"TRAINING LAYOUT PREVIEW · {description} · "
        f"ORACLE {preview.oracle_status.upper()} · "
        f"LAYOUT {preview.layout_digest[:10]}"
    )
    _, preview_column, _ = st.columns([1, 1, 1])
    with preview_column:
        st.image(
            preview.frame,
            caption=preview.mission,
            width=280,
        )


def _start(values: dict) -> None:
    st.session_state["training_config"] = dict(values)
    st.session_state["training_summary"] = None
    st.session_state["training_error"] = None
    try:
        job = start_training_job(TrainingConfig(
            environment=EnvironmentSpec(
                task_id=values["task_id"], grid_size=values["grid_size"]
            ),
            total_timesteps=values["total_timesteps"], seed=values["seed"],
            training_mode=values["training_mode"], layout_seed=values["layout_seed"],
        ))
    except Exception as error:
        st.session_state["training_status"] = "failed"
        st.session_state["training_error"] = str(error)
        append_console_log(f"[ERROR] Compact training failed to start: {error}")
        return
    st.session_state["training_job"] = job.to_dict()
    st.session_state["training_status"] = "running"
    append_console_log(f"[INFO] Compact scratch run {job.run_id} started")


@st.fragment(run_every=1.0)
def _monitor() -> None:
    payload = st.session_state.get("training_job")
    if not isinstance(payload, dict):
        st.session_state["training_status"] = "failed"
        st.session_state["training_error"] = "Training job metadata is missing."
        return
    state = read_training_job(TrainingJob.from_dict(payload))
    if state.status == "running":
        st.info("STATUS: TRAINING IN PROGRESS")
        return
    st.session_state["training_status"] = state.status
    st.session_state["training_summary"] = state.summary if state.status == "succeeded" else None
    st.session_state["training_error"] = state.error
    if state.status == "succeeded" and isinstance(state.summary, dict):
        evaluation = dict(st.session_state["evaluation_config"])
        evaluation["checkpoint_path"] = state.summary["checkpoint_path"]
        st.session_state["evaluation_config"] = evaluation
    st.rerun(scope="app")


def _render_state(alias: str) -> None:
    status = st.session_state["training_status"]
    if status == "idle":
        st.info(f"READY · {alias}")
    elif status == "failed":
        st.error(f"FAILED · {st.session_state['training_error']}")
    elif status == "succeeded":
        summary = st.session_state.get("training_summary")
        st.success("TRAINING COMPLETE")
        if isinstance(summary, dict):
            st.caption(
                f"RUN {summary.get('run_id', '?')} · "
                f"{int(summary.get('completed_timesteps', 0)):,} steps · "
                f"{float(summary.get('elapsed_seconds', 0)):.2f}s"
            )
