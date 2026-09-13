"""Minimal configuration form for Training v0."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import streamlit as st

from environments.catalog import (
    LEGACY_OBJECT_INTERACTION_TASK_ID,
    interactive_task_definitions,
)
from environments.factory import DEFAULT_GRID_SIZE
from environments.factory import EnvironmentSpec
from environments.generation import (
    DEFAULT_SEED,
    MAX_SEED,
    GeneratedPreview,
    GenerationRequest,
    generate_preview,
)
from training.trainer import (
    DEFAULT_TIMESTEPS,
    FIXED_LAYOUT_TRAINING_MODE,
    MIN_TIMESTEPS,
    PROCEDURAL_TRAINING_MODE,
)
from ui.display_aliases import task_code
from ui.runtime_warmup import wait_for_environment_support


@st.cache_data(show_spinner=False, max_entries=24, persist="disk")
def _cached_train_form_preview(
    task_id: str,
    grid_size: int,
    seed: int,
) -> GeneratedPreview:
    wait_for_environment_support(task_id)
    return generate_preview(GenerationRequest(
        environment=EnvironmentSpec(task_id=task_id, grid_size=grid_size),
        seed=seed,
    ))


@dataclass(frozen=True, slots=True)
class TrainFormConfig:
    """User-editable values supported by the first Train workspace."""

    total_timesteps: int = DEFAULT_TIMESTEPS
    grid_size: int = DEFAULT_GRID_SIZE
    seed: int = DEFAULT_SEED
    training_mode: str = PROCEDURAL_TRAINING_MODE
    layout_seed: int | None = None
    task_id: str | None = "door_key-v1"


def render_train_form(
    configuration: Mapping[str, object],
    *,
    disabled: bool = False,
) -> TrainFormConfig | None:
    """Render the Training v0 form and return values only when submitted."""
    saved_layout_seed = configuration.get("layout_seed", DEFAULT_SEED)
    layout_seed_default = (
        int(saved_layout_seed)
        if type(saved_layout_seed) is int
        else DEFAULT_SEED
    )
    definitions = interactive_task_definitions()
    task_ids = tuple(definition.task_id for definition in definitions)
    definitions_by_id = {
        definition.task_id: definition
        for definition in definitions
    }
    saved_task_id = configuration.get("task_id")
    if saved_task_id in (None, LEGACY_OBJECT_INTERACTION_TASK_ID):
        saved_task_id = "door_key-v1"
    if saved_task_id not in task_ids:
        saved_task_id = task_ids[0]
    current_widget_task = st.session_state.get("train_task_selector")
    if current_widget_task not in task_ids:
        st.session_state["train_task_selector"] = saved_task_id
    previous_rendered_task = st.session_state.get(
        "train_task_last_rendered", saved_task_id
    )
    task_id = st.selectbox(
        "TASK",
        options=task_ids,
        format_func=task_code,
        disabled=disabled,
        key="train_task_selector",
    )
    definition = definitions_by_id[task_id]
    # Apply task-specific defaults once when the selector changes. Comparing
    # against the last submitted configuration made every ordinary rerun look
    # like another task change until the form was submitted, which discarded
    # a user's budget edit on the next interaction.
    task_changed = task_id != previous_rendered_task
    st.session_state["train_task_last_rendered"] = task_id
    if task_changed:
        st.session_state.pop("train_steps_input", None)
        st.session_state.pop("train_grid_input", None)
    default_steps = (
        5_017_600
        if task_id == "official-lava-crossing-s9n1-v1"
        else DEFAULT_TIMESTEPS
        if task_changed
        else int(configuration["total_timesteps"])
    )
    preview_seed = int(configuration.get("seed", DEFAULT_SEED))
    try:
        preview = _cached_train_form_preview(
            task_id, definition.default_grid_size, preview_seed
        )
    except Exception as error:
        st.warning(f"TRAINING LAYOUT PREVIEW UNAVAILABLE · {error}")
    else:
        st.caption(
            "TRAINING LAYOUT PREVIEW · PROCEDURAL EXAMPLE · "
            f"SEED {preview_seed} · ORACLE {preview.oracle_status.upper()} · "
            f"LAYOUT {preview.layout_digest[:10]}"
        )
        _, preview_column, _ = st.columns([1, 1, 1])
        with preview_column:
            st.image(preview.frame, caption=preview.mission, width=280)

    with st.form("train_model_form", border=False):
        st.selectbox(
            "ALGORITHM",
            options=("PPO",),
            index=0,
            format_func=lambda _: "PPO · baseline",
            disabled=True,
            key="train_algorithm_selector",
        )
        total_timesteps = st.number_input(
            "TRAINING STEPS",
            min_value=MIN_TIMESTEPS,
            value=max(MIN_TIMESTEPS, default_steps),
            step=MIN_TIMESTEPS,
            disabled=disabled,
            key="train_steps_input",
            help=(
                "PPO collects 128 steps in each of 8 environments per rollout "
                "(1024 transitions), so the completed count may exceed this "
                "requested minimum."
            ),
        )
        grid_size = st.number_input(
            "GRID SIZE",
            min_value=definition.minimum_grid_size,
            value=definition.default_grid_size,
            step=1,
            disabled=True,
            help="New governed benchmark runs use the catalog default grid.",
            key="train_grid_input",
        )
        seed = st.number_input(
            "TRAINING SEED",
            min_value=0,
            max_value=MAX_SEED,
            value=int(configuration.get("seed", DEFAULT_SEED)),
            step=1,
            disabled=disabled,
            key="train_seed_input",
        )
        training_mode = st.selectbox(
            "TRAINING MODE",
            options=(PROCEDURAL_TRAINING_MODE, FIXED_LAYOUT_TRAINING_MODE),
            index=(
                1
                if configuration.get("training_mode") == FIXED_LAYOUT_TRAINING_MODE
                else 0
            ),
            format_func=lambda mode: (
                "Fixed Layout · repeat one layout"
                if mode == FIXED_LAYOUT_TRAINING_MODE
                else "Procedural · new layout each episode"
            ),
            disabled=disabled,
            key="train_mode_selector",
        )
        layout_seed = st.number_input(
            "LAYOUT SEED",
            min_value=0,
            max_value=MAX_SEED,
            value=layout_seed_default,
            step=1,
            disabled=disabled or training_mode == PROCEDURAL_TRAINING_MODE,
            key="train_layout_seed_input",
            help="Fixed Layout mode resets every training episode with this seed.",
        )
        submitted = st.form_submit_button(
            "START TRAINING",
            use_container_width=True,
            disabled=disabled,
        )

    if not submitted:
        return None
    return TrainFormConfig(
        total_timesteps=int(total_timesteps),
        grid_size=int(grid_size),
        seed=int(seed),
        training_mode=str(training_mode),
        layout_seed=(
            int(layout_seed)
            if training_mode == FIXED_LAYOUT_TRAINING_MODE
            else None
        ),
        task_id=task_id,
    )
