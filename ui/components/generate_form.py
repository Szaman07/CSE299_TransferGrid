"""Configuration form for the Object Interaction generator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import streamlit as st

from environments.catalog import (
    LEGACY_OBJECT_INTERACTION_TASK_ID,
    interactive_task_definitions,
)
from environments.factory import (
    DEFAULT_GRID_SIZE,
    MEDIUM_DIFFICULTY,
    OBJECT_INTERACTION_CAPABILITY,
)
from environments.generation import DEFAULT_SEED, MAX_SEED
from ui.display_aliases import task_code


@dataclass(frozen=True, slots=True)
class GenerateConfig:
    """The supported configuration for generating an environment preview."""

    capability: str = OBJECT_INTERACTION_CAPABILITY
    seed: int = DEFAULT_SEED
    grid_size: int = DEFAULT_GRID_SIZE
    difficulty: str = MEDIUM_DIFFICULTY
    task_id: str | None = "goal_navigation-v1"


def render_generate_form(
    configuration: Mapping[str, object],
) -> GenerateConfig | None:
    """Render the minimal Generate form and return its submitted configuration."""
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
    with st.form("generate_environment_form", border=False):
        task_id = st.selectbox(
            "TASK",
            options=task_ids,
            index=task_ids.index(saved_task_id),
            format_func=task_code,
            key="generate_task_selector",
        )
        definition = definitions_by_id[task_id]
        seed = st.number_input(
            "RANDOM SEED",
            min_value=0,
            max_value=MAX_SEED,
            value=int(configuration["seed"]),
            step=1,
            key="generate_seed_input",
        )
        grid_size = st.number_input(
            "GRID SIZE",
            min_value=definition.minimum_grid_size,
            value=definition.default_grid_size,
            step=1,
            disabled=True,
            help="New governed benchmark runs use the catalog default grid.",
            key="generate_grid_input",
        )
        submitted = st.form_submit_button("GENERATE", use_container_width=True)

    if not submitted:
        return None

    return GenerateConfig(
        seed=int(seed),
        grid_size=int(grid_size),
        task_id=task_id,
    )
