"""Checkpoint-first presentation controls for automatic Evaluation routing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import streamlit as st

from ui.display_aliases import artifact_route_label
from training.artifact_catalog import (
    TRANSFER_FINAL_CHECKPOINT,
    TRANSFER_INITIALIZATION_CHECKPOINT,
    classify_checkpoint_metadata,
)
from training.checkpoints import load_checkpoint_metadata
from training.evaluation import (
    CANONICAL_EVALUATION_DISTRIBUTIONS,
    DEFAULT_EVALUATION_DISTRIBUTION_NAME,
    FINAL_TEST_EVALUATION_DISTRIBUTION_NAME,
    MAX_EVALUATION_EPISODE_COUNT,
    VALIDATION_EVALUATION_DISTRIBUTION_NAME,
    format_evaluation_seeds,
)


DISTRIBUTION_OPTIONS = (
    DEFAULT_EVALUATION_DISTRIBUTION_NAME,
    VALIDATION_EVALUATION_DISTRIBUTION_NAME,
    FINAL_TEST_EVALUATION_DISTRIBUTION_NAME,
    "custom",
)

CANONICAL_SEED_TEXTS = {
    DEFAULT_EVALUATION_DISTRIBUTION_NAME: format_evaluation_seeds(
        CANONICAL_EVALUATION_DISTRIBUTIONS[DEFAULT_EVALUATION_DISTRIBUTION_NAME].seeds
    ),
    VALIDATION_EVALUATION_DISTRIBUTION_NAME: format_evaluation_seeds(
        CANONICAL_EVALUATION_DISTRIBUTIONS[VALIDATION_EVALUATION_DISTRIBUTION_NAME].seeds
    ),
    FINAL_TEST_EVALUATION_DISTRIBUTION_NAME: format_evaluation_seeds(
        CANONICAL_EVALUATION_DISTRIBUTIONS[FINAL_TEST_EVALUATION_DISTRIBUTION_NAME].seeds
    ),
}


@dataclass(frozen=True, slots=True)
class EvaluationFormConfig:
    """Raw user input forwarded to the backend evaluation configuration."""

    checkpoint_path: str
    distribution_name: str
    seed_text: str


def render_evaluation_form(
    configuration: Mapping[str, object],
    checkpoints: tuple[Path, ...],
) -> EvaluationFormConfig | None:
    """Backward-compatible checkpoint-first wrapper for legacy Evaluation."""
    checkpoint_path = render_evaluation_checkpoint_selector(
        configuration,
        checkpoints,
    )
    if checkpoint_path is None:
        return None
    return render_legacy_evaluation_form(configuration, checkpoint_path)


def render_evaluation_checkpoint_selector(
    configuration: Mapping[str, object],
    checkpoints: tuple[Path, ...],
) -> str | None:
    """Select an exact checkpoint before showing protocol-owned controls."""
    checkpoint_options = tuple(str(path) for path in checkpoints)
    saved_path = str(configuration.get("checkpoint_path", ""))
    if saved_path and saved_path not in checkpoint_options:
        resolved_saved_path = str(Path(saved_path).resolve())
        if resolved_saved_path in checkpoint_options:
            saved_path = resolved_saved_path
    widget_key = "evaluation_checkpoint_selector"
    widget_path = st.session_state.get(widget_key)
    if widget_path not in checkpoint_options:
        st.session_state.pop(widget_key, None)
        widget_path = saved_path
    selected_index = (
        checkpoint_options.index(widget_path)
        if widget_path in checkpoint_options
        else None
    )
    return st.selectbox(
        "CHECKPOINT",
        options=checkpoint_options,
        index=selected_index,
        format_func=_checkpoint_label,
        placeholder="Select the exact checkpoint to evaluate",
        help="Only checkpoints with a valid TransferGrid metadata sidecar are listed.",
        key=widget_key,
    )


def render_legacy_evaluation_form(
    configuration: Mapping[str, object],
    checkpoint_path: str,
) -> EvaluationFormConfig | None:
    """Collect Evaluation V1.2 conditions for an already-selected checkpoint."""
    saved_dist = str(
        configuration.get("distribution_name", DEFAULT_EVALUATION_DISTRIBUTION_NAME)
    )
    selected_dist_index = (
        DISTRIBUTION_OPTIONS.index(saved_dist)
        if saved_dist in DISTRIBUTION_OPTIONS
        else 0
    )

    with st.form("evaluation_v1_form", border=False):
        selected_distribution = st.selectbox(
            "EVALUATION DISTRIBUTION",
            options=DISTRIBUTION_OPTIONS,
            index=selected_dist_index,
            format_func=_format_distribution_label,
            help=(
                "Select a governed evaluation distribution. Development (1001-1010), "
                "Validation (1011-1050), Final Test (50001-50050), or Custom."
            ),
            key="legacy_evaluation_distribution_selector",
        )

        if selected_distribution == "custom":
            distribution_name = st.text_input(
                "CUSTOM DISTRIBUTION NAME",
                value=str(configuration.get("distribution_name", "custom-evaluation")),
                help="A unique name recorded with your custom seed set.",
            )
            seed_text = st.text_input(
                "EVALUATION SEEDS",
                value=str(configuration.get("seed_text", "1001, 1002, 1003")),
                help=(
                    "Comma- or space-separated seeds. Evaluation V1.2 accepts at most "
                    f"{MAX_EVALUATION_EPISODE_COUNT} seeds per report."
                ),
            )
        else:
            distribution_name = selected_distribution
            canonical_seeds = CANONICAL_SEED_TEXTS[selected_distribution]
            seed_text = st.text_input(
                "EVALUATION SEEDS",
                value=canonical_seeds,
                help=f"Immutable canonical seed set for {selected_distribution}.",
            )

        st.checkbox(
            "DETERMINISTIC POLICY ACTIONS",
            value=True,
            disabled=True,
            help="Evaluation Protocol V1.2 fixes deterministic PPO actions.",
        )
        submitted = st.form_submit_button(
            "RUN EVALUATION V1.2",
            width="stretch",
        )

    if not submitted:
        return None
    return EvaluationFormConfig(
        checkpoint_path=checkpoint_path,
        distribution_name=distribution_name,
        seed_text=seed_text,
    )


def _format_distribution_label(opt: str) -> str:
    """Format distribution option labels for the dropdown selector."""
    if opt in CANONICAL_EVALUATION_DISTRIBUTIONS:
        defn = CANONICAL_EVALUATION_DISTRIBUTIONS[opt]
        seeds = defn.seeds
        count = len(seeds)
        first_seed, last_seed = seeds[0], seeds[-1]
        role_tag = defn.role.upper()
        return f"{opt} ({role_tag} · Seeds {first_seed}–{last_seed} · {count} ep)"
    return "Custom Seed Distribution"


def _checkpoint_label(checkpoint_path: str) -> str:
    """Present checkpoint identity without exposing a long filesystem path."""
    try:
        metadata = load_checkpoint_metadata(checkpoint_path)
        environment = metadata.get("environment", {})
        grid_size = environment.get("grid_size", "?")
        seed = metadata.get("seed", "?")
        steps = _compact_count(metadata.get("requested_timesteps"))
        mode = (
            "Fixed layout"
            if metadata.get("training_mode") == "fixed_layout"
            else "Procedural"
        )
        run_id = metadata.get("run_id")
        checkpoint_kind = classify_checkpoint_metadata(metadata)
        kind_label = {
            TRANSFER_FINAL_CHECKPOINT: "TRANSFER · FINAL POLICY",
            TRANSFER_INITIALIZATION_CHECKPOINT: "TRANSFER · INITIALIZATION 0%",
        }.get(checkpoint_kind, "NORMAL TRAINING · FINAL POLICY")
        task_id = environment.get("task_id")
        initialization = metadata.get("initialization")
        source_task_id = (
            initialization.get("source_task_id")
            if isinstance(initialization, dict)
            else None
        )
        try:
            task_name = artifact_route_label(task_id, source_task_id)
        except (TypeError, ValueError):
            task_name = "Legacy DoorKey"
        identity = (
            f"Run {str(run_id)[-8:]}"
            if isinstance(run_id, str) and run_id
            else "Legacy"
        )
        return (
            f"{kind_label} · {task_name} · Grid {grid_size} · "
            f"{mode} · Seed {seed} · "
            f"{steps} steps · {identity}"
        )
    except (OSError, TypeError, ValueError):
        return Path(checkpoint_path).stem


def _compact_count(value: object) -> str:
    """Format large step counts for compact selector labels."""
    if type(value) is not int:
        return "?"
    if value >= 1_000_000 and value % 1_000_000 == 0:
        return f"{value // 1_000_000}M"
    if value >= 1_000 and value % 1_000 == 0:
        return f"{value // 1_000}k"
    return f"{value:,}"
