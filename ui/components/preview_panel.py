"""Reusable environment preview panel."""

from __future__ import annotations

from dataclasses import dataclass
from html import escape
from typing import Any

import streamlit as st


@dataclass(frozen=True)
class PreviewModel:
    """Display data for a right-hand launcher preview."""

    panel_title: str = "ENVIRONMENT PREVIEW"
    env_name: str = "OBJECT INTERACTION"
    description: str = "Collect the key, unlock the door, and reach the goal."
    image: Any | None = None
    seed: int = 42
    grid: str = "8 x 8"
    difficulty: str = "MEDIUM"
    mission: str | None = None
    observation: str | None = None
    error_message: str | None = None
    task_id: str | None = None
    layout_digest: str | None = None
    oracle_status: str | None = None
    barrier_orientation: str | None = None
    safe_gap: tuple[int, int] | None = None


def render_preview_panel(model: PreviewModel) -> None:
    """Render a framed preview from presentation data supplied by a page."""
    with st.container(border=True):
        st.markdown(
            f'<h2 class="tg-panel-title">{escape(model.panel_title)}</h2>',
            unsafe_allow_html=True,
        )
        copy_column, image_column = st.columns([4, 5], gap="small")
        with copy_column:
            st.markdown(
                f'<h3 class="tg-env-name">{escape(model.env_name)}</h3>'
                f'<p class="tg-description">{escape(model.description)}</p>',
                unsafe_allow_html=True,
            )
            _render_details(model)
        with image_column:
            if model.image is not None:
                st.image(model.image, width=300)
        st.markdown('<div class="tg-divider"></div>', unsafe_allow_html=True)
        st.markdown(
            "<div class=\"tg-meta\">"
            f"<span>SEED: {model.seed}</span>"
            f"<span>GRID: {model.grid}</span>"
            f"<span>DIFFICULTY: {model.difficulty}</span>"
            "</div>",
            unsafe_allow_html=True,
        )


def _render_details(model: PreviewModel) -> None:
    """Render optional mission, observation, or error text without side effects."""
    if model.error_message is not None:
        st.markdown(
            f'<p class="tg-description"><strong>ERROR:</strong> '
            f"{escape(model.error_message)}</p>",
            unsafe_allow_html=True,
        )
        return

    details = []
    if model.mission is not None:
        details.append(f"MISSION: {escape(model.mission)}")
    if model.observation is not None:
        details.append(f"OBSERVATION: {escape(model.observation)}")
    if model.task_id is not None:
        details.append(f"TASK ID: {escape(model.task_id)}")
    if model.layout_digest is not None:
        details.append(f"LAYOUT: {escape(model.layout_digest[:16])}…")
    if model.oracle_status is not None:
        details.append(f"ORACLE: {escape(model.oracle_status.upper())}")
    if model.barrier_orientation is not None:
        details.append(f"BARRIER: {escape(model.barrier_orientation.upper())}")
    if model.safe_gap is not None:
        details.append(f"SAFE GAP: {model.safe_gap}")
    if details:
        st.markdown(
            f'<p class="tg-description">{"<br>".join(details)}</p>',
            unsafe_allow_html=True,
        )
