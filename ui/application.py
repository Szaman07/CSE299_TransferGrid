"""Top-level presentation-mode router."""

from __future__ import annotations

from pathlib import Path

import streamlit as st


def render_application(workspace_path: Path) -> None:
    """Render the focused public workbench."""
    from ui.compact import render_compact_home

    render_compact_home(workspace_path)
