"""Status footer component."""

from __future__ import annotations

from html import escape

import streamlit as st


def _shorten_path(path: str, maximum_length: int = 48) -> str:
    """Preserve the start and project folder when shortening a workspace path."""
    if len(path) <= maximum_length:
        return path
    prefix_length = (maximum_length - 3) // 2
    suffix_length = maximum_length - prefix_length - 3
    return f"{path[:prefix_length]}...{path[-suffix_length:]}"


def render_status_bar(
    workspace: str,
    version: str = "v0.1.0",
    hints: str = "↑↓ select    Enter open",
) -> None:
    """Render the quiet launcher status line."""
    safe_workspace = escape(_shorten_path(workspace))
    st.markdown(
        "<div class=\"tg-status\">"
        f"workspace: {safe_workspace} &nbsp; | &nbsp; {version} &nbsp; | &nbsp; {hints}"
        "</div>",
        unsafe_allow_html=True,
    )
