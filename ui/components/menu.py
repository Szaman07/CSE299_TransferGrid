"""Launcher menu component."""

from __future__ import annotations

from collections.abc import Sequence

import streamlit as st

from ui.constants import MENU_ITEMS, WORKSPACE_LABELS


def _activate_view(item: str) -> None:
    """Commit navigation before Streamlit begins the button-triggered rerun."""
    if st.session_state.get("active_view") != item:
        st.session_state["navigation_log_pending"] = item
    st.session_state["active_view"] = item


def render_main_menu(
    selected: str,
    items: Sequence[str] = MENU_ITEMS,
) -> str:
    """Render the fixed launcher menu and return the selected item."""
    for index, item in enumerate(items, start=1):
        st.button(
            WORKSPACE_LABELS[item],
            key=f"menu_item_{index}",
            type="primary" if item == selected else "secondary",
            use_container_width=True,
            on_click=_activate_view,
            args=(item,),
        )
    return str(st.session_state.get("active_view", selected))
