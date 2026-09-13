"""Read-only launcher console component."""

from __future__ import annotations

from collections.abc import Sequence
from html import escape

import streamlit as st


def render_console(lines: Sequence[str]) -> None:
    """Render the latest status messages in a selectable console strip."""
    log_text = "\n".join(f"> {escape(line)}" for line in lines[-4:])
    st.markdown(
        "<section class=\"tg-console\">"
        "<div class=\"tg-console-label\">CONSOLE</div>"
        f"<pre class=\"tg-console-body\">{log_text}\n&gt; █</pre>"
        "</section>",
        unsafe_allow_html=True,
    )
