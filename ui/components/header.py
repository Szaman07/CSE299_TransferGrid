"""TransferGrid product header."""

from __future__ import annotations

import streamlit as st


def render_header(version: str = "v0.1.0") -> None:
    """Render the logo, product name, subtitle, and version."""
    st.markdown(
        f"""
        <header class="tg-header">
          <span class="tg-version">{version}</span>
          <div class="tg-brand">
            <svg class="tg-logo" viewBox="0 0 48 48" aria-hidden="true">
              <rect x="1" y="1" width="46" height="46" rx="10" fill="none"
                    stroke="#FFFFFF" stroke-width="1.7"/>
              <path d="M1 16h46M1 32h46M16 1v46M32 1v46" fill="none"
                    stroke="#FFFFFF" stroke-width="1.4"/>
              <path d="M24 21l6 10H18z" fill="none" stroke="#FFFFFF"
                    stroke-width="1.5"/>
            </svg>
            <div>
              <h1 class="tg-title">TRANSFERGRID</h1>
              <div class="tg-subtitle">REINFORCEMENT LEARNING EXPERIMENTS</div>
            </div>
          </div>
        </header>
        """,
        unsafe_allow_html=True,
    )
