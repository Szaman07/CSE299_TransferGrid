"""Streamlit entry point for the TransferGrid workbench."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from ui.application import render_application
from ui.runtime_warmup import start_environment_support_warmup


st.set_page_config(
    page_title="TransferGrid",
    layout="wide",
    initial_sidebar_state="collapsed",
)

start_environment_support_warmup()
render_application(Path(__file__).resolve().parent)
