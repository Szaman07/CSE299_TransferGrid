"""Default four-workflow TransferGrid interface."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from ui.components import render_header, render_status_bar
from ui.components.layout import initialize_home_state
from ui.theme import inject_theme_css


WORKFLOWS = ("train", "evaluate", "transfer", "replay")
WORKFLOW_LABELS = {name: name.upper() for name in WORKFLOWS}


def _select_workflow(workflow: str) -> None:
    st.session_state["compact_workflow"] = workflow


def render_compact_home(workspace_path: Path) -> None:
    inject_theme_css()
    initialize_home_state(workspace_path)
    st.session_state.setdefault("compact_workflow", "train")
    if st.session_state["compact_workflow"] not in WORKFLOWS:
        st.session_state["compact_workflow"] = "train"

    render_header()
    st.markdown("### COMPACT WORKBENCH")
    st.caption("TRAIN · EVALUATE · TRANSFER · REPLAY")

    columns = st.columns(4, gap="small")
    active = str(st.session_state["compact_workflow"])
    for column, workflow in zip(columns, WORKFLOWS, strict=True):
        with column:
            st.button(
                WORKFLOW_LABELS[workflow],
                key=f"compact_nav_{workflow}",
                type="primary" if workflow == active else "secondary",
                width="stretch",
                on_click=_select_workflow,
                args=(workflow,),
            )

    st.divider()
    snapshot = st.session_state.get("compact_local_snapshot_v3")
    if active == "train":
        from ui.compact.train import render_compact_train

        render_compact_train(snapshot if isinstance(snapshot, dict) else None)
    elif active == "transfer":
        from ui.compact.transfer import render_compact_transfer

        render_compact_transfer(snapshot if isinstance(snapshot, dict) else None)
    else:
        evidence_error: str | None = None
        if not isinstance(snapshot, dict):
            try:
                from ui.workflow_services import local_artifact_snapshot

                with st.spinner("Discovering verified local artifacts..."):
                    snapshot = local_artifact_snapshot()
            except Exception as error:
                evidence_error = str(error)

        if active == "evaluate":
            from ui.compact.evaluation import render_compact_evaluation

            render_compact_evaluation(
                snapshot if isinstance(snapshot, dict) else None,
                evidence_error=evidence_error,
            )
        else:
            from ui.compact.replay import render_compact_replay

            if isinstance(snapshot, dict):
                render_compact_replay(snapshot)
            else:
                st.error("NO VERIFIED LOCAL REPLAY ARTIFACT IS AVAILABLE")
                st.caption("Train and evaluate a checkpoint on this machine first.")

    render_status_bar(str(workspace_path))
