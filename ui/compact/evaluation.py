"""Compact governed-Validation workflow."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from training.checkpoints import load_checkpoint_metadata
from ui.components.layout import append_console_log
from ui.display_aliases import artifact_route_label, artifact_selection_label
from ui.runtime_warmup import wait_for_environment_support
from ui.workflow_services import execute_governed_validation


def render_compact_evaluation(
    snapshot: dict | None,
    *,
    evidence_error: str | None = None,
) -> None:
    st.markdown("## EVALUATE")
    if not isinstance(snapshot, dict):
        st.error("NO VERIFIED LOCAL CHECKPOINT IS AVAILABLE")
        if evidence_error:
            st.caption("Train and evaluate a checkpoint locally to populate this view.")
        return
    local_scope = snapshot.get("scope") in {
        "individually-verified-local-artifacts", "indexed-local-artifacts",
    }
    st.caption(
        "Run deterministic validation for one individually verified checkpoint."
        if local_scope else
        "Run frozen deterministic Validation for one certified checkpoint."
    )
    coverage = tuple(snapshot.get("condition_coverage", ()))
    if coverage:
        st.caption(
            "PPO STUDY COVERAGE · " + " · ".join(
                condition for condition in coverage if condition != "R&D"
            )
        )
    runs = {run["alias"]: run for run in snapshot["runs"]}
    default = str(snapshot["recommended_evaluation_alias"])
    if not runs:
        st.error("NO LOCALLY VERIFIABLE CHECKPOINT IS AVAILABLE")
        return
    aliases = tuple(runs)
    key = "compact_evaluation_artifact"
    if st.session_state.get(key) not in aliases:
        st.session_state[key] = default if default in aliases else aliases[0]
    selected = st.selectbox(
        "VERIFIED ARTIFACT" if local_scope else "CERTIFIED ARTIFACT",
        aliases,
        key=key,
        format_func=lambda alias: artifact_selection_label(runs[alias]),
    )
    run = runs[selected]
    route_label = artifact_route_label(
        run["task_id"], run.get("source_task_id")
    )
    st.markdown(
        '<div class="tg-meta">'
        f'<span>ARTIFACT: {selected}</span>'
        f'<span>ROUTE: {route_label}</span>'
        f'<span>STUDY: {run["study_label"]}</span>'
        '<span>PROTOCOL: V2 VALIDATION</span>'
        '<span>EPISODES: 40</span>'
        f'<span>SHA: {run["checkpoint_sha256"][:12]}</span>'
        '</div>',
        unsafe_allow_html=True,
    )
    if st.button(
        "RUN VALIDATION",
        key="compact_evaluation_start",
        type="primary",
        width="stretch",
        disabled=st.session_state["evaluation_status"] == "running",
    ):
        _run(run)
    _render_state()


def _run(run: dict) -> None:
    checkpoint = Path(run["checkpoint_path"])
    st.session_state["evaluation_status"] = "running"
    st.session_state["evaluation_report"] = None
    st.session_state["evaluation_report_path"] = None
    st.session_state["evaluation_error"] = None
    append_console_log(f"[INFO] Compact Validation requested for {run['alias']}")
    try:
        metadata = load_checkpoint_metadata(checkpoint)
        environment = metadata.get("environment")
        task_id = environment.get("task_id") if isinstance(environment, dict) else None
        wait_for_environment_support(task_id)
        with st.spinner("Executing 40 deterministic Validation episodes..."):
            report, report_path = execute_governed_validation(checkpoint)
    except Exception as error:
        st.session_state["evaluation_status"] = "failed"
        st.session_state["evaluation_error"] = str(error)
        append_console_log(f"[ERROR] Compact Validation failed: {error}")
        return
    st.session_state["evaluation_status"] = (
        "succeeded" if report["aggregate_metrics"]["report_complete"] else "incomplete"
    )
    st.session_state["evaluation_report"] = report
    st.session_state["evaluation_report_path"] = str(report_path)
    append_console_log("[SUCCESS] Compact Validation report saved")


def _render_state() -> None:
    status = st.session_state["evaluation_status"]
    if status == "idle":
        st.info("READY · deterministic held-out Validation")
        return
    if status == "running":
        st.info("EVALUATION IN PROGRESS")
        return
    if status == "failed":
        st.error(f"FAILED · {st.session_state['evaluation_error']}")
        return
    report = st.session_state.get("evaluation_report")
    if not isinstance(report, dict):
        st.error("Evaluation completed without report metadata.")
        return
    metrics = report["aggregate_metrics"]
    if status == "succeeded":
        st.success("VALIDATION COMPLETE")
    else:
        st.error("VALIDATION INCOMPLETE")
    columns = st.columns(3)
    columns[0].metric("SUCCESS", f"{float(metrics['success_rate']):.1%}")
    columns[1].metric("COMPLETED", metrics["completed_episode_count"])
    columns[2].metric("ERRORS", metrics.get("error_count", 0))
