"""Compact governed full-policy and selective-transfer workflow."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from training.initialization import initialization_from_dict
from training.jobs import TrainingJob, read_training_job, start_training_job
from training.trainer import (
    FIXED_LAYOUT_TRAINING_MODE,
    PROCEDURAL_TRAINING_MODE,
    check_transfer_preflight,
)
from ui.components.layout import append_console_log
from ui.display_aliases import compact_count, task_code
from ui.workflow_services import (
    SELECTIVE_TRANSFER_CONDITIONS,
    governed_source_catalog,
    preferred_source_checkpoint,
    run_transfer_preflight,
    source_checkpoint_label,
    target_budget,
    transfer_configuration,
    transfer_pair_catalog,
    transfer_pair_id,
    transfer_selection_key,
)


def _pairs() -> tuple[dict[str, str], ...]:
    return transfer_pair_catalog()


def render_compact_transfer(snapshot: dict | None = None) -> None:
    st.markdown("## TRANSFER")
    st.caption("Choose a source, target and component initialization.")
    pairs = _pairs()
    if not pairs:
        st.info("NO ELIGIBLE GOVERNED SOURCE")
        return
    by_id = {pair["id"]: pair for pair in pairs}
    recommended = (
        snapshot["recommended_transfer"]
        if isinstance(snapshot, dict)
        else {
            "source_task_id": "door_key-v1",
            "target_task_id": "lava_door_key-v2",
            "target_seed": 19,
            "source_checkpoint_path": None,
        }
    )
    preferred_pair = transfer_pair_id(
        str(recommended["source_task_id"]),
        str(recommended["target_task_id"]),
    )
    key = "compact_transfer_pair"
    if st.session_state.get(key) not in by_id:
        st.session_state[key] = (
            preferred_pair if preferred_pair in by_id else pairs[0]["id"]
        )
    pair_id = st.selectbox(
        "SOURCE → TARGET PAIR", tuple(by_id), key=key,
        format_func=lambda value: by_id[value]["label"],
        disabled=st.session_state["transfer_status"] == "running",
    )
    pair = by_id[pair_id]
    source_catalog = governed_source_catalog()
    sources = source_catalog[pair["source_task_id"]]
    condition_key = "compact_transfer_condition"
    condition_ids = tuple(SELECTIVE_TRANSFER_CONDITIONS)
    if st.session_state.get(condition_key) not in condition_ids:
        st.session_state[condition_key] = "C1"
    condition = st.selectbox(
        "INITIALIZATION CONDITION",
        condition_ids,
        key=condition_key,
        format_func=lambda value: str(
            SELECTIVE_TRANSFER_CONDITIONS[value]["label"]
        ),
        disabled=st.session_state["transfer_status"] == "running",
    )
    saved = dict(st.session_state["transfer_config"])
    pair_changed = st.session_state.get("compact_transfer_pair_applied") != pair_id
    if pair_changed:
        st.session_state["compact_transfer_pair_applied"] = pair_id
        for widget_key in (
            "compact_transfer_source_checkpoint",
            "compact_transfer_seed",
            "compact_transfer_budget",
            "compact_transfer_mode",
            "compact_transfer_layout_seed",
        ):
            st.session_state.pop(widget_key, None)
        seed = int(recommended["target_seed"])
        budget = target_budget(pair["target_task_id"])
        mode = PROCEDURAL_TRAINING_MODE
        layout_seed = 42
    else:
        seed = int(saved.get("seed", recommended["target_seed"]))
        budget = int(
            saved.get(
                "total_timesteps",
                target_budget(pair["target_task_id"]),
            )
        )
        mode = str(saved.get("training_mode", PROCEDURAL_TRAINING_MODE))
        layout_seed = int(saved.get("layout_seed") or 42)
    preferred_checkpoint = (
        recommended.get("source_checkpoint_path")
        if pair["source_task_id"] == recommended.get("source_task_id")
        else None
    )
    default_source = preferred_source_checkpoint(
        pair["source_task_id"],
        preferred=preferred_checkpoint,
        source_catalog=source_catalog,
    )
    with st.expander("Advanced options", expanded=False):
        source = st.selectbox(
            "GOVERNED SOURCE CHECKPOINT",
            sources,
            index=sources.index(default_source),
            format_func=source_checkpoint_label,
            key="compact_transfer_source_checkpoint",
        )
        seed = int(st.number_input(
            "TARGET PPO SEED", min_value=0, max_value=2_147_483_647,
            value=seed, step=1, key="compact_transfer_seed",
        ))
        budget = int(st.number_input(
            "TARGET BUDGET", min_value=64, step=102_400,
            value=budget, key="compact_transfer_budget",
        ))
        mode = st.selectbox(
            "TRAINING MODE", (PROCEDURAL_TRAINING_MODE, FIXED_LAYOUT_TRAINING_MODE),
            index=1 if mode == FIXED_LAYOUT_TRAINING_MODE else 0,
            format_func=lambda value: "Fixed Layout" if value == FIXED_LAYOUT_TRAINING_MODE else "Procedural",
            key="compact_transfer_mode",
        )
        layout_seed = int(st.number_input(
            "LAYOUT SEED", min_value=0, max_value=2_147_483_647,
            value=layout_seed, step=1,
            disabled=mode == PROCEDURAL_TRAINING_MODE,
            key="compact_transfer_layout_seed",
        ))
    values = {
        "source_checkpoint_path": str(source),
        "source_task_id": pair["source_task_id"],
        "target_task_id": pair["target_task_id"],
        "condition": condition,
        "seed": seed,
        "total_timesteps": budget,
        "training_mode": mode,
        "layout_seed": layout_seed if mode == FIXED_LAYOUT_TRAINING_MODE else None,
    }
    selection = transfer_selection_key(values)
    running = st.session_state["transfer_status"] == "running"
    if not running and st.session_state.get("transfer_preflight_selection") != selection:
        _preflight(values, selection)
    compatible = _render_compatibility(selection)
    st.markdown(
        '<div class="tg-meta">'
        f'<span>CONDITION: {condition}</span>'
        f'<span>PAIR: {task_code(pair["source_task_id"])} → '
        f'{task_code(pair["target_task_id"])}</span>'
        f'<span>BUDGET: {compact_count(budget)}</span>'
        '<span>OPTIMIZER: FRESH</span>'
        '<span>MODE: INTERACTIVE RUN</span>'
        '</div>', unsafe_allow_html=True,
    )
    if condition == "C2b":
        st.info(
            "C2b copies the actor body and critic and resets the action head. "
            "The optimizer is always initialized fresh for the target run."
        )
    if st.button(
        "START TRANSFER", key="compact_transfer_start", type="primary",
        width="stretch", disabled=running or not compatible,
    ):
        _start(values)
    if st.session_state["transfer_status"] == "running":
        _monitor()
    else:
        _render_state()


def _preflight(values: dict, selection: str) -> None:
    st.session_state["transfer_status"] = "checking"
    st.session_state["transfer_preflight_selection"] = selection
    st.session_state["transfer_config"] = dict(values)
    try:
        with st.spinner("Checking compatibility automatically..."):
            initialization, report = run_transfer_preflight(values)
    except Exception as error:
        st.session_state["transfer_status"] = "incompatible"
        st.session_state["transfer_error"] = str(error)
        st.session_state["transfer_initialization"] = None
        st.session_state["transfer_compatibility"] = None
        return
    st.session_state["transfer_initialization"] = initialization
    st.session_state["transfer_compatibility"] = report
    st.session_state["transfer_status"] = "compatible" if report.get("ready") else "incompatible"


def _render_compatibility(selection: str) -> bool:
    report = st.session_state.get("transfer_compatibility")
    if st.session_state.get("transfer_preflight_selection") != selection or not isinstance(report, dict):
        st.info("CHECKING COMPATIBILITY")
        return False
    if report.get("ready") is True:
        st.success("COMPATIBLE · OBSERVATION PASS · ACTION PASS · POLICY PASS")
        return True
    technical = report.get("technical_compatibility", {})
    competence = report.get("source_competence", {})
    failures = tuple(technical.get("failure_codes", ())) + tuple(competence.get("failure_codes", ()))
    st.error("INCOMPATIBLE · " + ", ".join(failures or ("preflight failed",)))
    return False


def _start(values: dict) -> None:
    try:
        initialization = initialization_from_dict(st.session_state["transfer_initialization"])
        configuration = transfer_configuration(values, initialization)
        fresh = check_transfer_preflight(configuration)
        cached = st.session_state.get("transfer_compatibility")
        if not fresh.ready or not isinstance(cached, dict) or cached.get("binding_digest") != fresh.binding_digest:
            raise ValueError("Transfer compatibility became stale before launch.")
        job = start_training_job(configuration)
    except Exception as error:
        st.session_state["transfer_status"] = "failed"
        st.session_state["transfer_error"] = str(error)
        append_console_log(f"[ERROR] Compact transfer failed to start: {error}")
        return
    st.session_state["transfer_config"] = dict(values)
    st.session_state["transfer_job"] = job.to_dict()
    st.session_state["transfer_summary"] = None
    st.session_state["transfer_status"] = "running"
    append_console_log(f"[INFO] Compact transfer {job.run_id} started")


@st.fragment(run_every=1.0)
def _monitor() -> None:
    payload = st.session_state.get("transfer_job")
    if not isinstance(payload, dict):
        st.session_state["transfer_status"] = "failed"
        st.session_state["transfer_error"] = "Transfer job metadata is missing."
        return
    state = read_training_job(TrainingJob.from_dict(payload))
    if state.status == "running":
        st.info("TRANSFER IN PROGRESS")
        return
    st.session_state["transfer_status"] = state.status
    st.session_state["transfer_summary"] = state.summary if state.status == "succeeded" else None
    st.session_state["transfer_error"] = state.error
    st.rerun(scope="app")


def _render_state() -> None:
    status = st.session_state["transfer_status"]
    if status in {"idle", "compatible", "checking", "incompatible"}:
        return
    if status == "failed":
        st.error(f"FAILED · {st.session_state['transfer_error']}")
    elif status == "succeeded":
        st.success("TRANSFER COMPLETE")
        summary = st.session_state.get("transfer_summary")
        if isinstance(summary, dict):
            st.caption(f"RUN {summary.get('run_id', '?')} · FINAL SHA {str(summary.get('checkpoint_sha256', ''))[:12]}")
