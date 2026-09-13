"""Compact deterministic episode Replay workflow."""

from __future__ import annotations

from pathlib import Path
from time import monotonic

import streamlit as st

from ui.display_aliases import artifact_selection_label
from ui.runtime_warmup import wait_for_environment_support
from ui.workflow_services import cached_replay_trace


SPEEDS = (1, 2, 4, 8)
BASE_FRAME_DELAY_SECONDS = 0.8
PLAYBACK_TICK_SECONDS = 0.2


def _checkpoint_changed() -> None:
    st.session_state.pop("compact_replay_episode_seed", None)


def _playback_delay_seconds(speed: int) -> float:
    """Return a visible, deterministic interval for each playback speed."""
    if speed not in SPEEDS:
        raise ValueError(f"Unsupported compact replay speed: {speed}")
    return BASE_FRAME_DELAY_SECONDS / speed


def _playback_speed_changed() -> None:
    """Re-anchor active playback so a speed change takes effect immediately."""
    if st.session_state.get("compact_replay_playing", False):
        st.session_state["compact_replay_anchor_step"] = int(
            st.session_state.get("compact_replay_step", 0)
        )
        st.session_state["compact_replay_anchor_time"] = monotonic()


def _playback_index_for_elapsed(
    current_index: int,
    last_index: int,
    *,
    anchor_index: int,
    elapsed_seconds: float,
    speed: int,
) -> int:
    """Map elapsed wall time to a trace index, catching up without UI floods."""
    elapsed_steps = int(
        max(0.0, elapsed_seconds) / _playback_delay_seconds(speed)
    )
    return min(last_index, max(current_index, anchor_index + elapsed_steps))


def _clear_playback_clock() -> None:
    st.session_state.pop("compact_replay_anchor_step", None)
    st.session_state.pop("compact_replay_anchor_time", None)


def _start_playback_clock(current_index: int) -> None:
    st.session_state["compact_replay_anchor_step"] = current_index
    st.session_state["compact_replay_anchor_time"] = monotonic()


def render_compact_replay(snapshot: dict) -> None:
    st.markdown("## REPLAY")
    st.caption("Choose one verified checkpoint and one exact deterministic episode seed.")

    runs = {run["alias"]: run for run in snapshot["runs"]}
    checkpoint_key = "compact_replay_checkpoint"
    recommended_alias = str(snapshot["recommended_replay_alias"])
    if st.session_state.get(checkpoint_key) not in runs:
        st.session_state[checkpoint_key] = (
            recommended_alias if recommended_alias in runs else tuple(runs)[0]
        )
    selected_alias = st.selectbox(
        "EVALUATION CHECKPOINT",
        tuple(runs),
        key=checkpoint_key,
        format_func=lambda alias: artifact_selection_label(runs[alias]),
        on_change=_checkpoint_changed,
    )
    run = runs[selected_alias]
    choices = {choice["label"]: choice for choice in run["episodes"]}
    seed_key = "compact_replay_episode_seed"
    if st.session_state.get(seed_key) not in choices:
        st.session_state[seed_key] = tuple(choices)[0]
    selected_seed = st.selectbox(
        "EPISODE SEED",
        tuple(choices),
        key=seed_key,
    )
    st.caption(
        f"Checkpoint: {Path(run['checkpoint_path']).name} · "
        f"Report: {run['validation_report_id']}"
    )
    choice = choices[selected_seed]
    st.markdown(
        '<div class="tg-meta">'
        f'<span>ARTIFACT: {choice["alias"]}</span>'
        f'<span>SEED: {choice["seed"]}</span>'
        f'<span>OUTCOME: {choice["outcome"]}</span>'
        '<span>SOURCE: EXACT CHECKPOINT</span>'
        '</div>', unsafe_allow_html=True,
    )
    selection = f"{choice['report_path']}:{choice['episode_index']}"
    if st.button("OPEN REPLAY", key="compact_replay_open", type="primary", width="stretch"):
        st.session_state["compact_replay_loaded"] = selection
        st.session_state["compact_replay_step"] = 0
        st.session_state["compact_replay_playing"] = False
        _clear_playback_clock()
    if st.session_state.get("compact_replay_loaded") != selection:
        st.info("READY · select Open Replay to regenerate this episode")
        return
    try:
        with st.spinner("Recreating checkpoint, environment, and deterministic actions..."):
            wait_for_environment_support(choice["task_id"])
            trace = cached_replay_trace(choice["report_path"], int(choice["episode_index"]))
    except Exception as error:
        st.error(f"REPLAY FAILED · {error}")
        return
    _render_trace(trace, choice["alias"])


@st.fragment(run_every=PLAYBACK_TICK_SECONDS)
def _render_trace(trace, alias: str) -> None:
    last = len(trace.steps) - 1
    current_index = min(int(st.session_state.get("compact_replay_step", 0)), last)
    current = trace.steps[current_index]
    if current_index >= last:
        st.session_state["compact_replay_playing"] = False
        _clear_playback_clock()
    _, frame_column, _ = st.columns([1, 1, 1])
    with frame_column:
        st.image(
            current.frame,
            caption=f"{alias} · STEP {current_index}/{last}",
            width="stretch",
        )
    metrics = st.columns(4)
    metrics[0].metric("ACTION", current.action_name)
    metrics[1].metric("REWARD", f"{current.immediate_reward:.3f}")
    metrics[2].metric("CUMULATIVE", f"{current.cumulative_reward:.3f}")
    metrics[3].metric("SUCCESS", "YES" if current.success else "NO")
    previous, next_column, play, pause, reset, speed_column = st.columns([1, 1, 1, 1, 1, 2])
    if previous.button("PREVIOUS", key="compact_replay_previous", disabled=current_index == 0, width="stretch"):
        st.session_state["compact_replay_step"] = max(0, current_index - 1)
        st.session_state["compact_replay_playing"] = False
        _clear_playback_clock()
        st.rerun(scope="fragment")
    if next_column.button("NEXT", key="compact_replay_next", disabled=current_index == last, width="stretch"):
        st.session_state["compact_replay_step"] = min(last, current_index + 1)
        st.session_state["compact_replay_playing"] = False
        _clear_playback_clock()
        st.rerun(scope="fragment")
    if play.button("PLAY", key="compact_replay_play", disabled=current_index == last, width="stretch"):
        st.session_state["compact_replay_playing"] = True
        _start_playback_clock(current_index)
        st.rerun(scope="fragment")
    if pause.button("PAUSE", key="compact_replay_pause", disabled=not st.session_state.get("compact_replay_playing", False), width="stretch"):
        st.session_state["compact_replay_playing"] = False
        _clear_playback_clock()
        st.rerun(scope="fragment")
    if reset.button("RESET", key="compact_replay_reset", disabled=current_index == 0, width="stretch"):
        st.session_state["compact_replay_step"] = 0
        st.session_state["compact_replay_playing"] = False
        _clear_playback_clock()
        st.rerun(scope="fragment")
    speed = speed_column.selectbox(
        "SPEED",
        SPEEDS,
        format_func=lambda value: f"{value}×",
        key="compact_replay_speed",
        on_change=_playback_speed_changed,
    )
    speed_column.caption(f"{_playback_delay_seconds(int(speed)):.1f}s per frame")
    if st.session_state.get("compact_replay_playing", False) and current_index < last:
        now = monotonic()
        anchor_index = int(st.session_state.setdefault(
            "compact_replay_anchor_step", current_index,
        ))
        anchor_time = float(st.session_state.setdefault(
            "compact_replay_anchor_time", now,
        ))
        next_index = _playback_index_for_elapsed(
            current_index,
            last,
            anchor_index=anchor_index,
            elapsed_seconds=now - anchor_time,
            speed=int(speed),
        )
        if next_index != current_index:
            st.session_state["compact_replay_step"] = next_index
