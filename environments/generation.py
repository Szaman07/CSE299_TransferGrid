"""One-shot environment generation service used by the workbench."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from typing import Any

import numpy as np

from environments.factory import EnvironmentSpec, make_environment
from utils.wrappers import stable_minigrid_layout_signature
from environments.goal_navigation_oracle import shortest_goal_actions
from environments.key_pickup_oracle import shortest_key_pickup_actions
from environments.door_key_oracle import shortest_door_key_actions
from environments.lava_navigation_oracle import shortest_safe_lava_navigation_actions
from environments.lava_door_key_oracle import shortest_lava_door_key_actions
from environments.lava_door_key_v2_oracle import (
    shortest_lava_door_key_v2_actions,
    shortest_lava_door_key_v2_failure_actions,
)

DEFAULT_SEED = 42
MAX_SEED = 2_147_483_647


@dataclass(frozen=True, slots=True)
class GenerationRequest:
    """Input required to generate one deterministic environment preview."""

    environment: EnvironmentSpec = field(default_factory=EnvironmentSpec)
    seed: int = DEFAULT_SEED

    def __post_init__(self) -> None:
        if type(self.seed) is not int or not 0 <= self.seed <= MAX_SEED:
            raise ValueError(f"seed must be an integer from 0 to {MAX_SEED}.")


@dataclass(frozen=True, slots=True)
class GeneratedPreview:
    """Serializable preview data with no live environment resources."""

    request: GenerationRequest
    frame: np.ndarray
    mission: str
    observation_summary: str
    layout_signature: tuple[tuple[str, tuple[int, int]], ...]
    task_id: str
    task_contract_digest: str | None
    layout_digest: str
    oracle_status: str
    barrier_orientation: str | None = None
    safe_gap: tuple[int, int] | None = None


def generate_preview(request: GenerationRequest) -> GeneratedPreview:
    """Generate, render, and close one deterministic environment instance."""
    environment = make_environment(
        request.environment,
        render_mode="rgb_array",
        observation_mode="raw",
    )
    try:
        observation, _ = environment.reset(seed=request.seed)
        frame = environment.render()
        if not isinstance(frame, np.ndarray):
            raise RuntimeError("MiniGrid did not return an RGB array.")
        if frame.ndim != 3 or frame.shape[-1] != 3:
            raise RuntimeError(f"Expected an RGB frame, received shape {frame.shape}.")

        layout_signature = stable_minigrid_layout_signature(environment)
        direction = int(observation["direction"])
        canonical = json.dumps(
            {"layout": layout_signature, "direction": direction},
            sort_keys=True,
            separators=(",", ":"),
        )
        oracle_status = "not-available"
        if request.environment.task_id == "goal_navigation-v1":
            layout = environment.unwrapped.layout
            shortest_goal_actions(layout["agent"], direction, layout["goal"], 7)
            oracle_status = "verified"
        elif request.environment.task_id in {"key_pickup-v1", "key_pickup-v2"}:
            layout = environment.unwrapped.layout
            shortest_key_pickup_actions(
                layout["agent"],
                direction,
                layout["key"],
                7,
            )
            oracle_status = "verified"
        elif request.environment.task_id == "door_key-v1":
            layout = environment.unwrapped.layout
            shortest_door_key_actions(
                request.environment.grid_size,
                layout["agent"],
                direction,
                layout["key"],
                layout["door"],
                layout["goal"],
            )
            oracle_status = "verified"
        elif request.environment.task_id == "lava_navigation-v1":
            layout = environment.unwrapped.layout
            lava = tuple(
                value for name, value in layout.items() if name.startswith("lava_")
            )
            shortest_safe_lava_navigation_actions(
                7, layout["agent"], direction, layout["goal"], lava
            )
            oracle_status = "verified"
        elif request.environment.task_id == "lava_door_key-v1":
            layout = environment.unwrapped.layout
            shortest_lava_door_key_actions(
                7,
                layout["agent"],
                direction,
                layout["key"],
                layout["door"],
                layout["goal"],
                layout["lava"],
            )
            oracle_status = "verified"
        elif request.environment.task_id == "lava_door_key-v2":
            layout = environment.unwrapped.layout
            lava = tuple(layout[f"lava_{index}"] for index in range(4))
            chamber_walls = tuple(
                layout[f"wall_{index}"] for index in range(2)
            )
            shortest_lava_door_key_v2_actions(
                7,
                layout["agent"],
                direction,
                layout["key"],
                layout["door"],
                layout["goal"],
                lava,
                chamber_walls,
            )
            shortest_lava_door_key_v2_failure_actions(
                7,
                layout["agent"],
                direction,
                layout["key"],
                layout["door"],
                layout["goal"],
                lava,
                chamber_walls,
            )
            oracle_status = "verified"
        return GeneratedPreview(
            request=request,
            frame=np.array(frame, copy=True),
            mission=str(observation["mission"]),
            observation_summary=_summarize_observation(observation),
            layout_signature=layout_signature,
            task_id=request.environment.resolved_task_id,
            task_contract_digest=request.environment.task_contract_digest,
            layout_digest=hashlib.sha256(canonical.encode()).hexdigest(),
            oracle_status=oracle_status,
            barrier_orientation=getattr(
                environment.unwrapped,
                "barrier_orientation",
                getattr(environment.unwrapped, "river_orientation", None),
            ),
            safe_gap=getattr(environment.unwrapped, "safe_gap", None),
        )
    finally:
        environment.close()


def _summarize_observation(observation: dict[str, Any]) -> str:
    """Return concise metadata for the current MiniGrid observation."""
    image = np.asarray(observation["image"])
    image_shape = " x ".join(str(dimension) for dimension in image.shape)
    return f"IMAGE {image_shape}; DIRECTION {observation['direction']}"
