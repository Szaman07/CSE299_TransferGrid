"""Key Pickup v2 candidate with a decoupled seed-to-layout mapping."""

from __future__ import annotations

from functools import lru_cache

from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Key

from environments.base import ProceduralMiniGridEnv
from environments.key_pickup import (
    KEY_PICKUP_GRID_SIZE,
    KEY_PICKUP_MAX_STEPS,
    valid_key_pickup_layouts,
)
from environments.procedural import add_outer_walls


KEY_PICKUP_V2_STRATUM_OFFSET = 97


@lru_cache(maxsize=1)
def key_pickup_v2_index_permutation() -> tuple[int, ...]:
    """Map every v1 support index to a different index with the same direction."""
    support = valid_key_pickup_layouts()
    direction_indices = {
        direction: tuple(
            index for index, item in enumerate(support) if item[2] == direction
        )
        for direction in range(4)
    }
    assert all(len(indices) == 195 for indices in direction_indices.values())
    ranks = {
        index: rank
        for indices in direction_indices.values()
        for rank, index in enumerate(indices)
    }
    return tuple(
        direction_indices[support[index][2]][
            (ranks[index] + KEY_PICKUP_V2_STRATUM_OFFSET) % 195
        ]
        for index in range(len(support))
    )


class KeyPickupV2Env(ProceduralMiniGridEnv):
    """Pick up a key using the v1 support with a new deterministic seed map."""

    MISSION = "pick up the yellow key"
    MIN_SIZE = KEY_PICKUP_GRID_SIZE

    def __init__(
        self,
        size: int = KEY_PICKUP_GRID_SIZE,
        max_steps: int | None = None,
        render_mode: str | None = None,
    ) -> None:
        if size != KEY_PICKUP_GRID_SIZE:
            raise ValueError("key_pickup-v2 requires an exact 7 x 7 grid.")
        if max_steps is not None and max_steps != KEY_PICKUP_MAX_STEPS:
            raise ValueError("key_pickup-v2 requires max_steps=196.")
        super().__init__(
            mission_space=MissionSpace(mission_func=lambda: self.MISSION),
            grid_size=size,
            max_steps=KEY_PICKUP_MAX_STEPS,
            see_through_walls=True,
            render_mode=render_mode,
        )

    def _gen_grid(self, width: int, height: int) -> None:
        self.grid = Grid(width, height)
        add_outer_walls(self.grid, width, height)
        support = valid_key_pickup_layouts()
        sampled_index = self.choose_layout_integer(0, len(support))
        permuted_index = key_pickup_v2_index_permutation()[sampled_index]
        agent, key, direction = support[permuted_index]
        self.put_obj(Key("yellow"), *key)
        self.agent_pos = agent
        self.agent_dir = direction
        self.record_layout(agent=agent, key=key)
        self.mission = self.MISSION

    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)
        carrying = self.carrying
        success = bool(
            carrying is not None
            and carrying.type == "key"
            and carrying.color == "yellow"
        )
        if success and not terminated:
            terminated = True
            truncated = False
            reward = self._reward()
        outcome = dict(info)
        outcome["success"] = success
        outcome["timeout"] = bool(truncated)
        outcome["failure_reason"] = None
        return observation, reward, terminated, truncated, outcome
