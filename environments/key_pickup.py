"""Implementation of the frozen Key Pickup task contract."""

from __future__ import annotations

from functools import lru_cache

from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Key

from environments.base import ProceduralMiniGridEnv
from environments.key_pickup_oracle import shortest_key_pickup_actions
from environments.procedural import add_outer_walls, interior_positions

KEY_PICKUP_GRID_SIZE = 7
KEY_PICKUP_MAX_STEPS = 196
KEY_PICKUP_MIN_ORACLE_ACTIONS = 6

LayoutTuple = tuple[tuple[int, int], tuple[int, int], int]


@lru_cache(maxsize=1)
def valid_key_pickup_layouts() -> tuple[LayoutTuple, ...]:
    """Enumerate the complete valid support in canonical lexicographic order."""
    positions = tuple(
        interior_positions(KEY_PICKUP_GRID_SIZE, KEY_PICKUP_GRID_SIZE)
    )
    return tuple(
        (agent, key, direction)
        for agent in positions
        for key in positions
        if key != agent
        for direction in range(4)
        if len(
            shortest_key_pickup_actions(
                agent,
                direction,
                key,
                KEY_PICKUP_GRID_SIZE,
            )
        )
        >= KEY_PICKUP_MIN_ORACLE_ACTIONS
    )


class KeyPickupEnv(ProceduralMiniGridEnv):
    """Pick up one yellow key in an otherwise empty 7 x 7 room."""

    MISSION = "pick up the yellow key"
    MIN_SIZE = KEY_PICKUP_GRID_SIZE

    def __init__(
        self,
        size: int = KEY_PICKUP_GRID_SIZE,
        max_steps: int | None = None,
        render_mode: str | None = None,
    ) -> None:
        if size != KEY_PICKUP_GRID_SIZE:
            raise ValueError("key_pickup-v1 requires an exact 7 x 7 grid.")
        if max_steps is not None and max_steps != KEY_PICKUP_MAX_STEPS:
            raise ValueError("key_pickup-v1 requires max_steps=196.")
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
        agent, key, direction = support[
            self.choose_layout_integer(0, len(support))
        ]
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
