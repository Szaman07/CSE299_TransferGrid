"""Draft implementation of the frozen Lava Navigation candidate contract."""

from __future__ import annotations

from functools import lru_cache

from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Goal, Lava

from environments.base import ProceduralMiniGridEnv
from environments.lava_navigation_oracle import (
    shortest_safe_lava_navigation_actions,
)
from environments.procedural import add_outer_walls, interior_positions

LAVA_NAVIGATION_GRID_SIZE = 7
LAVA_NAVIGATION_MAX_STEPS = 196
LAVA_NAVIGATION_MIN_ORACLE_ACTIONS = 8
Orientation = str
Position = tuple[int, int]
LayoutTuple = tuple[Orientation, int, Position, Position, Position, int]


def _barrier_positions(orientation: Orientation, coordinate: int) -> tuple[Position, ...]:
    return (
        tuple((coordinate, index) for index in range(1, 6))
        if orientation == "vertical"
        else tuple((index, coordinate) for index in range(1, 6))
    )


@lru_cache(maxsize=1)
def valid_lava_navigation_layouts() -> tuple[LayoutTuple, ...]:
    """Enumerate every valid tuple, then freeze canonical ordering."""
    positions = tuple(interior_positions(7, 7))
    candidates: list[LayoutTuple] = []
    for orientation in ("horizontal", "vertical"):
        for coordinate in (2, 3, 4):
            barrier = _barrier_positions(orientation, coordinate)
            for gap in barrier:
                lava = tuple(position for position in barrier if position != gap)
                negative = tuple(
                    position for position in positions
                    if (
                        position[1] < coordinate
                        if orientation == "horizontal"
                        else position[0] < coordinate
                    )
                )
                positive = tuple(
                    position for position in positions
                    if (
                        position[1] > coordinate
                        if orientation == "horizontal"
                        else position[0] > coordinate
                    )
                )
                for agent_side, goal_side in ((negative, positive), (positive, negative)):
                    for agent in agent_side:
                        for goal in goal_side:
                            for direction in range(4):
                                actions = shortest_safe_lava_navigation_actions(
                                    7, agent, direction, goal, lava
                                )
                                if len(actions) >= LAVA_NAVIGATION_MIN_ORACLE_ACTIONS:
                                    candidates.append(
                                        (orientation, coordinate, gap, agent, goal, direction)
                                    )
    return tuple(candidates)


class LavaNavigationEnv(ProceduralMiniGridEnv):
    """Cross one four-tile Lava barrier through its single safe gap."""

    MISSION = "cross the lava barrier through the safe gap and reach the goal"

    def __init__(
        self,
        size: int = LAVA_NAVIGATION_GRID_SIZE,
        max_steps: int | None = None,
        render_mode: str | None = None,
    ) -> None:
        if size != LAVA_NAVIGATION_GRID_SIZE:
            raise ValueError("lava_navigation-v1 requires an exact 7 x 7 grid.")
        if max_steps is not None and max_steps != LAVA_NAVIGATION_MAX_STEPS:
            raise ValueError("lava_navigation-v1 requires max_steps=196.")
        self.barrier_orientation = ""
        self.barrier_coordinate = 0
        self.safe_gap: Position = (0, 0)
        super().__init__(
            mission_space=MissionSpace(mission_func=lambda: self.MISSION),
            grid_size=size,
            max_steps=LAVA_NAVIGATION_MAX_STEPS,
            see_through_walls=True,
            render_mode=render_mode,
        )

    def _gen_grid(self, width: int, height: int) -> None:
        self.grid = Grid(width, height)
        add_outer_walls(self.grid, width, height)
        support = valid_lava_navigation_layouts()
        orientation, coordinate, gap, agent, goal, direction = support[
            self.choose_layout_integer(0, len(support))
        ]
        barrier = _barrier_positions(orientation, coordinate)
        lava = tuple(position for position in barrier if position != gap)
        for position in lava:
            self.put_obj(Lava(), *position)
        self.put_obj(Goal(), *goal)
        self.agent_pos = agent
        self.agent_dir = direction
        self.barrier_orientation = orientation
        self.barrier_coordinate = coordinate
        self.safe_gap = gap
        self.record_layout(
            agent=agent,
            goal=goal,
            gap=gap,
            **{f"lava_{index}": position for index, position in enumerate(lava)},
        )
        self.mission = self.MISSION

    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)
        position = tuple(int(value) for value in self.agent_pos)
        reached_goal = position == self.layout["goal"]
        lava_positions = {
            value for name, value in self.layout.items() if name.startswith("lava_")
        }
        hit_lava = position in lava_positions
        outcome = dict(info)
        outcome["success"] = bool(terminated and reached_goal)
        outcome["timeout"] = bool(truncated)
        outcome["failure_reason"] = "lava" if terminated and hit_lava else None
        return observation, reward, terminated, truncated, outcome
