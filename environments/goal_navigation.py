"""Draft implementation of the frozen Goal Navigation task contract."""

from __future__ import annotations

from functools import lru_cache

from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Goal

from environments.base import ProceduralMiniGridEnv
from environments.goal_navigation_oracle import shortest_goal_actions
from environments.procedural import add_outer_walls, interior_positions

GOAL_NAVIGATION_GRID_SIZE = 7
GOAL_NAVIGATION_MAX_STEPS = 196
GOAL_NAVIGATION_MIN_ORACLE_ACTIONS = 6

LayoutTuple = tuple[tuple[int, int], tuple[int, int], int]


@lru_cache(maxsize=1)
def valid_goal_navigation_layouts() -> tuple[LayoutTuple, ...]:
    """Enumerate the complete valid support in canonical lexicographic order."""
    positions = tuple(interior_positions(
        GOAL_NAVIGATION_GRID_SIZE, GOAL_NAVIGATION_GRID_SIZE
    ))
    return tuple(
        (agent, goal, direction)
        for agent in positions
        for goal in positions
        if goal != agent
        for direction in range(4)
        if len(shortest_goal_actions(
            agent, direction, goal, GOAL_NAVIGATION_GRID_SIZE
        )) >= GOAL_NAVIGATION_MIN_ORACLE_ACTIONS
    )


class GoalNavigationEnv(ProceduralMiniGridEnv):
    """Reach one green goal in an otherwise empty 7 x 7 room."""

    MISSION = "reach the green goal"
    MIN_SIZE = GOAL_NAVIGATION_GRID_SIZE

    def __init__(
        self,
        size: int = GOAL_NAVIGATION_GRID_SIZE,
        max_steps: int | None = None,
        render_mode: str | None = None,
    ) -> None:
        if size != GOAL_NAVIGATION_GRID_SIZE:
            raise ValueError("goal_navigation-v1 requires an exact 7 x 7 grid.")
        if max_steps is not None and max_steps != GOAL_NAVIGATION_MAX_STEPS:
            raise ValueError("goal_navigation-v1 requires max_steps=196.")
        super().__init__(
            mission_space=MissionSpace(mission_func=lambda: self.MISSION),
            grid_size=size,
            max_steps=GOAL_NAVIGATION_MAX_STEPS,
            see_through_walls=True,
            render_mode=render_mode,
        )

    def _gen_grid(self, width: int, height: int) -> None:
        self.grid = Grid(width, height)
        add_outer_walls(self.grid, width, height)
        support = valid_goal_navigation_layouts()
        agent, goal, direction = support[
            self.choose_layout_integer(0, len(support))
        ]
        self.put_obj(Goal(), *goal)
        self.agent_pos = agent
        self.agent_dir = direction
        self.record_layout(agent=agent, goal=goal)
        self.mission = self.MISSION

    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)
        goal = self.layout["goal"]
        reached_goal = tuple(int(value) for value in self.agent_pos) == goal
        outcome = dict(info)
        outcome["success"] = bool(terminated and reached_goal)
        outcome["timeout"] = bool(truncated)
        outcome["failure_reason"] = None
        return observation, reward, terminated, truncated, outcome
