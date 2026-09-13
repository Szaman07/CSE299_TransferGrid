"""The first TransferGrid capability environment: object interaction."""

from __future__ import annotations

from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Door, Goal, Key, Wall

from environments.base import ProceduralMiniGridEnv
from environments.procedural import (
    Position,
    add_outer_walls,
    interior_positions,
    is_reachable,
)


class ObjectInteractionEnv(ProceduralMiniGridEnv):
    """Collect a key, unlock a door, and reach a goal in a random layout."""

    MIN_SIZE = 7
    COLOR = "yellow"
    MISSION = "pick up the yellow key, unlock the yellow door, and reach the goal"

    def __init__(
        self,
        size: int = 9,
        max_steps: int | None = None,
        render_mode: str | None = None,
    ) -> None:
        if size < self.MIN_SIZE:
            raise ValueError(f"size must be at least {self.MIN_SIZE}.")

        mission_space = MissionSpace(
            mission_func=lambda: self.MISSION
        )
        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            max_steps=max_steps if max_steps is not None else 4 * size * size,
            see_through_walls=True,
            render_mode=render_mode,
        )

    def _gen_grid(self, width: int, height: int) -> None:
        """Generate a solvable two-room key-and-door layout."""
        self.grid = Grid(width, height)
        add_outer_walls(self.grid, width, height)

        divider_x = width // 2
        door_y = self.choose_layout_integer(2, height - 2)
        door_position = (divider_x, door_y)
        for y in range(1, height - 1):
            if y != door_y:
                self.put_obj(Wall(), divider_x, y)

        left_positions = [
            position
            for position in interior_positions(width, height)
            if position[0] < divider_x
        ]
        right_positions = [
            position
            for position in interior_positions(width, height)
            if position[0] > divider_x
        ]

        key_position = self.choose_position(left_positions)
        agent_position = self.choose_position(left_positions, excluded=(key_position,))
        goal_position = self.choose_position(right_positions)

        self.put_obj(Key(self.COLOR), *key_position)
        self.put_obj(Door(self.COLOR, is_locked=True), *door_position)
        self.put_obj(Goal(), *goal_position)
        self.agent_pos = agent_position
        self.agent_dir = self.choose_layout_integer(0, 4)

        self._validate_layout(
            agent_position=agent_position,
            key_position=key_position,
            door_position=door_position,
            goal_position=goal_position,
            divider_x=divider_x,
        )
        self.record_layout(
            agent=agent_position,
            key=key_position,
            door=door_position,
            goal=goal_position,
        )
        self.mission = self.MISSION

    def step(self, action):
        """Report task completion independently from MiniGrid reward semantics."""
        observation, reward, terminated, truncated, info = super().step(action)
        goal_position = self.layout.get("goal")
        reached_goal = (
            goal_position is not None
            and tuple(int(value) for value in self.agent_pos) == goal_position
        )
        outcome_info = dict(info)
        outcome_info["success"] = bool(terminated and reached_goal)
        outcome_info["timeout"] = bool(truncated)
        return observation, reward, terminated, truncated, outcome_info

    def _validate_layout(
        self,
        *,
        agent_position: Position,
        key_position: Position,
        door_position: Position,
        goal_position: Position,
        divider_x: int,
    ) -> None:
        """Assert that the generated layout supports the intended task sequence."""
        boundary = {
            (x, y)
            for x in range(self.width)
            for y in range(self.height)
            if x in (0, self.width - 1) or y in (0, self.height - 1)
        }
        divider_walls = {
            (divider_x, y)
            for y in range(1, self.height - 1)
            if (divider_x, y) != door_position
        }
        blocked = boundary | divider_walls

        if not is_reachable(
            self.width, self.height, agent_position, key_position, blocked
        ):
            raise RuntimeError("Generated key is unreachable from the agent.")
        if not is_reachable(
            self.width, self.height, key_position, goal_position, blocked
        ):
            raise RuntimeError("Generated goal is unreachable after unlocking the door.")
