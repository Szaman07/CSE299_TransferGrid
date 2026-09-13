"""Draft-only Lava DoorKey release candidate."""

from __future__ import annotations

from functools import lru_cache

from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Door, Goal, Key, Lava, Wall

from environments.base import ProceduralMiniGridEnv
from environments.lava_door_key_oracle import shortest_lava_door_key_actions
from environments.procedural import add_outer_walls

GRID_SIZE = 7
MAX_STEPS = 196
MIN_ORACLE_ACTIONS = 14
COLOR = "yellow"
Position = tuple[int, int]
# door, barrier row, gap, lava, key, agent, goal, direction
LayoutTuple = tuple[Position, int, Position, Position, Position, Position, Position, int]


@lru_cache(maxsize=1)
def valid_lava_door_key_layouts() -> tuple[LayoutTuple, ...]:
    """Enumerate the finite candidate support with no hidden retries."""
    divider_x = GRID_SIZE // 2
    # The bounded release candidate deliberately uses the four left-room corner
    # placements.  This keeps the finite support quick to enumerate while still
    # varying both key/agent sides and every initial direction.
    left = ((1, 1), (1, 5), (2, 1), (2, 5))
    candidates: list[LayoutTuple] = []
    for door_y in (2, 3, 4):
        door = (divider_x, door_y)
        for barrier_y in (2, 3, 4):
            if barrier_y == door_y:
                continue
            entry_above = door_y < barrier_y
            goal_positions = tuple(
                (x, y)
                for x in (4, 5)
                for y in range(1, 6)
                if (y > barrier_y if entry_above else y < barrier_y)
            )
            for gap_x in (4, 5):
                gap = (gap_x, barrier_y)
                lava = (9 - gap_x, barrier_y)
                for key in left:
                    for agent in left:
                        if agent == key:
                            continue
                        for goal in goal_positions:
                            for direction in range(4):
                                actions = shortest_lava_door_key_actions(
                                    GRID_SIZE, agent, direction, key, door, goal, lava
                                )
                                if len(actions) >= MIN_ORACLE_ACTIONS:
                                    candidates.append((
                                        door, barrier_y, gap, lava, key, agent, goal, direction
                                    ))
    return tuple(candidates)


class LavaDoorKeyEnv(ProceduralMiniGridEnv):
    """Acquire a key, unlock a door, cross a safe gap, and reach the goal."""

    MISSION = (
        "pick up the yellow key, unlock the yellow door, cross the safe gap, "
        "and reach the goal"
    )

    def __init__(
        self,
        size: int = GRID_SIZE,
        max_steps: int | None = None,
        render_mode: str | None = None,
    ) -> None:
        if size != GRID_SIZE:
            raise ValueError("lava_door_key-v1 requires an exact 7 x 7 grid.")
        if max_steps is not None and max_steps != MAX_STEPS:
            raise ValueError("lava_door_key-v1 requires max_steps=196.")
        self.barrier_orientation = "horizontal"
        self.barrier_row = 0
        self.safe_gap: Position = (0, 0)
        super().__init__(
            mission_space=MissionSpace(mission_func=lambda: self.MISSION),
            grid_size=size,
            max_steps=MAX_STEPS,
            see_through_walls=True,
            render_mode=render_mode,
        )

    def _gen_grid(self, width: int, height: int) -> None:
        self.grid = Grid(width, height)
        add_outer_walls(self.grid, width, height)
        door, barrier_row, gap, lava, key, agent, goal, direction = (
            valid_lava_door_key_layouts()[
                self.choose_layout_integer(0, len(valid_lava_door_key_layouts()))
            ]
        )
        divider_x = width // 2
        for y in range(1, height - 1):
            if (divider_x, y) != door:
                self.put_obj(Wall(), divider_x, y)
        self.put_obj(Key(COLOR), *key)
        self.put_obj(Door(COLOR, is_locked=True), *door)
        self.put_obj(Lava(), *lava)
        self.put_obj(Goal(), *goal)
        self.agent_pos = agent
        self.agent_dir = direction
        self.barrier_row = barrier_row
        self.safe_gap = gap
        self.record_layout(
            agent=agent, key=key, door=door, lava=lava, gap=gap, goal=goal
        )
        self.mission = self.MISSION

    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)
        position = tuple(int(value) for value in self.agent_pos)
        reached_goal = position == self.layout["goal"]
        hit_lava = position == self.layout["lava"]
        outcome = dict(info)
        outcome["success"] = bool(terminated and reached_goal)
        outcome["timeout"] = bool(truncated)
        outcome["failure_reason"] = "lava" if terminated and hit_lava else None
        return observation, reward, terminated, truncated, outcome
