"""Draft-only Candidate B: Lava river plus a locked goal chamber."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Door, Goal, Key, Lava, Wall

from environments.base import ProceduralMiniGridEnv
from environments.lava_door_key_v2_oracle import (
    shortest_lava_door_key_v2_actions,
)
from environments.procedural import add_outer_walls

GRID_SIZE = 7
RIVER_COORDINATE = 3
MAX_STEPS = 196
MIN_ORACLE_ACTIONS = 16
COLOR = "yellow"
Position = tuple[int, int]

HORIZONTAL = "horizontal"
VERTICAL = "vertical"
ORIENTATIONS = (HORIZONTAL, VERTICAL)
SOURCE_SIDES = {
    HORIZONTAL: ("north", "south"),
    VERTICAL: ("west", "east"),
}


@dataclass(frozen=True, slots=True)
class LavaDoorKeyV2Layout:
    """One immutable member of the uniformly sampled finite support."""

    orientation: str
    source_side: str
    chamber_corner: Position
    gap: Position
    lava: tuple[Position, ...]
    chamber_walls: tuple[Position, Position]
    door: Position
    goal: Position
    agent: Position
    key: Position
    direction: int
    oracle_length: int

    @property
    def template(self) -> tuple[str, str, Position, Position]:
        """Return the orientation/source/chamber/gap template identity."""
        return (
            self.orientation,
            self.source_side,
            self.chamber_corner,
            self.gap,
        )


def _source_positions(orientation: str, source_side: str) -> tuple[Position, ...]:
    if orientation == HORIZONTAL:
        source_rows = (1, 2) if source_side == "north" else (4, 5)
        return tuple((x, y) for y in source_rows for x in range(1, 6))
    source_columns = (1, 2) if source_side == "west" else (4, 5)
    return tuple((x, y) for y in range(1, 6) for x in source_columns)


def _chamber_corners(
    orientation: str,
    source_side: str,
) -> tuple[Position, Position]:
    if orientation == HORIZONTAL:
        target_y = 5 if source_side == "north" else 1
        return ((1, target_y), (5, target_y))
    target_x = 5 if source_side == "west" else 1
    return ((target_x, 1), (target_x, 5))


def _chamber_geometry(
    orientation: str,
    source_side: str,
    corner: Position,
) -> tuple[Position, tuple[Position, Position]]:
    """Return the sole door and exactly two added walls for a corner chamber."""
    goal_x, goal_y = corner
    if orientation == HORIZONTAL:
        door_x = 2 if goal_x == 1 else 4
        target_y = 5 if source_side == "north" else 1
        inward_y = 4 if target_y == 5 else 2
        door = (door_x, target_y)
        walls = tuple(sorted(((goal_x, inward_y), (door_x, inward_y))))
    else:
        door_y = 2 if goal_y == 1 else 4
        target_x = 5 if source_side == "west" else 1
        inward_x = 4 if target_x == 5 else 2
        door = (target_x, door_y)
        walls = tuple(sorted(((inward_x, goal_y), (inward_x, door_y))))
    return door, walls  # type: ignore[return-value]


def _river_positions(orientation: str) -> tuple[Position, ...]:
    if orientation == HORIZONTAL:
        return tuple((x, RIVER_COORDINATE) for x in range(1, 6))
    return tuple((RIVER_COORDINATE, y) for y in range(1, 6))


def _eligible_gaps(
    orientation: str,
    river: tuple[Position, ...],
    chamber_walls: tuple[Position, Position],
) -> tuple[Position, ...]:
    """Exclude crossings blocked by the chamber's two-cell inward wall strip."""
    if orientation == HORIZONTAL:
        blocked_coordinates = {x for x, _ in chamber_walls}
        return tuple(position for position in river if position[0] not in blocked_coordinates)
    blocked_coordinates = {y for _, y in chamber_walls}
    return tuple(position for position in river if position[1] not in blocked_coordinates)


@lru_cache(maxsize=1)
def valid_lava_door_key_v2_layouts() -> tuple[LavaDoorKeyV2Layout, ...]:
    """Construct and exactly filter the finite support without generation retries."""
    candidates: list[LavaDoorKeyV2Layout] = []
    for orientation in ORIENTATIONS:
        river = _river_positions(orientation)
        for source_side in SOURCE_SIDES[orientation]:
            source_positions = _source_positions(orientation, source_side)
            for corner in _chamber_corners(orientation, source_side):
                door, chamber_walls = _chamber_geometry(
                    orientation, source_side, corner
                )
                for gap in _eligible_gaps(
                    orientation, river, chamber_walls
                ):
                    lava = tuple(position for position in river if position != gap)
                    for agent in source_positions:
                        for key in source_positions:
                            if agent == key:
                                continue
                            for direction in range(4):
                                actions = shortest_lava_door_key_v2_actions(
                                    GRID_SIZE,
                                    agent,
                                    direction,
                                    key,
                                    door,
                                    corner,
                                    lava,
                                    chamber_walls,
                                )
                                if len(actions) < MIN_ORACLE_ACTIONS:
                                    continue
                                candidates.append(
                                    LavaDoorKeyV2Layout(
                                        orientation=orientation,
                                        source_side=source_side,
                                        chamber_corner=corner,
                                        gap=gap,
                                        lava=lava,
                                        chamber_walls=chamber_walls,
                                        door=door,
                                        goal=corner,
                                        agent=agent,
                                        key=key,
                                        direction=direction,
                                        oracle_length=len(actions),
                                    )
                                )
    return tuple(candidates)


class LavaDoorKeyV2Env(ProceduralMiniGridEnv):
    """Acquire a key, cross the Lava river gap, and unlock the goal chamber."""

    MISSION = (
        "pick up the yellow key, cross the safe river gap, unlock the yellow "
        "door, and reach the goal"
    )

    def __init__(
        self,
        size: int = GRID_SIZE,
        max_steps: int | None = None,
        render_mode: str | None = None,
    ) -> None:
        if size != GRID_SIZE:
            raise ValueError("The draft lava_door_key-v2 candidate requires Grid 7.")
        if max_steps is not None and max_steps != MAX_STEPS:
            raise ValueError(
                "The draft lava_door_key-v2 candidate requires max_steps=196."
            )
        self.layout_spec: LavaDoorKeyV2Layout | None = None
        self.river_orientation = HORIZONTAL
        self.source_side = "north"
        self.chamber_corner: Position = (0, 0)
        self.safe_gap: Position = (0, 0)
        self.lava_positions: tuple[Position, ...] = ()
        self.chamber_walls: tuple[Position, Position] = ((0, 0), (0, 0))
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
        support = valid_lava_door_key_v2_layouts()
        layout = support[self.choose_layout_integer(0, len(support))]

        for wall in layout.chamber_walls:
            self.put_obj(Wall(), *wall)
        for lava in layout.lava:
            self.put_obj(Lava(), *lava)
        self.put_obj(Key(COLOR), *layout.key)
        self.put_obj(Door(COLOR, is_locked=True), *layout.door)
        self.put_obj(Goal(), *layout.goal)
        self.agent_pos = layout.agent
        self.agent_dir = layout.direction

        self.layout_spec = layout
        self.river_orientation = layout.orientation
        self.source_side = layout.source_side
        self.chamber_corner = layout.chamber_corner
        self.safe_gap = layout.gap
        self.lava_positions = layout.lava
        self.chamber_walls = layout.chamber_walls
        self.record_layout(
            agent=layout.agent,
            key=layout.key,
            door=layout.door,
            goal=layout.goal,
            gap=layout.gap,
            lava_0=layout.lava[0],
            lava_1=layout.lava[1],
            lava_2=layout.lava[2],
            lava_3=layout.lava[3],
            wall_0=layout.chamber_walls[0],
            wall_1=layout.chamber_walls[1],
        )
        self.mission = self.MISSION

    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)
        position = tuple(int(value) for value in self.agent_pos)
        layout = self.layout_spec
        if layout is None:
            raise RuntimeError("Lava DoorKey v2 layout was not generated.")
        reached_goal = position == layout.goal
        hit_lava = position in layout.lava
        outcome = dict(info)
        outcome["success"] = bool(terminated and reached_goal)
        outcome["timeout"] = bool(truncated)
        outcome["failure_reason"] = "lava" if terminated and hit_lava else None
        return observation, reward, terminated, truncated, outcome
