"""Exact product-state oracle for the draft Lava DoorKey v2 candidate."""

from __future__ import annotations

from collections import deque
from functools import lru_cache

Position = tuple[int, int]
LEFT, RIGHT, FORWARD, PICKUP, DROP, TOGGLE, DONE = range(7)
_VECTORS = ((1, 0), (0, 1), (-1, 0), (0, -1))

# x, y, direction, key_x, key_y, carrying, door_locked, door_open
State = tuple[int, int, int, int, int, bool, bool, bool]
Terminal = str


@lru_cache(maxsize=None)
def _blocked_static(
    size: int,
    chamber_walls: tuple[Position, Position],
) -> frozenset[Position]:
    boundary = {
        (x, y)
        for x in range(size)
        for y in range(size)
        if x in (0, size - 1) or y in (0, size - 1)
    }
    return frozenset(boundary | set(chamber_walls))


def lava_door_key_v2_transition(
    state: State,
    action: int,
    *,
    size: int,
    door: Position,
    goal: Position,
    lava: tuple[Position, ...],
    chamber_walls: tuple[Position, Position],
) -> State | Terminal:
    """Apply one exact MiniGrid Discrete(7) action to a candidate state."""
    if action not in range(7):
        raise ValueError(f"Unsupported MiniGrid action index: {action}")

    x, y, facing, key_x, key_y, carrying, door_locked, door_open = state
    dx, dy = _VECTORS[facing]
    front = (x + dx, y + dy)
    key_position = None if key_x < 0 else (key_x, key_y)
    blocked = _blocked_static(size, chamber_walls)
    lava_positions = frozenset(lava)

    if action == LEFT:
        return (
            x,
            y,
            (facing - 1) % 4,
            key_x,
            key_y,
            carrying,
            door_locked,
            door_open,
        )
    if action == RIGHT:
        return (
            x,
            y,
            (facing + 1) % 4,
            key_x,
            key_y,
            carrying,
            door_locked,
            door_open,
        )
    if action == FORWARD and front not in blocked and front != key_position:
        if front == door and not door_open:
            return state
        if front in lava_positions:
            return "lava"
        if front == goal:
            return "success"
        return (
            front[0],
            front[1],
            facing,
            key_x,
            key_y,
            carrying,
            door_locked,
            door_open,
        )
    if action == PICKUP and not carrying and front == key_position:
        return (x, y, facing, -1, -1, True, door_locked, door_open)
    if action == DROP and carrying:
        front_is_empty = (
            front not in blocked
            and front not in lava_positions
            and front not in (door, goal)
            and key_position is None
        )
        if front_is_empty:
            return (
                x,
                y,
                facing,
                front[0],
                front[1],
                False,
                door_locked,
                door_open,
            )
    if action == TOGGLE and front == door:
        if door_locked and carrying:
            return (x, y, facing, key_x, key_y, carrying, False, True)
        if not door_locked:
            return (
                x,
                y,
                facing,
                key_x,
                key_y,
                carrying,
                False,
                not door_open,
            )

    # Blocked forward/pickup/drop/toggle and DONE are exact no-ops.
    return state


def _search(
    *,
    size: int,
    agent: Position,
    direction: int,
    key: Position,
    door: Position,
    goal: Position,
    lava: tuple[Position, ...],
    chamber_walls: tuple[Position, Position],
    terminal: Terminal,
) -> tuple[int, ...]:
    start: State = (
        agent[0],
        agent[1],
        direction,
        key[0],
        key[1],
        False,
        True,
        False,
    )
    queue: deque[State] = deque([start])
    predecessor: dict[State, tuple[State, int] | None] = {start: None}

    while queue:
        state = queue.popleft()
        for action in range(7):
            successor = lava_door_key_v2_transition(
                state,
                action,
                size=size,
                door=door,
                goal=goal,
                lava=lava,
                chamber_walls=chamber_walls,
            )
            if successor == terminal:
                actions = [action]
                cursor = state
                while predecessor[cursor] is not None:
                    previous, previous_action = predecessor[cursor]
                    actions.append(previous_action)
                    cursor = previous
                actions.reverse()
                return tuple(actions)
            if isinstance(successor, tuple) and successor not in predecessor:
                predecessor[successor] = (state, action)
                queue.append(successor)

    raise RuntimeError(f"No {terminal} trace exists for Lava DoorKey v2.")


@lru_cache(maxsize=None)
def shortest_lava_door_key_v2_actions(
    size: int,
    agent: Position,
    direction: int,
    key: Position,
    door: Position,
    goal: Position,
    lava: tuple[Position, ...],
    chamber_walls: tuple[Position, Position],
) -> tuple[int, ...]:
    """Return a shortest successful sequence under exact MiniGrid mechanics."""
    return _search(
        size=size,
        agent=agent,
        direction=direction,
        key=key,
        door=door,
        goal=goal,
        lava=lava,
        chamber_walls=chamber_walls,
        terminal="success",
    )


@lru_cache(maxsize=None)
def shortest_lava_door_key_v2_lava_actions(
    size: int,
    agent: Position,
    direction: int,
    key: Position,
    door: Position,
    goal: Position,
    lava: tuple[Position, ...],
    chamber_walls: tuple[Position, Position],
) -> tuple[int, ...]:
    """Return a deterministic shortest sequence whose last step enters Lava."""
    return _search(
        size=size,
        agent=agent,
        direction=direction,
        key=key,
        door=door,
        goal=goal,
        lava=lava,
        chamber_walls=chamber_walls,
        terminal="lava",
    )


# A descriptive alias matching the v1 oracle's public failure helper.
shortest_lava_door_key_v2_failure_actions = (
    shortest_lava_door_key_v2_lava_actions
)
