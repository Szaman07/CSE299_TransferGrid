"""Product-state oracle for the bounded Lava DoorKey candidate."""

from __future__ import annotations

from collections import deque
from functools import lru_cache

Position = tuple[int, int]
LEFT, RIGHT, FORWARD, PICKUP, DROP, TOGGLE, DONE = range(7)
_VECTORS = ((1, 0), (0, 1), (-1, 0), (0, -1))

# x, y, direction, key_x, key_y, carrying, door_locked, door_open
State = tuple[int, int, int, int, int, bool, bool, bool]


@lru_cache(maxsize=None)
def _blocked_static(size: int, door: Position) -> frozenset[Position]:
    divider = size // 2
    boundary = {
        (x, y)
        for x in range(size)
        for y in range(size)
        if x in (0, size - 1) or y in (0, size - 1)
    }
    divider_walls = {
        (divider, y) for y in range(1, size - 1) if (divider, y) != door
    }
    return frozenset(boundary | divider_walls)


def lava_door_key_transition(
    state: State,
    action: int,
    *,
    size: int,
    door: Position,
    goal: Position,
    lava: Position,
) -> State | str:
    """Apply one exact MiniGrid Discrete(7) action to the product state."""
    if action not in range(7):
        raise ValueError(f"Unsupported MiniGrid action index: {action}")
    x, y, facing, key_x, key_y, carrying, door_locked, door_open = state
    dx, dy = _VECTORS[facing]
    front = (x + dx, y + dy)
    key_position = None if key_x < 0 else (key_x, key_y)
    blocked = _blocked_static(size, door)
    unchanged = state
    if action == LEFT:
        return (x, y, (facing - 1) % 4, key_x, key_y, carrying, door_locked, door_open)
    if action == RIGHT:
        return (x, y, (facing + 1) % 4, key_x, key_y, carrying, door_locked, door_open)
    if action == FORWARD and front not in blocked and front != key_position:
        if front == door and not door_open:
            return unchanged
        elif front == lava:
            return "lava"
        elif front == goal:
            return "success"
        return (
            front[0], front[1], facing, key_x, key_y,
            carrying, door_locked, door_open,
        )
    if action == PICKUP and not carrying and front == key_position:
        return (x, y, facing, -1, -1, True, door_locked, door_open)
    if action == DROP and carrying:
        front_is_empty = (
            front not in blocked
            and front not in (door, goal, lava)
            and key_position is None
        )
        if front_is_empty:
            return (
                x, y, facing, front[0], front[1], False,
                door_locked, door_open,
            )
    if action == TOGGLE and front == door:
        if door_locked and carrying:
            return (x, y, facing, key_x, key_y, carrying, False, True)
        elif not door_locked:
            return (
                x, y, facing, key_x, key_y,
                carrying, False, not door_open,
            )
    # Blocked forward/pickup/drop/toggle and DONE are exact no-ops.
    return unchanged


def _search(
    *,
    size: int,
    agent: Position,
    direction: int,
    key: Position,
    door: Position,
    goal: Position,
    lava: Position,
    terminal: str,
) -> tuple[int, ...]:
    start: State = (agent[0], agent[1], direction, key[0], key[1], False, True, False)
    queue = deque([(start, ())])
    visited = {start}
    while queue:
        state, actions = queue.popleft()
        for action in range(7):
            successor = lava_door_key_transition(
                state, action, size=size, door=door, goal=goal, lava=lava
            )
            if successor == terminal:
                return actions + (action,)
            if isinstance(successor, tuple) and successor not in visited:
                visited.add(successor)
                queue.append((successor, actions + (action,)))
    raise RuntimeError(f"No {terminal} trace exists for Lava DoorKey.")


@lru_cache(maxsize=None)
def shortest_lava_door_key_actions(
    size: int,
    agent: Position,
    direction: int,
    key: Position,
    door: Position,
    goal: Position,
    lava: Position,
) -> tuple[int, ...]:
    """Return a shortest successful sequence under exact MiniGrid mechanics."""
    return _search(
        size=size, agent=agent, direction=direction, key=key, door=door,
        goal=goal, lava=lava, terminal="success",
    )


@lru_cache(maxsize=None)
def shortest_lava_door_key_failure_actions(
    size: int,
    agent: Position,
    direction: int,
    key: Position,
    door: Position,
    goal: Position,
    lava: Position,
) -> tuple[int, ...]:
    """Return a deterministic sequence whose final step enters Lava."""
    return _search(
        size=size, agent=agent, direction=direction, key=key, door=door,
        goal=goal, lava=lava, terminal="lava",
    )
