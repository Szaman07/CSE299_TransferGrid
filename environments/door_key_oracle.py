"""Shortest symbolic oracle for the single-key, single-door DoorKey task."""

from __future__ import annotations

from collections import deque
from functools import lru_cache

Position = tuple[int, int]
State = tuple[int, int, int, bool, bool, bool]

LEFT, RIGHT, FORWARD, PICKUP, TOGGLE = 0, 1, 2, 3, 5
_VECTORS = ((1, 0), (0, 1), (-1, 0), (0, -1))


@lru_cache(maxsize=None)
def shortest_door_key_actions(
    size: int,
    agent: Position,
    direction: int,
    key: Position,
    door: Position,
    goal: Position,
) -> tuple[int, ...]:
    """Return a deterministic shortest real MiniGrid action sequence."""
    divider_x = size // 2
    start: State = (agent[0], agent[1], direction, False, True, False)
    queue = deque([(start, ())])
    visited = {start}
    while queue:
        state, actions = queue.popleft()
        x, y, facing, carrying, door_locked, door_open = state
        dx, dy = _VECTORS[facing]
        front = (x + dx, y + dy)
        for action in (FORWARD, LEFT, RIGHT, PICKUP, TOGGLE):
            successor: State | None = None
            if action == LEFT:
                successor = (x, y, (facing - 1) % 4, carrying, door_locked, door_open)
            elif action == RIGHT:
                successor = (x, y, (facing + 1) % 4, carrying, door_locked, door_open)
            elif action == PICKUP:
                if not carrying and front == key:
                    successor = (x, y, facing, True, door_locked, door_open)
            elif action == TOGGLE:
                if front == door:
                    if door_locked and carrying:
                        successor = (x, y, facing, carrying, False, True)
                    elif not door_locked:
                        successor = (
                            x, y, facing, carrying, False, not door_open
                        )
            else:
                if not (1 <= front[0] < size - 1 and 1 <= front[1] < size - 1):
                    continue
                if front[0] == divider_x and front != door:
                    continue
                if front == door and not door_open:
                    continue
                if front == goal:
                    return actions + (FORWARD,)
                successor = (
                    front[0], front[1], facing, carrying, door_locked, door_open
                )
            if successor is not None and successor not in visited:
                visited.add(successor)
                queue.append((successor, actions + (action,)))
    raise RuntimeError("DoorKey layout is unreachable under the task contract.")


def door_key_trace(
    size: int,
    agent: Position,
    direction: int,
    key: Position,
    door: Position,
    goal: Position,
) -> tuple[int, ...]:
    """Named boundary for deterministic audit/replay fixture generation."""
    return shortest_door_key_actions(size, agent, direction, key, door, goal)
