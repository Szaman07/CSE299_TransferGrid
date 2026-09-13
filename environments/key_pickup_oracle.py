"""Shortest-path oracle for the bounded Key Pickup state space."""

from __future__ import annotations

from collections import deque
from functools import lru_cache

Position = tuple[int, int]
OracleState = tuple[int, int, int]

LEFT = 0
RIGHT = 1
FORWARD = 2
PICKUP = 3
_VECTORS = ((1, 0), (0, 1), (-1, 0), (0, -1))


@lru_cache(maxsize=None)
def shortest_key_pickup_actions(
    agent: Position,
    direction: int,
    key: Position,
    size: int = 7,
) -> tuple[int, ...]:
    """Return a deterministic shortest pose path followed by pickup."""
    start = (agent[0], agent[1], direction)
    queue = deque([(start, ())])
    visited = {start}
    while queue:
        state, actions = queue.popleft()
        x, y, facing = state
        dx, dy = _VECTORS[facing]
        if (x + dx, y + dy) == key:
            return actions + (PICKUP,)
        for action in (FORWARD, LEFT, RIGHT):
            if action == LEFT:
                successor = (x, y, (facing - 1) % 4)
            elif action == RIGHT:
                successor = (x, y, (facing + 1) % 4)
            else:
                nx, ny = x + dx, y + dy
                if (
                    not 1 <= nx < size - 1
                    or not 1 <= ny < size - 1
                    or (nx, ny) == key
                ):
                    continue
                successor = (nx, ny, facing)
            if successor not in visited:
                visited.add(successor)
                queue.append((successor, actions + (action,)))
    raise RuntimeError("Key Pickup layout is unreachable.")


def trace_key_pickup_actions(
    agent: Position,
    direction: int,
    key: Position,
    size: int = 7,
) -> tuple[OracleState, ...]:
    """Return the pose trace induced before the terminal pickup action."""
    states = [(agent[0], agent[1], direction)]
    x, y, facing = states[0]
    for action in shortest_key_pickup_actions(agent, direction, key, size):
        if action == PICKUP:
            states.append((x, y, facing))
        elif action == LEFT:
            facing = (facing - 1) % 4
            states.append((x, y, facing))
        elif action == RIGHT:
            facing = (facing + 1) % 4
            states.append((x, y, facing))
        else:
            dx, dy = _VECTORS[facing]
            x, y = x + dx, y + dy
            states.append((x, y, facing))
    return tuple(states)
