"""Shortest-path oracle for the bounded Goal Navigation state space."""

from __future__ import annotations

from collections import deque
from functools import lru_cache

Position = tuple[int, int]
OracleState = tuple[int, int, int]

LEFT = 0
RIGHT = 1
FORWARD = 2
_VECTORS = ((1, 0), (0, 1), (-1, 0), (0, -1))


@lru_cache(maxsize=None)
def shortest_goal_actions(
    agent: Position,
    direction: int,
    goal: Position,
    size: int = 7,
) -> tuple[int, ...]:
    """Return a deterministic shortest left/right/forward action sequence."""
    start = (agent[0], agent[1], direction)
    queue = deque([(start, ())])
    visited = {start}
    while queue:
        state, actions = queue.popleft()
        x, y, facing = state
        for action in (FORWARD, LEFT, RIGHT):
            if action == LEFT:
                successor = (x, y, (facing - 1) % 4)
            elif action == RIGHT:
                successor = (x, y, (facing + 1) % 4)
            else:
                dx, dy = _VECTORS[facing]
                nx, ny = x + dx, y + dy
                if not (1 <= nx < size - 1 and 1 <= ny < size - 1):
                    continue
                if (nx, ny) == goal:
                    return actions + (FORWARD,)
                successor = (nx, ny, facing)
            if successor not in visited:
                visited.add(successor)
                queue.append((successor, actions + (action,)))
    raise RuntimeError("Goal Navigation layout is unreachable.")


def trace_goal_actions(
    agent: Position,
    direction: int,
    goal: Position,
    size: int = 7,
) -> tuple[OracleState, ...]:
    """Return the deterministic state trace induced by the shortest oracle."""
    states = [(agent[0], agent[1], direction)]
    x, y, facing = states[0]
    for action in shortest_goal_actions(agent, direction, goal, size):
        if action == LEFT:
            facing = (facing - 1) % 4
        elif action == RIGHT:
            facing = (facing + 1) % 4
        else:
            dx, dy = _VECTORS[facing]
            x, y = x + dx, y + dy
        states.append((x, y, facing))
    return tuple(states)
