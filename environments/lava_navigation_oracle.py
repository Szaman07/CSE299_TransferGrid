"""Shortest safe and hazard-reaching oracles for Lava Navigation."""

from __future__ import annotations

from collections import deque
from functools import lru_cache

Position = tuple[int, int]
LEFT, RIGHT, FORWARD = 0, 1, 2
_VECTORS = ((1, 0), (0, 1), (-1, 0), (0, -1))


def _search(
    size: int,
    agent: Position,
    direction: int,
    target: Position,
    forbidden: frozenset[Position],
) -> tuple[int, ...]:
    queue = deque([((agent[0], agent[1], direction), ())])
    visited = {(agent[0], agent[1], direction)}
    while queue:
        (x, y, facing), actions = queue.popleft()
        for action in (FORWARD, LEFT, RIGHT):
            if action == LEFT:
                successor = (x, y, (facing - 1) % 4)
            elif action == RIGHT:
                successor = (x, y, (facing + 1) % 4)
            else:
                dx, dy = _VECTORS[facing]
                position = (x + dx, y + dy)
                if not (1 <= position[0] < size - 1 and 1 <= position[1] < size - 1):
                    continue
                if position == target:
                    return actions + (FORWARD,)
                if position in forbidden:
                    continue
                successor = (position[0], position[1], facing)
            if successor not in visited:
                visited.add(successor)
                queue.append((successor, actions + (action,)))
    raise RuntimeError("No oracle path exists for the Lava Navigation layout.")


@lru_cache(maxsize=None)
def shortest_safe_lava_navigation_actions(
    size: int,
    agent: Position,
    direction: int,
    goal: Position,
    lava: tuple[Position, ...],
) -> tuple[int, ...]:
    """Return the shortest pose-space path that never enters Lava."""
    return _search(size, agent, direction, goal, frozenset(lava))


@lru_cache(maxsize=None)
def shortest_lava_failure_actions(
    size: int,
    agent: Position,
    direction: int,
    lava: tuple[Position, ...],
) -> tuple[int, ...]:
    """Return a deterministic shortest trace whose final forward step enters Lava."""
    candidates = [
        _search(size, agent, direction, target, frozenset(lava) - {target})
        for target in lava
    ]
    return min(candidates, key=lambda actions: (len(actions), actions))
