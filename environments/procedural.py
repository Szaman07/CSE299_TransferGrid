"""Small helpers shared by procedural MiniGrid layouts."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable

from minigrid.core.grid import Grid

Position = tuple[int, int]


def interior_positions(width: int, height: int) -> list[Position]:
    """Return all positions inside a grid's outer wall."""
    return [
        (x, y)
        for y in range(1, height - 1)
        for x in range(1, width - 1)
    ]


def add_outer_walls(grid: Grid, width: int, height: int) -> None:
    """Add MiniGrid's standard outer wall boundary."""
    grid.wall_rect(0, 0, width, height)


def is_reachable(
    width: int,
    height: int,
    start: Position,
    target: Position,
    blocked: Iterable[Position],
) -> bool:
    """Return whether a four-directional path exists between two cells."""
    blocked_positions = set(blocked)

    if start in blocked_positions or target in blocked_positions:
        return False

    queue: deque[Position] = deque([start])
    visited = {start}

    while queue:
        current_x, current_y = queue.popleft()
        if (current_x, current_y) == target:
            return True

        for next_position in (
            (current_x + 1, current_y),
            (current_x - 1, current_y),
            (current_x, current_y + 1),
            (current_x, current_y - 1),
        ):
            next_x, next_y = next_position
            if not (0 <= next_x < width and 0 <= next_y < height):
                continue
            if next_position in blocked_positions or next_position in visited:
                continue
            visited.add(next_position)
            queue.append(next_position)

    return False
