"""Base class for TransferGrid procedural MiniGrid environments."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from gymnasium.core import ObsType
from minigrid.minigrid_env import MiniGridEnv

from environments.procedural import Position

_FIXED_LAYOUT_SEED_OPTION = "_transfergrid_fixed_layout_seed"


class ProceduralMiniGridEnv(MiniGridEnv):
    """MiniGrid base with deterministic layout helpers and metadata."""

    def __init__(self, *args, **kwargs) -> None:
        self._layout_random: np.random.Generator | None = None
        super().__init__(*args, **kwargs)
        self._layout: dict[str, Position] = {}

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, object] | None = None,
    ) -> tuple[ObsType, dict[str, object]]:
        """Reset normal RNG state while optionally fixing only layout generation."""
        reset_options = dict(options) if options is not None else None
        layout_seed = (
            reset_options.pop(_FIXED_LAYOUT_SEED_OPTION, None)
            if reset_options is not None
            else None
        )
        self._layout_random = (
            np.random.default_rng(layout_seed) if layout_seed is not None else None
        )
        try:
            return super().reset(seed=seed, options=reset_options)
        finally:
            self._layout_random = None

    @property
    def layout(self) -> dict[str, Position]:
        """Return a copy of the spatial positions generated for this episode."""
        return dict(self._layout)

    def choose_layout_integer(self, low: int, high: int) -> int:
        """Draw from the active layout RNG without consuming normal episode RNG."""
        if self._layout_random is None:
            return int(self._rand_int(low, high))
        return int(self._layout_random.integers(low, high))

    def choose_position(
        self,
        candidates: Iterable[Position],
        *,
        excluded: Iterable[Position] = (),
    ) -> Position:
        """Choose one valid position using the active layout RNG."""
        excluded_positions = set(excluded)
        available = [
            position for position in candidates if position not in excluded_positions
        ]
        if not available:
            raise ValueError("No valid positions are available for procedural placement.")
        return available[self.choose_layout_integer(0, len(available))]

    def record_layout(self, **positions: Position) -> None:
        """Store generated positions for debugging and deterministic verification."""
        self._layout = dict(positions)

    def layout_signature(self) -> tuple[tuple[str, Position], ...]:
        """Return stable spatial positions, excluding agent direction and dynamics."""
        return tuple(sorted(self._layout.items()))
