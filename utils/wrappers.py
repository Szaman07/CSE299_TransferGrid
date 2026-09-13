"""Gymnasium wrappers shared by TransferGrid environments."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np

MAX_LAYOUT_SEED = 2_147_483_647
_FIXED_LAYOUT_SEED_OPTION = "_transfergrid_fixed_layout_seed"


def validate_layout_seed(layout_seed: int) -> None:
    """Validate a fixed-layout seed before any environment resources are created."""
    if type(layout_seed) is not int or not 0 <= layout_seed <= MAX_LAYOUT_SEED:
        raise ValueError(
            f"layout_seed must be an integer from 0 to {MAX_LAYOUT_SEED}."
        )


class FlatMiniGridObservation(gym.ObservationWrapper, gym.utils.RecordConstructorArgs):
    """Flatten MiniGrid image and direction observations for SB3 MLP policies."""

    def __init__(self, env: gym.Env) -> None:
        gym.utils.RecordConstructorArgs.__init__(self)
        super().__init__(env)
        if not isinstance(env.observation_space, gym.spaces.Dict):
            raise TypeError("FlatMiniGridObservation requires a Dict observation space.")

        image_space = env.observation_space["image"]
        if not isinstance(image_space, gym.spaces.Box):
            raise TypeError("The MiniGrid image observation must be a Box space.")

        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(int(np.prod(image_space.shape)) + 1,),
            dtype=np.uint8,
        )

    def observation(self, observation: dict[str, Any]) -> np.ndarray:
        """Return image pixels followed by the agent direction."""
        image = np.asarray(observation["image"], dtype=np.uint8).reshape(-1)
        direction = np.asarray([observation["direction"]], dtype=np.uint8)
        return np.concatenate((image, direction))


def stable_minigrid_layout_signature(
    environment: gym.Env,
) -> tuple[tuple[str, tuple[int, int]], ...]:
    """Return a deterministic reset-state signature for any MiniGrid task.

    TransferGrid environments expose their own concise signature.  Official
    upstream MiniGrid tasks do not, so their grid encoding is recorded without
    changing the environment's dynamics.
    """
    base = environment.unwrapped
    native = getattr(base, "layout_signature", None)
    if callable(native):
        return tuple(native())

    grid = getattr(base, "grid", None)
    width = getattr(base, "width", None)
    height = getattr(base, "height", None)
    if grid is None or type(width) is not int or type(height) is not int:
        raise TypeError("MiniGrid environment has no inspectable grid layout.")

    signature: list[tuple[str, tuple[int, int]]] = []
    for y in range(height):
        for x in range(width):
            cell = grid.get(x, y)
            if cell is None:
                continue
            encoded = tuple(int(value) for value in cell.encode())
            signature.append(
                (f"cell:{encoded[0]}:{encoded[1]}:{encoded[2]}", (x, y))
            )
    agent = tuple(int(value) for value in base.agent_pos)
    signature.append(("agent", agent))
    return tuple(signature)


class OfficialMiniGridEvidenceAdapter(
    gym.Wrapper, gym.utils.RecordConstructorArgs
):
    """Add evaluation metadata to an official MiniGrid task.

    The adapter never changes observations, actions, rewards, transitions,
    termination, truncation, or environment state.  It only makes the terminal
    outcome explicit for TransferGrid's existing evaluators.
    """

    def __init__(self, env: gym.Env) -> None:
        gym.utils.RecordConstructorArgs.__init__(self)
        super().__init__(env)

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        outcome = dict(info)
        success = bool(terminated and float(reward) > 0.0)
        failure_reason = None
        if terminated and not success:
            base = self.env.unwrapped
            position = tuple(int(value) for value in base.agent_pos)
            cell = base.grid.get(*position)
            failure_reason = "lava" if getattr(cell, "type", None) == "lava" else "terminal"
        outcome["success"] = success
        outcome["failure_reason"] = failure_reason
        outcome["timeout"] = bool(truncated)
        return observation, reward, terminated, truncated, outcome


class FixedLayoutSeed(gym.Wrapper, gym.utils.RecordConstructorArgs):
    """Fix layout generation without replacing Gymnasium's normal reset seed."""

    def __init__(self, env: gym.Env, *, layout_seed: int) -> None:
        validate_layout_seed(layout_seed)
        gym.utils.RecordConstructorArgs.__init__(self, layout_seed=layout_seed)
        super().__init__(env)
        self.layout_seed = layout_seed

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ):
        """Pass through caller seeding while declaring a separate layout seed."""
        reset_options = dict(options) if options is not None else {}
        reset_options[_FIXED_LAYOUT_SEED_OPTION] = self.layout_seed
        return self.env.reset(seed=seed, options=reset_options)
