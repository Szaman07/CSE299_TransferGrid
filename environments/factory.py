"""Canonical construction API for TransferGrid environments."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

import gymnasium as gym

from environments.catalog import (
    LEGACY_OBJECT_INTERACTION_TASK_ID,
    require_released_task,
)
from environments.object_interaction import ObjectInteractionEnv
from environments.registration import (
    OBJECT_INTERACTION_ENV_ID,
    register_environments,
)
from environments.official_minigrid_contracts import verify_official_runtime_versions
from utils.wrappers import (
    FixedLayoutSeed,
    FlatMiniGridObservation,
    OfficialMiniGridEvidenceAdapter,
    validate_layout_seed,
)

OBJECT_INTERACTION_CAPABILITY = "object_interaction"
MEDIUM_DIFFICULTY = "medium"
DEFAULT_GRID_SIZE = 7
MIN_GRID_SIZE = ObjectInteractionEnv.MIN_SIZE

ObservationMode = Literal["raw", "flat"]


@dataclass(frozen=True, slots=True)
class EnvironmentSpec:
    """Backend-owned configuration required to construct an environment."""

    capability: str = OBJECT_INTERACTION_CAPABILITY
    grid_size: int = DEFAULT_GRID_SIZE
    difficulty: str = MEDIUM_DIFFICULTY
    task_id: str | None = None
    task_contract_digest: str | None = None

    def __post_init__(self) -> None:
        if self.task_id is not None:
            definition = require_released_task(self.task_id)
            if (
                type(self.grid_size) is not int
                or (
                    definition.exact_grid_size
                    and self.grid_size != definition.default_grid_size
                )
                or self.grid_size < definition.minimum_grid_size
            ):
                raise ValueError(
                    "grid_size is outside the released contract for task_id "
                    f"{self.task_id!r}."
                )
            if (
                self.task_contract_digest is not None
                and self.task_contract_digest != definition.contract_digest
            ):
                raise ValueError("task contract digest does not match the released task.")
            object.__setattr__(
                self, "task_contract_digest", definition.contract_digest
            )
            return
        if self.task_contract_digest is not None:
            raise ValueError("Legacy environments cannot declare a task digest.")
        if self.capability != OBJECT_INTERACTION_CAPABILITY:
            raise ValueError(f"Unsupported capability: {self.capability!r}.")
        if type(self.grid_size) is not int or self.grid_size < MIN_GRID_SIZE:
            raise ValueError(f"grid_size must be an integer of at least {MIN_GRID_SIZE}.")
        if self.difficulty != MEDIUM_DIFFICULTY:
            raise ValueError(
                f"Unsupported difficulty for {self.capability}: "
                f"{self.difficulty!r}."
            )

    @property
    def resolved_task_id(self) -> str:
        """Return the explicit released ID or the supported legacy identity."""
        return self.task_id or LEGACY_OBJECT_INTERACTION_TASK_ID

    def to_dict(self) -> dict[str, Any]:
        """Serialize one environment contract without leaking lifecycle state."""
        if self.task_id is not None:
            return {
                "task_id": self.task_id,
                "grid_size": self.grid_size,
                "task_contract_digest": self.task_contract_digest,
            }
        return {
            "capability": self.capability,
            "grid_size": self.grid_size,
            "difficulty": self.difficulty,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> EnvironmentSpec:
        """Normalize released or exact supported-legacy environment payloads."""
        if not isinstance(payload, Mapping):
            raise ValueError("environment must be a mapping.")
        values = dict(payload)
        if "task_id" in values:
            allowed = {"task_id", "grid_size", "task_contract_digest"}
            unexpected = sorted(set(values) - allowed)
            if unexpected:
                raise ValueError(
                    "task_id environment payload has unsupported fields: "
                    + ", ".join(unexpected)
                )
            if "grid_size" not in values:
                raise ValueError(
                    "task_id environment payload requires an explicit grid_size."
                )
            if "task_contract_digest" not in values:
                raise ValueError("task_id environment payload requires a contract digest.")
            return cls(
                task_id=values["task_id"],
                grid_size=values["grid_size"],
                task_contract_digest=values["task_contract_digest"],
            )

        required = {"capability", "grid_size", "difficulty"}
        missing = sorted(required - set(values))
        unexpected = sorted(set(values) - required)
        if missing:
            raise ValueError(
                "legacy environment payload is missing required fields: "
                + ", ".join(missing)
            )
        if unexpected:
            raise ValueError(
                "legacy environment payload has unsupported fields: "
                + ", ".join(unexpected)
            )
        return cls(
            capability=values["capability"],
            grid_size=values["grid_size"],
            difficulty=values["difficulty"],
        )


def make_environment(
    spec: EnvironmentSpec,
    *,
    render_mode: str | None = None,
    observation_mode: ObservationMode = "raw",
    fixed_layout_seed: int | None = None,
) -> gym.Env:
    """Construct a registered environment with the requested observation format."""
    if observation_mode not in ("raw", "flat"):
        raise ValueError(f"Unsupported observation mode: {observation_mode!r}.")
    if fixed_layout_seed is not None:
        validate_layout_seed(fixed_layout_seed)

    register_environments()
    environment_id = (
        require_released_task(spec.task_id).gym_id
        if spec.task_id is not None
        else OBJECT_INTERACTION_ENV_ID
    )
    environment = gym.make(
        environment_id,
        size=spec.grid_size,
        render_mode=render_mode,
    )
    if (
        spec.task_id is not None
        and require_released_task(spec.task_id).benchmark_role
        == "official-minigrid-external-validation"
    ):
        verify_official_runtime_versions()
        environment = OfficialMiniGridEvidenceAdapter(environment)
    if fixed_layout_seed is not None:
        environment = FixedLayoutSeed(environment, layout_seed=fixed_layout_seed)
    if observation_mode == "raw":
        return environment

    try:
        return FlatMiniGridObservation(environment)
    except Exception:
        environment.close()
        raise
