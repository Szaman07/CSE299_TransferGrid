"""Static task-identity boundaries for the TransferGrid benchmark."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Mapping

from environments.official_minigrid_contracts import official_contract_digest
from environments.registration import (
    DOOR_KEY_ENV_ID,
    GOAL_NAVIGATION_ENV_ID,
    KEY_PICKUP_ENV_ID,
    KEY_PICKUP_V2_ENV_ID,
    LAVA_DOOR_KEY_ENV_ID,
    LAVA_DOOR_KEY_V2_ENV_ID,
    LAVA_NAVIGATION_ENV_ID,
    OBJECT_INTERACTION_ENV_ID,
)

LEGACY_OBJECT_INTERACTION_TASK_ID = "legacy/object_interaction-v0"

RESERVED_TASK_IDS = frozenset(
    {
        "distractor_door_key-v1",
    }
)
DRAFT_TASK_IDS = frozenset(
    {
        "draft/key_pickup",
        "draft/key_pickup_v2",
        "draft/distractor_door_key",
    }
)


class TaskIdentityState(str, Enum):
    """Lifecycle classification used to keep unreleased tasks undiscoverable."""

    LEGACY = "legacy"
    RESERVED = "reserved"
    DRAFT = "draft"
    RELEASED = "released"
    UNKNOWN = "unknown"
    MALFORMED = "malformed"


class UnsupportedTaskIdentityError(ValueError):
    """Raised when a serialized task identity is not governed and released."""


@dataclass(frozen=True, slots=True)
class TaskDefinition:
    """Small auditable definition for one supported environment identity."""

    task_id: str
    gym_id: str
    display_name: str
    artifact_namespace: str
    default_grid_size: int
    minimum_grid_size: int
    observation_contract_id: str
    action_contract_id: str
    lifecycle: TaskIdentityState
    contract_digest: str | None = None
    mission: str | None = None
    exact_grid_size: bool = False
    new_runs_visible: bool = True
    benchmark_role: str = "standard"
    superseded_by: str | None = None


LEGACY_OBJECT_INTERACTION_TASK = TaskDefinition(
    task_id=LEGACY_OBJECT_INTERACTION_TASK_ID,
    gym_id=OBJECT_INTERACTION_ENV_ID,
    display_name="Object Interaction (Legacy DoorKey)",
    artifact_namespace="object_interaction",
    default_grid_size=9,
    minimum_grid_size=7,
    observation_contract_id="minigrid-egocentric-flat148-v1",
    action_contract_id="minigrid-discrete7-v1",
    lifecycle=TaskIdentityState.LEGACY,
    new_runs_visible=False,
    benchmark_role="historical-compatibility",
)

GOAL_NAVIGATION_TASK = TaskDefinition(
    task_id="goal_navigation-v1",
    gym_id=GOAL_NAVIGATION_ENV_ID,
    display_name="Goal Navigation",
    artifact_namespace="goal_navigation",
    default_grid_size=7,
    minimum_grid_size=7,
    observation_contract_id="minigrid-egocentric-flat148-v1",
    action_contract_id="minigrid-discrete7-v1",
    lifecycle=TaskIdentityState.RELEASED,
    contract_digest="cd9aae37e20d54ada4c3a2be92f997723be2077b17f4f24fee72a63fddbde82f",
    mission="reach the green goal",
    exact_grid_size=True,
)

KEY_PICKUP_TASK = TaskDefinition(
    task_id="key_pickup-v1",
    gym_id=KEY_PICKUP_ENV_ID,
    display_name="Key Pickup",
    artifact_namespace="key_pickup",
    default_grid_size=7,
    minimum_grid_size=7,
    observation_contract_id="minigrid-egocentric-flat148-v1",
    action_contract_id="minigrid-discrete7-v1",
    lifecycle=TaskIdentityState.RELEASED,
    contract_digest="db632b76257525a51106845a23f3c5e867dd2298cb08da401272d099258aa402",
    mission="pick up the yellow key",
    exact_grid_size=True,
    new_runs_visible=False,
    superseded_by="key_pickup-v2",
)

KEY_PICKUP_V2_TASK = TaskDefinition(
    task_id="key_pickup-v2",
    gym_id=KEY_PICKUP_V2_ENV_ID,
    display_name="Key Pickup v2",
    artifact_namespace="key_pickup_v2",
    default_grid_size=7,
    minimum_grid_size=7,
    observation_contract_id="minigrid-egocentric-flat148-v1",
    action_contract_id="minigrid-discrete7-v1",
    lifecycle=TaskIdentityState.RELEASED,
    contract_digest="a09eb4ad1594d49cf95f40a51174bdf7a77df5deb1b7ad32dd2692d19770cdd0",
    mission="pick up the yellow key",
    exact_grid_size=True,
)

_RELEASED_TASKS: Mapping[str, TaskDefinition] = MappingProxyType(
    {
        GOAL_NAVIGATION_TASK.task_id: GOAL_NAVIGATION_TASK,
        KEY_PICKUP_TASK.task_id: KEY_PICKUP_TASK,
        KEY_PICKUP_V2_TASK.task_id: KEY_PICKUP_V2_TASK,
        "door_key-v1": TaskDefinition(
            task_id="door_key-v1",
            gym_id=DOOR_KEY_ENV_ID,
            display_name="DoorKey",
            artifact_namespace="door_key",
            default_grid_size=7,
            minimum_grid_size=7,
            observation_contract_id="minigrid-egocentric-flat148-v1",
            action_contract_id="minigrid-discrete7-v1",
            lifecycle=TaskIdentityState.RELEASED,
            contract_digest="c2c944f75d849693c6b477640b3dc89f8307ee030af1aec081ab951331673d5d",
            mission=(
                "pick up the yellow key, unlock the yellow door, and reach the goal"
            ),
            exact_grid_size=False,
        ),
        "lava_navigation-v1": TaskDefinition(
            task_id="lava_navigation-v1",
            gym_id=LAVA_NAVIGATION_ENV_ID,
            display_name="Lava Navigation",
            artifact_namespace="lava_navigation",
            default_grid_size=7,
            minimum_grid_size=7,
            observation_contract_id="minigrid-egocentric-flat148-v1",
            action_contract_id="minigrid-discrete7-v1",
            lifecycle=TaskIdentityState.RELEASED,
            contract_digest="d9ee63610c5a3cf50c8c38037d6c2024ca155871548ccae42a461567e42727e8",
            mission="cross the lava barrier through the safe gap and reach the goal",
            exact_grid_size=True,
        ),
        "lava_door_key-v1": TaskDefinition(
            task_id="lava_door_key-v1",
            gym_id=LAVA_DOOR_KEY_ENV_ID,
            display_name="Lava DoorKey v1 · Simple Composite",
            artifact_namespace="lava_door_key",
            default_grid_size=7,
            minimum_grid_size=7,
            observation_contract_id="minigrid-egocentric-flat148-v1",
            action_contract_id="minigrid-discrete7-v1",
            lifecycle=TaskIdentityState.RELEASED,
            contract_digest="85d34203a5bf4c184b55574b6503347bc5a16ab81c785ee256675c6b48df3961",
            mission=(
                "pick up the yellow key, unlock the yellow door, cross the "
                "safe gap, and reach the goal"
            ),
            exact_grid_size=True,
            benchmark_role="simple-composite",
            superseded_by="lava_door_key-v2",
        ),
        "lava_door_key-v2": TaskDefinition(
            task_id="lava_door_key-v2",
            gym_id=LAVA_DOOR_KEY_V2_ENV_ID,
            display_name="Lava DoorKey v2 · Flagship",
            artifact_namespace="lava_door_key_v2",
            default_grid_size=7,
            minimum_grid_size=7,
            observation_contract_id="minigrid-egocentric-flat148-v1",
            action_contract_id="minigrid-discrete7-v1",
            lifecycle=TaskIdentityState.RELEASED,
            contract_digest="6d51edc09b2e96a7485a8cb3ba9e7d7238131e22842a0ae4636622923b9ac9dc",
            mission=(
                "pick up the yellow key, cross the safe river gap, unlock the "
                "yellow door, and reach the goal"
            ),
            exact_grid_size=True,
            benchmark_role="flagship",
        ),
    }
)

_EXTERNAL_VALIDATION_TASKS: Mapping[str, TaskDefinition] = MappingProxyType(
    {
        "official-empty-random-6x6-v1": TaskDefinition(
            task_id="official-empty-random-6x6-v1",
            gym_id="MiniGrid-Empty-Random-6x6-v0",
            display_name="Official MiniGrid Empty Random 6x6",
            artifact_namespace="official_minigrid_empty_random_6x6",
            default_grid_size=6,
            minimum_grid_size=6,
            observation_contract_id="minigrid-egocentric-flat148-v1",
            action_contract_id="minigrid-discrete7-v1",
            lifecycle=TaskIdentityState.RELEASED,
            contract_digest=official_contract_digest("official-empty-random-6x6-v1"),
            mission="get to the green goal square",
            exact_grid_size=True,
            new_runs_visible=False,
            benchmark_role="official-minigrid-external-validation",
        ),
        "official-door-key-6x6-v1": TaskDefinition(
            task_id="official-door-key-6x6-v1",
            gym_id="MiniGrid-DoorKey-6x6-v0",
            display_name="Official MiniGrid DoorKey 6x6",
            artifact_namespace="official_minigrid_door_key_6x6",
            default_grid_size=6,
            minimum_grid_size=6,
            observation_contract_id="minigrid-egocentric-flat148-v1",
            action_contract_id="minigrid-discrete7-v1",
            lifecycle=TaskIdentityState.RELEASED,
            contract_digest=official_contract_digest("official-door-key-6x6-v1"),
            mission="use the key to open the door and then get to the goal",
            exact_grid_size=True,
            new_runs_visible=False,
            benchmark_role="official-minigrid-external-validation",
        ),
        "official-lava-crossing-s9n1-v1": TaskDefinition(
            task_id="official-lava-crossing-s9n1-v1",
            gym_id="MiniGrid-LavaCrossingS9N1-v0",
            display_name="Official MiniGrid LavaCrossing S9N1",
            artifact_namespace="official_minigrid_lava_crossing_s9n1",
            default_grid_size=9,
            minimum_grid_size=9,
            observation_contract_id="minigrid-egocentric-flat148-v1",
            action_contract_id="minigrid-discrete7-v1",
            lifecycle=TaskIdentityState.RELEASED,
            contract_digest=official_contract_digest(
                "official-lava-crossing-s9n1-v1"
            ),
            mission="avoid the lava and get to the green goal square",
            exact_grid_size=True,
            new_runs_visible=False,
            benchmark_role="official-minigrid-external-validation",
        ),
    }
)


def classify_task_identity(task_id: object) -> TaskIdentityState:
    """Classify an identity without promoting drafts or reserved names."""
    if not isinstance(task_id, str) or not task_id.strip():
        return TaskIdentityState.MALFORMED
    if task_id == LEGACY_OBJECT_INTERACTION_TASK_ID:
        return TaskIdentityState.LEGACY
    if task_id in RESERVED_TASK_IDS:
        return TaskIdentityState.RESERVED
    if task_id in DRAFT_TASK_IDS:
        return TaskIdentityState.DRAFT
    if task_id in _RELEASED_TASKS or task_id in _EXTERNAL_VALIDATION_TASKS:
        return TaskIdentityState.RELEASED
    return TaskIdentityState.UNKNOWN


def released_task_definitions() -> tuple[TaskDefinition, ...]:
    """Return governed released tasks only; external tasks stay separate."""
    return tuple(_RELEASED_TASKS.values())


def new_run_task_definitions() -> tuple[TaskDefinition, ...]:
    """Return released tasks intentionally visible in new-run workflows."""
    return tuple(
        definition
        for definition in _RELEASED_TASKS.values()
        if definition.new_runs_visible
    )


def external_validation_task_definitions() -> tuple[TaskDefinition, ...]:
    """Return governed official tasks kept out of product UI discovery."""
    return tuple(_EXTERNAL_VALIDATION_TASKS.values())


def interactive_task_definitions() -> tuple[TaskDefinition, ...]:
    """Return tasks explicitly exposed by interactive Train/Generate workflows.

    Official MiniGrid contracts remain separate from TransferGrid's released
    benchmark catalog and certified evidence.  This additive view only makes
    those already-governed external tasks selectable for new interactive runs.
    """
    return (*new_run_task_definitions(), *external_validation_task_definitions())


def require_released_task(task_id: object) -> TaskDefinition:
    """Return one governed definition or reject its exact lifecycle state."""
    state = classify_task_identity(task_id)
    if state is TaskIdentityState.RELEASED:
        key = str(task_id)
        return _RELEASED_TASKS.get(key) or _EXTERNAL_VALIDATION_TASKS[key]
    raise UnsupportedTaskIdentityError(
        f"task_id {task_id!r} is {state.value}, not a released governed task."
    )


def legacy_task_definition() -> TaskDefinition:
    """Return the supported historical Object Interaction identity."""
    return LEGACY_OBJECT_INTERACTION_TASK


def task_display_name(task_id: object) -> str:
    """Return the catalog-owned label for a released or supported legacy task."""
    if task_id is None or task_id == LEGACY_OBJECT_INTERACTION_TASK_ID:
        return LEGACY_OBJECT_INTERACTION_TASK.display_name
    return require_released_task(task_id).display_name
