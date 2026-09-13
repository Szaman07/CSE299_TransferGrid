# TransferGrid
"""Custom procedural capability environments for TransferGrid."""

from environments.catalog import (
    DRAFT_TASK_IDS,
    LEGACY_OBJECT_INTERACTION_TASK,
    LEGACY_OBJECT_INTERACTION_TASK_ID,
    RESERVED_TASK_IDS,
    TaskDefinition,
    TaskIdentityState,
    UnsupportedTaskIdentityError,
    classify_task_identity,
    legacy_task_definition,
    new_run_task_definitions,
    released_task_definitions,
    require_released_task,
    task_display_name,
)
from environments.factory import EnvironmentSpec, make_environment
from environments.generation import (
    GeneratedPreview,
    GenerationRequest,
    generate_preview,
)
from environments.object_interaction import ObjectInteractionEnv
from environments.registration import (
    DOOR_KEY_ENV_ID,
    GOAL_NAVIGATION_ENV_ID,
    KEY_PICKUP_ENV_ID,
    KEY_PICKUP_V2_ENV_ID,
    LAVA_DOOR_KEY_ENV_ID,
    LAVA_DOOR_KEY_V2_ENV_ID,
    LAVA_NAVIGATION_ENV_ID,
    OBJECT_INTERACTION_ENV_ID,
    register_environments,
)

register_environments()

__all__ = [
    "DRAFT_TASK_IDS",
    "DOOR_KEY_ENV_ID",
    "GOAL_NAVIGATION_ENV_ID",
    "KEY_PICKUP_ENV_ID",
    "KEY_PICKUP_V2_ENV_ID",
    "LAVA_DOOR_KEY_ENV_ID",
    "LAVA_DOOR_KEY_V2_ENV_ID",
    "LAVA_NAVIGATION_ENV_ID",
    "LEGACY_OBJECT_INTERACTION_TASK",
    "LEGACY_OBJECT_INTERACTION_TASK_ID",
    "OBJECT_INTERACTION_ENV_ID",
    "RESERVED_TASK_IDS",
    "EnvironmentSpec",
    "GeneratedPreview",
    "GenerationRequest",
    "ObjectInteractionEnv",
    "TaskDefinition",
    "TaskIdentityState",
    "UnsupportedTaskIdentityError",
    "classify_task_identity",
    "generate_preview",
    "legacy_task_definition",
    "make_environment",
    "new_run_task_definitions",
    "register_environments",
    "released_task_definitions",
    "require_released_task",
    "task_display_name",
]
