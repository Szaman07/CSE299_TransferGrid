"""Stable workspace identities and unchanged visible launcher labels."""

from environments.catalog import legacy_task_definition, require_released_task

WORKSPACE_LABELS: dict[str, str] = {
    "generate": "01 GENERATE",
    "train": "02 TRAIN",
    "evaluate": "03 EVALUATE",
    "transfer": "04 TRANSFER",
    "compare": "05 COMPARE",
    "replay": "06 REPLAY",
    "reports": "07 REPORTS",
    "settings": "08 SETTINGS",
}
MENU_ITEMS = tuple(WORKSPACE_LABELS)
LEGACY_VIEW_MIGRATION = {label: workspace_id for workspace_id, label in WORKSPACE_LABELS.items()}

GENERATE_VIEW = "generate"
TRAIN_VIEW = "train"
EVALUATE_VIEW = "evaluate"
TRANSFER_VIEW = "transfer"
COMPARE_VIEW = "compare"
REPLAY_VIEW = "replay"
REPORTS_VIEW = "reports"
SETTINGS_VIEW = "settings"
DEFAULT_ACTIVE_VIEW = GENERATE_VIEW
OBJECT_INTERACTION_LABEL = legacy_task_definition().display_name
GOAL_NAVIGATION_LABEL = require_released_task(
    "goal_navigation-v1"
).display_name
DOOR_KEY_LABEL = require_released_task("door_key-v1").display_name
LAVA_NAVIGATION_LABEL = require_released_task(
    "lava_navigation-v1"
).display_name
LAVA_DOOR_KEY_LABEL = require_released_task(
    "lava_door_key-v1"
).display_name
