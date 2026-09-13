"""Gymnasium registration for TransferGrid environments."""

from __future__ import annotations

from gymnasium.envs.registration import register, registry

OBJECT_INTERACTION_ENV_ID = "TransferGrid-ObjectInteraction-v0"
GOAL_NAVIGATION_ENV_ID = "TransferGrid-GoalNavigation-v1"
KEY_PICKUP_ENV_ID = "TransferGrid-KeyPickup-v1"
KEY_PICKUP_V2_ENV_ID = "TransferGrid-KeyPickup-v2"
DOOR_KEY_ENV_ID = "TransferGrid-DoorKey-v1"
LAVA_NAVIGATION_ENV_ID = "TransferGrid-LavaNavigation-v1"
LAVA_DOOR_KEY_ENV_ID = "TransferGrid-LavaDoorKey-v1"
LAVA_DOOR_KEY_V2_ENV_ID = "TransferGrid-LavaDoorKey-v2"


def register_environments() -> None:
    """Register TransferGrid environments once per Python process."""
    if OBJECT_INTERACTION_ENV_ID not in registry:
        register(
            id=OBJECT_INTERACTION_ENV_ID,
            entry_point="environments.object_interaction:ObjectInteractionEnv",
        )
    if GOAL_NAVIGATION_ENV_ID not in registry:
        register(
            id=GOAL_NAVIGATION_ENV_ID,
            entry_point="environments.goal_navigation:GoalNavigationEnv",
        )
    if KEY_PICKUP_ENV_ID not in registry:
        register(
            id=KEY_PICKUP_ENV_ID,
            entry_point="environments.key_pickup:KeyPickupEnv",
        )
    if KEY_PICKUP_V2_ENV_ID not in registry:
        register(
            id=KEY_PICKUP_V2_ENV_ID,
            entry_point="environments.key_pickup_v2:KeyPickupV2Env",
        )
    if DOOR_KEY_ENV_ID not in registry:
        register(
            id=DOOR_KEY_ENV_ID,
            entry_point="environments.object_interaction:ObjectInteractionEnv",
            kwargs={"size": 7},
        )
    if LAVA_NAVIGATION_ENV_ID not in registry:
        register(
            id=LAVA_NAVIGATION_ENV_ID,
            entry_point="environments.lava_navigation:LavaNavigationEnv",
        )
    if LAVA_DOOR_KEY_ENV_ID not in registry:
        register(
            id=LAVA_DOOR_KEY_ENV_ID,
            entry_point="environments.lava_door_key:LavaDoorKeyEnv",
        )
    if LAVA_DOOR_KEY_V2_ENV_ID not in registry:
        register(
            id=LAVA_DOOR_KEY_V2_ENV_ID,
            entry_point="environments.lava_door_key_v2:LavaDoorKeyV2Env",
        )
