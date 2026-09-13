"""Frozen upstream MiniGrid contracts for the external replication study.

These definitions deliberately live outside TransferGrid's product task list.  They
pin the exact upstream environments and wrapper semantics used by the CLI-only
replication without modifying the environments' transition or reward functions.
"""

from __future__ import annotations

import hashlib
import json
from importlib import metadata
from types import MappingProxyType
from typing import Any, Mapping


OFFICIAL_MINIGRID_CONTRACT_SCHEMA = 1
EXPECTED_MINIGRID_VERSION = "3.1.0"
EXPECTED_GYMNASIUM_VERSION = "1.3.0"
OBSERVATION_CONTRACT_ID = "minigrid-egocentric-flat148-v1"
ACTION_CONTRACT_ID = "minigrid-discrete7-v1"


def _descriptor(
    *,
    task_id: str,
    gym_id: str,
    grid_size: int,
    maximum_steps: int,
    mission: str,
) -> dict[str, Any]:
    return {
        "schema_version": OFFICIAL_MINIGRID_CONTRACT_SCHEMA,
        "task_id": task_id,
        "upstream_gym_id": gym_id,
        "grid_size": grid_size,
        "maximum_steps": maximum_steps,
        "mission": mission,
        "minigrid_version": EXPECTED_MINIGRID_VERSION,
        "gymnasium_version": EXPECTED_GYMNASIUM_VERSION,
        "observation_contract_id": OBSERVATION_CONTRACT_ID,
        "action_contract_id": ACTION_CONTRACT_ID,
        "observation_wrapper": "FlatMiniGridObservation-v1",
        "terminal_metadata_adapter": "OfficialMiniGridEvidenceAdapter-v1",
        "dynamics_policy": (
            "upstream observations, actions, rewards, transitions, termination, "
            "truncation, layout generation, and episode horizon are unchanged"
        ),
    }


OFFICIAL_MINIGRID_CONTRACTS: Mapping[str, Mapping[str, Any]] = MappingProxyType(
    {
        "official-empty-random-6x6-v1": MappingProxyType(
            _descriptor(
                task_id="official-empty-random-6x6-v1",
                gym_id="MiniGrid-Empty-Random-6x6-v0",
                grid_size=6,
                maximum_steps=144,
                mission="get to the green goal square",
            )
        ),
        "official-door-key-6x6-v1": MappingProxyType(
            _descriptor(
                task_id="official-door-key-6x6-v1",
                gym_id="MiniGrid-DoorKey-6x6-v0",
                grid_size=6,
                maximum_steps=360,
                mission="use the key to open the door and then get to the goal",
            )
        ),
        "official-lava-crossing-s9n1-v1": MappingProxyType(
            _descriptor(
                task_id="official-lava-crossing-s9n1-v1",
                gym_id="MiniGrid-LavaCrossingS9N1-v0",
                grid_size=9,
                maximum_steps=324,
                mission="avoid the lava and get to the green goal square",
            )
        ),
    }
)


def canonical_contract_json(descriptor: Mapping[str, Any]) -> str:
    """Return the byte-stable representation used for task identity."""
    return json.dumps(dict(descriptor), sort_keys=True, separators=(",", ":"))


def official_contract_digest(task_id: str) -> str:
    """Return the frozen SHA-256 identity for one official task contract."""
    descriptor = OFFICIAL_MINIGRID_CONTRACTS[task_id]
    return hashlib.sha256(canonical_contract_json(descriptor).encode()).hexdigest()


def verify_official_runtime_versions() -> None:
    """Fail closed when upstream environment semantics may have changed."""
    actual = {
        "minigrid": metadata.version("minigrid"),
        "gymnasium": metadata.version("gymnasium"),
    }
    expected = {
        "minigrid": EXPECTED_MINIGRID_VERSION,
        "gymnasium": EXPECTED_GYMNASIUM_VERSION,
    }
    if actual != expected:
        raise RuntimeError(
            "Official MiniGrid replication runtime mismatch: "
            f"expected={expected!r} actual={actual!r}."
        )
