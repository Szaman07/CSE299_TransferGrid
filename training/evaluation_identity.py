"""Pure comparison-condition identity helpers for governed evaluations."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from typing import Any

EVALUATION_V2_COMPARISON_FIELDS = (
    "protocol_version",
    "task_id",
    "task_contract_digest",
    "grid_size",
    "maximum_steps",
    "observation_contract_id",
    "action_contract_id",
    "evidence_role",
    "distribution_id",
    "distribution_version",
    "ordered_seed_digest",
    "deterministic_actions",
)
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def build_evaluation_v2_comparison_condition(
    *,
    protocol_version: str,
    task_id: str,
    task_contract_digest: str,
    grid_size: int,
    maximum_steps: int,
    observation_contract_id: str,
    action_contract_id: str,
    evidence_role: str,
    distribution_id: str,
    distribution_version: int,
    ordered_seed_digest: str,
    deterministic_actions: bool,
) -> dict[str, Any]:
    """Return the validated condition payload shared by writers and readers."""
    condition = {
        "protocol_version": protocol_version,
        "task_id": task_id,
        "task_contract_digest": task_contract_digest,
        "grid_size": grid_size,
        "maximum_steps": maximum_steps,
        "observation_contract_id": observation_contract_id,
        "action_contract_id": action_contract_id,
        "evidence_role": evidence_role,
        "distribution_id": distribution_id,
        "distribution_version": distribution_version,
        "ordered_seed_digest": ordered_seed_digest,
        "deterministic_actions": deterministic_actions,
    }
    return normalize_evaluation_v2_comparison_condition(condition)


def normalize_evaluation_v2_comparison_condition(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate and normalize exactly one Evaluation V2 comparison condition."""
    if not isinstance(payload, Mapping):
        raise ValueError("Evaluation V2 comparison condition must be a mapping.")
    condition = dict(payload)
    unexpected = sorted(set(condition) - set(EVALUATION_V2_COMPARISON_FIELDS))
    missing = sorted(set(EVALUATION_V2_COMPARISON_FIELDS) - set(condition))
    if missing or unexpected:
        details = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if unexpected:
            details.append("unexpected " + ", ".join(unexpected))
        raise ValueError(
            "Evaluation V2 comparison condition has invalid fields: "
            + "; ".join(details)
        )
    for key in (
        "protocol_version",
        "task_id",
        "observation_contract_id",
        "action_contract_id",
        "evidence_role",
        "distribution_id",
    ):
        value = condition[key]
        if not isinstance(value, str) or not value:
            raise ValueError(
                f"Evaluation V2 comparison condition {key} must be a "
                "non-empty string."
            )
    for key in ("task_contract_digest", "ordered_seed_digest"):
        value = condition[key]
        if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
            raise ValueError(
                f"Evaluation V2 comparison condition {key} must be a "
                "lowercase 64-character SHA-256."
            )
    for key in ("grid_size", "maximum_steps", "distribution_version"):
        value = condition[key]
        if type(value) is not int or value <= 0:
            raise ValueError(
                f"Evaluation V2 comparison condition {key} must be a "
                "positive integer."
            )
    if type(condition["deterministic_actions"]) is not bool:
        raise ValueError(
            "Evaluation V2 comparison condition deterministic_actions must "
            "be a boolean."
        )
    return {
        key: condition[key]
        for key in EVALUATION_V2_COMPARISON_FIELDS
    }


def evaluation_v2_comparison_condition_digest(
    payload: Mapping[str, Any],
) -> str:
    """Hash one normalized condition while excluding checkpoint provenance."""
    condition = normalize_evaluation_v2_comparison_condition(payload)
    canonical = json.dumps(condition, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
