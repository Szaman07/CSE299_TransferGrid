"""Checkpoint persistence helpers for TransferGrid PPO models."""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import gymnasium as gym
from stable_baselines3 import PPO

from environments.factory import EnvironmentSpec
from environments.catalog import require_released_task
from training.artifacts import publish_new_file, write_text_exclusive

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT_ROOT = PROJECT_ROOT / "checkpoints"
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")
WINDOWS_RESERVED_BASENAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{number}" for number in range(1, 10)),
    *(f"LPT{number}" for number in range(1, 10)),
}


def build_checkpoint_path(
    environment: EnvironmentSpec,
    *,
    seed: int,
    total_timesteps: int,
    learning_rate: float,
    training_mode: str = "procedural",
    layout_seed: int | None = None,
    run_id: str | None = None,
    checkpoint_root: Path = DEFAULT_CHECKPOINT_ROOT,
) -> Path:
    """Return a legacy-compatible or run-specific checkpoint path."""
    learning_rate_label = (
        format(learning_rate, ".12g")
        .replace(".", "p")
        .replace("+", "")
        .replace("-", "m")
    )
    mode_label = (
        f"_fixedlayout{layout_seed}"
        if training_mode == "fixed_layout"
        else ""
    )
    namespace = (
        require_released_task(environment.task_id).artifact_namespace
        if environment.task_id is not None
        else environment.capability
    )
    identity = (
        environment.task_id.replace("-", "_")
        if environment.task_id is not None
        else environment.difficulty
    )
    filename = (
        f"ppo_{identity}_grid{environment.grid_size}"
        f"_seed{seed}{mode_label}_steps{total_timesteps}_lr{learning_rate_label}.zip"
    )
    capability_root = Path(checkpoint_root) / namespace
    if run_id is None:
        return capability_root / filename
    return capability_root / validate_training_run_id(run_id) / filename


def save_model(model: PPO, checkpoint_path: str | Path) -> Path:
    """Atomically publish a new PPO model without overwriting an existing run."""
    path = _normalize_checkpoint_path(checkpoint_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Checkpoint already exists: {path}")

    temporary_path = path.with_name(
        f".{path.stem}.{uuid.uuid4().hex}.tmp.zip"
    )
    try:
        model.save(temporary_path)
        if not temporary_path.is_file():
            raise RuntimeError(
                f"Stable-Baselines3 did not create temporary checkpoint "
                f"{temporary_path}."
            )
        publish_new_file(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return path


def load_model(
    checkpoint_path: str | Path,
    *,
    environment: gym.Env | None = None,
) -> PPO:
    """Load PPO, first validating any metadata sidecar bound to the checkpoint."""
    path = _normalize_checkpoint_path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {path}")
    if checkpoint_metadata_path(path).is_file():
        load_checkpoint_metadata(path)
    return PPO.load(path, env=environment, device="cpu")


def save_checkpoint_metadata(
    checkpoint_path: str | Path,
    metadata: Mapping[str, Any],
) -> Path:
    """Atomically publish new reproducibility metadata beside a checkpoint."""
    normalized_checkpoint_path = _normalize_checkpoint_path(checkpoint_path)
    if not normalized_checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint does not exist: {normalized_checkpoint_path}"
        )
    metadata_path = checkpoint_metadata_path(normalized_checkpoint_path)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    if metadata_path.exists():
        raise FileExistsError(f"Checkpoint metadata already exists: {metadata_path}")

    payload = dict(metadata)
    schema_version = _validate_metadata_schema_version(payload)
    if schema_version in (2, 3):
        expected_hash = payload.get("checkpoint_sha256")
        actual_hash = checkpoint_sha256(normalized_checkpoint_path)
        if expected_hash != actual_hash:
            raise ValueError(
                "Schema v2 checkpoint metadata must contain the checkpoint's "
                "exact SHA-256 digest."
            )

    return write_text_exclusive(
        metadata_path,
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
    )


def load_checkpoint_metadata(checkpoint_path: str | Path) -> dict[str, Any]:
    """Load metadata, validating checkpoint binding for current schemas."""
    path = checkpoint_metadata_path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint metadata does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Checkpoint metadata must contain a JSON object: {path}")
    schema_version = _validate_metadata_schema_version(payload)
    if schema_version in (2, 3):
        expected_hash = payload.get("checkpoint_sha256")
        if not isinstance(expected_hash, str) or len(expected_hash) != 64:
            raise ValueError(
                f"Schema v2 checkpoint metadata has no valid SHA-256 digest: {path}"
            )
        actual_hash = checkpoint_sha256(checkpoint_path)
        if expected_hash != actual_hash:
            raise ValueError(
                f"Checkpoint metadata SHA-256 does not match the model: {path}"
            )
    return payload


def _validate_metadata_schema_version(metadata: Mapping[str, Any]) -> int:
    """Accept only checkpoint metadata schemas understood by this codebase."""
    schema_version = metadata.get("schema_version")
    if type(schema_version) is not int or schema_version not in (1, 2, 3):
        raise ValueError(
            "Checkpoint metadata schema_version must be the integer 1, 2, or 3; "
            f"received {schema_version!r}."
        )
    return schema_version


def checkpoint_metadata_path(checkpoint_path: str | Path) -> Path:
    """Return the JSON sidecar path for a checkpoint."""
    return _normalize_checkpoint_path(checkpoint_path).with_suffix(".metadata.json")


def checkpoint_sha256(checkpoint_path: str | Path) -> str:
    """Return the SHA-256 digest for one existing checkpoint archive."""
    path = _normalize_checkpoint_path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as checkpoint_file:
        for chunk in iter(lambda: checkpoint_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_training_run_id(run_id: str) -> str:
    """Validate a conservative, filename-safe training run identifier."""
    if (
        not isinstance(run_id, str)
        or RUN_ID_PATTERN.fullmatch(run_id) is None
        or run_id.upper() in WINDOWS_RESERVED_BASENAMES
    ):
        raise ValueError(
            "run_id must contain 1-128 ASCII letters, digits, underscores, or "
            "hyphens, must start with a letter or digit, and must not be a "
            "reserved Windows device name."
        )
    return run_id


def _normalize_checkpoint_path(checkpoint_path: str | Path) -> Path:
    """Normalize user paths to the .zip extension used by SB3."""
    path = Path(checkpoint_path)
    if path.suffix.lower() != ".zip":
        path = path.with_suffix(".zip")
    return path
