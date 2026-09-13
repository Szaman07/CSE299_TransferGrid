"""Optional policy-only snapshots for governed representation trajectories."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import torch
from stable_baselines3.common.callbacks import BaseCallback

from training.artifacts import write_text_exclusive


SNAPSHOT_SCHEMA_VERSION = 1


def policy_state_digest(policy: Any) -> str:
    """Hash policy tensors canonically, independent of archive metadata."""
    digest = hashlib.sha256()
    for key, value in sorted(policy.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class RepresentationSnapshotCallback(BaseCallback):
    """Capture policy weights at exact post-update training boundaries.

    The callback never evaluates the model and never reads or mutates the
    optimizer, rollout buffer, schedules, environments, or RNG streams.
    """

    def __init__(
        self,
        destination: Path,
        *,
        run_id: str,
        total_timesteps: int,
        fractions: Iterable[float],
    ) -> None:
        super().__init__(verbose=0)
        self.destination = Path(destination)
        self.run_id = run_id
        self.total_timesteps = int(total_timesteps)
        self.fractions = tuple(float(value) for value in fractions)
        self.thresholds = tuple(
            (fraction, int(round(fraction * self.total_timesteps)))
            for fraction in self.fractions
        )
        self.records: list[dict[str, Any]] = []
        self._captured_steps: set[int] = set()

    def _on_training_start(self) -> None:
        self.destination.mkdir(parents=True, exist_ok=False)
        self._capture_due(0)

    def _on_rollout_start(self) -> None:
        # At this boundary, the previous rollout's optimizer update has
        # completed. This is the only callback boundary used for intermediates.
        self._capture_due(int(self.num_timesteps))

    def _on_step(self) -> bool:
        return True

    def capture_final(self) -> None:
        """Capture the exact saved-policy boundary after learn() returns."""
        self._capture_due(int(self.model.num_timesteps), final=True)

    def _capture_due(self, current_step: int, *, final: bool = False) -> None:
        for fraction, requested_step in self.thresholds:
            if requested_step in self._captured_steps:
                continue
            if requested_step > current_step:
                continue
            if fraction == 1.0 and not final:
                continue
            if current_step != requested_step:
                raise RuntimeError(
                    "Representation snapshot boundary is not aligned with a "
                    f"completed PPO rollout: requested={requested_step}, "
                    f"current={current_step}."
                )
            path = self.destination / f"policy_step{current_step:09d}.pth"
            state = {
                key: value.detach().cpu().clone()
                for key, value in self.model.policy.state_dict().items()
            }
            torch.save(state, path)
            self.records.append({
                "fraction": fraction,
                "step": current_step,
                "path": str(path),
                "sha256": file_sha256(path),
                "policy_state_sha256": policy_state_digest(self.model.policy),
                "tensor_count": len(state),
            })
            self._captured_steps.add(requested_step)

    def save_manifest(self, path: Path, *, context: dict[str, Any]) -> Path:
        expected = {step for _, step in self.thresholds}
        if self._captured_steps != expected:
            raise RuntimeError(
                "Representation snapshots are incomplete: "
                f"expected={sorted(expected)}, captured={sorted(self._captured_steps)}"
            )
        payload = {
            "schema_version": SNAPSHOT_SCHEMA_VERSION,
            "artifact_type": "transfergrid-policy-representation-snapshots",
            "run_id": self.run_id,
            "policy_only": True,
            "post_optimizer_update_boundaries": True,
            "total_timesteps": self.total_timesteps,
            "fractions": list(self.fractions),
            "snapshots": self.records,
            **context,
        }
        write_text_exclusive(
            path, json.dumps(payload, indent=2, sort_keys=True) + "\n"
        )
        return path
