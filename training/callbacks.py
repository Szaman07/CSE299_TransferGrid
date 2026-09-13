"""Training callbacks layered on Stable-Baselines3 logging."""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Any, Mapping

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

OUTCOMES_PREFIX = "TransferGrid/Outcomes"
POLICY_PREFIX = "TransferGrid/Policy"
EXPERIMENT_PREFIX = "TransferGrid/Experiment"
METADATA_PREFIX = "TransferGrid/Metadata"


@dataclass(frozen=True, slots=True)
class TrainingMetrics:
    """Terminal aggregate of environment outcomes and policy actions."""

    episode_count: int
    success_count: int
    failure_count: int
    timeout_count: int
    success_rate: float | None
    action_counts: dict[str, int]
    action_fractions: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ExperimentLoggingCallback(BaseCallback):
    """Extend SB3 logs without replacing any standard SB3 metrics."""

    def __init__(self, context: Mapping[str, Any]) -> None:
        super().__init__(verbose=0)
        self._context = dict(context)
        self._started_timer = perf_counter()
        self._rollout_iteration = 0
        self._rollout_episodes = 0
        self._rollout_successes = 0
        self._rollout_failures = 0
        self._rollout_timeouts = 0
        self._rollout_actions: Counter[int] = Counter()
        self._total_episodes = 0
        self._total_successes = 0
        self._total_failures = 0
        self._total_timeouts = 0
        self._total_actions: Counter[int] = Counter()
        self._action_labels: dict[int, str] = {}

    def _on_training_start(self) -> None:
        declared_labels = self._context.get("action_labels")
        self._action_labels = (
            {int(index): str(label) for index, label in declared_labels.items()}
            if isinstance(declared_labels, dict)
            else _discover_action_labels(self.model.action_space)
        )
        exclusions = ("stdout", "log", "json", "csv")
        for key in (
            "run_id",
            "capability",
            "training_mode",
            "checkpoint_path",
            "experiment_manifest_path",
        ):
            self.logger.record(
                f"{METADATA_PREFIX}/{key}",
                str(self._context[key]),
                exclude=exclusions,
            )
        self.logger.record(
            f"{METADATA_PREFIX}/configuration",
            json.dumps(self._context, sort_keys=True),
            exclude=exclusions,
        )

    def _on_rollout_start(self) -> None:
        self._rollout_episodes = 0
        self._rollout_successes = 0
        self._rollout_failures = 0
        self._rollout_timeouts = 0
        self._rollout_actions.clear()

    def _on_step(self) -> bool:
        for action in np.asarray(self.locals.get("actions", ())).reshape(-1):
            action_index = int(action)
            self._rollout_actions[action_index] += 1
            self._total_actions[action_index] += 1

        infos = self.locals.get("infos", ())
        dones = self.locals.get("dones", ())
        for info, done in zip(infos, dones, strict=False):
            if not bool(done):
                continue
            success = info.get("success")
            if not isinstance(success, (bool, np.bool_)):
                raise RuntimeError(
                    "Completed training episodes must provide boolean "
                    "info['success']."
                )
            timeout = bool(
                info.get("timeout", info.get("TimeLimit.truncated", False))
            )
            self._rollout_episodes += 1
            self._total_episodes += 1
            if bool(success):
                self._rollout_successes += 1
                self._total_successes += 1
            elif timeout:
                self._rollout_timeouts += 1
                self._total_timeouts += 1
            else:
                self._rollout_failures += 1
                self._total_failures += 1
        return True

    def _on_rollout_end(self) -> None:
        self._rollout_iteration += 1
        self.logger.record(
            f"{OUTCOMES_PREFIX}/episode_count",
            self._rollout_episodes,
        )
        self.logger.record(
            f"{OUTCOMES_PREFIX}/success_count",
            self._rollout_successes,
        )
        self.logger.record(
            f"{OUTCOMES_PREFIX}/failure_count",
            self._rollout_failures,
        )
        self.logger.record(
            f"{OUTCOMES_PREFIX}/timeout_count",
            self._rollout_timeouts,
        )
        if self._rollout_episodes:
            denominator = self._rollout_episodes
            self.logger.record(
                f"{OUTCOMES_PREFIX}/success_rate",
                self._rollout_successes / denominator,
            )
            self.logger.record(
                f"{OUTCOMES_PREFIX}/failure_rate",
                self._rollout_failures / denominator,
            )
            self.logger.record(
                f"{OUTCOMES_PREFIX}/timeout_rate",
                self._rollout_timeouts / denominator,
            )
        action_total = sum(self._rollout_actions.values())
        if action_total:
            for action, count in sorted(self._rollout_actions.items()):
                self.logger.record(
                    f"{POLICY_PREFIX}/action_fraction/"
                    f"{self._action_label(action)}",
                    count / action_total,
                )
        self.logger.record(
            f"{EXPERIMENT_PREFIX}/rollout_iteration",
            self._rollout_iteration,
        )
        self.logger.record(
            f"{EXPERIMENT_PREFIX}/total_timesteps",
            self.num_timesteps,
        )
        self.logger.record(
            f"{EXPERIMENT_PREFIX}/elapsed_seconds",
            perf_counter() - self._started_timer,
        )

    def _on_training_end(self) -> None:
        # On-policy SB3 records its last optimizer metrics after its normal final
        # dump. One explicit dump preserves that final update in TensorBoard.
        self.logger.dump(step=self.num_timesteps)

    def metrics(self) -> TrainingMetrics:
        action_total = sum(self._total_actions.values())
        return TrainingMetrics(
            episode_count=self._total_episodes,
            success_count=self._total_successes,
            failure_count=self._total_failures,
            timeout_count=self._total_timeouts,
            success_rate=(
                self._total_successes / self._total_episodes
                if self._total_episodes
                else None
            ),
            action_counts={
                self._action_label(action): count
                for action, count in sorted(self._total_actions.items())
            },
            action_fractions={
                self._action_label(action): count / action_total
                for action, count in sorted(self._total_actions.items())
            }
            if action_total
            else {},
        )

    def _action_label(self, action: int) -> str:
        return f"{action}_{self._action_labels.get(action, 'action')}"


def _discover_action_labels(action_space: Any) -> dict[int, str]:
    """Return stable numeric labels, enriched when an enum is available."""
    count = getattr(action_space, "n", 0)
    labels = {index: "action" for index in range(int(count))}
    enum_type = getattr(action_space, "enum_class", None)
    if enum_type is not None:
        labels.update(
            {
                int(member): re.sub(r"[^a-z0-9]+", "-", member.name.lower())
                for member in enum_type
            }
        )
    return labels
