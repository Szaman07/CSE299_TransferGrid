"""Isolated deterministic Development evaluation during governed transfer."""

from __future__ import annotations

import random
from collections import Counter
from time import perf_counter
from typing import Any

import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback

from environments.factory import EnvironmentSpec, make_environment
from training.benchmark_contracts import DEVELOPMENT_SEEDS

DEVELOPMENT_EVALUATION_INTERVAL = 102_400


def evaluate_live_policy(
    model: Any,
    environment: EnvironmentSpec,
    seeds: tuple[int, ...] = DEVELOPMENT_SEEDS,
) -> dict[str, Any]:
    """Evaluate without advancing training environments or global RNG streams."""
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    started = perf_counter()
    records: list[dict[str, Any]] = []
    try:
        evaluation_environment = make_environment(
            environment, observation_mode="flat"
        )
        try:
            for seed in seeds:
                observation, _ = evaluation_environment.reset(seed=seed)
                total_reward = 0.0
                actions: list[int] = []
                terminated = truncated = False
                info: dict[str, Any] = {}
                while not (terminated or truncated):
                    action, _ = model.predict(observation, deterministic=True)
                    action_value = int(np.asarray(action).item())
                    actions.append(action_value)
                    observation, reward, terminated, truncated, info = (
                        evaluation_environment.step(action_value)
                    )
                    total_reward += float(reward)
                records.append({
                    "seed": seed,
                    "total_reward": total_reward,
                    "episode_length": len(actions),
                    "success": bool(info.get("success", False)),
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    "failure_reason": info.get("failure_reason"),
                    "action_counts": dict(Counter(actions)),
                })
        finally:
            evaluation_environment.close()
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)
    successes = sum(record["success"] for record in records)
    return {
        "success_count": successes,
        "success_rate": successes / len(records),
        "mean_reward": float(np.mean([
            record["total_reward"] for record in records
        ])),
        "mean_episode_length": float(np.mean([
            record["episode_length"] for record in records
        ])),
        "evaluation_seconds": perf_counter() - started,
        "records": records,
    }


class DevelopmentEvaluationCallback(BaseCallback):
    """Record exact unsmoothed Development points without early stopping."""

    def __init__(
        self,
        environment: EnvironmentSpec,
        *,
        interval: int = DEVELOPMENT_EVALUATION_INTERVAL,
    ) -> None:
        super().__init__(verbose=0)
        self.environment = environment
        self.interval = interval
        self.next_step = interval
        self.points: list[dict[str, Any]] = []
        self.evaluation_seconds = 0.0

    def evaluate_point(self, step: int, model: Any | None = None) -> dict[str, Any]:
        result = evaluate_live_policy(
            self.model if model is None else model, self.environment
        )
        result["step"] = step
        self.points.append(result)
        self.evaluation_seconds += result["evaluation_seconds"]
        return result

    def _on_step(self) -> bool:
        if self.num_timesteps >= self.next_step:
            self.evaluate_point(self.next_step)
            self.next_step += self.interval
        return True
