"""Common reward function implementations using Pydantic models."""

from __future__ import annotations

from typing import Any, Dict

from gymkhana.core.rewards.base import RewardFunction, register_reward_function
from gymkhana.core.models import TrajectoryMetrics


@register_reward_function("simple")
class SimpleCorrectness(RewardFunction):
    """
    Simple binary reward: +1 for correct, -0.5 for incorrect, 0 for unknown.
    """

    def compute(self, metrics: TrajectoryMetrics) -> Dict[str, Any]:
        if metrics.answer_correct is True:
            final_reward = 1.0
        elif metrics.answer_correct is False:
            final_reward = -0.5
        else:
            final_reward = 0.0

        total = sum(metrics.intermediate_rewards) + final_reward

        return {
            "total_reward": total,
            "final_step_reward": final_reward,
            "metadata": {
                "reward_function": self.name,
            }
        }


@register_reward_function("efficiency")
class EfficiencyAware(RewardFunction):
    """
    Reward that considers both correctness and efficiency.
    """
    base_correct_reward: float = 1.0
    max_efficiency_bonus: float = 0.5
    error_penalty_per_error: float = 0.1
    base_incorrect_penalty: float = -0.5
    max_effort_penalty: float = -0.3

    def compute(self, metrics: TrajectoryMetrics) -> Dict[str, Any]:
        if metrics.answer_correct is True:
            reward = self.base_correct_reward

            if metrics.max_turns > 0:
                efficiency_ratio = 1.0 - (metrics.num_code_blocks / metrics.max_turns)
                efficiency_bonus = self.max_efficiency_bonus * max(0.0, efficiency_ratio)
                reward += efficiency_bonus
            else:
                efficiency_bonus = 0.0

            quality_penalty = self.error_penalty_per_error * metrics.num_errors
            reward -= quality_penalty

            final_reward = reward
            metadata = {
                "reward_function": self.name,
                "base_reward": self.base_correct_reward,
                "efficiency_bonus": efficiency_bonus if metrics.max_turns > 0 else 0.0,
                "quality_penalty": quality_penalty,
            }

        elif metrics.answer_correct is False:
            reward = self.base_incorrect_penalty

            if metrics.max_turns > 0:
                effort_ratio = metrics.num_code_blocks / metrics.max_turns
                effort_penalty = self.max_effort_penalty * effort_ratio
                reward += effort_penalty
            else:
                effort_penalty = 0.0

            final_reward = reward
            metadata = {
                "reward_function": self.name,
                "base_penalty": self.base_incorrect_penalty,
                "effort_penalty": effort_penalty if metrics.max_turns > 0 else 0.0,
            }
        else:
            final_reward = 0.0
            metadata = {
                "reward_function": self.name,
                "note": "unknown_correctness",
            }

        total = sum(metrics.intermediate_rewards) + final_reward

        return {
            "total_reward": total,
            "final_step_reward": final_reward,
            "metadata": metadata,
        }


@register_reward_function("normalized")
class NormalizedReward(RewardFunction):
    """
    Normalized reward that scales by trajectory length.
    """
    base_correct: float = 1.0
    base_incorrect: float = -0.5

    def compute(self, metrics: TrajectoryMetrics) -> Dict[str, Any]:
        if metrics.answer_correct is True:
            final_reward = self.base_correct
        elif metrics.answer_correct is False:
            final_reward = self.base_incorrect
        else:
            final_reward = 0.0

        total = sum(metrics.intermediate_rewards) + final_reward

        reward_per_turn = total / max(1, metrics.num_turns)
        reward_per_code = total / max(1, metrics.num_code_blocks)

        return {
            "total_reward": total,
            "final_step_reward": final_reward,
            "metadata": {
                "reward_function": self.name,
                "reward_per_turn": reward_per_turn,
                "reward_per_code_block": reward_per_code,
            }
        }


# For backwards compatibility
from gymkhana.core.rewards.base import get_reward_function, list_reward_functions

__all__ = [
    "RewardFunction",
    "SimpleCorrectness",
    "EfficiencyAware",
    "NormalizedReward",
    "get_reward_function",
    "list_reward_functions",
    "register_reward_function",
]
