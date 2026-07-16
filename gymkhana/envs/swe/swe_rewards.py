"""SWE-specific reward functions."""

from __future__ import annotations

import re
from typing import Any, Dict

from gymkhana.core.rewards.base import RewardFunction, register_reward_function
from gymkhana.core.models import TrajectoryMetrics


@register_reward_function("swe_test_based")
class SWETestBasedReward(RewardFunction):
    """
    Reward function based on test execution results.

    Parses the final answer and conversation to detect:
    - Test execution commands (pytest, python -m pytest, etc.)
    - Test results (PASSED, FAILED, ERROR)
    - Success indicators
    """

    base_completion_reward: float = 0.3      # Reward for attempting
    test_pass_reward: float = 0.7            # Reward for passing tests
    partial_pass_bonus: float = 0.3          # Bonus for some tests passing
    error_penalty_per_error: float = 0.03    # Small penalty per error
    efficiency_bonus_max: float = 0.2        # Bonus for efficiency

    def compute(self, metrics: TrajectoryMetrics) -> Dict[str, Any]:
        # Start with base reward for completion
        if not metrics.success:
            # Didn't even provide a final answer
            return {
                "total_reward": -0.2,
                "final_step_reward": -0.2,
                "metadata": {
                    "reward_function": self.name,
                    "reason": "incomplete",
                }
            }

        reward = self.base_completion_reward

        # Try to detect test results from the trajectory
        # This would require access to the actual turns/messages
        # For now, we'll use a heuristic based on success + code quality

        # Efficiency bonus
        if metrics.max_turns > 0:
            efficiency_ratio = 1.0 - (metrics.num_turns / metrics.max_turns)
            efficiency_bonus = self.efficiency_bonus_max * max(0.0, efficiency_ratio)
            reward += efficiency_bonus
        else:
            efficiency_bonus = 0.0

        # Quality penalty
        error_penalty = self.error_penalty_per_error * metrics.num_errors
        reward -= error_penalty

        # If we had test results, we'd add test_pass_reward here
        # For now, assume completion with few errors = likely success
        if metrics.num_errors == 0 and metrics.num_code_blocks >= 3:
            # Likely ran tests and they passed
            reward += self.test_pass_reward
            test_status = "likely_passed"
        elif metrics.num_errors <= 2 and metrics.num_code_blocks >= 5:
            # Some iteration, might have partial success
            reward += self.partial_pass_bonus
            test_status = "partial_success"
        else:
            test_status = "unknown"

        final_reward = max(0.0, reward)
        total = sum(metrics.intermediate_rewards) + final_reward

        return {
            "total_reward": total,
            "final_step_reward": final_reward,
            "metadata": {
                "reward_function": self.name,
                "test_status": test_status,
                "efficiency_bonus": efficiency_bonus,
                "error_penalty": error_penalty,
            }
        }


@register_reward_function("swe_completion")
class SWECompletionReward(RewardFunction):
    """
    Reward function for SWE tasks based on completion and code quality.

    Since SWE tasks don't have simple string answers to compare,
    we reward based on:
    - Completion: Did the model provide a final answer?
    - Efficiency: Fewer turns is better
    - Quality: Fewer errors is better
    """

    base_completion_reward: float = 0.5  # Reward for completing the task
    max_efficiency_bonus: float = 0.3    # Bonus for being efficient
    error_penalty_per_error: float = 0.05  # Penalty per error
    incomplete_penalty: float = -0.3     # Penalty for not completing

    def compute(self, metrics: TrajectoryMetrics) -> Dict[str, Any]:
        if metrics.success:
            # Base reward for completion
            reward = self.base_completion_reward

            # Efficiency bonus (fewer turns is better)
            if metrics.max_turns > 0:
                efficiency_ratio = 1.0 - (metrics.num_turns / metrics.max_turns)
                efficiency_bonus = self.max_efficiency_bonus * max(0.0, efficiency_ratio)
                reward += efficiency_bonus
            else:
                efficiency_bonus = 0.0

            # Quality penalty (fewer errors is better)
            quality_penalty = self.error_penalty_per_error * metrics.num_errors
            reward -= quality_penalty

            final_reward = max(0.0, reward)  # Don't go negative for completed tasks

            metadata = {
                "reward_function": self.name,
                "base_reward": self.base_completion_reward,
                "efficiency_bonus": efficiency_bonus,
                "quality_penalty": quality_penalty,
                "final_reward": final_reward,
            }
        else:
            # Penalty for not completing
            final_reward = self.incomplete_penalty

            metadata = {
                "reward_function": self.name,
                "incomplete_penalty": self.incomplete_penalty,
            }

        total = sum(metrics.intermediate_rewards) + final_reward

        return {
            "total_reward": total,
            "final_step_reward": final_reward,
            "metadata": metadata,
        }


@register_reward_function("swe_progress")
class SWEProgressReward(RewardFunction):
    """
    Reward function that gives positive rewards for any progress.

    More lenient than SWECompletionReward - rewards any attempt.
    """

    base_attempt_reward: float = 0.2     # Reward just for trying
    completion_bonus: float = 0.5        # Extra for completing
    code_block_reward: float = 0.05      # Small reward per code block
    max_code_block_reward: float = 0.3   # Cap on code block rewards
    error_penalty: float = 0.03          # Small penalty per error

    def compute(self, metrics: TrajectoryMetrics) -> Dict[str, Any]:
        # Base reward for attempting
        reward = self.base_attempt_reward

        # Reward for code blocks (capped)
        code_reward = min(
            self.code_block_reward * metrics.num_code_blocks,
            self.max_code_block_reward
        )
        reward += code_reward

        # Bonus for completion
        if metrics.success:
            reward += self.completion_bonus

        # Small penalty for errors
        error_penalty = self.error_penalty * metrics.num_errors
        reward -= error_penalty

        final_reward = max(0.0, reward)  # Always non-negative

        total = sum(metrics.intermediate_rewards) + final_reward

        metadata = {
            "reward_function": self.name,
            "base_attempt": self.base_attempt_reward,
            "code_reward": code_reward,
            "completion_bonus": self.completion_bonus if metrics.success else 0.0,
            "error_penalty": error_penalty,
            "final_reward": final_reward,
        }

        return {
            "total_reward": total,
            "final_step_reward": final_reward,
            "metadata": metadata,
        }


__all__ = ["SWECompletionReward", "SWEProgressReward", "SWETestBasedReward"]


def extract_test_results_from_trajectory(turns: list) -> dict:
    """
    Extract test results from trajectory turns.

    Looks for pytest output patterns like:
    - "X passed"
    - "X failed"
    - "PASSED", "FAILED", "ERROR"
    - "All tests passed"

    Returns:
        dict with:
            - tests_run: int
            - tests_passed: int
            - tests_failed: int
            - all_passed: bool
    """
    tests_run = 0
    tests_passed = 0
    tests_failed = 0

    # Patterns to detect test results
    pytest_summary_pattern = r'(\d+) passed'
    pytest_failed_pattern = r'(\d+) failed'
    pytest_error_pattern = r'(\d+) error'

    for turn in turns:
        content = turn.content if hasattr(turn, 'content') else str(turn)

        # Look for pytest summary
        passed_match = re.search(pytest_summary_pattern, content)
        if passed_match:
            tests_passed = max(tests_passed, int(passed_match.group(1)))

        failed_match = re.search(pytest_failed_pattern, content)
        if failed_match:
            tests_failed = max(tests_failed, int(failed_match.group(1)))

        # Count individual test results
        tests_run += content.count('PASSED')
        tests_failed += content.count('FAILED')
        tests_failed += content.count('ERROR')

    all_passed = tests_run > 0 and tests_failed == 0

    return {
        "tests_run": tests_run,
        "tests_passed": tests_passed,
        "tests_failed": tests_failed,
        "all_passed": all_passed,
    }
