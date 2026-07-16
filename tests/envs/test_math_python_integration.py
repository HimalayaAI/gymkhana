"""Integration tests for math-python environment with rollout tracking."""

import pytest
from datetime import datetime, timezone

from gymkhana.envs.math_python import MathPythonEnv
from gymkhana.envs.config import (
    EnvConfig,
    DatasetSettings,
    EnvironmentType,
    REPLSettings,
    RolloutTerminationPolicy,
)
from gymkhana.envs.environment import Task
from gymkhana.core.models.trajectory import RolloutStatus, RolloutState


class TestMathPythonEnvironmentSetup:
    """Test basic math-python environment setup with new config."""

    def test_default_config_includes_termination_policy(self):
        """Test that default config includes termination policy."""
        env = MathPythonEnv()

        assert env.config.rollout_termination_policy is not None
        assert isinstance(env.config.rollout_termination_policy, RolloutTerminationPolicy)

        # Check default values
        policy = env.config.rollout_termination_policy
        assert policy.terminate_on_format_violation is True
        assert policy.max_consecutive_errors == 3
        assert policy.max_total_errors == 5

    def test_custom_termination_policy(self):
        """Test creating environment with custom termination policy."""
        config = EnvConfig(
            name="math-python-test",
            dataset=DatasetSettings(
                environment=EnvironmentType.MATH_PYTHON,
                limit=1
            ),
            rollout_termination_policy=RolloutTerminationPolicy(
                terminate_on_format_violation=False,
                max_consecutive_errors=5,
                max_total_errors=10
            )
        )

        env = MathPythonEnv(config=config)

        policy = env.config.rollout_termination_policy
        assert policy.terminate_on_format_violation is False
        assert policy.max_consecutive_errors == 5
        assert policy.max_total_errors == 10

    def test_policy_checker_available(self):
        """Test that policy checker method is available."""
        env = MathPythonEnv()

        assert hasattr(env, '_check_termination_policy')
        assert callable(env._check_termination_policy)


class TestRolloutStateTracking:
    """Test rollout state tracking in math environment."""

    def test_rollout_state_creation(self):
        """Test creating rollout states for math tasks."""
        env = MathPythonEnv()

        # Create rollout states for a multi-rollout task
        num_rollouts = 4
        states = [
            RolloutState(
                rollout_id=g,
                status=RolloutStatus.ACTIVE,
                started_at=datetime.now(timezone.utc)
            )
            for g in range(num_rollouts)
        ]

        assert len(states) == 4
        for g, state in enumerate(states):
            assert state.rollout_id == g
            assert state.status == RolloutStatus.ACTIVE
            assert state.num_errors == 0
            assert state.num_format_violations == 0

    def test_rollout_state_execution_tracking(self):
        """Test tracking execution results in rollout state."""
        state = RolloutState(
            rollout_id=0,
            status=RolloutStatus.ACTIVE,
            started_at=datetime.now(timezone.utc)
        )

        # Simulate successful execution
        state.record_execution(success=True, reward=1.0)
        assert state.num_code_blocks == 1
        assert state.num_errors == 0
        assert state.consecutive_errors == 0
        assert state.total_reward == 1.0

        # Simulate error
        state.record_execution(success=False, reward=-0.1)
        assert state.num_code_blocks == 2
        assert state.num_errors == 1
        assert state.consecutive_errors == 1
        assert state.total_reward == 0.9

        # Simulate recovery
        state.record_execution(success=True, reward=0.5)
        assert state.num_code_blocks == 3
        assert state.num_errors == 1  # Total doesn't reset
        assert state.consecutive_errors == 0  # Consecutive resets
        assert state.total_reward == 1.4


class TestTerminationPolicyInMathEnv:
    """Test termination policy checking in math environment context."""

    def setup_method(self):
        """Set up test environment."""
        config = EnvConfig(
            name="math-python-test",
            dataset=DatasetSettings(
                environment=EnvironmentType.MATH_PYTHON,
                limit=1
            ),
            repl=REPLSettings(max_turns=10),
            rollout_termination_policy=RolloutTerminationPolicy(
                terminate_on_format_violation=True,
                max_consecutive_errors=3,
                max_total_errors=5
            )
        )
        self.env = MathPythonEnv(config=config)

    def test_math_specific_format_violations(self):
        """Test format violations in math context (hallucinated REPL tags)."""
        state = RolloutState(
            rollout_id=0,
            status=RolloutStatus.ACTIVE,
            started_at=datetime.now(timezone.utc)
        )

        # Simulate format violation (hallucinated <repl> tag)
        state.record_execution(
            success=True,
            has_format_violation=True,
            format_violation_type="hallucinated_repl"
        )

        should_terminate, reason = self.env._check_termination_policy(
            state, [state], turn_idx=0
        )

        assert should_terminate is True
        assert "Format violation" in reason

    def test_math_error_patterns(self):
        """Test typical math error patterns."""
        state = RolloutState(
            rollout_id=0,
            status=RolloutStatus.ACTIVE,
            started_at=datetime.now(timezone.utc)
        )

        # Simulate typical math errors (syntax, name errors, etc.)
        for _ in range(3):
            state.record_execution(success=False, reward=-0.1)

        should_terminate, reason = self.env._check_termination_policy(
            state, [state], turn_idx=0
        )

        assert should_terminate is True
        assert "Consecutive errors: 3/3" in reason

    def test_math_recovery_scenario(self):
        """Test that math rollout can recover from errors."""
        state = RolloutState(
            rollout_id=0,
            status=RolloutStatus.ACTIVE,
            started_at=datetime.now(timezone.utc)
        )

        # Error, error, success pattern
        state.record_execution(success=False)
        state.record_execution(success=False)

        # Should not terminate yet (2 < 3)
        should_terminate, _ = self.env._check_termination_policy(
            state, [state], turn_idx=0
        )
        assert should_terminate is False

        # Recovery
        state.record_execution(success=True)

        # Should still not terminate (consecutive reset)
        should_terminate, _ = self.env._check_termination_policy(
            state, [state], turn_idx=0
        )
        assert should_terminate is False
        assert state.consecutive_errors == 0
        assert state.num_errors == 2  # Total still tracked


class TestPipelineStatsIntegration:
    """Test that pipeline stats work with rollout tracking."""

    def test_stats_record_rollout(self):
        """Test recording rollout states in pipeline stats."""
        from gymkhana.core.models.trajectory import PipelineStats

        stats = PipelineStats()

        # Create some rollout states
        completed_state = RolloutState(
            rollout_id=0,
            status=RolloutStatus.COMPLETED,
            started_at=datetime.now(timezone.utc)
        )
        completed_state.mark_completed()

        failed_state = RolloutState(
            rollout_id=1,
            status=RolloutStatus.FAILED,
            started_at=datetime.now(timezone.utc)
        )
        failed_state.mark_failed("Consecutive errors: 3")

        # Record them
        stats.record_rollout(completed_state)
        stats.record_rollout(failed_state)

        assert stats.total_rollouts == 2
        assert stats.rollouts_completed == 1
        assert stats.rollouts_terminated_early == 1
        assert "Consecutive errors: 3" in stats.termination_reasons

    def test_stats_report_includes_rollouts(self):
        """Test that stats report includes rollout information."""
        from gymkhana.core.models.trajectory import PipelineStats

        stats = PipelineStats()

        # Add some rollout data
        for i in range(4):
            state = RolloutState(
                rollout_id=i,
                status=RolloutStatus.COMPLETED if i < 2 else RolloutStatus.FAILED,
                started_at=datetime.now(timezone.utc)
            )
            if i >= 2:
                state.mark_failed("Test termination")
            stats.record_rollout(state)

        report = stats.report()

        assert "Rollout Statistics" in report
        assert "Total rollouts:" in report
        assert "4" in report
        assert "Completed:" in report
        assert "2" in report
        assert "Terminated early:" in report


class TestConfigurationPresets:
    """Test different configuration presets for math environment."""

    def test_conservative_policy(self):
        """Test conservative termination policy for math."""
        config = EnvConfig(
            name="math-python-conservative",
            dataset=DatasetSettings(
                environment=EnvironmentType.MATH_PYTHON,
                limit=1
            ),
            rollout_termination_policy=RolloutTerminationPolicy(
                terminate_on_format_violation=True,
                max_consecutive_errors=5,  # More lenient
                max_total_errors=8,
                min_code_blocks_before_answer=1
            )
        )

        env = MathPythonEnv(config=config)
        policy = env.config.rollout_termination_policy

        assert policy.max_consecutive_errors == 5
        assert policy.max_total_errors == 8

    def test_balanced_policy(self):
        """Test balanced termination policy for math (default)."""
        env = MathPythonEnv()
        policy = env.config.rollout_termination_policy

        assert policy.max_consecutive_errors == 3
        assert policy.max_total_errors == 5
        assert policy.min_code_blocks_before_answer == 1

    def test_aggressive_policy(self):
        """Test aggressive termination policy for math."""
        config = EnvConfig(
            name="math-python-aggressive",
            dataset=DatasetSettings(
                environment=EnvironmentType.MATH_PYTHON,
                limit=1
            ),
            rollout_termination_policy=RolloutTerminationPolicy(
                terminate_on_format_violation=True,
                max_consecutive_errors=2,  # Strict
                max_total_errors=3,
                min_code_blocks_before_answer=1
            )
        )

        env = MathPythonEnv(config=config)
        policy = env.config.rollout_termination_policy

        assert policy.max_consecutive_errors == 2
        assert policy.max_total_errors == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
