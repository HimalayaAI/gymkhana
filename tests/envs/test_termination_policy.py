"""Tests for rollout termination policy configuration and checking."""

import pytest
from datetime import datetime, timezone

from gymkhana.envs.config import RolloutTerminationPolicy, EnvConfig
from gymkhana.core.models.trajectory import RolloutStatus, RolloutState


class TestRolloutTerminationPolicy:
    """Test RolloutTerminationPolicy configuration."""

    def test_default_policy(self):
        """Test default policy values."""
        policy = RolloutTerminationPolicy()

        assert policy.terminate_on_format_violation is True
        assert policy.max_format_violations == 1
        assert policy.max_consecutive_errors == 3
        assert policy.max_total_errors == 5
        assert policy.enable_max_turns_termination is True
        assert policy.min_code_blocks_before_answer == 1
        assert policy.enable_comparative_termination is False

    def test_custom_policy(self):
        """Test custom policy configuration."""
        policy = RolloutTerminationPolicy(
            terminate_on_format_violation=False,
            max_format_violations=2,
            max_consecutive_errors=5,
            max_total_errors=8,
            min_code_blocks_before_answer=2
        )

        assert policy.terminate_on_format_violation is False
        assert policy.max_format_violations == 2
        assert policy.max_consecutive_errors == 5
        assert policy.max_total_errors == 8
        assert policy.min_code_blocks_before_answer == 2

    def test_policy_in_env_config(self):
        """Test that policy is integrated into EnvConfig."""
        config = EnvConfig(
            name="test",
            rollout_termination_policy=RolloutTerminationPolicy(
                max_consecutive_errors=2
            )
        )

        assert config.rollout_termination_policy.max_consecutive_errors == 2

    def test_policy_validation(self):
        """Test that policy validates constraints."""
        # Should not allow negative values
        with pytest.raises(Exception):  # Pydantic validation error
            RolloutTerminationPolicy(max_consecutive_errors=-1)

        with pytest.raises(Exception):
            RolloutTerminationPolicy(max_total_errors=0)


class TestTerminationPolicyChecker:
    """Test the _check_termination_policy method."""

    def setup_method(self):
        """Set up test environment."""
        from gymkhana.envs.math_python import MathPythonEnv
        from gymkhana.envs.config import EnvConfig, DatasetSettings, EnvironmentType

        config = EnvConfig(
            name="test",
            dataset=DatasetSettings(
                environment=EnvironmentType.MATH_PYTHON,
                limit=1
            ),
            rollout_termination_policy=RolloutTerminationPolicy(
                terminate_on_format_violation=True,
                max_consecutive_errors=3,
                max_total_errors=5
            )
        )
        self.env = MathPythonEnv(config=config)

    def test_no_termination_clean_state(self):
        """Test that clean state doesn't trigger termination."""
        state = RolloutState(
            rollout_id=0,
            status=RolloutStatus.ACTIVE,
            started_at=datetime.now(timezone.utc)
        )

        should_terminate, reason = self.env._check_termination_policy(
            state, [state], turn_idx=0
        )

        assert should_terminate is False
        assert reason is None

    def test_format_violation_termination(self):
        """Test termination on format violation."""
        state = RolloutState(
            rollout_id=0,
            status=RolloutStatus.ACTIVE,
            num_format_violations=1,
            started_at=datetime.now(timezone.utc)
        )

        should_terminate, reason = self.env._check_termination_policy(
            state, [state], turn_idx=0
        )

        assert should_terminate is True
        assert "Format violation" in reason

    def test_consecutive_errors_termination(self):
        """Test termination on consecutive errors."""
        state = RolloutState(
            rollout_id=0,
            status=RolloutStatus.ACTIVE,
            consecutive_errors=3,
            num_errors=3,
            started_at=datetime.now(timezone.utc)
        )

        should_terminate, reason = self.env._check_termination_policy(
            state, [state], turn_idx=0
        )

        assert should_terminate is True
        assert "Consecutive errors" in reason
        assert "3/3" in reason

    def test_total_errors_termination(self):
        """Test termination on total errors."""
        state = RolloutState(
            rollout_id=0,
            status=RolloutStatus.ACTIVE,
            consecutive_errors=1,  # Not enough to trigger consecutive
            num_errors=5,  # But enough total errors
            started_at=datetime.now(timezone.utc)
        )

        should_terminate, reason = self.env._check_termination_policy(
            state, [state], turn_idx=0
        )

        assert should_terminate is True
        assert "Total errors" in reason
        assert "5/5" in reason

    def test_max_turns_without_progress(self):
        """Test termination at max turns without code execution."""
        state = RolloutState(
            rollout_id=0,
            status=RolloutStatus.ACTIVE,
            num_code_blocks=0,  # No successful executions
            started_at=datetime.now(timezone.utc)
        )

        # At max_turns - 1 (last turn)
        max_turns = self.env.config.repl.max_turns
        should_terminate, reason = self.env._check_termination_policy(
            state, [state], turn_idx=max_turns - 1
        )

        assert should_terminate is True
        assert "Max turns" in reason
        assert "without any successful code execution" in reason

    def test_max_turns_with_progress_no_termination(self):
        """Test that max turns doesn't terminate if code was executed."""
        state = RolloutState(
            rollout_id=0,
            status=RolloutStatus.ACTIVE,
            num_code_blocks=5,  # Has successful executions
            started_at=datetime.now(timezone.utc)
        )

        max_turns = self.env.config.repl.max_turns
        should_terminate, reason = self.env._check_termination_policy(
            state, [state], turn_idx=max_turns - 1
        )

        assert should_terminate is False
        assert reason is None

    def test_priority_format_over_errors(self):
        """Test that format violations have highest priority."""
        state = RolloutState(
            rollout_id=0,
            status=RolloutStatus.ACTIVE,
            num_format_violations=1,
            consecutive_errors=3,
            num_errors=5,
            started_at=datetime.now(timezone.utc)
        )

        should_terminate, reason = self.env._check_termination_policy(
            state, [state], turn_idx=0
        )

        assert should_terminate is True
        # Should mention format violation, not errors
        assert "Format violation" in reason

    def test_disabled_format_termination(self):
        """Test that format termination can be disabled."""
        # Create env with format termination disabled
        from gymkhana.envs.math_python import MathPythonEnv
        from gymkhana.envs.config import EnvConfig, DatasetSettings, EnvironmentType

        config = EnvConfig(
            name="test",
            dataset=DatasetSettings(
                environment=EnvironmentType.MATH_PYTHON,
                limit=1
            ),
            rollout_termination_policy=RolloutTerminationPolicy(
                terminate_on_format_violation=False,
                max_format_violations=2
            )
        )
        env = MathPythonEnv(config=config)

        state = RolloutState(
            rollout_id=0,
            status=RolloutStatus.ACTIVE,
            num_format_violations=1,
            started_at=datetime.now(timezone.utc)
        )

        should_terminate, reason = env._check_termination_policy(
            state, [state], turn_idx=0
        )

        # Should not terminate on first violation
        assert should_terminate is False

        # But should terminate on second
        state.num_format_violations = 2
        should_terminate, reason = env._check_termination_policy(
            state, [state], turn_idx=0
        )
        assert should_terminate is True

    def test_comparative_termination_not_implemented(self):
        """Test that comparative termination is not yet implemented."""
        # Create env with comparative enabled
        from gymkhana.envs.math_python import MathPythonEnv
        from gymkhana.envs.config import EnvConfig, DatasetSettings, EnvironmentType

        config = EnvConfig(
            name="test",
            dataset=DatasetSettings(
                environment=EnvironmentType.MATH_PYTHON,
                limit=1
            ),
            rollout_termination_policy=RolloutTerminationPolicy(
                enable_comparative_termination=True
            )
        )
        env = MathPythonEnv(config=config)

        # Create multiple states with different error counts
        states = [
            RolloutState(
                rollout_id=0,
                status=RolloutStatus.ACTIVE,
                num_errors=0,
                started_at=datetime.now(timezone.utc)
            ),
            RolloutState(
                rollout_id=1,
                status=RolloutStatus.ACTIVE,
                num_errors=5,  # Much worse than rollout 0
                started_at=datetime.now(timezone.utc)
            ),
        ]

        # Should not terminate based on comparison (not implemented)
        should_terminate, reason = env._check_termination_policy(
            states[1], states, turn_idx=0
        )

        # Will only terminate if hits other thresholds
        assert should_terminate is True  # Because num_errors=5 >= max_total_errors=5
        assert "Total errors" in reason  # Not comparative reason


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
