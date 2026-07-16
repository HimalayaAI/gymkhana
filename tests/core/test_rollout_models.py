"""Unit tests for rollout tracking models."""

import json
from datetime import datetime, timezone
from uuid import uuid4

import pytest

from gymkhana.core.models.trajectory import (
    RolloutGroup,
    RolloutState,
    RolloutStatus,
    TrajectoryResult,
)
from gymkhana.core.models.db import (
    RolloutGroupDB,
    RolloutStateDB,
)


class TestRolloutGroup:
    """Tests for RolloutGroup model."""

    def test_create_rollout_group(self):
        """Test creating a rollout group."""
        group = RolloutGroup(
            task_id="test_task_123",
            environment="math_reasoning",
            num_rollouts=8
        )

        assert group.task_id == "test_task_123"
        assert group.environment == "math_reasoning"
        assert group.num_rollouts == 8
        assert group.num_completed == 0
        assert group.num_failed == 0
        assert group.best_reward == 0.0
        assert group.reward_mean == 0.0
        assert group.id is not None
        assert group.created_at is not None

    def test_update_statistics(self):
        """Test updating reward statistics."""
        group = RolloutGroup(
            task_id="test_task",
            environment="test_env",
            num_rollouts=5
        )

        rewards = [0.5, 0.7, 0.9, 0.6, 0.8]
        group.update_statistics(rewards)

        assert group.reward_mean == pytest.approx(0.7, rel=1e-2)
        assert group.reward_std > 0
        assert group.reward_min == 0.5
        assert group.best_reward == 0.9

    def test_update_statistics_single_rollout(self):
        """Test statistics with single rollout (std dev should be 0)."""
        group = RolloutGroup(
            task_id="test_task",
            environment="test_env",
            num_rollouts=1
        )

        rewards = [0.75]
        group.update_statistics(rewards)

        assert group.reward_mean == 0.75
        assert group.reward_std == 0.0
        assert group.reward_min == 0.75
        assert group.best_reward == 0.75

    def test_update_statistics_empty(self):
        """Test statistics with empty rewards list."""
        group = RolloutGroup(
            task_id="test_task",
            environment="test_env",
            num_rollouts=0
        )

        group.update_statistics([])

        # Should not crash, values remain at defaults
        assert group.reward_mean == 0.0
        assert group.reward_std == 0.0

    def test_config_serialization(self):
        """Test config dictionary serialization."""
        config = {
            "termination_policy": {
                "max_consecutive_errors": 3,
                "max_total_errors": 5
            },
            "reward_function": "efficiency_aware"
        }

        group = RolloutGroup(
            task_id="test_task",
            environment="test_env",
            num_rollouts=4,
            config=config
        )

        assert group.config == config
        assert "termination_policy" in group.config


class TestRolloutGroupDB:
    """Tests for RolloutGroupDB model."""

    def test_create_db_model(self):
        """Test creating database model."""
        group_db = RolloutGroupDB(
            task_id="test_task",
            environment="test_env",
            num_rollouts=8
        )

        assert group_db.task_id == "test_task"
        assert group_db.environment == "test_env"
        assert group_db.num_rollouts == 8
        assert group_db.id is not None

    def test_config_property(self):
        """Test config JSON serialization property."""
        config = {"key": "value", "nested": {"a": 1}}

        group_db = RolloutGroupDB(
            task_id="test_task",
            environment="test_env",
            num_rollouts=4
        )

        # Set via property
        group_db.config = config

        # Verify JSON storage
        assert group_db.config_json == json.dumps(config)

        # Get via property
        assert group_db.config == config

    def test_config_property_empty(self):
        """Test config property with empty/default value."""
        group_db = RolloutGroupDB(
            task_id="test_task",
            environment="test_env",
            num_rollouts=4
        )

        assert group_db.config == {}


class TestRolloutState:
    """Tests for RolloutState model."""

    def test_create_rollout_state(self):
        """Test creating a rollout state."""
        state = RolloutState(
            rollout_id=0,
            status=RolloutStatus.ACTIVE
        )

        assert state.rollout_id == 0
        assert state.status == RolloutStatus.ACTIVE
        assert state.num_turns == 0
        assert state.num_errors == 0
        assert state.consecutive_errors == 0
        assert state.total_reward == 0.0
        assert state.is_active()

    def test_mark_completed(self):
        """Test marking rollout as completed."""
        state = RolloutState(rollout_id=0)

        assert state.completed_at is None
        state.mark_completed()

        assert state.status == RolloutStatus.COMPLETED
        assert state.completed_at is not None

    def test_mark_failed(self):
        """Test marking rollout as failed."""
        state = RolloutState(rollout_id=0)

        reason = "Consecutive errors: 3"
        state.mark_failed(reason)

        assert state.status == RolloutStatus.FAILED
        assert state.termination_reason == reason
        assert state.completed_at is not None
        assert not state.is_active()

    def test_mark_error(self):
        """Test marking rollout as error."""
        state = RolloutState(rollout_id=0)

        error_msg = "Exception during execution"
        state.mark_error(error_msg)

        assert state.status == RolloutStatus.ERROR
        assert state.termination_reason == error_msg
        assert state.completed_at is not None

    def test_mark_timeout(self):
        """Test marking rollout as timeout."""
        state = RolloutState(rollout_id=0)

        state.mark_timeout()

        assert state.status == RolloutStatus.TIMEOUT
        assert "Max turns" in state.termination_reason
        assert state.completed_at is not None

    def test_record_execution_success(self):
        """Test recording successful execution."""
        state = RolloutState(rollout_id=0)

        state.record_execution(success=True, reward=0.5)

        assert state.num_code_blocks == 1
        assert state.num_errors == 0
        assert state.consecutive_errors == 0
        assert state.total_reward == 0.5
        assert state.last_reward == 0.5

    def test_record_execution_failure(self):
        """Test recording failed execution."""
        state = RolloutState(rollout_id=0)

        state.record_execution(success=False, reward=-0.1)

        assert state.num_code_blocks == 1
        assert state.num_errors == 1
        assert state.consecutive_errors == 1
        assert state.total_reward == -0.1

    def test_record_execution_consecutive_errors(self):
        """Test consecutive error tracking."""
        state = RolloutState(rollout_id=0)

        # First error
        state.record_execution(success=False)
        assert state.consecutive_errors == 1

        # Second error
        state.record_execution(success=False)
        assert state.consecutive_errors == 2

        # Success resets consecutive errors
        state.record_execution(success=True)
        assert state.consecutive_errors == 0
        assert state.num_errors == 2  # Total errors not reset

    def test_record_execution_format_violation(self):
        """Test format violation tracking."""
        state = RolloutState(rollout_id=0)

        state.record_execution(
            success=True,
            has_format_violation=True,
            format_violation_type="hallucinated_tags"
        )

        assert state.num_format_violations == 1
        assert state.consecutive_format_violations == 1
        assert state.last_format_violation_type == "hallucinated_tags"
        assert "hallucinated_tags" in state.format_violation_history

    def test_record_execution_format_violation_reset(self):
        """Test format violation consecutive counter reset."""
        state = RolloutState(rollout_id=0)

        # Violation
        state.record_execution(success=True, has_format_violation=True)
        assert state.consecutive_format_violations == 1

        # No violation resets consecutive
        state.record_execution(success=True, has_format_violation=False)
        assert state.consecutive_format_violations == 0
        assert state.num_format_violations == 1  # Total not reset

    def test_duration_ms(self):
        """Test duration calculation."""
        state = RolloutState(rollout_id=0)

        # No completion time
        assert state.duration_ms() is None

        # Set completion time
        state.started_at = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        state.completed_at = datetime(2026, 1, 1, 12, 0, 5, tzinfo=timezone.utc)

        assert state.duration_ms() == pytest.approx(5000.0, rel=1e-2)


class TestRolloutStateDB:
    """Tests for RolloutStateDB model."""

    def test_create_db_model(self):
        """Test creating database model."""
        group_id = uuid4()

        state_db = RolloutStateDB(
            rollout_group_id=group_id,
            rollout_index=0,
            status="active",
            started_at=datetime.now(timezone.utc)
        )

        assert state_db.rollout_group_id == group_id
        assert state_db.rollout_index == 0
        assert state_db.status == "active"
        assert state_db.id is not None

    def test_format_violation_history_property(self):
        """Test format violation history JSON property."""
        state_db = RolloutStateDB(
            rollout_group_id=uuid4(),
            rollout_index=0,
            status="active",
            started_at=datetime.now(timezone.utc)
        )

        history = ["hallucinated_tags", "malformed_xml", "hallucinated_tags"]

        # Set via property
        state_db.format_violation_history = history

        # Verify JSON storage
        assert state_db.format_violation_history_json == json.dumps(history)

        # Get via property
        assert state_db.format_violation_history == history

    def test_format_violation_history_empty(self):
        """Test format violation history with empty value."""
        state_db = RolloutStateDB(
            rollout_group_id=uuid4(),
            rollout_index=0,
            status="active",
            started_at=datetime.now(timezone.utc)
        )

        assert state_db.format_violation_history == []


class TestTrajectoryResultRolloutFields:
    """Tests for new rollout fields in TrajectoryResult."""

    def test_create_with_rollout_fields(self):
        """Test creating trajectory with rollout tracking fields."""
        rollout_id = uuid4()
        group_id = uuid4()

        result = TrajectoryResult(
            success=True,
            final_answer="42",
            rollout_id=rollout_id,
            rollout_group_id=group_id,
            rollout_index=3,
            answer_correct=True,
            reward_function="efficiency_aware",
            efficiency_score=0.85,
            quality_score=0.92
        )

        assert result.rollout_id == rollout_id
        assert result.rollout_group_id == group_id
        assert result.rollout_index == 3
        assert result.answer_correct is True
        assert result.reward_function == "efficiency_aware"
        assert result.efficiency_score == 0.85
        assert result.quality_score == 0.92

    def test_rollout_fields_optional(self):
        """Test that rollout fields are optional."""
        result = TrajectoryResult(
            success=True,
            final_answer="answer"
        )

        assert result.rollout_id is None
        assert result.rollout_group_id is None
        assert result.rollout_index is None
        assert result.answer_correct is None
        assert result.reward_function is None
        assert result.efficiency_score is None
        assert result.quality_score is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
