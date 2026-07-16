"""Tests for rollout tracking models."""

import pytest
from datetime import datetime, timezone

from gymkhana.core.models.trajectory import (
    RolloutStatus,
    RolloutState,
    PipelineStats,
)


class TestRolloutStatus:
    """Test RolloutStatus enum."""

    def test_status_values(self):
        """Test that all status values are defined."""
        assert RolloutStatus.ACTIVE == "active"
        assert RolloutStatus.COMPLETED == "completed"
        assert RolloutStatus.FAILED == "failed"
        assert RolloutStatus.ERROR == "error"
        assert RolloutStatus.TIMEOUT == "timeout"


class TestRolloutState:
    """Test RolloutState model."""

    def test_initialization(self):
        """Test RolloutState initialization with defaults."""
        state = RolloutState(
            rollout_id=0,
            status=RolloutStatus.ACTIVE,
            started_at=datetime.now(timezone.utc)
        )

        assert state.rollout_id == 0
        assert state.status == RolloutStatus.ACTIVE
        assert state.num_errors == 0
        assert state.consecutive_errors == 0
        assert state.num_format_violations == 0
        assert state.total_reward == 0.0
        assert state.termination_reason is None
        assert state.completed_at is None

    def test_is_active(self):
        """Test is_active method."""
        state = RolloutState(
            rollout_id=0,
            status=RolloutStatus.ACTIVE,
            started_at=datetime.now(timezone.utc)
        )
        assert state.is_active() is True

        state.status = RolloutStatus.COMPLETED
        assert state.is_active() is False

    def test_mark_completed(self):
        """Test mark_completed method."""
        state = RolloutState(
            rollout_id=0,
            status=RolloutStatus.ACTIVE,
            started_at=datetime.now(timezone.utc)
        )

        state.mark_completed()

        assert state.status == RolloutStatus.COMPLETED
        assert state.completed_at is not None
        assert state.termination_reason is None

    def test_mark_failed(self):
        """Test mark_failed method."""
        state = RolloutState(
            rollout_id=0,
            status=RolloutStatus.ACTIVE,
            started_at=datetime.now(timezone.utc)
        )

        reason = "Consecutive errors: 3"
        state.mark_failed(reason)

        assert state.status == RolloutStatus.FAILED
        assert state.termination_reason == reason
        assert state.completed_at is not None

    def test_mark_error(self):
        """Test mark_error method."""
        state = RolloutState(
            rollout_id=0,
            status=RolloutStatus.ACTIVE,
            started_at=datetime.now(timezone.utc)
        )

        error_msg = "Exception during execution"
        state.mark_error(error_msg)

        assert state.status == RolloutStatus.ERROR
        assert state.termination_reason == error_msg
        assert state.completed_at is not None

    def test_mark_timeout(self):
        """Test mark_timeout method."""
        state = RolloutState(
            rollout_id=0,
            status=RolloutStatus.ACTIVE,
            started_at=datetime.now(timezone.utc)
        )

        state.mark_timeout()

        assert state.status == RolloutStatus.TIMEOUT
        assert state.termination_reason == "Max turns reached without completion"
        assert state.completed_at is not None

    def test_record_execution_success(self):
        """Test record_execution with successful execution."""
        state = RolloutState(
            rollout_id=0,
            status=RolloutStatus.ACTIVE,
            started_at=datetime.now(timezone.utc)
        )

        state.record_execution(success=True, reward=1.0)

        assert state.num_code_blocks == 1
        assert state.num_errors == 0
        assert state.consecutive_errors == 0
        assert state.total_reward == 1.0
        assert state.last_reward == 1.0

    def test_record_execution_error(self):
        """Test record_execution with error."""
        state = RolloutState(
            rollout_id=0,
            status=RolloutStatus.ACTIVE,
            started_at=datetime.now(timezone.utc)
        )

        state.record_execution(success=False, reward=-0.1)

        assert state.num_code_blocks == 1
        assert state.num_errors == 1
        assert state.consecutive_errors == 1
        assert state.total_reward == -0.1
        assert state.last_reward == -0.1

    def test_consecutive_errors_reset(self):
        """Test that consecutive errors reset on success."""
        state = RolloutState(
            rollout_id=0,
            status=RolloutStatus.ACTIVE,
            started_at=datetime.now(timezone.utc)
        )

        # Record 2 errors
        state.record_execution(success=False)
        state.record_execution(success=False)
        assert state.consecutive_errors == 2
        assert state.num_errors == 2

        # Success should reset consecutive but not total
        state.record_execution(success=True)
        assert state.consecutive_errors == 0
        assert state.num_errors == 2

    def test_format_violation_tracking(self):
        """Test format violation tracking."""
        state = RolloutState(
            rollout_id=0,
            status=RolloutStatus.ACTIVE,
            started_at=datetime.now(timezone.utc)
        )

        state.record_execution(
            success=True,
            has_format_violation=True,
            format_violation_type="hallucinated_tags"
        )

        assert state.num_format_violations == 1
        assert state.consecutive_format_violations == 1
        assert state.last_format_violation_type == "hallucinated_tags"
        assert "hallucinated_tags" in state.format_violation_history

    def test_format_violation_reset(self):
        """Test that consecutive format violations reset on valid format."""
        state = RolloutState(
            rollout_id=0,
            status=RolloutStatus.ACTIVE,
            started_at=datetime.now(timezone.utc)
        )

        # Record 2 format violations
        state.record_execution(success=True, has_format_violation=True)
        state.record_execution(success=True, has_format_violation=True)
        assert state.consecutive_format_violations == 2
        assert state.num_format_violations == 2

        # Valid format should reset consecutive but not total
        state.record_execution(success=True, has_format_violation=False)
        assert state.consecutive_format_violations == 0
        assert state.num_format_violations == 2

    def test_duration_ms(self):
        """Test duration_ms calculation."""
        import time

        state = RolloutState(
            rollout_id=0,
            status=RolloutStatus.ACTIVE,
            started_at=datetime.now(timezone.utc)
        )

        # No completion yet
        assert state.duration_ms() is None

        # Complete after small delay
        time.sleep(0.01)
        state.mark_completed()

        duration = state.duration_ms()
        assert duration is not None
        assert duration > 0


class TestPipelineStatsRolloutTracking:
    """Test PipelineStats rollout tracking features."""

    def test_record_rollout_completed(self):
        """Test recording a completed rollout."""
        stats = PipelineStats()
        state = RolloutState(
            rollout_id=0,
            status=RolloutStatus.COMPLETED,
            started_at=datetime.now(timezone.utc)
        )

        stats.record_rollout(state)

        assert stats.total_rollouts == 1
        assert stats.rollouts_completed == 1
        assert stats.rollouts_terminated_early == 0

    def test_record_rollout_failed(self):
        """Test recording a failed rollout."""
        stats = PipelineStats()
        state = RolloutState(
            rollout_id=0,
            status=RolloutStatus.FAILED,
            started_at=datetime.now(timezone.utc)
        )
        state.mark_failed("Consecutive errors: 3")

        stats.record_rollout(state)

        assert stats.total_rollouts == 1
        assert stats.rollouts_completed == 0
        assert stats.rollouts_terminated_early == 1
        assert stats.termination_reasons["Consecutive errors: 3"] == 1

    def test_record_rollout_error(self):
        """Test recording an error rollout."""
        stats = PipelineStats()
        state = RolloutState(
            rollout_id=0,
            status=RolloutStatus.ERROR,
            started_at=datetime.now(timezone.utc)
        )

        stats.record_rollout(state)

        assert stats.total_rollouts == 1
        assert stats.rollouts_error == 1

    def test_record_rollout_timeout(self):
        """Test recording a timeout rollout."""
        stats = PipelineStats()
        state = RolloutState(
            rollout_id=0,
            status=RolloutStatus.TIMEOUT,
            started_at=datetime.now(timezone.utc)
        )

        stats.record_rollout(state)

        assert stats.total_rollouts == 1
        assert stats.rollouts_timeout == 1

    def test_multiple_rollouts(self):
        """Test recording multiple rollouts."""
        stats = PipelineStats()

        # 2 completed
        for i in range(2):
            state = RolloutState(
                rollout_id=i,
                status=RolloutStatus.COMPLETED,
                started_at=datetime.now(timezone.utc)
            )
            stats.record_rollout(state)

        # 1 failed
        state = RolloutState(
            rollout_id=2,
            status=RolloutStatus.FAILED,
            started_at=datetime.now(timezone.utc)
        )
        state.mark_failed("Total errors: 5")
        stats.record_rollout(state)

        # 1 timeout
        state = RolloutState(
            rollout_id=3,
            status=RolloutStatus.TIMEOUT,
            started_at=datetime.now(timezone.utc)
        )
        stats.record_rollout(state)

        assert stats.total_rollouts == 4
        assert stats.rollouts_completed == 2
        assert stats.rollouts_terminated_early == 1
        assert stats.rollouts_timeout == 1

    def test_report_with_rollouts(self):
        """Test that report includes rollout statistics."""
        stats = PipelineStats()

        # Add some rollout data
        state = RolloutState(
            rollout_id=0,
            status=RolloutStatus.COMPLETED,
            started_at=datetime.now(timezone.utc)
        )
        stats.record_rollout(state)

        state = RolloutState(
            rollout_id=1,
            status=RolloutStatus.FAILED,
            started_at=datetime.now(timezone.utc)
        )
        state.mark_failed("Consecutive errors: 3")
        stats.record_rollout(state)

        report = stats.report()

        assert "Rollout Statistics" in report
        assert "Total rollouts:" in report
        assert "Completed:" in report
        assert "Terminated early:" in report
        assert "Termination Reasons" in report
        assert "Consecutive errors: 3" in report


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
