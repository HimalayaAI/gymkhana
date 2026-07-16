"""Trajectory models for gymkhana.

This module defines the core trajectory-related entities:
- Turn: A single conversation turn (user/assistant/tool)
- TrajectoryResult: Complete result of a trajectory rollout
- PipelineStats: Aggregate statistics for batch processing
- TrajectoryMetrics: Metrics for reward calculation
- RolloutStatus: Status enum for active rollout tracking
- RolloutState: Runtime state for individual rollouts

Pydantic models (Entity) are used for business logic.
SQLAlchemy models (Base + DBEntityMixin) are used for persistence.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, TYPE_CHECKING
from uuid import UUID

if TYPE_CHECKING:
    from gymkhana.core.models.execution import ExecutionResult

from pydantic import Field, ConfigDict

from gymkhana.core.models.entity import Entity


# =============================================================================
# Rollout Tracking Models (for active rollout tracking and early termination)
# =============================================================================

class RolloutStatus(str, Enum):
    """Status of an individual rollout during execution.

    Used for active rollout tracking to determine which rollouts should
    continue and which should be terminated early.
    """
    ACTIVE = "active"           # Currently executing
    COMPLETED = "completed"     # Successfully reached final answer
    FAILED = "failed"           # Terminated due to failure policy
    ERROR = "error"             # Exception during execution
    TIMEOUT = "timeout"         # Exceeded max turns without completion


class RolloutState(Entity):
    """Runtime state for a single rollout.

    Tracks execution progress, errors, format violations, and termination
    status for one rollout in a multi-rollout batch. Used to make termination
    decisions and filter active rollouts for batched inference.
    """
    model_config = ConfigDict(use_enum_values=True)

    # Identification
    rollout_id: int = Field(..., description="Index in batch (0 to G-1)")
    status: RolloutStatus = Field(
        default=RolloutStatus.ACTIVE,
        description="Current execution status"
    )
    session_id: Optional[str] = Field(
        default=None,
        description="Sandbox session ID for this rollout"
    )

    # Execution tracking
    num_turns: int = Field(default=0, description="Number of turns completed")
    num_code_blocks: int = Field(default=0, description="Number of code blocks executed")
    num_errors: int = Field(default=0, description="Total execution errors")
    consecutive_errors: int = Field(
        default=0,
        description="Consecutive errors (resets on success)"
    )

    # Format adherence tracking
    num_format_violations: int = Field(
        default=0,
        description="Count of format violations (hallucinated tags, malformed XML)"
    )
    consecutive_format_violations: int = Field(
        default=0,
        description="Consecutive format violations (resets on valid format)"
    )
    last_format_violation_type: Optional[str] = Field(
        default=None,
        description="Type of last format violation (e.g., 'hallucinated_tags')"
    )
    format_violation_history: List[str] = Field(
        default_factory=list,
        description="History of format violation types"
    )

    # Reward tracking
    total_reward: float = Field(default=0.0, description="Cumulative reward")
    last_reward: float = Field(default=0.0, description="Most recent step reward")

    # Termination tracking
    termination_reason: Optional[str] = Field(
        default=None,
        description="Reason for termination if status is FAILED"
    )
    started_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Rollout start timestamp"
    )
    completed_at: Optional[datetime] = Field(
        default=None,
        description="Rollout completion timestamp"
    )

    def duration_ms(self) -> Optional[float]:
        """Calculate rollout duration in milliseconds."""
        if self.completed_at is None:
            return None
        return (self.completed_at - self.started_at).total_seconds() * 1000

    def is_active(self) -> bool:
        """Check if rollout is still active."""
        return self.status == RolloutStatus.ACTIVE

    def mark_completed(self) -> None:
        """Mark rollout as completed."""
        self.status = RolloutStatus.COMPLETED
        if self.completed_at is None:
            self.completed_at = datetime.now(timezone.utc)

    def mark_failed(self, reason: str) -> None:
        """Mark rollout as failed with reason."""
        self.status = RolloutStatus.FAILED
        self.termination_reason = reason
        if self.completed_at is None:
            self.completed_at = datetime.now(timezone.utc)

    def mark_error(self, error_msg: str) -> None:
        """Mark rollout as error with message."""
        self.status = RolloutStatus.ERROR
        self.termination_reason = error_msg
        if self.completed_at is None:
            self.completed_at = datetime.now(timezone.utc)

    def mark_timeout(self) -> None:
        """Mark rollout as timeout."""
        self.status = RolloutStatus.TIMEOUT
        self.termination_reason = "Max turns reached without completion"
        if self.completed_at is None:
            self.completed_at = datetime.now(timezone.utc)

    def record_execution(
        self,
        success: bool,
        reward: float = 0.0,
        has_format_violation: bool = False,
        format_violation_type: Optional[str] = None
    ) -> None:
        """Record the result of a code execution."""
        self.num_code_blocks += 1
        self.total_reward += reward
        self.last_reward = reward

        if not success:
            self.num_errors += 1
            self.consecutive_errors += 1
        else:
            self.consecutive_errors = 0

        if has_format_violation:
            self.num_format_violations += 1
            self.consecutive_format_violations += 1
            if format_violation_type:
                self.last_format_violation_type = format_violation_type
                self.format_violation_history.append(format_violation_type)
        else:
            self.consecutive_format_violations = 0


# =============================================================================
# Trajectory Models
# =============================================================================

class Turn(Entity):
    """A single turn in the conversation.

    Represents one message in the multi-turn conversation between
    user, assistant, and tool (REPL or CallableTool).
    """
    role: str = Field(..., description='Role: "user", "assistant", or "tool"')
    content: str = Field(..., description="Message content")
    code: Optional[str] = Field(default=None, description="Code extracted from this turn")
    turn_index: int = Field(default=0, description="Position in trajectory (0-indexed)")
    execution: Optional[Any] = Field(default=None, description="Execution result if this was a tool turn")
    reasoning_content: Optional[str] = Field(
        default=None,
        description="Reasoning/thinking content from reasoning models (o1, o3, DeepSeek R1, GLM-5)"
    )
    # Tool-call fields (for TOOL_CALL / TOOL_CALL_INTERLEAVED modes)
    tool_calls: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Tool calls made in this assistant turn"
    )
    tool_call_id: Optional[str] = Field(
        default=None,
        description="Tool-call ID that this tool-result turn responds to"
    )


class TrajectoryResult(Entity):
    """Result of generating a trajectory.

    Captures the complete outcome of a multi-turn reasoning trajectory,
    including all conversation turns, final answer, and reward information.
    """
    # Success tracking
    success: bool = Field(..., description="Whether trajectory completed successfully")
    final_answer: str = Field(default="", description="Extracted final answer")
    turns: List[Turn] = Field(default_factory=list, description="Conversation turns")

    # Execution metrics
    num_code_blocks: int = Field(default=0, description="Number of code blocks executed")
    num_errors: int = Field(default=0, description="Count of execution errors")
    num_turns: int = Field(default=0, description="Actual number of turns taken")

    # Interaction mode tracking
    interaction_mode: Optional[str] = Field(default=None, description="Interaction mode used")
    conversation_manager: Optional[str] = Field(default=None, description="Conversation manager used")
    max_turns: Optional[int] = Field(default=None, description="Maximum turns allowed")

    # Task context
    task_id: Optional[str] = Field(default=None, description="Associated task ID")
    environment: Optional[str] = Field(default=None, description="Environment name")
    system_prompt: str = Field(default="", description="System prompt used")

    # Reward tracking
    total_reward: float = Field(default=0.0, description="Cumulative reward")
    step_rewards: List[float] = Field(default_factory=list, description="Per-step rewards")

    # Model tracking
    model_name: Optional[str] = Field(default=None, description="LLM model used")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Flexible metadata")

    # Sandbox tracking
    session_id: Optional[str] = Field(default=None, description="Sandbox session ID")
    sandbox_state: Optional[Any] = Field(default=None, description="Final sandbox state")

    # Rollout tracking
    rollout_id: Optional[UUID] = Field(default=None, description="Parent rollout ID")
    rollout_group_id: Optional[UUID] = Field(default=None, description="Parent rollout group ID")
    rollout_index: Optional[int] = Field(default=None, description="Index in rollout batch")

    # Evaluation metrics
    answer_correct: Optional[bool] = Field(default=None, description="Whether answer was correct")
    reward_function: Optional[str] = Field(default=None, description="Reward function used")
    efficiency_score: Optional[float] = Field(default=None, description="Efficiency metric (0-1)")
    quality_score: Optional[float] = Field(default=None, description="Quality metric (0-1)")


class PipelineStats(Entity):
    """Track pipeline statistics across multiple trajectories."""
    total: int = Field(default=0, description="Total trajectories processed")
    successful: int = Field(default=0, description="Successful trajectories")
    failed: int = Field(default=0, description="Failed trajectories")
    answers_correct: int = Field(default=0, description="Correct answers")
    answers_incorrect: int = Field(default=0, description="Incorrect answers")
    answers_unknown: int = Field(default=0, description="Unknown answers")
    avg_turns: float = Field(default=0.0, description="Average turns per trajectory")
    avg_code_blocks: float = Field(default=0.0, description="Average code blocks")
    total_turns: int = Field(default=0, description="Total turns")
    total_code_blocks: int = Field(default=0, description="Total code blocks")

    # Rollout tracking statistics
    total_rollouts: int = Field(default=0, description="Total rollouts")
    rollouts_completed: int = Field(default=0, description="Completed rollouts")
    rollouts_terminated_early: int = Field(default=0, description="Terminated rollouts")
    rollouts_error: int = Field(default=0, description="Error rollouts")
    rollouts_timeout: int = Field(default=0, description="Timeout rollouts")
    termination_reasons: Dict[str, int] = Field(
        default_factory=dict,
        description="Count of each termination reason"
    )

    def record(
        self, result: TrajectoryResult, answer_correct: Optional[bool] = None
    ) -> None:
        """Record a trajectory result."""
        self.total += 1
        if result.success:
            self.successful += 1
        else:
            self.failed += 1

        if answer_correct is True:
            self.answers_correct += 1
        elif answer_correct is False:
            self.answers_incorrect += 1
        else:
            self.answers_unknown += 1

        self.total_turns += len(result.turns)
        self.total_code_blocks += result.num_code_blocks
        self.avg_turns = self.total_turns / self.total if self.total > 0 else 0.0
        self.avg_code_blocks = self.total_code_blocks / self.total if self.total > 0 else 0.0

    def record_rollout(self, state: RolloutState) -> None:
        """Record a rollout state for statistics."""
        self.total_rollouts += 1

        if state.status == RolloutStatus.COMPLETED:
            self.rollouts_completed += 1
        elif state.status == RolloutStatus.FAILED:
            self.rollouts_terminated_early += 1
            if state.termination_reason:
                self.termination_reasons[state.termination_reason] = (
                    self.termination_reasons.get(state.termination_reason, 0) + 1
                )
        elif state.status == RolloutStatus.ERROR:
            self.rollouts_error += 1
        elif state.status == RolloutStatus.TIMEOUT:
            self.rollouts_timeout += 1

    def report(self) -> str:
        """Generate a human-readable statistics report."""
        success_rate = (self.successful / self.total * 100) if self.total > 0 else 0
        validated = self.answers_correct + self.answers_incorrect
        accuracy = (self.answers_correct / validated * 100) if validated > 0 else 0

        rollout_completion_rate = (
            (self.rollouts_completed / self.total_rollouts * 100)
            if self.total_rollouts > 0 else 0
        )

        report = f"""
=== Pipeline Statistics ===
Total processed:    {self.total:,}
Successful:         {self.successful:,}
Failed:             {self.failed:,}
Success rate:       {success_rate:.1f}%
Avg turns:          {self.avg_turns:.1f}
Avg code blocks:    {self.avg_code_blocks:.1f}

=== Answer Validation ===
Correct:            {self.answers_correct:,}
Incorrect:          {self.answers_incorrect:,}
Unknown:            {self.answers_unknown:,}
Accuracy:           {accuracy:.1f}% (of {validated} validated)
"""

        if self.total_rollouts > 0:
            report += f"""
=== Rollout Statistics ===
Total rollouts:     {self.total_rollouts:,}
Completed:          {self.rollouts_completed:,} ({rollout_completion_rate:.1f}%)
Terminated early:   {self.rollouts_terminated_early:,}
Errors:             {self.rollouts_error:,}
Timeouts:           {self.rollouts_timeout:,}
"""

            if self.termination_reasons:
                report += "\n=== Termination Reasons ===\n"
                for reason, count in sorted(
                    self.termination_reasons.items(),
                    key=lambda x: x[1],
                    reverse=True
                ):
                    report += f"{reason:30s} {count:,}\n"

        return report


class TrajectoryMetrics(Entity):
    """Metrics extracted from a trajectory for reward calculation."""
    answer_correct: Optional[bool] = Field(default=None, description="True/False/None")
    num_turns: int = Field(..., description="Number of turns in trajectory")
    num_code_blocks: int = Field(..., description="Number of code blocks executed")
    num_errors: int = Field(..., description="Number of execution errors")
    max_turns: int = Field(..., description="Maximum turns allowed")
    intermediate_rewards: List[Any] = Field(
        default_factory=list, description="Per-step rewards"
    )
    success: bool = Field(..., description="Whether trajectory completed")


# =============================================================================
# Rollout Group Models (for GRPO training)
# =============================================================================

class RolloutGroup(Entity):
    """Group of parallel rollouts for the same task (for GRPO training)."""
    task_id: str = Field(..., description="Task ID from dataset")
    environment: str = Field(..., description="Environment name")
    num_rollouts: int = Field(..., description="Number of parallel rollouts (G)")

    # Aggregate statistics
    num_completed: int = Field(default=0, description="Completed rollouts")
    num_failed: int = Field(default=0, description="Failed rollouts")
    num_error: int = Field(default=0, description="Error rollouts")
    num_timeout: int = Field(default=0, description="Timeout rollouts")

    # Best rollout tracking
    best_rollout_id: Optional[UUID] = Field(default=None, description="Best rollout ID")
    best_reward: float = Field(default=0.0, description="Highest reward")

    # Reward distribution
    reward_mean: float = Field(default=0.0, description="Mean reward")
    reward_std: float = Field(default=0.0, description="Std dev of rewards")
    reward_min: float = Field(default=0.0, description="Minimum reward")

    # Configuration
    config: Dict[str, Any] = Field(default_factory=dict, description="Rollout config")
    completed_at: Optional[datetime] = Field(default=None, description="Completion time")

    def update_statistics(self, rollout_rewards: List[float]) -> None:
        """Update reward distribution statistics from rollout rewards."""
        if not rollout_rewards:
            return

        import statistics
        self.reward_mean = statistics.mean(rollout_rewards)
        self.reward_std = statistics.stdev(rollout_rewards) if len(rollout_rewards) > 1 else 0.0
        self.reward_min = min(rollout_rewards)
        self.best_reward = max(rollout_rewards)


__all__ = [
    "Turn",
    "TrajectoryResult",
    "PipelineStats",
    "TrajectoryMetrics",
    "RolloutStatus",
    "RolloutState",
    "RolloutGroup",
]
