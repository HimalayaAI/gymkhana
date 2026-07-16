"""Database models for trajectory data.

These models should ONLY be imported by storage/migration code.
"""
from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from sqlalchemy import String, Integer, Float, Boolean, DateTime, Text, ForeignKey, Index
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.dialects.postgresql import UUID as PGUUID

from gymkhana.core.models.entity import Base, DBEntityMixin


class TurnDB(Base, DBEntityMixin):
    """Database model for Turn."""
    __tablename__ = "turns"

    trajectory_id: Mapped[Optional[UUID]] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("trajectories.id", ondelete="CASCADE"),
        nullable=True,
        index=True
    )
    role: Mapped[str] = mapped_column(String(50), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    code: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    turn_index: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    reasoning_content: Mapped[Optional[str]] = mapped_column(Text, nullable=True)


class TrajectoryResultDB(Base, DBEntityMixin):
    """Database model for TrajectoryResult."""
    __tablename__ = "trajectories"
    __table_args__ = (
        Index('idx_trajectories_task_id', 'task_id'),
        Index('idx_trajectories_environment', 'environment'),
        Index('idx_trajectories_model_name', 'model_name'),
        Index('idx_trajectories_answer_correct', 'answer_correct'),
        Index('idx_trajectories_total_reward', 'total_reward'),
    )

    # Rollout references
    rollout_id: Mapped[Optional[UUID]] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("rollouts.id", ondelete="SET NULL"),
        nullable=True,
        index=True
    )
    rollout_group_id: Mapped[Optional[UUID]] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("rollout_groups.id", ondelete="SET NULL"),
        nullable=True,
        index=True
    )
    rollout_index: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    # Task reference
    task_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    environment: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)

    # Interaction mode tracking
    interaction_mode: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    conversation_manager: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    max_turns: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    # Core fields
    success: Mapped[bool] = mapped_column(Boolean, nullable=False)
    final_answer: Mapped[str] = mapped_column(Text, default="", nullable=False)
    expected_answer: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    answer_correct: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)

    num_code_blocks: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    num_turns: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    system_prompt: Mapped[str] = mapped_column(Text, default="", nullable=False)
    num_errors: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    # Reward fields
    total_reward: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    step_rewards_json: Mapped[str] = mapped_column(Text, default="[]", nullable=False)
    reward_function: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)

    # Quality metrics
    efficiency_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    quality_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Model tracking
    model_name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    metadata_json: Mapped[str] = mapped_column(Text, default="{}", nullable=False)

    @property
    def step_rewards(self) -> List[float]:
        """Deserialize step rewards from JSON."""
        return json.loads(self.step_rewards_json) if self.step_rewards_json else []

    @step_rewards.setter
    def step_rewards(self, value: List[float]) -> None:
        """Serialize step rewards to JSON."""
        self.step_rewards_json = json.dumps(value)

    @property
    def extra_metadata(self) -> Dict[str, Any]:
        """Deserialize metadata from JSON."""
        return json.loads(self.metadata_json) if self.metadata_json else {}

    @extra_metadata.setter
    def extra_metadata(self, value: Dict[str, Any]) -> None:
        """Serialize metadata to JSON."""
        self.metadata_json = json.dumps(value)


class RolloutGroupDB(Base, DBEntityMixin):
    """Database model for RolloutGroup."""
    __tablename__ = "rollout_groups"
    __table_args__ = (
        Index('idx_rollout_groups_task_id', 'task_id'),
        Index('idx_rollout_groups_environment', 'environment'),
    )

    task_id: Mapped[str] = mapped_column(String(255), nullable=False)
    environment: Mapped[str] = mapped_column(String(100), nullable=False)
    num_rollouts: Mapped[int] = mapped_column(Integer, nullable=False)
    num_completed: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    num_failed: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    num_error: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    num_timeout: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    best_rollout_id: Mapped[Optional[UUID]] = mapped_column(PGUUID(as_uuid=True), nullable=True)
    best_reward: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    reward_mean: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    reward_std: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    reward_min: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    config_json: Mapped[str] = mapped_column(Text, default="{}", nullable=False)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    @property
    def config(self) -> Dict[str, Any]:
        """Deserialize config from JSON."""
        return json.loads(self.config_json) if self.config_json else {}

    @config.setter
    def config(self, value: Dict[str, Any]) -> None:
        """Serialize config to JSON."""
        self.config_json = json.dumps(value)


class RolloutStateDB(Base, DBEntityMixin):
    """Database model for RolloutState."""
    __tablename__ = "rollouts"
    __table_args__ = (
        Index('idx_rollouts_group_id', 'rollout_group_id'),
        Index('idx_rollouts_trajectory_id', 'trajectory_id'),
        Index('idx_rollouts_status', 'status'),
        Index('idx_rollouts_total_reward', 'total_reward'),
    )

    rollout_group_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("rollout_groups.id", ondelete="CASCADE"),
        nullable=False
    )
    trajectory_id: Mapped[Optional[UUID]] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("trajectories.id", ondelete="SET NULL"),
        nullable=True
    )
    rollout_index: Mapped[int] = mapped_column(Integer, nullable=False)

    # Status
    status: Mapped[str] = mapped_column(String(50), nullable=False)
    termination_reason: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Execution metrics
    num_turns: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    num_code_blocks: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    num_errors: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    consecutive_errors: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    # Format violations
    num_format_violations: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    consecutive_format_violations: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    last_format_violation_type: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    format_violation_history_json: Mapped[str] = mapped_column(Text, default="[]", nullable=False)

    # Rewards
    total_reward: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)

    # Timing
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    duration_ms: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Sandbox
    session_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)

    @property
    def format_violation_history(self) -> List[str]:
        """Deserialize format violation history from JSON."""
        return json.loads(self.format_violation_history_json) if self.format_violation_history_json else []

    @format_violation_history.setter
    def format_violation_history(self, value: List[str]) -> None:
        """Serialize format violation history to JSON."""
        self.format_violation_history_json = json.dumps(value)


__all__ = [
    "TurnDB",
    "TrajectoryResultDB",
    "RolloutGroupDB",
    "RolloutStateDB",
]
