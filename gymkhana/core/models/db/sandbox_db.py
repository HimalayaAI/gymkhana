"""Database models for sandbox data.

These models should ONLY be imported by storage/migration code.
"""
from __future__ import annotations

from datetime import datetime
from json import dumps, loads
from typing import Any, Dict, Optional
from uuid import UUID

from sqlalchemy import String, Integer, Float, Text, DateTime, Index
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.dialects.postgresql import UUID as PGUUID

from gymkhana.core.models.entity import Base, DBEntityMixin


class SandboxSessionDB(Base, DBEntityMixin):
    """Database model for SessionState."""
    __tablename__ = "sandbox_sessions"
    __table_args__ = (
        Index('idx_sandbox_sessions_session_id', 'session_id'),
        Index('idx_sandbox_sessions_environment', 'environment'),
        Index('idx_sandbox_sessions_trajectory_id', 'trajectory_id'),
    )

    session_id: Mapped[str] = mapped_column(String(255), nullable=False)
    status: Mapped[str] = mapped_column(String(50), nullable=False)

    # Timestamps
    ready_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    last_execution_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    # Metrics
    total_reward: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    total_executions: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    successful_executions: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    failed_executions: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    total_execution_time_ms: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    # JSON states
    interpreter_json: Mapped[str] = mapped_column(Text, default="{}", nullable=False)
    episode_json: Mapped[str] = mapped_column(Text, default="{}", nullable=False)
    config_json: Mapped[str] = mapped_column(Text, default="{}", nullable=False)

    # Relationships/Metadata
    environment: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    trajectory_id: Mapped[Optional[UUID]] = mapped_column(PGUUID(as_uuid=True), nullable=True)

    @property
    def interpreter_state(self) -> Dict[str, Any]:
        return loads(self.interpreter_json) if self.interpreter_json else {}

    @interpreter_state.setter
    def interpreter_state(self, value: Dict[str, Any]) -> None:
        self.interpreter_json = dumps(value)

    @property
    def episode_state(self) -> Dict[str, Any]:
        return loads(self.episode_json) if self.episode_json else {}

    @episode_state.setter
    def episode_state(self, value: Dict[str, Any]) -> None:
        self.episode_json = dumps(value)

    @property
    def session_config(self) -> Dict[str, Any]:
        return loads(self.config_json) if self.config_json else {}

    @session_config.setter
    def session_config(self, value: Dict[str, Any]) -> None:
        self.config_json = dumps(value)


__all__ = ["SandboxSessionDB"]
