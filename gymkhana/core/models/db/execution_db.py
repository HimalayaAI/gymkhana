"""Database models for execution data.

These models should ONLY be imported by storage/migration code.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional
from uuid import UUID

from sqlalchemy import String, Integer, Float, Boolean, Text, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.dialects.postgresql import UUID as PGUUID

from gymkhana.core.models.entity import Base, DBEntityMixin


class SubAgentCallDB(Base, DBEntityMixin):
    """Database model for SubAgentCall."""
    __tablename__ = "sub_agent_calls"

    execution_id: Mapped[Optional[UUID]] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("executions.id", ondelete="CASCADE"),
        nullable=True,
        index=True
    )
    task: Mapped[str] = mapped_column(Text, nullable=False)
    system_prompt: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    response: Mapped[str] = mapped_column(Text, default="", nullable=False)


class ExecutionResultDB(Base, DBEntityMixin):
    """Database model for ExecutionResult."""
    __tablename__ = "executions"

    # Reference to parent turn/trajectory
    turn_id: Mapped[Optional[UUID]] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("turns.id", ondelete="CASCADE"),
        nullable=True,
        index=True
    )

    # Core fields
    success: Mapped[bool] = mapped_column(Boolean, nullable=False)
    output: Mapped[str] = mapped_column(Text, default="", nullable=False)
    error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    truncated: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    execution_time_ms: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    # JSON-serialized complex fields
    files_created_json: Mapped[str] = mapped_column(Text, default="[]", nullable=False)
    variables_json: Mapped[str] = mapped_column(Text, default="{}", nullable=False)
    state_json: Mapped[str] = mapped_column(Text, default="{}", nullable=False)
    state_formatted: Mapped[str] = mapped_column(Text, default="(empty state)", nullable=False)

    # RL fields
    done: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    final_answer: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    iteration: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    reward: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    episode_state_json: Mapped[str] = mapped_column(Text, default="{}", nullable=False)

    @property
    def files_created(self) -> List[str]:
        """Deserialize files_created from JSON."""
        return json.loads(self.files_created_json) if self.files_created_json else []

    @files_created.setter
    def files_created(self, value: List[str]) -> None:
        """Serialize files_created to JSON."""
        self.files_created_json = json.dumps(value)

    @property
    def variables(self) -> Dict[str, str]:
        """Deserialize variables from JSON."""
        return json.loads(self.variables_json) if self.variables_json else {}

    @variables.setter
    def variables(self, value: Dict[str, str]) -> None:
        """Serialize variables to JSON."""
        self.variables_json = json.dumps(value)

    @property
    def state(self) -> Dict[str, Any]:
        """Deserialize state from JSON."""
        return json.loads(self.state_json) if self.state_json else {}

    @state.setter
    def state(self, value: Dict[str, Any]) -> None:
        """Serialize state to JSON."""
        self.state_json = json.dumps(value)

    @property
    def episode_state(self) -> Dict[str, Any]:
        """Deserialize episode_state from JSON."""
        return json.loads(self.episode_state_json) if self.episode_state_json else {}

    @episode_state.setter
    def episode_state(self, value: Dict[str, Any]) -> None:
        """Serialize episode_state to JSON."""
        self.episode_state_json = json.dumps(value)


__all__ = [
    "SubAgentCallDB",
    "ExecutionResultDB",
]
