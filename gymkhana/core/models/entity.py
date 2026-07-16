"""Base entity classes for gymkhana models.

This module provides the foundation for all gymkhana entities:
- Entity: Pydantic base class with UUID, timestamps, and serialization
- Base: SQLAlchemy declarative base for database models

The design separates business logic (Pydantic) from persistence (SQLAlchemy).
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, ConfigDict
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy import DateTime, String, event
from sqlalchemy.dialects.postgresql import UUID as PGUUID


class Entity(BaseModel):
    """Base class for all gymkhana entities.

    Provides:
    - Automatic UUID generation
    - Created timestamp
    - JSON serialization with proper type encoding
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
    )

    id: UUID = Field(
        default_factory=uuid4,
        description="Unique identifier for this entity instance"
    )

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp when this entity was created"
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert entity to dictionary with JSON-safe values."""
        return self.model_dump(mode="json")


class Base(DeclarativeBase):
    """SQLAlchemy declarative base for all database models."""
    pass


class DBEntityMixin:
    """Mixin for common database model fields.

    Provides standard fields that all DB models should have:
    - id: Primary key UUID
    - created_at: Creation timestamp
    - updated_at: Last update timestamp
    """

    id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
        nullable=False
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False
    )

    updated_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        default=None,
        nullable=True,
        onupdate=lambda: datetime.now(timezone.utc)
    )


@event.listens_for(Base, "init", propagate=True)
def _initialize_entity_defaults(target: Any, args: Any, kwargs: Any) -> None:
    """Apply identity/timestamp defaults before a model is flushed.

    SQLAlchemy column defaults normally run at INSERT time, while Gymkhana
    links rollout objects before persistence. Assigning them on construction
    keeps the domain and persistence identity contracts aligned.
    """
    state = target.__dict__
    state.setdefault("id", uuid4())
    state.setdefault("created_at", datetime.now(timezone.utc))


__all__ = ["Entity", "Base", "DBEntityMixin"]
