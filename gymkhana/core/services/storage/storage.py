"""Base storage service abstractions for Gymkhana.

Defines the abstract base class and specialized implementations for persistence.
Follows the same pattern as SandboxService for consistency.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict

from gymkhana.core.models.trajectory import Turn
from gymkhana.core.models.execution import ExecutionResult


class StorageService(BaseModel, ABC):
    """Abstract base class for storage services.

    Any class implementing these methods can serve as a storage backend.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the storage backend (connections, schema, etc.)."""
        ...

    @abstractmethod
    async def close(self) -> None:
        """Close the storage backend."""
        ...

    @abstractmethod
    def create_session(self, trajectory_id: Optional[UUID] = None) -> "StorageSession":
        """Create a scoped session for recording related entities."""
        ...


class StorageSession(BaseModel, ABC):
    """Abstract base class for storage sessions.

    Provides a scoped interface for recording turns and executions
    linked to a specific trajectory.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    trajectory_id: Optional[UUID] = None

    @abstractmethod
    async def store_turn(self, turn: Turn, trajectory_id: Optional[UUID] = None) -> UUID:
        """Store a conversation turn."""
        ...

    @abstractmethod
    async def store_execution(self, execution: ExecutionResult, turn_id: UUID) -> UUID:
        """Store an execution result."""
        ...
