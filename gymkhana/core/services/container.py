"""Base service container for dependency injection."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field

from gymkhana.core.services.sandboxes.sandbox import SandboxService
from gymkhana.core.services.storage.storage import StorageService
from gymkhana.core.services.inference.base import InferenceService


class ServiceContainer(BaseModel):
    """Dependency container for environment services.

    Provides a typed container for injecting services into environments.
    Services are optional - if not provided, environments fall back to
    their default behavior.

    Attributes:
        sandbox: SandboxService implementation for code execution
        inference: InferenceService implementation for LLM calls
        storage: StorageService implementation for persistence
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    sandbox: Optional[SandboxService] = Field(
        default=None,
        description="SandboxService implementation for code execution",
    )
    inference: Optional[InferenceService] = Field(
        default=None,
        description="InferenceService implementation for LLM calls",
    )
    storage: Optional[StorageService] = Field(
        default=None,
        description="StorageService implementation for persistence",
    )


ServiceContainer.model_rebuild()

__all__ = ["ServiceContainer"]
