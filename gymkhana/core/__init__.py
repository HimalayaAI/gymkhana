"""Core module for Gymkhana shared abstractions.

This module re-exports the core entities, services, and models for
convenient access throughout the codebase.

NOTE: DB models are NOT exported here to avoid triggering SQLAlchemy
at import time. Import them directly from their modules when needed.
"""

# Entity base classes
from gymkhana.core.models import Entity, Base, DBEntityMixin

# Services
from gymkhana.core.services import (
    REPLSandbox,
    REPLClient,
    SandboxService,
    ServiceContainer,
    StorageService,
    SQLStorageService,
    EnvStorageService,
    StorageSession,
    STORAGE_AVAILABLE,
)

# Rewards
from gymkhana.core.rewards import (
    RewardFunction,
    TrajectoryMetrics,
    get_reward_function,
)

# Models (Pydantic only - no DB models)
from gymkhana.core.models import (
    Turn,
    TrajectoryResult,
    PipelineStats,
    ExecutionResult,
    SubAgentCall,
    AnswerParser,
    BoxedAnswerParser,
    AnswerVerifier,
    SimpleEqualityVerifier,
    MathematicalVerifier,
)

__all__ = [
    # Entity base
    "Entity",
    "Base",
    "DBEntityMixin",
    # Services
    "REPLSandbox",
    "REPLClient",
    "SandboxService",
    "ServiceContainer",
    "StorageService",
    "SQLStorageService",
    "EnvStorageService",
    "StorageSession",
    "STORAGE_AVAILABLE",
    # Reward System
    "RewardFunction",
    "TrajectoryMetrics",
    "get_reward_function",
    # Core models (Pydantic only)
    "Turn",
    "TrajectoryResult",
    "PipelineStats",
    "ExecutionResult",
    "SubAgentCall",
    # Parsers
    "AnswerParser",
    "BoxedAnswerParser",
    "AnswerVerifier",
    "SimpleEqualityVerifier",
    "MathematicalVerifier",
]
