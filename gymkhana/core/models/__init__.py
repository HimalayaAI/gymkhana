"""Core models re-exported for convenience.

This module exposes all core models from their respective submodules.
DB models are NOT imported at module level to avoid triggering SQLAlchemy
at import time. Import them directly from their modules when needed.
"""

# Base entity classes
from gymkhana.core.models.entity import Entity, Base, DBEntityMixin

# Execution models (Pydantic only)
from gymkhana.core.models.execution import (
    ExecutionResult,
    SubAgentCall,
)

# Parser protocols
from gymkhana.core.models.parsers import (
    AnswerParser,
    AnswerVerifier,
    BoxedAnswerParser,
    SimpleEqualityVerifier,
    MathematicalVerifier,
)

# Trajectory models (Pydantic only)
from gymkhana.core.models.trajectory import (
    PipelineStats,
    TrajectoryMetrics,
    TrajectoryResult,
    Turn,
    RolloutStatus,
    RolloutState,
    RolloutGroup,
)

# Environment models
from gymkhana.core.models.tasks import (
    TaskMetadata,
    TestResult,
    CodePatch,
)

# Sandbox models
from gymkhana.core.models.sandbox import (
    # Enums
    SandboxBackend,
    SessionStatus,
    # Configuration
    ResourceConfig,
    TimeoutConfig,
    RewardConfig,
    SubAgentConfig,
    SessionConfig,
    SandboxConfig,
    # State
    ExecutionMetrics,
    SessionMetrics,
    InterpreterState,
    EpisodeState,
    SessionState,
)


__all__ = [
    # Entity base classes
    "Entity",
    "Base",
    "DBEntityMixin",
    # Execution
    "ExecutionResult",
    "SubAgentCall",
    # Parsers
    "AnswerParser",
    "AnswerVerifier",
    "BoxedAnswerParser",
    "SimpleEqualityVerifier",
    "MathematicalVerifier",
    # Trajectory
    "Turn",
    "TrajectoryResult",
    "PipelineStats",
    "TrajectoryMetrics",
    "RolloutStatus",
    "RolloutState",
    "RolloutGroup",
    # Environment
    "TaskMetadata",
    "TestResult",
    "CodePatch",
    # Sandbox - Enums
    "SandboxBackend",
    "SessionStatus",
    # Sandbox - Configuration
    "ResourceConfig",
    "TimeoutConfig",
    "RewardConfig",
    "SubAgentConfig",
    "SessionConfig",
    "SandboxConfig",
    # Sandbox - State
    "ExecutionMetrics",
    "SessionMetrics",
    "InterpreterState",
    "EpisodeState",
    "SessionState",
]
