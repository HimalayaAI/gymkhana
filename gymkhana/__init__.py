"""
Gymkhana: Interleaved Reasoning + Code Training Data Generation.

Generates training data for LLMs to learn:
- Brief natural language planning
- Executable Python actions
- Iterative refinement based on observations
- Filesystem-based dynamic context management
"""

# Version
__version__ = "0.1.0"

# Main components
from gymkhana.envs.config import (
    EnvConfig,
    REPLSettings,
    SubLLMSettings,
    DatasetSettings,
    LLMClientType,
    EnvironmentType,
    InferenceConfig,
)
from gymkhana.envs import get_environment, ENVIRONMENTS

# Core models and services
from gymkhana.core import (
    Entity,
    Base,
    DBEntityMixin,
    TrajectoryResult,
    Turn,
    ExecutionResult,
    SandboxService,
    REPLSandbox,
    REPLClient,
    StorageService,
    EnvStorageService,
    StorageSession,
    ServiceContainer,
    RewardFunction,
    get_reward_function,
)

__all__ = [
    # Config
    "EnvConfig",
    "REPLSettings",
    "SubLLMSettings",
    "DatasetSettings",
    "EnvironmentType",
    "LLMClientType",
    "InferenceConfig",
    # Abstractions
    "get_environment",
    "ENVIRONMENTS",
    "ServiceContainer",
    # Core Entities
    "Entity",
    "Base",
    "DBEntityMixin",
    "TrajectoryResult",
    "Turn",
    "ExecutionResult",
    # Services
    "SandboxService",
    "REPLSandbox",
    "REPLClient",
    "StorageService",
    "EnvStorageService",
    "StorageSession",
    # Reward System
    "RewardFunction",
    "get_reward_function",
]
