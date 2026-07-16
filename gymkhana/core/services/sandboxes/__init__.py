"""Sandbox service abstractions and implementations for Gymkhana.

Provides local and remote code execution environments for reasoning agents,
with support for resource management, session lifecycle, and RLM-compatible
state tracking.

Architecture:
- SandboxService: Abstract base class defining the sandbox interface
- SandboxConfig: Configuration models for resources, timeouts, rewards
- SessionState: State tracking models for session lifecycle
- REPLSandbox: HTTP-based REPL server implementation
- DockerSandboxService: Docker container-based implementation
- REPLClient: Low-level HTTP client for REPL server

Usage:
    # Simple REPL sandbox (math problems)
    from gymkhana.core.services.sandboxes import REPLSandbox, SandboxConfig

    config = SandboxConfig.for_math()
    sandbox = REPLSandbox(config=config)

    async with sandbox.session(context="data") as state:
        result = await sandbox.execute("print(1 + 1)")
        print(f"Output: {result.output}")
        print(f"Iteration: {state.episode.iteration}")

    # Docker sandbox (SWE tasks)
    from gymkhana.core.services.sandboxes import DockerSandboxService

    config = SandboxConfig.for_swe("swebench/swesmith.x86_64.repo")
    sandbox = DockerSandboxService(config=config, instance_id="task-001")

    async with sandbox.session() as state:
        result = await sandbox.execute_bash("ls -la /testbed")
        result = await sandbox.execute("import os; print(os.getcwd())")
"""

# Configuration and state models from core/models (canonical location)
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

# Base service and exceptions
from gymkhana.core.services.sandboxes.sandbox import (
    SandboxService,
    SandboxError,
    SandboxNotReadyError,
    SandboxTimeoutError,
    SandboxResourceError,
)

# Implementations
from gymkhana.core.services.sandboxes.client import REPLClient
from gymkhana.core.services.sandboxes.repl import REPLSandbox, REPLSandboxService
from gymkhana.core.services.sandboxes.docker_sandbox import (
    DockerContainerSession,
    DockerSandboxService,
    DOCKER_AVAILABLE,
)

__all__ = [
    # Configuration
    "SandboxBackend",
    "ResourceConfig",
    "TimeoutConfig",
    "RewardConfig",
    "SubAgentConfig",
    "SessionConfig",
    "SandboxConfig",
    # State
    "SessionStatus",
    "ExecutionMetrics",
    "SessionMetrics",
    "InterpreterState",
    "EpisodeState",
    "SessionState",
    # Base service
    "SandboxService",
    "SandboxError",
    "SandboxNotReadyError",
    "SandboxTimeoutError",
    "SandboxResourceError",
    # Implementations
    "REPLClient",
    "REPLSandbox",
    "REPLSandboxService",  # Legacy alias
    "DockerContainerSession",
    "DockerSandboxService",
    "DOCKER_AVAILABLE",
]
