"""Sandbox configuration and state models.

Provides Pydantic models for sandbox resource allocation, session configuration,
RLM-compatible reward settings, and session state tracking. These models are
environment-agnostic and can be used across different sandbox implementations.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from json import dumps, loads
from typing import Any, Dict, List, Optional
from uuid import uuid4, UUID

from pydantic import BaseModel, Field, ConfigDict, computed_field


# =============================================================================
# Enums
# =============================================================================

class SandboxBackend(str, Enum):
    """Available sandbox backend types."""
    REPL_HTTP = "repl_http"       # Remote REPL server over HTTP
    DOCKER = "docker"             # Docker container with REPL
    LOCAL = "local"               # Local in-process execution
    REMOTE = "remote"             # Remote cloud sandbox


class SessionStatus(str, Enum):
    """Session lifecycle status."""
    PENDING = "pending"       # Session created but not ready
    READY = "ready"           # Session active and available
    EXECUTING = "executing"   # Currently executing code
    PAUSED = "paused"         # Temporarily paused
    COMPLETED = "completed"   # Episode completed (FINAL() called)
    FAILED = "failed"         # Session failed due to error
    EXPIRED = "expired"       # Session timed out
    DELETED = "deleted"       # Session cleaned up


# =============================================================================
# Configuration Models
# =============================================================================

class ResourceConfig(BaseModel):
    """Resource allocation configuration for sandboxes."""
    model_config = ConfigDict(frozen=True)

    cpu_cores: int = Field(
        default=2,
        ge=1,
        le=32,
        description="Number of CPU cores to allocate"
    )

    memory_gb: float = Field(
        default=4.0,
        ge=0.5,
        le=128.0,
        description="Memory allocation in GB"
    )

    disk_size_gb: float = Field(
        default=10.0,
        ge=1.0,
        le=500.0,
        description="Disk space allocation in GB"
    )

    @property
    def memory_bytes(self) -> int:
        """Memory allocation in bytes for Docker."""
        return int(self.memory_gb * 1024 * 1024 * 1024)

    @property
    def memory_limit_docker(self) -> str:
        """Memory limit string for Docker (e.g., '4g')."""
        return f"{self.memory_gb}g"


class TimeoutConfig(BaseModel):
    """Timeout configuration for sandbox operations."""
    model_config = ConfigDict(frozen=True)

    execution_seconds: int = Field(
        default=120,
        ge=1,
        description="Timeout for single code execution"
    )

    bash_seconds: int = Field(
        default=60,
        ge=1,
        description="Timeout for bash commands"
    )

    startup_seconds: int = Field(
        default=120,
        ge=10,
        description="Timeout for sandbox startup"
    )

    session_minutes: int = Field(
        default=60,
        ge=1,
        description="Total session lifetime in minutes"
    )


class RewardConfig(BaseModel):
    """RL reward configuration for RLM-style training."""
    model_config = ConfigDict(frozen=True)

    on_success: float = Field(
        default=1.0,
        description="Reward when FINAL() is called successfully"
    )

    on_iteration: float = Field(
        default=0.0,
        description="Reward per step"
    )

    on_error: float = Field(
        default=-0.05,
        description="Penalty for execution errors"
    )

    on_failure: float = Field(
        default=-0.1,
        description="Penalty when max iterations reached without FINAL()"
    )

    on_timeout: float = Field(
        default=-0.1,
        description="Penalty for execution timeout"
    )


class SubAgentConfig(BaseModel):
    """Configuration for sub-agent LLM calls from within sandbox."""
    model_config = ConfigDict(frozen=True)

    enabled: bool = Field(
        default=False,
        description="Whether sub-agent calls are enabled"
    )

    model: str = Field(
        default="",
        description="Model to use for sub-agent calls"
    )

    client: str = Field(
        default="litellm",
        description="Client for sub-agent calls"
    )

    temperature: float = Field(
        default=0.3,
        ge=0.0,
        le=2.0,
        description="Temperature for sub-agent responses"
    )

    max_tokens: int = Field(
        default=2048,
        ge=1,
        description="Max tokens for sub-agent responses"
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for server payload."""
        if not self.enabled:
            return {}
        return {
            "model": self.model,
            "client": self.client,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }


class SessionConfig(BaseModel):
    """Session-level configuration for sandbox instances."""
    model_config = ConfigDict(validate_assignment=True)

    max_output_chars: int = Field(
        default=8192,
        ge=256,
        description="Maximum characters in output before truncation"
    )

    max_iterations: int = Field(
        default=30,
        ge=1,
        description="Maximum iterations before episode ends"
    )

    enable_bash: bool = Field(
        default=False,
        description="Whether to enable bash command execution"
    )

    enable_file_io: bool = Field(
        default=True,
        description="Whether to enable file read/write operations"
    )

    enable_network: bool = Field(
        default=False,
        description="Whether to enable network operations in code"
    )

    packages: List[str] = Field(
        default_factory=lambda: ["numpy", "pandas", "sympy", "scipy"],
        description="Python packages to pre-import"
    )

    startup_commands: List[str] = Field(
        default_factory=list,
        description="Shell commands to run at container startup (e.g., ['pip install -e /testbed'])"
    )

    setup_code: Optional[str] = Field(
        default=None,
        description="Optional setup code to run at session start"
    )


class SandboxConfig(BaseModel):
    """Complete sandbox configuration."""
    model_config = ConfigDict(validate_assignment=True)

    id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique identifier for this configuration"
    )

    backend: SandboxBackend = Field(
        default=SandboxBackend.REPL_HTTP,
        description="Sandbox backend to use"
    )

    server_url: str = Field(
        default="http://localhost:5003",
        description="URL for remote REPL server"
    )

    docker_image: str = Field(
        default="python:3.11-slim",
        description="Docker image for container-based sandboxes"
    )

    resources: ResourceConfig = Field(
        default_factory=ResourceConfig,
        description="Resource allocation limits"
    )

    timeouts: TimeoutConfig = Field(
        default_factory=TimeoutConfig,
        description="Timeout configuration"
    )

    session: SessionConfig = Field(
        default_factory=SessionConfig,
        description="Session-level configuration"
    )

    rewards: RewardConfig = Field(
        default_factory=RewardConfig,
        description="RL reward configuration"
    )

    sub_agent: SubAgentConfig = Field(
        default_factory=SubAgentConfig,
        description="Sub-agent LLM configuration"
    )

    @classmethod
    def for_math(cls) -> "SandboxConfig":
        """Factory for math problem environments."""
        return cls(
            backend=SandboxBackend.REPL_HTTP,
            session=SessionConfig(
                max_iterations=10,
                enable_bash=False,
                packages=["numpy", "sympy", "scipy"],
            ),
        )

    @classmethod
    def for_swe(cls, docker_image: str) -> "SandboxConfig":
        """Factory for software engineering environments."""
        return cls(
            backend=SandboxBackend.DOCKER,
            docker_image=docker_image,
            resources=ResourceConfig(
                cpu_cores=4,
                memory_gb=8.0,
                disk_size_gb=20.0,
            ),
            session=SessionConfig(
                max_iterations=50,
                max_output_chars=16384,
                enable_bash=True,
                packages=["pytest", "git"],
                startup_commands=[
                    "pip install -q -e /testbed 2>&1 || echo 'Note: /testbed installation failed or not applicable'"
                ],
            ),
        )

    @classmethod
    def for_long_context(cls) -> "SandboxConfig":
        """Factory for long-context environments with sub-agent support."""
        return cls(
            backend=SandboxBackend.REPL_HTTP,
            session=SessionConfig(
                max_iterations=20,
                max_output_chars=16384,
                enable_bash=False,
            ),
            sub_agent=SubAgentConfig(
                enabled=True,
                model="Hermes-4-70B",
                temperature=0.3,
            ),
        )


# =============================================================================
# State Tracking Models
# =============================================================================

class ExecutionMetrics(BaseModel):
    """Metrics for a single execution."""
    model_config = ConfigDict(frozen=True)

    execution_time_ms: int = Field(
        default=0,
        description="Wall-clock execution time in milliseconds"
    )

    output_chars: int = Field(
        default=0,
        description="Number of characters in output"
    )

    truncated: bool = Field(
        default=False,
        description="Whether output was truncated"
    )


class SessionMetrics(BaseModel):
    """Aggregated metrics for an entire session."""
    model_config = ConfigDict(validate_assignment=True)

    total_executions: int = Field(default=0)
    successful_executions: int = Field(default=0)
    failed_executions: int = Field(default=0)
    total_execution_time_ms: int = Field(default=0)
    total_output_chars: int = Field(default=0)
    truncation_count: int = Field(default=0)

    # RLM metrics
    total_reward: float = Field(default=0.0)
    step_rewards: List[float] = Field(default_factory=list)

    @computed_field
    @property
    def avg_execution_time_ms(self) -> float:
        """Average execution time per step."""
        if self.total_executions == 0:
            return 0.0
        return self.total_execution_time_ms / self.total_executions

    @computed_field
    @property
    def success_rate(self) -> float:
        """Fraction of successful executions."""
        if self.total_executions == 0:
            return 0.0
        return self.successful_executions / self.total_executions

    def record_execution(
        self,
        success: bool,
        metrics: ExecutionMetrics,
        reward: float = 0.0
    ) -> None:
        """Record metrics from an execution."""
        self.total_executions += 1
        if success:
            self.successful_executions += 1
        else:
            self.failed_executions += 1

        self.total_execution_time_ms += metrics.execution_time_ms
        self.total_output_chars += metrics.output_chars
        if metrics.truncated:
            self.truncation_count += 1

        self.total_reward += reward
        self.step_rewards.append(reward)


class InterpreterState(BaseModel):
    """Snapshot of Python interpreter state."""
    model_config = ConfigDict(validate_assignment=True)

    variables: Dict[str, Any] = Field(
        default_factory=dict,
        description="Variable name -> representation"
    )

    functions: Dict[str, Any] = Field(
        default_factory=dict,
        description="Function name -> signature/info"
    )

    classes: Dict[str, Any] = Field(
        default_factory=dict,
        description="Class name -> metadata (methods, bases, etc.)"
    )

    modules: List[str] = Field(
        default_factory=list,
        description="Imported module names"
    )

    files_created: List[str] = Field(
        default_factory=list,
        description="Files created in the workspace"
    )

    def format_compact(self) -> str:
        """Format state as compact string for display."""
        parts = []

        if self.modules:
            parts.append(f"imports: {', '.join(self.modules)}")

        if self.functions:
            parts.append(f"functions: {', '.join(self.functions.keys())}")

        if self.classes:
            parts.append(f"classes: {', '.join(self.classes.keys())}")

        if self.variables:
            var_items = []
            for k, v in list(self.variables.items())[:5]:
                if isinstance(v, dict) and 'type' in v:
                    val_repr = v['type']
                else:
                    val_repr = str(v)
                var_items.append(f"{k}={val_repr}")

            if len(self.variables) > 5:
                var_items.append(f"... (+{len(self.variables) - 5} more)")
            parts.append(f"vars: {', '.join(var_items)}")

        if self.files_created:
            parts.append(f"files: {', '.join(self.files_created)}")

        return " | ".join(parts) if parts else "(empty state)"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "variables": self.variables,
            "functions": self.functions,
            "classes": self.classes,
            "modules": self.modules,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "InterpreterState":
        """Create from dictionary."""
        return cls(
            variables=data.get("variables", {}),
            functions=data.get("functions", {}),
            classes=data.get("classes", {}),
            modules=data.get("modules", []),
            files_created=data.get("files_created", []),
        )


class EpisodeState(BaseModel):
    """RLM-style episode state for reinforcement learning."""
    model_config = ConfigDict(validate_assignment=True)

    iteration: int = Field(
        default=0,
        ge=0,
        description="Current iteration count"
    )

    max_iterations: int = Field(
        default=30,
        description="Maximum allowed iterations"
    )

    done: bool = Field(
        default=False,
        description="Whether episode is complete"
    )

    final_answer: Optional[str] = Field(
        default=None,
        description="Final answer if done"
    )

    termination_reason: Optional[str] = Field(
        default=None,
        description="Reason for episode termination"
    )

    @computed_field
    @property
    def iterations_remaining(self) -> int:
        """Number of iterations remaining."""
        return max(0, self.max_iterations - self.iteration)

    @computed_field
    @property
    def progress_ratio(self) -> float:
        """Progress through episode (0.0 to 1.0)."""
        if self.max_iterations == 0:
            return 1.0
        return min(1.0, self.iteration / self.max_iterations)

    def step(self) -> None:
        """Increment iteration counter."""
        self.iteration += 1
        if self.iteration >= self.max_iterations and not self.done:
            self.done = True
            self.termination_reason = "max_iterations_reached"

    def complete(self, answer: str) -> None:
        """Mark episode as complete with final answer."""
        self.done = True
        self.final_answer = answer
        self.termination_reason = "final_answer"

    def fail(self, reason: str) -> None:
        """Mark episode as failed."""
        self.done = True
        self.termination_reason = reason


class SessionState(BaseModel):
    """Complete state for a sandbox session."""
    model_config = ConfigDict(validate_assignment=True)

    session_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Session identifier"
    )

    status: SessionStatus = Field(
        default=SessionStatus.PENDING,
        description="Current session status"
    )

    # Timestamps
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Session creation timestamp (UTC)"
    )

    ready_at: Optional[datetime] = Field(
        default=None,
        description="When session became ready"
    )

    last_execution_at: Optional[datetime] = Field(
        default=None,
        description="Last execution timestamp"
    )

    completed_at: Optional[datetime] = Field(
        default=None,
        description="Session completion timestamp"
    )

    # State components
    interpreter: InterpreterState = Field(
        default_factory=InterpreterState,
        description="Python interpreter state"
    )

    episode: EpisodeState = Field(
        default_factory=EpisodeState,
        description="RLM episode state"
    )

    metrics: SessionMetrics = Field(
        default_factory=SessionMetrics,
        description="Aggregated session metrics"
    )

    # Context and configuration
    context_loaded: bool = Field(
        default=False,
        description="Whether context data has been loaded"
    )

    context_size_chars: int = Field(
        default=0,
        description="Size of loaded context in characters"
    )

    # Answer tracking
    answer: Dict[str, Any] = Field(
        default_factory=lambda: {"content": "", "ready": False},
        description="Answer variable state"
    )

    # Extra metadata
    extra: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional state data"
    )

    def mark_ready(self) -> None:
        """Mark session as ready."""
        self.status = SessionStatus.READY
        self.ready_at = datetime.now(timezone.utc)

    def mark_executing(self) -> None:
        """Mark session as currently executing."""
        self.status = SessionStatus.EXECUTING

    def mark_completed(self, answer: Optional[str] = None) -> None:
        """Mark session as completed."""
        self.status = SessionStatus.COMPLETED
        self.completed_at = datetime.now(timezone.utc)
        if answer:
            self.episode.complete(answer)

    def mark_failed(self, reason: str) -> None:
        """Mark session as failed."""
        self.status = SessionStatus.FAILED
        self.completed_at = datetime.now(timezone.utc)
        self.episode.fail(reason)

    def record_execution(
        self,
        success: bool,
        metrics: ExecutionMetrics,
        interpreter_state: Optional[InterpreterState] = None,
        reward: float = 0.0,
    ) -> None:
        """Record an execution result."""
        self.last_execution_at = datetime.now(timezone.utc)
        self.metrics.record_execution(success, metrics, reward)
        self.episode.step()

        if interpreter_state:
            self.interpreter = interpreter_state

        if self.status == SessionStatus.EXECUTING:
            self.status = SessionStatus.READY

    def to_summary(self) -> Dict[str, Any]:
        """Generate a summary for logging/display."""
        return {
            "session_id": str(self.session_id),
            "status": self.status.value,
            "iteration": self.episode.iteration,
            "max_iterations": self.episode.max_iterations,
            "done": self.episode.done,
            "total_reward": self.metrics.total_reward,
            "executions": self.metrics.total_executions,
            "success_rate": self.metrics.success_rate,
            "avg_time_ms": self.metrics.avg_execution_time_ms,
        }


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Enums
    "SandboxBackend",
    "SessionStatus",
    # Configuration
    "ResourceConfig",
    "TimeoutConfig",
    "RewardConfig",
    "SubAgentConfig",
    "SessionConfig",
    "SandboxConfig",
    # State
    "ExecutionMetrics",
    "SessionMetrics",
    "InterpreterState",
    "EpisodeState",
    "SessionState",
]
