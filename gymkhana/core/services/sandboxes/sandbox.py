"""Sandbox service abstraction.

Defines the abstract base class for code execution backends with
support for resource management, session lifecycle, and RLM-compatible
state tracking.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Callable,
    Dict,
    List,
    Optional,
    TypeVar,
)
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from gymkhana.core.models.sandbox import (
    SandboxConfig,
    SessionConfig,
    ResourceConfig,
    TimeoutConfig,
    RewardConfig,
    SubAgentConfig,
    SessionState,
    SessionStatus,
    InterpreterState,
    EpisodeState,
    ExecutionMetrics,
)

if TYPE_CHECKING:
    from gymkhana.core.models.execution import ExecutionResult


logger = logging.getLogger(__name__)

# Type variable for sandbox service subclasses
SandboxT = TypeVar("SandboxT", bound="SandboxService")


class SandboxError(Exception):
    """Base exception for sandbox errors."""
    pass


class SandboxNotReadyError(SandboxError):
    """Raised when sandbox is not ready for operations."""
    pass


class SandboxTimeoutError(SandboxError):
    """Raised when sandbox operation times out."""
    pass


class SandboxResourceError(SandboxError):
    """Raised when sandbox resource allocation fails."""
    pass


class SandboxService(BaseModel, ABC):
    """Abstract base class for code execution sandboxes.

    Provides a unified interface for various sandbox backends (HTTP REPL,
    Docker containers, local execution, remote cloud sandboxes).

    Features:
    - Resource management (CPU, memory, GPU)
    - Session lifecycle with state tracking
    - RLM-compatible episode management
    - Async context manager support
    - Cleanup hooks for graceful shutdown

    Example:
        ```python
        service = REPLSandbox(config=SandboxConfig.for_math())

        async with service.session(context="some data") as session:
            result = await service.execute("print('hello')")
            print(f"Output: {result.output}")
            print(f"State: {session.interpreter.format_compact()}")
        ```
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Configuration
    config: SandboxConfig = Field(
        default_factory=SandboxConfig,
        description="Complete sandbox configuration"
    )

    # Private state
    _active_sessions: Dict[str, SessionState] = PrivateAttr(default_factory=dict)
    _current_session_id: Optional[str] = PrivateAttr(default=None)
    _cleanup_hooks: List[Callable[[], None]] = PrivateAttr(default_factory=list)

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        self._active_sessions = {}
        self._current_session_id = None
        self._cleanup_hooks = []

    # ========================================================================
    # Properties
    # ========================================================================

    @property
    def session_id(self) -> Optional[str]:
        """Get current session ID."""
        return self._current_session_id

    @property
    def current_session(self) -> Optional[SessionState]:
        """Get current session state."""
        if self._current_session_id:
            return self._active_sessions.get(self._current_session_id)
        return None

    @property
    def active_session_count(self) -> int:
        """Number of active sessions."""
        return len(self._active_sessions)

    @property
    def resources(self) -> ResourceConfig:
        """Resource configuration."""
        return self.config.resources

    @property
    def timeouts(self) -> TimeoutConfig:
        """Timeout configuration."""
        return self.config.timeouts

    # ========================================================================
    # Abstract Methods - Must be implemented by subclasses
    # ========================================================================

    @abstractmethod
    async def create_session(
        self,
        *,
        context: Optional[str] = None,
        config_override: Optional[SessionConfig] = None,
    ) -> str:
        """Create a new sandbox session.

        Args:
            context: Optional large context to make available in session
            config_override: Override session configuration

        Returns:
            Session ID string
        """
        ...

    @abstractmethod
    async def execute(
        self, code: str, session_id: Optional[str] = None
    ) -> "ExecutionResult":
        """Execute Python code in the sandbox.

        Args:
            code: Python code to execute
            session_id: Session ID (uses current session if not provided)

        Returns:
            ExecutionResult with output, errors, state
        """
        ...

    @abstractmethod
    async def execute_bash(
        self,
        code: str,
        timeout: Optional[int] = None,
        session_id: Optional[str] = None,
    ) -> "ExecutionResult":
        """Execute bash commands in the sandbox.

        Args:
            code: Bash commands to execute
            timeout: Optional timeout in seconds
            session_id: Session ID (uses current session if not provided)

        Returns:
            ExecutionResult with output, errors
        """
        ...

    @abstractmethod
    async def get_state(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Get current session state.

        Args:
            session_id: Session ID (uses current session if not provided)

        Returns:
            Dictionary containing session state
        """
        ...

    @abstractmethod
    async def reset_session(self, session_id: Optional[str] = None) -> None:
        """Reset session to initial state.

        Args:
            session_id: Session ID (uses current session if not provided)
        """
        ...

    @abstractmethod
    async def delete_session(self, session_id: Optional[str] = None) -> None:
        """Delete session and cleanup resources.

        Args:
            session_id: Session ID (uses current session if not provided)
        """
        ...

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the sandbox backend is healthy.

        Returns:
            True if healthy, False otherwise
        """
        ...

    # ========================================================================
    # Session State Management
    # ========================================================================

    def _create_session_state(
        self,
        session_id: str,
        config: SessionConfig,
        context_size: int = 0,
    ) -> SessionState:
        """Create and register a new session state."""
        state = SessionState(
            session_id=session_id,  # Use the session_id directly
            episode=EpisodeState(max_iterations=config.max_iterations),
            context_loaded=context_size > 0,
            context_size_chars=context_size,
        )

        self._active_sessions[session_id] = state
        self._current_session_id = session_id
        return state

    def get_session_state(self, session_id: Optional[str] = None) -> Optional[SessionState]:
        """Get session state by ID."""
        sid = session_id or self._current_session_id
        if sid:
            return self._active_sessions.get(sid)
        return None

    def update_session_state(
        self,
        session_id: str,
        *,
        interpreter: Optional[InterpreterState] = None,
        execution_metrics: Optional[ExecutionMetrics] = None,
        reward: float = 0.0,
        done: bool = False,
        final_answer: Optional[str] = None,
    ) -> None:
        """Update session state after execution."""
        state = self._active_sessions.get(session_id)
        if not state:
            return

        if execution_metrics:
            state.record_execution(
                success=True,  # TODO: pass actual success
                metrics=execution_metrics,
                interpreter_state=interpreter,
                reward=reward,
            )

        if done:
            if final_answer:
                state.mark_completed(final_answer)
            else:
                state.episode.done = True

    def _cleanup_session_state(self, session_id: str) -> None:
        """Remove session state from tracking."""
        if session_id in self._active_sessions:
            del self._active_sessions[session_id]
        if self._current_session_id == session_id:
            self._current_session_id = None

    # ========================================================================
    # Lifecycle Hooks
    # ========================================================================

    def register_cleanup_hook(self, hook: Callable[[], None]) -> None:
        """Register a cleanup hook to run on shutdown."""
        self._cleanup_hooks.append(hook)

    def run_cleanup_hooks(self) -> None:
        """Run all registered cleanup hooks."""
        for hook in self._cleanup_hooks:
            try:
                hook()
            except Exception as e:
                logger.warning(f"Cleanup hook failed: {e}")

    async def cleanup_all_sessions(self) -> None:
        """Delete all active sessions."""
        session_ids = list(self._active_sessions.keys())
        for sid in session_ids:
            try:
                await self.delete_session(sid)
            except Exception as e:
                logger.warning(f"Failed to delete session {sid}: {e}")

    # ========================================================================
    # Context Managers
    # ========================================================================

    def __enter__(self) -> "SandboxService":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        """Context manager exit - runs cleanup hooks only. Use async session() for session cleanup."""
        self.run_cleanup_hooks()
        return False

    @asynccontextmanager
    async def session(
        self,
        *,
        context: Optional[str] = None,
        config_override: Optional[SessionConfig] = None,
    ) -> AsyncIterator[SessionState]:
        """Async context manager for session lifecycle.

        Creates a session, yields the session state, then cleans up.

        Example:
            ```python
            service = REPLSandbox()
            async with service.session(context="data") as state:
                result = await service.execute("print('hello')")
                print(f"Iteration: {state.episode.iteration}")
            ```

        Args:
            context: Optional context
            config_override: Override session configuration

        Yields:
            SessionState for the active session
        """
        session_id = await self.create_session(
            context=context,
            config_override=config_override,
        )
        try:
            state = self._active_sessions.get(session_id)
            if state:
                state.mark_ready()
            yield state or SessionState()
        finally:
            await self.delete_session(session_id)

    # ========================================================================
    # Utility Methods
    # ========================================================================

    def get_effective_session_config(
        self,
        override: Optional[SessionConfig] = None
    ) -> SessionConfig:
        """Get effective session config with optional override."""
        if override:
            return override
        return self.config.session

    def build_create_session_payload(
        self,
        context: Optional[str] = None,
        config: Optional[SessionConfig] = None,
    ) -> Dict[str, Any]:
        """Build payload for session creation (useful for HTTP APIs)."""
        cfg = config or self.config.session

        payload: Dict[str, Any] = {
            "max_output_chars": cfg.max_output_chars,
            "max_iterations": cfg.max_iterations,
            "enable_bash": cfg.enable_bash,
        }

        if context:
            payload["context"] = context

        if cfg.packages:
            payload["packages"] = cfg.packages

        if self.config.sub_agent.enabled:
            payload["sub_agent_config"] = self.config.sub_agent.to_dict()

        # Add reward config
        payload["reward_config"] = {
            "on_success": self.config.rewards.on_success,
            "on_iteration": self.config.rewards.on_iteration,
            "on_error": self.config.rewards.on_error,
            "on_failure": self.config.rewards.on_failure,
        }

        return payload


SandboxService.model_rebuild()

__all__ = [
    "SandboxService",
    "SandboxError",
    "SandboxNotReadyError",
    "SandboxTimeoutError",
    "SandboxResourceError",
]
