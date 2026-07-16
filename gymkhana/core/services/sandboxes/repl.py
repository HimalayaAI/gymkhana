"""REPL sandbox service implementation.

Provides SandboxService implementations that use HTTP-based REPL servers
for code execution. This implementation is environment-agnostic and can
be used across different task types (math, long-context, etc.).
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Dict, List, Optional

from pydantic import ConfigDict, Field, PrivateAttr

from gymkhana.core.models.execution import ExecutionResult
from .client import REPLClient
from .sandbox import SandboxService, SandboxNotReadyError
from gymkhana.core.models.sandbox import (
    SandboxConfig,
    SessionConfig,
    SessionState,
    SessionStatus,
    InterpreterState,
    ExecutionMetrics,
)


logger = logging.getLogger(__name__)


class REPLSandbox(SandboxService):
    """Sandbox implementation using REPLClient (HTTP REPL server).

    Wraps the existing REPLClient to conform to the SandboxService protocol.
    Provides both sync context manager and async session management.

    Example:
        ```python
        # Using new config-based initialization
        config = SandboxConfig.for_math()
        service = REPLSandbox(config=config)

        async with service.session(context="some data") as state:
            result = service.execute("print('hello')")
            print(f"Output: {result.output}")
            print(f"Iteration: {state.episode.iteration}")

        # Legacy usage still supported
        service = REPLSandbox(server_url="http://localhost:5003")
        service.create_session(context="some data")
        result = service.execute("print('hello')")
        service.delete_session()
        ```
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Allow direct server_url for backward compatibility
    server_url: str = Field(
        default="http://localhost:5003",
        description="URL of the REPL server"
    )

    timeout_seconds: int = Field(
        default=120,
        description="Timeout in seconds for code execution"
    )

    # Private state
    _client: Optional[REPLClient] = PrivateAttr(default=None)

    def __init__(self, **data: Any) -> None:
        """Initialize the sandbox service.

        Args:
            server_url: URL of the REPL server (for backward compatibility)
            timeout_seconds: Timeout in seconds for code execution
            config: SandboxConfig instance (preferred)
        """
        # Handle backward compatibility - server_url takes precedence
        if "server_url" in data and "config" not in data:
            data["config"] = SandboxConfig(server_url=data["server_url"])
        elif "config" in data and "server_url" not in data:
            data["server_url"] = data["config"].server_url

        super().__init__(**data)

        self._client = REPLClient(self.server_url, timeout_seconds=self.timeout_seconds)

    @property
    def client(self) -> REPLClient:
        """Get the underlying REPL client."""
        if self._client is None:
            self._client = REPLClient(self.server_url, timeout_seconds=self.timeout_seconds)
        return self._client

    # ========================================================================
    # SandboxService Implementation
    # ========================================================================

    async def create_session(
        self,
        *,
        context: Optional[str] = None,
        config_override: Optional[SessionConfig] = None,
        # Legacy parameters for backward compatibility
        packages: Optional[List[str]] = None,
        max_output_chars: Optional[int] = None,
        max_iterations: Optional[int] = None,
        enable_bash: Optional[bool] = None,
        sub_agent_config: Optional[Dict[str, Any]] = None,
        reward_config: Optional[Dict[str, float]] = None,
    ) -> str:
        """Create a new sandbox session.

        Args:
            context: Optional large context to make available
            config_override: Override session configuration
            packages: Python packages to pre-import (legacy)
            max_output_chars: Max output truncation limit (legacy)
            max_iterations: Max iterations (legacy)
            enable_bash: Enable bash (legacy)
            sub_agent_config: Sub-agent config (legacy)
            reward_config: Reward config (legacy)

        Returns:
            Session ID
        """
        # Build effective configuration
        cfg = self.get_effective_session_config(config_override)

        # Apply legacy overrides
        effective_packages = packages if packages is not None else cfg.packages
        effective_max_output = max_output_chars if max_output_chars is not None else cfg.max_output_chars
        effective_max_iters = max_iterations if max_iterations is not None else cfg.max_iterations
        effective_bash = enable_bash if enable_bash is not None else cfg.enable_bash

        # Build sub-agent config
        effective_sub_agent = sub_agent_config
        if effective_sub_agent is None and self.config.sub_agent.enabled:
            effective_sub_agent = self.config.sub_agent.to_dict()

        # Build reward config
        effective_rewards = reward_config
        if effective_rewards is None:
            effective_rewards = {
                "on_success": self.config.rewards.on_success,
                "on_iteration": self.config.rewards.on_iteration,
                "on_error": self.config.rewards.on_error,
                "on_failure": self.config.rewards.on_failure,
            }

        # Create session via client
        session_id = await self.client.create_session(
            context=context,
            packages=effective_packages,
            max_output_chars=effective_max_output,
            max_iterations=effective_max_iters,
            enable_bash=effective_bash,
            sub_agent_config=effective_sub_agent,
            reward_config=effective_rewards,
        )

        # Create session state
        effective_cfg = SessionConfig(
            packages=effective_packages or [],
            max_output_chars=effective_max_output,
            max_iterations=effective_max_iters,
            enable_bash=effective_bash,
        )
        self._create_session_state(
            session_id,
            effective_cfg,
            context_size=len(context) if context else 0,
        )

        logger.debug(f"Created REPL session: {session_id}")
        return session_id

    async def execute(
        self, code: str, session_id: Optional[str] = None
    ) -> ExecutionResult:
        """Execute Python code in the sandbox.

        Args:
            code: Python code to execute
            session_id: Session ID (uses current session if not provided)

        Returns:
            ExecutionResult with output, errors, state
        """
        sid = session_id or self._current_session_id
        if not sid:
            raise SandboxNotReadyError("No active session. Call create_session() first.")

        # Mark session as executing
        state = self.get_session_state(sid)
        if state:
            state.mark_executing()

        # Execute via client
        result = await self.client.execute(code, sid)

        # Update session state
        if state:
            metrics = ExecutionMetrics(
                execution_time_ms=result.execution_time_ms,
                output_chars=len(result.output),
                truncated=result.truncated,
            )

            interpreter_state = InterpreterState.from_dict(result.state)
            interpreter_state.files_created = result.files_created

            self.update_session_state(
                sid,
                interpreter=interpreter_state,
                execution_metrics=metrics,
                reward=result.reward,
                done=result.done,
                final_answer=result.final_answer,
            )

        return result

    async def execute_bash(
        self,
        code: str,
        timeout: Optional[int] = None,
        session_id: Optional[str] = None,
    ) -> ExecutionResult:
        """Execute bash commands in the sandbox.

        Args:
            code: Bash commands to execute
            timeout: Optional timeout in seconds
            session_id: Session ID (uses current session if not provided)

        Returns:
            ExecutionResult with output, errors
        """
        sid = session_id or self._current_session_id
        if not sid:
            raise SandboxNotReadyError("No active session. Call create_session() first.")

        effective_timeout = timeout or self.config.timeouts.bash_seconds
        return await self.client.execute_bash(code, effective_timeout, sid)

    async def get_state(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Get current session state.

        Args:
            session_id: Session ID (uses current session if not provided)

        Returns:
            Dictionary containing session state
        """
        sid = session_id or self._current_session_id
        if not sid:
            return {}
        return await self.client.get_state(sid)

    async def reset_session(self, session_id: Optional[str] = None) -> None:
        """Reset session to initial state.

        Args:
            session_id: Session ID (uses current session if not provided)
        """
        sid = session_id or self._current_session_id
        if not sid:
            return

        await self.client.reset_session(sid)

        # Reset local state tracking
        state = self.get_session_state(sid)
        if state:
            state.interpreter = InterpreterState()
            state.episode = type(state.episode)(max_iterations=state.episode.max_iterations)
            state.metrics = type(state.metrics)()

    async def delete_session(self, session_id: Optional[str] = None) -> None:
        """Delete session and cleanup resources.

        Args:
            session_id: Session ID (uses current session if not provided)
        """
        sid = session_id or self._current_session_id
        if not sid:
            return

        await self.client.delete_session(sid)
        self._cleanup_session_state(sid)
        logger.debug(f"Deleted REPL session: {sid}")

    async def health_check(self) -> bool:
        """Check if the REPL server is healthy.

        Returns:
            True if healthy, False otherwise
        """
        return await self.client.health_check()

    # ========================================================================
    # Additional Methods
    # ========================================================================

    async def execute_stateless(
        self, code: str, context: Optional[str] = None
    ) -> ExecutionResult:
        """Execute code without session management.

        Creates a temporary sandbox, executes, and cleans up.
        Use for one-off executions.

        Args:
            code: Python code to execute
            context: Optional context

        Returns:
            ExecutionResult
        """
        return await self.client.execute_stateless(code, context)

    # ========================================================================
    # Context Managers (override for sync support)
    # ========================================================================

    def __enter__(self) -> "REPLSandbox":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        """Context manager exit. Use async session() for session cleanup."""
        return False

    @asynccontextmanager
    async def session(
        self,
        *,
        context: Optional[str] = None,
        config_override: Optional[SessionConfig] = None,
        # Legacy parameters
        packages: Optional[List[str]] = None,
        max_output_chars: int = 8192,
        max_iterations: int = 30,
        enable_bash: bool = False,
        sub_agent_config: Optional[Dict[str, Any]] = None,
        reward_config: Optional[Dict[str, float]] = None,
    ) -> AsyncIterator[SessionState]:
        """Async context manager for session lifecycle.

        Creates a session, yields session state, then cleans up.

        Args:
            context: Optional context
            config_override: Override session configuration
            packages: Packages to import (legacy)
            max_output_chars: Output limit (legacy)
            max_iterations: Max iterations (legacy)
            enable_bash: Enable bash (legacy)
            sub_agent_config: Sub-agent config (legacy)
            reward_config: Reward config (legacy)

        Yields:
            SessionState for the active session
        """
        session_id = await self.create_session(
            context=context,
            config_override=config_override,
            packages=packages,
            max_output_chars=max_output_chars,
            max_iterations=max_iterations,
            enable_bash=enable_bash,
            sub_agent_config=sub_agent_config,
            reward_config=reward_config,
        )
        try:
            state = self.get_session_state(session_id)
            if state:
                state.mark_ready()
            yield state or SessionState()
        finally:
            await self.delete_session(session_id)


# Legacy alias for backward compatibility
REPLSandboxService = REPLSandbox

REPLSandbox.model_rebuild()

__all__ = ["REPLSandbox", "REPLSandboxService"]
