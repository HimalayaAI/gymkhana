"""
REPL Client for interacting with the Gymkhana server.

Used by both:
- Pipeline (dataset generation)
- Inference (running trained models)

All I/O is async (aiohttp). Use await when calling methods.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import aiohttp

from gymkhana.core.models.execution import ExecutionResult, SubAgentCall


class REPLClient:
    """
    Async client for the Gymkhana REPL server.

    Manages sessions and executes code via HTTP API. All methods are async.
    """

    def __init__(self, server_url: str = "http://localhost:5003", timeout_seconds: int = 120):
        self.server_url = server_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.session_id: Optional[str] = None
        self._http_session: Optional[aiohttp.ClientSession] = None

    async def _session(self) -> aiohttp.ClientSession:
        """Get or create the aiohttp client session."""
        if self._http_session is None or self._http_session.closed:
            self._http_session = aiohttp.ClientSession()
        return self._http_session

    async def close(self) -> None:
        """Close the HTTP session. Call when done with the client."""
        if self._http_session and not self._http_session.closed:
            await self._http_session.close()
            self._http_session = None

    async def __aenter__(self) -> "REPLClient":
        await self._session()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.close()

    async def health_check(self) -> bool:
        """Check if server is healthy."""
        try:
            session = await self._session()
            async with session.get(f"{self.server_url}/health", timeout=aiohttp.ClientTimeout(total=5)) as resp:
                return resp.status == 200
        except Exception:
            return False

    async def create_session(
        self,
        context: Optional[str] = None,
        packages: Optional[List[str]] = None,
        max_output_chars: int = 8192,
        max_iterations: int = 30,
        enable_bash: bool = False,
        sub_agent_config: Optional[Dict[str, Any]] = None,
        reward_config: Optional[Dict[str, float]] = None,
    ) -> str:
        """
        Create a new REPL session.

        Args:
            context: Optional large context to make available
            packages: Python packages to pre-import
            max_output_chars: Max output truncation limit
            max_iterations: Max iterations before episode ends
            enable_bash: Enable bash command execution (default: False)
            sub_agent_config: Config for sub-agent calls (model, client, temperature, max_tokens)
            reward_config: RL reward configuration (on_success, on_iteration, on_error, on_failure)

        Returns:
            Session ID
        """
        payload: Dict[str, Any] = {
            "max_output_chars": max_output_chars,
            "max_iterations": max_iterations,
            "enable_bash": enable_bash,
        }
        if context:
            payload["context"] = context
        if packages:
            payload["packages"] = packages
        if sub_agent_config:
            payload["sub_agent_config"] = sub_agent_config
        if reward_config:
            payload["reward_config"] = reward_config

        session = await self._session()
        async with session.post(
            f"{self.server_url}/session/create",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=60),  # Increased from 30s to 60s for startup commands
        ) as resp:
            resp.raise_for_status()
            data = await resp.json()
        self.session_id = data["session_id"]
        return self.session_id

    async def execute(self, code: str, session_id: Optional[str] = None) -> ExecutionResult:
        """
        Execute code in a session.

        Args:
            code: Python code to execute
            session_id: Session ID (uses current session if not provided)

        Returns:
            ExecutionResult with output, errors, state
        """
        sid = session_id or self.session_id
        if not sid:
            raise ValueError("No session ID. Call create_session() first.")

        session = await self._session()
        async with session.post(
            f"{self.server_url}/session/{sid}/execute",
            json={"code": code},
            timeout=aiohttp.ClientTimeout(total=self.timeout_seconds),
        ) as resp:
            body = await resp.text()
            if resp.status >= 400:
                try:
                    err_data = json.loads(body)
                    msg = err_data.get("error") or body[:500]
                    if err_data.get("traceback"):
                        msg += "\nServer traceback:\n" + err_data.get("traceback", "")[-2000:]
                except Exception:
                    if resp.status == 500 and body.strip().lower().startswith("<!") and "500" in body:
                        msg = (
                            "REPL server returned HTML 500. Restart the REPL server with the latest code "
                            "(e.g. python -m gymkhana.core.services.sandboxes.server.app) so it returns JSON "
                            "with the real error and traceback."
                        )
                    else:
                        msg = body[:1000] if body else f"HTTP {resp.status}"
                raise aiohttp.ClientResponseError(
                    resp.request_info,
                    resp.history,
                    status=resp.status,
                    message=msg,
                    headers=resp.headers,
                )
            data = json.loads(body)

        sub_agent_calls = [
            SubAgentCall(
                task=c.get("task", ""),
                system_prompt=c.get("system_prompt"),
                response=c.get("response", ""),
            )
            for c in data.get("sub_agent_calls", [])
        ]

        return ExecutionResult(
            success=data.get("success", False),
            output=data.get("output", ""),
            error=data.get("error"),
            truncated=data.get("truncated", False),
            execution_time_ms=data.get("execution_time_ms", 0),
            answer=data.get("answer", {"content": "", "ready": False}),
            files_created=data.get("files_created", []),
            variables=data.get("variables", {}),
            state=data.get("state", {"variables": {}, "functions": {}, "classes": {}, "modules": []}),
            state_formatted=data.get("state_formatted", "(empty state)"),
            sub_agent_calls=sub_agent_calls,
            done=data.get("done", False),
            final_answer=data.get("final_answer"),
            iteration=data.get("iteration", 0),
            reward=data.get("reward", 0.0),
            episode_state=data.get("episode_state", {}),
        )

    async def execute_bash(
        self, code: str, timeout: Optional[int] = None, session_id: Optional[str] = None
    ) -> ExecutionResult:
        """
        Execute bash commands in a session.

        Args:
            code: Bash commands to execute
            timeout: Optional timeout in seconds
            session_id: Session ID (uses current session if not provided)

        Returns:
            ExecutionResult with output, errors
        """
        sid = session_id or self.session_id
        if not sid:
            raise ValueError("No session ID. Call create_session() first.")

        payload: Dict[str, Any] = {"code": code}
        if timeout is not None:
            payload["timeout"] = timeout

        session = await self._session()
        async with session.post(
            f"{self.server_url}/session/{sid}/execute_bash",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=max(timeout or 120, 120)),
        ) as resp:
            resp.raise_for_status()
            data = await resp.json()

        return ExecutionResult(
            success=data.get("success", False),
            output=data.get("output", ""),
            error=data.get("error"),
            truncated=data.get("truncated", False),
            execution_time_ms=data.get("execution_time_ms", 0),
            files_created=data.get("files_created", []),
        )

    async def get_state(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Get current session state."""
        sid = session_id or self.session_id
        if not sid:
            raise ValueError("No session ID.")

        session = await self._session()
        async with session.get(
            f"{self.server_url}/session/{sid}/state",
            timeout=aiohttp.ClientTimeout(total=30),
        ) as resp:
            resp.raise_for_status()
            return await resp.json()

    async def reset_session(self, session_id: Optional[str] = None) -> None:
        """Reset session to initial state."""
        sid = session_id or self.session_id
        if not sid:
            return

        session = await self._session()
        async with session.post(
            f"{self.server_url}/session/{sid}/reset",
            timeout=aiohttp.ClientTimeout(total=30),
        ) as resp:
            resp.raise_for_status()

    async def delete_session(self, session_id: Optional[str] = None) -> None:
        """Delete session and cleanup resources."""
        sid = session_id or self.session_id
        if not sid:
            return

        try:
            session = await self._session()
            async with session.delete(
                f"{self.server_url}/session/{sid}",
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                resp.raise_for_status()
        except Exception:
            pass

        if sid == self.session_id:
            self.session_id = None

    async def execute_stateless(
        self,
        code: str,
        context: Optional[str] = None,
    ) -> ExecutionResult:
        """
        Execute code without session management.

        Creates a temporary sandbox, executes, and cleans up.
        Use for one-off executions.
        """
        payload: Dict[str, Any] = {"code": code}
        if context:
            payload["context"] = context

        session = await self._session()
        async with session.post(
            f"{self.server_url}/execute",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=120),
        ) as resp:
            resp.raise_for_status()
            data = await resp.json()

        return ExecutionResult(
            success=data.get("success", False),
            output=data.get("output", ""),
            error=data.get("error"),
            execution_time_ms=data.get("execution_time_ms", 0),
            answer=data.get("answer", {"content": "", "ready": False}),
        )

    def __enter__(self) -> "REPLClient":
        """Sync context manager entry. Prefer async with for proper cleanup."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        """Sync context manager exit. Does not close HTTP session; use async with for that."""
        return False
