"""Tests for sandbox service implementations."""

import pytest
from unittest.mock import Mock, patch, MagicMock, AsyncMock

from gymkhana.core.services.sandboxes.sandbox import SandboxService
from gymkhana.core.services.sandboxes.repl import REPLSandbox
from gymkhana.core.services.container import ServiceContainer
from gymkhana.core.models.execution import ExecutionResult


def create_mock_execution_result(**kwargs):
    """Create a mock ExecutionResult with defaults."""
    defaults = {
        "success": True,
        "output": "hello",
        "error": None,
        "truncated": False,
        "execution_time_ms": 100,
        "state": {
            "variables": {},
            "modules": [],
            "functions": {},
            "classes": {},
        },
        "files_created": [],
        "reward": 0.0,
        "done": False,
        "iteration": 1,
    }
    defaults.update(kwargs)
    return ExecutionResult(**defaults)


class TestREPLSandbox:
    """Tests for REPLSandbox."""

    def test_init_default_url(self):
        """Test default server URL."""
        service = REPLSandbox()
        assert service.server_url == "http://localhost:5003"

    def test_init_custom_url(self):
        """Test custom server URL."""
        service = REPLSandbox(server_url="http://custom:8000")
        assert service.server_url == "http://custom:8000"

    def test_session_id_initially_none(self):
        """Test session_id is None before create_session."""
        service = REPLSandbox()
        assert service.session_id is None

    @pytest.mark.asyncio
    @patch("gymkhana.core.services.sandboxes.repl.REPLClient")
    async def test_create_session_delegates_to_client(self, mock_client_class):
        """Test create_session calls REPLClient.create_session."""
        mock_client = MagicMock()
        mock_client.create_session = AsyncMock(return_value="session-123")
        mock_client_class.return_value = mock_client

        service = REPLSandbox()
        session_id = await service.create_session(context="test context")

        mock_client.create_session.assert_called_once()
        assert session_id == "session-123"
        assert service.session_id == "session-123"

    @pytest.mark.asyncio
    @patch("gymkhana.core.services.sandboxes.repl.REPLClient")
    async def test_execute_uses_current_session(self, mock_client_class):
        """Test execute uses current session if not specified."""
        mock_client = MagicMock()
        mock_client.create_session = AsyncMock(return_value="session-123")
        mock_client.execute = AsyncMock(return_value=create_mock_execution_result())
        mock_client_class.return_value = mock_client

        service = REPLSandbox()
        await service.create_session()
        result = await service.execute("print('hello')")

        mock_client.execute.assert_called_with("print('hello')", "session-123")
        assert result.success is True

    @pytest.mark.asyncio
    @patch("gymkhana.core.services.sandboxes.repl.REPLClient")
    async def test_execute_with_explicit_session(self, mock_client_class):
        """Test execute with explicit session_id."""
        mock_client = MagicMock()
        mock_client.create_session = AsyncMock(return_value="dummy-session")
        mock_client.execute = AsyncMock(return_value=create_mock_execution_result())
        mock_client_class.return_value = mock_client

        service = REPLSandbox()
        await service.create_session()
        await service.execute("print('hello')", session_id="explicit-session")

        mock_client.execute.assert_called_with("print('hello')", "explicit-session")

    @pytest.mark.asyncio
    @patch("gymkhana.core.services.sandboxes.repl.REPLClient")
    async def test_delete_session_clears_current(self, mock_client_class):
        """Test delete_session clears current session ID."""
        mock_client = MagicMock()
        mock_client.create_session = AsyncMock(return_value="session-123")
        mock_client.delete_session = AsyncMock()
        mock_client_class.return_value = mock_client

        service = REPLSandbox()
        await service.create_session()
        assert service.session_id == "session-123"

        await service.delete_session()
        assert service.session_id is None

    @pytest.mark.asyncio
    @patch("gymkhana.core.services.sandboxes.repl.REPLClient")
    async def test_async_session_context_cleanup(self, mock_client_class):
        """Test async session() context manager cleans up on exit."""
        mock_client = MagicMock()
        mock_client.create_session = AsyncMock(return_value="session-123")
        mock_client.delete_session = AsyncMock()
        mock_client_class.return_value = mock_client

        async with REPLSandbox().session() as state:
            assert state is not None

        mock_client.create_session.assert_called_once()
        mock_client.delete_session.assert_called_once()

    @pytest.mark.asyncio
    @patch("gymkhana.core.services.sandboxes.repl.REPLClient")
    async def test_health_check_delegates(self, mock_client_class):
        """Test health_check delegates to client."""
        mock_client = MagicMock()
        mock_client.health_check = AsyncMock(return_value=True)
        mock_client_class.return_value = mock_client

        service = REPLSandbox()
        result = await service.health_check()

        assert result is True
        mock_client.health_check.assert_called_once()


class TestServiceContainer:
    """Tests for ServiceContainer."""

    def test_empty_container(self):
        """Test creating empty container."""
        container = ServiceContainer()
        assert container.sandbox is None
        assert container.inference is None
        assert container.storage is None

    def test_container_with_sandbox(self):
        """Test container with sandbox service."""
        mock_sandbox = MagicMock(spec=SandboxService)
        container = ServiceContainer(sandbox=mock_sandbox)
        assert container.sandbox is mock_sandbox

    def test_container_model_copy(self):
        """Test container can be copied."""
        mock_sandbox = MagicMock(spec=SandboxService)
        container = ServiceContainer(sandbox=mock_sandbox)
        copied = container.model_copy()
        assert copied.sandbox is mock_sandbox
