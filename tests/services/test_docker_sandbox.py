import pytest
from unittest.mock import MagicMock, patch

# Mock docker module if not available
try:
    import docker
except ImportError:
    import sys
    docker = MagicMock()
    # Mock error classes
    class MockError(Exception): pass
    docker.errors = MagicMock()
    docker.errors.ImageNotFound = MockError
    docker.errors.NotFound = MockError
    docker.errors.APIError = MockError
    sys.modules["docker"] = docker

from gymkhana.core.models.sandbox import (
    SandboxConfig,
    ResourceConfig,
    SessionStatus,
)
from gymkhana.core.services.sandboxes.docker_sandbox import (
    DockerSandboxService,
    DockerContainerSession,
)


class TestDockerContainerSession:
    """Tests for DockerContainerSession management."""

    @pytest.fixture
    def session(self):
        mock_client = MagicMock()
        resources = ResourceConfig(cpu_cores=4, memory_gb=8.0)
        session = DockerContainerSession(
            image_name="test-image",
            instance_id="test-id",
            resources=resources,
            client=mock_client,
        )
        return session

    def test_start_pulls_image_if_missing(self, session):
        """Test that start() pulls the image if it's not found locally."""
        # Ensure we use the correct exception class
        import docker
        session.client.images.get.side_effect = docker.errors.ImageNotFound("Missing")
        session.client.containers.run.return_value = MagicMock()
        session.client.api.pull.return_value = [{"status": "Done"}]

        with patch.object(DockerContainerSession, "_wait_for_server", return_value=True), \
             patch.object(DockerContainerSession, "_remove_existing_container"):
            session.start()

        session.client.api.pull.assert_called()

    def test_start_configures_resources(self, session):
        """Test that start() correctly maps ResourceConfig to Docker limits."""
        session.client.containers.run.return_value = MagicMock()

        with patch.object(DockerContainerSession, "_wait_for_server", return_value=True), \
             patch.object(DockerContainerSession, "_remove_existing_container"):
            session.start()

        # Check call to containers.run
        args, kwargs = session.client.containers.run.call_args
        assert kwargs["mem_limit"] == "8.0g"
        assert "nano_cpus" in kwargs
        assert kwargs["nano_cpus"] == int(4 * 1e9)

    def test_stop_removes_container(self, session):
        """Test that stop() removes the container."""
        mock_container = MagicMock()
        session.client.containers.run.return_value = mock_container

        with patch.object(DockerContainerSession, "_wait_for_server", return_value=True), \
             patch.object(DockerContainerSession, "_remove_existing_container"):
            session.start()

        session.stop()
        mock_container.remove.assert_called()


class TestDockerSandboxService:
    """Tests for DockerSandboxService coordination."""

    @pytest.fixture
    def service(self):
        config = SandboxConfig.for_swe("test-image")
        service = DockerSandboxService(
            config=config,
            instance_id="test-001"
        )
        return service

    @pytest.mark.asyncio
    async def test_create_session_starts_container(self, service):
        """Test that create_session() orchestrates container and REPL startup."""
        from unittest.mock import AsyncMock
        with patch("gymkhana.core.services.sandboxes.docker_sandbox.DockerContainerSession") as mock_container_class, \
             patch("gymkhana.core.services.sandboxes.docker_sandbox.REPLClient") as mock_repl_client_class:

            mock_container = MagicMock()
            mock_container.start.return_value = "http://localhost:1234"
            mock_container_class.return_value = mock_container

            mock_repl_client = MagicMock()
            mock_repl_client.create_session = AsyncMock(return_value="test-session-id")
            mock_repl_client_class.return_value = mock_repl_client

            session_id = await service.create_session()

            mock_container.start.assert_called_once()
            mock_repl_client_class.assert_called_with(server_url="http://localhost:1234")
            assert session_id == "test-session-id"

    @pytest.mark.asyncio
    async def test_delete_session_stops_container(self, service):
        """Test that delete_session() stops the container."""
        from unittest.mock import AsyncMock
        with patch("gymkhana.core.services.sandboxes.docker_sandbox.DockerContainerSession") as mock_container_class, \
             patch("gymkhana.core.services.sandboxes.docker_sandbox.REPLClient") as mock_repl_client_class:

            mock_container = MagicMock()
            mock_container_class.return_value = mock_container

            mock_repl_client = MagicMock()
            mock_repl_client.create_session = AsyncMock(return_value="test-session-id")
            mock_repl_client.delete_session = AsyncMock()
            mock_repl_client_class.return_value = mock_repl_client

            await service.create_session()
            await service.delete_session()

            mock_container.stop.assert_called_once()
            assert service.session_id is None
