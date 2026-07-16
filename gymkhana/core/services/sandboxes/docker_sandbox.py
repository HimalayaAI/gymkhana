"""Docker-based sandbox service for containerized code execution.

Provides SandboxService implementation using Docker containers for
isolated code execution. Supports both simple Python containers and
SWE-bench style containers with repository code mounted at /testbed.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List, Optional

from pydantic import Field, PrivateAttr

from gymkhana.core.models.execution import ExecutionResult
from gymkhana.core.services.sandboxes.client import REPLClient
from gymkhana.core.services.sandboxes.sandbox import (
    SandboxService,
    SandboxError,
    SandboxNotReadyError,
    SandboxTimeoutError,
    SandboxResourceError,
)
from gymkhana.core.models.sandbox import (
    SandboxConfig,
    SandboxBackend,
    SessionConfig,
    ResourceConfig,
    SessionState,
    SessionStatus,
    InterpreterState,
    ExecutionMetrics,
)

# Optional Docker import
try:
    import docker
    DOCKER_AVAILABLE = True
except ImportError:
    docker = None
    DOCKER_AVAILABLE = False

logger = logging.getLogger(__name__)


class DockerContainerSession:
    """Manages a single Docker container running REPL server.

    This is a low-level helper class that handles container lifecycle:
    - Pulling images
    - Starting containers with resource limits
    - Waiting for REPL server to be ready
    - Stopping and cleaning up containers
    """

    def __init__(
        self,
        image_name: str,
        instance_id: str,
        *,
        resources: ResourceConfig,
        client: Optional[Any] = None,
        server_port: int = 5003,
        startup_timeout: int = 120,
        cleanup: bool = True,
        volumes: Optional[Dict[str, Dict[str, str]]] = None,
        environment: Optional[Dict[str, str]] = None,
        startup_commands: Optional[List[str]] = None,
    ):
        """Initialize Docker container session.

        Args:
            image_name: Docker image name
            instance_id: Unique instance identifier
            resources: Resource configuration
            server_port: Port for REPL server inside container
            startup_timeout: Timeout for container startup
            cleanup: Whether to remove container on stop
            volumes: Volume mounts for container
            environment: Environment variables for container
            startup_commands: Shell commands to run at container startup
        """
        if not DOCKER_AVAILABLE:
            raise ImportError("docker package required: pip install docker")

        self.image_name = image_name
        self.instance_id = instance_id
        self.resources = resources
        self.server_port = server_port
        self.startup_timeout = startup_timeout
        self.cleanup = cleanup
        self.volumes = volumes or {}
        self.environment = environment or {}
        self.startup_commands = startup_commands or []

        # Automatically mount current project root to /gymkhana
        # Get gymkhana project root (5 levels up from this file)
        try:
            from pathlib import Path
            gymkhana_path = Path(__file__).resolve().parent.parent.parent.parent.parent
            if '/gymkhana' not in self.volumes:
                self.volumes[str(gymkhana_path)] = {
                    'bind': '/gymkhana',
                    'mode': 'ro'
                }
        except Exception as e:
            logger.warning(f"Failed to auto-detect gymkhana path for mounting: {e}")

        # Automatically propagate DB_* and LITELLM_* environment variables
        import os
        for k, v in os.environ.items():
            if (k.startswith('DB_') or k.startswith('LITELLM_')) and k not in self.environment:
                self.environment[k] = v

        if client:
            self.client = client
        else:
            self.client = docker.from_env()
        self.container = None
        self.host_port: Optional[int] = None
        self._ready = False

    @property
    def repl_url(self) -> str:
        """Get REPL server URL."""
        if not self.host_port:
            raise SandboxNotReadyError("Container not started")
        return f"http://localhost:{self.host_port}"

    @property
    def is_ready(self) -> bool:
        """Check if container is ready."""
        return self._ready

    def start(self) -> str:
        """Start Docker container with REPL server.

        Returns:
            URL of the REPL server
        """
        logger.info(f"Starting Docker container for {self.instance_id}")

        # Pull image if needed
        self._ensure_image()

        # Remove existing container with same name
        self._remove_existing_container()

        # Build startup script
        startup_script = self._build_startup_script()

        # Start container with resource limits
        try:
            self.container = self.client.containers.run(
                image=self.image_name,
                command=["/bin/bash", "-c", startup_script],
                detach=True,
                ports={f'{self.server_port}/tcp': None},
                remove=self.cleanup,
                name=f"sandbox-{self.instance_id}",
                volumes=self.volumes,
                environment={
                    'PYTHONUNBUFFERED': '1',
                    **self.environment,
                },
                # Resource limits from config
                mem_limit=self.resources.memory_limit_docker,
                nano_cpus=int(self.resources.cpu_cores * 1e9),
            )
        except docker.errors.APIError as e:
            raise SandboxResourceError(f"Failed to start container: {e}")

        # Get mapped host port
        self._wait_for_port_mapping()

        # Wait for server to be ready
        self._wait_for_server()

        self._ready = True
        logger.info(f"Container started: {self.container.short_id}, port: {self.host_port}")

        return self.repl_url

    def _ensure_image(self) -> None:
        """Pull Docker image if not available locally."""
        try:
            self.client.images.get(self.image_name)
            logger.info(f"Using cached image: {self.image_name}")
        except docker.errors.ImageNotFound:
            logger.info(f"Pulling image: {self.image_name}")
            print(f"[Docker] Pulling image: {self.image_name}")
            print(f"[Docker] This may take several minutes...")

            # Use specific pull if it's the real client, otherwise just call it
            if hasattr(self.client, "api"):
                for line in self.client.api.pull(self.image_name, stream=True, decode=True):
                    if 'status' in line:
                        status = line['status']
                        if 'id' in line:
                            layer_id = line['id']
                            progress = line.get('progress', '')
                            print(f"[Docker]   {layer_id}: {status} {progress}", end='\r')
            else:
                self.client.pull(self.image_name)

            print(f"\n[Docker] Pull complete!")

    def _remove_existing_container(self) -> None:
        """Remove container with same name if exists."""
        container_name = f"sandbox-{self.instance_id}"
        try:
            old_container = self.client.containers.get(container_name)
            logger.info(f"Removing existing container: {container_name}")
            old_container.remove(force=True)
        except (docker.errors.NotFound, docker.errors.APIError):
            pass

    def _build_startup_script(self) -> str:
        """Build the container startup script."""
        script_parts = [
            "set -e",
            "echo 'Installing REPL server dependencies...'",
            "pip install -q flask 2>&1 | grep -v 'already satisfied' || true",
        ]

        # Add startup commands if provided
        if self.startup_commands:
            script_parts.append("echo 'Running startup commands...'")
            for cmd in self.startup_commands:
                script_parts.append(f"echo '  > {cmd}'")
                script_parts.append(cmd)

        # Start REPL server
        script_parts.extend([
            f"echo 'Starting REPL server on port {self.server_port}...'",
            "cd /gymkhana/gymkhana/core/services/sandboxes/server",
            f"python app.py --port {self.server_port} --host 0.0.0.0"
        ])

        return "\n".join(script_parts)

    def _wait_for_port_mapping(self) -> None:
        """Wait for container port mapping to be available."""
        self.container.reload()
        port_mapping = self.container.ports.get(f'{self.server_port}/tcp')
        if not port_mapping:
            raise SandboxResourceError("Failed to get port mapping")
        self.host_port = int(port_mapping[0]['HostPort'])

    def _wait_for_server(self) -> None:
        """Wait for REPL server to be ready."""
        import requests

        url = f"http://localhost:{self.host_port}/health"
        start_time = time.time()

        while time.time() - start_time < self.startup_timeout:
            try:
                response = requests.get(url, timeout=1)
                if response.status_code == 200:
                    logger.info("REPL server is ready")
                    return
            except requests.exceptions.RequestException:
                pass
            time.sleep(0.5)

        raise SandboxTimeoutError(f"REPL server did not start within {self.startup_timeout}s")

    def stop(self) -> None:
        """Stop and optionally remove container."""
        if not self.container:
            return

        logger.info(f"Stopping container: {self.container.short_id}")
        self._ready = False

        try:
            self.container.stop(timeout=10)
            if self.cleanup:
                self.container.remove(force=True)
                logger.info("Container removed")
        except Exception as e:
            logger.error(f"Error stopping container: {e}")

        self.container = None
        self.host_port = None

    def get_logs(self, tail: int = 100) -> str:
        """Get container logs."""
        if not self.container:
            return ""
        return self.container.logs(tail=tail).decode('utf-8')

    def __enter__(self) -> str:
        """Context manager entry - returns REPL URL."""
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """Context manager exit - stops container."""
        self.stop()
        return False


class DockerSandboxService(SandboxService):
    """Docker-based sandbox service.

    Manages Docker containers for isolated code execution. Each session
    creates a new container with the REPL server running inside.

    Example:
        ```python
        config = SandboxConfig.for_swe("swebench/swesmith.x86_64.oauthlib")
        service = DockerSandboxService(
            config=config,
            instance_id="task-001",
        )

        async with service.session() as state:
            result = service.execute("import os; print(os.getcwd())")
            bash_result = service.execute_bash("ls -la /testbed")
        ```
    """

    # Configuration
    instance_id: str = Field(
        ...,
        description="Unique instance ID for the container"
    )

    # Private state
    _container_session: Optional[DockerContainerSession] = PrivateAttr(default=None)
    _repl_client: Optional[REPLClient] = PrivateAttr(default=None)
    _repl_session_id: Optional[str] = PrivateAttr(default=None)

    def __init__(self, **data: Any) -> None:
        # Set backend to docker if not specified
        if "config" in data and hasattr(data["config"], "backend"):
            if data["config"].backend != SandboxBackend.DOCKER:
                data["config"] = data["config"].model_copy(update={"backend": SandboxBackend.DOCKER})

        super().__init__(**data)
        self._container_session = None
        self._repl_client = None
        self._repl_session_id = None

    @property
    def container_logs(self) -> str:
        """Get container logs."""
        if self._container_session:
            return self._container_session.get_logs()
        return ""

    # ========================================================================
    # SandboxService Implementation
    # ========================================================================

    async def create_session(
        self,
        *,
        context: Optional[str] = None,
        config_override: Optional[SessionConfig] = None,
        volumes: Optional[Dict[str, Dict[str, str]]] = None,
        environment: Optional[Dict[str, str]] = None,
        # Legacy/Compatibility parameters
        packages: Optional[List[str]] = None,
        max_output_chars: Optional[int] = None,
        max_iterations: Optional[int] = None,
        enable_bash: Optional[bool] = None,
        sub_agent_config: Optional[Dict[str, Any]] = None,
        reward_config: Optional[Dict[str, float]] = None,
    ) -> str:
        """Create a new Docker sandbox session.

        Starts a Docker container and initializes REPL session inside.

        Args:
            context: Optional context to load
            config_override: Override session configuration
            volumes: Additional volume mounts
            environment: Additional environment variables
            sub_agent_config: Optional sub-agent configuration

        Returns:
            Session ID
        """
        cfg = self.get_effective_session_config(config_override)

        # Build volumes - always mount gymkhana
        from pathlib import Path
        gymkhana_path = Path(__file__).resolve().parent.parent.parent.parent.parent

        effective_volumes = {
            str(gymkhana_path): {
                'bind': '/gymkhana',
                'mode': 'ro'
            },
        }
        if volumes:
            effective_volumes.update(volumes)

        # Build environment
        effective_env = {
            'PYTHONPATH': '/gymkhana',
            **{k: v for k, v in os.environ.items()
               if k.startswith('DB_') or k.startswith('LITELLM_')},
        }
        # Remap localhost DB_HOST for Docker transparency (especially on Mac/Windows)
        if effective_env.get('DB_HOST') in ('localhost', '127.0.0.1'):
            effective_env['DB_HOST'] = 'host.docker.internal'

        if environment:
            effective_env.update(environment)

        # Start container and REPL client only when we don't have one yet (multi-session: one container, many sessions)
        if not self._container_session:
            logger.info(f"Starting Docker sandbox for {self.instance_id}")
            self._container_session = DockerContainerSession(
                image_name=self.config.docker_image,
                instance_id=self.instance_id,
                resources=self.config.resources,
                startup_timeout=self.config.timeouts.startup_seconds,
                cleanup=True,
                volumes=effective_volumes,
                environment=effective_env,
                startup_commands=self.config.session.startup_commands,
            )
            try:
                repl_url = self._container_session.start()
            except Exception as e:
                logger.error(f"Failed to start Docker sandbox: {e}")
                raise
            self._repl_client = REPLClient(server_url=repl_url)

        # Fallback for legacy parameters if config_override is not provided
        if config_override is None:
            if packages is not None: cfg.packages = packages
            if max_output_chars is not None: cfg.max_output_chars = max_output_chars
            if max_iterations is not None: cfg.max_iterations = max_iterations
            if enable_bash is not None: cfg.enable_bash = enable_bash

        # Prepare sub_agent_config - prioritize passed argument, then config
        effective_sub_agent = sub_agent_config
        if effective_sub_agent is None and self.config.sub_agent.enabled:
            effective_sub_agent = self.config.sub_agent.to_dict()

        session_id = await self._repl_client.create_session(
            context=context,
            packages=cfg.packages,
            max_output_chars=cfg.max_output_chars,
            max_iterations=cfg.max_iterations,
            enable_bash=cfg.enable_bash,
            sub_agent_config=effective_sub_agent,
            reward_config=reward_config or {
                "on_success": self.config.rewards.on_success,
                "on_iteration": self.config.rewards.on_iteration,
                "on_error": self.config.rewards.on_error,
                "on_failure": self.config.rewards.on_failure,
            },
        )

        self._repl_session_id = session_id
        self._create_session_state(
            session_id,
            cfg,
            context_size=len(context) if context else 0,
        )

        return session_id

    async def execute(
        self, code: str, session_id: Optional[str] = None
    ) -> ExecutionResult:
        """Execute Python code in the sandbox."""
        if not self._repl_client:
            raise SandboxNotReadyError("No active session")

        sid = session_id or self._repl_session_id
        state = self.get_session_state(sid)

        if state:
            state.mark_executing()

        result = await self._repl_client.execute(code, sid)

        # Update state
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
        """Execute bash commands in the sandbox."""
        if not self._repl_client:
            raise SandboxNotReadyError("No active session")

        sid = session_id or self._repl_session_id
        effective_timeout = timeout or self.config.timeouts.bash_seconds

        return await self._repl_client.execute_bash(code, effective_timeout, sid)

    async def get_state(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Get current session state."""
        if not self._repl_client:
            return {}

        sid = session_id or self._repl_session_id
        return await self._repl_client.get_state(sid)

    async def reset_session(self, session_id: Optional[str] = None) -> None:
        """Reset session state."""
        if not self._repl_client:
            return

        sid = session_id or self._repl_session_id
        await self._repl_client.reset_session(sid)

        state = self.get_session_state(sid)
        if state:
            state.interpreter = InterpreterState()
            state.episode = type(state.episode)(max_iterations=state.episode.max_iterations)
            state.metrics = type(state.metrics)()

    async def delete_session(self, session_id: Optional[str] = None) -> None:
        """Delete one REPL session. If this was the last session, stop the container."""
        sid = session_id or self._repl_session_id
        if not sid:
            return

        if self._repl_client:
            try:
                await self._repl_client.delete_session(sid)
            except Exception:
                pass

        self._cleanup_session_state(sid)
        if sid == self._repl_session_id:
            self._repl_session_id = next(iter(self._active_sessions), None) if self._active_sessions else None

        # When no sessions left, stop container and clear client
        if not self._active_sessions:
            if self._container_session:
                try:
                    self._container_session.stop()
                except Exception as e:
                    logger.error(f"Error stopping container: {e}")
                self._container_session = None
            self._repl_client = None
            self._repl_session_id = None

    async def health_check(self) -> bool:
        """Check if container and REPL are responsive."""
        if not self._repl_client:
            return False
        return await self._repl_client.health_check()


DockerSandboxService.model_rebuild()

__all__ = [
    "DockerContainerSession",
    "DockerSandboxService",
    "DOCKER_AVAILABLE",
]
