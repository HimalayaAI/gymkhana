"""Unit tests for sandbox services.

Tests the refactored sandbox abstractions including:
- Configuration models (SandboxConfig, ResourceConfig, etc.)
- State tracking models (SessionState, InterpreterState, etc.)
- REPLSandbox service (mock-based)
"""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from gymkhana.core.services.sandboxes.server.engine import PythonSandbox
from gymkhana.core.models.sandbox import (
    SandboxBackend,
    ResourceConfig,
    TimeoutConfig,
    RewardConfig,
    SubAgentConfig,
    SessionConfig,
    SandboxConfig,
    SessionStatus,
    ExecutionMetrics,
    SessionMetrics,
    InterpreterState,
    EpisodeState,
    SessionState,
)
from gymkhana.core.models.execution import ExecutionResult
from gymkhana.core.services.sandboxes.sandbox import (
    SandboxError,
    SandboxNotReadyError,
)


class TestResourceConfig:
    """Tests for ResourceConfig."""

    def test_defaults(self):
        """Test default resource values."""
        config = ResourceConfig()
        assert config.cpu_cores == 2
        assert config.memory_gb == 4.0
        assert config.disk_size_gb == 10.0

    def test_memory_bytes(self):
        """Test memory_bytes property."""
        config = ResourceConfig(memory_gb=2.0)
        expected = int(2.0 * 1024 * 1024 * 1024)
        assert config.memory_bytes == expected

    def test_memory_limit_docker(self):
        """Test memory_limit_docker property."""
        config = ResourceConfig(memory_gb=8.0)
        assert config.memory_limit_docker == "8.0g"

    def test_validation_bounds(self):
        """Test validation of resource bounds."""
        with pytest.raises(ValueError):
            ResourceConfig(cpu_cores=0)  # Must be >= 1

        with pytest.raises(ValueError):
            ResourceConfig(memory_gb=0.1)  # Must be >= 0.5


class TestTimeoutConfig:
    """Tests for TimeoutConfig."""

    def test_defaults(self):
        """Test default timeout values."""
        config = TimeoutConfig()
        assert config.execution_seconds == 120
        assert config.bash_seconds == 60
        assert config.startup_seconds == 120
        assert config.session_minutes == 60


class TestRewardConfig:
    """Tests for RewardConfig (RLM-compatible)."""

    def test_defaults(self):
        """Test default reward values."""
        config = RewardConfig()
        assert config.on_success == 1.0
        assert config.on_iteration == 0.0
        assert config.on_error == -0.05
        assert config.on_failure == -0.1
        assert config.on_timeout == -0.1

    def test_custom_rewards(self):
        """Test custom reward configuration."""
        config = RewardConfig(
            on_success=2.0,
            on_iteration=-0.01,  # Penalty per step
        )
        assert config.on_success == 2.0
        assert config.on_iteration == -0.01


class TestSubAgentConfig:
    """Tests for SubAgentConfig."""

    def test_disabled_by_default(self):
        """Test sub-agent is disabled by default."""
        config = SubAgentConfig()
        assert config.enabled is False

    def test_to_dict_when_disabled(self):
        """Test to_dict returns empty when disabled."""
        config = SubAgentConfig(enabled=False)
        assert config.to_dict() == {}

    def test_to_dict_when_enabled(self):
        """Test to_dict when enabled."""
        config = SubAgentConfig(
            enabled=True,
            model="Hermes-4-70B",
            temperature=0.5,
        )
        d = config.to_dict()
        assert d["model"] == "Hermes-4-70B"
        assert d["temperature"] == 0.5


class TestSessionConfig:
    """Tests for SessionConfig."""

    def test_defaults(self):
        """Test default session values."""
        config = SessionConfig()
        assert config.max_output_chars == 8192
        assert config.max_iterations == 30
        assert config.enable_bash is False
        assert "numpy" in config.packages

    def test_swe_config(self):
        """Test SWE-style configuration."""
        config = SessionConfig(
            max_iterations=50,
            enable_bash=True,
            packages=["pytest", "git"],
        )
        assert config.enable_bash is True
        assert config.max_iterations == 50


class TestSandboxConfig:
    """Tests for SandboxConfig."""

    def test_defaults(self):
        """Test default sandbox config."""
        config = SandboxConfig()
        assert config.backend == SandboxBackend.REPL_HTTP
        assert config.server_url == "http://localhost:5003"
        assert isinstance(config.resources, ResourceConfig)
        assert isinstance(config.session, SessionConfig)

    def test_for_math_factory(self):
        """Test for_math factory method."""
        config = SandboxConfig.for_math()
        assert config.backend == SandboxBackend.REPL_HTTP
        assert config.session.max_iterations == 10
        assert config.session.enable_bash is False

    def test_for_swe_factory(self):
        """Test for_swe factory method."""
        config = SandboxConfig.for_swe("my-image:latest")
        assert config.backend == SandboxBackend.DOCKER
        assert config.docker_image == "my-image:latest"
        assert config.session.enable_bash is True
        assert config.resources.cpu_cores == 4

    def test_for_long_context_factory(self):
        """Test for_long_context factory method."""
        config = SandboxConfig.for_long_context()
        assert config.sub_agent.enabled is True
        assert config.session.max_iterations == 20


class TestInterpreterState:
    """Tests for InterpreterState."""

    def test_empty_state(self):
        """Test empty interpreter state."""
        state = InterpreterState()
        assert state.format_compact() == "(empty state)"

    def test_format_compact(self):
        """Test compact formatting."""
        state = InterpreterState(
            modules=["numpy", "pandas"],
            variables={"x": "int", "y": "list[10]"},
            functions={"solve": "solve(a, b)"},
        )
        formatted = state.format_compact()
        assert "imports: numpy, pandas" in formatted
        assert "functions: solve" in formatted
        assert "vars:" in formatted

    def test_from_dict(self):
        """Test creation from dictionary."""
        data = {
            "variables": {"x": "int"},
            "modules": ["sympy"],
            "functions": {},
            "classes": {},
        }
        state = InterpreterState.from_dict(data)
        assert state.variables == {"x": "int"}
        assert state.modules == ["sympy"]


class TestEpisodeState:
    """Tests for EpisodeState (RLM-compatible)."""

    def test_initial_state(self):
        """Test initial episode state."""
        state = EpisodeState()
        assert state.iteration == 0
        assert state.done is False
        assert state.final_answer is None

    def test_step(self):
        """Test stepping through episode."""
        state = EpisodeState(max_iterations=3)

        state.step()
        assert state.iteration == 1
        assert state.done is False

        state.step()
        state.step()
        assert state.iteration == 3
        assert state.done is True
        assert state.termination_reason == "max_iterations_reached"

    def test_complete(self):
        """Test completing episode with answer."""
        state = EpisodeState()
        state.step()
        state.complete("42")

        assert state.done is True
        assert state.final_answer == "42"
        assert state.termination_reason == "final_answer"

    def test_progress_ratio(self):
        """Test progress ratio calculation."""
        state = EpisodeState(max_iterations=10)
        assert state.progress_ratio == 0.0

        state.step()
        state.step()
        state.step()
        assert state.progress_ratio == 0.3

    def test_iterations_remaining(self):
        """Test iterations remaining."""
        state = EpisodeState(max_iterations=5)
        assert state.iterations_remaining == 5

        state.step()
        state.step()
        assert state.iterations_remaining == 3


class TestSessionMetrics:
    """Tests for SessionMetrics."""

    def test_record_execution(self):
        """Test recording execution metrics."""
        metrics = SessionMetrics()

        exec_metrics = ExecutionMetrics(
            execution_time_ms=100,
            output_chars=500,
            truncated=False,
        )

        metrics.record_execution(True, exec_metrics, reward=0.5)

        assert metrics.total_executions == 1
        assert metrics.successful_executions == 1
        assert metrics.total_execution_time_ms == 100
        assert metrics.total_reward == 0.5
        assert len(metrics.step_rewards) == 1

    def test_success_rate(self):
        """Test success rate calculation."""
        metrics = SessionMetrics()

        success = ExecutionMetrics(execution_time_ms=50)
        failure = ExecutionMetrics(execution_time_ms=50)

        metrics.record_execution(True, success)
        metrics.record_execution(True, success)
        metrics.record_execution(False, failure)

        assert metrics.success_rate == pytest.approx(2/3)

    def test_avg_execution_time(self):
        """Test average execution time calculation."""
        metrics = SessionMetrics()

        m1 = ExecutionMetrics(execution_time_ms=100)
        m2 = ExecutionMetrics(execution_time_ms=200)

        metrics.record_execution(True, m1)
        metrics.record_execution(True, m2)

        assert metrics.avg_execution_time_ms == 150.0


class TestSessionState:
    """Tests for SessionState."""

    def test_initial_state(self):
        """Test initial session state."""
        state = SessionState()
        assert state.status == SessionStatus.PENDING
        assert state.episode.iteration == 0

    def test_mark_ready(self):
        """Test marking session as ready."""
        state = SessionState()
        state.mark_ready()

        assert state.status == SessionStatus.READY
        assert state.ready_at is not None

    def test_record_execution(self):
        """Test recording execution."""
        state = SessionState()
        state.mark_ready()
        state.mark_executing()

        metrics = ExecutionMetrics(execution_time_ms=100)
        interpreter = InterpreterState(modules=["numpy"])

        state.record_execution(
            success=True,
            metrics=metrics,
            interpreter_state=interpreter,
            reward=0.5,
        )

        assert state.status == SessionStatus.READY
        assert state.metrics.total_executions == 1
        assert state.episode.iteration == 1
        assert "numpy" in state.interpreter.modules

    def test_mark_completed(self):
        """Test marking session as completed."""
        state = SessionState()
        state.mark_ready()
        state.mark_completed(answer="42")

        assert state.status == SessionStatus.COMPLETED
        assert state.episode.done is True
        assert state.episode.final_answer == "42"

    def test_to_summary(self):
        """Test summary generation."""
        state = SessionState()
        state.mark_ready()

        summary = state.to_summary()

        assert "session_id" in summary
        assert summary["status"] == "ready"
        assert summary["iteration"] == 0


class TestREPLSandboxWithMock:
    """Tests for REPLSandbox with mocked client."""

    @pytest.fixture
    def mock_client(self):
        """Create a mock REPL client (async methods)."""
        client = MagicMock()
        client.health_check = AsyncMock(return_value=True)
        client.create_session = AsyncMock(return_value="test-session-123")
        client.execute = AsyncMock(return_value=ExecutionResult(success=True, output="ok"))
        client.delete_session = AsyncMock()
        return client

    @pytest.fixture
    def sandbox(self, mock_client):
        """Create sandbox with mocked client."""
        from gymkhana.core.services.sandboxes.repl import REPLSandbox

        sandbox = REPLSandbox(server_url="http://localhost:5003")
        sandbox._client = mock_client
        return sandbox

    @pytest.mark.asyncio
    async def test_health_check(self, sandbox, mock_client):
        """Test health check."""
        result = await sandbox.health_check()
        assert result is True
        mock_client.health_check.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_session(self, sandbox, mock_client):
        """Test session creation."""
        session_id = await sandbox.create_session(context="test data")

        assert session_id == "test-session-123"
        assert sandbox.session_id == "test-session-123"
        mock_client.create_session.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_session_creates_state(self, sandbox, mock_client):
        """Test that create_session creates session state."""
        await sandbox.create_session()

        state = sandbox.current_session
        assert state is not None
        assert state.status == SessionStatus.PENDING

    @pytest.mark.asyncio
    async def test_execute_without_session_raises(self, sandbox):
        """Test execute without session raises error."""
        with pytest.raises(SandboxNotReadyError):
            await sandbox.execute("print('hello')")

    @pytest.mark.asyncio
    async def test_execute_updates_state(self, sandbox, mock_client):
        """Test that execute updates session state."""
        mock_client.execute.return_value = ExecutionResult(
            success=True,
            output="hello",
            execution_time_ms=100,
            state={
                "variables": {"x": "int"},
                "modules": ["numpy"],
                "functions": {},
                "classes": {},
            },
            files_created=[],
            reward=0.5,
            done=False,
            iteration=1,
        )

        await sandbox.create_session()
        result = await sandbox.execute("print('hello')")

        assert result.output == "hello"

        state = sandbox.current_session
        assert state.metrics.total_executions == 1
        assert state.episode.iteration == 1

    @pytest.mark.asyncio
    async def test_delete_session_cleans_up_state(self, sandbox, mock_client):
        """Test that delete_session cleans up state."""
        await sandbox.create_session()
        assert sandbox.current_session is not None

        await sandbox.delete_session()

        assert sandbox.session_id is None
        assert sandbox.current_session is None
        mock_client.delete_session.assert_called_once()


class TestSandboxFileReadingUTF8:
    """Regression coverage for issue #17: read_text() calls must pin UTF-8."""

    def test_read_file_preserves_devanagari(self, tmp_path):
        (tmp_path / "note.txt").write_text(
            "यसको नाम परीक्षण नियमावली हो।", encoding="utf-8"
        )
        sandbox = PythonSandbox(workspace_dir=str(tmp_path))
        content = sandbox._read_file("note.txt", raw=True)
        assert content == "यसको नाम परीक्षण नियमावली हो।"

    def test_read_file_json_preserves_devanagari(self, tmp_path):
        import json
        (tmp_path / "data.json").write_text(
            json.dumps({"title": "नेपालको राजधानी"}, ensure_ascii=False),
            encoding="utf-8",
        )
        sandbox = PythonSandbox(workspace_dir=str(tmp_path))
        parsed = sandbox._read_file("data.json")
        assert parsed["title"] == "नेपालको राजधानी"

    def test_search_files_matches_devanagari(self, tmp_path):
        (tmp_path / "doc.txt").write_text(
            "काठमाडौं नेपालको राजधानी हो।", encoding="utf-8"
        )
        sandbox = PythonSandbox(workspace_dir=str(tmp_path))
        results = sandbox._search_files("राजधानी")
        assert results