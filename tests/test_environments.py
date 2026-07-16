"""Smoke tests for the Gymkhana environment abstraction."""

from contextlib import asynccontextmanager
from typing import Any, Dict, List

import pytest

from gymkhana.envs.config import EnvironmentType
from gymkhana.envs import ENVIRONMENTS, Environment, Task, register_environment
from gymkhana.envs.config import EnvConfig
from gymkhana.envs.math_python import MathPythonEnv
from gymkhana.core.models import TrajectoryResult, Turn


@pytest.fixture
def config() -> EnvConfig:
    return EnvConfig(name="dummy-test")


@register_environment(name="dummy-test")
class DummyEnvironment(Environment):
    name: str = "dummy-test"

    def load_tasks(self, limit=None):  # type: ignore[override]
        count = limit or 1
        return [Task(id=f"task-{i}", prompt=f"prompt-{i}") for i in range(count)]

    async def run_task(self, task: Task) -> TrajectoryResult:  # type: ignore[override]
        turns = [Turn(role="user", content=task.prompt)]
        return TrajectoryResult(
            success=True,
            final_answer="42",
            turns=turns,
            num_code_blocks=0,
            system_prompt="",
        )


@pytest.mark.asyncio
async def test_environment_generate_runs(config: EnvConfig):
    env = ENVIRONMENTS.create("dummy-test", config)
    summary = await env.generate(limit=2)
    assert summary.environment == "dummy-test"
    assert summary.total_tasks == 2
    assert summary.successful == 2
    assert summary.failed == 0
    assert env.stats.total == 2


@pytest.mark.asyncio
async def test_registry_lookup_by_type(config: EnvConfig):
    # Register a temporary class for lookup by EnvironmentType if needed later
    env = ENVIRONMENTS.create("dummy-test", config)
    assert isinstance(env, Environment)


from unittest.mock import MagicMock, AsyncMock, patch
import gymkhana.envs.environment
from gymkhana.core.services import ServiceContainer, SandboxService

@pytest.mark.asyncio
async def test_environment_uses_injected_sandbox(config: EnvConfig):
    # Create mock sandbox service (async methods)
    mock_sandbox = MagicMock(spec=SandboxService)
    mock_sandbox.create_session = AsyncMock(return_value="session-123")
    mock_sandbox.delete_session = AsyncMock()

    # Mock execute to return a result with the answer "42"
    mock_result = MagicMock()
    mock_result.output = "42"
    mock_result.success = True
    mock_result.files_created = []
    mock_result.files_modified = []
    mock_sandbox.execute = AsyncMock(return_value=mock_result)

    container = ServiceContainer(sandbox=mock_sandbox)

    # Create environment with injected services
    env = MathPythonEnv(config=config, services=container)

    # Mock generate_response on the class/instance using patch
    # We use a context manager to patch the method on the class for the duration of the test
    # or just patch the instance method if pydantic allows, but here patch.object is safer
    with patch.object(MathPythonEnv, 'generate_response', new_callable=AsyncMock) as mock_generate:
        mock_generate.side_effect = [
            "Let's calculate.\n<python>\nprint(42)\n</python>",
            "The answer is \\boxed{42}.",
            ""  # Safety stop
        ]

        task = Task(id="test-1", prompt="Calculate 42")
        result = await env.run_task(task)

    # Verify sandbox usage (async methods)
    mock_sandbox.create_session.assert_called_once()
    mock_sandbox.execute.assert_called_with("print(42)", session_id="session-123")
    mock_sandbox.delete_session.assert_any_call("session-123")

    assert result.success
    assert result.final_answer == "The answer is \\boxed{42}."


@pytest.mark.asyncio
async def test_environment_multi_rollout_uses_repl_sessions(config: EnvConfig):
    """With num_rollouts=2, env creates 2 sandbox sessions and returns best by total_reward."""
    mock_sandbox = MagicMock(spec=SandboxService)
    mock_sandbox.create_session = AsyncMock(side_effect=["session-1", "session-2"])
    mock_sandbox.delete_session = AsyncMock()

    mock_result = MagicMock()
    mock_result.output = "42"
    mock_result.success = True
    mock_result.files_created = []
    mock_result.files_modified = []
    mock_result.reward = 0.5
    mock_sandbox.execute = AsyncMock(return_value=mock_result)

    mock_inference = MagicMock()
    mock_inference.batch_generate_identical = AsyncMock(
        return_value=[
            "Let's calculate.\n<python>\nprint(42)\n</python>",
            "Let's calculate.\n<python>\nprint(42)\n</python>",
        ]
    )
    mock_inference.batch_generate_conversations = AsyncMock(
        return_value=[
            "<final_answer>The answer is \\boxed{42}.</final_answer>",
            "<final_answer>The answer is \\boxed{42}.</final_answer>",
        ]
    )

    container = ServiceContainer(sandbox=mock_sandbox)
    config.dataset.num_rollouts = 2
    config.repl.max_turns = 2
    env = MathPythonEnv(config=config, services=container, num_rollouts=2)
    env._inference_service = mock_inference

    task = Task(id="test-1", prompt="Calculate 42")
    result = await env.run_task(task)

    assert mock_sandbox.create_session.call_count == 2
    assert mock_sandbox.delete_session.call_count == 2
    mock_inference.batch_generate_identical.assert_called_once()
    mock_inference.batch_generate_conversations.assert_called()
    assert result.success
    assert "42" in result.final_answer


if __name__ == "__main__":
    import asyncio
    async def run():
        pass
    asyncio.run(run())
