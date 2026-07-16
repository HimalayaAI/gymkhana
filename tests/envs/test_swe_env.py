"""Unit tests for SWEEnv."""

from __future__ import annotations

from unittest.mock import MagicMock

from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional
from unittest.mock import patch

import pytest

from gymkhana.envs.config import EnvironmentType
from gymkhana.envs.swe.swe_env import SWEEnv, SWETaskMetadata, SWEPatch
from gymkhana.envs import Task
from gymkhana.core.models.execution import ExecutionResult


@pytest.fixture
def swe_config():
    """Create a default SWE config for testing."""
    config = SWEEnv.default_config.model_copy(deep=True)
    config.dataset.environment = EnvironmentType.SWE
    config.dataset.limit = 2
    # Mock dataset name so we can intercept load_dataset
    config.dataset.dataset_name = "mock/swe-dataset"
    return config


@pytest.mark.asyncio
async def test_swe_environment_load_tasks(swe_config, monkeypatch):
    """Test loading tasks from a mock dataset."""

    # Mock dataset records
    mock_records = [
        {
            "instance_id": "test-task-1",
            "problem_statement": "Fix bug A",
            "repo": "test/repo1",
            "base_commit": "abc1234",
            "docker_image": "image:v1",
            "hints_text": "Check file X",
            "patch": "diff --git a/file.py...",
        },
        {
            "instance_id": "test-task-2",
            "problem_statement": "Fix bug B",
            "repo": "test/repo2",
            # Missing some fields to test defaults
        }
    ]

    # Mock _load_dataset to return our records
    monkeypatch.setattr(SWEEnv, "_load_dataset", lambda self: mock_records)

    env = SWEEnv(config=swe_config)
    tasks = env.load_tasks()

    assert len(tasks) == 2

    # Verify first task
    t1 = tasks[0]
    assert t1.id == "test-task-1"
    assert t1.prompt == "Fix bug A"

    # Check metadata conversion
    meta1 = SWETaskMetadata(**t1.metadata)
    assert meta1.repo == "test/repo1"
    assert meta1.commit == "abc1234"
    assert meta1.image_name == "image:v1"
    assert meta1.hints == "Check file X"
    assert meta1.patch == "diff --git a/file.py..."

    # Verify second task (defaults)
    t2 = tasks[1]
    assert t2.id == "test-task-2"
    meta2 = SWETaskMetadata(**t2.metadata)
    assert meta2.repo == "test/repo2"
    assert meta2.commit is None


def test_swe_environment_formatting(swe_config):
    """Test prompt and instruction formatting."""
    env = SWEEnv(config=swe_config)

    metadata = SWETaskMetadata(
        task_id="t1",
        repo="my/repo",
        problem_statement="Fix it",
        hints="Look here"
    )
    task = Task(id="t1", prompt="Fix it", metadata=metadata.to_dict())

    # Check initial message format
    msg = env.format_initial_message(task)
    assert "Fix it" in msg
    assert "Repository: my/repo" in msg
    assert "Hints: Look here" in msg

    # Check instructions
    instr = env.get_environment_instructions(task)
    assert "SWE Task Instructions" in instr
    assert "bash commands" in instr


@pytest.mark.asyncio
async def test_swe_rollout_mocked(swe_config, monkeypatch):
    """Test a full rollout with mocked LLM and Sandbox."""

    # Mock dataset
    mock_records = [{"instance_id": "t1", "problem_statement": "Fix it", "repo": "r1", "docker_image": "img1"}]
    monkeypatch.setattr(SWEEnv, "_load_dataset", lambda self: mock_records)

    # Mock LLM response generation
    async def fake_generate_response(self, messages, system_prompt, **kwargs):
        # Return a sequence of responses
        if not hasattr(self, "_call_count"):
            self._call_count = 0

        responses = [
            # Turn 1: Explore file
            "<bash>\nls -la\n</bash>",
            # Turn 2: Final answer
            "<final_answer>Fixed it</final_answer>"
        ]

        if self._call_count < len(responses):
            resp = responses[self._call_count]
            self._call_count += 1
            return resp
        return ""

    # Mock the method on the class for the test duration
    # We can't set it on the instance if pydantic validates assignment
    # But we can patch the class method or set it if we subclass (as done in integration test)
    # Using monkeypatch on the class works:
    monkeypatch.setattr(SWEEnv, "generate_response", fake_generate_response)

    # Mock DockerSandboxService
    # We need to mock the import in repl_session
    # Since it's imported inside the method, we patch sys.modules or patch the class where it's used
    # But repl_session does local import.
    # We can patch `gymkhana.core.services.sandboxes.DockerSandboxService`

    from unittest.mock import AsyncMock
    with patch("gymkhana.core.services.sandboxes.DockerSandboxService") as MockService:
        mock_service_instance = MockService.return_value

        # Mock create_session (async)
        mock_service_instance.create_session = AsyncMock(return_value="session-1")
        mock_service_instance.delete_session = AsyncMock()

        # Mock SandboxSession (the proxy) - execute/execute_bash are async
        with patch("gymkhana.envs.environment.SandboxSession") as MockSessionProxy:
            mock_proxy = MagicMock()
            MockSessionProxy.return_value = mock_proxy
            mock_proxy.execute = AsyncMock(return_value=ExecutionResult(success=True, output="ok"))
            mock_proxy.execute_bash = AsyncMock(return_value=ExecutionResult(success=True, output="file1"))

            env = SWEEnv(config=swe_config)

            # Since load_tasks is mocked
            tasks = env.load_tasks()

            # Run task
            result = await env.run_task(tasks[0])

            # Verify result
            assert result.success is True
            assert result.final_answer == "Fixed it"

            # Verify DockerSandboxService was used
            from unittest.mock import ANY
            MockService.assert_called_with(config=ANY, instance_id="t1")
            mock_service_instance.create_session.assert_called()

            # Verify execution
            # The environment calls repl.execute_bash for bash block
            mock_proxy.execute_bash.assert_called()
