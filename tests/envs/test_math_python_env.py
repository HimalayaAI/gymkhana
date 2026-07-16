"""Unit tests for MathPythonEnv."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any, Dict, List

import pytest

from gymkhana.envs.config import EnvironmentType
from gymkhana.envs.math_python import MathPythonEnv
from gymkhana.envs import Task


@pytest.mark.asyncio
async def test_math_environment_rollout_with_mocked_repl(monkeypatch: pytest.MonkeyPatch):
    config = MathPythonEnv.default_config.model_copy(deep=True)
    config.dataset.environment = EnvironmentType.MATH_PYTHON
    config.dataset.limit = 1

    dataset_records: List[Dict[str, Any]] = [
        {
            "uuid": "math-1",
            "problem": "Compute 2 + 2",
            "expected_answer": "4",
        }
    ]

    async def fake_generate_response(self, messages, system_prompt):
        if not getattr(self, "_response_calls", None):
            self._response_calls = 0
        responses = [
            "<python>\nprint(2 + 2)\n</python>",
            "<final_answer>The answer is \\boxed{4}</final_answer>",
        ]
        if self._response_calls < len(responses):
            result = responses[self._response_calls]
        else:
            result = ""
        self._response_calls += 1
        return result

    class DummyExecution:
        def __init__(self) -> None:
            self.output = "4"
            self.execution_time_ms = 1
            self.error = None
            self.truncated = False
            self.files_created: List[str] = []
            self.sub_agent_calls: List[Any] = []
            self.state_formatted = "(empty state)"
            self.reward = 0.0

    class DummyRepl:
        async def execute(self, code: str) -> DummyExecution:
            return DummyExecution()

    @asynccontextmanager
    async def fake_repl_session(self, **_: Any):
        yield DummyRepl()

    monkeypatch.setattr(MathPythonEnv, "_load_dataset", lambda self: dataset_records)
    monkeypatch.setattr(MathPythonEnv, "generate_response", fake_generate_response)
    monkeypatch.setattr(MathPythonEnv, "repl_session", fake_repl_session)

    env = MathPythonEnv(config=config)

    tasks = env.load_tasks(limit=1)
    assert len(tasks) == 1

    result = await env.run_task(tasks[0])

    assert result.success is True
    assert "4" in result.final_answer


def test_math_environment_load_tasks_respects_mapping(monkeypatch: pytest.MonkeyPatch):
    config = MathPythonEnv.default_config.model_copy(deep=True)
    config.dataset.field_mapping = {
        "id": "uuid",
        "prompt": "problem",
        "expected_answer": "expected_answer",
        "context": "context",
    }
    config.dataset.limit = None

    dataset_records: List[Dict[str, Any]] = [
        {
            "uuid": "math-1",
            "problem": "Compute 2 + 2",
            "expected_answer": "4",
            "context": "Given integers",
            "difficulty": "easy",
        },
        {
            "uuid": "math-2",
            "problem": "Compute 3 + 3",
            "expected_answer": "6",
            "context": None,
            "difficulty": "easy",
        },
        {
            "uuid": "math-3",
            "problem": "Compute 5 + 5",
            "expected_answer": "10",
            "context": None,
            "difficulty": "medium",
        },
    ]

    monkeypatch.setattr(MathPythonEnv, "_load_dataset", lambda self: dataset_records)

    env = MathPythonEnv(config=config)
    tasks = env.load_tasks(limit=2)

    assert len(tasks) == 2
    first, second = tasks
    assert first.id == "math-1"
    assert first.context == "Given integers"
    assert first.metadata.get("difficulty") == "easy"
    assert second.id == "math-2"
    assert second.context is None
    assert second.metadata.get("difficulty") == "easy"


def test_math_environment_instructions_toggle():
    config = MathPythonEnv.default_config.model_copy(deep=True)
    env = MathPythonEnv(config=config)
    task = Task(id="t1", prompt="Solve")
    instructions = env.get_environment_instructions(task)
    assert instructions.strip().startswith("## Math Problem Instructions")

    config_without = MathPythonEnv.default_config.model_copy(deep=True)
    config_without.dataset.include_instructions = False
    env_without = MathPythonEnv(config=config_without)
    assert env_without.get_environment_instructions(task) == ""
