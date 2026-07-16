"""Unit tests for OolongEnv."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any, Dict, List

import pytest

from gymkhana.envs.config import EnvironmentType
from gymkhana.envs.oolong import OolongEnv
from gymkhana.envs import Task


@pytest.mark.asyncio
async def test_oolong_environment_rollout_with_mocked_repl(monkeypatch: pytest.MonkeyPatch):
    config = OolongEnv.default_config.model_copy(deep=True)
    config.dataset.environment = EnvironmentType.OOLONG
    config.dataset.limit = 1

    dataset_records: List[Dict[str, Any]] = [
        {
            "id": "oolong-1",
            "question": "Total number of rolls in this episode?",
            "answer": "42",
            "context_window_text": "Long context text with rolls...",
        }
    ]

    async def fake_generate_response(self, messages, system_prompt):
        if not getattr(self, "_response_calls", None):
            self._response_calls = 0
        responses = [
            "<python>\ncontext = read_file('context.txt')\nprint(context.count('rolls'))\n</python>",
            "<final_answer>The total number of rolls is \\boxed{42}</final_answer>",
        ]
        if self._response_calls < len(responses):
            result = responses[self._response_calls]
        else:
            result = ""
        self._response_calls += 1
        return result

    class DummyExecution:
        def __init__(self) -> None:
            self.output = "42"
            self.execution_time_ms = 1
            self.error = None
            self.truncated = False
            self.files_created: List[str] = ["context.txt"]
            self.sub_agent_calls: List[Any] = []
            self.state_formatted = "imports: re | vars: context: str"
            self.reward = 0.0

    class DummyRepl:
        async def execute(self, code: str) -> DummyExecution:
            return DummyExecution()

    @asynccontextmanager
    async def fake_repl_session(self, **_: Any):
        yield DummyRepl()

    monkeypatch.setattr(OolongEnv, "_load_dataset", lambda self: dataset_records)
    monkeypatch.setattr(OolongEnv, "generate_response", fake_generate_response)
    monkeypatch.setattr(OolongEnv, "repl_session", fake_repl_session)

    env = OolongEnv(config=config)

    tasks = env.load_tasks(limit=1)
    assert len(tasks) == 1

    result = await env.run_task(tasks[0])

    assert result.success is True
    assert "42" in result.final_answer


def test_oolong_environment_load_tasks_respects_mapping(monkeypatch: pytest.MonkeyPatch):
    config = OolongEnv.default_config.model_copy(deep=True)
    config.dataset.field_mapping = {
        "id": "id",
        "prompt": "question",
        "expected_answer": "answer",
        "context": "context_window_text",
    }
    config.dataset.limit = None

    dataset_records: List[Dict[str, Any]] = [
        {
            "id": "oolong-1",
            "question": "Total number of rolls?",
            "answer": "42",
            "context_window_text": "Long context with rolls...",
            "difficulty": "medium",
        },
        {
            "id": "oolong-2",
            "question": "How many attacks?",
            "answer": "15",
            "context_window_text": None,
            "difficulty": "easy",
        },
        {
            "id": "oolong-3",
            "question": "Count the spells?",
            "answer": "8",
            "context_window_text": "Spell casting context...",
            "difficulty": "hard",
        },
    ]

    monkeypatch.setattr(OolongEnv, "_load_dataset", lambda self: dataset_records)

    env = OolongEnv(config=config)
    tasks = env.load_tasks(limit=2)

    assert len(tasks) == 2
    first, second = tasks
    assert first.id == "oolong-1"
    assert first.context == "Long context with rolls..."
    assert first.metadata.get("difficulty") == "medium"
    assert second.id == "oolong-2"
    assert second.context is None
    assert second.metadata.get("difficulty") == "easy"


def test_oolong_environment_instructions_toggle():
    config = OolongEnv.default_config.model_copy(deep=True)
    env = OolongEnv(config=config)
    task = Task(id="t1", prompt="Count rolls", context="Long context...")
    instructions = env.get_environment_instructions(task)
    assert "Strategy for long-context information retrieval" in instructions

    config_without = OolongEnv.default_config.model_copy(deep=True)
    config_without.dataset.include_instructions = False
    env_without = OolongEnv(config=config_without)
    assert env_without.get_environment_instructions(task) == ""


def test_oolong_environment_context_write_snippets():
    """Test context chunking and snippet generation."""
    env = OolongEnv()

    # Test with short context (single chunk)
    short_context = "Short context text"
    snippets = env._generate_context_write_snippets(short_context)
    assert len(snippets) == 1
    assert "with open('context.txt', 'w')" in snippets[0]
    assert "Short context text" in snippets[0]

    # Test with long context (multiple chunks)
    long_context = "A" * 25000  # 25KB
    snippets = env._generate_context_write_snippets(long_context)
    assert len(snippets) == 3  # Should be split into 3 chunks

    # First chunk should use 'w' mode
    assert "with open('context.txt', 'w')" in snippets[0]
    # Subsequent chunks should use 'a' mode
    assert "with open('context.txt', 'a')" in snippets[1]
    assert "with open('context.txt', 'a')" in snippets[2]

    # Check escaping works
    context_with_special_chars = "Text with 'single quotes' and\nnewlines and\rcarriage returns"
    snippets = env._generate_context_write_snippets(context_with_special_chars)
    assert "\\'" in snippets[0]  # Single quotes escaped
    assert "\\n" in snippets[0]  # Newlines escaped
    assert "\\r" in snippets[0]  # Carriage returns escaped


def test_oolong_environment_format_initial_message():
    """Test message formatting with file descriptor."""
    env = OolongEnv()
    task = Task(id="t1", prompt="Count rolls", context="Long context text...")

    message = env.format_initial_message(task)

    # Should include the original prompt
    assert "Count rolls" in message

    # Should include file descriptor with context length
    assert "<file name=\"context.txt\"" in message
    assert "chars=" in message
    assert "[Content saved to workspace - use read_file('context.txt') to load]" in message

    # Should handle None context
    task_no_context = Task(id="t2", prompt="Simple task", context=None)
    message_no_context = env.format_initial_message(task_no_context)
    assert "Simple task" in message_no_context
    assert "<file" not in message_no_context


@pytest.mark.asyncio
async def test_oolong_environment_upload_context_to_workspace():
    """Test context upload functionality."""
    env = OolongEnv()

    class MockRepl:
        def __init__(self):
            self.executed_snippets = []

        async def execute(self, code: str):
            self.executed_snippets.append(code)
            execution = type('Execution', (), {
                'error': None,
                'output': 'Success'
            })()
            return execution

    # Test with context
    mock_repl = MockRepl()
    test_context = "Test context for upload"

    await env._upload_context_to_workspace(mock_repl, test_context)

    # Should have executed snippets
    assert len(mock_repl.executed_snippets) > 0
    assert any("context.txt" in snippet for snippet in mock_repl.executed_snippets)

    # Test with None context
    mock_repl_empty = MockRepl()
    await env._upload_context_to_workspace(mock_repl_empty, None)

    # Should not execute anything
    assert len(mock_repl_empty.executed_snippets) == 0

    # Test with empty context
    mock_repl_empty_str = MockRepl()
    await env._upload_context_to_workspace(mock_repl_empty_str, "")

    # Should not execute anything
    assert len(mock_repl_empty_str.executed_snippets) == 0
