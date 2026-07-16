"""Focused tests for provider-neutral Pydantic AI tool integration."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from gymkhana.envs.config import ToolUseModeSettings
from gymkhana.envs.environment import Task
from gymkhana.envs.modes.tool_use import ToolUseMode
from gymkhana.envs.tool_bridge import EnvironmentToolkit


async def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


@pytest.mark.asyncio
async def test_toolkit_exposes_native_schema_and_executes() -> None:
    toolkit = EnvironmentToolkit([add])

    assert toolkit.tool_names == ["add"]
    native = toolkit.pydantic_tools[0]
    schema = native.function_schema.json_schema
    assert schema["required"] == ["a", "b"]
    assert schema["properties"]["a"]["type"] == "integer"
    assert await toolkit.execute_tool_call("add", {"a": 20, "b": 22}) == "42"


@pytest.mark.asyncio
async def test_tool_traces_are_isolated_and_capture_native_loop() -> None:
    toolkit = EnvironmentToolkit([add])
    token = toolkit.start_trace()
    await toolkit.execute("add", {"a": 2, "b": 5})
    calls = toolkit.finish_trace(token)

    assert calls == [
        {"name": "add", "arguments": {"a": 2, "b": 5}, "result": 7}
    ]


@pytest.mark.asyncio
async def test_tool_use_mode_delegates_tool_loop_to_inference_service() -> None:
    toolkit = EnvironmentToolkit([add])

    class FakeEnvironment:
        name = "fake"
        config = SimpleNamespace(
            get_mode_config=lambda: ToolUseModeSettings(max_turns=2),
            get_llm_config=lambda: SimpleNamespace(model="test:model"),
        )

        def get_tool_executor(self, task: Task) -> EnvironmentToolkit:
            return toolkit

        def build_system_prompt(self, task: Task) -> str:
            return "Use tools."

        def format_initial_message(self, task: Task) -> str:
            return task.prompt

        async def generate_response(self, **kwargs: Any) -> tuple[str, None]:
            # Simulate Pydantic AI's internal model -> tool -> model round trip.
            assert kwargs["tools"]
            await toolkit.execute("add", {"a": 20, "b": 22})
            return "The answer is 42.", None

    result = await ToolUseMode().execute_single(
        Task(id="task-1", prompt="What is 20 + 22?"), FakeEnvironment()  # type: ignore[arg-type]
    )

    assert result.success
    assert result.final_answer == "The answer is 42."
    assert result.num_code_blocks == 1
    assert [turn.role for turn in result.turns] == [
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
    assert result.turns[1].tool_calls == [
        {"id": "tool-0", "name": "add", "arguments": {"a": 20, "b": 22}}
    ]
    assert result.turns[2].content == "42"
