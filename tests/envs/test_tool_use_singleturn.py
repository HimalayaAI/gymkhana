"""Unit tests for ToolUseSingleTurnEnv."""

from __future__ import annotations

import json
from typing import Any, Dict, List

import pytest
from gymkhana.envs.tool_use_singleturn.tool_use_singleturn import (
    ToolUseSingleTurnEnv,
    convert_xlam_tools_to_openai,
    extract_tool_calls_from_native_response,
    HermesToolCallParser,
    tool_calls_match,
)
from gymkhana.envs import Task

@pytest.fixture
def tool_env() -> ToolUseSingleTurnEnv:
    # Trigger default_config initialization if needed
    if ToolUseSingleTurnEnv.default_config is None:
        ToolUseSingleTurnEnv(config=None)

    config = ToolUseSingleTurnEnv.default_config.model_copy(deep=True)
    config.dataset.limit = 1
    return ToolUseSingleTurnEnv(config=config)

def test_convert_xlam_tools_to_openai():
    xlam_tools = [
        {
            "name": "get_weather",
            "description": "Get weather info",
            "parameters": {
                "city": {"description": "The city name", "type": "str", "default": "London"},
                "units": {"description": "Celsius or Fahrenheit", "type": "str"}
            }
        }
    ]

    openai_tools = convert_xlam_tools_to_openai(xlam_tools)

    assert len(openai_tools) == 1
    tool = openai_tools[0]
    assert tool["type"] == "function"
    assert tool["function"]["name"] == "get_weather"
    assert tool["function"]["description"] == "Get weather info"

    params = tool["function"]["parameters"]
    assert params["type"] == "object"
    assert "city" in params["properties"]
    assert params["properties"]["city"]["type"] == "string"
    assert "units" in params["properties"]
    assert params["properties"]["units"]["type"] == "string"

    # units should be required because it has no default
    assert "units" in params["required"]
    # city should NOT be required because it has a default
    assert "city" not in params["required"]

def test_extract_tool_calls_from_native_response():
    # Test single call
    response_single = json.dumps([{"name": "test_fn", "arguments": {"a": 1}, "tool_call_id": "call_1"}])
    calls = extract_tool_calls_from_native_response(response_single)
    assert len(calls) == 1
    assert calls[0]["name"] == "test_fn"
    assert calls[0]["arguments"] == {"a": 1}

    # Test multiple calls
    response_multi = json.dumps([
        {"name": "fn1", "arguments": {"x": 10}},
        {"name": "fn2", "arguments": {"y": 20}}
    ])
    calls = extract_tool_calls_from_native_response(response_multi)
    assert len(calls) == 2
    assert calls[1]["name"] == "fn2"

    # Test invalid JSON
    assert extract_tool_calls_from_native_response("not json") == []

def test_hermes_parser():
    response = (
        "Thinking... <think>I should call a tool</think>\n"
        "<tool_call>{\"name\": \"get_user\", \"arguments\": {\"id\": 123}}</tool_call>"
    )
    calls = HermesToolCallParser.parse(response)
    assert len(calls) == 1
    assert calls[0]["name"] == "get_user"
    assert calls[0]["arguments"] == {"id": 123}

    # Test array inside hermes tags
    response_array = "<tool_call>[{\"name\": \"call1\", \"arguments\": {}}, {\"name\": \"call2\", \"arguments\": {}}]</tool_call>"
    calls = HermesToolCallParser.parse(response_array)
    assert len(calls) == 2
    assert calls[1]["name"] == "call2"

def test_tool_calls_match():
    expected = [{"name": "fn1", "arguments": {"a": 1}}]

    # Exact match
    assert tool_calls_match([{"name": "fn1", "arguments": {"a": 1}}], expected)

    # Order independent
    expected_multi = [
        {"name": "a", "arguments": {}},
        {"name": "b", "arguments": {}}
    ]
    assert tool_calls_match([{"name": "b", "arguments": {}}, {"name": "a", "arguments": {}}], expected_multi)

    # Mismatch name
    assert not tool_calls_match([{"name": "fn2", "arguments": {"a": 1}}], expected)

    # Mismatch args
    assert not tool_calls_match([{"name": "fn1", "arguments": {"a": 2}}], expected)

    # Mismatch count
    assert not tool_calls_match([{"name": "fn1", "arguments": {"a": 1}}, {"name": "fn1", "arguments": {"a": 1}}], expected)

@pytest.mark.asyncio
async def test_load_tasks(monkeypatch: pytest.MonkeyPatch, tool_env: ToolUseSingleTurnEnv):
    records = [
        {
            "query": "What is the weather in Paris?",
            "tools": json.dumps([{"name": "get_weather", "parameters": {"city": {"type": "str"}}}]),
            "answers": json.dumps([{"name": "get_weather", "arguments": {"city": "Paris"}}])
        }
    ]

    monkeypatch.setattr(tool_env, "_load_dataset", lambda *_: records)

    tasks = tool_env.load_tasks(limit=1)
    assert len(tasks) == 1
    task = tasks[0]

    assert task.prompt == "What is the weather in Paris?"
    assert task.metadata["expected_tool_calls"][0]["name"] == "get_weather"
    assert len(task.metadata["tools_openai"]) == 1
    assert task.metadata["tools_openai"][0]["function"]["name"] == "get_weather"

@pytest.mark.asyncio
async def test_compute_reward(tool_env: ToolUseSingleTurnEnv):
    # Test with correct answer
    task = Task(id="1", prompt="p", metadata={"expected_tool_calls": [{"name": "fn", "arguments": {}}]})
    result = type("Result", (), {"final_answer": json.dumps([{"name": "fn", "arguments": {}}]), "id": "1"})

    reward = await tool_env.compute_reward(result, answer_correct=True, task=task)
    assert reward == 1.0

    reward_incorrect = await tool_env.compute_reward(result, answer_correct=False, task=task)
    assert reward_incorrect == 0.0

    # Test verification from scratch
    reward_scratch = await tool_env.compute_reward(result, answer_correct=None, task=task)
    assert reward_scratch == 1.0
