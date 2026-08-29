from typing import Any, Dict, List, Optional

import pytest
from packaging.version import Version

from gymkhana.core.services.inference.base import InferenceService
from gymkhana.core.services.inference.rollouts import RolloutRequest, generate_rollout_group
from gymkhana.core.services.inference.pydantic_ai import PydanticAIInferenceService


class FakeInferenceService(InferenceService):
    async def generate(self, *, messages: List[Dict[str, str]], **kwargs: Any) -> str:
        return messages[-1]["content"]

    async def batch_generate(
        self, *, prompts: List[str], system_prompt: Optional[str] = None, **kwargs: Any
    ) -> List[str]:
        return [f"{prompt}:{index}" for index, prompt in enumerate(prompts)]


@pytest.mark.asyncio
async def test_rollout_group_is_ordered_and_identified() -> None:
    group = await generate_rollout_group(
        FakeInferenceService(), RolloutRequest(task_id="t-1", prompt="answer", group_size=4)
    )

    assert [candidate.index for candidate in group.candidates] == [0, 1, 2, 3]
    assert [candidate.output for candidate in group.candidates] == [
        "answer:0", "answer:1", "answer:2", "answer:3"
    ]
    assert {candidate.group_id for candidate in group.candidates} == {group.group_id}


def test_runtime_uses_pydantic_ai_v2() -> None:
    import pydantic_ai

    assert Version(pydantic_ai.__version__).major == 2


def test_conversation_preserves_native_roles() -> None:
    prompt, history = PydanticAIInferenceService._conversation([
        {"role": "user", "content": "remember yak"},
        {"role": "assistant", "content": "remembered"},
        {"role": "user", "content": "what did I say?"},
    ])

    assert prompt == "what did I say?"
    assert [message.kind for message in history] == ["request", "response"]


@pytest.mark.asyncio
async def test_batch_failure_isolated_and_order_preserved(monkeypatch) -> None:
    service = PydanticAIInferenceService(max_concurrency=2)

    async def generate(self, *, messages, **kwargs):
        prompt = messages[-1]["content"]
        if prompt == "bad":
            raise RuntimeError("provider unavailable")
        return prompt.upper()

    monkeypatch.setattr(PydanticAIInferenceService, "generate", generate)
    outputs = await service.batch_generate(prompts=["one", "bad", "three"])

    assert outputs == ["ONE", "", "THREE"]


@pytest.mark.asyncio
async def test_schema_tools_are_returned_as_deferred_calls_json() -> None:
    """OpenAI-style tool dicts become external tools; the model's calls come back as JSON."""
    import json

    from pydantic_ai.models.test import TestModel

    from gymkhana.core.services.inference.pydantic_ai import _tool_definitions

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Weather for a city",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }
    ]
    definitions = _tool_definitions(tools)
    assert [d.name for d in definitions] == ["get_weather"]
    assert definitions[0].parameters_json_schema["required"] == ["city"]

    service = PydanticAIInferenceService()
    raw = await service.generate(
        messages=[{"role": "user", "content": "weather in Paris?"}],
        system_prompt="call tools",
        model=TestModel(),
        tools=tools,
    )
    calls = json.loads(raw)
    assert calls and calls[0]["name"] == "get_weather"
    assert isinstance(calls[0]["arguments"], dict) and "city" in calls[0]["arguments"]
    assert calls[0]["tool_call_id"]

    plain = await service.generate(messages=[{"role": "user", "content": "hi"}], model=TestModel())
    assert isinstance(plain, str)


def test_litellm_prefix_binds_to_configured_endpoint(monkeypatch) -> None:
    from gymkhana.core.services.inference.pydantic_ai import resolve_model

    monkeypatch.delenv("LITELLM_PROXY_API_BASE", raising=False)
    monkeypatch.delenv("LITELLM_PROXY_API_KEY", raising=False)
    monkeypatch.setenv("LITELLM_ENDPOINT", "https://tarka.rest/v1/chat/completions")
    monkeypatch.setenv("LITELLM_API_KEY", "test-key")

    model = resolve_model("litellm:himalaya-gemma-4-bf16")
    assert type(model).__name__ == "OpenAIChatModel"
    assert model.model_name == "himalaya-gemma-4-bf16"
    assert str(model._provider.base_url).rstrip("/") == "https://tarka.rest/v1"

    assert resolve_model("openai:gpt-4.1-mini") == "openai:gpt-4.1-mini"

    monkeypatch.delenv("LITELLM_ENDPOINT")
    with pytest.raises(ValueError, match="LITELLM_ENDPOINT"):
        resolve_model("litellm:anything")
