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


@pytest.mark.asyncio
async def test_lenient_transport_repairs_offspec_chat_completion() -> None:
    import json

    import httpx

    from gymkhana.core.services.inference.pydantic_ai import (
        _LenientOpenAITransport,
        normalize_chat_completion,
    )

    offspec = {
        "id": "x",
        "object": "chat.completion",
        "created": "1788013978",
        "model": "himalaya-gemma-4-bf16",
        "choices": [{"message": {"role": "assistant", "content": "नमस्ते"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": "17", "completion_tokens": "5", "total_tokens": "22"},
    }
    assert normalize_chat_completion(json.loads(json.dumps(offspec))) is True
    assert normalize_chat_completion({"choices": [{"index": 0}], "created": 1, "usage": {"a": 1}}) is False

    def fake(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=offspec, headers={"content-length": "999"})

    transport = _LenientOpenAITransport(httpx.MockTransport(fake))
    async with httpx.AsyncClient(transport=transport) as client:
        response = await client.post("https://tarka.rest/v1/chat/completions", json={})
    data = response.json()
    assert data["choices"][0]["index"] == 0
    assert data["created"] == 1788013978
    assert data["usage"]["total_tokens"] == 22
    assert data["choices"][0]["message"]["content"] == "नमस्ते"

    # Now the strict OpenAI client type accepts it.
    from openai.types.chat import ChatCompletion

    ChatCompletion.model_validate(data)


def test_reasoning_helpers_split_inline_and_native_thinking() -> None:
    from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, ThinkingPart, UserPromptPart

    from gymkhana.core.services.inference.pydantic_ai import _thinking_text, split_think_tags

    assert split_think_tags("plain answer") == ("plain answer", None)
    assert split_think_tags("<think>\nstep 1\n</think>\nfinal") == ("final", "step 1")
    assert split_think_tags("<THINK>a</THINK> x <think>b</think>") == ("x", "a\n\nb")

    messages = [
        ModelRequest(parts=[UserPromptPart("q")]),
        ModelResponse(parts=[ThinkingPart(content=" native reasoning "), TextPart("answer")]),
    ]
    assert _thinking_text(messages) == "native reasoning"
    assert _thinking_text([ModelResponse(parts=[TextPart("answer")])]) is None


@pytest.mark.asyncio
async def test_generate_with_reasoning_returns_native_thinking(monkeypatch) -> None:
    from pydantic_ai.models.function import FunctionModel
    from pydantic_ai.messages import ModelResponse, TextPart, ThinkingPart

    def respond(messages, info):
        return ModelResponse(parts=[ThinkingPart(content="I should greet."), TextPart("नमस्ते")])

    service = PydanticAIInferenceService()
    content, reasoning = await service.generate_with_reasoning(
        messages=[{"role": "user", "content": "hi"}], model=FunctionModel(respond)
    )
    assert content == "नमस्ते"
    assert reasoning == "I should greet."


@pytest.mark.asyncio
async def test_lenient_transport_handles_gzip_encoded_bodies() -> None:
    import gzip
    import json

    import httpx

    from gymkhana.core.services.inference.pydantic_ai import _LenientOpenAITransport

    payload = {"id": "x", "object": "chat.completion", "created": 1, "model": "m",
               "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
               "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}

    def fake(request: httpx.Request) -> httpx.Response:
        raw = gzip.compress(json.dumps(payload).encode())
        return httpx.Response(200, content=raw, headers={"content-encoding": "gzip", "content-type": "application/json", "content-length": str(len(raw))})

    async with httpx.AsyncClient(transport=_LenientOpenAITransport(httpx.MockTransport(fake))) as client:
        response = await client.post("https://example.test/v1/chat/completions", json={})
    assert response.json()["choices"][0]["message"]["content"] == "ok"


@pytest.mark.asyncio
async def test_transient_errors_are_retried_with_server_delay(monkeypatch) -> None:
    from pydantic_ai.exceptions import ModelHTTPError
    from pydantic_ai.messages import ModelResponse, TextPart
    from pydantic_ai.models.function import FunctionModel

    import gymkhana.core.services.inference.pydantic_ai as mod

    sleeps: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr(mod.asyncio, "sleep", fake_sleep)

    calls = {"n": 0}

    def flaky(messages, info):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ModelHTTPError(429, "m", body={"error": {"message": "Please retry in 22.2s"}}, headers={})
        if calls["n"] == 2:
            raise ModelHTTPError(503, "m", body="overloaded", headers={"Retry-After": "5"})
        return ModelResponse(parts=[TextPart("ok")])

    service = PydanticAIInferenceService(retry_base_seconds=1.0, retry_max_seconds=60.0)
    out = await service.generate(messages=[{"role": "user", "content": "hi"}], model=FunctionModel(flaky))
    assert out == "ok"
    assert calls["n"] == 3
    assert sleeps == [22.2, 5.0]  # server-suggested delays win over backoff


@pytest.mark.asyncio
async def test_non_retryable_errors_propagate_immediately(monkeypatch) -> None:
    from pydantic_ai.exceptions import ModelHTTPError
    from pydantic_ai.models.function import FunctionModel

    import gymkhana.core.services.inference.pydantic_ai as mod

    monkeypatch.setattr(mod.asyncio, "sleep", lambda s: (_ for _ in ()).throw(AssertionError("should not sleep")))
    calls = {"n": 0}

    def bad_request(messages, info):
        calls["n"] += 1
        raise ModelHTTPError(400, "m", body="bad request")

    service = PydanticAIInferenceService()
    with pytest.raises(ModelHTTPError):
        await service.generate(messages=[{"role": "user", "content": "hi"}], model=FunctionModel(bad_request))
    assert calls["n"] == 1

    # exhausted retries re-raise the last error
    always = {"n": 0}

    def always_503(messages, info):
        always["n"] += 1
        raise ModelHTTPError(503, "m", body="down")

    async def no_sleep(_: float) -> None:
        return None

    monkeypatch.setattr(mod.asyncio, "sleep", no_sleep)
    service = PydanticAIInferenceService(max_retries=2)
    with pytest.raises(ModelHTTPError):
        await service.generate(messages=[{"role": "user", "content": "hi"}], model=FunctionModel(always_503))
    assert always["n"] == 3
