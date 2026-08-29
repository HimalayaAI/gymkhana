"""Pydantic AI v2 inference routing for Gymkhana rollouts."""

from __future__ import annotations

import asyncio
import json
import os
import re

import httpx
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from .base import InferenceService, StructuredOutputT


def normalize_chat_completion(data: Any) -> bool:
    """Coerce off-spec OpenAI-compatible responses in place. Returns True if changed.

    Seen from self-hosted endpoints (e.g. Tarka): ``choices[].index`` missing,
    ``created`` and ``usage`` counts as strings. Strict clients reject these.
    """
    if not isinstance(data, dict):
        return False
    changed = False
    for position, choice in enumerate(data.get("choices") or []):
        if isinstance(choice, dict) and not isinstance(choice.get("index"), int):
            choice["index"] = position
            changed = True
    created = data.get("created")
    if isinstance(created, str) and created.isdigit():
        data["created"] = int(created)
        changed = True
    usage = data.get("usage")
    if isinstance(usage, dict):
        for key, value in list(usage.items()):
            if isinstance(value, str) and value.isdigit():
                usage[key] = int(value)
                changed = True
    return changed


class _LenientOpenAITransport(httpx.AsyncBaseTransport):
    """httpx transport that repairs off-spec chat-completion JSON bodies."""

    def __init__(self, inner: Optional[httpx.AsyncBaseTransport] = None) -> None:
        self._inner = inner or httpx.AsyncHTTPTransport()

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        response = await self._inner.handle_async_request(request)
        content_type = response.headers.get("content-type", "")
        if "chat/completions" not in str(request.url) or "application/json" not in content_type:
            return response
        # ``aread`` returns the *decoded* body, so the rebuilt response must not
        # carry the original transfer headers (content-encoding / content-length).
        body = await response.aread()
        headers = {
            k: v
            for k, v in response.headers.items()
            if k.lower() not in {"content-length", "content-encoding", "transfer-encoding"}
        }
        try:
            data = json.loads(body)
        except ValueError:
            data = None
        if data is not None and normalize_chat_completion(data):
            body = json.dumps(data).encode("utf-8")
        return httpx.Response(response.status_code, headers=headers, content=body, request=request)

    async def aclose(self) -> None:
        await self._inner.aclose()


def lenient_openai_http_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=_LenientOpenAITransport(), timeout=httpx.Timeout(600.0))


THINK_RE = re.compile(r"<think>\s*(.*?)\s*</think>", re.DOTALL | re.IGNORECASE)


def split_think_tags(text: str) -> tuple[str, Optional[str]]:
    """Separate inline ``<think>…</think>`` blocks from the visible answer."""
    if not text or "<think" not in text.lower():
        return text, None
    blocks = [block for block in THINK_RE.findall(text) if block.strip()]
    visible = THINK_RE.sub("", text).strip()
    return visible, ("\n\n".join(blocks) or None)


def _thinking_text(messages: Any) -> Optional[str]:
    """Collect provider-native reasoning (Pydantic AI ``ThinkingPart``) from a run."""
    from pydantic_ai.messages import ModelResponse, ThinkingPart

    chunks = [
        part.content
        for message in messages
        if isinstance(message, ModelResponse)
        for part in message.parts
        if isinstance(part, ThinkingPart) and getattr(part, "content", None)
    ]
    return "\n\n".join(chunk.strip() for chunk in chunks if chunk.strip()) or None


LITELLM_PREFIX = "litellm:"


def _litellm_api_base() -> Optional[str]:
    """Base URL for the LiteLLM / OpenAI-compatible endpoint.

    ``LITELLM_PROXY_API_BASE`` wins; otherwise ``LITELLM_ENDPOINT`` (which the
    repo's ``.env`` stores as a full ``.../v1/chat/completions`` URL) is trimmed
    to its ``/v1`` base.
    """
    explicit = os.getenv("LITELLM_PROXY_API_BASE")
    if explicit:
        return explicit
    endpoint = os.getenv("LITELLM_ENDPOINT")
    if not endpoint:
        return None
    return re.sub(r"/chat/completions/?$", "", endpoint.strip())


def resolve_model(model: Any) -> Any:
    """Turn ``litellm:<name>`` into a model bound to the configured endpoint.

    Pydantic AI's ``litellm:`` provider reads no environment variables, so
    without this every ``litellm:`` model silently hits api.openai.com. Any other
    value (other prefixes, or Model objects such as ``TestModel``) passes through.
    """
    if not isinstance(model, str) or not model.startswith(LITELLM_PREFIX):
        return model
    api_base = _litellm_api_base()
    api_key = os.getenv("LITELLM_PROXY_API_KEY") or os.getenv("LITELLM_API_KEY")
    if not api_base or not api_key:
        raise ValueError(
            f"{model!r} needs LITELLM_ENDPOINT (or LITELLM_PROXY_API_BASE) and "
            "LITELLM_API_KEY (or LITELLM_PROXY_API_KEY) in the environment"
        )
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.litellm import LiteLLMProvider

    return OpenAIChatModel(
        model[len(LITELLM_PREFIX):],
        provider=LiteLLMProvider(
            api_key=api_key, api_base=api_base, http_client=lenient_openai_http_client()
        ),
    )


def _tool_definitions(tools: Any) -> list[Any]:
    """Convert OpenAI-style function tool dicts into Pydantic AI ``ToolDefinition``s.

    Accepts ``{"type": "function", "function": {name, description, parameters}}``
    and bare ``{name, description, parameters}`` shapes. Unknown entries are skipped.
    """
    from pydantic_ai.tools import ToolDefinition

    definitions: list[Any] = []
    for tool in tools or ():
        if not isinstance(tool, dict):
            continue
        spec = tool.get("function") if tool.get("type") == "function" else tool
        if not isinstance(spec, dict) or not spec.get("name"):
            continue
        parameters = spec.get("parameters") or {"type": "object", "properties": {}}
        definitions.append(
            ToolDefinition(
                name=str(spec["name"]),
                description=spec.get("description") or None,
                parameters_json_schema=parameters,
            )
        )
    return definitions


def _agent_tool_kwargs(tools: Any) -> Dict[str, Any]:
    """Agent kwargs for tools supplied as schema dicts (external, never executed).

    The model's tool calls come back as a ``DeferredToolRequests`` output, which
    ``generate`` serializes as ``[{"name", "arguments", "tool_call_id"}]`` JSON so
    tool-use environments can verify them against ground truth.
    """
    definitions = _tool_definitions(tools)
    if not definitions:
        return {"output_type": str}
    from pydantic_ai import DeferredToolRequests
    from pydantic_ai.toolsets import ExternalToolset

    return {
        "output_type": [str, DeferredToolRequests],
        "toolsets": [ExternalToolset(definitions)],
    }


def _serialize_output(output: Any) -> str:
    from pydantic_ai import DeferredToolRequests

    if isinstance(output, DeferredToolRequests):
        return json.dumps(
            [
                {
                    "name": call.tool_name,
                    "arguments": call.args_as_dict(),
                    "tool_call_id": call.tool_call_id,
                }
                for call in output.calls
            ]
        )
    return output if isinstance(output, str) else str(output)


class PydanticAIInferenceService(InferenceService):
    """Provider-neutral inference service backed by Pydantic AI v2.

    Model names use Pydantic AI's ``provider:model`` syntax. The model object is
    resolved lazily, so importing Gymkhana never requires provider credentials.
    """

    default_model: str = "openai:gpt-4.1-mini"
    default_temperature: Optional[float] = None
    default_max_tokens: int = 4096
    max_concurrency: int = Field(default=8, ge=1)

    @staticmethod
    def _conversation(messages: List[Dict[str, str]]) -> tuple[str, list[Any]]:
        """Convert chat messages without collapsing provider-native roles."""
        from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart

        if not messages:
            return "", []
        prompt_index = max(
            (index for index, message in enumerate(messages) if message.get("role") == "user"),
            default=len(messages) - 1,
        )
        prompt = messages[prompt_index].get("content", "")
        history: list[Any] = []
        for message in messages[:prompt_index]:
            content = message.get("content", "")
            if message.get("role") == "assistant":
                history.append(ModelResponse(parts=[TextPart(content)]))
            else:
                history.append(ModelRequest(parts=[UserPromptPart(content)]))
        return prompt, history

    async def _run(
        self,
        *,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> tuple[str, Optional[str]]:
        from pydantic_ai import Agent
        from pydantic_ai.settings import ModelSettings

        agent = Agent(
            resolve_model(model or self.default_model),
            instructions=system_prompt,
            **_agent_tool_kwargs(kwargs.get("tools")),
        )
        settings_values: Dict[str, Any] = {
            "max_tokens": max_tokens if max_tokens is not None else self.default_max_tokens,
        }
        resolved_temperature = (
            temperature if temperature is not None else self.default_temperature
        )
        if resolved_temperature is not None:
            settings_values["temperature"] = resolved_temperature
        if kwargs.get("seed") is not None:
            settings_values["seed"] = kwargs["seed"]
        settings = ModelSettings(**settings_values)
        prompt, history = self._conversation(messages)
        result = await agent.run(
            prompt,
            message_history=history or None,
            model_settings=settings,
        )
        content, inline_reasoning = split_think_tags(_serialize_output(result.output))
        reasoning = _thinking_text(result.new_messages()) or inline_reasoning
        return content, reasoning

    async def generate(
        self,
        *,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> str:
        content, _ = await self._run(
            messages=messages,
            system_prompt=system_prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
        return content

    async def generate_with_reasoning(
        self,
        *,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> tuple[str, Optional[str]]:
        """Return ``(content, reasoning)``; reasoning is provider-native thinking
        (``reasoning`` / ``reasoning_content`` fields) or inline ``<think>`` blocks."""
        return await self._run(
            messages=messages,
            system_prompt=system_prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )


    async def generate_structured(
        self,
        *,
        messages: List[Dict[str, str]],
        output_type: type[StructuredOutputT],
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> StructuredOutputT:
        """Generate a Pydantic-validated response using Pydantic AI."""

        from pydantic_ai import Agent
        from pydantic_ai.settings import ModelSettings

        agent = Agent(
            resolve_model(model or self.default_model),
            instructions=system_prompt,
            output_type=output_type,
        )
        settings_values: Dict[str, Any] = {
            "max_tokens": max_tokens if max_tokens is not None else self.default_max_tokens,
        }
        resolved_temperature = (
            temperature if temperature is not None else self.default_temperature
        )
        if resolved_temperature is not None:
            settings_values["temperature"] = resolved_temperature
        if kwargs.get("seed") is not None:
            settings_values["seed"] = kwargs["seed"]
        prompt, history = self._conversation(messages)
        result = await agent.run(
            prompt,
            message_history=history or None,
            model_settings=ModelSettings(**settings_values),
        )
        output = result.output
        if isinstance(output, BaseModel):
            return output_type.model_validate(output.model_dump())
        return output_type.model_validate(output)

    async def batch_generate(
        self,
        *,
        prompts: List[str],
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> List[str]:
        semaphore = asyncio.Semaphore(self.max_concurrency)

        async def run(index: int, prompt: str) -> str:
            async with semaphore:
                seed = kwargs.get("seed")
                return await self.generate(
                    messages=[{"role": "user", "content": prompt}],
                    system_prompt=system_prompt,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    seed=None if seed is None else seed + index,
                )

        results = await asyncio.gather(
            *(run(i, prompt) for i, prompt in enumerate(prompts)),
            return_exceptions=True,
        )
        outputs: List[str] = []
        for index, result in enumerate(results):
            if isinstance(result, BaseException):
                logging.getLogger(__name__).warning(
                    "Inference candidate %s failed: %s", index, type(result).__name__
                )
                outputs.append("")
            else:
                outputs.append(result)
        return outputs
