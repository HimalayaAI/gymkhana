"""Pydantic AI v2 inference routing for Gymkhana rollouts."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from .base import InferenceService, StructuredOutputT


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
        from pydantic_ai import Agent
        from pydantic_ai.settings import ModelSettings

        agent = Agent(
            model or self.default_model,
            instructions=system_prompt,
            output_type=str,
            tools=kwargs.get("tools") or (),
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
        return result.output

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
            model or self.default_model,
            instructions=system_prompt,
            output_type=output_type,
            tools=kwargs.get("tools") or (),
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
