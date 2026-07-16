"""Pydantic AI v2 sub-LLM service for RLM helper calls."""

from __future__ import annotations

import asyncio
from typing import Any, Callable, Dict, List, Optional

from pydantic import Field, PrivateAttr

from gymkhana.core.services.inference.pydantic_ai import PydanticAIInferenceService


class SubLLMOrchestrator(PydanticAIInferenceService):
    """Bounded, timeout-aware helper inference with optional registered tools."""

    model: str = "openai:gpt-4.1-mini"
    max_parallel: int = Field(default=8, ge=1)
    timeout_seconds: int = Field(default=60, ge=1)
    llm_client: Any = None
    _tools: Dict[str, Callable[..., Any]] = PrivateAttr(default_factory=dict)

    def register_tool(
        self,
        name: str,
        func: Callable[..., Any],
        description: str,
        parameters: Dict[str, Any],
    ) -> None:
        """Register a callable; its signature/docstring define the Pydantic AI tool."""
        del description, parameters
        self._tools[name] = func

    async def generate(
        self,
        *,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        tools: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> str:
        from pydantic_ai import Agent
        from pydantic_ai.settings import ModelSettings

        selected_tools = [self._tools[name] for name in tools or [] if name in self._tools]
        agent = Agent(
            model or self.model,
            instructions=system_prompt,
            output_type=str,
            tools=selected_tools,
        )
        prompt, history = self._conversation(messages)
        result = await asyncio.wait_for(
            agent.run(
                prompt,
                message_history=history or None,
                model_settings=ModelSettings(
                    temperature=temperature if temperature is not None else self.default_temperature,
                    max_tokens=max_tokens if max_tokens is not None else self.default_max_tokens,
                ),
            ),
            timeout=self.timeout_seconds,
        )
        return result.output

    async def batch_generate(
        self,
        *,
        prompts: List[str],
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        tools: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        semaphore = asyncio.Semaphore(self.max_parallel)

        async def run(prompt: str) -> str:
            async with semaphore:
                try:
                    return await self.generate(
                        messages=[{"role": "user", "content": prompt}],
                        system_prompt=system_prompt,
                        model=model,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        tools=tools,
                        **kwargs,
                    )
                except asyncio.TimeoutError:
                    return f"[Sub-LLM timeout after {self.timeout_seconds}s]"
                except Exception as exc:
                    return f"[Sub-LLM error: {type(exc).__name__}]"

        return list(await asyncio.gather(*(run(prompt) for prompt in prompts)))


__all__ = ["SubLLMOrchestrator"]
