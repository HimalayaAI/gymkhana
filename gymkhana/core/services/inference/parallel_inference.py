"""Compatibility adapter for DeepGym's former parallel inference service."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from gymkhana.core.services.inference.pydantic_ai import PydanticAIInferenceService


class ParallelInferenceService(PydanticAIInferenceService):
    """Deprecated name backed entirely by Pydantic AI v2.

    ``llm_client`` is accepted while old environment configurations are being
    migrated; routing is determined by the provider-qualified model identifier.
    """

    llm_client: Any = None

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
        output = await self.generate(
            messages=messages,
            system_prompt=system_prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
        return output, None


__all__ = ["ParallelInferenceService"]
