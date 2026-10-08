"""Typed decisions through OpenAI's Decisions API.

This is deliberately separate from :class:`InferenceService`: Decisions API
answers typed questions and does not generate assistant text.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union


DecisionInput = Union[str, List[Dict[str, Any]]]
DecisionQuestion = Dict[str, Any]


class OpenAIDecisionService:
    """Small async client wrapper for OpenAI's typed Decisions API."""

    def __init__(self, *, model: str = "gpt-6-luna", client: Any = None) -> None:
        self.model = model
        if client is None:
            try:
                from openai import AsyncOpenAI
            except ImportError as exc:  # pragma: no cover - package is a core dependency
                raise RuntimeError(
                    "OpenAI Decisions API support requires openai>=3.26.0"
                ) from exc
            client = AsyncOpenAI()
        self.client = client

    async def decide(
        self,
        *,
        input: DecisionInput,
        questions: List[DecisionQuestion],
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        """Ask typed questions about text and return the SDK decision result.

        String inputs are wrapped as one user text message. Structured inputs
        are passed through for callers needing multiple supported text parts.
        The raw SDK result is returned so callers can inspect predicates,
        choices, scores, probabilities, confidence, and refusals without loss.
        """
        if not questions:
            raise ValueError("questions must contain at least one decision question")
        api_input: Any = input
        if isinstance(input, str):
            api_input = [
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": input}],
                }
            ]
        return await self.client.decisions.create(
            model=model or self.model,
            input=api_input,
            questions=questions,
            **kwargs,
        )
