"""Base abstractions for inference services."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, TypeVar

from pydantic import BaseModel, ConfigDict


StructuredOutputT = TypeVar("StructuredOutputT", bound=BaseModel)


class InferenceService(BaseModel, ABC):
    """Abstract base class for LLM inference services."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    data_inserter: Optional[Any] = None

    @abstractmethod
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
        """Generate a response from an LLM.

        Args:
            messages: List of chat messages (role/content)
            system_prompt: Optional system prompt
            model: Optional model override
            temperature: Optional temperature override
            max_tokens: Optional token limit override
            **kwargs: Implementation-specific options

        Returns:
            Generated text response
        """
        ...

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
        """Generate a response with reasoning content if available.

        Default implementation calls generate() and returns (response, None).
        Subclasses should override to return reasoning content when available.

        Args:
            messages: List of chat messages (role/content)
            system_prompt: Optional system prompt
            model: Optional model override
            temperature: Optional temperature override
            max_tokens: Optional token limit override
            **kwargs: Implementation-specific options

        Returns:
            Tuple of (content, reasoning_content) where reasoning_content is None if not available
        """
        response = await self.generate(
            messages=messages,
            system_prompt=system_prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
        return response, None

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
        """Generate provider-validated structured output.

        Inference backends that support native structured output should override
        this method. Keeping this separate from :meth:`generate` prevents
        verifiers from parsing provider text or accepting unvalidated payloads.
        """

        raise NotImplementedError(
            f"{type(self).__name__} does not support structured generation"
        )

    @abstractmethod
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
        """Generate responses for multiple prompts in parallel.

        Args:
            prompts: List of user prompts
            system_prompt: Optional system prompt for all calls
            model: Optional model override
            temperature: Optional temperature override
            max_tokens: Optional token limit override

        Returns:
            List of generated responses
        """
        ...

    async def batch_generate_identical(
        self,
        *,
        prompt: str,
        system_prompt: Optional[str] = None,
        n: int = 1,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Generate N responses for the same prompt (homogeneous batch, e.g., turn 0).

        Default implementation uses batch_generate with N copies of the prompt.
        Subclasses can override for more efficient implementations (e.g., using n parameter).

        Args:
            prompt: Single prompt to replicate
            system_prompt: Optional system prompt
            n: Number of identical responses to generate
            model: Optional model override
            temperature: Optional temperature override
            max_tokens: Optional token limit override

        Returns:
            List of N generated responses
        """
        return await self.batch_generate(
            prompts=[prompt] * n,
            system_prompt=system_prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

    async def batch_generate_conversations(
        self,
        *,
        conversations: List[tuple[List[Dict[str, str]], Optional[str]]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Generate responses for multiple conversations with different histories (heterogeneous batch, e.g., turn > 0).

        Default implementation calls generate() for each conversation sequentially.
        Subclasses should override for parallel execution.

        Args:
            conversations: List of (messages, system_prompt) tuples, one per conversation
            model: Optional model override
            temperature: Optional temperature override
            max_tokens: Optional token limit override

        Returns:
            List of generated responses, one per conversation
        """
        results = []
        for messages, system_prompt in conversations:
            response = await self.generate(
                messages=messages,
                system_prompt=system_prompt,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )
            results.append(response)
        return results
