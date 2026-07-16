"""Compatibility tests for the Pydantic AI-backed legacy service name."""

import pytest

from gymkhana.core.services.inference.parallel_inference import ParallelInferenceService


@pytest.mark.asyncio
async def test_generate_with_reasoning_delegates_to_v2_service(monkeypatch) -> None:
    async def generate(self, **kwargs):
        return "hello"

    monkeypatch.setattr(ParallelInferenceService, "generate", generate)
    service = ParallelInferenceService(llm_client="anthropic")
    content, reasoning = await service.generate_with_reasoning(
        messages=[{"role": "user", "content": "hi"}]
    )

    assert content == "hello"
    assert reasoning is None


@pytest.mark.asyncio
async def test_empty_parallel_batch() -> None:
    service = ParallelInferenceService()
    assert await service.batch_generate(prompts=[]) == []
