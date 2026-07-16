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
