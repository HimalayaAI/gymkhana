from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gymkhana.core.services.inference.openai_decisions import OpenAIDecisionService


@pytest.mark.asyncio
async def test_decide_wraps_text_and_returns_sdk_result() -> None:
    expected = SimpleNamespace(answers=[])
    create = AsyncMock(return_value=expected)
    service = OpenAIDecisionService(
        client=SimpleNamespace(decisions=SimpleNamespace(create=create))
    )
    questions = [
        {
            "type": "predicate",
            "name": "answers_query",
            "instructions": "Does the passage answer the query?",
        }
    ]

    result = await service.decide(input="query: ...\npassage: ...", questions=questions)

    assert result is expected
    create.assert_awaited_once_with(
        model="gpt-6-luna",
        input=[
            {
                "role": "user",
                "content": [{"type": "input_text", "text": "query: ...\npassage: ..."}],
            }
        ],
        questions=questions,
    )


@pytest.mark.asyncio
async def test_decide_passes_structured_input_and_model_override() -> None:
    create = AsyncMock(return_value="result")
    service = OpenAIDecisionService(
        model="default", client=SimpleNamespace(decisions=SimpleNamespace(create=create))
    )
    input_messages = [{"role": "user", "content": [{"type": "input_text", "text": "x"}]}]

    await service.decide(input=input_messages, questions=[{"type": "predicate", "name": "ok"}], model="override")

    create.assert_awaited_once_with(
        model="override", input=input_messages, questions=[{"type": "predicate", "name": "ok"}]
    )


@pytest.mark.asyncio
async def test_decide_rejects_empty_questions() -> None:
    service = OpenAIDecisionService(client=SimpleNamespace())

    with pytest.raises(ValueError, match="at least one"):
        await service.decide(input="text", questions=[])
