"""Focused tests for the sandbox server's inference boundary."""

from unittest.mock import AsyncMock, patch

import pytest

pytest.importorskip("flask")

from gymkhana.core.services.inference.pydantic_ai import PydanticAIInferenceService
from gymkhana.core.services.sandboxes.server.app import app


def test_sub_agent_uses_pydantic_ai_and_preserves_context():
    generate = AsyncMock(return_value="नेपाली उत्तर")

    with patch.object(PydanticAIInferenceService, "generate", generate):
        response = app.test_client().post(
            "/sub_agent",
            json={
                "task": "Translate this",
                "context": "namaste",
                "model": "anthropic:claude-sonnet-4-5",
                "temperature": 0.2,
                "max_tokens": 256,
            },
        )

    assert response.status_code == 200
    assert response.get_json() == {
        "response": "नेपाली उत्तर",
        "model": "anthropic:claude-sonnet-4-5",
        "client": "anthropic",
    }
    kwargs = generate.await_args.kwargs
    assert kwargs["messages"] == [
        {
            "role": "user",
            "content": '<file name="context">\nnamaste\n</file>\n\nTranslate this',
        }
    ]
    assert kwargs["temperature"] == 0.2
    assert kwargs["max_tokens"] == 256


def test_sub_agent_qualifies_legacy_model_with_provider_hint():
    generate = AsyncMock(return_value="ok")

    with patch.object(PydanticAIInferenceService, "generate", generate):
        response = app.test_client().post(
            "/sub_agent",
            json={"task": "Answer", "model": "gpt-4o-mini", "client": "openai"},
        )

    assert response.status_code == 200
    assert response.get_json()["model"] == "openai:gpt-4o-mini"
    assert generate.await_args.kwargs["model"] == "openai:gpt-4o-mini"


def test_sub_agent_requires_task():
    response = app.test_client().post("/sub_agent", json={})

    assert response.status_code == 400
    assert response.get_json() == {"error": "No task provided"}
