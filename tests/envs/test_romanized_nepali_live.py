"""Opt-in live Anthropic smoke test; skipped during normal test runs."""

import os

import pytest

from gymkhana.envs.config import InferenceConfig, LLMClientType
from gymkhana.envs.romanized_nepali import RomanizedNepaliEnv


pytestmark = [pytest.mark.live, pytest.mark.anthropic]


@pytest.mark.asyncio
async def test_live_anthropic_easy_transliteration() -> None:
    if os.getenv("GYMKHANA_RUN_LIVE_ANTHROPIC") != "1" or not os.getenv("ANTHROPIC_API_KEY"):
        pytest.skip("set GYMKHANA_RUN_LIVE_ANTHROPIC=1 and ANTHROPIC_API_KEY")

    config = RomanizedNepaliEnv.default_config.model_copy(deep=True)
    config.dataset.num_rollouts = 1
    config.llm = InferenceConfig(
        client=LLMClientType.ANTHROPIC,
        model=os.getenv(
            "GYMKHANA_ANTHROPIC_MODEL",
            os.getenv("ANTHROPIC_MODEL", "anthropic:claude-sonnet-4-5"),
        ),
        temperature=None,
        max_tokens=64,
    )
    env = RomanizedNepaliEnv(
        config=config,
        records=[{"id": "live-nepal", "direction": "devanagari_to_romanized", "source": "नेपाल", "reference": "nepal"}],
    )

    summary = await env.generate(limit=1)

    assert summary.successful == 1
    assert summary.results[0].final_answer.strip()
    assert 0.7 <= summary.results[0].total_reward <= 1.0
