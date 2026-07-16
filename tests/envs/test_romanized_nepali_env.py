import pytest
from typing import Any, Dict, List, Optional
from pydantic import Field

from gymkhana.core.models import TrajectoryResult
from gymkhana.core.services import ServiceContainer
from gymkhana.core.services.inference import InferenceService
from gymkhana.envs import ENVIRONMENTS
from gymkhana.envs.romanized_nepali import RomanizedNepaliEnv, normalize_translation
from envs.fixtures.romanized_nepali_cases import (
    NORMALIZATION_CASES,
    TRANSLITERATION_CASES,
)


class ScriptedInference(InferenceService):
    responses: List[str]
    calls: List[Dict[str, Any]] = Field(default_factory=list)

    async def generate(
        self, *, messages: List[Dict[str, str]], system_prompt: Optional[str] = None, **kwargs: Any
    ) -> str:
        self.calls.append({"messages": messages, "system_prompt": system_prompt})
        return self.responses.pop(0)

    async def batch_generate(
        self, *, prompts: List[str], system_prompt: Optional[str] = None, **kwargs: Any
    ) -> List[str]:
        return [self.responses.pop(0) for _ in prompts]


def test_environment_is_registered_and_bidirectional() -> None:
    env = RomanizedNepaliEnv()
    tasks = env.load_tasks()

    assert ENVIRONMENTS.get("romanized-nepali") is RomanizedNepaliEnv
    assert {task.metadata["direction"] for task in tasks} == {
        "devanagari_to_romanized", "romanized_to_devanagari"
    }
    assert all(task.metadata["reference"] for task in tasks)


@pytest.mark.asyncio
async def test_normalized_exact_and_partial_rewards() -> None:
    env = RomanizedNepaliEnv(records=[{
        "id": "one", "direction": "devanagari_to_romanized",
        "source": "नेपाल", "reference": "Nepal"
    }])
    task = env.load_tasks()[0]
    exact = TrajectoryResult(success=True, final_answer="  NEPAL  ")
    partial = TrajectoryResult(success=True, final_answer="nepa")

    assert normalize_translation("  NEPAL  ") == "nepal"
    assert await env.compute_reward(exact, task=task) == 1.0
    assert 0.0 < await env.compute_reward(partial, task=task) < 1.0


@pytest.mark.parametrize("direction,source,reference", TRANSLITERATION_CASES)
def test_translator_regression_fixtures(direction: str, source: str, reference: str) -> None:
    assert RomanizedNepaliEnv().translate(direction, source) == reference


@pytest.mark.parametrize("source,expected", NORMALIZATION_CASES)
def test_normalization_fixtures(source: str, expected: str) -> None:
    assert normalize_translation(source) == expected


def test_explicit_empty_records_do_not_load_defaults() -> None:
    assert RomanizedNepaliEnv(records=[]).load_tasks() == []


@pytest.mark.asyncio
async def test_run_task_applies_reward_without_storage() -> None:
    inference = ScriptedInference(responses=["Nepal"])
    env = RomanizedNepaliEnv(
        records=[{
            "id": "one", "direction": "devanagari_to_romanized",
            "source": "नेपाल", "reference": "Nepal"
        }],
        services=ServiceContainer(inference=inference),
    )
    env.config.dataset.num_rollouts = 1
    await env.setup()
    result = await env.run_task(env.load_tasks()[0])

    assert result.final_answer == "Nepal"
    assert result.total_reward == 1.0
    assert result.answer_correct is True
    assert inference.calls[0]["system_prompt"]
    assert "Transliterate" in inference.calls[0]["messages"][0]["content"]


@pytest.mark.asyncio
async def test_grouped_run_selects_best_reward_without_storage() -> None:
    inference = ScriptedInference(responses=["wrong", "Nepa", "Nepal"])
    env = RomanizedNepaliEnv(
        records=[{
            "id": "group", "direction": "devanagari_to_romanized",
            "source": "नेपाल", "reference": "Nepal"
        }],
        services=ServiceContainer(inference=inference),
    )
    env.config.dataset.num_rollouts = 3
    await env.setup()
    best = await env.run_task(env.load_tasks()[0])

    assert best.final_answer == "Nepal"
    assert best.total_reward == 1.0
    assert len(inference.calls) == 3
