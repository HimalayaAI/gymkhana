import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

import pytest
from pydantic import Field

from gymkhana.core.models import TrajectoryResult
from gymkhana.core.services import ServiceContainer
from gymkhana.core.services.inference import InferenceService
from gymkhana.envs import ENVIRONMENTS
from gymkhana.envs.config import EnvironmentType
from gymkhana.envs.english_sharegpt_nepali import (
    EnglishShareGPTToNepaliEnv,
    evaluate_translation,
    normalize_conversations,
    parse_translation_output,
)


def encoded(conversations: List[Dict[str, str]]) -> str:
    return json.dumps({"conversations": conversations}, ensure_ascii=False)


class ScriptedInference(InferenceService):
    responses: List[str]
    calls: List[Dict[str, Any]] = Field(default_factory=list)

    async def generate(
        self,
        *,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        self.calls.append({"messages": messages, "system_prompt": system_prompt})
        return self.responses.pop(0)

    async def batch_generate(
        self,
        *,
        prompts: List[str],
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> List[str]:
        return [self.responses.pop(0) for _ in prompts]


SOURCE = [
    {"from": "human", "value": "What is the capital of Nepal?"},
    {"from": "gpt", "value": "The capital of Nepal is Kathmandu."},
]

NEPALI = [
    {"from": "human", "value": "नेपालको राजधानी के हो?"},
    {"from": "gpt", "value": "नेपालको राजधानी काठमाडौं हो।"},
]

JUDGE_PASS = json.dumps(
    {"score": 10, "reasoning": "Faithful and complete translation."}
)
JUDGE_FAIL = json.dumps(
    {"score": 0, "reasoning": "The candidate does not preserve the source meaning."}
)


class CapturingInserter:
    def __init__(self) -> None:
        self.sharegpt_rows: List[Dict[str, Any]] = []

    async def insert_trajectory(self, **kwargs: Any) -> Any:
        return uuid4()

    async def insert_sharegpt_dataset(self, **kwargs: Any) -> bool:
        self.sharegpt_rows.append(kwargs)
        return True

    async def insert_rollout_group(self, group: Any) -> Any:
        return uuid4()

    async def insert_rollout(self, rollout: Any, rollout_group_id: Any) -> None:
        return None

    async def update_rollout(self, rollout: Any) -> None:
        return None

    async def link_rollout_to_trajectory(
        self, rollout_id: Any, trajectory_id: Any
    ) -> None:
        return None

    async def update_rollout_group_statistics(self, *args: Any) -> None:
        return None


def test_environment_is_registered_with_canonical_aliases() -> None:
    assert (
        ENVIRONMENTS.get("english-sharegpt-to-nepali")
        is EnglishShareGPTToNepaliEnv
    )
    assert (
        ENVIRONMENTS.get("english_sharegpt_to_nepali")
        is EnglishShareGPTToNepaliEnv
    )
    assert EnvironmentType.ENGLISH_SHAREGPT_TO_NEPALI.value == (
        "english-sharegpt-to-nepali"
    )


def test_prompt_explicitly_preserves_ascii_numeric_tokens() -> None:
    env = EnglishShareGPTToNepaliEnv(records=[{"conversations": SOURCE}])
    task = env.load_tasks()[0]

    instructions = env.get_environment_instructions(task)

    assert "same ASCII form" in instructions
    assert "keep `7`, `28`, and `196`" in instructions


def test_loads_sharegpt_openai_and_flattened_hermes_rows() -> None:
    records = [
        {
            "id": "sharegpt",
            "source": "OpenHermes",
            "conversations": SOURCE,
        },
        {
            "id": "openai",
            "messages": [
                {"role": "user", "content": "Say hello."},
                {"role": "assistant", "content": "Hello!"},
            ],
        },
        {
            "_id": "flattened",
            "_source": "platypus",
            "condition": "direct",
            "instruction": "Add two and three.",
            "response": "The answer is five.",
        },
    ]
    tasks = EnglishShareGPTToNepaliEnv(records=records).load_tasks()

    assert [task.id for task in tasks] == ["sharegpt", "openai", "flattened"]
    assert [task.metadata["input_format"] for task in tasks] == [
        "sharegpt",
        "openai-messages",
        "hermes-instruction-response",
    ]
    assert tasks[0].metadata["source_provenance"]["source"] == "OpenHermes"
    assert tasks[2].metadata["source_provenance"]["_source"] == "platypus"
    assert EnglishShareGPTToNepaliEnv(records=records).load_tasks(limit=0) == []
    assert len(EnglishShareGPTToNepaliEnv(records=records).load_tasks(limit=2)) == 2


def test_field_mapping_onboards_compatible_custom_schema() -> None:
    config = EnglishShareGPTToNepaliEnv.default_config.model_copy(deep=True)
    config.dataset.field_mapping = {
        "id": "uid",
        "conversations": "dialogue",
    }
    env = EnglishShareGPTToNepaliEnv(
        config=config,
        records=[
            {
                "uid": "custom-row",
                "dialogue": SOURCE,
                "license": "cc-by-4.0",
            }
        ],
    )

    task = env.load_tasks()[0]

    assert task.id == "custom-row"
    assert task.metadata["source_conversations"] == SOURCE
    assert task.metadata["source_provenance"]["license"] == "cc-by-4.0"
    assert "dialogue" not in task.metadata["source_provenance"]


def test_local_jsonl_loader_and_stable_hash_id(tmp_path: Path) -> None:
    path = tmp_path / "hermes.jsonl"
    path.write_text(
        json.dumps({"conversations": SOURCE}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    config = EnglishShareGPTToNepaliEnv.default_config.model_copy(deep=True)
    config.dataset.dataset_name = str(path)
    config.dataset.limit = None

    first = EnglishShareGPTToNepaliEnv(config=config).load_tasks()[0]
    second = EnglishShareGPTToNepaliEnv(config=config).load_tasks()[0]

    assert first.id == second.id
    assert first.id.startswith("sharegpt-")
    assert first.metadata["source_provenance"]["dataset_name"] == str(path)


def test_huggingface_rows_backend_supports_bounded_offset_loading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: List[Dict[str, Any]] = []

    class Response:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> Dict[str, Any]:
            return {
                "rows": [
                    {"row": {"id": "row-25", "conversations": SOURCE}},
                    {"row": {"id": "row-26", "conversations": SOURCE}},
                ],
                "num_rows_total": 1000,
            }

    def fake_get(url: str, **kwargs: Any) -> Response:
        calls.append({"url": url, **kwargs})
        return Response()

    monkeypatch.setattr("requests.get", fake_get)
    config = EnglishShareGPTToNepaliEnv.default_config.model_copy(deep=True)
    config.dataset.dataset_name = "teknium/OpenHermes-2.5"
    config.dataset.dataset_backend = "huggingface-rows"
    config.dataset.dataset_config = "default"
    config.dataset.dataset_offset = 25
    config.dataset.limit = 2

    tasks = EnglishShareGPTToNepaliEnv(config=config).load_tasks()

    assert [task.id for task in tasks] == ["row-25", "row-26"]
    assert [
        task.metadata["source_provenance"]["row_index"] for task in tasks
    ] == [25, 26]
    assert calls[0]["params"]["offset"] == 25
    assert calls[0]["params"]["length"] == 2


def test_normalizes_openai_text_parts_and_rejects_bad_roles() -> None:
    normalized = normalize_conversations(
        [
            {"role": "user", "content": [{"type": "text", "text": "Hello"}]},
            {"role": "assistant", "content": "Hi"},
        ]
    )
    assert normalized == [
        {"from": "human", "value": "Hello"},
        {"from": "gpt", "value": "Hi"},
    ]

    with pytest.raises(ValueError, match="unsupported role"):
        normalize_conversations(
            [
                {"from": "moderator", "value": "Hello"},
                {"from": "gpt", "value": "Hi"},
            ]
        )


@pytest.mark.parametrize(
    "output,error",
    [
        ("", "empty"),
        ("not json", "valid JSON"),
        ("```json\n{}\n```", "Markdown fences"),
        (json.dumps({"conversations": SOURCE, "note": "extra"}), "only"),
    ],
)
def test_strict_output_parser_rejects_malformed_results(output: str, error: str) -> None:
    with pytest.raises(ValueError, match=error):
        parse_translation_output(output)


def test_reference_exact_match_gets_full_reward_without_prompt_leakage() -> None:
    env = EnglishShareGPTToNepaliEnv(
        records=[
            {
                "id": "reference",
                "conversations": SOURCE,
                "reference_conversations": NEPALI,
            }
        ]
    )
    task = env.load_tasks()[0]
    evaluation = evaluate_translation(
        encoded(NEPALI),
        task.metadata["source_conversations"],
        task.metadata["reference_conversations"],
    )

    assert evaluation.score == pytest.approx(1.0)
    assert evaluation.reference_similarity == pytest.approx(1.0)
    assert "काठमाडौं" not in env.format_initial_message(task)
    assert "काठमाडौं" not in task.prompt


def test_reward_rejects_english_copy_and_malformed_json() -> None:
    copied = evaluate_translation(encoded(SOURCE), SOURCE)
    malformed = evaluate_translation("not json", SOURCE)

    assert copied.score < 0.5
    assert copied.devanagari == 0.0
    assert malformed.score == 0.0
    assert malformed.error


def test_reward_requires_protected_code_math_urls_and_numbers() -> None:
    source = [
        {
            "from": "human",
            "value": "Explain `2 ** 3`, $x=4$, and https://example.com in Python.",
        },
        {"from": "gpt", "value": "The result is 8."},
    ]
    preserved = [
        {
            "from": "human",
            "value": (
                "Python मा `2 ** 3`, $x=4$, र https://example.com "
                "व्याख्या गर्नुहोस्।"
            ),
        },
        {"from": "gpt", "value": "नतिजा 8 हो।"},
    ]
    corrupted = [
        {
            "from": "human",
            "value": "Python मा घाताङ्क र वेबसाइटबारे व्याख्या गर्नुहोस्।",
        },
        {"from": "gpt", "value": "नतिजा नौ हो।"},
    ]

    good = evaluate_translation(encoded(preserved), source)
    bad = evaluate_translation(encoded(corrupted), source)

    assert good.protected_spans == pytest.approx(1.0)
    assert good.score >= 0.8
    assert bad.protected_spans < 0.5
    assert bad.score < 0.8


@pytest.mark.asyncio
async def test_run_task_scores_and_exports_only_translated_conversation() -> None:
    inference = ScriptedInference(responses=[encoded(NEPALI), JUDGE_PASS])
    env = EnglishShareGPTToNepaliEnv(
        records=[{"id": "one", "conversations": SOURCE}],
        services=ServiceContainer(inference=inference),
    )
    env.config.dataset.num_rollouts = 1
    await env.setup()
    task = env.load_tasks()[0]
    result = await env.run_task(task)

    assert result.total_reward == pytest.approx(1.0)
    assert result.answer_correct is True
    assert result.reward_function == "sharegpt-nepali-semantic-v2"
    assert env.build_sharegpt_conversations(result, task) == NEPALI
    assert len(inference.calls) == 2
    assert inference.calls[0]["system_prompt"]
    assert "<source_conversation>" in inference.calls[0]["messages"][0]["content"]


@pytest.mark.asyncio
async def test_semantic_judge_rejects_unrelated_devanagari_without_reference() -> None:
    unrelated = [
        {"from": "human", "value": "क"},
        {"from": "gpt", "value": "क"},
    ]
    inference = ScriptedInference(responses=[encoded(unrelated), JUDGE_FAIL])
    env = EnglishShareGPTToNepaliEnv(
        records=[{"id": "unrelated", "conversations": SOURCE}],
        services=ServiceContainer(inference=inference),
    )
    env.config.dataset.num_rollouts = 1
    await env.setup()
    task = env.load_tasks()[0]

    result = await env.run_task(task)

    assert result.total_reward < 0.8
    assert result.answer_correct is False
    assert result.metadata["translation_semantic_evaluation"]["score"] == 0.0
    assert env.build_sharegpt_conversations(result, task) is None


@pytest.mark.asyncio
async def test_reference_free_reward_fails_closed_without_a_judge() -> None:
    inference = ScriptedInference(responses=[encoded(NEPALI)])
    config = EnglishShareGPTToNepaliEnv.default_config.model_copy(deep=True)
    config.llm_judge = None
    config.dataset.num_rollouts = 1
    env = EnglishShareGPTToNepaliEnv(
        config=config,
        records=[{"id": "no-judge", "conversations": SOURCE}],
        services=ServiceContainer(inference=inference),
    )
    await env.setup()

    result = await env.run_task(env.load_tasks()[0])

    assert result.total_reward == 0.0
    assert result.answer_correct is False
    assert result.metadata["translation_semantic_evaluation"]["error"]


@pytest.mark.asyncio
async def test_reviewed_reference_is_a_deterministic_judge_alternative() -> None:
    inference = ScriptedInference(responses=[encoded(NEPALI)])
    config = EnglishShareGPTToNepaliEnv.default_config.model_copy(deep=True)
    config.llm_judge = None
    config.dataset.num_rollouts = 1
    env = EnglishShareGPTToNepaliEnv(
        config=config,
        records=[
            {
                "id": "reviewed",
                "conversations": SOURCE,
                "reference_conversations": NEPALI,
            }
        ],
        services=ServiceContainer(inference=inference),
    )
    await env.setup()

    result = await env.run_task(env.load_tasks()[0])

    assert result.total_reward == pytest.approx(1.0)
    assert result.answer_correct is True
    assert result.metadata["translation_semantic_evaluation"]["method"] == (
        "reviewed-reference"
    )
    assert len(inference.calls) == 1


@pytest.mark.asyncio
async def test_grouped_rollout_selects_best_translation_without_api_key() -> None:
    inference = ScriptedInference(
        responses=["not json", encoded(SOURCE), encoded(NEPALI), JUDGE_PASS]
    )
    env = EnglishShareGPTToNepaliEnv(
        records=[{"id": "group", "conversations": SOURCE}],
        services=ServiceContainer(inference=inference),
    )
    env.config.dataset.num_rollouts = 3
    await env.setup()
    task = env.load_tasks()[0]
    best = await env.run_task(task)

    assert best.final_answer == encoded(NEPALI)
    assert best.total_reward == pytest.approx(1.0)
    assert best.answer_correct is True
    assert len(inference.calls) == 4


@pytest.mark.asyncio
@pytest.mark.parametrize("num_rollouts", [1, 2])
async def test_export_preserves_source_provenance_for_all_rollout_paths(
    num_rollouts: int,
    tmp_path: Path,
) -> None:
    responses = [encoded(NEPALI)] * num_rollouts + [JUDGE_PASS] * num_rollouts
    inference = ScriptedInference(responses=responses)
    inserter = CapturingInserter()
    env = EnglishShareGPTToNepaliEnv(
        records=[
            {
                "id": "licensed-source-row",
                "source": "OpenHermes",
                "license": "apache-2.0",
                "conversations": SOURCE,
            }
        ],
        services=ServiceContainer(inference=inference),
        data_inserter=inserter,
        num_rollouts=num_rollouts,
    )
    env.config.dataset.num_rollouts = num_rollouts
    env.config.dataset.output_dir = str(tmp_path)

    if num_rollouts == 1:
        await env.run()
    else:
        await env.setup()
        await env.run_task(env.load_tasks()[0])

    assert len(inserter.sharegpt_rows) == num_rollouts
    for row in inserter.sharegpt_rows:
        assert row["metadata"]["source_provenance"]["source"] == "OpenHermes"
        assert row["metadata"]["source_provenance"]["license"] == "apache-2.0"
        assert row["metadata"]["input_format"] == "sharegpt"


@pytest.mark.asyncio
async def test_standard_run_writes_local_dataset_and_audit_without_storage(
    tmp_path: Path,
) -> None:
    inference = ScriptedInference(responses=[encoded(NEPALI), JUDGE_PASS])
    config = EnglishShareGPTToNepaliEnv.default_config.model_copy(deep=True)
    config.dataset.output_dir = str(tmp_path)
    config.dataset.output_basename = "openhermes_preview"
    config.dataset.num_rollouts = 1
    env = EnglishShareGPTToNepaliEnv(
        config=config,
        records=[
            {
                "id": "source-row",
                "source": "OpenHermes",
                "license": "apache-2.0",
                "conversations": SOURCE,
            }
        ],
        services=ServiceContainer(inference=inference),
    )

    summary = await env.run()

    dataset_path = tmp_path / "openhermes_preview.jsonl"
    audit_path = tmp_path / "openhermes_preview_audit.jsonl"
    summary_path = tmp_path / "openhermes_preview_summary.json"
    dataset_rows = [json.loads(line) for line in dataset_path.read_text().splitlines()]
    audit_rows = [json.loads(line) for line in audit_path.read_text().splitlines()]
    saved_summary = json.loads(summary_path.read_text())

    assert dataset_rows[0]["conversations"] == NEPALI
    assert dataset_rows[0]["source_provenance"]["source"] == "OpenHermes"
    assert dataset_rows[0]["source_provenance"]["license"] == "apache-2.0"
    assert audit_rows[0]["accepted"] is True
    assert audit_rows[0]["source_conversations"] == SOURCE
    assert audit_rows[0]["translation_semantic_evaluation"]["score"] == 1.0
    assert saved_summary["accepted"] == 1
    assert saved_summary["rejected"] == 0
    assert summary.artifacts["sharegpt_jsonl"] == str(dataset_path.resolve())


def test_builder_refuses_low_reward_or_role_corruption() -> None:
    env = EnglishShareGPTToNepaliEnv(
        records=[{"id": "one", "conversations": SOURCE}]
    )
    task = env.load_tasks()[0]
    copied = TrajectoryResult(success=True, final_answer=encoded(SOURCE))
    wrong_roles = TrajectoryResult(
        success=True,
        final_answer=encoded(
            [
                {"from": "gpt", "value": "नेपालको राजधानी के हो?"},
                {
                    "from": "human",
                    "value": "नेपालको राजधानी काठमाडौं हो।",
                },
            ]
        ),
    )

    role_evaluation = evaluate_translation(
        wrong_roles.final_answer,
        task.metadata["source_conversations"],
    )

    assert env.build_sharegpt_conversations(copied, task) is None
    assert env.build_sharegpt_conversations(wrong_roles, task) is None
    assert role_evaluation.structure < 1.0
    assert role_evaluation.score < 0.8
