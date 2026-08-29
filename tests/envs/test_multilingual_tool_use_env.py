import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest
from pydantic import Field, ValidationError

from gymkhana.core.services import ServiceContainer
from gymkhana.core.services.inference import InferenceService
from gymkhana.envs import ENVIRONMENTS, Task
from gymkhana.envs.languages import BUILTIN_LANGUAGES, LanguageSpec, resolve_language
from gymkhana.envs.multilingual_tool_use import (
    LocalizationSettings,
    MultilingualToolUseEnv,
    argument_literals,
    check_localization,
    protected_tokens,
)
from gymkhana.run import load_environment_config


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
        self.calls.append(
            {"messages": list(messages), "system_prompt": system_prompt, "kwargs": kwargs}
        )
        return self.responses.pop(0)

    async def batch_generate(self, *, prompts: List[str], **kwargs: Any) -> List[str]:
        return [self.responses.pop(0) for _ in prompts]


EN_QUERY = "What is the weather in Paris on 2024-05-01? Use 3 day forecast."
EXPECTED = [
    {"name": "get_weather", "arguments": {"city": "Paris", "date": "2024-05-01", "days": 3}}
]
NE_QUERY = "2024-05-01 मा Paris को मौसम कस्तो छ? 3 दिनको पूर्वानुमान प्रयोग गर्नुहोस्।"
NE_SPEC = BUILTIN_LANGUAGES["ne-Deva"]


def make_env(inference: ScriptedInference) -> MultilingualToolUseEnv:
    if MultilingualToolUseEnv.default_config is None:
        MultilingualToolUseEnv(config=None)
    config = MultilingualToolUseEnv.default_config.model_copy(deep=True)
    config.dataset.limit = 1
    config.dataset.num_rollouts = 1
    return MultilingualToolUseEnv(config=config, services=ServiceContainer(inference=inference))


def make_task() -> Task:
    return Task(
        id="t1",
        prompt=EN_QUERY,
        metadata={
            "tools_openai": [
                {
                    "type": "function",
                    "function": {"name": "get_weather", "parameters": {"type": "object"}},
                }
            ],
            "tools_raw": [{"name": "get_weather"}],
            "expected_tool_calls": EXPECTED,
        },
    )


# ---------------------------------------------------------------------------
# Deterministic gate
# ---------------------------------------------------------------------------

def test_environment_is_registered() -> None:
    assert ENVIRONMENTS.get("multilingual-tool-use") is MultilingualToolUseEnv


def test_protected_tokens_and_argument_literals() -> None:
    assert protected_tokens("mail bob@example.com and open https://x.io/a?b=1 by 5pm") == {
        "bob@example.com": 1,
        "https://x.io/a?b=1": 1,
    }
    assert protected_tokens(EN_QUERY) == {}  # plain numbers are not censused
    literals = argument_literals(EXPECTED, EN_QUERY)
    assert literals == ["Paris", "2024-05-01", "3"]  # numeric args count, as ASCII digits
    nested = [{"name": "f", "arguments": {"filters": {"ids": ["ab", "x"], "n": 7, "on": True, "ratio": 2.0}}}]
    assert argument_literals(nested, "give me ab and x, 7 of them at ratio 2") == ["ab", "7", "2"]


def test_ideal_localization_passes() -> None:
    assert check_localization(
        source_query=EN_QUERY, localized_query=NE_QUERY, expected_calls=EXPECTED, spec=NE_SPEC
    ) == []


@pytest.mark.parametrize(
    ("localized", "expected_reason"),
    [
        ("", "empty_localization"),
        (EN_QUERY, "not_localized"),
        ("What is the weather in Paris on 2024-05-01? Use 3 day forecast!", "insufficient_script_ratio:ne-Deva"),
        ("२०२४-०५-०१ मा Paris को मौसम कस्तो छ? 3 दिनको पूर्वानुमान।", "missing_argument_literal:2024-05-01"),
        ("2024-05-01 मा पेरिस को मौसम कस्तो छ? 3 दिनको पूर्वानुमान।", "missing_argument_literal:Paris"),
        ("2024-05-01 मा Paris को मौसम कस्तो छ? तीन दिनको पूर्वानुमान।", "missing_argument_literal:3"),
    ],
)
def test_localization_gate_rejects(localized: str, expected_reason: str) -> None:
    reasons = check_localization(
        source_query=EN_QUERY, localized_query=localized, expected_calls=EXPECTED, spec=NE_SPEC
    )
    assert any(reason.startswith(expected_reason) for reason in reasons), reasons


# ---------------------------------------------------------------------------
# Environment flow
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_localized_query_reaches_policy_and_correct_call_is_rewarded() -> None:
    policy_output = json.dumps(
        [{"name": "get_weather", "arguments": {"city": "Paris", "date": "2024-05-01", "days": 3}}]
    )
    inference = ScriptedInference(responses=[NE_QUERY, policy_output])
    env = make_env(inference)
    task = make_task()

    result = await env.run_task(task)

    localizer_call, policy_call = inference.calls
    assert localizer_call["kwargs"]["model"] == env.ml_config.localizer_llm.model_identifier
    assert "Nepali" in localizer_call["system_prompt"]
    assert policy_call["messages"] == [{"role": "user", "content": NE_QUERY}]
    assert policy_call["kwargs"]["tools"] == task.metadata["tools_openai"]
    assert "Nepali" in policy_call["system_prompt"]
    assert "verbatim" in policy_call["system_prompt"]

    assert result.answer_correct is True
    assert result.total_reward == 1.0
    assert result.metadata["localized_query"] == NE_QUERY
    assert result.metadata["source_query"] == EN_QUERY
    assert task.metadata["localization"]["passed"] is True

    conversations = env.build_sharegpt_conversations(result, task)
    assert conversations and any(NE_QUERY in m.get("value", "") for m in conversations)
    meta = env.build_sharegpt_metadata(result, task)
    assert meta["target_language"] == "ne-Deva"
    assert meta["expected_tool_calls"] == EXPECTED


@pytest.mark.asyncio
async def test_wrong_call_scores_zero_and_is_not_exported() -> None:
    inference = ScriptedInference(
        responses=[NE_QUERY, json.dumps([{"name": "get_weather", "arguments": {"city": "पेरिस"}}])]
    )
    env = make_env(inference)
    task = make_task()

    result = await env.run_task(task)

    assert result.answer_correct is False
    assert result.total_reward == 0.0
    assert env.build_sharegpt_conversations(result, task) is None


@pytest.mark.asyncio
async def test_failed_localization_rejects_row_without_calling_policy() -> None:
    inference = ScriptedInference(responses=["२०२४-०५-०१ मा पेरिस को मौसम कस्तो छ?", "unused"])
    env = make_env(inference)
    task = make_task()

    result = await env.run_task(task)

    assert len(inference.calls) == 1  # localizer only; hard-fail, no policy rollout
    assert result.success is False and result.total_reward == 0.0
    assert result.metadata["rejection"] == "localization_failed"
    reasons = result.metadata["localization"]["attempts"][0]["issues"]
    assert "missing_argument_literal:Paris" in reasons
    assert "missing_argument_literal:2024-05-01" in reasons
    assert env.build_sharegpt_conversations(result, task) is None


@pytest.mark.asyncio
async def test_localization_is_cached_across_rollouts() -> None:
    policy_output = json.dumps([{"name": "get_weather", "arguments": EXPECTED[0]["arguments"]}])
    inference = ScriptedInference(responses=[NE_QUERY, policy_output, policy_output])
    env = make_env(inference)
    env.config.dataset.num_rollouts = 2
    task = make_task()

    result = await env.run_task(task)

    assert result.total_reward == 1.0
    assert sum(1 for c in inference.calls if "tools" in c["kwargs"]) == 2
    assert sum(1 for c in inference.calls if "tools" not in c["kwargs"]) == 1


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def test_checked_in_config_loads() -> None:
    env_name, config = load_environment_config(Path("configs/multilingual_tool_use/xlam_nepali.yaml"))
    assert env_name == "multilingual-tool-use"
    assert config.localization.language_spec.code == "ne-Deva"
    assert config.localizer_llm.model_identifier == "google:gemini-3.6-flash"
    assert config.dataset.reward_function == "tool_call_match"


def test_custom_language_in_yaml(tmp_path: Path) -> None:
    path = tmp_path / "hindi.yaml"
    path.write_text(
        "\n".join(
            [
                "name: multilingual-tool-use",
                "localization:",
                "  target_language: hi-Deva",
                "  languages:",
                "    hi-Deva:",
                "      code: hi-Deva",
                "      name: Hindi (Devanagari)",
                "      instruction: Write natural Hindi in Devanagari.",
                '      script_regex: "[\\u0900-\\u097f]"',
            ]
        ),
        encoding="utf-8",
    )
    _, config = load_environment_config(path)
    assert config.localization.language_spec.name == "Hindi (Devanagari)"
    with pytest.raises(ValidationError, match="unknown target_language"):
        LocalizationSettings(target_language="xx-Zzzz")


def test_language_registry_shim_still_importable() -> None:
    from gymkhana.envs.multi_turn_qa.languages import LanguageSpec as Shim

    assert Shim is LanguageSpec
    assert resolve_language("ne-Latn").context_label == "Sandarbh"


def test_hermes_rows_become_tasks_with_english_ground_truth(monkeypatch: pytest.MonkeyPatch) -> None:
    inference = ScriptedInference(responses=[])
    env = make_env(inference)
    env.config.source_format = "hermes"
    tools = [{"type": "function", "function": {"name": "get_camera_live_feed", "parameters": {"type": "object", "properties": {"camera_id": {"type": "string"}}}}}]
    records = [
        {
            "id": "row-1",
            "category": "IoT and Home Automation",
            "subcategory": "Security",
            "task": "View camera feed",
            "tools": json.dumps(tools),
            "conversations": [
                {"from": "system", "value": "You are a function calling AI model. <tools>" + json.dumps(tools) + "</tools>"},
                {"from": "human", "value": "Show me the front_door camera in 1080p."},
                {"from": "gpt", "value": '<tool_call>\n{"name": "get_camera_live_feed", "arguments": {"camera_id": "front_door", "stream_quality": "1080p"}}\n</tool_call>'},
            ],
        },
        {"id": "row-2", "tools": "[]", "conversations": [{"from": "human", "value": "no tools here"}]},
    ]
    monkeypatch.setattr(env, "_load_dataset", lambda *_: records)

    tasks = env.load_tasks(limit=10)

    assert [t.id for t in tasks] == ["row-1"]
    task = tasks[0]
    assert task.prompt == "Show me the front_door camera in 1080p."
    assert task.metadata["expected_tool_calls"] == [
        {"name": "get_camera_live_feed", "arguments": {"camera_id": "front_door", "stream_quality": "1080p"}}
    ]
    assert task.metadata["tools_openai"][0]["function"]["name"] == "get_camera_live_feed"
    assert task.metadata["source_provenance"]["category"] == "IoT and Home Automation"
    assert argument_literals(task.metadata["expected_tool_calls"], task.prompt) == ["front_door", "1080p"]


def test_hermes_config_loads() -> None:
    env_name, config = load_environment_config(
        Path("configs/multilingual_tool_use/hermes_singleturn_nepali.yaml")
    )
    assert env_name == "multilingual-tool-use"
    assert config.source_format == "hermes"
    assert config.dataset.dataset_config == "func_calling_singleturn"
    assert config.dataset.num_rollouts == 4


def test_localizer_prompt_asks_for_spoken_requests() -> None:
    env = make_env(ScriptedInference(responses=[]))
    prompt = env._localizer_system_prompt()
    assert "Siri or Alexa" in prompt
    assert "not a word-for-word translation" in prompt
    assert "never as key=value pairs" in prompt


@pytest.mark.parametrize(
    ("mode", "expected_fragment"),
    [("english", None), ("target", "Think through the request in Nepali (Devanagari)"), ("hybrid", "keeping English for tool names")],
)
def test_policy_reasoning_mode_controls_system_prompt(mode: str, expected_fragment: Optional[str]) -> None:
    env = make_env(ScriptedInference(responses=[]))
    env.config.policy_reasoning = mode
    prompt = env.build_system_prompt(make_task())
    if expected_fragment is None:
        assert "Think through" not in prompt
    else:
        assert expected_fragment in prompt
    assert "verbatim" in prompt
