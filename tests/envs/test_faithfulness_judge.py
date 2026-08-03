"""Offline tests for FaithfulnessJudgeEnvironment.

Does NOT mock generate_response(). Under the real TOOL_CALL flow,
ToolUseMode.execute_single() calls generate_response() once and lets
Pydantic AI's Agent drive the model -> tool -> model loop internally,
populating Turn.tool_calls from the toolkit's execution trace. Mocking
generate_response alone would never actually invoke the toolkit, so the
trace (and therefore tool_calls) would stay empty regardless of what we
return -- that would test nothing real.

Instead, these tests exercise:
1. The submit_verification tool function directly (no API needed --
   it's a plain Python function).
2. evaluate_answer()/compute_reward() against manually-constructed
   TrajectoryResult/Turn objects, in the exact shape
   ToolUseMode.execute_single() is confirmed to produce (see
   gymkhana/envs/modes/tool_use.py): a Turn with
   tool_calls=[{"id", "name", "arguments"}].

This needs no OPENAI_API_KEY / ANTHROPIC_API_KEY and never touches a
real inference provider.

Run with:
    pytest gymkhana/envs/faithfulness_judge/test_faithfulness_judge.py -v
"""

from __future__ import annotations

import pytest

from gymkhana.core.models import TrajectoryResult, Turn
from gymkhana.envs.environment import Task
from gymkhana.envs.faithfulness_judge.environment import (
    FaithfulnessJudgeEnvironment,
    SUBMIT_VERIFICATION_TOOL_NAME,
    VerifierOutput,
    submit_verification,
)


def _result_with_tool_call(name: str, arguments: dict, *, success: bool = True) -> TrajectoryResult:
    """Build a TrajectoryResult carrying one tool-call turn, matching the
    shape ToolUseMode.execute_single() produces."""
    turns = [
        Turn(role="user", content="CONTEXT:\n...\n\nANSWER:\n...", turn_index=0),
        Turn(
            role="assistant",
            content="",
            tool_calls=[{"id": "tool-0", "name": name, "arguments": arguments}],
            turn_index=1,
        ),
        Turn(role="tool", content="{}", tool_call_id="tool-0", turn_index=2),
        Turn(role="assistant", content="Done.", turn_index=3),
    ]
    return TrajectoryResult(
        success=success,
        final_answer="Done.",
        turns=turns,
        task_id="t1",
        environment="faithfulness_judge",
    )


def _result_with_no_tool_call() -> TrajectoryResult:
    """Simulates the model ignoring the tool entirely and answering in
    plain text -- must degrade gracefully, not crash."""
    turns = [
        Turn(role="user", content="CONTEXT:\n...\n\nANSWER:\n...", turn_index=0),
        Turn(role="assistant", content="I think it's fine.", turn_index=1),
    ]
    return TrajectoryResult(
        success=True,
        final_answer="I think it's fine.",
        turns=turns,
        task_id="t1",
        environment="faithfulness_judge",
    )


@pytest.fixture
def env() -> FaithfulnessJudgeEnvironment:
    return FaithfulnessJudgeEnvironment()


@pytest.fixture
def sample_task() -> Task:
    return Task(
        id="t1",
        prompt="CONTEXT:\nThe sky is blue.\n\nANSWER:\nThe sky is blue.",
        metadata={
            "context": "The sky is blue.",
            "answer": "The sky is blue.",
            "reference_is_supported": True,
        },
    )


# ---------------------------------------------------------------------------
# submit_verification -- the tool itself, called directly, no API involved
# ---------------------------------------------------------------------------
class TestSubmitVerificationTool:
    @pytest.mark.asyncio
    async def test_valid_call_returns_validated_dict(self):
        result = await submit_verification(
            critique="Fully entailed.", is_supported=True, reward_score=1.0
        )
        assert result == {"critique": "Fully entailed.", "is_supported": True, "reward_score": 1.0}

    @pytest.mark.asyncio
    async def test_reward_out_of_range_returns_error_not_raise(self):
        """The tool must never raise -- an exception here would surface as
        a hard failure inside Pydantic AI's agent loop rather than a
        recoverable, scoreable rollout."""
        result = await submit_verification(
            critique="x", is_supported=True, reward_score=5.0
        )
        assert "error" in result

    def test_toolkit_registers_the_tool(self, env):
        assert SUBMIT_VERIFICATION_TOOL_NAME in env._toolkit.tool_names

    def test_get_tool_executor_returns_the_toolkit(self, env, sample_task):
        assert env.get_tool_executor(sample_task) is env._toolkit


# ---------------------------------------------------------------------------
# VerifierOutput schema validation
# ---------------------------------------------------------------------------
class TestVerifierOutputSchema:
    def test_valid_payload_parses(self):
        verdict = VerifierOutput.model_validate(
            {"critique": "ok", "is_supported": True, "reward_score": 1.0}
        )
        assert verdict.is_supported is True

    def test_reward_score_out_of_range_rejected(self):
        with pytest.raises(Exception):
            VerifierOutput.model_validate(
                {"critique": "ok", "is_supported": True, "reward_score": 1.5}
            )

    def test_missing_field_rejected(self):
        with pytest.raises(Exception):
            VerifierOutput.model_validate({"critique": "ok", "is_supported": True})


# ---------------------------------------------------------------------------
# evaluate_answer / compute_reward -- against constructed trajectories
# ---------------------------------------------------------------------------
class TestScoring:
    def test_evaluate_answer_reads_supported_verdict(self, env, sample_task):
        result = _result_with_tool_call(
            SUBMIT_VERIFICATION_TOOL_NAME,
            {"critique": "ok", "is_supported": True, "reward_score": 1.0},
        )
        assert env.evaluate_answer(sample_task, result) is True

    def test_evaluate_answer_reads_unsupported_verdict(self, env, sample_task):
        result = _result_with_tool_call(
            SUBMIT_VERIFICATION_TOOL_NAME,
            {"critique": "contradicts", "is_supported": False, "reward_score": 0.0},
        )
        assert env.evaluate_answer(sample_task, result) is False

    def test_evaluate_answer_none_when_tool_never_called(self, env, sample_task):
        result = _result_with_no_tool_call()
        assert env.evaluate_answer(sample_task, result) is None

    def test_evaluate_answer_none_for_wrong_tool_name(self, env, sample_task):
        result = _result_with_tool_call(
            "some_other_tool", {"critique": "ok", "is_supported": True, "reward_score": 1.0}
        )
        assert env.evaluate_answer(sample_task, result) is None

    def test_evaluate_answer_none_for_malformed_arguments(self, env, sample_task):
        result = _result_with_tool_call(
            SUBMIT_VERIFICATION_TOOL_NAME,
            {"critique": "ok", "is_supported": True, "reward_score": 5.0},  # out of range
        )
        assert env.evaluate_answer(sample_task, result) is None

    @pytest.mark.asyncio
    async def test_compute_reward_matches_reward_score(self, env, sample_task):
        result = _result_with_tool_call(
            SUBMIT_VERIFICATION_TOOL_NAME,
            {"critique": "partially supported", "is_supported": False, "reward_score": 0.4},
        )
        reward = await env.compute_reward(result, task=sample_task)
        assert reward == pytest.approx(0.4)
        assert result.total_reward == pytest.approx(0.4)
        assert result.reward_function == "faithfulness-judge"

    @pytest.mark.asyncio
    async def test_compute_reward_zero_when_no_tool_call(self, env, sample_task):
        """If the model ignores the tool, this must degrade to zero
        reward, never raise -- one bad rollout can't take down an RLVR
        training batch."""
        result = _result_with_no_tool_call()
        reward = await env.compute_reward(result, task=sample_task)
        assert reward == pytest.approx(0.0)

    @pytest.mark.asyncio
    async def test_compute_reward_zero_for_malformed_call(self, env, sample_task):
        result = _result_with_tool_call(
            SUBMIT_VERIFICATION_TOOL_NAME,
            {"critique": "ok", "is_supported": True, "reward_score": -1.0},
        )
        reward = await env.compute_reward(result, task=sample_task)
        assert reward == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# load_tasks
# ---------------------------------------------------------------------------
class TestLoadTasks:
    def test_loads_all_rows_by_default(self, env):
        tasks = env.load_tasks()
        assert len(tasks) >= 1
        assert all(isinstance(t, Task) for t in tasks)

    def test_respects_limit(self, env):
        tasks = env.load_tasks(limit=1)
        assert len(tasks) == 1

    def test_prompt_contains_context_and_answer(self, env):
        tasks = env.load_tasks(limit=1)
        assert "CONTEXT:" in tasks[0].prompt
        assert "ANSWER:" in tasks[0].prompt

    def test_reference_label_not_leaked_into_prompt(self, env):
        """The human reference label must stay in metadata only -- if it
        leaked into the prompt the judge would be grading against its own
        answer key."""
        tasks = env.load_tasks(limit=1)
        assert "reference_is_supported" not in tasks[0].prompt
        assert "reference_is_supported" in tasks[0].metadata
