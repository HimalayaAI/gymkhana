"""
Faithfulness / Hallucination Judge environment for Gymkhana.

Given a (context, answer) pair, an LLM judge produces a structured
critique, a boolean support verdict, and a reward score in [0.0, 1.0].

Pattern confirmed against gymkhana/envs/generic_qa/generic_qa.py and
gymkhana/envs/modes/tool_use.py (the correct TOOL_CALL reference,
per maintainer feedback -- NOT tool_use_singleturn.py, which predates
the EnvironmentToolkit convention and manually parses tool-call JSON
out of raw response text instead of using it):

- The tool is a plain Python callable, registered via EnvironmentToolkit.
  The framework derives its JSON schema from the function's type hints
  (see EnvironmentToolkit._arguments_schema in tool_bridge.py).
- `get_tool_executor(task)` is the only hook needed to wire it up.
- The base Environment.run_task() already dispatches TOOL_CALL mode to
  ToolUseMode.execute_single(), which calls generate_response() ONCE --
  Pydantic AI's Agent owns the full model -> tool -> model loop
  internally. We never call generate_response ourselves.
- ToolUseMode.execute_single() builds Turn objects with structured
  `tool_calls` (raw arguments dict, already parsed) attached directly --
  so evaluate_answer()/compute_reward() read those off result.turns
  rather than re-parsing response text.

This means no execute_task/run_task override is needed at all -- just
load_tasks(), get_tool_executor(), get_environment_instructions(),
evaluate_answer(), and compute_reward(), same shape as GenericQAEnv.
"""

from __future__ import annotations

import logging
import os
from typing import Any, ClassVar, Dict, List, Optional, Sequence

from pydantic import BaseModel, Field, ValidationError

from gymkhana.core.models import TrajectoryResult
from gymkhana.envs.config import (
    DatasetSettings,
    EnvConfig,
    EnvironmentType,
    InferenceConfig,
    InteractionMode,
    LLMClientType,
    ToolUseModeSettings,
)
from gymkhana.envs.environment import Environment, Task, register_environment
from gymkhana.envs.tool_bridge import EnvironmentToolkit

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Verifier output schema
# ---------------------------------------------------------------------------
class VerifierOutput(BaseModel):
    """
    Reward contract: reward_score is a continuous value in [0.0, 1.0].
    1.0 = fully supported by context, no fabrication. 0.0 = contradicts
    or fabricates from the context entirely. Partial credit is valid and
    intentionally graded, not binary -- e.g. an answer that is mostly
    supported but adds one unstated detail should score in the
    mid-range, not collapse to 0 or 1. is_supported is a hard boolean
    threshold derived from the same judgment for callers that need a
    pass/fail signal (e.g. best-of-N filtering); reward_score is the
    signal actually used for RLVR.
    """

    critique: str = Field(
        description="Step-by-step reasoning evaluating the answer against the context."
    )
    is_supported: bool = Field(
        description="True if the answer is fully supported by the context, False otherwise."
    )
    reward_score: float = Field(
        ge=0.0,
        le=1.0,
        description="A score between 0.0 and 1.0 reflecting how well-supported the answer is.",
    )


SUBMIT_VERIFICATION_TOOL_NAME = "submit_verification"


# ---------------------------------------------------------------------------
# 2. The tool itself -- a plain Python callable, same shape as add/multiply
#    in generic_qa.py. EnvironmentToolkit derives the JSON schema the model
#    sees from these type hints; it does NOT need a hand-built OpenAI dict.
# ---------------------------------------------------------------------------
async def submit_verification(critique: str, is_supported: bool, reward_score: float) -> Dict[str, Any]:
    """Submit your faithfulness verdict for the given context/answer pair.

    Args:
        critique: Step-by-step reasoning evaluating the answer against the context.
        is_supported: True if the answer is fully supported by the context, False otherwise.
        reward_score: A score between 0.0 and 1.0 reflecting how well-supported the answer is.
    """
    try:
        verdict = VerifierOutput(
            critique=critique, is_supported=is_supported, reward_score=reward_score
        )
    except ValidationError as exc:
        # Never raise out of a tool call -- Pydantic AI's internal loop
        # would otherwise surface this as a hard failure rather than a
        # recoverable, zero-reward rollout. Errors are read back out of
        # the tool-call arguments by evaluate_answer/compute_reward, not
        # this return value, so this is mostly a defensive echo.
        return {"error": str(exc)}
    return verdict.model_dump()


# ---------------------------------------------------------------------------
# 3. Dataset rows: replace with real loading (jsonl / HF dataset / your own
#    pipeline's saved context+answer pairs) when ready.
# ---------------------------------------------------------------------------
ROWS = (
    {
        "id": "faithfulness-example-1",
        "context": "The Annapurna Circuit is roughly 160-230 km depending on the route and takes 15-20 days to complete.",
        "answer": "The Annapurna Circuit is about 500 km long and takes a month to finish.",
        "reference_is_supported": False,  # human label, used for offline judge eval only
    },
    {
        "id": "faithfulness-example-2",
        "context": "TAAN (Trekking Agencies' Association of Nepal) is the umbrella body representing trekking agencies in Nepal.",
        "answer": "TAAN represents trekking agencies in Nepal.",
        "reference_is_supported": True,
    },
)


JUDGE_INSTRUCTIONS = (
    "You are a strict faithfulness judge. You are given a CONTEXT and an "
    "ANSWER. Decide whether the ANSWER is fully supported by the CONTEXT "
    "alone, with no outside knowledge. Penalize unsupported claims, "
    "fabricated numbers, and plausible-but-unstated generalizations. "
    "Always respond by calling the submit_verification tool -- never "
    "answer in plain text."
)


def _get_default_config() -> EnvConfig:
    """Reads a dedicated FAITHFULNESS_JUDGE_MODEL env var rather than
    LITELLM_MODEL -- keeps the judge's model config isolated from whatever
    policy model produced the answers being judged (README guidance on
    LLM judges: document why one is needed and isolate it from the
    policy model)."""
    client_str = os.getenv("LITELLM_CLIENT", "litellm").lower()
    client_map = {client.value: client for client in LLMClientType}

    return EnvConfig(
        name="faithfulness-judge",
        llm=InferenceConfig(
            client=client_map.get(client_str, LLMClientType.LITELLM),
            model=os.getenv("FAITHFULNESS_JUDGE_MODEL", "gpt-4o"),
            temperature=float(os.getenv("FAITHFULNESS_JUDGE_TEMPERATURE", "0.0")),
            max_tokens=int(os.getenv("FAITHFULNESS_JUDGE_MAX_TOKENS", "1024")),
        ),
        interaction_mode=InteractionMode.TOOL_CALL,
        mode_config=ToolUseModeSettings(max_turns=1),
        dataset=DatasetSettings(
            environment="faithfulness-judge",
            num_rollouts=int(os.getenv("FAITHFULNESS_JUDGE_ROLLOUTS", "1")),
            limit=int(os.getenv("FAITHFULNESS_JUDGE_LIMIT", "100")),
            output_dir="outputs/faithfulness_judge",
            output_sharegpt=True,
            enable_rewards=True,
        ),
        debug=False,
    )


@register_environment(name="faithfulness-judge", env_type=EnvironmentType.FAITHFULNESS_JUDGE)
class FaithfulnessJudgeEnvironment(Environment):
    """Self-RAG style context/answer faithfulness judge.

    Important: this environment's LLM call IS the judge, not a policy
    model being scored. This environment should only ever be pointed at
    (context, answer) pairs produced *elsewhere* -- never at its own
    judge output -- and it reads its own FAITHFULNESS_JUDGE_MODEL env
    var rather than LITELLM_MODEL so its config never silently shares
    identity with a policy run.
    """

    name: str = "faithfulness_judge"
    default_config: ClassVar[Optional[EnvConfig]] = None

    def __init__(self, *, config: Optional[EnvConfig] = None, **data: Any) -> None:
        if FaithfulnessJudgeEnvironment.default_config is None:
            FaithfulnessJudgeEnvironment.default_config = _get_default_config()

        if config is None:
            config = FaithfulnessJudgeEnvironment.default_config.model_copy(deep=True)
        elif isinstance(config, dict):
            config = EnvConfig(**config)

        data["config"] = config
        super().__init__(**data)

        self._toolkit = EnvironmentToolkit(tools=[submit_verification])

    # ------------------------------------------------------------------
    # Tool wiring -- the only hook TOOL_CALL mode strictly needs.
    # ------------------------------------------------------------------
    def get_tool_executor(self, task: Task) -> Optional[EnvironmentToolkit]:
        return self._toolkit

    # ------------------------------------------------------------------
    # Dataset handling
    # ------------------------------------------------------------------
    def load_tasks(self, limit: Optional[int] = None) -> Sequence[Task]:
        rows = ROWS if limit is None else ROWS[:limit]
        tasks: List[Task] = []
        for row in rows:
            tasks.append(
                Task(
                    id=row["id"],
                    prompt=f"CONTEXT:\n{row['context']}\n\nANSWER:\n{row['answer']}",
                    metadata={
                        "context": row["context"],
                        "answer": row["answer"],
                        # Offline judge-quality eval only -- never shown to the judge model.
                        "reference_is_supported": row["reference_is_supported"],
                    },
                )
            )
        return tasks

    def get_environment_instructions(self, task: Task) -> str:
        return JUDGE_INSTRUCTIONS

    # format_initial_message: base class default (returns task.prompt) is
    # exactly what we want -- no override needed.

    # ------------------------------------------------------------------
    # Scoring -- read the structured tool call straight off result.turns.
    # ------------------------------------------------------------------
    def evaluate_answer(self, task: Task, result: TrajectoryResult) -> Optional[bool]:
        verdict = self._find_verdict(result)
        return verdict.is_supported if verdict is not None else None

    async def compute_reward(
        self,
        result: TrajectoryResult,
        answer_correct: Optional[bool] = None,
        task: Optional[Task] = None,
    ) -> float:
        verdict = self._find_verdict(result)
        is_supported = verdict.is_supported if verdict is not None else False
        reward = float(verdict.reward_score) if verdict is not None else 0.0

        # Set independently of the base class's _score_trajectory having
        # already called evaluate_answer() first -- per README section 5,
        # compute_reward must score candidates correctly even when called
        # standalone (e.g. outside the normal run_task flow).
        result.answer_correct = is_supported
        result.total_reward = reward
        result.reward_function = "faithfulness-judge"
        return reward

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _find_verdict(result: TrajectoryResult) -> Optional[VerifierOutput]:
        """Scan turns for the submit_verification tool call and validate
        its arguments. Returns None on a missing or malformed call --
        never raises, so one bad rollout can't crash an RLVR batch."""
        for turn in result.turns:
            if not turn.tool_calls:
                continue
            for call in turn.tool_calls:
                if call.get("name") != SUBMIT_VERIFICATION_TOOL_NAME:
                    continue
                args = call.get("arguments", {})
                try:
                    return VerifierOutput.model_validate(args)
                except ValidationError:
                    logger.warning("Malformed submit_verification args: %s", args)
                    return None
        return None
