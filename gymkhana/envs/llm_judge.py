"""LLM-as-Judge reward function with rubric support.

Evaluates model responses using a secondary LLM when verifiable ground-truth
answers are not available. Supports configurable rubrics for structured scoring.

Usage::

    from gymkhana.envs.llm_judge import LLMJudgeReward

    judge = LLMJudgeReward(settings=my_llm_judge_settings)
    score = await judge.score(prompt=task_prompt, response=model_answer)

Or as a registered reward function via config::

    config = EnvConfig(
        ...,
        llm_judge=LLMJudgeSettings(model="gpt-4o-mini", rubric_prompt=MY_RUBRIC),
    )
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from gymkhana.envs.config import LLMJudgeSettings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Default rubrics
# ---------------------------------------------------------------------------

DEFAULT_RUBRIC = """You are an expert evaluator. Score the following response on a scale of 0 to 10.

## Evaluation Criteria
- **Accuracy** (0-3): Is the response factually correct?
- **Completeness** (0-3): Does it answer all parts of the question?
- **Clarity** (0-2): Is it well-organized and easy to understand?
- **Instruction Following** (0-2): Does it follow any specific formatting or style constraints?

## Task
**Prompt:** {prompt}

**Response:** {response}

## Output Format
Return ONLY a JSON object with the following fields:
```json
{{
  "accuracy": <0-3>,
  "completeness": <0-3>,
  "clarity": <0-2>,
  "instruction_following": <0-2>,
  "total_score": <0-10>,
  "reasoning": "<brief explanation>"
}}
```"""

BINARY_RUBRIC = """You are an expert evaluator. Determine if the response correctly and completely answers the question.

**Prompt:** {prompt}

**Response:** {response}

## Output Format
Return ONLY a JSON object:
```json
{{
  "correct": true or false,
  "reasoning": "<brief explanation>"
}}
```"""

COMPARATIVE_RUBRIC = """You are an expert evaluator. Compare the response against the reference answer.

**Prompt:** {prompt}

**Response:** {response}

**Reference Answer:** {reference}

## Output Format
Return ONLY a JSON object:
```json
{{
  "score": <0-10>,
  "matches_reference": true or false,
  "reasoning": "<brief explanation>"
}}
```"""

BUILTIN_RUBRICS = {
    "default": DEFAULT_RUBRIC,
    "binary": BINARY_RUBRIC,
    "comparative": COMPARATIVE_RUBRIC,
}


# ---------------------------------------------------------------------------
# Judge result model
# ---------------------------------------------------------------------------

class JudgeResult(BaseModel):
    """Structured result from a single judge evaluation."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    score: float = Field(default=0.0, ge=0.0, le=10.0)
    normalized_score: float = Field(default=0.0, ge=0.0, le=1.0)
    raw_output: str = ""
    parsed: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# LLM Judge
# ---------------------------------------------------------------------------

class LLMJudge:
    """Stateless LLM-as-judge scorer.

    Instantiate with :class:`LLMJudgeSettings` and call :meth:`score`
    to evaluate a single (prompt, response) pair.
    """

    def __init__(self, settings: "LLMJudgeSettings") -> None:
        self.settings = settings
        self._rubric_template = self._resolve_rubric(settings.rubric_prompt)

    # ------------------------------------------------------------------
    # Rubric resolution
    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_rubric(rubric_prompt: Optional[str]) -> str:
        """Resolve a rubric: built-in name, custom template, or default."""
        if rubric_prompt is None:
            return DEFAULT_RUBRIC
        if rubric_prompt in BUILTIN_RUBRICS:
            return BUILTIN_RUBRICS[rubric_prompt]
        # Treat as custom template string
        return rubric_prompt

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------
    async def score(
        self,
        prompt: str,
        response: str,
        *,
        reference: Optional[str] = None,
        inference_service: Any = None,
        extra_context: Optional[Dict[str, str]] = None,
    ) -> JudgeResult:
        """Score a single response.

        Args:
            prompt: The original task prompt.
            response: The model's response to evaluate.
            reference: Optional reference/gold answer for comparative rubrics.
            inference_service: The inference service to use for LLM calls.
            extra_context: Additional template variables for the rubric.

        Returns:
            A :class:`JudgeResult` with normalized score in [0, 1].
        """
        if inference_service is None:
            return JudgeResult(
                error="No inference service provided for judge call"
            )

        # Build the evaluation prompt from rubric template
        template_vars = {
            "prompt": prompt,
            "response": response,
            "reference": reference or "(no reference provided)",
        }
        if extra_context:
            template_vars.update(extra_context)

        try:
            eval_prompt = self._rubric_template.format(**template_vars)
        except KeyError as e:
            return JudgeResult(error=f"Rubric template missing variable: {e}")

        # Collect multiple judge samples if num_judges > 1
        results: List[JudgeResult] = []
        for _ in range(self.settings.num_judges):
            result = await self._single_judge_call(eval_prompt, inference_service)
            results.append(result)

        if len(results) == 1:
            return results[0]

        # Majority vote / average for multi-judge
        return self._aggregate_results(results)

    async def _single_judge_call(
        self,
        eval_prompt: str,
        inference_service: Any,
    ) -> JudgeResult:
        """Make a single judge LLM call and parse the result."""
        try:
            raw_output = await inference_service.generate(
                messages=[{"role": "user", "content": eval_prompt}],
                system_prompt="You are a precise evaluation judge. Follow the output format exactly.",
                model=self.settings.model,
                temperature=self.settings.temperature,
                max_tokens=self.settings.max_tokens,
            )

            parsed = self._parse_judge_output(raw_output)
            score = self._extract_score(parsed)
            normalized = min(max(score / 10.0, 0.0), 1.0)

            return JudgeResult(
                score=score,
                normalized_score=normalized,
                raw_output=raw_output,
                parsed=parsed,
            )

        except Exception as e:
            logger.exception("Judge call failed")
            return JudgeResult(error=str(e), raw_output="")

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_judge_output(raw: str) -> Dict[str, Any]:
        """Extract JSON from the judge's response."""
        # Try direct JSON parse
        text = raw.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try to find JSON block in markdown
        json_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1).strip())
            except json.JSONDecodeError:
                pass

        # Try to find any JSON object
        brace_match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
        if brace_match:
            try:
                return json.loads(brace_match.group(0))
            except json.JSONDecodeError:
                pass

        logger.warning("Could not parse judge output as JSON: %s", text[:200])
        return {"raw": text}

    @staticmethod
    def _extract_score(parsed: Dict[str, Any]) -> float:
        """Extract a numeric score from the parsed judge output.

        Handles different rubric output formats:
        - ``{"total_score": N}`` (default rubric)
        - ``{"score": N}`` (comparative rubric)
        - ``{"correct": bool}`` (binary rubric)
        """
        if "total_score" in parsed:
            return float(parsed["total_score"])
        if "score" in parsed:
            return float(parsed["score"])
        if "correct" in parsed:
            return 10.0 if parsed["correct"] else 0.0
        # Fallback: look for any numeric value
        for v in parsed.values():
            if isinstance(v, (int, float)) and 0 <= v <= 10:
                return float(v)
        return 0.0

    # ------------------------------------------------------------------
    # Multi-judge aggregation
    # ------------------------------------------------------------------
    @staticmethod
    def _aggregate_results(results: List[JudgeResult]) -> JudgeResult:
        """Average scores across multiple judge samples."""
        valid = [r for r in results if r.error is None]
        if not valid:
            return JudgeResult(
                error="All judge calls failed",
                parsed={"individual_errors": [r.error for r in results]},
            )

        avg_score = sum(r.score for r in valid) / len(valid)
        avg_norm = sum(r.normalized_score for r in valid) / len(valid)

        return JudgeResult(
            score=round(avg_score, 2),
            normalized_score=round(avg_norm, 4),
            raw_output=valid[0].raw_output,  # Keep first for reference
            parsed={
                "aggregation": "average",
                "n_judges": len(valid),
                "n_failed": len(results) - len(valid),
                "individual_scores": [r.score for r in valid],
            },
        )


# ---------------------------------------------------------------------------
# Convenience: integrate with Environment.compute_reward
# ---------------------------------------------------------------------------

async def judge_reward(
    *,
    settings: "LLMJudgeSettings",
    prompt: str,
    response: str,
    reference: Optional[str] = None,
    inference_service: Any = None,
    extra_context: Optional[Dict[str, str]] = None,
) -> float:
    """One-shot convenience function returning a normalized [0, 1] reward.

    Typical usage inside ``Environment.compute_reward``::

        if self.config.llm_judge:
            return await judge_reward(
                settings=self.config.llm_judge,
                prompt=task.prompt,
                response=result.final_answer,
                inference_service=self._inference_service,
            )
    """
    judge = LLMJudge(settings)
    result = await judge.score(
        prompt=prompt,
        response=response,
        reference=reference,
        inference_service=inference_service,
        extra_context=extra_context,
    )
    if result.error:
        logger.warning("Judge reward failed, returning 0.0: %s", result.error)
        return 0.0
    return result.normalized_score
