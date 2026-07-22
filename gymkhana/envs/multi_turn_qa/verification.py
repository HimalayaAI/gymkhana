"""Question validation and answer verification for generated QA conversations."""

from __future__ import annotations

import json
import math
import re
import unicodedata
from typing import Any, Optional

from gymkhana.envs.config import LLMJudgeSettings
from gymkhana.envs.llm_judge import LLMJudge

from .models import (
    AnswerType,
    ContextPolicy,
    ConversationEvaluation,
    QAGenerationSettings,
    QATurnPlan,
    QuestionDraft,
    SourceDocument,
    TurnEvaluation,
    VerifierType,
)
from .profiles import DomainProfile
from .sources import json_safe


QA_JUDGE_RUBRIC = """You are a strict QA dataset verifier for the {profile} domain.

Evaluate whether the candidate answer is correct, complete, and supported by the
VISIBLE conversation. The private source is provided for factual checking, but
it must not excuse a question that was unanswerable from the visible conversation.
Compare against the private reference answer without requiring identical wording.
For legal, health, finance, or banking content, penalize unsupported certainty,
missing qualifications, and personalized advice.

Domain profile requirements:
{profile_instructions}

Source-grounding requirement:
{source_grounding_requirement}

Question:
{prompt}

Candidate answer:
{response}

Private reference answer:
{reference}

Visible conversation available to the answer agent:
{visible_context}

Private source used by the questioner and verifier:
{source_context}

Return ONLY this JSON object:
{{
  "accuracy": <0-4>,
  "visible_context_sufficiency": <0-2>,
  "grounding": <0-2>,
  "clarity_and_instruction_following": <0-2>,
  "total_score": <0-10>,
  "reasoning": "<brief explanation>"
}}
total_score must equal the four component scores and therefore use the 0–10
scale. The environment normalizes that total to 0–1 before acceptance.
"""

DEVANAGARI_RE = re.compile(r"[\u0900-\u097f]")
LETTER_RE = re.compile(r"[^\W\d_]", re.UNICODE)
NUMBER_RE = re.compile(r"[-+]?\d+(?:[.,]\d+)?")
NEPALI_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")
ROMANIZED_NEPALI_WORDS = {
    "chha",
    "chhan",
    "cha",
    "ho",
    "huncha",
    "garchha",
    "garna",
    "ko",
    "ka",
    "ki",
    "le",
    "lai",
    "ma",
    "ra",
    "bhaneko",
    "bhane",
    "yo",
    "tyo",
    "kina",
    "kasari",
    "nepal",
    "nepali",
    "uttar",
    "prashna",
}


def normalize_text(value: str) -> str:
    return " ".join(unicodedata.normalize("NFC", value).casefold().split())


class QAVerifier:
    """Route QA turns through structural, deterministic, or judge checks."""

    def __init__(
        self,
        settings: QAGenerationSettings,
        judge_settings: Optional[LLMJudgeSettings] = None,
        inference_service: Any = None,
    ) -> None:
        self.settings = settings
        self.judge_settings = judge_settings
        self.inference_service = inference_service

    def language_issues(self, text: str) -> list[str]:
        language = self.settings.target_language
        if not text.strip():
            return ["empty_text"]
        letters = LETTER_RE.findall(text)
        if language == "ne-Deva":
            ratio = len(DEVANAGARI_RE.findall(text)) / max(1, len(letters))
            if ratio < self.settings.min_devanagari_ratio:
                return [f"insufficient_devanagari_ratio:{ratio:.3f}"]
        elif language == "ne-Latn":
            if DEVANAGARI_RE.search(text):
                return ["unexpected_devanagari_in_romanized_nepali"]
            words = set(re.findall(r"[A-Za-z]+", text.casefold()))
            hits = len(words & ROMANIZED_NEPALI_WORDS)
            if hits < self.settings.min_romanized_nepali_tokens:
                return [f"insufficient_romanized_nepali_tokens:{hits}"]
        return []

    def validate_draft(
        self,
        draft: QuestionDraft,
        source: SourceDocument,
        policy: ContextPolicy,
        subcategory: str,
    ) -> list[str]:
        issues = self.language_issues(draft.question)
        if draft.subcategory != subcategory:
            issues.append("subcategory_mismatch")
        if len(draft.visible_context) > self.settings.max_context_chars:
            issues.append("visible_context_too_long")
        if policy == ContextPolicy.INLINE_EXCERPT:
            if not draft.visible_context:
                issues.append("missing_visible_context")
            else:
                issues.extend(
                    f"visible_context_{issue}"
                    for issue in self.language_issues(draft.visible_context)
                )
            if not draft.evidence:
                issues.append("missing_source_evidence")
            elif not any(
                normalize_text(evidence) in normalize_text(source.text)
                for evidence in draft.evidence
                if evidence.strip()
            ):
                issues.append("evidence_not_found_in_source")
        elif draft.visible_context:
            issues.append("unexpected_visible_context")
        if policy == ContextPolicy.CONVERSATION_GROUNDED and draft.standalone:
            issues.append("followup_incorrectly_marked_standalone")
        if policy != ContextPolicy.CONVERSATION_GROUNDED and not draft.standalone:
            issues.append("initial_question_not_standalone")
        compatible = {
            AnswerType.EXACT: {VerifierType.EXACT, VerifierType.RUBRIC},
            AnswerType.MULTIPLE_CHOICE: {VerifierType.MULTIPLE_CHOICE},
            AnswerType.NUMERIC: {VerifierType.NUMERIC},
            AnswerType.SYMBOLIC: {VerifierType.SYMBOLIC},
            AnswerType.SOURCE_GROUNDED: {VerifierType.SOURCE_GROUNDED},
            AnswerType.RUBRIC: {VerifierType.RUBRIC},
        }
        if draft.verifier not in compatible[draft.answer_type]:
            issues.append("answer_type_verifier_mismatch")
        return issues

    @staticmethod
    def _numbers(text: str) -> list[float]:
        normalized = text.translate(NEPALI_DIGITS)
        values: list[float] = []
        for match in NUMBER_RE.findall(normalized):
            try:
                values.append(float(match.replace(",", "")))
            except ValueError:
                continue
        return values

    def _deterministic_score(
        self, plan: QATurnPlan, answer: str
    ) -> tuple[float, dict[str, Any]]:
        expected = normalize_text(plan.expected_answer)
        candidate = normalize_text(answer)
        if plan.verifier in {VerifierType.EXACT, VerifierType.MULTIPLE_CHOICE}:
            passed = bool(expected and (candidate == expected or expected in candidate))
            return float(passed), {"normalized_expected": expected}
        if plan.verifier == VerifierType.NUMERIC:
            expected_numbers = self._numbers(plan.expected_answer)
            candidate_numbers = self._numbers(answer)
            passed = bool(expected_numbers and candidate_numbers) and any(
                math.isclose(
                    expected_value,
                    candidate_value,
                    rel_tol=self.settings.numeric_tolerance,
                    abs_tol=self.settings.numeric_tolerance,
                )
                for expected_value in expected_numbers
                for candidate_value in candidate_numbers
            )
            return float(passed), {
                "expected_numbers": expected_numbers,
                "candidate_numbers": candidate_numbers,
            }
        return 0.0, {}

    async def evaluate_turn(
        self,
        *,
        plan: QATurnPlan,
        answer: str,
        source: SourceDocument,
        profile: DomainProfile,
        visible_messages: list[dict[str, str]],
    ) -> TurnEvaluation:
        reasons = self.language_issues(answer)
        details: dict[str, Any] = {}
        deterministic = plan.verifier in {
            VerifierType.EXACT,
            VerifierType.MULTIPLE_CHOICE,
            VerifierType.NUMERIC,
        }
        score = 0.0
        if deterministic:
            score, deterministic_details = self._deterministic_score(plan, answer)
            details["deterministic"] = deterministic_details
            if score < 1.0:
                reasons.append("deterministic_verification_failed")

        requires_judge = (
            profile.require_source_grounding
            or not deterministic
            or self.settings.judge_deterministic_answers
        )
        if requires_judge:
            if self.judge_settings is None:
                reasons.append("llm_judge_not_configured")
                score = 0.0
            elif self.inference_service is None:
                reasons.append("inference_service_unavailable_for_judge")
                score = 0.0
            else:
                judge = await LLMJudge(self.judge_settings).score(
                    prompt=plan.question,
                    response=answer,
                    reference=plan.expected_answer,
                    inference_service=self.inference_service,
                    extra_context={
                        "profile": profile.name,
                        "profile_instructions": (
                            "Questioner rules:\n"
                            f"{profile.questioner_instructions}\n\n"
                            "Answerer rules:\n"
                            f"{profile.answerer_instructions}"
                        ),
                        "source_grounding_requirement": (
                            "Required. Reject answers that are not fully supported by "
                            "the visible context, even if they happen to agree with the "
                            "private source."
                            if profile.require_source_grounding
                            else "Apply the context policy and domain rules above."
                        ),
                        "source_context": source.text[: self.settings.max_context_chars],
                        "visible_context": json.dumps(
                            visible_messages, ensure_ascii=False
                        ),
                    },
                )
                details["judge"] = {
                    "raw_score": judge.score,
                    "normalized_score": judge.normalized_score,
                    "score_scale": "0_to_1",
                    "error": judge.error,
                    "parsed": json_safe(judge.parsed),
                }
                if judge.error:
                    reasons.append(f"llm_judge_error:{judge.error}")
                    score = 0.0
                elif deterministic:
                    score = min(score, judge.normalized_score)
                else:
                    score = judge.normalized_score
                if profile.require_source_grounding and not judge.error:
                    grounding = judge.parsed.get("grounding")
                    visible_context_sufficiency = judge.parsed.get(
                        "visible_context_sufficiency"
                    )
                    if not isinstance(grounding, (int, float)) or grounding < 2:
                        reasons.append("source_grounding_below_required")
                    if (
                        not isinstance(visible_context_sufficiency, (int, float))
                        or visible_context_sufficiency < 2
                    ):
                        reasons.append("visible_context_sufficiency_below_required")
                if score < self.settings.acceptance_threshold:
                    reasons.append("judge_score_below_threshold")

        accepted = not reasons and score >= self.settings.acceptance_threshold
        return TurnEvaluation(
            turn_index=plan.turn_index,
            accepted=accepted,
            score=score,
            verifier=plan.verifier,
            reasons=reasons,
            details=details,
        )

    @staticmethod
    def evaluate_conversation(
        plans: list[QATurnPlan],
        evaluations: list[TurnEvaluation],
        expected_turns: int,
    ) -> ConversationEvaluation:
        reasons: list[str] = []
        normalized_questions = [normalize_text(plan.question) for plan in plans]
        if len(set(normalized_questions)) != len(normalized_questions):
            reasons.append("duplicate_questions")
        if len(plans) != expected_turns or len(evaluations) != expected_turns:
            reasons.append("incomplete_conversation")
        for evaluation in evaluations:
            if not evaluation.accepted:
                reasons.append(f"turn_{evaluation.turn_index + 1}_rejected")
        score = min((item.score for item in evaluations), default=0.0)
        return ConversationEvaluation(
            accepted=not reasons,
            score=score,
            reasons=reasons,
            turns=evaluations,
        )


__all__ = ["QAVerifier", "QA_JUDGE_RUBRIC", "normalize_text"]
