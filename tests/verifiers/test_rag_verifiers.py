"""Offline tests for reusable external RAG verifiers."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pytest
from pydantic import BaseModel, Field

from gymkhana.core.services.inference import InferenceService
from gymkhana.core.services.inference.pydantic_ai import PydanticAIInferenceService
from gymkhana.envs.config import LLMJudgeSettings
from gymkhana.verifiers.rag import (
    ClaimVerdict,
    ContextPrecisionVerifier,
    ContextRelevanceJudgment,
    ContextRelevanceVerifier,
    ContextVerdict,
    FaithfulnessJudgment,
    FaithfulnessVerifier,
    GroundednessJudgment,
    GroundednessLevel,
    GroundednessVerifier,
    RAGSample,
    RelevanceLevel,
    ResponseRelevanceJudgment,
    ResponseRelevanceVerifier,
)


class ScriptedInferenceService(InferenceService):
    """Return preconstructed structured outputs without a provider call."""

    outputs: List[Any]
    requests: List[Dict[str, Any]] = Field(default_factory=list)

    async def generate(self, *, messages: List[Dict[str, str]], **kwargs: Any) -> str:
        raise AssertionError("RAG verifiers must use generate_structured")

    async def batch_generate(
        self,
        *,
        prompts: List[str],
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> List[str]:
        raise AssertionError("RAG verifiers must use generate_structured")

    async def generate_structured(
        self,
        *,
        messages: List[Dict[str, str]],
        output_type: type[BaseModel],
        **kwargs: Any,
    ) -> BaseModel:
        self.requests.append(
            {"messages": messages, "output_type": output_type, **kwargs}
        )
        output = self.outputs.pop(0)
        if isinstance(output, Exception):
            raise output
        return output_type.model_validate(output)


@pytest.fixture
def settings() -> LLMJudgeSettings:
    return LLMJudgeSettings(model="test:external-judge", temperature=0.0)


@pytest.fixture
def sample() -> RAGSample:
    return RAGSample(
        question="What does TAAN represent?",
        answer="TAAN represents trekking agencies in Nepal and was founded in 1978.",
        contexts=[
            "TAAN is the umbrella body representing trekking agencies in Nepal.",
            "Nepal contains eight of the world's fourteen highest mountains.",
            "TAAN publishes guidance for member trekking agencies.",
        ],
    )


@pytest.mark.asyncio
async def test_faithfulness_score_is_computed_from_claim_verdicts(
    settings: LLMJudgeSettings, sample: RAGSample
) -> None:
    service = ScriptedInferenceService(
        outputs=[
            FaithfulnessJudgment(
                claims=[
                    ClaimVerdict(
                        claim="TAAN represents trekking agencies in Nepal.",
                        supported=True,
                        evidence=["representing trekking agencies in Nepal"],
                        reason="Directly stated.",
                    ),
                    ClaimVerdict(
                        claim="TAAN was founded in 1978.",
                        supported=False,
                        reason="No founding date is supplied.",
                    ),
                ]
            )
        ]
    )
    verifier = FaithfulnessVerifier(
        settings=settings, inference_service=service, threshold=1.0
    )

    result = await verifier.verify(sample)

    assert result.metric == "faithfulness"
    assert result.score == pytest.approx(0.5)
    assert result.passed is False
    assert result.details["supported_claims"] == 1
    assert result.details["total_claims"] == 2
    assert "reward_score" not in FaithfulnessJudgment.model_json_schema()["properties"]
    assert service.requests[0]["output_type"] is FaithfulnessJudgment


@pytest.mark.asyncio
async def test_faithfulness_rejects_empty_or_failed_judgments(
    settings: LLMJudgeSettings, sample: RAGSample
) -> None:
    no_claims = FaithfulnessVerifier(
        settings=settings,
        inference_service=ScriptedInferenceService(
            outputs=[FaithfulnessJudgment(claims=[])]
        ),
    )
    failed = FaithfulnessVerifier(
        settings=settings,
        inference_service=ScriptedInferenceService(
            outputs=[RuntimeError("judge unavailable")]
        ),
    )

    empty_result = await no_claims.verify(sample)
    failed_result = await failed.verify(sample)

    assert empty_result.score == 0.0
    assert empty_result.error == "judge returned no factual claims"
    assert failed_result.score == 0.0
    assert failed_result.error == "RuntimeError: judge unavailable"


@pytest.mark.asyncio
async def test_groundedness_maps_external_label_in_trusted_code(
    settings: LLMJudgeSettings, sample: RAGSample
) -> None:
    service = ScriptedInferenceService(
        outputs=[
            GroundednessJudgment(
                level=GroundednessLevel.PARTIALLY_GROUNDED,
                evidence=["representing trekking agencies in Nepal"],
                reason="The representation claim is supported but the date is not.",
            )
        ]
    )
    verifier = GroundednessVerifier(
        settings=settings, inference_service=service, threshold=1.0
    )

    result = await verifier.verify(sample)

    assert result.metric == "groundedness"
    assert result.score == 0.5
    assert result.passed is False
    assert result.details["level"] == "partially_grounded"
    assert "score" not in GroundednessJudgment.model_json_schema()["properties"]


@pytest.mark.asyncio
async def test_response_relevance_is_distinct_from_grounding(
    settings: LLMJudgeSettings, sample: RAGSample
) -> None:
    service = ScriptedInferenceService(
        outputs=[
            ResponseRelevanceJudgment(
                level=RelevanceLevel.FULLY_RELEVANT,
                reason="The answer directly states what TAAN represents.",
            )
        ]
    )
    verifier = ResponseRelevanceVerifier(
        settings=settings, inference_service=service
    )

    result = await verifier.verify(sample)

    assert result.metric == "response_relevance"
    assert result.score == 1.0
    assert result.passed is True
    request_payload = service.requests[0]["messages"][0]["content"]
    assert "contexts" not in request_payload


@pytest.mark.asyncio
async def test_context_relevance_scores_fraction_of_relevant_contexts(
    settings: LLMJudgeSettings, sample: RAGSample
) -> None:
    service = ScriptedInferenceService(
        outputs=[
            ContextRelevanceJudgment(
                verdicts=[
                    ContextVerdict(
                        context_index=2, relevant=True, reason="Provides supporting detail."
                    ),
                    ContextVerdict(
                        context_index=0, relevant=True, reason="Directly answers the question."
                    ),
                    ContextVerdict(
                        context_index=1, relevant=False, reason="Only discusses mountains."
                    ),
                ]
            )
        ]
    )
    verifier = ContextRelevanceVerifier(
        settings=settings, inference_service=service, threshold=0.6
    )

    result = await verifier.verify(sample)

    assert result.score == pytest.approx(2 / 3)
    assert result.passed is True
    assert [item["context_index"] for item in result.details["contexts"]] == [0, 1, 2]


@pytest.mark.asyncio
async def test_context_precision_rewards_relevant_contexts_earlier(
    settings: LLMJudgeSettings, sample: RAGSample
) -> None:
    judgment = ContextRelevanceJudgment(
        verdicts=[
            ContextVerdict(context_index=0, relevant=True, reason="Relevant."),
            ContextVerdict(context_index=1, relevant=False, reason="Irrelevant."),
            ContextVerdict(context_index=2, relevant=True, reason="Relevant."),
        ]
    )
    verifier = ContextPrecisionVerifier(
        settings=settings,
        inference_service=ScriptedInferenceService(outputs=[judgment]),
    )

    result = await verifier.verify(sample)

    assert result.score == pytest.approx((1.0 + 2 / 3) / 2)
    assert result.details["relevant_at_k"] == [True, False, True]


@pytest.mark.asyncio
async def test_context_verifier_rejects_missing_or_duplicate_indices(
    settings: LLMJudgeSettings, sample: RAGSample
) -> None:
    duplicate = ContextRelevanceJudgment(
        verdicts=[
            ContextVerdict(context_index=0, relevant=True, reason="Relevant."),
            ContextVerdict(context_index=0, relevant=False, reason="Duplicate."),
            ContextVerdict(context_index=2, relevant=True, reason="Relevant."),
        ]
    )
    verifier = ContextRelevanceVerifier(
        settings=settings,
        inference_service=ScriptedInferenceService(outputs=[duplicate]),
    )

    result = await verifier.verify(sample)

    assert result.score == 0.0
    assert result.passed is False
    assert result.error == "judge returned duplicate context indices"


@pytest.mark.asyncio
async def test_pydantic_ai_structured_generation_uses_output_model() -> None:
    from pydantic_ai.models.test import TestModel

    class StructuredAnswer(BaseModel):
        supported: bool

    service = PydanticAIInferenceService()
    result = await service.generate_structured(
        messages=[{"role": "user", "content": "Evaluate this."}],
        output_type=StructuredAnswer,
        model=TestModel(custom_output_args={"supported": True}),
    )

    assert result == StructuredAnswer(supported=True)


def test_rag_sample_rejects_blank_contexts() -> None:
    with pytest.raises(ValueError, match="blank entries"):
        RAGSample(question="Question?", contexts=["valid", "  "])


def test_verifier_rejects_invalid_threshold(settings: LLMJudgeSettings) -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        FaithfulnessVerifier(
            settings=settings,
            inference_service=ScriptedInferenceService(outputs=[]),
            threshold=1.1,
        )
