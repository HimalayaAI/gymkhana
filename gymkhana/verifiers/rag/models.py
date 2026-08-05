"""Typed contracts for retrieval-augmented generation verification."""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from pydantic import BaseModel, Field, field_validator


class RAGSample(BaseModel):
    """Inputs shared by built-in RAG metrics.

    ``reference_answer`` is optional because faithfulness and context relevance
    do not require ground-truth answers.
    """

    question: str = Field(min_length=1)
    answer: str = ""
    contexts: List[str] = Field(min_length=1)
    reference_answer: Optional[str] = None

    @field_validator("question", "answer", "reference_answer")
    @classmethod
    def strip_text(cls, value: Optional[str]) -> Optional[str]:
        return value.strip() if value is not None else None

    @field_validator("contexts")
    @classmethod
    def validate_contexts(cls, values: List[str]) -> List[str]:
        normalized = [value.strip() for value in values]
        if any(not value for value in normalized):
            raise ValueError("contexts must not contain blank entries")
        return normalized


class ClaimVerdict(BaseModel):
    """External judge decision for one atomic factual claim."""

    claim: str = Field(min_length=1)
    supported: bool
    evidence: List[str] = Field(default_factory=list)
    reason: str = Field(min_length=1)


class FaithfulnessJudgment(BaseModel):
    """Structured judge output; the trusted score is computed separately."""

    claims: List[ClaimVerdict]


class ContextVerdict(BaseModel):
    """External judge decision for one retrieved context at its original rank."""

    context_index: int = Field(ge=0)
    relevant: bool
    reason: str = Field(min_length=1)


class ContextRelevanceJudgment(BaseModel):
    """Structured relevance judgments for every retrieved context."""

    verdicts: List[ContextVerdict]


class RelevanceLevel(str, Enum):
    """Coarse response relevance labels mapped to trusted numeric scores."""

    IRRELEVANT = "irrelevant"
    PARTIALLY_RELEVANT = "partially_relevant"
    FULLY_RELEVANT = "fully_relevant"


class ResponseRelevanceJudgment(BaseModel):
    """Holistic external judgment of whether an answer addresses its question."""

    level: RelevanceLevel
    reason: str = Field(min_length=1)


class GroundednessLevel(str, Enum):
    """Coarse evidence-grounding labels mapped to trusted numeric scores."""

    UNGROUNDED = "ungrounded"
    PARTIALLY_GROUNDED = "partially_grounded"
    FULLY_GROUNDED = "fully_grounded"


class GroundednessJudgment(BaseModel):
    """Token-efficient holistic grounding judgment."""

    level: GroundednessLevel
    evidence: List[str] = Field(default_factory=list)
    reason: str = Field(min_length=1)


class RAGMetricResult(BaseModel):
    """Auditable, normalized output shared by all built-in RAG metrics."""

    metric: str
    score: float = Field(ge=0.0, le=1.0)
    passed: bool
    threshold: float = Field(ge=0.0, le=1.0)
    details: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None


@runtime_checkable
class RAGVerifier(Protocol):
    """Common asynchronous interface implemented by built-in RAG metrics."""

    async def verify(self, sample: RAGSample) -> RAGMetricResult:
        ...


__all__ = [
    "ClaimVerdict",
    "ContextRelevanceJudgment",
    "ContextVerdict",
    "FaithfulnessJudgment",
    "GroundednessJudgment",
    "GroundednessLevel",
    "RAGMetricResult",
    "RAGSample",
    "RAGVerifier",
    "RelevanceLevel",
    "ResponseRelevanceJudgment",
]
