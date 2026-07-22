"""Typed models for source-grounded multi-turn QA generation."""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from gymkhana.envs.config import EnvConfig, InferenceConfig


class ContextPolicy(str, Enum):
    """How information needed by an answer is exposed to the answer agent."""

    AUTO = "auto"
    CLOSED_BOOK = "closed_book"
    INLINE_EXCERPT = "inline_excerpt"
    SELF_CONTAINED_PROBLEM = "self_contained_problem"
    CONVERSATION_GROUNDED = "conversation_grounded"


class AnswerType(str, Enum):
    """Shape of the expected answer, independent of the question topic."""

    EXACT = "exact"
    MULTIPLE_CHOICE = "multiple_choice"
    NUMERIC = "numeric"
    SYMBOLIC = "symbolic"
    SOURCE_GROUNDED = "source_grounded"
    RUBRIC = "rubric"


class VerifierType(str, Enum):
    """Verification strategy selected for one generated answer."""

    EXACT = "exact"
    MULTIPLE_CHOICE = "multiple_choice"
    NUMERIC = "numeric"
    SYMBOLIC = "symbolic"
    SOURCE_GROUNDED = "source_grounded"
    RUBRIC = "rubric"


class QAGenerationSettings(BaseModel):
    """Environment-specific controls for QA generation."""

    model_config = ConfigDict(validate_assignment=True)

    profile: str = "textbook"
    subcategory: str = "auto"
    turns: int = Field(default=3, ge=1, le=8)
    target_language: Literal["en", "ne-Deva", "ne-Latn"] = "ne-Deva"
    source_language: Optional[str] = None
    source_license: Optional[str] = None
    source_kind: Literal["auto", "text", "pdf"] = "auto"
    context_policy: ContextPolicy = ContextPolicy.AUTO
    context_mix: Dict[str, float] = Field(default_factory=dict)
    subjects: List[str] = Field(default_factory=list)
    grades: List[str] = Field(default_factory=list)
    difficulty_profile: List[str] = Field(
        default_factory=lambda: ["recall", "application", "analysis"]
    )
    max_question_attempts: int = Field(default=2, ge=1, le=5)
    max_context_chars: int = Field(default=4000, ge=200)
    min_source_chars: int = Field(default=160, ge=1)
    chunk_size_chars: int = Field(default=3500, ge=500)
    chunk_overlap_chars: int = Field(default=250, ge=0)
    max_chunks_per_document: int = Field(default=4, ge=1)
    skip_frontmatter: bool = True
    include_source_text_in_audit: bool = True
    acceptance_threshold: float = Field(default=0.75, ge=0.0, le=1.0)
    judge_deterministic_answers: bool = False
    numeric_tolerance: float = Field(default=1e-6, ge=0.0)
    min_devanagari_ratio: float = Field(default=0.45, ge=0.0, le=1.0)
    min_romanized_nepali_tokens: int = Field(default=2, ge=0)

    @field_validator("profile")
    @classmethod
    def normalize_profile(cls, value: str) -> str:
        normalized = value.strip().lower().replace("-", "_")
        if not normalized:
            raise ValueError("generation.profile cannot be empty")
        return normalized

    @field_validator("difficulty_profile")
    @classmethod
    def validate_difficulty_profile(cls, value: List[str]) -> List[str]:
        cleaned = [item.strip() for item in value if item.strip()]
        if not cleaned:
            raise ValueError("difficulty_profile must contain at least one level")
        return cleaned

    @model_validator(mode="after")
    def validate_chunking_and_mix(self) -> "QAGenerationSettings":
        if self.chunk_overlap_chars >= self.chunk_size_chars:
            raise ValueError("chunk_overlap_chars must be smaller than chunk_size_chars")
        invalid = [key for key, weight in self.context_mix.items() if weight < 0]
        if invalid:
            raise ValueError(f"context_mix weights cannot be negative: {invalid}")
        if self.context_mix and sum(self.context_mix.values()) <= 0:
            raise ValueError("context_mix must have a positive total weight")
        return self


class MultiTurnQAConfig(EnvConfig):
    """Validated configuration for the multi-turn QA environment."""

    questioner_llm: InferenceConfig = Field(default_factory=InferenceConfig)
    generation: QAGenerationSettings = Field(default_factory=QAGenerationSettings)


class SourceDocument(BaseModel):
    """Canonical source unit used to generate one QA conversation."""

    id: str
    text: str
    source: str
    title: Optional[str] = None
    subject: Optional[str] = None
    grade: Optional[str] = None
    language: Optional[str] = None
    license: Optional[str] = None
    jurisdiction: Optional[str] = None
    document_date: Optional[str] = None
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class QuestionDraft(BaseModel):
    """Structured output requested from the questioner agent."""

    question: str = Field(min_length=3)
    visible_context: str = ""
    expected_answer: str = Field(min_length=1)
    answer_type: AnswerType
    verifier: VerifierType
    evidence: List[str] = Field(default_factory=list)
    learning_objective: str = Field(min_length=3)
    subcategory: str = Field(min_length=1)
    standalone: bool = True

    @field_validator(
        "question",
        "visible_context",
        "expected_answer",
        "learning_objective",
        "subcategory",
    )
    @classmethod
    def strip_text(cls, value: str) -> str:
        return value.strip()


class QATurnPlan(BaseModel):
    """Question draft plus the policy selected by the environment."""

    turn_index: int
    difficulty: str
    context_policy: ContextPolicy
    question: str
    user_message: str
    visible_context: str = ""
    expected_answer: str
    answer_type: AnswerType
    verifier: VerifierType
    evidence: List[str] = Field(default_factory=list)
    learning_objective: str
    subcategory: str
    standalone: bool


class TurnEvaluation(BaseModel):
    """Auditable evaluation result for one QA pair."""

    turn_index: int
    accepted: bool
    score: float = Field(ge=0.0, le=1.0)
    verifier: VerifierType
    reasons: List[str] = Field(default_factory=list)
    details: Dict[str, Any] = Field(default_factory=dict)


class ConversationEvaluation(BaseModel):
    """Whole-conversation acceptance decision."""

    accepted: bool
    score: float = Field(ge=0.0, le=1.0)
    reasons: List[str] = Field(default_factory=list)
    turns: List[TurnEvaluation] = Field(default_factory=list)


__all__ = [
    "AnswerType",
    "ContextPolicy",
    "ConversationEvaluation",
    "MultiTurnQAConfig",
    "QAGenerationSettings",
    "QATurnPlan",
    "QuestionDraft",
    "SourceDocument",
    "TurnEvaluation",
    "VerifierType",
]
