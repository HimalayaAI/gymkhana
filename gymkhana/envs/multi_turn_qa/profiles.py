"""Domain profiles for the reusable multi-turn QA generator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

from .models import ContextPolicy


@dataclass(frozen=True)
class DomainProfile:
    """Prompt and policy bundle for a QA content domain."""

    name: str
    description: str
    subcategories: Tuple[str, ...]
    questioner_instructions: str
    answerer_instructions: str
    context_mix: Dict[str, float]
    require_source_grounding: bool = False
    jurisdiction: str | None = None


TEXTBOOK = DomainProfile(
    name="textbook",
    description="Grade-appropriate educational QA derived from textbook material.",
    subcategories=(
        "conceptual",
        "factual",
        "math_stem",
        "literature",
        "procedural",
        "source_analysis",
    ),
    questioner_instructions="""
Create pedagogically valuable questions appropriate for the stated subject and
grade. Conceptual questions should test understanding, not copied wording.
Math/STEM problems must include every number, unit, assumption, and formula
needed to solve them. Literature and passage-analysis questions must include a
short relevant excerpt in the visible context. Avoid questions that require an
image or diagram unless all necessary information is represented in text.
""".strip(),
    answerer_instructions="""
Answer as a careful teacher. Match the learner's grade level, explain important
steps, and preserve equations, units, code, and technical terms exactly.
""".strip(),
    context_mix={
        ContextPolicy.CLOSED_BOOK.value: 0.35,
        ContextPolicy.INLINE_EXCERPT.value: 0.35,
        ContextPolicy.SELF_CONTAINED_PROBLEM.value: 0.30,
    },
)


LEGAL = DomainProfile(
    name="legal",
    description="Source-grounded educational QA about Nepalese legal documents.",
    subcategories=(
        "definitions",
        "rights_and_duties",
        "procedure",
        "statutory_interpretation",
        "scenario_application",
    ),
    questioner_instructions="""
Generate educational questions answerable strictly from the supplied legal
text. Preserve section/rule numbers and legal dates. Prefer definitions,
rights, duties, procedures, and restrained hypothetical applications. Never
invent current legal status, amendments, court holdings, or personalized legal
advice. Put the minimum sufficient legal excerpt in visible_context.
""".strip(),
    answerer_instructions="""
Answer only from the visible legal excerpt and prior visible conversation.
Identify the relevant provision when present, distinguish the text from any
inference, and do not present the response as personalized legal advice.
""".strip(),
    context_mix={ContextPolicy.INLINE_EXCERPT.value: 1.0},
    require_source_grounding=True,
    jurisdiction="Nepal",
)


HEALTH = DomainProfile(
    name="health",
    description="Evidence-grounded health education and public-health QA.",
    subcategories=(
        "patient_education",
        "prevention",
        "symptoms",
        "treatment_literacy",
        "public_health",
    ),
    questioner_instructions="""
Create health-literacy questions grounded in the supplied trusted material.
Avoid diagnosis, individualized treatment, unsupported dosage advice, or
claims beyond the source. Include sufficient visible evidence and retain
important warnings, populations, quantities, and dates.
""".strip(),
    answerer_instructions="""
Provide cautious educational information using only the visible evidence.
Preserve safety qualifications and recommend professional or emergency help
when the question itself presents an urgent or individualized situation.
""".strip(),
    context_mix={ContextPolicy.INLINE_EXCERPT.value: 1.0},
    require_source_grounding=True,
)


FINANCE = DomainProfile(
    name="finance",
    description="Financial literacy, accounting, markets, and quantitative QA.",
    subcategories=(
        "financial_literacy",
        "accounting",
        "markets",
        "corporate_finance",
        "quantitative",
    ),
    questioner_instructions="""
Create questions that clearly state currency, dates, rates, assumptions, and
calculation conventions. Source-specific or time-sensitive claims require a
visible excerpt. Do not turn educational examples into personalized investment
recommendations or guarantees.
""".strip(),
    answerer_instructions="""
Show calculations clearly, label assumptions, preserve currencies and dates,
and separate source-grounded facts from general explanation. Do not provide
personalized investment recommendations.
""".strip(),
    context_mix={
        ContextPolicy.INLINE_EXCERPT.value: 0.50,
        ContextPolicy.SELF_CONTAINED_PROBLEM.value: 0.35,
        ContextPolicy.CLOSED_BOOK.value: 0.15,
    },
)


AGRICULTURE = DomainProfile(
    name="agriculture",
    description="Agronomy, livestock, soil, irrigation, and agribusiness QA.",
    subcategories=("crops", "livestock", "soil", "irrigation", "agribusiness"),
    questioner_instructions="""
Create practical agricultural questions while preserving crop, climate,
location, season, dosage, and measurement constraints. Use visible context for
region-specific recommendations and avoid unsafe pesticide improvisation.
""".strip(),
    answerer_instructions="""
Give practical, source-grounded explanations and make environmental,
geographic, seasonal, and safety assumptions explicit.
""".strip(),
    context_mix={
        ContextPolicy.INLINE_EXCERPT.value: 0.55,
        ContextPolicy.CLOSED_BOOK.value: 0.25,
        ContextPolicy.SELF_CONTAINED_PROBLEM.value: 0.20,
    },
)


ECOMMERCE = DomainProfile(
    name="ecommerce",
    description="Marketplace, catalog, logistics, and customer-support QA.",
    subcategories=(
        "catalog",
        "marketplace_operations",
        "customer_support",
        "logistics",
        "policy",
    ),
    questioner_instructions="""
Create realistic operational questions. Product, seller, returns, shipping, or
marketplace-policy questions must expose the relevant policy or scenario facts.
Avoid fabricating company-specific rules.
""".strip(),
    answerer_instructions="""
Answer from the visible scenario and policy details, clearly distinguishing a
general best practice from a company-specific rule.
""".strip(),
    context_mix={
        ContextPolicy.INLINE_EXCERPT.value: 0.55,
        ContextPolicy.CLOSED_BOOK.value: 0.25,
        ContextPolicy.SELF_CONTAINED_PROBLEM.value: 0.20,
    },
)


BANKING = DomainProfile(
    name="banking",
    description="Retail banking, lending, payments, compliance, and risk QA.",
    subcategories=("retail_banking", "lending", "payments", "compliance", "risk"),
    questioner_instructions="""
Create banking questions with explicit rates, fees, dates, currencies, and
policy constraints. Regulatory and institution-specific claims require visible
source context. Avoid personalized credit decisions or claims of guaranteed
approval.
""".strip(),
    answerer_instructions="""
Explain banking concepts and calculations precisely. Use only visible context
for institution-specific or regulatory claims and label assumptions.
""".strip(),
    context_mix={
        ContextPolicy.INLINE_EXCERPT.value: 0.60,
        ContextPolicy.SELF_CONTAINED_PROBLEM.value: 0.30,
        ContextPolicy.CLOSED_BOOK.value: 0.10,
    },
)


GENERAL = DomainProfile(
    name="general",
    description="General-purpose factual, conceptual, procedural, and scenario QA.",
    subcategories=("conceptual", "factual", "procedural", "scenario"),
    questioner_instructions="""
Create clear, useful questions. Use a visible excerpt for claims tied to the
specific source; otherwise make the question fully self-contained.
""".strip(),
    answerer_instructions="Answer accurately, directly, and with enough explanation to teach the concept.",
    context_mix={
        ContextPolicy.CLOSED_BOOK.value: 0.45,
        ContextPolicy.INLINE_EXCERPT.value: 0.35,
        ContextPolicy.SELF_CONTAINED_PROBLEM.value: 0.20,
    },
)


PROFILES = {
    profile.name: profile
    for profile in (TEXTBOOK, LEGAL, HEALTH, FINANCE, AGRICULTURE, ECOMMERCE, BANKING, GENERAL)
}

PROFILE_ALIASES = {
    "domain": "general",
    "textbook_qa": "textbook",
    "legal_qa": "legal",
    "health_qa": "health",
    "finance_qa": "finance",
    "agriculture_qa": "agriculture",
    "e_commerce": "ecommerce",
    "ecommerce_qa": "ecommerce",
    "banking_qa": "banking",
}


def get_profile(name: str) -> DomainProfile:
    """Resolve a profile name or a stable alias."""

    key = name.strip().lower().replace("-", "_")
    key = PROFILE_ALIASES.get(key, key)
    try:
        return PROFILES[key]
    except KeyError as exc:
        supported = ", ".join(sorted(PROFILES))
        raise ValueError(f"unknown QA profile {name!r}; choose one of: {supported}") from exc


def select_qa_subcategory(
    profile: DomainProfile,
    subject: str,
    title: str,
    text: str,
) -> str:
    """Map authoritative source metadata to a QA subtype.

    ``subject`` is never inferred or replaced. The returned value is a separate
    generation strategy such as ``math_stem`` or ``literature``.
    """

    normalized_subject = subject.casefold().replace("-", "_").replace(" ", "_")
    content_signals = " ".join((title, text[:1200])).casefold()
    if profile.name == "textbook":
        if any(
            token in normalized_subject
            for token in ("math", "गणित", "physics", "भौतिक", "chemistry")
        ):
            return "math_stem"
        if any(
            token in content_signals
            for token in ("कथा", "कविता", "story", "poem", "नाटक")
        ):
            return "literature"
        if any(
            token in content_signals
            for token in ("प्रयोग", "विधि", "चरण", "procedure")
        ):
            return "procedural"
        return "conceptual"
    if profile.name == "legal":
        if any(
            token in content_signals
            for token in ("परिभाषा", "सम्झनु", "definition")
        ):
            return "definitions"
        if any(
            token in content_signals
            for token in ("प्रक्रिया", "कार्यविधि", "निवेदन")
        ):
            return "procedure"
        return "statutory_interpretation"
    return profile.subcategories[0]


__all__ = ["DomainProfile", "PROFILES", "get_profile", "select_qa_subcategory"]
