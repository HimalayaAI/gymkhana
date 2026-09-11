"""Target-language registry shared by multilingual environments.

Every language-specific behaviour of the environment (prompt instruction,
visible-message labels, and the deterministic script/marker checks that gate
rewards) lives in a :class:`LanguageSpec`. Built-in specs cover English and
Nepali; additional languages are declared in config under
``generation.languages`` and selected with ``generation.target_language``
without touching code.
"""

from __future__ import annotations

import re
from typing import Dict, Mapping, Optional, Set

from pydantic import BaseModel, ConfigDict, Field, field_validator


class LanguageSpec(BaseModel):
    """Prompting and verification rules for one target language + script."""

    model_config = ConfigDict(frozen=True)

    code: str = Field(description="BCP-47 style code, e.g. ne-Deva, hi-Deva, taj-Latn")
    name: str = Field(description="Human-readable name used in prompts")
    instruction: str = Field(
        description="Sentence appended to questioner/answerer system prompts",
    )
    context_label: str = "Context"
    question_label: str = "Question"
    script_regex: Optional[str] = Field(
        default=None,
        description=(
            "Regex matching one letter of the expected script. When set, the share "
            "of letters matching it must reach min_script_ratio."
        ),
    )
    min_script_ratio: float = Field(default=0.45, ge=0.0, le=1.0)
    forbidden_script_regex: Optional[str] = Field(
        default=None,
        description="Regex whose presence rejects the text (e.g. Devanagari in a Latin-script target)",
    )
    marker_words: Set[str] = Field(
        default_factory=set,
        description=(
            "Lowercase high-frequency words of the language. When non-empty, at "
            "least min_marker_words distinct hits are required (Latin-script targets)."
        ),
    )
    min_marker_words: int = Field(default=2, ge=0)

    @field_validator("code")
    @classmethod
    def normalize_code(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("language code cannot be empty")
        return cleaned

    @field_validator("script_regex", "forbidden_script_regex")
    @classmethod
    def compile_check(cls, value: Optional[str]) -> Optional[str]:
        if value is not None:
            re.compile(value)
        return value

    @field_validator("marker_words")
    @classmethod
    def lower_marker_words(cls, value: Set[str]) -> Set[str]:
        return {word.strip().casefold() for word in value if word.strip()}


LETTER_RE = re.compile(r"[^\W\d_]", re.UNICODE)
DEVANAGARI = r"[ऀ-ॿ]"

ROMANIZED_NEPALI_WORDS = {
    "chha", "chhan", "cha", "ho", "huncha", "garchha", "garna", "ko", "ka", "ki",
    "le", "lai", "ma", "ra", "bhaneko", "bhane", "yo", "tyo", "kina", "kasari",
    "nepal", "nepali", "uttar", "prashna",
}

BUILTIN_LANGUAGES: Dict[str, LanguageSpec] = {
    spec.code: spec
    for spec in (
        LanguageSpec(
            code="en",
            name="English",
            instruction="Write clear natural English.",
        ),
        LanguageSpec(
            code="ne-Deva",
            name="Nepali (Devanagari)",
            instruction=(
                "Write natural Nepali in Devanagari. Preserve numbers, formulas, units, "
                "URLs, code, and necessary technical Latin terms."
            ),
            context_label="सन्दर्भ",
            question_label="प्रश्न",
            script_regex=DEVANAGARI,
            min_script_ratio=0.45,
        ),
        LanguageSpec(
            code="ne-Latn",
            name="Nepali (romanized)",
            instruction=(
                "Write natural Nepali transliterated into the Latin alphabet (romanized "
                "Nepali / 'Neplish', e.g. 'malai yo kaam garna parcha'). Never use "
                "Devanagari characters — every character must be Latin. This is Nepali "
                "written in Latin letters, not an English translation. Use one "
                "consistent romanization style and preserve technical tokens."
            ),
            context_label="Sandarbh",
            question_label="Prashna",
            forbidden_script_regex=DEVANAGARI,
            marker_words=ROMANIZED_NEPALI_WORDS,
            min_marker_words=2,
        ),
    )
}


def resolve_language(
    code: str, extra: Optional[Mapping[str, LanguageSpec]] = None
) -> LanguageSpec:
    """Return the spec for ``code``, preferring config-declared specs over built-ins."""

    registry: Dict[str, LanguageSpec] = dict(BUILTIN_LANGUAGES)
    if extra:
        registry.update(extra)
    try:
        return registry[code]
    except KeyError:
        known = ", ".join(sorted(registry))
        raise ValueError(
            f"unknown target_language {code!r}; known: {known}. "
            "Declare new languages under generation.languages."
        ) from None


def language_issues(text: str, spec: LanguageSpec) -> list[str]:
    """Deterministic script/marker checks. Empty list means the text passes."""

    if not text.strip():
        return ["empty_text"]
    if spec.forbidden_script_regex and re.search(spec.forbidden_script_regex, text):
        return [f"forbidden_script:{spec.code}"]
    if spec.script_regex:
        letters = LETTER_RE.findall(text)
        ratio = len(re.findall(spec.script_regex, text)) / max(1, len(letters))
        if ratio < spec.min_script_ratio:
            return [f"insufficient_script_ratio:{spec.code}:{ratio:.3f}"]
    if spec.marker_words:
        words = set(re.findall(r"[^\W\d_]+", text.casefold()))
        hits = len(words & spec.marker_words)
        if hits < spec.min_marker_words:
            return [f"insufficient_marker_words:{spec.code}:{hits}"]
    return []
