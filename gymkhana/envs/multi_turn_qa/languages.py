"""Compatibility re-export; the registry now lives in :mod:`gymkhana.envs.languages`."""

from gymkhana.envs.languages import (  # noqa: F401
    BUILTIN_LANGUAGES,
    DEVANAGARI,
    LETTER_RE,
    ROMANIZED_NEPALI_WORDS,
    LanguageSpec,
    language_issues,
    resolve_language,
)

__all__ = [
    "BUILTIN_LANGUAGES",
    "DEVANAGARI",
    "LETTER_RE",
    "ROMANIZED_NEPALI_WORDS",
    "LanguageSpec",
    "language_issues",
    "resolve_language",
]
