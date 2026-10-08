"""Corpus-derived vocabularies that sharpen Preeti-vs-English decisions.

The Nepali vocabulary should come from clean, hand-typed Unicode text (for
the Nepal gov corpus: the HTML pages). PDF extractions carry font damage, and
a vocabulary built from them learns the broken spellings.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Iterable

_DEVA_WORD_RE = re.compile(r"[\u0900-\u0963\u0970-\u097F]+")
_DEVA_NUMERAL_RE = re.compile(r"[\u0964\u0965\u0966-\u096F]+")
_EN_FUNCTION_WORDS = frozenset("the and of to in for is with by on be are this that shall".split())


def devanagari_words(text: str) -> list[str]:
    """Devanagari word tokens, excluding bare dandas and numerals."""

    return [w for w in _DEVA_WORD_RE.findall(text) if not _DEVA_NUMERAL_RE.fullmatch(w)]


def build_nepali_vocab(texts: Iterable[str], min_freq: int = 3) -> set[str]:
    """Nepali words seen at least ``min_freq`` times.

    Build this from clean Unicode text (e.g. HTML pages); garbled PDF
    extractions would teach the vocabulary their broken spellings.
    """

    counts: Counter[str] = Counter()
    for text in texts:
        counts.update(devanagari_words(text))
    return {w for w, c in counts.items() if c >= min_freq}


def build_english_vocab(texts: Iterable[str], min_freq: int = 3) -> set[str]:
    """Lower-cased words from English-dominant texts (by function-word share)."""

    counts: Counter[str] = Counter()
    for text in texts:
        tokens = re.findall(r"[A-Za-z]+", text)
        if len(tokens) < 30:
            continue
        lower = [t.lower() for t in tokens]
        if sum(1 for t in lower if t in _EN_FUNCTION_WORDS) / len(lower) >= 0.12:
            counts.update(lower)
    return {w for w, c in counts.items() if c >= min_freq and len(w) > 1}


def save_vocab(words: Iterable[str], path: str | Path) -> None:
    """One word per line, sorted, UTF-8."""

    Path(path).write_text("\n".join(sorted(set(words))) + "\n", encoding="utf-8")


def load_vocab(path: str | Path) -> set[str]:
    return {w for w in Path(path).read_text(encoding="utf-8").split("\n") if w}
