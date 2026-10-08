"""Drop training rows that overlap an evaluation set.

Word 5-gram shingles, compared as the share of a text's shingles that also
occur in the eval texts. Tokens are whitespace-separated with punctuation
trimmed, so Devanagari vowel signs and combining marks stay inside words (a
``\\w+`` regex splits Devanagari words at the matras and silently hides overlap).
"""

from __future__ import annotations

import unicodedata
from typing import Any, Iterable, Mapping, Optional, Sequence

N = 5


def _tokens(text: str) -> list[str]:
    words = []
    for raw in unicodedata.normalize("NFC", text).casefold().split():
        word = raw.strip("".join(c for c in raw if unicodedata.category(c)[0] in "PS"))
        if word:
            words.append(word)
    return words


def shingles(text: str, n: int = N) -> set[str]:
    words = _tokens(text)
    if len(words) < n:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i : i + n]) for i in range(len(words) - n + 1)}


def build_index(eval_texts: Iterable[str], n: int = N) -> set[str]:
    index: set[str] = set()
    for text in eval_texts:
        index |= shingles(text, n)
    return index


def overlap(text: str, index: set[str], n: int = N) -> float:
    sh = shingles(text, n)
    return len(sh & index) / len(sh) if sh else 0.0


def filter_rows(
    rows: Iterable[Mapping[str, Any]],
    eval_texts: Sequence[str],
    *,
    threshold: float = 0.5,
    fields: Sequence[str] = ("query", "positive"),
    n: int = N,
    negative_target: Optional[int] = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split rows into (kept, dropped). A row is dropped when any checked field
    overlaps the eval set by ``threshold`` or more; dropped rows get ``decontam``
    metadata naming the field and the share. ``negatives`` entries are checked
    one by one and removed individually instead of dropping the row. When
    ``negative_target`` is given, ``below_target`` is recomputed after pruning
    (a row that falls under the target is flagged, never silently kept as complete)."""
    index = build_index(eval_texts, n)
    kept: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    for row in rows:
        row = dict(row)
        hit = None
        for field in fields:
            value = row.get(field)
            if isinstance(value, str) and overlap(value, index, n) >= threshold:
                hit = {"field": field, "overlap": round(overlap(value, index, n), 3)}
                break
        if hit:
            row["decontam"] = hit
            dropped.append(row)
            continue
        negatives = row.get("negatives")
        if isinstance(negatives, list):
            keep = [i for i, t in enumerate(negatives) if not (isinstance(t, str) and overlap(t, index, n) >= threshold)]
            if len(keep) != len(negatives):
                row["negatives"] = [negatives[i] for i in keep]
                if isinstance(row.get("negative_types"), list):
                    row["negative_types"] = [row["negative_types"][i] for i in keep if i < len(row["negative_types"])]
                row["negatives_pruned"] = len(negatives) - len(keep)
            if negative_target is not None:
                row["below_target"] = len(row["negatives"]) < negative_target
        kept.append(row)
    return kept, dropped


__all__ = ["build_index", "filter_rows", "overlap", "shingles"]
