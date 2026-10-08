"""Apply :class:`PreetiFixer` to a table column and summarize what changed."""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from .detect import PreetiFixer


@dataclass
class FrameFixResult:
    frame: pd.DataFrame
    report: dict[str, Any]
    pairs: list[dict[str, Any]] = field(default_factory=list)


def _context(text: str, needle: str, width: int = 80) -> str:
    i = text.find(needle)
    if i < 0:
        return text[: 2 * width]
    return text[max(0, i - width): i + len(needle) + width]


def fix_frame(
    df: pd.DataFrame,
    fixer: PreetiFixer,
    column: str = "text",
    id_column: str | None = "chunk_id",
    n_pairs: int = 50,
    seed: int = 0,
    top_k: int = 100,
) -> FrameFixResult:
    """Fix ``column`` in place of a copy of ``df``.

    The returned frame keeps the original text in ``<column>_original`` and
    adds ``preeti_touched`` / ``preeti_tokens``. ``pairs`` holds ``n_pairs``
    seeded before/after samples from touched rows for human review.
    """

    texts = df[column].fillna("").tolist()
    fixed: list[str] = []
    n_tokens: list[int] = []
    counts: Counter[tuple[str, str]] = Counter()
    for text in texts:
        result = fixer.fix(text)
        fixed.append(result.text)
        n_tokens.append(len(result.converted))
        counts.update(result.converted)

    out = df.copy()
    out[f"{column}_original"] = df[column]
    out[column] = fixed
    out["preeti_tokens"] = n_tokens
    out["preeti_touched"] = out["preeti_tokens"] > 0

    touched_rows = [i for i, n in enumerate(n_tokens) if n]
    rng = random.Random(seed)
    pairs = []
    for i in sorted(rng.sample(touched_rows, min(n_pairs, len(touched_rows)))):
        converted = fixer.fix(texts[i]).converted
        src, dst = converted[0]
        row_id = df.iloc[i][id_column] if id_column and id_column in df.columns else i
        pairs.append(
            {
                "row": i,
                "id": str(row_id),
                "conversions": [list(c) for c in converted[:20]],
                "before": _context(texts[i], src),
                "after": _context(fixed[i], dst),
            }
        )
    report = {
        "rows": len(df),
        "rows_touched": len(touched_rows),
        "tokens_converted": int(sum(n_tokens)),
        "top_conversions": [[a, b, n] for (a, b), n in counts.most_common(top_k)],
    }
    return FrameFixResult(out, report, pairs)
