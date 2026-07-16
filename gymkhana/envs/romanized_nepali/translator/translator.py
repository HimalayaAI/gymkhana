from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .devanagari import DevanagariRomanizer
from .romanized import RomanizedDevanagariConverter


class Transliterator(Protocol):
    def convert(self, text: str) -> str:
        """Convert text in one transliteration direction."""


@dataclass(frozen=True)
class RoundTripResult:
    input_devanagari: str
    romanized: str
    output_devanagari: str


class BidirectionalTranslator:
    """Deterministic local Nepali transliteration in both directions."""

    def __init__(
        self,
        *,
        devanagari_to_romanized: Transliterator | None = None,
        romanized_to_devanagari: Transliterator | None = None,
    ) -> None:
        self.devanagari_to_romanized_adapter = (
            devanagari_to_romanized or DevanagariRomanizer()
        )
        self.romanized_to_devanagari_adapter = (
            romanized_to_devanagari or RomanizedDevanagariConverter()
        )

    def devanagari_to_romanized(self, text: str) -> str:
        return self.devanagari_to_romanized_adapter.convert(text)

    def romanized_to_devanagari(self, text: str) -> str:
        return self.romanized_to_devanagari_adapter.convert(text)

    def round_trip(self, text: str) -> RoundTripResult:
        romanized = self.devanagari_to_romanized(text)
        output_devanagari = self.romanized_to_devanagari(romanized)
        return RoundTripResult(
            input_devanagari=text,
            romanized=romanized,
            output_devanagari=output_devanagari,
        )


def create_translator() -> BidirectionalTranslator:
    return BidirectionalTranslator()


# Backwards-compatible aliases for the local development scripts.
LocalBidirectionalTranslator = BidirectionalTranslator
create_best_local_translator = create_translator
