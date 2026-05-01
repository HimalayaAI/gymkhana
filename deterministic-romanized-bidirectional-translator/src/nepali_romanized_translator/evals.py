from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Protocol

from .devanagari import DevanagariRomanizer


class Transliterator(Protocol):
    def convert(self, text: str) -> str:
        """Convert text in one transliteration direction."""


@dataclass(frozen=True)
class TransliterationCase:
    name: str
    devanagari: str
    expected_romanized: str


@dataclass(frozen=True)
class TransliterationScore:
    name: str
    expected: str
    actual: str
    exact: bool


DEFAULT_DEVANAGARI_TO_ROMAN_CASES = (
    TransliterationCase(
        name="bank_ai",
        devanagari="बैंकहरुले एआई कसरी प्रयोग गर्न सक्छन्?",
        expected_romanized="bankharule AI kasari prayog garna sakchhan?",
    ),
    TransliterationCase(
        name="nepal",
        devanagari="नेपाल",
        expected_romanized="nepal",
    ),
    TransliterationCase(
        name="customer_support",
        devanagari="ग्राहक सहायता",
        expected_romanized="grahak sahayata",
    ),
)


def score_devanagari_to_roman(
    transliterator: Transliterator,
    cases: Iterable[TransliterationCase] = DEFAULT_DEVANAGARI_TO_ROMAN_CASES,
) -> list[TransliterationScore]:
    scores: list[TransliterationScore] = []
    for case in cases:
        actual = transliterator.convert(case.devanagari)
        scores.append(
            TransliterationScore(
                name=case.name,
                expected=case.expected_romanized,
                actual=actual,
                exact=actual == case.expected_romanized,
            )
        )
    return scores


def main() -> None:
    scores = score_devanagari_to_roman(DevanagariRomanizer())
    passed = sum(1 for score in scores if score.exact)

    for score in scores:
        status = "PASS" if score.exact else "MISS"
        print(f"{status} {score.name}")
        print(f"  expected: {score.expected}")
        print(f"  actual:   {score.actual}")

    print(f"\n{passed}/{len(scores)} exact matches")


if __name__ == "__main__":
    main()
