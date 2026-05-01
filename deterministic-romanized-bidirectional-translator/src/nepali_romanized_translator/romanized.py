from __future__ import annotations

from dataclasses import dataclass
import re

from .normalization import (
    cleanup_spacing,
    protect_roman_terms,
    protect_romanized_overrides,
)
from .tables import (
    ASCII_DIGITS_TO_DEVANAGARI,
    ASCII_PUNCTUATION_TO_DEVANAGARI,
    ROMAN_CONSONANT_KEYS,
    ROMAN_CONSONANTS_TO_DEVANAGARI,
    ROMAN_DIACRITIC_TRANSLATION,
    ROMAN_INDEPENDENT_VOWELS_TO_DEVANAGARI,
    ROMAN_VOWEL_KEYS,
    ROMAN_VOWEL_SIGNS_TO_DEVANAGARI,
)

ROMAN_WORD_PATTERN = r"[A-Za-zāīūṛṝṅñṭḍṇśṣ]+(?:[-'][A-Za-zāīūṛṝṅñṭḍṇśṣ]+)*"
ROMAN_TOKEN_RE = re.compile(rf"\{{[^{{}}]*\}}|{ROMAN_WORD_PATTERN}|\d+|[^\w\s]|\s+")
ROMAN_WORD_RE = re.compile(ROMAN_WORD_PATTERN)


@dataclass(frozen=True)
class RomanizedDevanagariOptions:
    preserve_ascii_digits: bool = True
    use_nepali_danda: bool = True


class RomanizedDevanagariConverter:
    """Deterministic romanized Nepali to Devanagari converter.

    The converter is deliberately layered:

    1. High-confidence phrase and word overrides.
    2. Protected Latin technical/product/domain terms.
    3. A greedy phonetic fallback for the long tail.

    This avoids the old Node subprocess boundary while keeping output
    predictable enough for regression fixtures.
    """

    def __init__(self, options: RomanizedDevanagariOptions | None = None) -> None:
        self.options = options or RomanizedDevanagariOptions()
        self._consonant_keys = ROMAN_CONSONANT_KEYS
        self._vowel_keys = ROMAN_VOWEL_KEYS

    def convert(self, text: str) -> str:
        protected, placeholders = protect_romanized_overrides(text)
        protected = protect_roman_terms(protected)
        converted = "".join(
            self._convert_token(token) for token in self._tokens(protected)
        )
        for placeholder, value in sorted(
            placeholders.items(), key=lambda item: -len(item[0])
        ):
            converted = converted.replace(placeholder, value)
        return cleanup_spacing(converted)

    def _tokens(self, text: str) -> list[str]:
        tokens = ROMAN_TOKEN_RE.findall(text)
        if not tokens:
            return [text] if text else []
        return tokens

    def _convert_token(self, token: str) -> str:
        if token.startswith("{") and token.endswith("}"):
            return token[1:-1]
        if token.isspace():
            return token
        if token.isdigit():
            if self.options.preserve_ascii_digits:
                return token
            return "".join(ASCII_DIGITS_TO_DEVANAGARI.get(char, char) for char in token)
        if token in ASCII_PUNCTUATION_TO_DEVANAGARI and self.options.use_nepali_danda:
            return ASCII_PUNCTUATION_TO_DEVANAGARI[token]
        if ROMAN_WORD_RE.fullmatch(token):
            return self._convert_word(token)
        return token

    def _convert_word(self, word: str) -> str:
        pieces = re.split(r"([-'])", word)
        return "".join(
            self._convert_plain_word(piece) if piece not in "-'" else piece
            for piece in pieces
        )

    def _convert_plain_word(self, word: str) -> str:
        if not word:
            return word
        normalized = self._normalize_roman_word(word)
        output: list[str] = []
        index = 0
        while index < len(normalized):
            vowel = self._match_vowel(normalized, index)
            if vowel:
                output.append(ROMAN_INDEPENDENT_VOWELS_TO_DEVANAGARI[vowel])
                index += len(vowel)
                continue

            consonant = self._match_consonant(normalized, index)
            if consonant is None:
                output.append(normalized[index])
                index += 1
                continue

            devanagari = ROMAN_CONSONANTS_TO_DEVANAGARI[consonant]
            index += len(consonant)
            vowel = self._match_vowel(normalized, index)
            if vowel:
                output.append(devanagari + ROMAN_VOWEL_SIGNS_TO_DEVANAGARI[vowel])
                index += len(vowel)
            elif self._starts_with_consonant(normalized, index):
                output.append(devanagari + "्")
            else:
                output.append(devanagari)
        return "".join(output)

    def _normalize_roman_word(self, word: str) -> str:
        return word.lower().translate(ROMAN_DIACRITIC_TRANSLATION)

    def _match_vowel(self, word: str, index: int) -> str | None:
        for vowel in self._vowel_keys:
            if word.startswith(vowel, index):
                return vowel
        return None

    def _match_consonant(self, word: str, index: int) -> str | None:
        for consonant in self._consonant_keys:
            if word.startswith(consonant, index):
                return consonant
        return None

    def _starts_with_consonant(self, word: str, index: int) -> bool:
        if index >= len(word):
            return False
        return self._match_consonant(word, index) is not None
