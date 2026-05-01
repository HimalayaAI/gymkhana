from __future__ import annotations

from dataclasses import dataclass
import re

from .normalization import (
    cleanup_spacing,
    protect_devanagari_terms,
    protect_existing_roman_terms,
    restore_placeholders,
)
from .tables import (
    DEVANAGARI_CONSONANTS,
    DEVANAGARI_DIACRITICS,
    DEVANAGARI_INDEPENDENT_VOWELS,
    DEVANAGARI_MARKS,
    DEVANAGARI_TO_ROMAN_CHARACTERS,
)


LOAN_WORDS = {
    "रेडियो": "radio",
    "कोट": "coat",
    "डाक्टर": "doctor",
    "स्कुल": "school",
    "क्यान्सर": "cancer",
    "ब्याग": "bag",
    "टिकट": "ticket",
    "सिनेमा": "cinema",
    "फोटो": "photo",
    "टेलिफोन": "telephone",
    "फोन": "phone",
    "क्यामेरा": "camera",
    "कम्प्युटर": "computer",
    "मोटर": "motor",
    "मोबाइल": "mobile",
    "क्यालेन्डर": "calendar",
    "क्याम्पस": "campus",
    "स्टुडियो": "studio",
    "कलेज": "college",
    "पुलिस": "police",
    "इन्जिनियर": "engineer",
    "टुरिस्ट": "tourist",
    "कफी": "coffee",
    "कर्फ्यु": "curfew",
}

HIGH_CONFIDENCE_WORDS = {
    "काठमाडौँ": "kathmandu",
    "काठमाडौं": "kathmandu",
    "गीत": "geet",
    "तर": "tara",
    "मञ्च": "manch",
    "तँ": "ta",
    "संयोजक": "samyojak",
    "प्रशंसा": "prasamsha",
    "संलग्न": "samlagna",
    "वर्ष": "barsha",
    "नम्बर": "number",
    "न.": "No.",
    "नाम": "naam",
    "छैन": "chhaina",
    "दिन": "din",
    "तल": "tala",
    "बाट": "bata",
    "सेवा": "sewa",
    "सुधार": "sudhar",
    "केन्द्र": "kendra",
    "गर्छ": "garcha",
    "गर्छन्": "garchhan",
    "गर्छन": "garchhan",
    "सक्छ": "sakchha",
    "सक्छन्": "sakchhan",
    "सक्छन": "sakchhan",
    "हुन्छ": "hunchha",
    "चाहिन्छ": "chahinchha",
    "खोज्नुहोस्": "khojnuhos",
}

PREDEFINED = {**LOAN_WORDS, **HIGH_CONFIDENCE_WORDS}

DEVANAGARI_WORD_RE = re.compile(r"[\u0900-\u097f.]+")
MIN_SUFFIX_STEM_BASES = 2

SUFFIXES = (
    "द्वारा",
    "देखि",
    "लाई",
    "बाट",
    "हरु",
    "हरू",
    "सँग",
    "संग",
    "स्थित",
    "ले",
    "को",
    "का",
    "की",
    "मा",
    "कै",
)


@dataclass(frozen=True)
class DevanagariRomanizerOptions:
    lowercase: bool = True


class DevanagariRomanizer:
    """Deterministic Devanagari to romanized Nepali converter.

    This is an in-package Python implementation based on source-derived rules,
    high-confidence lexical fixes, and protected Latin terms.
    """

    def __init__(self, options: DevanagariRomanizerOptions | None = None) -> None:
        self.options = options or DevanagariRomanizerOptions()

    def convert(self, text: str) -> str:
        protected, placeholders = protect_devanagari_terms(text)
        protected, roman_placeholders = protect_existing_roman_terms(protected)
        placeholders = {**placeholders, **roman_placeholders}
        romanized = DEVANAGARI_WORD_RE.sub(
            lambda match: self._romanize_word(match.group(0)),
            protected.replace("ज्ञ", "ग्य"),
        )
        if self.options.lowercase:
            romanized = romanized.lower()
        romanized = restore_placeholders(romanized, placeholders)
        return cleanup_spacing(romanized).replace(" .", ".")

    def _romanize_word(self, word: str) -> str:
        if word in PREDEFINED:
            return PREDEFINED[word]

        stem, suffixes = self._split_suffixes(word)
        if stem in PREDEFINED:
            romanized = PREDEFINED[stem]
        else:
            romanized = self._romanize_stem(stem)
        if suffixes:
            romanized += "".join(self._romanize_word(suffix) for suffix in suffixes)
        return romanized

    def _split_suffixes(self, word: str) -> tuple[str, list[str]]:
        suffixes: list[str] = []
        stem = word
        changed = True
        while changed:
            changed = False
            for suffix in SUFFIXES:
                if not stem.endswith(suffix):
                    continue
                candidate = stem[: -len(suffix)]
                if not self._can_split_suffix(candidate):
                    continue
                stem = candidate
                suffixes.insert(0, suffix)
                changed = True
                break
        return stem, suffixes

    def _can_split_suffix(self, stem: str) -> bool:
        if not stem:
            return False
        if stem in PREDEFINED:
            return True
        base_count = sum(char not in DEVANAGARI_MARKS for char in stem)
        return base_count >= MIN_SUFFIX_STEM_BASES

    def _romanize_stem(self, word: str) -> str:
        output: list[str] = []
        length = len(word)
        for index, char in enumerate(word):
            transliterated = DEVANAGARI_TO_ROMAN_CHARACTERS.get(char, char)
            if char in DEVANAGARI_DIACRITICS and index > 0 and (
                word[index - 1] in DEVANAGARI_MARKS
                or word[index - 1] in DEVANAGARI_INDEPENDENT_VOWELS
            ):
                transliterated = "" if index == length - 1 else transliterated[1:]
            if char in DEVANAGARI_CONSONANTS:
                transliterated = self._apply_inherent_vowel_rule(
                    word, char, index, transliterated
                )
            output.append(transliterated)
        return "".join(output)

    def _apply_inherent_vowel_rule(
        self,
        word: str,
        char: str,
        index: int,
        transliterated: str,
    ) -> str:
        if index < len(word) - 1 and word[index + 1] in DEVANAGARI_MARKS:
            if transliterated.endswith("a"):
                return transliterated[:-1]
            return transliterated
        if index == len(word) - 1 and char == "ङ":
            return "ng"
        if index == len(word) - 1 and len(word) > 1:
            if not self._pronounce_final_inherent_vowel(word, char, index):
                if char == "व":
                    transliterated = "va"
                if transliterated.endswith("a"):
                    return transliterated[:-1]
        return transliterated

    def _pronounce_final_inherent_vowel(self, word: str, char: str, index: int) -> bool:
        prev = word[index - 1]
        if char in {"छ", "य", "ह"}:
            return True
        if prev == "्" or prev in DEVANAGARI_DIACRITICS:
            return True
        if prev == "े" and char == "र" and len(word) > 3:
            return True
        if char == "न":
            return True
        return prev in DEVANAGARI_INDEPENDENT_VOWELS
