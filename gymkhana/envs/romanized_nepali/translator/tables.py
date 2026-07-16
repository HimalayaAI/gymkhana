from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ConsonantRow:
    devanagari: str
    romanized_from_devanagari: str
    romanized_inputs: tuple[str, ...]


CONSONANT_ROWS = (
    ConsonantRow("क", "ka", ("k", "c", "q")),
    ConsonantRow("ख", "kha", ("kh",)),
    ConsonantRow("ग", "ga", ("g",)),
    ConsonantRow("घ", "gha", ("gh",)),
    ConsonantRow("ङ", "na", ("ng",)),
    ConsonantRow("च", "cha", ("ch",)),
    ConsonantRow("छ", "chha", ("chh",)),
    ConsonantRow("ज", "ja", ("j", "z")),
    ConsonantRow("झ", "jha", ("jh",)),
    ConsonantRow("ञ", "na", ("ny",)),
    ConsonantRow("ट", "ta", ("ṭ",)),
    ConsonantRow("ठ", "tha", ("ṭh",)),
    ConsonantRow("ड", "da", ("ḍ",)),
    ConsonantRow("ढ", "dha", ("ḍh",)),
    ConsonantRow("ण", "na", ("n",)),
    ConsonantRow("त", "ta", ("t",)),
    ConsonantRow("थ", "tha", ("th",)),
    ConsonantRow("द", "da", ("d",)),
    ConsonantRow("ध", "dha", ("dh",)),
    ConsonantRow("न", "na", ("n",)),
    ConsonantRow("प", "pa", ("p",)),
    ConsonantRow("फ", "pha", ("ph", "f")),
    ConsonantRow("ब", "ba", ("b",)),
    ConsonantRow("भ", "bha", ("bh",)),
    ConsonantRow("म", "ma", ("m",)),
    ConsonantRow("य", "ya", ("y",)),
    ConsonantRow("र", "ra", ("r",)),
    ConsonantRow("ल", "la", ("l",)),
    ConsonantRow("व", "wa", ("v", "w")),
    ConsonantRow("श", "sha", ("sh", "ś")),
    ConsonantRow("ष", "sha", ("ṣ",)),
    ConsonantRow("स", "sa", ("s",)),
    ConsonantRow("ह", "ha", ("h",)),
)

ROMAN_CONJUNCT_INPUTS = {
    "ksh": "क्ष",
    "gya": "ज्ञ",
    "dny": "ज्ञ",
    "x": "क्स",
}

DEVANAGARI_CONSONANTS_TO_ROMAN = {
    row.devanagari: row.romanized_from_devanagari for row in CONSONANT_ROWS
}
ROMAN_CONSONANTS_TO_DEVANAGARI = {
    romanized: row.devanagari
    for row in CONSONANT_ROWS
    for romanized in row.romanized_inputs
} | ROMAN_CONJUNCT_INPUTS

DEVANAGARI_CONSONANTS = frozenset(DEVANAGARI_CONSONANTS_TO_ROMAN)
ROMAN_CONSONANT_KEYS = tuple(sorted(ROMAN_CONSONANTS_TO_DEVANAGARI, key=len, reverse=True))

DEVANAGARI_INDEPENDENT_VOWELS_TO_ROMAN = {
    "अ": "a",
    "आ": "aa",
    "इ": "i",
    "ई": "i",
    "उ": "u",
    "ऊ": "oo",
    "ए": "e",
    "ऐ": "ai",
    "ओ": "o",
    "औ": "au",
}

ROMAN_INDEPENDENT_VOWELS_TO_DEVANAGARI = {
    "aa": "आ",
    "a": "अ",
    "ai": "ऐ",
    "au": "औ",
    "ee": "ई",
    "ii": "ई",
    "i": "इ",
    "oo": "ऊ",
    "uu": "ऊ",
    "u": "उ",
    "ri": "ऋ",
    "e": "ए",
    "o": "ओ",
}

DEVANAGARI_VOWEL_SIGNS_TO_ROMAN = {
    "ा": "a",
    "ि": "i",
    "ी": "i",
    "ु": "u",
    "ू": "u",
    "े": "e",
    "ै": "ai",
    "ो": "o",
    "ौ": "au",
    "ृ": "ri",
}

ROMAN_VOWEL_SIGNS_TO_DEVANAGARI = {
    "aa": "ा",
    "a": "",
    "ai": "ै",
    "au": "ौ",
    "ee": "ी",
    "ii": "ी",
    "i": "ि",
    "oo": "ू",
    "uu": "ू",
    "u": "ु",
    "ri": "ृ",
    "e": "े",
    "o": "ो",
}

ROMAN_VOWEL_KEYS = tuple(sorted(ROMAN_VOWEL_SIGNS_TO_DEVANAGARI, key=len, reverse=True))
DEVANAGARI_INDEPENDENT_VOWELS = frozenset(DEVANAGARI_INDEPENDENT_VOWELS_TO_ROMAN)
DEVANAGARI_VOWEL_SIGNS = frozenset(DEVANAGARI_VOWEL_SIGNS_TO_ROMAN)

DEVANAGARI_DIACRITICS_TO_ROMAN = {
    "ं": "an",
    "ः": "ah",
    "ँ": "an",
}
DEVANAGARI_DIACRITICS = frozenset(DEVANAGARI_DIACRITICS_TO_ROMAN)

DEVANAGARI_DIGITS_TO_ASCII = {
    "०": "0",
    "१": "1",
    "२": "2",
    "३": "3",
    "४": "4",
    "५": "5",
    "६": "6",
    "७": "7",
    "८": "8",
    "९": "9",
}

ASCII_DIGITS_TO_DEVANAGARI = {
    ascii_digit: devanagari_digit
    for devanagari_digit, ascii_digit in DEVANAGARI_DIGITS_TO_ASCII.items()
}

DEVANAGARI_PUNCTUATION_TO_ASCII = {
    "।": ".",
}

ASCII_PUNCTUATION_TO_DEVANAGARI = {
    ".": "।",
}

DEVANAGARI_TO_ROMAN_CHARACTERS = (
    DEVANAGARI_CONSONANTS_TO_ROMAN
    | DEVANAGARI_INDEPENDENT_VOWELS_TO_ROMAN
    | DEVANAGARI_VOWEL_SIGNS_TO_ROMAN
    | DEVANAGARI_DIACRITICS_TO_ROMAN
    | DEVANAGARI_DIGITS_TO_ASCII
    | DEVANAGARI_PUNCTUATION_TO_ASCII
    | {
        "्": "",
        "ॠ": "ri",
        "ऋ": "ri",
        "ॐ": "om",
    }
)

DEVANAGARI_MARKS = (
    DEVANAGARI_VOWEL_SIGNS
    | DEVANAGARI_DIACRITICS
    | {
        "्",
    }
)

ROMAN_DIACRITIC_TRANSLATION = str.maketrans(
    {
        "ā": "aa",
        "ī": "ii",
        "ū": "uu",
        "ṛ": "ri",
        "ṝ": "rri",
        "ṅ": "ng",
        "ñ": "ny",
        "ś": "sh",
        "ṣ": "sh",
        "ṇ": "n",
    }
)
