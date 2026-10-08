"""Preeti glyph map and whole-string Preeti -> Unicode conversion.

Preeti is a legacy ASCII font: each key renders a Devanagari glyph, the
i-matra (``l``) is typed before its consonant, and the reph (``{``) after
its syllable. :func:`convert_preeti` maps glyphs and then re-orders those
marks into Unicode logical order. It assumes the whole input is Preeti; use
:class:`gymkhana.text.preeti.PreetiFixer` for mixed Unicode/Preeti text.
"""

from __future__ import annotations

import re

# Multi-character glyph combinations, applied before the single-char map.
# ``m`` is a modifier glyph in Preeti that turns the preceding glyph into
# another letter (k+m = फ, e+m = झ, q+m = क्र, p+m = ऊ).
PREETI_COMBOS: tuple[tuple[str, str], ...] = (
    ("qm", "क्र"),
    ("Qm", "क्त"),
    ("km", "फ"),
    ("em", "झ"),
    ("pm", "ऊ"),
    ("O{", "ई"),
)

PREETI_CHARS: dict[str, str] = {
    # lowercase
    "a": "ब", "b": "द", "c": "अ", "d": "म", "e": "भ", "f": "ा", "g": "न",
    "h": "ज", "i": "ष्", "j": "व", "k": "प", "l": "ि", "m": "ं", "n": "ल",
    "o": "य", "p": "उ", "q": "त्र", "r": "च", "s": "क", "t": "त", "u": "ग",
    "v": "ख", "w": "ध", "x": "ह", "y": "थ", "z": "श",
    # uppercase (mostly half forms)
    "A": "ब्", "B": "द्य", "C": "ऋ", "D": "म्", "E": "भ्", "F": "ँ", "G": "न्",
    "H": "ज्", "I": "क्ष्", "J": "व्", "K": "प्", "L": "ी", "M": "ः", "N": "ल्",
    "O": "इ", "P": "ए", "Q": "त्त", "R": "च्", "S": "क्", "T": "त्", "U": "ग्",
    "V": "ख्", "W": "ध्", "X": "ह्", "Y": "थ्", "Z": "श्",
    # digit keys are letters in Preeti
    "0": "ण्", "1": "ज्ञ", "2": "द्द", "3": "घ", "4": "द्ध", "5": "छ", "6": "ट",
    "7": "ठ", "8": "ड", "9": "ढ",
    # shifted digit keys are the numerals
    "!": "१", "@": "२", "#": "३", "$": "४", "%": "५", "^": "६", "&": "७",
    "*": "८", "(": "९", ")": "०",
    # punctuation keys
    "`": "ञ", "~": "ञ्", "-": "(", "_": ")", "+": "ं", "=": ".", "[": "ृ",
    "]": "े", "}": "ै", "\\": "्", "|": "्र", ";": "स", ":": "स्", "'": "ु",
    '"': "ू", ",": ",", ".": "।", "/": "र", "<": "?", ">": "श्र", "?": "रु",
    # extended (Latin-1) glyphs that survive PDF extraction
    "÷": "/", "Ö": "=", "¿": "रू", "Ø": "्य", "å": "द्व", "ß": "द्म",
    "Í": "ङ्क", "Ë": "ङ्ग", "Ì": "न्न", "§": "ट्ट", "ˆ": "फ्", "æ": "“", "Æ": "”",
}

# Placeholder for the reph key ``{`` so it can be re-ordered without being
# confused with a र् that came from ``/\``.
_REPH = "\ue000"
PREETI_CHARS["{"] = _REPH

PREETI_ALPHABET = frozenset(PREETI_CHARS) | frozenset(c for pair in PREETI_COMBOS for c in pair[0])

_CONS = "\u0915-\u0939\u0958-\u095F"
_MATRA = "ािीुूृेैोौ"
_PRE_I_RE = re.compile(rf"\u093F((?:[{_CONS}]\u094D)*[{_CONS}])")
_REPH_RE = re.compile(rf"((?:[{_CONS}]\u094D)*[{_CONS}][{_MATRA}]*[\u0902\u0901]?){_REPH}")
_POST_FIXES: tuple[tuple[str, str], ...] = (
    ("अाे", "ओ"),
    ("अाै", "औ"),
    ("अा", "आ"),
    ("एे", "ऐ"),
    ("ाे", "ो"),
    ("ाै", "ौ"),
)


def convert_preeti(text: str) -> str:
    """Convert a string that is entirely Preeti-encoded into Unicode."""

    out = text
    for src, dst in PREETI_COMBOS:
        out = out.replace(src, dst)
    out = "".join(PREETI_CHARS.get(ch, ch) for ch in out)
    # half form + aa-bar renders as the full consonant
    out = out.replace("्ा", "")
    # the i-matra is typed before the consonant cluster it follows
    out = _PRE_I_RE.sub(r"\1ि", out)
    # the reph is typed after the syllable it sits on
    out = _REPH_RE.sub(r"र्\1", out)
    out = out.replace(_REPH, "र्")
    for src, dst in _POST_FIXES:
        out = out.replace(src, dst)
    return out
