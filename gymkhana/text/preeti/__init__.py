"""Preeti (legacy Nepali font) detection and conversion.

Quick use::

    from gymkhana.text.preeti import PreetiFixer, convert_preeti

    convert_preeti("g]kfn ;/sf/")             # 'नेपाल सरकार' (input is all Preeti)
    PreetiFixer().fix("Ao'l6 पार्लर 5 .").text  # 'ब्युटि पार्लर छ ।' (mixed text)

See ``README.md`` in this package for the design and the command line.
"""

from .detect import COUNTER_WORDS, FixResult, PreetiFixer
from .frame import FrameFixResult, fix_frame
from .mapping import PREETI_CHARS, PREETI_COMBOS, convert_preeti
from .vocab import (
    build_english_vocab,
    build_nepali_vocab,
    devanagari_words,
    load_vocab,
    save_vocab,
)

__all__ = [
    "COUNTER_WORDS",
    "FixResult",
    "FrameFixResult",
    "PREETI_CHARS",
    "PREETI_COMBOS",
    "PreetiFixer",
    "build_english_vocab",
    "build_nepali_vocab",
    "convert_preeti",
    "devanagari_words",
    "fix_frame",
    "load_vocab",
    "save_vocab",
]
