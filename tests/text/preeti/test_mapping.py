import pytest

from gymkhana.text.preeti import convert_preeti


@pytest.mark.parametrize(
    ("preeti", "unicode"),
    [
        # examples from the Nepal gov corpus
        ("%÷)*)", "५/०८०"),
        ("!@=", "१२."),
        ("-3_", "(घ)"),
        ("Ao'l6", "ब्युटि"),
        ("?=", "रु."),
        # common words
        ("g]kfn", "नेपाल"),
        (";/sf/", "सरकार"),
        ("k|b]z", "प्रदेश"),
        ("Ifdtf", "क्षमता"),
        ("Go\"gtd", "न्यूनतम"),
        ("cleofg", "अभियान"),
    ],
)
def test_glyph_map(preeti: str, unicode: str) -> None:
    assert convert_preeti(preeti) == unicode


@pytest.mark.parametrize(
    ("preeti", "unicode", "rule"),
    [
        ("sfo{qmd", "कार्यक्रम", "reph moves before its syllable; qm is क्र"),
        ("lg0f{o", "निर्णय", "i-matra moves after its cluster; reph"),
        ("a'emfpg'", "बुझाउनु", "em is झ"),
        ("cf]", "ओ", "अ + ा + े composes"),
        ("If", "क्ष", "half form + aa-bar is the full consonant"),
    ],
)
def test_reordering_rules(preeti: str, unicode: str, rule: str) -> None:
    assert convert_preeti(preeti) == unicode, rule
