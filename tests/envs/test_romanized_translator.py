import pytest

from envs.fixtures.romanized_smoke_cases import SMOKE_CASES
from envs.fixtures.romanized_translator_cases import (
    DEVANAGARI_TO_ROMANIZED_CASES,
    ROMANIZED_TO_DEVANAGARI_CASES,
    ROUND_TRIP_CASES,
)
from gymkhana.envs.romanized_nepali.translator import (
    DevanagariRomanizer,
    RomanizedDevanagariConverter,
    create_translator,
)


def test_devanagari_to_romanized() -> None:
    romanizer = DevanagariRomanizer()

    assert romanizer.convert("एआई प्रयोग") == "AI prayog"
    assert romanizer.convert("ग्राहक सेवा सुधार गर्न सक्छ") == (
        "grahak sewa sudhar garna sakchha"
    )


def test_romanized_to_devanagari() -> None:
    converter = RomanizedDevanagariConverter()

    assert converter.convert("bank le AI prayog garna sakcha") == (
        "बैंक ले AI प्रयोग गर्न सक्छ"
    )


def test_bidirectional_round_trip() -> None:
    translator = create_translator()
    result = translator.round_trip("ग्राहक सेवा सुधार गर्न सक्छ")

    assert result.romanized == "grahak sewa sudhar garna sakchha"
    assert result.output_devanagari == "ग्राहक सेवा सुधार गर्न सक्छ"


@pytest.mark.parametrize("case", DEVANAGARI_TO_ROMANIZED_CASES)
def test_devanagari_regression_fixtures(case) -> None:
    actual = create_translator().devanagari_to_romanized(case.source)
    assert actual == case.expected, case.name


@pytest.mark.parametrize("case", ROMANIZED_TO_DEVANAGARI_CASES)
def test_romanized_regression_fixtures(case) -> None:
    actual = create_translator().romanized_to_devanagari(case.source)
    assert actual == case.expected, case.name


@pytest.mark.parametrize("case", ROUND_TRIP_CASES)
def test_round_trip_regression_fixtures(case) -> None:
    actual = create_translator().round_trip(case.source).output_devanagari
    assert actual == case.expected, case.name


@pytest.mark.parametrize("source", SMOKE_CASES)
def test_round_trip_smoke_cases(source: str) -> None:
    result = create_translator().round_trip(source)
    assert result.romanized
    assert result.output_devanagari
