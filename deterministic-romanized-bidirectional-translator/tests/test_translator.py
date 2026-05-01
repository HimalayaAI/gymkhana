from nepali_romanized_translator import (
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
