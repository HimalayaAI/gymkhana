from gymkhana.text.preeti import (
    build_english_vocab,
    build_nepali_vocab,
    devanagari_words,
    load_vocab,
    save_vocab,
)


def test_devanagari_words_skip_numerals() -> None:
    assert devanagari_words("नेपाल। १२३ सरकार") == ["नेपाल", "सरकार"]


def test_build_nepali_vocab_min_freq() -> None:
    assert build_nepali_vocab(["नेपाल सरकार ।", "नेपाल सरकार", "नेपाल"], min_freq=2) == {"नेपाल", "सरकार"}


def test_build_english_vocab_needs_english_dominant_text() -> None:
    english = " ".join(["the office of the municipality and the ward"] * 5)
    assert {"the", "office", "municipality"} <= build_english_vocab([english], min_freq=3)
    assert build_english_vocab(["नेपाल सरकार the"], min_freq=1) == set()


def test_save_load_roundtrip(tmp_path) -> None:
    path = tmp_path / "v.txt"
    save_vocab({"सरकार", "नेपाल"}, path)
    assert load_vocab(path) == {"नेपाल", "सरकार"}
