from __future__ import annotations

from nepali_romanized_translator import create_translator

from fixtures.transliteration_cases import (
    DEVANAGARI_TO_ROMANIZED_CASES,
    ROMANIZED_TO_DEVANAGARI_CASES,
    ROUND_TRIP_CASES,
)


def assert_equal(name: str, actual: str, expected: str) -> None:
    if actual != expected:
        raise AssertionError(
            f"{name}\nexpected: {expected!r}\nactual:   {actual!r}"
        )


def main() -> None:
    translator = create_translator()
    checks = 0

    for case in DEVANAGARI_TO_ROMANIZED_CASES:
        assert_equal(
            f"devanagari_to_romanized:{case.name}",
            translator.devanagari_to_romanized(case.source),
            case.expected,
        )
        checks += 1

    for case in ROMANIZED_TO_DEVANAGARI_CASES:
        assert_equal(
            f"romanized_to_devanagari:{case.name}",
            translator.romanized_to_devanagari(case.source),
            case.expected,
        )
        checks += 1

    for case in ROUND_TRIP_CASES:
        result = translator.round_trip(case.source)
        assert_equal(f"round_trip:{case.name}", result.output_devanagari, case.expected)
        checks += 1

    print(f"{checks} local transliteration checks passed")


if __name__ == "__main__":
    main()
