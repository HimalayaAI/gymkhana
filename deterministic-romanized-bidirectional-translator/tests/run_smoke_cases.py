from __future__ import annotations

from nepali_romanized_translator import create_translator

from fixtures.smoke_cases import SMOKE_CASES


def main() -> None:
    translator = create_translator()
    for index, source in enumerate(SMOKE_CASES, start=1):
        result = translator.round_trip(source)
        print(f"## {index}. {source}")
        print(f"romanized: {result.romanized}")
        print(f"devanagari: {result.output_devanagari}")
        print()


if __name__ == "__main__":
    main()
