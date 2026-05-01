from __future__ import annotations

import argparse

from .translator import create_best_local_translator


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the best current local no-API Nepali transliterator."
    )
    parser.add_argument(
        "direction",
        choices=("dev-to-roman", "roman-to-dev", "round-trip"),
        help="Transliteration direction.",
    )
    parser.add_argument("text", help="Text to transliterate.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    translator = create_best_local_translator()

    if args.direction == "dev-to-roman":
        print(translator.devanagari_to_romanized(args.text))
    elif args.direction == "roman-to-dev":
        print(translator.romanized_to_devanagari(args.text))
    else:
        result = translator.round_trip(args.text)
        print(f"romanized: {result.romanized}")
        print(f"devanagari: {result.output_devanagari}")


if __name__ == "__main__":
    main()
