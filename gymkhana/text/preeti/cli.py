"""Command line for the Preeti fixer.

Subcommands::

    # quick check on a string (or stdin)
    python -m gymkhana.text.preeti convert "Ao'l6 पार्लर सञ्चालन 5 ."
    python -m gymkhana.text.preeti convert --whole "g]kfn ;/sf/"

    # build vocabularies once (Nepali from clean rows, e.g. HTML pages)
    python -m gymkhana.text.preeti build-vocab --hf voidash/previllage-nepal-gov-corpus \\
        --hf-config chunks --where document_is_current=1 --nepali-where doc_type=html --out vocab/

    # fix a whole table (parquet / jsonl / HF dataset)
    python -m gymkhana.text.preeti fix --hf voidash/previllage-nepal-gov-corpus --hf-config chunks \\
        --revision <sha> --where document_is_current=1 --vocab-dir vocab/ --output fixed.parquet

``fix`` writes the fixed table (original text in ``<column>_original``),
``<output stem>.report.json`` (counts and top conversions) and
``<output stem>.pairs.jsonl`` (seeded before/after samples for review).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

from .detect import PreetiFixer
from .frame import fix_frame
from .mapping import convert_preeti
from .vocab import build_english_vocab, build_nepali_vocab, load_vocab, save_vocab

logger = logging.getLogger(__name__)

NEPALI_VOCAB_FILE = "nepali.txt"
ENGLISH_VOCAB_FILE = "english.txt"


def _add_source_args(p: argparse.ArgumentParser) -> None:
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--input", help="Input .parquet or .jsonl file")
    src.add_argument("--hf", help="Hugging Face dataset repo id")
    p.add_argument("--hf-config")
    p.add_argument("--split", default="train")
    p.add_argument("--revision", help="Dataset revision (commit sha) for reproducibility")
    p.add_argument("--column", default="text")
    p.add_argument("--where", action="append", default=[], metavar="COL=VALUE",
                   help="Only use rows where COL == VALUE (repeatable)")


def _load(args: argparse.Namespace) -> pd.DataFrame:
    if args.input:
        path = Path(args.input)
        if path.suffix == ".jsonl":
            df = pd.read_json(path, lines=True)
        else:
            df = pd.read_parquet(path)
    else:
        from datasets import load_dataset

        df = load_dataset(args.hf, args.hf_config, split=args.split, revision=args.revision).to_pandas()
    return df[_mask(df, args.where)].reset_index(drop=True)


def _mask(df: pd.DataFrame, conds: list[str]) -> pd.Series:
    m = pd.Series(True, index=df.index)
    for cond in conds:
        col, _, val = cond.partition("=")
        m &= df[col].astype(str) == val
    return m


def _fixer_from_dir(vocab_dir: str | None) -> PreetiFixer:
    if not vocab_dir:
        return PreetiFixer()
    d = Path(vocab_dir)
    return PreetiFixer(load_vocab(d / NEPALI_VOCAB_FILE), load_vocab(d / ENGLISH_VOCAB_FILE))


def cmd_convert(args: argparse.Namespace) -> int:
    text = args.text if args.text is not None else sys.stdin.read()
    if args.whole:
        print(convert_preeti(text))
        return 0
    result = _fixer_from_dir(args.vocab_dir).fix(text)
    print(result.text)
    for src, dst in result.converted:
        print(f"  {src!r} -> {dst!r}", file=sys.stderr)
    return 0


def cmd_build_vocab(args: argparse.Namespace) -> int:
    df = _load(args)
    texts = df[args.column].fillna("")
    nepali = build_nepali_vocab(texts[_mask(df, args.nepali_where)], args.min_freq)
    english = build_english_vocab(texts, args.min_freq)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    save_vocab(nepali, out / NEPALI_VOCAB_FILE)
    save_vocab(english, out / ENGLISH_VOCAB_FILE)
    logger.info("rows=%d nepali=%d english=%d -> %s", len(df), len(nepali), len(english), out)
    return 0


def cmd_fix(args: argparse.Namespace) -> int:
    df = _load(args)
    if args.vocab_dir:
        fixer = _fixer_from_dir(args.vocab_dir)
    else:
        texts = df[args.column].fillna("")
        fixer = PreetiFixer(
            build_nepali_vocab(texts[_mask(df, args.nepali_where)], args.min_freq),
            build_english_vocab(texts, args.min_freq),
        )
    logger.info("rows=%d nepali_vocab=%d english_vocab=%d", len(df), len(fixer.nepali_vocab), len(fixer.english_vocab))
    res = fix_frame(df, fixer, column=args.column, id_column=args.id_column, n_pairs=args.pairs, seed=args.seed)
    res.report.update({"nepali_vocab": len(fixer.nepali_vocab), "english_vocab": len(fixer.english_vocab)})

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    res.frame.to_parquet(out, index=False)
    with open(out.with_suffix(".report.json"), "w", encoding="utf-8") as fh:
        json.dump(res.report, fh, ensure_ascii=False, indent=2)
    with open(out.with_suffix(".pairs.jsonl"), "w", encoding="utf-8") as fh:
        for row in res.pairs:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    logger.info("touched %d/%d rows, %d tokens converted -> %s",
                res.report["rows_touched"], res.report["rows"], res.report["tokens_converted"], out)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m gymkhana.text.preeti", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("convert", help="Fix (or fully convert) one string")
    p.add_argument("text", nargs="?", help="Text to convert; reads stdin if omitted")
    p.add_argument("--whole", action="store_true", help="Treat the whole input as Preeti")
    p.add_argument("--vocab-dir", help="Directory from build-vocab")
    p.set_defaults(func=cmd_convert)

    p = sub.add_parser("build-vocab", help="Build Nepali/English vocabularies from a corpus")
    _add_source_args(p)
    p.add_argument("--nepali-where", action="append", default=[], metavar="COL=VALUE",
                   help="Rows used for the Nepali vocab (use clean Unicode rows, e.g. doc_type=html)")
    p.add_argument("--min-freq", type=int, default=3)
    p.add_argument("--out", required=True, help="Output directory")
    p.set_defaults(func=cmd_build_vocab)

    p = sub.add_parser("fix", help="Fix a text column of a table")
    _add_source_args(p)
    p.add_argument("--output", required=True, help="Output .parquet")
    p.add_argument("--vocab-dir", help="Vocab directory from build-vocab (otherwise built from the input)")
    p.add_argument("--nepali-where", action="append", default=[], metavar="COL=VALUE")
    p.add_argument("--min-freq", type=int, default=3)
    p.add_argument("--id-column", default="chunk_id")
    p.add_argument("--pairs", type=int, default=50, help="Before/after samples to write")
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_fix)
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
