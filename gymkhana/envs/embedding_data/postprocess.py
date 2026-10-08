"""CLI: verify negatives and decontaminate an exported retrieval JSONL.

    python -m gymkhana.envs.embedding_data.postprocess mine rows.jsonl out.jsonl \\
        --retriever intfloat/multilingual-e5-large-instruct --query-prefix "query: " --doc-prefix "passage: "
    python -m gymkhana.envs.embedding_data.postprocess decontam out.jsonl clean.jsonl --eval-jsonl bench.jsonl

``mine`` needs sentence-transformers (and a GPU for large pools).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .decontam import filter_rows
from .negatives import cross_encoder_scorer, mine_verified_negatives, sentence_transformer_retriever


def _read(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write(path: str, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    mine = sub.add_parser("mine", help="replace negatives with mined, reranker-verified ones")
    mine.add_argument("input")
    mine.add_argument("output")
    mine.add_argument("--retriever", default="intfloat/multilingual-e5-large-instruct")
    mine.add_argument("--query-prefix", default="query: ")
    mine.add_argument("--doc-prefix", default="passage: ")
    mine.add_argument("--reranker", default="BAAI/bge-reranker-v2-m3")
    mine.add_argument("--negatives", type=int, default=5)
    mine.add_argument("--extra-passages", help="jsonl with a 'text' field to enlarge the pool")
    dec = sub.add_parser("decontam", help="drop rows that overlap an eval set")
    dec.add_argument("input")
    dec.add_argument("output")
    dec.add_argument("--eval-jsonl", nargs="+", required=True, help="eval files; every string value is checked")
    dec.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args(argv)

    rows = _read(args.input)
    if args.cmd == "mine":
        extra = [r["text"] for r in _read(args.extra_passages)] if args.extra_passages else []
        out, stats = mine_verified_negatives(
            rows,
            retrieve=sentence_transformer_retriever(args.retriever, args.query_prefix, args.doc_prefix),
            rerank=cross_encoder_scorer(args.reranker),
            extra_passages=extra,
            n_negatives=args.negatives,
        )
        _write(args.output, out)
        print(json.dumps(stats))
    else:
        texts = [v for path in args.eval_jsonl for r in _read(path) for v in r.values() if isinstance(v, str)]
        kept, dropped = filter_rows(rows, texts, threshold=args.threshold, fields=("query", "positive"))
        _write(args.output, kept)
        _write(str(Path(args.output).with_suffix(".dropped.jsonl")), dropped)
        print(json.dumps({"kept": len(kept), "dropped": len(dropped)}))


if __name__ == "__main__":
    main()
