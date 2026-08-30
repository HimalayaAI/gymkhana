"""Push Gymkhana ShareGPT exports to the Hugging Face Hub.

Every Gymkhana environment exports ShareGPT rows in Hermes format (a system turn
with ``<tools>`` when tools exist, assistant turns with ``<think>`` and
``<tool_call>`` blocks). This script merges one or more export files, keeps
``id`` + ``conversations``, flattens ``source_provenance`` into top-level
columns, JSON-encodes structured extras (``tools``, ``expected_tool_calls``,
lists/dicts), drops nothing else, and pushes with a dataset card.

Usage:
    python scripts/push_sharegpt_to_hub.py \
        --input outputs/.../export.jsonl [--input .../pass2/export.jsonl] \
        --repo himalaya-ai/hermes-function-calling-nepali \
        --source-dataset NousResearch/hermes-function-calling-v1 \
        --environment multilingual-tool-use [--private] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

TOOLS_RE = re.compile(r"<tools>\s*(.*?)\s*</tools>", re.DOTALL)
PROVENANCE_KEY = "source_provenance"
LEADING_COLUMNS = ("id", "conversations", "tools", "category", "subcategory", "task")


def tools_from_system_prompt(conversations: List[Dict[str, str]]) -> Optional[List[Any]]:
    """Recover the tool list from the Hermes ``<tools>`` block (first JSON list wins)."""
    system = next((m.get("value", "") for m in conversations if m.get("from") == "system"), "")
    for match in TOOLS_RE.finditer(system):
        try:
            parsed = json.loads(match.group(1))
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, list):
            # Hermes prompts store the bare function objects; restore the OpenAI
            # ``{"type": "function", "function": {...}}`` shape used by source datasets.
            return [
                tool if isinstance(tool, dict) and tool.get("type") == "function" else {"type": "function", "function": tool}
                for tool in parsed
            ]
    return None


HUB_API = "https://huggingface.co/api"


def resolve_token(explicit: Optional[str]) -> str:
    """CLI flag, then $HF_TOKEN, then the cached `hf auth login` token."""
    if explicit:
        return explicit
    env = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if env:
        return env
    cached = Path.home() / ".cache" / "huggingface" / "token"
    if cached.exists():
        return cached.read_text(encoding="utf-8").strip()
    raise SystemExit("no Hugging Face token: pass --hf-token, set HF_TOKEN, or run `hf auth login`")


def create_repo_rest(repo: str, *, token: str, private: bool) -> None:
    import requests

    owner, _, name = repo.partition("/")
    response = requests.post(
        f"{HUB_API}/repos/create",
        headers={"Authorization": f"Bearer {token}"},
        json={"name": name, "organization": owner, "type": "dataset", "private": private},
        timeout=60,
    )
    if response.status_code == 409 or "already created" in response.text:
        print(f"[push] repo {repo} already exists")
        return
    response.raise_for_status()
    print(f"[push] created {repo}")


def commit_files_rest(
    repo: str,
    *,
    token: str,
    files: Dict[str, bytes],
    summary: str,
    delete: Optional[List[str]] = None,
) -> None:
    """Single commit over the Hub's NDJSON commit endpoint (no LFS, no xet)."""
    import base64

    import requests

    lines = [json.dumps({"key": "header", "value": {"summary": summary, "description": ""}})]
    for path in delete or []:
        lines.append(json.dumps({"key": "deletedFile", "value": {"path": path}}))
    for path, blob in files.items():
        lines.append(
            json.dumps(
                {
                    "key": "file",
                    "value": {
                        "path": path,
                        "content": base64.b64encode(blob).decode(),
                        "encoding": "base64",
                    },
                }
            )
        )
    response = requests.post(
        f"{HUB_API}/datasets/{repo}/commit/main",
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/x-ndjson"},
        data="\n".join(lines).encode("utf-8"),
        timeout=600,
    )
    response.raise_for_status()
    print(f"[push] commit {response.json().get('commitOid', '')[:10]} ({', '.join(files)})")


def flatten(row: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"id": row["id"], "conversations": row["conversations"]}
    provenance = row.get(PROVENANCE_KEY) or {}
    for key, value in provenance.items():
        if key == "id":
            continue
        out.setdefault(key, value)
    tools = row.get("tools")
    if not tools:
        tools = tools_from_system_prompt(row["conversations"])
    if tools is not None:
        out["tools"] = json.dumps(tools, ensure_ascii=False)
    for key, value in row.items():
        if key in ("id", "conversations", PROVENANCE_KEY, "tools") or key in out:
            continue
        out[key] = json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value
    ordered = {k: out[k] for k in LEADING_COLUMNS if k in out}
    ordered.update({k: v for k, v in out.items() if k not in ordered})
    return ordered


def align_columns(splits: Dict[str, List[Dict[str, Any]]]) -> None:
    """Give every row the same keys, in the same order, filling gaps with None.

    Exports written before and after a metadata change carry different columns;
    the Hub infers features from the first split and fails to cast the rest
    ("column names don't match"). Aligning up front keeps the splits castable.
    """
    ordered: List[str] = []
    for rows in splits.values():
        for row in rows:
            for key in row:
                if key not in ordered:
                    ordered.append(key)
    lead = [c for c in LEADING_COLUMNS if c in ordered]
    ordered = lead + [c for c in ordered if c not in lead]
    for split, rows in splits.items():
        splits[split] = [{key: row.get(key) for key in ordered} for row in rows]


def merge(paths: Iterable[Path]) -> List[Dict[str, Any]]:
    """Merge exports, keeping the first row per (id, target_language).

    The same source row localized into two scripts is two distinct examples, so
    ``target_language`` is part of the key — deduping on ``id`` alone would drop
    every variant after the first.
    """
    rows: Dict[tuple, Dict[str, Any]] = {}
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = flatten(json.loads(line))
                    key = (str(row["id"]), row.get("target_language"))
                    rows.setdefault(key, row)
    return list(rows.values())


def build_card(
    *, repo: str, rows: List[Dict[str, Any]], splits: Dict[str, List[Dict[str, Any]]], args: argparse.Namespace
) -> str:
    columns = sorted({k for row in rows for k in row})
    split_yaml = "\n".join(
        f"      - split: {split}\n        path: data/{split}.jsonl" for split in splits
    )
    configs_block = (
        "configs:\n  - config_name: default\n    data_files:\n" + split_yaml + "\n"
    )
    split_lines = "\n".join(f"- `{split}`: {len(r)} rows" for split, r in splits.items())
    size = "n<1K" if len(rows) < 1000 else "1K<n<10K" if len(rows) < 10000 else "10K<n<100K"
    languages = "\n".join(f"- {lang}" for lang in args.language)
    tags = "\n".join(f"- {tag}" for tag in args.tag)
    source = (
        f"Derived from [{args.source_dataset}](https://huggingface.co/datasets/{args.source_dataset})."
        if args.source_dataset
        else ""
    )
    description = args.description or ""
    column_lines = "\n".join(f"| `{c}` | |" for c in columns)
    return f"""---
{configs_block}license: {args.license}
language:
{languages}
task_categories:
- text-generation
tags:
{tags}
size_categories:
- {size}
---

# {repo.split('/')[-1]}

{description}

{source}

Generated with [HimalayaAI/gymkhana](https://github.com/HimalayaAI/gymkhana)
environment `{args.environment}`. Conversations are ShareGPT in Hermes format:
`system` carries tool schemas inside `<tools>` when tools exist, assistant turns
carry reasoning inside `<think>…</think>` and calls inside `<tool_call>…</tool_call>`.
Only rows that passed the environment's verifier are included.

## Columns

| column | notes |
|---|---|
{column_lines}

`conversations` is a list of `{{"from": "system|human|gpt|tool", "value": "..."}}`;
`tools` and `expected_tool_calls` are JSON strings.

## Splits

{split_lines}

## Stats

- rows: {len(rows)}
{args.stats or ""}
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        help="export jsonl (repeatable); prefix with 'split=' to target a split, e.g. train_latn=path.jsonl",
    )
    parser.add_argument("--repo", required=True, help="owner/name on the Hub")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--environment", default="gymkhana")
    parser.add_argument("--source-dataset", default=None)
    parser.add_argument("--description", default=None, help="one paragraph for the card")
    parser.add_argument("--stats", default=None, help="extra markdown bullet lines for the card")
    parser.add_argument("--license", default="apache-2.0")
    parser.add_argument("--language", action="append", default=None, help="repeatable; default ne, en")
    parser.add_argument("--tag", action="append", default=None, help="repeatable")
    parser.add_argument("--split", default="train")
    parser.add_argument(
        "--via-datasets",
        action="store_true",
        help="push parquet with datasets.push_to_hub instead of committing the jsonl (uses hf_xet)",
    )
    parser.add_argument("--hf-token", default=None, help="defaults to $HF_TOKEN, then the cached login")
    parser.add_argument(
        "--delete-path",
        action="append",
        default=None,
        help="repo path to delete in the same commit (repeatable), e.g. stale data/train.jsonl",
    )
    parser.add_argument("--out", type=Path, default=None, help="also write the merged jsonl here")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.language = args.language or ["ne", "en"]
    args.tag = args.tag or ["sharegpt", "hermes", "synthetic"]

    # Group inputs by split: "split=path" targets that split, bare paths the default.
    by_split: Dict[str, List[Path]] = {}
    for item in args.input:
        split, sep, path = str(item).partition("=")
        by_split.setdefault(split if sep else args.split, []).append(Path(path if sep else item))

    splits = {split: merge(paths) for split, paths in by_split.items()}
    align_columns(splits)
    rows = [row for split_rows in splits.values() for row in split_rows]
    for split, split_rows in splits.items():
        print(f"merged rows: {len(split_rows)} -> split '{split}' from {len(by_split[split])} file(s)")
    print("columns:", list(rows[0]) if rows else [])
    if args.out:
        with args.out.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"wrote {args.out}")

    card = build_card(repo=args.repo, rows=rows, splits=splits, args=args)
    if args.dry_run:
        print(card)
        return

    token = resolve_token(args.hf_token)
    payloads = {
        f"data/{split}.jsonl": "".join(
            json.dumps(row, ensure_ascii=False) + "\n" for row in split_rows
        ).encode("utf-8")
        for split, split_rows in splits.items()
    }

    if args.via_datasets:
        # Parquet layout via `datasets` (like the MarketDataGenie uploaders).
        # Parquet is LFS-tracked, so this goes through hf_xet — which hangs on
        # some huggingface_hub/hf_xet combinations; keep --via-datasets opt-in.
        import tempfile

        from datasets import load_dataset
        from huggingface_hub import HfApi

        with tempfile.TemporaryDirectory() as tmpdir:
            data_files = {}
            for split, split_rows in splits.items():
                staged = Path(tmpdir) / f"{split}.jsonl"
                staged.write_text(
                    "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in split_rows), encoding="utf-8"
                )
                data_files[split] = str(staged)
            print(f"[push] datasets.push_to_hub -> {args.repo} (private={args.private})")
            load_dataset("json", data_files=data_files).push_to_hub(
                args.repo, token=token, private=args.private
            )
        HfApi().upload_file(
            path_or_fileobj=card.encode("utf-8"),
            path_in_repo="README.md",
            repo_id=args.repo,
            repo_type="dataset",
            token=token,
            commit_message="Add dataset card",
        )
    else:
        # Default: commit the JSONL directly over the Hub REST API. A .jsonl under
        # 10MB is not LFS-tracked, so this avoids the xet upload path entirely and
        # matches the layout of himalaya-ai/nepali-hermes-function-calling-v1.
        create_repo_rest(args.repo, token=token, private=args.private)
        commit_files_rest(
            args.repo,
            token=token,
            files={**payloads, "README.md": card.encode("utf-8")},
            summary=f"Upload {len(rows)} rows across {len(splits)} split(s)",
            delete=[p for p in (args.delete_path or []) if p not in payloads],
        )
    print(f"pushed {len(rows)} rows to https://huggingface.co/datasets/{args.repo}")


if __name__ == "__main__":
    main()
