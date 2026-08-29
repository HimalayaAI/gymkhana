"""Merge multilingual-tool-use exports and push them to the Hugging Face Hub.

Output schema mirrors NousResearch/hermes-function-calling-v1 (id, conversations,
tools, category, subcategory, task) and appends provenance columns:
source_query (original English), target_language, expected_tool_calls,
localizer_model, policy_model.

Usage:
    python scripts/push_hermes_nepali_to_hub.py \
        --input outputs/.../hermes_singleturn_nepali.jsonl \
        --input outputs/.../pass2/hermes_singleturn_nepali.jsonl \
        --repo himalaya-ai/hermes-function-calling-nepali --private
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List

TOOLS_RE = re.compile(r"<tools>\s*(.*?)\s*</tools>", re.DOTALL)


def _tools_from_row(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    tools = row.get("tools")
    if isinstance(tools, list) and tools:
        return tools
    if isinstance(tools, str) and tools.strip():
        try:
            return json.loads(tools)
        except json.JSONDecodeError:
            pass
    system = next((m["value"] for m in row["conversations"] if m.get("from") == "system"), "")
    match = TOOLS_RE.search(system)
    return json.loads(match.group(1)) if match else []


def reshape(row: Dict[str, Any], *, localizer_model: str, policy_model: str) -> Dict[str, Any]:
    provenance = row.get("source_provenance") or {}
    return {
        "id": row["id"],
        "conversations": row["conversations"],
        "tools": json.dumps(_tools_from_row(row), ensure_ascii=False),
        "category": provenance.get("category"),
        "subcategory": provenance.get("subcategory"),
        "task": provenance.get("task"),
        "source_query": row.get("source_query"),
        "target_language": row.get("target_language"),
        "expected_tool_calls": json.dumps(row.get("expected_tool_calls") or [], ensure_ascii=False),
        "localizer_model": row.get("localizer_model") or localizer_model,
        "policy_model": row.get("policy_model") or policy_model,
    }


def merge(paths: Iterable[Path], **models: str) -> List[Dict[str, Any]]:
    seen: Dict[str, Dict[str, Any]] = {}
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = reshape(json.loads(line), **models)
                seen.setdefault(row["id"], row)  # first occurrence wins
    return list(seen.values())


DATASET_CARD = """---
license: apache-2.0
language:
- ne
- en
task_categories:
- text-generation
tags:
- function-calling
- tool-use
- nepali
- sharegpt
- synthetic
size_categories:
- n<1K
---

# hermes-function-calling-nepali

Single-turn function-calling conversations with the **user request spoken in Nepali
(Devanagari)** and verified tool calls, derived from
[NousResearch/hermes-function-calling-v1](https://huggingface.co/datasets/NousResearch/hermes-function-calling-v1)
(`func_calling_singleturn`). Tool schemas and ground-truth calls are unchanged from the
source; only the user turn was re-spoken in Nepali, voice-assistant style. Each assistant
turn carries the policy model's reasoning in `<think>…</think>` followed by
`<tool_call>` blocks (Hermes format).

## How it was made

Generated with [HimalayaAI/gymkhana](https://github.com/HimalayaAI/gymkhana)'s
`multilingual-tool-use` environment:

1. **Localizer** ({localizer_model}) rewrites the English request as a Nepali speaker
   would say it to an assistant. A deterministic gate rejects rewrites that drop any
   identifier-like argument value, URL, or e-mail the tool needs, or that are not
   written in Devanagari.
2. **Policy** ({policy_model}) receives the English tool schemas via native tool calling
   and the Nepali request, and emits tool calls; its provider-returned reasoning is
   captured.
3. **Verifier**: the calls must match the English ground truth exactly (name and
   arguments, order-independent, all calls present). Rows are exported only when at
   least one of {rollouts} rollouts matched; that rollout is exported.

Rows whose ground truth could not be reproduced in {rollouts} attempts are excluded, so
every row here has a verified call set.

## Schema

Same columns as the source, plus provenance:

| column | description |
|---|---|
| `id` | source row id |
| `conversations` | ShareGPT: `system` (Hermes `<tools>` prompt), `human` (Nepali request), `gpt` (`<think>` + `<tool_call>`) |
| `tools` | JSON string, OpenAI function schemas (from the source) |
| `category`, `subcategory`, `task` | from the source |
| `source_query` | the original English request |
| `target_language` | `ne-Deva` |
| `expected_tool_calls` | JSON string, source ground truth the row was verified against |
| `localizer_model`, `policy_model` | models used |

## Stats

- rows: {rows}
- source rows attempted: {attempted}; excluded: localization gate {gate}, no matching call in {rollouts} rollouts {miss}

## License

Apache-2.0, inherited from the source dataset.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", action="append", required=True, type=Path, help="export jsonl (repeatable)")
    parser.add_argument("--repo", required=True, help="e.g. himalaya-ai/hermes-function-calling-nepali")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--localizer-model", default="litellm:openai/gpt-5.6-luna-pro")
    parser.add_argument("--policy-model", default="litellm:deepseek/deepseek-v4-flash-0731")
    parser.add_argument("--rollouts", type=int, default=4)
    parser.add_argument("--attempted", type=int, default=0)
    parser.add_argument("--gate-rejected", type=int, default=0)
    parser.add_argument("--no-match", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None, help="also write the merged jsonl here")
    parser.add_argument("--dry-run", action="store_true", help="build and report, do not push")
    args = parser.parse_args()

    rows = merge(args.input, localizer_model=args.localizer_model, policy_model=args.policy_model)
    print(f"merged rows: {len(rows)} from {len(args.input)} file(s)")
    if args.out:
        with args.out.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"wrote {args.out}")

    card = DATASET_CARD.format(
        localizer_model=args.localizer_model,
        policy_model=args.policy_model,
        rollouts=args.rollouts,
        rows=len(rows),
        attempted=args.attempted or "n/a",
        gate=args.gate_rejected or "n/a",
        miss=args.no_match or "n/a",
    )
    if args.dry_run:
        print(card)
        return

    from datasets import Dataset
    from huggingface_hub import HfApi

    dataset = Dataset.from_list(rows)
    dataset.push_to_hub(args.repo, private=args.private, split="train")
    HfApi().upload_file(
        path_or_fileobj=card.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=args.repo,
        repo_type="dataset",
    )
    print(f"pushed {len(rows)} rows to https://huggingface.co/datasets/{args.repo}")


if __name__ == "__main__":
    main()
