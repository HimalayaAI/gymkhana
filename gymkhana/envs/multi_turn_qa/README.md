# Multi-turn QA generation

`multi-turn-qa` is a two-agent dataset-generation environment. A questioner
sees the private source and produces a structured question plan. A separate
answer agent sees only the conversation that will be exported. Verification
can inspect the private source and reference answer, but neither is exposed to
the answer agent.

Set `generation.turns: 1` for single-turn QA. Values greater than one generate
follow-up questions against one persistent visible answer-agent conversation.

## Profiles

The built-in profiles are `textbook`, `legal`, `health`, `finance`,
`agriculture`, `ecommerce`, `banking`, and `general`. Each profile supplies its
own questioner rules, answerer rules, subcategories, and default context mix.
High-stakes profiles use visible source grounding and reject unsupported judge
outputs. When a profile requires grounding, even deterministic answers are sent
to the judge and must receive full grounding and visible-context-sufficiency
marks. Judge totals use a 0–10 rubric and are normalized to 0–1 before applying
`acceptance_threshold`.

Textbook subcategories are `conceptual`, `factual`, `math_stem`, `literature`,
`procedural`, and `source_analysis`. The dataset's `subject` is authoritative
and is copied directly into every task and export. With `subcategory: auto`, a
deterministic subject-first selector chooses only the QA generation strategy;
it never infers or replaces the source subject.

## Context policies

- `closed_book`: no source excerpt is exported; the question must stand alone.
- `inline_excerpt`: the first user message contains the minimum source excerpt.
- `self_contained_problem`: all facts and values are embedded in the question.
- `conversation_grounded`: later questions may rely on earlier visible turns.
- `auto`: select from the profile/configured context distribution.

The answer agent never receives hidden source context. A generated sample is
therefore usable under the same information boundary in which it was created.

## Included datasets

Generate Nepali textbook QA:

```bash
python -m gymkhana.run \
  --config configs/multi_turn_qa/nepali_textbooks.yaml \
  --limit 100 \
  --no-database
```

Generate source-grounded legal QA from PDFs:

```bash
python -m gymkhana.run \
  --config configs/multi_turn_qa/nepali_legal_pdf.yaml \
  --limit 100 \
  --no-database
```

The textbook corpus is already chunked. The legal dataset contains raw PDFs in
its `validation` split, so the environment extracts text, creates overlapping
page-aware chunks, and preserves PDF title and page range as provenance.

Use `--qa-turns 1` for a single-turn preview, `--target-language ne-Latn` for
Romanized Nepali, and `--questioner-model`, `--model`, and `--judge-model` to
configure the three model roles independently.

## Onboarding another dataset

Copy `configs/multi_turn_qa/domain_template.yaml`, choose a profile, and map the
dataset's fields to `id`, `text`, `source`, `subject`, `title`, `language`,
`license`, `jurisdiction`, and `document_date`. No dataset-specific runner or
shell script is needed. Text sources can be local JSON/JSONL or Hugging Face
datasets; PDF rows use the Hugging Face `Pdf` feature or a path/bytes mapping.

For PDFs without a mapped date field, `document_date_strategy` can fall back to
dates in the filename and then the document text. The legal configuration uses
`mapped_or_filename_or_text` and records `document_date_source` in provenance;
the original date text is preserved without guessing a calendar conversion.

Each run writes accepted ShareGPT JSONL, a full audit JSONL with private plans
and verifier results, a summary JSON, and `run.log` under `dataset.output_dir`.
