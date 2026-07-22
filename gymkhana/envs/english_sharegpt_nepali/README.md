# English ShareGPT to Nepali

`english-sharegpt-to-nepali` translates a complete English conversation into
Nepali Devanagari while retaining canonical ShareGPT roles. It uses Gymkhana's
plain-text inference service; it does not create a provider client or submit an
external batch from inside the environment.

## Accepted input

The loader accepts local JSON/JSONL files, Hugging Face datasets, injected test
records, and these common row shapes:

- ShareGPT: `{"conversations":[{"from":"human","value":"..."}, ...]}`
- OpenAI messages: `{"messages":[{"role":"user","content":"..."}, ...]}`
- Flattened Hermes rows: `{"instruction":"...", "response":"..."}`

Original row fields such as `id`, `source`, category, and license metadata are
kept in `Task.metadata.source_provenance` and copied into every exported
ShareGPT row. Optional reviewed translations may be supplied in
`nepali_conversations`, `reference_conversations`, or `translation_reference`;
references are verifier-only and are never put in the policy prompt.

## Configure credentials and run

Gymkhana does not ask for API keys interactively. Each user must configure the
API key for their selected model provider before running the environment.

From the Gymkhana repository root, create a local `.env` file:

```bash
cp .env.example .env
```
Open .env in an editor and add your own OpenAI API key:
```bash
OPENAI_API_KEY=your-openai-api-key
```
Alternatively, configure the key only for the current terminal session:
```bash
export OPENAI_API_KEY="your-openai-api-key"
```

Reference-free rows are evaluated by a semantic judge and fail closed if that
judge is unavailable or returns an invalid result. The default judge is
`openai:gpt-4.1-mini`; override it with `--judge-model` when needed. A row with
a reviewed Nepali reference uses that reference instead and does not make a
judge call.

## Standard OpenHermes run

Use the checked-in partial configuration with the standard environment runner:

```bash
python -m gymkhana.run \
  --config configs/english_sharegpt_to_nepali/openhermes.yaml \
  --no-database
```

The default OpenAI models can be replaced entirely from the CLI. For example:

```bash
python -m gymkhana.run \
  --config configs/english_sharegpt_to_nepali/openhermes.yaml \
  --model anthropic:claude-sonnet-4-6 \
  --client anthropic \
  --judge-model anthropic:claude-haiku-4-5 \
  --no-database
```

No database or helper script is required. `Environment.run()` writes these
artifacts atomically under `dataset.output_dir`:

- `<output_basename>.jsonl`: accepted ShareGPT translations only
- `<output_basename>_audit.jsonl`: every source row, raw output, reward, and
  rejection details
- `<output_basename>_summary.json`: counts, models, and artifact paths
- `run.log`: runner and per-task progress logs

Resume or inspect another bounded window with `--dataset-offset`; use `--limit`
to control its size. Both flags override YAML.

```bash
python -m gymkhana.run \
  --config configs/english_sharegpt_to_nepali/openhermes.yaml \
  --dataset-offset 1000 \
  --limit 100 \
  --no-database
```

## Onboard another dataset

No Python adapter is needed when a source uses one of the supported row shapes:
ShareGPT `conversations`, OpenAI `messages`, or flattened Hermes
`instruction`/`response`. Create a partial YAML config containing:

```yaml
name: english-sharegpt-to-nepali
dataset:
  dataset_name: organization/dataset-name
  dataset_backend: huggingface-rows
  dataset_config: default
  dataset_split: train
  dataset_offset: 0
  limit: 100
  output_dir: outputs/my_dataset_nepali
  output_basename: my_dataset_nepali
```

`huggingface-rows` is efficient for public datasets exposed by the Hugging Face
dataset viewer. Use `huggingface` for datasets that must be loaded through the
`datasets` library, or `local`/`auto` for JSON and JSONL files. Unsupported row
schemas should be added as a small normalization adapter with fixture-based
tests, rather than embedded in the runner.

When the schema is compatible but uses different field names, map it in YAML:

```yaml
dataset:
  field_mapping:
    id: uid
    conversations: dialogue
    # Or use instruction: prompt and response: completion
```

Mappings are applied before normalization, while unrelated row fields remain
attached as source provenance.

## Reward contract

The reward is in `[0, 1]`. A deterministic structural score combines strict
JSON/schema validity, exact message-role structure, non-empty messages,
Devanagari use, and exact preservation of code/math/URL/tag/number spans.
Role/count changes and empty messages are hard failures. Missing any protected
span caps the score below the `0.80` export threshold.

After the structural gate passes, reference-free rows receive a semantic judge
score for message-by-message fidelity. The final reward is the lower of the
structural and semantic scores, so both must reach `0.80` for export. When a
reviewed reference is present, normalized per-message similarity replaces the
judge and must independently reach the same threshold.

These automated checks catch common data-corruption and adequacy failures but
cannot prove translation accuracy. Review a statistically meaningful sample
with fluent Nepali speakers before training or publishing a generated dataset.

## Dataset and license policy

No OpenHermes or other third-party dataset rows are bundled. OpenHermes combines
multiple sources with different licensing and attribution requirements. Keep
the upstream provenance fields, select only sources you are entitled to use,
and retain all required notices when distributing translations.
