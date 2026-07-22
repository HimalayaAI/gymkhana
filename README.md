# Gymkhana

Gymkhana is a multilingual RLVR environment framework for generating scored
model trajectories and supervised fine-tuning datasets. For every task, an
environment can launch a group of independent LLM rollouts in parallel, verify
their outputs, assign rewards, and retain the task and rollout provenance needed
for training-data selection.

Gymkhana uses **GRPO-style rollout groups**: `G` candidates are sampled for the
same prompt and compared under the same reward function. Gymkhana generates and
scores those groups; it is not itself a GRPO optimizer. The resulting data can be
used for reward-based filtering, best-of-N selection, SFT export, or downstream
GRPO/RL training.

English and Nepali are the initial language focus. Environments are pluggable so
additional languages, verifiers, tools, and task families can be added without
coupling them to a particular model provider.

```text
dataset rows
    -> environment tasks and prompts
    -> G parallel model rollouts
    -> deterministic or task-specific verifier
    -> per-candidate rewards and grouped provenance
    -> selected SFT trajectories / downstream RLVR data
```

The execution core was ported from DeepGym and rebranded for HimalayaAI.
Inference is routed through [Pydantic AI v2](https://ai.pydantic.dev/) using
provider-qualified model names such as `anthropic:...` and `openai:...`.
Environment implementations depend on Gymkhana's typed inference service rather
than provider SDKs.

## Features

- GRPO-style parallel rollouts with stable task, group, and candidate indexes.
- Pydantic AI v2 inference routing across supported model providers.
- Plain-text, native tool-calling, interleaved tool-calling, and sandboxed RLM
  interaction modes.
- Deterministic rewards, answer verifiers, and optional task-specific judges.
- Storage-backed rollout tracking and reward-filtered ShareGPT/SFT export.
- First-class support for English and Nepali tasks, with no language hardcoding
  in the environment interface.
- Dependency injection for inference, storage, and sandbox services, making
  environments testable without paid API calls.

## Built-in environments

| Environment | Task family |
| --- | --- |
| `multi-turn-qa` | Two-agent single/multi-turn textbook and domain QA generation |
| `romanized-nepali` | Bidirectional Nepali Devanagari/romanized transliteration |
| `english-sharegpt-to-nepali` | Full-conversation English ShareGPT to Nepali ShareGPT translation |
| `ifeval` | Verifiable instruction-following constraints |
| `math-python` | Mathematical problem solving with a Python sandbox |
| `hotpotqa` | Multi-hop question answering |
| `oolong` | Long-context reasoning |
| `swe` | Software-engineering tasks in a sandbox |
| `tool-use-singleturn` | Verifiable single-turn tool use |

## Installation

Gymkhana supports Python 3.10 through 3.13.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

Provider credentials follow Pydantic AI conventions. Keep them in a local
`.env`, which is ignored by Git:

```bash
cp .env.example .env
# Edit .env locally. Never commit it or paste API keys into issues or chats.
```

## Generate rollout data

The CLI uses each environment's default configuration and accepts model, task,
and rollout overrides. This example runs four Claude candidates per Romanized
Nepali task:

```bash
python -m gymkhana.run \
  --env romanized-nepali \
  --model anthropic:claude-sonnet-5 \
  --client anthropic \
  --limit 2 \
  --num-rollouts 4
```

`--num-rollouts` controls the group size `G`. `--batch-size` controls how many
different tasks can run concurrently. Provider calls may incur cost.

For a bounded live smoke test that prints every candidate and reward:

```bash
python scripts/smoke_test_romanized_nepali.py --group-size 2
```

The smoke test requires `ANTHROPIC_API_KEY` and optionally reads
`ANTHROPIC_MODEL` from `.env`. It never prints the credential.

### Programmatic rollout groups

The lower-level rollout API is useful when a data pipeline needs every candidate
instead of the environment runner's best candidate:

```python
from gymkhana.core.services.inference import (
    PydanticAIInferenceService,
    RolloutRequest,
    generate_rollout_group,
)

service = PydanticAIInferenceService(
    default_model="openai:gpt-4.1-mini",
    max_concurrency=8,
)
group = await generate_rollout_group(
    service,
    RolloutRequest(
        task_id="example-1",
        prompt="Answer the task",
        group_size=8,
    ),
)

for candidate in group.candidates:
    print(candidate.group_id, candidate.index, candidate.output)
```

Candidates retain their task ID, rollout-group ID, and ordered sample index so
rewards and exports can be compared within the correct group.

## Romanized Nepali: LLM policy, deterministic verifier

`romanized-nepali` adapts the bidirectional translator contributed in
[Gymkhana PR #1](https://github.com/HimalayaAI/gymkhana/pull/1). It supports
Devanagari → romanized Nepali and romanized Nepali → Devanagari.

The environment deliberately separates policy generation from verification:

- Pydantic AI sends the task to the configured LLM for every rollout.
- A curated dataset reference is used whenever one is present.
- The deterministic translator supplies a bootstrap reference only when a row
  has no curated reference; it is not exposed to the LLM as a tool.
- The verifier compares the LLM output with the reference using normalized exact
  match and edit similarity for a graded non-exact reward.

Because Nepali can have several reasonable Romanizations, deterministic fallback
references are best treated as bootstrap labels. Production datasets should
prefer reviewed references or an explicitly documented variant-aware verifier.
Original acknowledgments, license, and third-party notices are retained under
`gymkhana/envs/romanized_nepali/`.

## English ShareGPT to Nepali

`english-sharegpt-to-nepali` translates complete ShareGPT conversations into
natural Nepali Devanagari and exports the translated conversation rather than
the translation instruction. It accepts ShareGPT `conversations`, OpenAI-style
`messages`, and flattened Hermes `instruction`/`response` rows from local
JSON/JSONL files or Hugging Face datasets.

The reward checks strict JSON, exact turn-role structure, non-empty messages,
Devanagari use, preservation of code/math/URLs/tags/numbers, and semantic
translation fidelity. Reference-free rows must pass a configurable LLM judge;
a reviewed Nepali reference can be used instead. Both paths fail closed below
the export threshold, and native-speaker review is still required.

Translate a bounded local Hermes file:

```bash
python -m gymkhana.run \
  --env english-sharegpt-to-nepali \
  --dataset-name /path/to/openhermes.jsonl \
  --model openai:gpt-4.1-mini \
  --judge-model openai:gpt-4.1-mini \
  --client openai \
  --limit 100 \
  --num-rollouts 4
```

Run the checked-in OpenHermes configuration through the same standard runner:

```bash
python -m gymkhana.run \
  --config configs/english_sharegpt_to_nepali/openhermes.yaml \
  --no-database
```

The runner writes accepted JSONL, a full audit JSONL, a summary, and `run.log`
under the configured `output_dir`; PostgreSQL and helper scripts are optional.
Use `--dataset-offset` and `--limit` for bounded windows. Compatible ShareGPT,
OpenAI-message, and flattened Hermes datasets can be onboarded with YAML alone;
see the environment README for backend and schema details.

For OpenHermes or another mixed-source corpus, keep every row's provenance and
verify the license and attribution obligations of each retained source. No
third-party dataset rows are bundled with this environment. The full loader and
reward contract are documented in
`gymkhana/envs/english_sharegpt_nepali/README.md`.

## Multi-turn QA generation

`multi-turn-qa` generates single- or multi-turn conversations with separate
questioner and answer agents. The questioner and verifier may inspect the
private source, while the answer agent sees exactly the visible conversation
that is exported. Built-in profiles cover textbook, legal, health, finance,
agriculture, ecommerce, banking, and general-domain QA.

Run the checked-in Nepali textbook or legal-PDF configurations:

```bash
python -m gymkhana.run \
  --config configs/multi_turn_qa/nepali_textbooks.yaml \
  --limit 100 \
  --no-database

python -m gymkhana.run \
  --config configs/multi_turn_qa/nepali_legal_pdf.yaml \
  --limit 100 \
  --no-database
```

Use `--qa-turns 1` for single-turn output or choose `en`, `ne-Deva`, or
`ne-Latn` with `--target-language`. Dataset mappings, context policies,
verifier routing, profile extension, and PDF chunk provenance are documented in
`gymkhana/envs/multi_turn_qa/README.md`.

## Contributing a new environment

An environment owns five things: dataset ingestion, prompt construction,
interaction mode, answer verification, and reward calculation. The shared runner
owns inference routing, parallel rollout execution, tracking, and lifecycle.

### 1. Define the task and verifier first

Before implementing model calls, specify:

- What one `Task` represents and which fields belong in `metadata`.
- What makes an answer verifiably correct.
- Whether partial credit is valid and how it is bounded.
- Which language, script, locale, and normalization rules apply.
- Which dataset license and attribution files must be retained.

Prefer deterministic verification for RLVR environments. If an LLM judge is
unavoidable, document why, isolate it from the policy model, and add tests for
judge parsing and failure behavior. Never include a gold answer in the model
prompt or expose an oracle as an agent tool unless that is explicitly the task.

### 2. Choose an interaction mode

| Mode | Use when | Common environment hooks |
| --- | --- | --- |
| `PLAIN_TEXT` | The model returns text without tools | Prompts, parser, verifier, reward |
| `TOOL_CALL` | The model uses native tools over one or more rounds | `get_tool_executor()` |
| `TOOL_CALL_INTERLEAVED` | Tool calls are interleaved with model reasoning | `get_tool_executor()` |
| `RLM` | The task needs Python/REPL or repository interaction | `prepare_repl_context()`, sandbox settings |

Use Pydantic AI tools through Gymkhana's `EnvironmentToolkit`. Do not import a
provider SDK or create a separate model client inside an environment.

### 3. Add the environment package

Create a self-contained package:

```text
gymkhana/envs/my_environment/
    __init__.py
    environment.py
```

A minimal plain-text environment looks like this:

```python
from typing import ClassVar, Optional, Sequence

from gymkhana.core.models import TrajectoryResult
from gymkhana.envs.config import (
    ChatModeSettings,
    DatasetSettings,
    EnvConfig,
    InferenceConfig,
    InteractionMode,
)
from gymkhana.envs.environment import Environment, Task, register_environment


ROWS = (
    {"id": "example-1", "prompt": "2 + 2 = ?", "reference": "4"},
)


@register_environment(name="my-environment", env_type="my-environment")
class MyEnvironment(Environment):
    name: str = "my-environment"
    default_config: ClassVar[EnvConfig] = EnvConfig(
        name="my-environment",
        llm=InferenceConfig(),
        interaction_mode=InteractionMode.PLAIN_TEXT,
        mode_config=ChatModeSettings(max_turns=1),
        dataset=DatasetSettings(
            environment="my-environment",
            num_rollouts=4,
            enable_rewards=True,
        ),
    )

    def __init__(self, *, config: Optional[EnvConfig] = None, **data):
        data["config"] = config or self.default_config.model_copy(deep=True)
        super().__init__(**data)

    def load_tasks(self, limit: Optional[int] = None) -> Sequence[Task]:
        rows = ROWS if limit is None else ROWS[:limit]
        return [
            Task(
                id=row["id"],
                prompt=row["prompt"],
                metadata={"reference": row["reference"]},
            )
            for row in rows
        ]

    def get_environment_instructions(self, task: Task) -> str:
        return "Return only the final answer."

    def evaluate_answer(self, task: Task, result: TrajectoryResult) -> bool:
        return result.final_answer.strip() == task.metadata["reference"]

    async def compute_reward(
        self,
        result: TrajectoryResult,
        answer_correct: Optional[bool] = None,
        task: Optional[Task] = None,
    ) -> float:
        if task is None:
            raise ValueError("A task is required to compute reward")
        reward = float(self.evaluate_answer(task, result))
        result.answer_correct = reward == 1.0
        result.total_reward = reward
        result.reward_function = "exact-match"
        return reward
```

The base runner formats the interaction, calls the configured Pydantic AI v2
inference service, executes `G` candidates concurrently, scores every candidate,
and chooses the highest-reward result. Override only the hooks your task needs.

### 4. Register every public entry point

- Export the class from its package `__init__.py`.
- Import it in `gymkhana/envs/__init__.py` so the decorator runs at startup.
- Add its canonical hyphenated name to `EnvironmentType` when appropriate.
- Add it to the CLI environment choices in `gymkhana/run.py`.
- Add it to the built-in environment table in this README.

Registry keys are normalized for case, hyphens, and underscores, but use one
canonical hyphenated name in configs, docs, and dataset metadata.

### 5. Design rewards for training, not just demos

Reward functions should be deterministic, side-effect free, and stable across
parallel calls whenever the task permits. Contributors should:

- Document the reward range and the meaning of partial credit.
- Normalize only what is genuinely equivalent for the task.
- Score candidates independently of whether storage is configured.
- Set `answer_correct`, `total_reward`, and `reward_function` on the trajectory.
- Keep references and verifier-only state out of prompts and agent tools.
- Define behavior for empty, malformed, timed-out, and failed outputs.
- Avoid rewarding verbosity, formatting artifacts, or leaked reference text.

For multilingual environments, preserve Unicode in source data and tests, state
the normalization form, cover punctuation and mixed-script input, and include
native-speaker-reviewed fixtures where practical.

### 6. Make parallel execution safe

Rollouts from the same group must be independent. Avoid mutable global state,
order-dependent rewards, reused sandbox sessions, and shared iterators that can
hand different answers to concurrent candidates. Bound external concurrency and
clean up files, clients, or sessions in `on_finalize()`.

Use stable task IDs and deterministic dataset ordering. Honor `limit`, preserve
dataset split information, and attach only JSON-serializable provenance to task
metadata.

### 7. Test without requiring an API key

Add tests under `tests/envs/` with a scripted `InferenceService`. At minimum,
cover:

- Registration by canonical name and supported aliases.
- Task loading, stable IDs, limits, and required metadata.
- Prompt construction without reference leakage.
- Exact, incorrect, partial, empty, and malformed reward cases.
- One rollout and a grouped rollout where the best reward is selected.
- Resource cleanup and error isolation for tool or sandbox environments.

Live provider tests must be opt-in, tightly bounded, and marked with `live` plus
the provider marker. They must skip when credentials are absent and must never
print secrets.

```bash
python -m pytest -q tests/envs/test_my_environment.py
python -m pytest -q
python -m gymkhana.run --help
```

An environment is ready to merge when its offline tests pass, registration and
CLI discovery work, its reward contract is documented, dataset licensing is
clear, and one bounded end-to-end smoke test has succeeded for any external
services it requires.

## Development

Run the full offline test suite:

```bash
python -m pytest -q
```

Local source checkouts belong under `reference_repos/`. The directory is
gitignored and excluded from test discovery. Keep `.env`, generated datasets,
logs, and service state out of commits.
