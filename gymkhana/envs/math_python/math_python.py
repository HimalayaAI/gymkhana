"""Math-Python environment implementation."""

from __future__ import annotations

from typing import Any, ClassVar, Dict, Iterable, List, Optional, Sequence

from datasets import load_dataset

from ..config import EnvironmentType, LLMClientType
from ..config import DatasetSettings, EnvConfig, RLMEnvConfig, REPLSettings, SubLLMSettings
from ..environment import Environment, EnvironmentError, Task, register_environment


ENV_INSTRUCTIONS = (
    "## Math Problem Instructions\n"
    "- Use sympy or numpy for precise computation.\n"
    "- Verify results with code before concluding.\n"
    "- Place numerical/symbolic answers inside \\boxed{...}.\n"
)


DEFAULT_MATH_CONFIG = RLMEnvConfig(
    name="math-python",
    main_model="Hermes-4-405B",
    main_client=LLMClientType.LITELLM,
    main_temperature=0.7,
    main_max_tokens=4096,
    enable_reasoning=False,  # Disable <think> blocks by default (enable with --enable-reasoning flag)
    repl=REPLSettings(
        server_url="http://localhost:5003",
        max_output_chars=8192,
        max_output_lines=500,
        timeout_seconds=120,
        max_turns=10,
    ),
    sub_llm=SubLLMSettings(
        model="Hermes-4-70B",
        client=LLMClientType.LITELLM,
        max_parallel=8,
        timeout_seconds=60,
        max_tokens=2048,
        temperature=0.5,
    ),
    dataset=DatasetSettings(
        environment=EnvironmentType.MATH_PYTHON,
        dataset_name="nvidia/Nemotron-Math-v2",
        dataset_config=None,
        dataset_split="low",  # Easiest split
        field_mapping={
            "id": "uuid",
            "prompt": "problem",
            "expected_answer": "expected_answer",
            "context": None,
        },
        batch_size=4,
        num_rollouts=1,
        limit=100,
        include_instructions=True,
        output_dir="outputs/gymkhana",
        output_sharegpt=True,
        mask_observations=False,
        enable_rewards=True,
        reward_function="simple",
    ),
    debug=False,
)


@register_environment(name="math-python", env_type=EnvironmentType.MATH_PYTHON)
class MathPythonEnv(Environment):
    """Standalone math reasoning environment powered by Python execution.

    Supports parallel rollouts per task: set config.dataset.num_rollouts > 1 (or
    --num-rollouts) for best-of-N / GRPO. Uses base repl_sessions to create G
    sandbox sessions per task; no env-specific overrides needed.
    """

    name: str = "math-python"
    default_config: ClassVar[RLMEnvConfig] = DEFAULT_MATH_CONFIG

    def __init__(self, *, config: Optional[EnvConfig] = None, **data: Any) -> None:  # type: ignore[override]
        if config is None:
            config = self.default_config.model_copy(deep=True)
        elif isinstance(config, dict):
            config = RLMEnvConfig(**config)
        elif not isinstance(config, RLMEnvConfig):
            config_data = config.model_dump()
            config_data["interaction_mode"] = "rlm"
            config_data["mode_config"] = None
            config = RLMEnvConfig(**config_data)

        data["config"] = config

        # Use MathematicalVerifier for math tasks to handle LaTeX and mathematical equivalence
        from gymkhana.core.models.parsers import MathematicalVerifier
        data["answer_verifier"] = MathematicalVerifier(tolerance=1e-6)

        super().__init__(**data)

    # ------------------------------------------------------------------
    # Dataset handling
    # ------------------------------------------------------------------
    def _load_dataset(self) -> Iterable[Dict[str, Any]]:
        cfg = self.config.dataset
        if not cfg.dataset_name:
            raise EnvironmentError("MathPythonEnv requires dataset.dataset_name to be set")

        try:
            if cfg.dataset_config:
                ds = load_dataset(cfg.dataset_name, cfg.dataset_config, split=cfg.dataset_split, streaming=True)
            else:
                ds = load_dataset(cfg.dataset_name, split=cfg.dataset_split, streaming=True)

            if cfg.dataset_seed is not None:
                ds = ds.shuffle(seed=cfg.dataset_seed, buffer_size=10)

            print(f"Using streaming mode for {cfg.dataset_name} (split={cfg.dataset_split}, seed={cfg.dataset_seed})")
            return ds
        except ImportError as exc:  # pragma: no cover - dependency missing
            raise EnvironmentError("datasets package is required for MathPythonEnv") from exc
        except Exception as exc:  # pragma: no cover - surface dataset errors
            raise EnvironmentError(f"Failed to load dataset '{cfg.dataset_name}': {exc}") from exc

    def load_tasks(self, limit: Optional[int] = None) -> Sequence[Task]:  # type: ignore[override]
        dataset_limit = limit or self.config.dataset.limit
        mapping = self.config.dataset.field_mapping
        records = self._load_dataset()

        tasks: List[Task] = []
        seen = 0
        id_field = mapping.get("id")
        prompt_field = mapping.get("prompt") or "prompt"
        expected_field = mapping.get("expected_answer")
        context_field = mapping.get("context")

        for record in records:
            if dataset_limit is not None and seen >= dataset_limit:
                break

            task_id = record.get(id_field) if id_field else record.get("id") or str(seen)
            prompt = record.get(prompt_field)

            if not prompt:
                continue

            context = record.get(context_field) if context_field else None
            expected_answer = record.get(expected_field) if expected_field else None

            metadata = {
                k: v
                for k, v in record.items()
                if k not in {id_field, prompt_field, context_field}
            }
            if expected_answer is not None:
                metadata["expected_answer"] = expected_answer

            tasks.append(
                Task(
                    id=str(task_id),
                    prompt=str(prompt),
                    context=str(context) if context is not None else None,
                    metadata=metadata,
                )
            )
            seen += 1

        return tasks

    # ------------------------------------------------------------------
    # Behaviour hooks
    # ------------------------------------------------------------------
    def get_environment_instructions(self, task: Task) -> str:
        if not self.config.dataset.include_instructions:
            return ""
        return ENV_INSTRUCTIONS

    def format_initial_message(self, task: Task) -> str:
        if not task.context:
            return task.prompt
        context_length = len(task.context)
        return (
            f"{task.prompt}\n\n[Context available as `context` variable or via "
            f"read_file('context.txt'); ~{context_length} characters]"
        )

    def accept_response_without_code(self, response: str, *, num_code_blocks: int) -> Optional[str]:
        if num_code_blocks > 0:
            candidates = self.extract_inline_answers(response)
            if candidates:
                return response
        return None
