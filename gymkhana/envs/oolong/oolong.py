"""OOLONG long-context environment implementation."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any, ClassVar, Dict, Iterable, List, Optional, Sequence

from datasets import load_dataset

from ..config import (
    DatasetSettings,
    EnvironmentType,
    LLMClientType,
    REPLSettings,
    RLMEnvConfig,
    SubLLMSettings,
)
from ..environment import Environment, EnvironmentError, Task, register_environment
from ..workspace_utils import format_single_file_context


INSTRUCTIONS = (
    "<instructions>\n"
    "Strategy for long-context information retrieval:\n"
    "1. Split the context into manageable chunks with overlap.\n"
    "2. Use Python tools (regex, counters, parsing) to narrow search targets.\n"
    "3. Leverage sub_agent() for semantic questions on individual chunks.\n"
    "4. Aggregate evidence programmatically before giving the final answer.\n"
    "</instructions>"
)

CONTEXT_FILENAME = "context.txt"
# Keep generated snippets below common request/proxy limits while uploading
# long contexts incrementally.
CONTEXT_CHUNK_SIZE = 10_000


DEFAULT_OOLONG_CONFIG = RLMEnvConfig(
    name="oolong",
    main_model="Hermes-4-405B",
    main_client=LLMClientType.LITELLM,
    main_temperature=0.7,
    main_max_tokens=4096,
    repl=REPLSettings(
        server_url="http://localhost:5003",
        max_output_chars=8192,
        max_output_lines=500,
        timeout_seconds=180,
        max_turns=10,
    ),
    sub_llm=SubLLMSettings(
        model="Hermes-4-70B",
        client=LLMClientType.LITELLM,
        max_parallel=8,
        timeout_seconds=60,
        max_tokens=2048,
        temperature=0.3,
    ),
    dataset=DatasetSettings(
        environment=EnvironmentType.OOLONG,
        dataset_name="oolongbench/oolong-real",
        dataset_config="dnd",
        dataset_split="validation",
        field_mapping={
            "id": "id",
            "prompt": "question",
            "expected_answer": "answer",
            "context": "context_window_text",
        },
        batch_size=4,
        num_rollouts=1,
        limit=None,
        include_instructions=True,
        output_dir="outputs/gymkhana_oolong",
        output_sharegpt=True,
        mask_observations=False,
        enable_rewards=False,
        reward_function="simple",
    ),
    debug=False,
)


@register_environment(name="oolong", env_type=EnvironmentType.OOLONG)
class OolongEnv(Environment):
    """Long-context reasoning environment for OOLONG benchmark datasets.

    Supports parallel rollouts per task (config.dataset.num_rollouts > 1):
    overrides repl_sessions to upload context to each of G sessions before yielding.
    """

    name: str = "oolong"
    default_config: ClassVar[RLMEnvConfig] = DEFAULT_OOLONG_CONFIG

    def __init__(self, *, config: Optional[RLMEnvConfig] = None, **data: Any) -> None:  # type: ignore[override]
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
        super().__init__(**data)

    # ------------------------------------------------------------------
    # Dataset handling
    # ------------------------------------------------------------------
    def _load_dataset(self) -> Iterable[Dict[str, Any]]:
        cfg = self.config.dataset
        if not cfg.dataset_name:
            raise EnvironmentError("OolongEnv requires dataset.dataset_name to be set")

        try:
            if cfg.dataset_config:
                ds = load_dataset(
                    cfg.dataset_name,
                    cfg.dataset_config,
                    split=cfg.dataset_split,
                    streaming=True,
                )
            else:
                ds = load_dataset(cfg.dataset_name, split=cfg.dataset_split, streaming=True)

            if cfg.dataset_seed is not None:
                ds = ds.shuffle(seed=cfg.dataset_seed, buffer_size=1)

            return ds
        except ImportError as exc:  # pragma: no cover - dependency missing
            raise EnvironmentError("datasets package is required for OolongEnv") from exc
        except Exception as exc:  # pragma: no cover - surface dataset errors
            raise EnvironmentError(f"Failed to load dataset '{cfg.dataset_name}': {exc}") from exc

    def load_tasks(self, limit: Optional[int] = None) -> Sequence[Task]:  # type: ignore[override]
        cfg = self.config.dataset
        dataset_limit = limit or cfg.limit
        mapping = cfg.field_mapping
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

            context_value = record.get(context_field) if context_field else None
            context_text = str(context_value) if context_value else None
            expected_answer = record.get(expected_field) if expected_field else None

            metadata = {
                k: v
                for k, v in record.items()
                if k not in {id_field, prompt_field, context_field}
            }
            if expected_answer is not None:
                metadata["expected_answer"] = expected_answer
            if context_text is not None:
                metadata["context_length"] = len(context_text)

            tasks.append(
                Task(
                    id=str(task_id),
                    prompt=str(prompt),
                    context=context_text,
                    metadata=metadata,
                )
            )
            seen += 1

        return tasks

    # ------------------------------------------------------------------
    # Behaviour hooks
    # ------------------------------------------------------------------
    def format_initial_message(self, task: Task) -> str:
        prompt = task.prompt
        if not task.context:
            return prompt

        context_length = len(task.context)
        instructions = self.get_environment_instructions(task)

        return format_single_file_context(
            prompt=prompt,
            filename=CONTEXT_FILENAME,
            content_length=context_length,
            file_type="txt",
            instructions=instructions if instructions else None,
        )

    def prepare_repl_context(self, task: Task) -> Optional[str]:  # type: ignore[override]
        # Provide context so our overridden session can upload it manually.
        return task.context

    def get_environment_instructions(self, task: Task) -> str:
        if not self.config.dataset.include_instructions:
            return ""
        return INSTRUCTIONS

    @staticmethod
    def _generate_context_write_snippets(context: str) -> List[str]:
        snippets: List[str] = []
        for index in range(0, len(context), CONTEXT_CHUNK_SIZE):
            chunk = context[index : index + CONTEXT_CHUNK_SIZE]
            mode = "w" if index == 0 else "a"
            escaped_chunk = (
                chunk.replace("\\", "\\\\")
                .replace("'", "\\'")
                .replace("\n", "\\n")
                .replace("\r", "\\r")
            )
            snippets.append(
                f"with open('{CONTEXT_FILENAME}', '{mode}') as f:\n"
                f"    f.write('{escaped_chunk}')"
            )
        return snippets

    async def _upload_context_to_workspace(self, repl, context: Optional[str]) -> None:
        if not context:
            return

        for snippet in self._generate_context_write_snippets(context):
            if self.config.debug:
                print(f"[DEBUG] Executing snippet:\n{snippet}")
            result = await repl.execute(snippet)
            if result.error:
                raise EnvironmentError(f"Failed to upload context chunk: {result.error}")

    @asynccontextmanager
    async def repl_session(  # type: ignore[override]
        self,
        *,
        context: Optional[str] = None,
        enable_bash: bool = False,
        sub_agent_config: Optional[Dict[str, Any]] = None,
    ):
        context_text = context
        async with super().repl_session(
            context=None,
            enable_bash=enable_bash,
            sub_agent_config=sub_agent_config,
        ) as repl:
            await self._upload_context_to_workspace(repl, context_text)
            yield repl

    @asynccontextmanager
    async def repl_sessions(  # type: ignore[override]
        self,
        *,
        context: Optional[str] = None,
        enable_bash: bool = False,
        sub_agent_config: Optional[Dict[str, Any]] = None,
        num_sessions: int = 1,
    ):
        context_text = context
        async with super().repl_sessions(
            context=None,
            enable_bash=enable_bash,
            sub_agent_config=sub_agent_config,
            num_sessions=num_sessions,
        ) as repls:
            for repl in repls:
                await self._upload_context_to_workspace(repl, context_text)
            yield repls


__all__ = ["OolongEnv", "DEFAULT_OOLONG_CONFIG"]
