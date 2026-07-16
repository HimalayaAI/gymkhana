"""HotpotQA multi-hop question answering environment."""

from __future__ import annotations

import json
import re
from contextlib import asynccontextmanager
from typing import Any, ClassVar, Dict, Iterable, List, Optional, Sequence

from datasets import load_dataset

from ..config import (
    DatasetSettings,
    EnvConfig,
    EnvironmentType,
    LLMClientType,
    REPLSettings,
    SubLLMSettings,
)
from ..environment import Environment, EnvironmentError, Task, register_environment
from ..workspace_utils import format_workspace_context

DOC_FILENAME_PREFIX = "doc"
DOC_FILENAME_PATTERN = "{prefix}_{index:02d}_{slug}.txt"
# Larger chunks for efficiency (REPL can handle up to ~1MB per request)
DOC_CHUNK_SIZE = 100_000

INSTRUCTIONS = (
    "<instructions>\n"
    "HotpotQA multi-hop strategy:\n"
    "1. Multiple documents are saved in your workspace with doc_XX_Title.txt naming.\n"
    "2. Each document is provided in a <file> tag showing filename, size, and title.\n"
    "3. Use read_file('filename.txt') to load specific documents.\n"
    "4. Combine evidence across documents with Python before finalizing your answer.\n"
    "5. Use sub_agent(...) when you need semantic extraction (names, dates, relations).\n"
    "</instructions>"
)

DEFAULT_HOTPOTQA_CONFIG = EnvConfig(
    name="hotpotqa",
    main_model="Hermes-4-405B",
    main_client=LLMClientType.LITELLM,
    main_temperature=0.7,
    main_max_tokens=4096,
    repl=REPLSettings(
        server_url="http://localhost:5003",
        max_output_chars=8_192,
        max_output_lines=500,
        timeout_seconds=180,
        max_turns=15,
    ),
    sub_llm=SubLLMSettings(
        model="Hermes-4-70B",
        client=LLMClientType.LITELLM,
        max_parallel=8,
        timeout_seconds=60,
        max_tokens=2_048,
        temperature=0.3,
    ),
    dataset=DatasetSettings(
        environment=EnvironmentType.HOTPOTQA,
        dataset_name="hotpotqa/hotpot_qa",
        dataset_config="distractor",
        dataset_split="validation",
        field_mapping={
            "id": "id",
            "prompt": "question",
            "expected_answer": "answer",
            "context": "context",
        },
        context_processor="hotpotqa",
        batch_size=4,
        num_rollouts=1,
        limit=None,
        include_instructions=True,
        output_dir="outputs/gymkhana_hotpotqa",
        output_sharegpt=True,
        mask_observations=False,
        enable_rewards=False,
        reward_function="simple",
    ),
    debug=False,
)


@register_environment(name="hotpotqa", env_type=EnvironmentType.HOTPOTQA)
class HotpotQAEnv(Environment):
    """Environment for multi-hop question answering across multiple documents.

    Supports parallel rollouts per task (config.dataset.num_rollouts > 1):
    overrides repl_sessions to upload documents to each of G sessions before yielding.
    """

    name: str = "hotpotqa"
    default_config: ClassVar[EnvConfig] = DEFAULT_HOTPOTQA_CONFIG

    def __init__(self, *, config: Optional[EnvConfig] = None, **data: Any) -> None:  # type: ignore[override]
        if config is None:
            config = self.default_config.model_copy(deep=True)
        elif isinstance(config, dict):
            config = EnvConfig(**config)

        data["config"] = config
        super().__init__(**data)

    # ------------------------------------------------------------------
    # Dataset handling
    # ------------------------------------------------------------------
    def _load_dataset(self) -> Iterable[Dict[str, Any]]:
        cfg = self.config.dataset
        if not cfg.dataset_name:
            raise EnvironmentError("HotpotQAEnv requires dataset.dataset_name to be set")

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
                ds = ds.shuffle(seed=cfg.dataset_seed, buffer_size=10)

            return ds
        except ImportError as exc:  # pragma: no cover
            raise EnvironmentError("datasets package is required for HotpotQAEnv") from exc
        except Exception as exc:  # pragma: no cover
            raise EnvironmentError(f"Failed to load dataset '{cfg.dataset_name}': {exc}") from exc

    def load_tasks(self, limit: Optional[int] = None) -> Sequence[Task]:  # type: ignore[override]
        dataset_limit = limit or self.config.dataset.limit
        mapping = self.config.dataset.field_mapping
        records = self._load_dataset()

        tasks: List[Task] = []
        seen = 0
        id_field = mapping.get("id")
        prompt_field = mapping.get("prompt") or "question"
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
            documents = self._normalize_documents(context_value)

            expected_answer = record.get(expected_field) if expected_field else None

            metadata = {
                k: v
                for k, v in record.items()
                if k not in {id_field, prompt_field, context_field}
            }
            metadata["documents"] = documents
            if expected_answer is not None:
                metadata["expected_answer"] = expected_answer

            tasks.append(
                Task(
                    id=str(task_id),
                    prompt=str(prompt),
                    context=json.dumps(documents),
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
        return INSTRUCTIONS

    def format_initial_message(self, task: Task) -> str:
        documents = self._get_documents(task)
        if not documents:
            return task.prompt

        files = [
            {
                "filename": doc["filename"],
                "content_length": len(doc["content"]),
                "title": doc.get("title", doc["filename"]),
            }
            for doc in documents
        ]

        instructions = self.get_environment_instructions(task)

        return format_workspace_context(
            prompt=task.prompt,
            files=files,
            file_type="txt",
            instructions=instructions if instructions else None,
        )

    def enable_bash_for_task(self, task: Task) -> bool:  # type: ignore[override]
        return False

    @asynccontextmanager
    async def repl_session(  # type: ignore[override]
        self,
        *,
        context: Optional[str] = None,
        enable_bash: bool = False,
        sub_agent_config: Optional[Dict[str, Any]] = None,
    ):
        documents = self._deserialize_documents(context) if context else []
        async with super().repl_session(
            context=None,
            enable_bash=enable_bash,
            sub_agent_config=sub_agent_config,
        ) as repl:
            await self._upload_documents(repl, documents)
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
        documents = self._deserialize_documents(context) if context else []
        async with super().repl_sessions(
            context=None,
            enable_bash=enable_bash,
            sub_agent_config=sub_agent_config,
            num_sessions=num_sessions,
        ) as repls:
            for repl in repls:
                await self._upload_documents(repl, documents)
            yield repls

    # ------------------------------------------------------------------
    # Document helpers
    # ------------------------------------------------------------------
    def _normalize_documents(self, context_value: Any) -> List[Dict[str, str]]:
        documents: List[Dict[str, str]] = []
        if not context_value:
            return documents

        # Handle dict-of-lists format (common in datasets library for this dataset)
        if isinstance(context_value, dict) and "title" in context_value and "sentences" in context_value:
            titles = context_value["title"]
            all_sentences = context_value["sentences"]
            entries = []
            for t, s in zip(titles, all_sentences):
                entries.append({"title": t, "sentences": s})
            context_value = entries

        for idx, entry in enumerate(context_value):
            if isinstance(entry, dict):
                title = entry.get("title") or entry.get("heading") or f"Doc {idx + 1}"
                sentences = entry.get("sentences") or entry.get("text") or ""
            elif isinstance(entry, (list, tuple)) and len(entry) >= 2:
                title, sentences = entry[0], entry[1]
            else:
                title = f"Document {idx + 1}"
                sentences = entry

            text = self._render_sentences(sentences)
            if not text.strip():
                continue

            filename = self._make_filename(idx, title)
            documents.append(
                {
                    "filename": filename,
                    "title": str(title),
                    "content": text,
                }
            )

        return documents

    @staticmethod
    def _render_sentences(sentences: Any) -> str:
        if sentences is None:
            return ""
        if isinstance(sentences, str):
            return sentences
        if isinstance(sentences, (list, tuple)):
            return "\n".join(str(s) for s in sentences)
        return str(sentences)

    @staticmethod
    def _slugify(title: str) -> str:
        slug = re.sub(r"[^A-Za-z0-9]+", "_", title).strip("_")
        return slug or "document"

    def _make_filename(self, index: int, title: str) -> str:
        slug = self._slugify(title)[:60]
        return DOC_FILENAME_PATTERN.format(prefix=DOC_FILENAME_PREFIX, index=index + 1, slug=slug)

    def _get_documents(self, task: Task) -> List[Dict[str, str]]:
        documents = task.metadata.get("documents") if task.metadata else None
        if not documents and task.context:
            documents = self._deserialize_documents(task.context)
        if not documents:
            return []
        normalized: List[Dict[str, str]] = []
        for i, doc in enumerate(documents):
            normalized.append(
                {
                    "filename": doc.get("filename", f"doc_{i+1:02d}.txt"),
                    "title": doc.get("title", f"Document {i+1}"),
                    "content": doc.get("content", ""),
                }
            )
        return normalized

    def _deserialize_documents(self, data: str) -> List[Dict[str, str]]:
        try:
            parsed = json.loads(data)
        except Exception:
            return []
        docs: List[Dict[str, str]] = []
        if isinstance(parsed, list):
            for i, item in enumerate(parsed):
                if not isinstance(item, dict):
                    continue
                docs.append(
                    {
                        "filename": item.get("filename", f"doc_{i+1:02d}.txt"),
                        "title": item.get("title", f"Document {i+1}"),
                        "content": item.get("content", ""),
                    }
                )
        return docs

    @staticmethod
    def _escape_chunk(chunk: str) -> str:
        return (
            chunk.replace("\\", "\\\\")
            .replace("'", "\\'")
            .replace("\n", "\\n")
            .replace("\r", "\\r")
        )

    @staticmethod
    def _escape_filename(filename: str) -> str:
        """Escape filename for use inside a single-quoted Python string."""
        return filename.replace("\\", "\\\\").replace("'", "\\'")

    def _generate_document_snippets(self, filename: str, content: str) -> List[str]:
        snippets: List[str] = []
        safe_name = self._escape_filename(filename)
        for idx in range(0, len(content), DOC_CHUNK_SIZE):
            chunk = content[idx : idx + DOC_CHUNK_SIZE]
            mode = "w" if idx == 0 else "a"
            escaped = self._escape_chunk(chunk)
            snippets.append(
                f"with open('{safe_name}', '{mode}') as f:\n"
                f"    f.write('{escaped}')"
            )
        if not snippets:
            snippets.append(f"open('{safe_name}', 'w').close()")
        return snippets

    async def _upload_documents(self, repl, documents: List[Dict[str, str]]) -> None:
        for doc in documents:
            filename = doc.get("filename")
            content = doc.get("content", "")
            if not filename:
                continue
            for snippet in self._generate_document_snippets(filename, content):
                if self.config.debug:
                    print(f"[DEBUG] Uploading snippet to {filename}:\n{snippet}")
                result = await repl.execute(snippet)
                if result.error:
                    raise EnvironmentError(
                        f"Failed to upload document '{filename}': {result.error}"
                    )


__all__ = ["HotpotQAEnv", "DEFAULT_HOTPOTQA_CONFIG"]
