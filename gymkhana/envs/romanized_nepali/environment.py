"""Verifiable Nepali Devanagari/romanized transliteration environment.

LLM agents generate rollout candidates through Gymkhana's inference service. The
deterministic bidirectional transliterator is not exposed to those agents; it is
used only to bootstrap reference answers when a dataset row has no curated
reference, and as a standalone conversion utility. Rewards are computed
deterministically by comparing each LLM candidate with the selected reference.
"""

from __future__ import annotations

import unicodedata
from difflib import SequenceMatcher
from typing import Any, ClassVar, Iterable, Optional, Sequence

from pydantic import PrivateAttr

from gymkhana.core.models import TrajectoryResult
from gymkhana.envs.config import (
    ChatModeSettings,
    DatasetSettings,
    EnvConfig,
    InferenceConfig,
    InteractionMode,
)
from gymkhana.envs.environment import Environment, Task, register_environment

from .translator import BidirectionalTranslator, create_translator


SYSTEM_PROMPT = """You transliterate Nepali text accurately.
Return only the transliterated text, without explanations or quotation marks.
Preserve punctuation, numbers, and technical Latin terms where appropriate."""


def normalize_translation(text: str) -> str:
    """Canonical form used by the deterministic verifier."""
    return " ".join(unicodedata.normalize("NFC", text).strip().split()).casefold()


@register_environment(name="romanized-nepali", env_type="romanized-nepali")
class RomanizedNepaliEnv(Environment):
    """RLVR task with LLM-generated candidates and deterministic verification.

    Dataset-provided references take precedence. The bundled transliterator
    supplies a fallback reference only; it does not produce policy rollouts and
    is not available to the LLM as a tool.
    """

    name: str = "romanized-nepali"
    _translator: BidirectionalTranslator = PrivateAttr()
    _records: Optional[list[dict[str, Any]]] = PrivateAttr(default=None)
    default_config: ClassVar[EnvConfig] = EnvConfig(
        name="romanized-nepali",
        llm=InferenceConfig(),
        interaction_mode=InteractionMode.PLAIN_TEXT,
        mode_config=ChatModeSettings(max_turns=1),
        dataset=DatasetSettings(
            environment="romanized-nepali",
            num_rollouts=4,
            reward_function="simple",
        ),
    )

    def __init__(
        self,
        *,
        config: Optional[EnvConfig] = None,
        translator: Optional[BidirectionalTranslator] = None,
        records: Optional[Iterable[dict[str, Any]]] = None,
        **data: Any,
    ) -> None:
        data["config"] = config or self.default_config.model_copy(deep=True)
        super().__init__(**data)
        self._translator = translator or create_translator()
        self._records = list(records) if records is not None else None

    def load_tasks(self, limit: Optional[int] = None) -> Sequence[Task]:
        records = self._records if self._records is not None else [
            {"id": "dev-rom-001", "direction": "devanagari_to_romanized", "source": "नेपाल सुन्दर छ।"},
            {"id": "rom-dev-001", "direction": "romanized_to_devanagari", "source": "nepal sundar chha."},
        ]
        tasks = []
        selected = records if limit is None else records[:limit]
        for record in selected:
            direction = record["direction"]
            source = record["source"]
            reference = record.get("reference") or self.translate(direction, source)
            tasks.append(
                Task(
                    id=str(record["id"]),
                    prompt=str(source),
                    metadata={"direction": direction, "reference": reference},
                )
            )
        return tasks

    def translate(self, direction: str, source: str) -> str:
        """Create a deterministic fallback reference or standalone conversion."""
        if direction == "devanagari_to_romanized":
            return self._translator.devanagari_to_romanized(source)
        if direction == "romanized_to_devanagari":
            return self._translator.romanized_to_devanagari(source)
        raise ValueError(f"Unsupported translation direction: {direction}")

    def get_environment_instructions(self, task: Task) -> str:
        return SYSTEM_PROMPT

    def format_initial_message(self, task: Task) -> str:
        target = (
            "romanized Nepali" if task.metadata["direction"] == "devanagari_to_romanized"
            else "Nepali Devanagari"
        )
        return f"Transliterate the following text into {target}:\n\n{task.prompt}"

    def score(self, prediction: str, reference: str) -> float:
        predicted = normalize_translation(prediction)
        expected = normalize_translation(reference)
        if predicted == expected:
            return 1.0
        return SequenceMatcher(None, predicted, expected).ratio()

    def evaluate_answer(self, task: Task, result: TrajectoryResult) -> bool:
        return self.score(result.final_answer, task.metadata["reference"]) == 1.0

    async def compute_reward(
        self,
        result: TrajectoryResult,
        answer_correct: Optional[bool] = None,
        task: Optional[Task] = None,
    ) -> float:
        if task is None:
            raise ValueError("RomanizedNepaliEnv.compute_reward requires a task")
        reward = self.score(result.final_answer, task.metadata["reference"])
        result.total_reward = reward
        result.answer_correct = reward == 1.0
        result.reward_function = "normalized-exact-plus-edit-similarity"
        return reward
