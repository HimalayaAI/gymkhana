"""IFEval (Instruction Following) environment implementation.

Loads tasks from ``allenai/RLVR-IFeval`` in streaming mode and evaluates
model responses against format-level constraints (lowercase, paragraph
count, JSON format, quotation marks, etc.).

Uses **ChatMode** with **SingleTurnManager** — single-turn text generation
where the model must follow specific formatting constraints embedded in the
instruction.

Dataset Structure:
    - messages: List with single user message containing instruction + constraint
    - ground_truth: JSON string with validator function name and parameters
    - constraint: Human-readable constraint description
    - constraint_type: Category (e.g., "Keyword Frequency", "All Lowercase")

Example:
    messages: [{"content": "Answer this question... In your response,
                the word nonsensorial should appear 17 times.", "role": "user"}]
    ground_truth: {"func_name": "verify_keyword_frequency", "N": 17,
                   "word": "nonsensorial", ...}
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, ClassVar, Dict, Iterable, List, Optional, Sequence

from datasets import load_dataset
from pydantic import ConfigDict

from ..config import (
    DatasetSettings,
    EnvConfig,
    InferenceConfig,
    InteractionMode,
    LLMClientType,
)
from ..environment import Environment, EnvironmentError, Task, register_environment
from ..modes import ChatMode
from ..managers import SingleTurnManager
from gymkhana.core.models import AnswerVerifier, TrajectoryResult
from gymkhana.core.rewards import RewardFunction, register_reward_function, TrajectoryMetrics

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constraint Verifier (implements AnswerVerifier protocol)
# ---------------------------------------------------------------------------

class ConstraintVerifier(AnswerVerifier):
    """Verifier for IFEval instruction-following constraints.

    Implements the AnswerVerifier protocol to validate model responses
    against format-level constraints specified in the task metadata.

    Each constraint is validated by a specific method that checks if the
    response satisfies the formatting requirement (e.g., all lowercase,
    specific word frequency, JSON format, etc.).
    """

    def verify(
        self,
        *,
        expected: Optional[str],
        candidates: List[str],
        task_metadata: Optional[Dict[str, Any]] = None,
        trajectory: Optional[TrajectoryResult] = None,
    ) -> Optional[bool]:
        """Verify that the response satisfies the constraint.

        Args:
            expected: Not used for IFEval (constraints are in metadata)
            candidates: List of candidate responses (typically just final_answer)
            task_metadata: Must contain 'ground_truth' with constraint details
            trajectory: Optional trajectory result for additional context

        Returns:
            True if constraint is satisfied, False if violated, None if cannot verify
        """
        if not candidates or not task_metadata:
            return None

        ground_truth = task_metadata.get("ground_truth", {})
        func_name = ground_truth.get("func_name")

        if not func_name:
            logger.warning("No func_name in ground_truth")
            return None

        # Get the validator method
        validator_method = getattr(self, func_name, None)
        if validator_method is None:
            logger.warning(f"Validator '{func_name}' not implemented")
            return None

        # Validate the first candidate (IFEval is single-response)
        text = candidates[0]

        try:
            return validator_method(text, **ground_truth)
        except Exception as e:
            logger.exception(f"Validator '{func_name}' raised exception: {e}")
            return False

    # -- casing / character constraints ----------------------------------------

    def validate_lowercase(self, text: str, **kwargs) -> bool:
        """No capital letters allowed."""
        return not any(c.isupper() for c in text)

    def validate_no_capital_letters(self, text: str, **kwargs) -> bool:
        return self.validate_lowercase(text, **kwargs)

    # -- punctuation constraints -----------------------------------------------

    def validate_no_commas(self, text: str, **kwargs) -> bool:
        return "," not in text

    def validate_quotation(self, text: str, **kwargs) -> bool:
        """Wrap entire response with double quotation marks."""
        s = text.strip()
        return s.startswith('"') and s.endswith('"')

    # -- structural constraints ------------------------------------------------

    def verify_paragraph_count(self, text: str, N: int = None, **kwargs) -> bool:
        """Paragraphs separated by markdown divider ``***``."""
        if N is None:
            return True
        parts = text.split("***")
        count = len([p for p in parts if p.strip()])
        return count == int(N)

    def validate_paragraphs(self, text: str, N: int = None, first_word: str = None, **kwargs) -> bool:
        """N paragraphs separated by two newlines, first word of i-th paragraph matches."""
        if N is None:
            return True
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        if len(paragraphs) != int(N):
            return False
        if first_word is not None:
            # first_word is typically a comma-separated list of expected first words
            # or a single word for a specific paragraph
            pass  # TODO: implement first-word checking per paragraph
        return True

    def validate_highlighted_sections(self, text: str, N: int = None, **kwargs) -> bool:
        """At least *N* sections wrapped in markdown asterisks ``*…*``."""
        if N is None:
            return True
        sections = re.findall(r"\*([^*]+)\*", text)
        return len(sections) >= int(N)

    # -- content frequency constraints -----------------------------------------

    def verify_keyword_frequency(self, text: str, keyword_list: list = None, word: str = None, N: int = None, **kwargs) -> bool:
        """The word(s) in *keyword_list* or *word* should each appear exactly *N* times."""
        if N is None:
            return True

        # Handle both keyword_list and word parameters
        keywords = keyword_list if keyword_list else ([word] if word else [])
        if not keywords:
            return True

        target = int(N)
        for kw in keywords:
            if text.lower().count(kw.lower()) != target:
                return False
        return True

    def verify_letter_frequency(self, text: str, letter: str = None, N: int = None,
                                quantifier: str = None, **kwargs) -> bool:
        """Letter *letter* should appear (exactly | at least | at most) *N* times."""
        if not letter or N is None:
            return True
        count = text.lower().count(letter.lower())
        target = int(N)
        if quantifier == "at least":
            return count >= target
        elif quantifier == "at most":
            return count <= target
        return count == target

    def validate_frequency_capital_words(self, text: str, N: int = None,
                                         quantifier: str = None, **kwargs) -> bool:
        """Words with ALL capital letters should appear (at least | at most) *N* times."""
        if N is None:
            return True
        words = text.split()
        count = sum(1 for w in words if w.isupper() and w.isalpha())
        target = int(N)
        if quantifier == "at least":
            return count >= target
        elif quantifier == "at most":
            return count <= target
        return count == target

    # -- format constraints ----------------------------------------------------

    def validate_json_format(self, text: str, **kwargs) -> bool:
        """Entire output should be valid JSON (optionally in a code block)."""
        s = text.strip()
        if s.startswith("```json") and s.endswith("```"):
            s = s[7:-3].strip()
        elif s.startswith("```") and s.endswith("```"):
            s = s[3:-3].strip()
        try:
            json.loads(s)
            return True
        except Exception:
            return False

    def validate_repeat_prompt(self, text: str, original_prompt: str = None, **kwargs) -> bool:
        """Response must begin with a verbatim repeat of the original prompt."""
        if not original_prompt:
            return True
        return text.strip().startswith(original_prompt.strip())

    # -- postscript constraint -------------------------------------------------

    def validate_postscript(self, text: str, postscript_marker: str = None, **kwargs) -> bool:
        """Response must end with a postscript starting with *postscript_marker*."""
        if not postscript_marker:
            return True
        lines = text.strip().split("\n")
        return any(line.strip().startswith(postscript_marker) for line in lines[-3:])

    # -- word / section constraints --------------------------------------------

    def validate_forbidden_words(self, text: str, forbidden_words: list = None, **kwargs) -> bool:
        """None of the *forbidden_words* may appear in the response."""
        if not forbidden_words:
            return True
        lower = text.lower()
        return not any(w.lower() in lower for w in forbidden_words)

    def validate_end_checker(self, text: str, end_phrase: str = None, **kwargs) -> bool:
        """Response must end with *end_phrase*."""
        if not end_phrase:
            return True
        return text.strip().endswith(end_phrase)

    def validate_word_constraint(self, text: str, word: str = None, **kwargs) -> bool:
        """The word *word* must appear in the response."""
        if not word:
            return True
        return word.lower() in text.lower()

    def validate_number_of_words(self, text: str, N: int = None, quantifier: str = None, **kwargs) -> bool:
        """Word count constraint."""
        if N is None:
            return True
        count = len(text.split())
        target = int(N)
        if quantifier == "at least":
            return count >= target
        elif quantifier == "at most":
            return count <= target
        return count == target

    def validate_sections(self, text: str, N: int = None, section_splitter: str = None, **kwargs) -> bool:
        """Response must contain exactly *N* sections split by *section_splitter*."""
        if N is None:
            return True
        splitter = section_splitter or "\n\n"
        parts = [p.strip() for p in text.split(splitter) if p.strip()]
        return len(parts) == int(N)

    def validate_choice(self, text: str, options: list = None, **kwargs) -> bool:
        """Response must be exactly one of the given *options*."""
        if not options:
            return True
        s = text.strip()
        return s in options

    # -- additional structural constraints -------------------------------------

    def validate_placeholders(self, text: str, N: int = None, **kwargs) -> bool:
        """Response must contain at least *N* placeholders (e.g., [placeholder])."""
        if N is None:
            return True
        # Count placeholders in square brackets
        placeholders = re.findall(r'\[([^\]]+)\]', text)
        return len(placeholders) >= int(N)

    def validate_title(self, text: str, **kwargs) -> bool:
        """Response must contain a title (line starting with # in markdown)."""
        lines = text.split('\n')
        return any(line.strip().startswith('#') for line in lines)

    def validate_two_responses(self, text: str, **kwargs) -> bool:
        """Response must contain two distinct responses/sections."""
        # Look for common separators or numbered sections
        # Check for patterns like "1." and "2." or "Response 1:" and "Response 2:"
        has_numbered = bool(re.search(r'1\.|Response 1', text, re.IGNORECASE)) and \
                      bool(re.search(r'2\.|Response 2', text, re.IGNORECASE))
        # Or check for markdown sections
        has_sections = text.count('***') >= 1 or text.count('\n\n') >= 1
        return has_numbered or has_sections

    def verify_bullet_points(self, text: str, N: int = None, **kwargs) -> bool:
        """Response must contain at least *N* bullet points."""
        if N is None:
            return True
        # Count lines starting with bullet markers (-, *, •)
        lines = text.split('\n')
        bullet_count = sum(1 for line in lines if re.match(r'^\s*[-*•]\s+', line))
        return bullet_count >= int(N)

    def verify_keywords(self, text: str, forbidden_words: list = None, **kwargs) -> bool:
        """Verify that certain keywords appear or don't appear.

        Note: Based on schema, this uses forbidden_words parameter (inverse logic).
        """
        if not forbidden_words:
            return True
        # This appears to be checking that keywords DO appear (not forbidden)
        # The parameter name is misleading in the dataset
        lower = text.lower()
        return all(kw.lower() in lower for kw in forbidden_words)


# ---------------------------------------------------------------------------
# Constraint Reward Function
# ---------------------------------------------------------------------------

@register_reward_function("constraint")
class ConstraintRewardFunction(RewardFunction):
    """Reward function for constraint-based evaluation.

    Returns 1.0 if the constraint is satisfied, 0.0 otherwise.
    Uses ConstraintVerifier to validate responses against task metadata.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self, **data):
        super().__init__(**data)
        # Create verifier after initialization (not as a field)
        object.__setattr__(self, '_verifier', ConstraintVerifier())

    @property
    def verifier(self) -> ConstraintVerifier:
        """Get the constraint verifier instance."""
        return getattr(self, '_verifier', ConstraintVerifier())

    def compute(self, metrics: TrajectoryMetrics) -> Dict[str, Any]:
        """Compute reward based on constraint satisfaction.

        Args:
            metrics: Trajectory metrics containing final_answer and task_metadata

        Returns:
            Dict with total_reward, final_step_reward, and metadata
        """
        # Extract final answer and task metadata from metrics
        final_answer = getattr(metrics, 'final_answer', None)
        task_metadata = getattr(metrics, 'task_metadata', None)

        if not final_answer or not task_metadata:
            return {
                "total_reward": 0.0,
                "final_step_reward": 0.0,
                "metadata": {
                    "reward_function": self.name,
                    "constraint_satisfied": False,
                    "reason": "missing_answer_or_metadata",
                }
            }

        # Use verifier to check constraint
        passed = self.verifier.verify(
            expected=None,
            candidates=[final_answer],
            task_metadata=task_metadata,
            trajectory=None,
        )

        if passed is None:
            final_reward = 0.0
            reason = "verification_failed"
        elif passed:
            final_reward = 1.0
            reason = "constraint_satisfied"
        else:
            final_reward = 0.0
            reason = "constraint_violated"

        # Include intermediate rewards if any
        intermediate_rewards = getattr(metrics, 'intermediate_rewards', [])
        total = sum(intermediate_rewards) + final_reward

        return {
            "total_reward": total,
            "final_step_reward": final_reward,
            "metadata": {
                "reward_function": self.name,
                "constraint_satisfied": passed if passed is not None else False,
                "reason": reason,
                "constraint_type": task_metadata.get("constraint_type", "unknown"),
            }
        }


# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------

ENV_INSTRUCTIONS = (
    "Follow the given instructions precisely. "
    "Pay close attention to any formatting constraints.\n"
)


def _get_default_config() -> EnvConfig:
    """Create default config with env vars loaded.

    This is a function instead of a module-level constant to ensure
    .env is loaded before reading environment variables.
    """
    from ..config import ChatModeSettings

    # Map environment value to the provider-neutral client identifier.
    client_str = os.getenv("LITELLM_CLIENT", "litellm").lower()
    client_map = {client.value: client for client in LLMClientType}

    return EnvConfig(
        name="ifeval",
        llm=InferenceConfig(
            client=client_map.get(client_str, LLMClientType.LITELLM),
            model=os.getenv("LITELLM_MODEL", "gpt-4o"),
            temperature=float(os.getenv("LITELLM_TEMPERATURE", "0.7")),
            max_tokens=int(os.getenv("LITELLM_MAX_TOKENS", "4096")),
        ),
        interaction_mode=InteractionMode.PLAIN_TEXT,
        mode_config=ChatModeSettings(
            max_turns=1,
            allow_self_refinement=False,
        ),
        dataset=DatasetSettings(
            environment="ifeval",
            dataset_name="allenai/RLVR-IFeval",
            dataset_config=None,
            dataset_split="train",
            field_mapping={
                "id": None,
                "prompt": "messages",
                "expected_answer": None,
                "context": None,
            },
            batch_size=4,
            num_rollouts=1,
            limit=int(os.getenv("IFEVAL_LIMIT", "100")),
            include_instructions=True,
            output_dir="outputs/ifeval",
            output_sharegpt=True,
            enable_rewards=True,
            reward_function="constraint",
        ),
        debug=False,
    )


# Public, copyable default. Each environment instance still receives a deep copy.
DEFAULT_IFEVAL_CONFIG: EnvConfig = _get_default_config()


# ---------------------------------------------------------------------------
# Environment class
# ---------------------------------------------------------------------------

@register_environment(name="ifeval", env_type="custom")
class IfEvalEnv(Environment):
    """Instruction-following evaluation environment.

    Loads ``allenai/RLVR-IFeval`` in streaming mode and scores model
    outputs against deterministic constraint validators.
    """

    name: str = "ifeval"
    default_config: ClassVar[EnvConfig] = DEFAULT_IFEVAL_CONFIG

    def __init__(self, *, config: Optional[EnvConfig] = None, **data: Any) -> None:
        if config is None:
            config = IfEvalEnv.default_config.model_copy(deep=True)
        elif isinstance(config, dict):
            if isinstance(config.get("llm"), dict):
                config = {**config, "llm": InferenceConfig(**config["llm"])}
            config = EnvConfig(**config)

        data["config"] = config
        super().__init__(**data)

        # Initialize interaction mode and conversation manager
        self._mode = ChatMode()
        self._manager = SingleTurnManager()

    # ------------------------------------------------------------------
    # Dataset handling
    # ------------------------------------------------------------------
    def _load_dataset(self) -> Iterable[Dict[str, Any]]:
        cfg = self.config.dataset
        if not cfg.dataset_name:
            raise EnvironmentError("IfEvalEnv requires dataset.dataset_name to be set")

        try:
            if cfg.dataset_config:
                ds = load_dataset(cfg.dataset_name, cfg.dataset_config,
                                  split=cfg.dataset_split, streaming=True)
            else:
                ds = load_dataset(cfg.dataset_name,
                                  split=cfg.dataset_split, streaming=True)

            if cfg.dataset_seed is not None:
                ds = ds.shuffle(seed=cfg.dataset_seed, buffer_size=1000)

            print(f"Using streaming mode for {cfg.dataset_name} "
                  f"(split={cfg.dataset_split}, seed={cfg.dataset_seed})")
            return ds
        except ImportError as exc:
            raise EnvironmentError("datasets package is required for IfEvalEnv") from exc
        except Exception as exc:
            raise EnvironmentError(
                f"Failed to load dataset '{cfg.dataset_name}': {exc}"
            ) from exc

    def load_tasks(self, limit: Optional[int] = None) -> Sequence[Task]:
        dataset_limit = limit or self.config.dataset.limit
        records = self._load_dataset()

        tasks: List[Task] = []
        seen = 0

        for record in records:
            if dataset_limit is not None and seen >= dataset_limit:
                break

            # Extract prompt from messages list
            messages = record.get("messages", [])
            prompt = messages[0].get("content", "") if messages else ""
            if not prompt:
                continue

            # Parse ground_truth JSON string
            gt_str = record.get("ground_truth", "{}")
            try:
                ground_truth = json.loads(gt_str) if isinstance(gt_str, str) else gt_str
            except (json.JSONDecodeError, TypeError):
                ground_truth = {}

            metadata = {
                "ground_truth": ground_truth,
                "constraint": record.get("constraint", ""),
                "constraint_type": record.get("constraint_type", ""),
                "dataset": record.get("dataset", "ifeval"),
            }

            tasks.append(
                Task(
                    id=str(seen),
                    prompt=prompt,
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
        """The prompt already contains the constraint instruction."""
        return task.prompt

    async def execute_task(self, task: Task) -> "TrajectoryResult":
        """Execute a single task using ChatMode.

        Args:
            task: The task to execute

        Returns:
            TrajectoryResult with the model's response and constraint validation
        """
        # Use ChatMode to execute the task
        result = await self._mode.execute_single(task, self, self._manager)

        # Compute reward based on constraint validation
        if self.config.dataset.enable_rewards:
            # Verify constraint
            verifier = ConstraintVerifier()
            passed = verifier.verify(
                expected=None,
                candidates=[result.final_answer],
                task_metadata=task.metadata,
                trajectory=result,
            )

            # Set answer_correct based on verification
            result.answer_correct = passed

            # Compute reward
            reward = await self.compute_reward(result, answer_correct=passed, task=task)
            result.total_reward = reward
            result.step_rewards = [reward]

        return result

    # ------------------------------------------------------------------
    # Reward / scoring
    # ------------------------------------------------------------------
    async def compute_reward(
        self,
        result: "TrajectoryResult",
        answer_correct: Optional[bool] = None,
        task: Optional[Task] = None,
    ) -> float:
        """Score 1.0 if the constraint is satisfied, else 0.0."""
        if not result.final_answer or task is None:
            return 0.0

        # Use ConstraintVerifier to validate the response
        verifier = ConstraintVerifier()
        passed = verifier.verify(
            expected=None,  # Not used for IFEval
            candidates=[result.final_answer],
            task_metadata=task.metadata,
            trajectory=result,
        )

        # Debug logging
        if self.config.debug:
            ground_truth = task.metadata.get("ground_truth", {})
            func_name = ground_truth.get("func_name", "unknown")
            constraint_type = task.metadata.get("constraint_type", "unknown")
            constraint_desc = task.metadata.get("constraint", "")

            print("─" * 100)
            print(f"Answer Verification - Task {task.id}")
            print("─" * 100)
            print(f"Constraint Type: {constraint_type}")
            print(f"Constraint: {constraint_desc}")
            print(f"Validator Function: {func_name}")

            # Show ground truth JSON
            print(f"Ground Truth: {json.dumps(ground_truth, indent=2)}")

            # Show relevant parameters (for quick reference)
            params = []
            if ground_truth.get("N") is not None:
                params.append(f"N={ground_truth['N']}")
            if ground_truth.get("word"):
                params.append(f"word='{ground_truth['word']}'")
            if ground_truth.get("keyword_list"):
                params.append(f"keyword_list={ground_truth['keyword_list']}")
            if ground_truth.get("letter"):
                params.append(f"letter='{ground_truth['letter']}'")
            if ground_truth.get("quantifier"):
                params.append(f"quantifier='{ground_truth['quantifier']}'")
            if ground_truth.get("end_phrase"):
                params.append(f"end_phrase='{ground_truth['end_phrase']}'")
            if ground_truth.get("forbidden_words"):
                params.append(f"forbidden_words={ground_truth['forbidden_words']}")
            if ground_truth.get("options"):
                params.append(f"options={ground_truth['options']}")

            if params:
                print(f"Key Parameters: {', '.join(params)}")

            print(f"Result: {'✅ PASSED' if passed else '❌ FAILED' if passed is not None else '⚠️  UNKNOWN'}")

            if passed is False:
                print(f"Reason: Constraint not satisfied")
            elif passed is None:
                print(f"Reason: Could not verify constraint (validator not found or error)")

            print()

        if passed is None:
            logger.warning("Could not verify constraint for task %s", task.id)
            return 0.0

        return 1.0 if passed else 0.0
