"""Environment abstractions for Gymkhana.

This module introduces a pluggable RL-style environment interface that unifies
pipeline, REPL, inference, and reward handling across task types. The goal is to
allow new environments (math puzzles, long-context reasoning, SWE, games, etc.)
to reuse a shared execution core while customizing context ingestion and rollout
strategy.
"""

from __future__ import annotations

import asyncio
import logging
import re
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Sequence, Tuple, Type, TypeVar, Union

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from .config import (
    ChatModeSettings,
    EnvConfig,
    InteractionMode,
    SubLLMSettings,
    ToolUseModeSettings,
)
from .tool_bridge import EnvironmentToolkit
from gymkhana.core.models import (
    AnswerParser,
    AnswerVerifier,
    BoxedAnswerParser,
    PipelineStats,
    SimpleEqualityVerifier,
    TrajectoryResult,
    Turn,
    RolloutStatus,
    RolloutState,
    RolloutGroup,
)
from gymkhana.core.services import ServiceContainer, SandboxService, InferenceService
if False:  # Type checking only
    from gymkhana.core.models.execution import ExecutionResult
from .prompts import BASE_SYSTEM_PROMPT
from . import parsers

# Debug utilities for pretty printing
try:
    from ..utils.debug import (
        print_repl_output, print_state, print_final_answer,
        print_task_start, print_task_result, print_messages,
        print_code_block, print_sub_agent_calls
    )
except ImportError:
    # Fallbacks if debug utils unavailable
    def print_repl_output(*args, **kwargs): pass
    def print_state(*args, **kwargs): pass
    def print_final_answer(*args, **kwargs): pass
    def print_task_start(*args, **kwargs): pass
    def print_task_result(*args, **kwargs): pass
    def print_messages(*args, **kwargs): pass
    def print_code_block(*args, **kwargs): pass

try:  # Database inserter is optional
    from ..core.services.storage.env_storage import EnvStorageService as GymkhanaDataInserter
except ImportError:  # pragma: no cover - optional dependency
    GymkhanaDataInserter = None  # type: ignore

from gymkhana.core import RewardFunction, TrajectoryMetrics, get_reward_function

try:
    from ..core.models.execution import ExecutionResult
except ImportError:
    ExecutionResult = None  # type: ignore

logger = logging.getLogger(__name__)


class SandboxSession:
    """Proxy for a specific sandbox session."""

    def __init__(self, service: SandboxService, session_id: str):
        self.service = service
        self.session_id = session_id

    async def execute(self, code: str) -> "ExecutionResult":
        return await self.service.execute(code, session_id=self.session_id)

    async def execute_bash(self, code: str, timeout: Optional[int] = None) -> "ExecutionResult":
        return await self.service.execute_bash(code, timeout=timeout, session_id=self.session_id)

    @property
    def state(self) -> Any:
        """Get the current state of the sandbox session."""
        try:
            return self.service.get_session_state(self.session_id)
        except Exception:
            return None


class EnvironmentError(RuntimeError):
    """Raised when an environment encounters a non-recoverable issue."""


class Task(BaseModel):
    """Description of a single dataset task to be processed."""

    id: str
    prompt: str
    context: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class TrajectoryState(BaseModel):
    """Mutable rollout state tracked across environment steps."""

    task: Task
    turns: List[Dict[str, Any]] = Field(default_factory=list)
    result: Optional[TrajectoryResult] = None
    extras: Dict[str, Any] = Field(default_factory=dict)


class EnvironmentRunSummary(BaseModel):
    """Aggregated results for a batch of tasks."""

    environment: str
    total_tasks: int
    successful: int
    failed: int
    num_errors: int = 0
    stats: PipelineStats
    results: List[TrajectoryResult] = Field(default_factory=list)

    model_config = ConfigDict(arbitrary_types_allowed=True)


EnvironmentFactory = Callable[[EnvConfig, Optional[Any]], "Environment"]
EnvT = TypeVar("EnvT", bound="Environment")


class EnvironmentRegistry:
    """Global registry for available environment classes."""

    def __init__(self) -> None:
        self._by_name: Dict[str, Type[Environment]] = {}
        self._by_type: Dict[str, Type[Environment]] = {}

    @staticmethod
    def _key(identifier: Any) -> str:
        value = getattr(identifier, "value", identifier)
        return str(value).strip().lower().replace("_", "-")

    def register(
        self,
        *,
        name: Optional[str] = None,
        env_type: Optional[str] = None,
    ) -> Callable[[Type[EnvT]], Type[EnvT]]:
        """Decorator for registering Environment subclasses."""

        def _decorator(cls: Type[EnvT]) -> Type[EnvT]:
            key_name = self._key(name or cls.__name__.lower())
            if key_name in self._by_name:
                raise EnvironmentError(f"Environment '{key_name}' is already registered")
            self._by_name[key_name] = cls

            if env_type is not None:
                type_key = self._key(env_type)
                if type_key in self._by_type:
                    raise EnvironmentError(
                        f"Environment type '{env_type}' already registered with "
                        f"{self._by_type[type_key].__name__}"
                    )
                self._by_type[type_key] = cls
            return cls

        return _decorator

    def create(
        self,
        identifier: Union[str],
        config: EnvConfig,
        data_inserter: Optional[GymkhanaDataInserter] = None,
        **extra,
    ) -> "Environment":
        """Instantiate an environment by name."""

        cls: Optional[Type[Environment]] = None
        key = self._key(identifier)
        cls = self._by_name.get(key) or self._by_type.get(key)

        if cls is None:
            raise EnvironmentError(f"Unknown environment '{identifier}'")

        return cls(config=config, data_inserter=data_inserter, **extra)

    def get(self, identifier: str) -> Type[Environment]:
        key = self._key(identifier)
        cls = self._by_name.get(key) or self._by_type.get(key)
        if cls is None:
            raise EnvironmentError(f"Unknown environment '{identifier}'")
        return cls

    def available(self) -> List[str]:
        return sorted(self._by_name.keys())


ENVIRONMENTS = EnvironmentRegistry()


def get_environment(identifier: str) -> Type[Environment]:
    """Convenience wrapper around the global environment registry."""
    return ENVIRONMENTS.get(identifier)


class Environment(BaseModel, ABC):
    """Base class for all Gymkhana environments."""

    config: EnvConfig
    data_inserter: Optional[Any] = Field(default=None, exclude=True)
    max_parallel_rollouts: int = Field(default=4, ge=1)
    num_rollouts: int = Field(default=1, ge=1)
    """Number of parallel rollouts per task (best-of-N, GRPO). 1 = single trajectory."""
    reward_function_name: Optional[str] = None
    enable_rewards: bool = True

    stats: PipelineStats = Field(default_factory=PipelineStats)
    name: str = Field(default_factory=lambda: "environment")
    services: Optional[ServiceContainer] = Field(default=None, description="Injected services")

    # Internal caches / runtime-only attributes
    _orchestrator: Optional[Any] = PrivateAttr(default=None)
    _inference_service: Optional[InferenceService] = PrivateAttr(default=None)
    _reward_function: Optional[RewardFunction] = PrivateAttr(default=None)
    _answer_parser: AnswerParser = PrivateAttr(default_factory=BoxedAnswerParser)
    _answer_verifier: AnswerVerifier = PrivateAttr(default_factory=SimpleEqualityVerifier)

    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    def __init__(self, **data: Any) -> None:  # type: ignore[override]
        answer_parser = data.pop("answer_parser", BoxedAnswerParser())
        answer_verifier = data.pop("answer_verifier", SimpleEqualityVerifier())

        config = data.get("config")
        if config is None:
            raise ValueError("Environment requires an 'config' argument of type EnvConfig")
        if not isinstance(config, EnvConfig):
            raise TypeError("config must be an instance of EnvConfig")

        super().__init__(**data)
        self._answer_parser = answer_parser
        self._answer_verifier = answer_verifier
        if self.reward_function_name and get_reward_function:
            self._reward_function = get_reward_function(self.reward_function_name)
        elif self.enable_rewards and get_reward_function:
            # Fall back to dataset setting when available
            reward_name = getattr(self.config.dataset, "reward_function", None)
            if reward_name:
                self._reward_function = get_reward_function(reward_name)
            else:
                self._reward_function = None

    # ------------------------------------------------------------------
    # Parser & verifier accessors
    # ------------------------------------------------------------------
    @property
    def answer_parser(self) -> AnswerParser:
        return self._answer_parser

    @answer_parser.setter
    def answer_parser(self, parser: AnswerParser) -> None:
        self._answer_parser = parser

    @property
    def answer_verifier(self) -> AnswerVerifier:
        return self._answer_verifier

    @answer_verifier.setter
    def answer_verifier(self, verifier: AnswerVerifier) -> None:
        self._answer_verifier = verifier

    # ---------------------------------------------------------------------
    # Lifecycle hooks
    # ---------------------------------------------------------------------
    async def setup(self) -> None:
        """Prepare environment resources (REPL sessions, orchestrators, etc.)."""
        self._ensure_orchestrator()

        # Link storage to legacy data_inserter if not already set
        if self.data_inserter is None and self.services and self.services.storage:
            self.data_inserter = self.services.storage

        await self.on_setup()

    async def on_setup(self) -> None:
        """Subclass hook executed during :meth:`setup`."""
        return None

    async def finalize(self) -> None:
        """Tear down resources. Subclasses should override as needed."""
        await self.on_finalize()

    async def on_finalize(self) -> None:
        return None

    # ---------------------------------------------------------------------
    # Dataset / task handling
    # ---------------------------------------------------------------------
    def load_tasks(self, limit: Optional[int] = None) -> Sequence[Task]:
        """Return Task objects to process. Must be implemented by subclass."""

        raise NotImplementedError("load_tasks must be implemented by Environment subclasses")

    async def run_task(self, task: Task) -> TrajectoryResult:
        """Execute a full trajectory for the supplied task."""
        mode = self.config.interaction_mode

        # Dispatch based on interaction mode
        if mode == InteractionMode.RLM:
            from . import rlm
            G = getattr(self.config.dataset, "num_rollouts", None) or self.num_rollouts
            if G > 1:
                return await rlm.run_rlm_multi_rollout_task(self, task, num_rollouts=G)
            return await rlm.run_rlm_task(self, task)

        elif mode == InteractionMode.TOOL_CALL:
            from .modes import ToolUseMode
            tool_mode = ToolUseMode()
            G = getattr(self.config.dataset, "num_rollouts", None) or self.num_rollouts
            if G > 1:
                # Use common batch tracking
                results = await self._execute_batch_with_tracking(
                    task, num_rollouts=G,
                    executor=lambda: tool_mode.execute_single(task, self)
                )
                return max(results, key=lambda r: (r.total_reward or 0.0, r.success))
            return await self._score_trajectory(task, await tool_mode.execute_single(task, self))

        elif mode == InteractionMode.TOOL_CALL_INTERLEAVED:
            from .modes import ToolUseInterleavedMode
            tool_mode = ToolUseInterleavedMode()
            G = getattr(self.config.dataset, "num_rollouts", None) or self.num_rollouts
            if G > 1:
                # Use common batch tracking
                results = await self._execute_batch_with_tracking(
                    task, num_rollouts=G,
                    executor=lambda: tool_mode.execute_single(task, self)
                )
                return max(results, key=lambda r: (r.total_reward or 0.0, r.success))
            return await self._score_trajectory(task, await tool_mode.execute_single(task, self))

        elif mode == InteractionMode.PLAIN_TEXT:
            from .modes import ChatMode
            from .managers import SingleTurnManager
            chat_mode = ChatMode()
            manager = SingleTurnManager()
            G = getattr(self.config.dataset, "num_rollouts", None) or self.num_rollouts
            if G > 1:
                # Use common batch tracking
                results = await self._execute_batch_with_tracking(
                    task, num_rollouts=G,
                    executor=lambda: chat_mode.execute_single(task, self, manager)
                )
                return max(results, key=lambda r: (r.total_reward or 0.0, r.success))
            return await self._score_trajectory(task, await chat_mode.execute_single(task, self, manager))

        else:
            raise EnvironmentError(f"Unsupported interaction mode: {mode}")

    async def _execute_batch_with_tracking(
        self,
        task: Task,
        num_rollouts: int,
        executor: callable,
    ) -> List[TrajectoryResult]:
        """Execute G parallel rollouts with full rollout tracking.

        This is the common rollout tracking logic used by all interaction modes.

        Args:
            task: The task to execute
            num_rollouts: Number of parallel rollouts (G)
            executor: Async callable that executes a single rollout

        Returns:
            List of G TrajectoryResults with rollout tracking
        """
        import asyncio
        import logging
        from gymkhana.core.models.trajectory import RolloutGroup, RolloutState, RolloutStatus
        from datetime import datetime

        logger = logging.getLogger("gymkhana.envs")

        # Create rollout group for tracking
        rollout_group_id = None
        rollout_states = []

        if self.data_inserter and hasattr(self.data_inserter, 'insert_rollout_group'):
            try:
                group = RolloutGroup(
                    task_id=task.id,
                    environment=self.name,
                    num_rollouts=num_rollouts,
                    config={
                        "mode": str(self.config.interaction_mode.value),
                        "max_turns": getattr(self.config.get_mode_config(), 'max_turns', None),
                    }
                )
                rollout_group_id = await self.data_inserter.insert_rollout_group(group)
                logger.debug(f"Created rollout group {rollout_group_id} for task {task.id}")

                # Create rollout states
                for i in range(num_rollouts):
                    rollout_state = RolloutState(
                        rollout_id=i,
                        status=RolloutStatus.ACTIVE,
                        started_at=datetime.now(),
                    )
                    await self.data_inserter.insert_rollout(rollout_state, rollout_group_id)
                    rollout_states.append(rollout_state)

            except Exception as e:
                logger.warning(f"Failed to create rollout group: {e}")
                import traceback
                logger.debug(traceback.format_exc())

        # Execute G rollouts in parallel
        tasks_list = [executor() for _ in range(num_rollouts)]
        results = await asyncio.gather(*tasks_list, return_exceptions=True)

        # Handle exceptions
        valid_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Rollout {i} failed: {result}")
                result = TrajectoryResult(
                    success=False,
                    final_answer="",
                    turns=[],
                    task_id=task.id,
                    environment=self.name,
                )
            valid_results.append(result)

        # Rewarding is part of execution semantics, not persistence. Always
        # score candidates so storage-free GRPO groups can select the best one.
        valid_results = [
            await self._score_trajectory(task, result) for result in valid_results
        ]

        # Set rollout tracking info on results BEFORE they're stored
        if rollout_group_id:
            for i, result in enumerate(valid_results):
                result.rollout_group_id = rollout_group_id
                result.rollout_index = i

        # Persist trajectories and update rollout tracking
        if rollout_group_id and self.data_inserter and rollout_states:
            try:
                rollout_rewards = []
                status_counts = {'completed': 0, 'failed': 0, 'error': 0, 'timeout': 0}

                for i, result in enumerate(valid_results):
                    if i >= len(rollout_states):
                        logger.warning(f"Rollout state missing for index {i}")
                        continue

                    # Update rollout state
                    rollout_states[i].status = RolloutStatus.COMPLETED if result.success else RolloutStatus.FAILED
                    rollout_states[i].completed_at = datetime.now()
                    rollout_states[i].total_reward = result.total_reward
                    rollout_states[i].num_turns = result.num_turns or 0
                    rollout_states[i].num_code_blocks = result.num_code_blocks
                    rollout_states[i].num_errors = result.num_errors

                    await self.data_inserter.update_rollout(rollout_states[i])

                    # Store rollout_id on result for later linking
                    result.rollout_id = rollout_states[i].id

                    # Persist trajectory with rollout references
                    llm_config = self.config.get_llm_config()
                    merged_metadata = {"environment": self.name, "model": llm_config.model}
                    if task.metadata:
                        merged_metadata.update(task.metadata)

                    trajectory_id = await self.data_inserter.insert_trajectory(
                        result=result,
                        task_id=task.id,
                        env_name=self.name,
                        metadata=merged_metadata,
                        rollout_id=rollout_states[i].id,
                        rollout_group_id=rollout_group_id
                    )

                    # Link rollout to trajectory
                    await self.data_inserter.link_rollout_to_trajectory(
                        rollout_states[i].id,
                        trajectory_id
                    )

                    logger.debug(f"Persisted trajectory {trajectory_id} for rollout {i}")

                    # Export to ShareGPT if enabled and answer is correct
                    if self.config.dataset.output_sharegpt and result.success and result.final_answer:
                        should_export = result.answer_correct is not False

                        if should_export:
                            # Use environment's ShareGPT builder if available
                            conversations = None
                            if hasattr(self, 'build_sharegpt_conversations'):
                                conversations = self.build_sharegpt_conversations(result, task)

                            # Fallback to standard format
                            if not conversations:
                                conversations = [{"from": "system", "value": result.system_prompt}]
                                role_map = {"user": "human", "assistant": "gpt", "tool": "tool"}

                                prev_role = "system"
                                valid_conversation = True

                                for turn in result.turns:
                                    sharegpt_role = role_map.get(turn.role, "unknown")

                                    # Validate: consecutive messages from same role should not happen
                                    if sharegpt_role == prev_role:
                                        logger.warning(
                                            f"ShareGPT validation failed for task {task.id} rollout {i}: "
                                            f"consecutive {sharegpt_role} messages detected. Skipping."
                                        )
                                        valid_conversation = False
                                        break

                                    # Build message value with reasoning if present
                                    message_value = turn.content
                                    if turn.reasoning_content and self.config.enable_reasoning:
                                        message_value = f"<think>\n{turn.reasoning_content}\n</think>\n\n{turn.content}"

                                    conversations.append({
                                        "from": sharegpt_role,
                                        "value": message_value
                                    })
                                    prev_role = sharegpt_role

                                if not valid_conversation:
                                    conversations = None

                            # Only insert if validation passed
                            if conversations and len(conversations) > 1:
                                logger.info(f"Inserting ShareGPT for task {task.id} rollout {i}")
                                export_metadata = {
                                    "env": self.name,
                                    "success": result.success,
                                    "final_answer": result.final_answer,
                                    "answer_correct": result.answer_correct,
                                    "rollout_index": i,
                                    "rollout_group_id": str(rollout_group_id),
                                }
                                export_metadata.update(
                                    self.build_sharegpt_metadata(result, task)
                                )
                                await self.data_inserter.insert_sharegpt_dataset(
                                    task_id=f"{task.id}_rollout_{i}",
                                    conversations=conversations,
                                    metadata=export_metadata,
                                )

                    # Track for group statistics
                    rollout_rewards.append(result.total_reward)
                    status_key = 'completed' if result.success else 'failed'
                    status_counts[status_key] += 1

                # Update group statistics
                await self.data_inserter.update_rollout_group_statistics(
                    rollout_group_id,
                    rollout_rewards,
                    status_counts
                )

                logger.info(f"Updated rollout group {rollout_group_id} with {num_rollouts} rollouts")

            except Exception as e:
                logger.warning(f"Failed to update rollout tracking: {e}")
                import traceback
                logger.debug(traceback.format_exc())

        return valid_results

    async def _score_trajectory(
        self, task: Task, result: TrajectoryResult
    ) -> TrajectoryResult:
        """Verify and reward one result regardless of storage configuration."""
        if not self.config.dataset.enable_rewards:
            return result
        if result.answer_correct is None:
            result.answer_correct = self.evaluate_answer(task, result)
        reward = await self.compute_reward(
            result, answer_correct=result.answer_correct, task=task
        )
        if isinstance(reward, (int, float)):
            result.total_reward = float(reward)
        if result.total_reward is None:
            result.total_reward = 0.0
        if not result.step_rewards:
            result.step_rewards = [result.total_reward]
        return result

    # ------------------------------------------------------------------
    # Hooks for subclasses to customize behaviour
    # ------------------------------------------------------------------
    def build_system_prompt(self, task: Task) -> str:
        """Build system prompt for the task.

        Combines:
        1. Base prompt: "You are a helpful AI assistant"
        2. Mode-specific instructions (from get_mode_instructions)
        3. Environment-specific instructions (from get_environment_instructions)
        """
        parts = ["You are a helpful AI assistant."]

        # Add mode-specific instructions
        mode_instructions = self.get_mode_instructions(task)
        if mode_instructions:
            parts.append(mode_instructions.strip())

        # Add environment-specific instructions
        env_instructions = self.get_environment_instructions(task)
        if env_instructions:
            parts.append(env_instructions.strip())

        return "\n\n".join(parts)

    def get_mode_instructions(self, task: Task) -> str:
        """Get mode-specific instructions (RLM, tool use, etc.).

        Override this in mode-specific base classes or environments.
        """
        from .config import InteractionMode, RLMModeSettings

        if self.config.interaction_mode == InteractionMode.RLM:
            # Build RLM-specific instructions
            mode_config = self.config.get_mode_config()
            if not isinstance(mode_config, RLMModeSettings):
                return ""

            from .prompts import (
                RLM_MODE_INSTRUCTIONS,
                REASONING_INSTRUCTIONS,
                ALLOWED_TAGS_WITH_REASONING_AND_BASH,
                ALLOWED_TAGS_WITH_REASONING_NO_BASH,
                ALLOWED_TAGS_NO_REASONING_WITH_BASH,
                ALLOWED_TAGS_NO_REASONING_NO_BASH,
            )

            reasoning_instructions = ""
            if self.config.enable_reasoning:
                reasoning_instructions = REASONING_INSTRUCTIONS
                if self.enable_bash_for_task(task):
                    allowed_tags = ALLOWED_TAGS_WITH_REASONING_AND_BASH
                else:
                    allowed_tags = ALLOWED_TAGS_WITH_REASONING_NO_BASH
            else:
                if self.enable_bash_for_task(task):
                    allowed_tags = ALLOWED_TAGS_NO_REASONING_WITH_BASH
                else:
                    allowed_tags = ALLOWED_TAGS_NO_REASONING_NO_BASH

            return RLM_MODE_INSTRUCTIONS.format(
                max_output=mode_config.repl.max_output_chars,
                reasoning_instructions=reasoning_instructions,
                allowed_tags=allowed_tags,
            )

        return ""

    def get_environment_instructions(self, task: Task) -> str:
        return ""

    def format_initial_message(self, task: Task) -> str:
        return task.prompt

    def prepare_repl_context(self, task: Task) -> Optional[str]:
        return task.context

    def enable_bash_for_task(self, task: Task) -> bool:
        return False

    def build_sub_agent_config(self) -> Dict[str, Any]:
        mode_config = self.config.get_mode_config()
        sub_llm = getattr(mode_config, "sub_llm", None) or SubLLMSettings()
        return {
            "model": sub_llm.model,
            "client": sub_llm.client.value,
            "temperature": sub_llm.temperature,
            "max_tokens": sub_llm.max_tokens,
        }

    # ------------------------------------------------------------------
    # Answer helpers (generic)
    # ------------------------------------------------------------------

    def get_expected_answer(self, task: Task) -> Optional[str]:
        meta = task.metadata or {}
        expected = meta.get("expected_answer") if isinstance(meta, dict) else None
        if isinstance(expected, str):
            return expected
        return None

    def build_sharegpt_conversations(
        self,
        result: TrajectoryResult,
        task: Task,
    ) -> Optional[List[Dict[str, Any]]]:
        """Build ShareGPT-formatted conversation from a trajectory result.

        Subclasses can override this to provide custom formatting
        (e.g. Hermes tool-calling format with <tools>, <tool_call> XML tags).

        Default implementation: generic role mapping with <think> tags for reasoning.

        Args:
            result: The TrajectoryResult to convert
            task: The original task

        Returns:
            ShareGPT conversation as list of {"from": role, "value": content} dicts,
            or None if conversion fails.
        """
        conversations: List[Dict[str, Any]] = [
            {"from": "system", "value": result.system_prompt}
        ]
        role_map = {"user": "human", "assistant": "gpt", "tool": "tool"}

        prev_role = "system"
        valid_conversation = True

        for turn in result.turns:
            sharegpt_role = role_map.get(turn.role, "unknown")

            # Skip correction prompts
            if turn.role == "user" and "[System]" in turn.content:
                if conversations and conversations[-1]["from"] == "gpt":
                    conversations.pop()
                    prev_role = conversations[-1]["from"] if conversations else "system"
                continue

            # Validate: no consecutive same-role messages
            if sharegpt_role == prev_role:
                logger.warning(
                    f"ShareGPT validation failed for task {task.id}: "
                    f"consecutive {sharegpt_role} messages. Skipping."
                )
                valid_conversation = False
                break

            # Build message with optional reasoning
            message_value = turn.content
            if turn.reasoning_content and self.config.enable_reasoning:
                message_value = f"<think>\n{turn.reasoning_content}\n</think>\n\n{turn.content}"

            conversations.append({"from": sharegpt_role, "value": message_value})
            prev_role = sharegpt_role

        return conversations if valid_conversation else None

    def build_sharegpt_metadata(
        self,
        result: TrajectoryResult,
        task: Task,
    ) -> Dict[str, Any]:
        """Return environment-specific metadata for ShareGPT exports."""
        return {}

    def extract_candidate_answers(self, result: TrajectoryResult) -> List[str]:
        """Collect candidate answers from a finished trajectory."""

        return self.answer_parser.extract_from_result(result)

    def normalize_expected_answer(self, expected: str) -> str:
        return self.answer_parser.normalize(expected)

    def normalize_candidate_answer(self, candidate: str) -> str:
        return self.answer_parser.normalize(candidate)

    def evaluate_answer(self, task: Task, result: TrajectoryResult) -> Optional[bool]:
        expected = self.get_expected_answer(task)
        if not expected or not result.final_answer:
            return None

        candidates = self.extract_candidate_answers(result)
        if not candidates:
            return None

        return self.answer_verifier.verify(
            expected=expected,
            candidates=[self.normalize_candidate_answer(candidate) for candidate in candidates],
            task_metadata=task.metadata if isinstance(task.metadata, dict) else None,
            trajectory=result,
        )

    # ------------------------------------------------------------------
    # Rollout termination policy checker
    # ------------------------------------------------------------------

    def _check_termination_policy(
        self,
        state: RolloutState,
        all_states: Sequence[RolloutState],
        *,
        turn_idx: int,
    ) -> Tuple[bool, Optional[str]]:
        """Evaluate deterministic early-termination rules in priority order.

        ``turn_idx`` is zero-based. ``all_states`` is accepted for the future
        comparative policy but deliberately does not affect decisions yet.
        """
        del all_states
        policy = self.config.rollout_termination_policy

        # Format integrity has highest priority so its actionable reason is not
        # hidden by execution errors recorded on the same turn.
        if state.num_format_violations > 0:
            format_limit = 1 if policy.terminate_on_format_violation else max(
                1, policy.max_format_violations
            )
            if state.num_format_violations >= format_limit:
                return (
                    True,
                    "Format violation: "
                    f"{state.num_format_violations}/{format_limit}",
                )

        if state.consecutive_errors >= policy.max_consecutive_errors:
            return (
                True,
                "Consecutive errors: "
                f"{state.consecutive_errors}/{policy.max_consecutive_errors}",
            )

        if state.num_errors >= policy.max_total_errors:
            return (
                True,
                f"Total errors: {state.num_errors}/{policy.max_total_errors}",
            )

        if policy.enable_max_turns_termination:
            max_turns = self.config.repl.max_turns
            if turn_idx >= max_turns - 1 and state.num_code_blocks == 0:
                return (
                    True,
                    f"Max turns ({max_turns}) reached without any successful "
                    "code execution",
                )

        return False, None


    # ------------------------------------------------------------------
    # Default execution loop
    # ------------------------------------------------------------------
    # RLM implementations moved to rlm.py

    def extract_inline_answers(self, response: str) -> List[str]:
        """Extract candidate answers directly from an assistant response."""
        return self.answer_parser.extract_inline(response)

    def get_tool_executor(self, task: Task) -> Optional[EnvironmentToolkit]:
        """Hook for subclasses to provide task-specific tools."""
        return None

    def get_refinement_prompt(self, task: Task, response: str, turn_idx: int) -> Optional[str]:
        """Hook for subclasses to provide refinement prompts in multi-turn modes."""
        return None

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------
        return self._build_simple_result(task, turns, final_answer, 0)

    def _build_simple_result(
        self,
        task: Task,
        turns: List[Turn],
        final_answer: Optional[str],
        num_code_blocks: int
    ) -> TrajectoryResult:
        """Helper to build TrajectoryResult for new modes."""
        result = TrajectoryResult(
            success=bool(final_answer),
            final_answer=final_answer or "",
            turns=turns,
            num_code_blocks=num_code_blocks,
            system_prompt=self.get_environment_instructions(task), # Simplified
            total_reward=0.0,
            step_rewards=[],
            num_errors=0,
            session_id=None,
            sandbox_state=None
        )

        # Compute rewards (synchronously for now until compute_reward is fully updated in subclass usage)
        # Note: run_task callers loop will handle compute_reward if we returned here,
        # but in _run_standard_task we do it inside.
        # For consistency, we adhere to the pattern where the runner returns a computed result.

        answer_correct = self.evaluate_answer(task, result)
        # Since we made compute_reward async, we cannot call it easily from synchronous helper.
        # But we are in async method _run_... so we should do it there.
        # Ideally, we call it here if we pass 'self' and 'await' it, but this is a helper.
        # I will let the caller handle reward computation? No, _run_standard_task does it.
        # I will revert to doing it in the caller methods.

        return result

    # ---------------------------------------------------------------------
    # Orchestration helpers
    # ---------------------------------------------------------------------
    def _ensure_orchestrator(self) -> None:
        """Initialize inference service and orchestrator."""
        # Prioritize services container
        if self.services and self.services.inference:
            self._inference_service = self.services.inference
            return

        # Default to the provider-neutral Pydantic AI v2 router.
        try:
            from gymkhana.core.services.inference import PydanticAIInferenceService

            llm_config = self.config.llm
            self._inference_service = PydanticAIInferenceService(
                default_model=getattr(
                    llm_config,
                    "model_identifier",
                    getattr(llm_config, "model", "openai:gpt-4.1-mini"),
                ),
                default_temperature=getattr(llm_config, "temperature", 0.7),
                default_max_tokens=getattr(llm_config, "max_tokens", 4096),
                data_inserter=self.data_inserter
            )
        except Exception as exc:  # pragma: no cover
            logger.warning("Failed to initialize Inference: %s", exc)
            self._orchestrator = None
            self._inference_service = None

    async def generate(self, limit: Optional[int] = None) -> EnvironmentRunSummary:
        """Run the environment over tasks selected by :meth:`load_tasks`."""

        await self.setup()
        try:
            tasks = list(self.load_tasks(limit))
            results = await self._run_tasks(tasks)
            successful = sum(1 for r in results if r and r.success)
            failed = sum(1 for r in results if not r or not r.success)
            return EnvironmentRunSummary(
                environment=self.name,
                total_tasks=len(tasks),
                successful=successful,
                failed=failed,
                stats=self.stats,
                results=[r for r in results if r],
            )
        finally:
            await self.finalize()

    async def _run_tasks(self, tasks: Sequence[Task]) -> List[Optional[TrajectoryResult]]:
        semaphore = asyncio.Semaphore(self.max_parallel_rollouts)

        async def _run(task: Task) -> Optional[TrajectoryResult]:
            async with semaphore:
                try:
                    result = await self.run_task(task)
                    self.stats.record(result)
                    return result
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    logger.exception("Task %s failed: %s", task.id, exc)
                    return None

        return list(await asyncio.gather(*[_run(task) for task in tasks]))

    # ---------------------------------------------------------------------
    # Inference utilities
    # ---------------------------------------------------------------------
    async def generate_response(
        self,
        *,
        messages: List[Dict[str, str]],
        system_prompt: str,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        tools: Optional[List[Dict]] = None,
    ) -> tuple[str, Optional[str]]:
        """Shared helper for subclasses to invoke the main LLM.

        Returns:
            Tuple of (content, reasoning_content) where reasoning_content is None if not available
        """

        if self._inference_service is None:
            raise EnvironmentError("Inference service is not available")

        response, reasoning = await self._inference_service.generate_with_reasoning(
            messages=messages,
            system_prompt=system_prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
        )

        return response, reasoning

    # ---------------------------------------------------------------------
    # Reward + verification hooks
    # ---------------------------------------------------------------------
    async def compute_reward(
        self,
        result: TrajectoryResult,
        *,
        answer_correct: Optional[bool] = None,
        metadata: Optional[Dict[str, Any]] = None,
        task: Optional[Task] = None,
    ) -> Optional[Dict[str, Any]]:
        if not self.enable_rewards or not self._reward_function:
            return None

        # Get max_turns from mode_config
        mode_config = self.config.get_mode_config()
        max_turns = None
        if mode_config:
            from .config import RLMModeSettings, ToolUseModeSettings, ChatModeSettings
            if isinstance(mode_config, RLMModeSettings):
                max_turns = mode_config.repl.max_turns
            elif isinstance(mode_config, (ToolUseModeSettings, ChatModeSettings)):
                max_turns = mode_config.max_turns

        metrics = TrajectoryMetrics(
            answer_correct=answer_correct,
            num_turns=len(result.turns),
            num_code_blocks=result.num_code_blocks,
            num_errors=result.num_errors,
            max_turns=max_turns or 20,  # Default fallback
            intermediate_rewards=result.step_rewards,
            success=result.success,
        )
        # Check for async compute support (LLM judges)
        if hasattr(self._reward_function, 'compute_async'):
            reward_result = await self._reward_function.compute_async(metrics, task=task, trajectory=result)
        else:
            reward_result = self._reward_function.compute(metrics)

        result.total_reward = reward_result.get("total_reward", result.total_reward)
        final_step = reward_result.get("final_step_reward")
        if final_step is not None:
            result.step_rewards.append(final_step)
        return reward_result.get("metadata")

    def verify_answer(self, result: TrajectoryResult, task: Task) -> Optional[bool]:
        """Optional hook for subclasses to validate final answers."""

        return None

    # ---------------------------------------------------------------------
    # REPL helpers (optional default implementation)
    # ---------------------------------------------------------------------
    @asynccontextmanager
    async def repl_session(
        self,
        *,
        context: Optional[str] = None,
        enable_bash: bool = False,
        sub_agent_config: Optional[Dict[str, Any]] = None,
    ) -> AsyncIterator[SandboxSession]:
        """Create a single async sandbox session. Requires services.sandbox (inject via ServiceContainer)."""
        if not self.services or not self.services.sandbox:
            raise EnvironmentError(
                "Sandbox service is required. Pass services=ServiceContainer(sandbox=...) when creating the environment."
            )

        # Get max_output_chars from RLM mode config
        mode_config = self.config.get_mode_config()
        from .config import RLMModeSettings
        max_output_chars = 8192  # Default
        if isinstance(mode_config, RLMModeSettings):
            max_output_chars = mode_config.repl.max_output_chars

        session_id = await self.services.sandbox.create_session(
            context=context,
            max_output_chars=max_output_chars,
            sub_agent_config=sub_agent_config,
            enable_bash=enable_bash,
        )
        try:
            yield SandboxSession(self.services.sandbox, session_id)
        finally:
            await self.services.sandbox.delete_session(session_id)

    @asynccontextmanager
    async def repl_sessions(
        self,
        *,
        context: Optional[str] = None,
        enable_bash: bool = False,
        sub_agent_config: Optional[Dict[str, Any]] = None,
        num_sessions: int = 1,
    ) -> AsyncIterator[List[SandboxSession]]:
        """Create multiple async sandbox sessions (e.g. for parallel rollouts). Requires services.sandbox."""
        if num_sessions < 1:
            yield []
            return
        if not self.services or not self.services.sandbox:
            raise EnvironmentError(
                "Sandbox service is required. Pass services=ServiceContainer(sandbox=...) when creating the environment."
            )

        # Get max_output_chars from RLM mode config
        mode_config = self.config.get_mode_config()
        from .config import RLMModeSettings
        max_output_chars = 8192  # Default
        if isinstance(mode_config, RLMModeSettings):
            max_output_chars = mode_config.repl.max_output_chars

        session_ids: List[str] = []
        try:
            # Create sessions with small delay to avoid overwhelming server
            for i in range(num_sessions):
                sid = await self.services.sandbox.create_session(
                    context=context,
                    max_output_chars=max_output_chars,
                    sub_agent_config=sub_agent_config,
                    enable_bash=enable_bash,
                )
                session_ids.append(sid)
                # Small delay every 10 sessions to avoid rate limiting
                if (i + 1) % 10 == 0 and i + 1 < num_sessions:
                    await asyncio.sleep(0.1)
            yield [SandboxSession(self.services.sandbox, sid) for sid in session_ids]
        finally:
            for sid in session_ids:
                try:
                    await self.services.sandbox.delete_session(sid)
                except Exception as e:
                    logger.warning("Failed to delete session %s: %s", sid, e)

    # ------------------------------------------------------------------
    # Response parsing utilities (ported from legacy pipeline)
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_python_blocks(text: str) -> List[str]:
        return parsers.extract_python_blocks(text)

    @staticmethod
    def _extract_boxed_answers(text: str) -> List[str]:
        return parsers.extract_boxed_answers(text)

    @staticmethod
    def _extract_final_answer(text: str) -> Optional[str]:
        return parsers.extract_final_answer(text)

    @staticmethod
    def _has_hallucinated_repl(text: str) -> bool:
        return parsers.has_hallucinated_repl(text)

    async def run(self, limit: Optional[int] = None) -> EnvironmentRunSummary:
        """Run the environment on a batch of tasks."""
        await self.setup()
        tasks = list(self.load_tasks(limit))

        if self.config.debug and tasks:
            task = tasks[0]
            print_task_start(task.id, task.prompt, getattr(task, 'expected_answer', None))
            print_messages([{"role": "system", "content": self.build_system_prompt(task)},
                           {"role": "user", "content": self.format_initial_message(task)}],
                           title="Initial Messages", truncate=False)

        # Run tasks in parallel
        results = await self._run_tasks(tasks)

        # Filter out None results and handle persistence
        valid_results = [r for r in results if r is not None]

        async def persist_result(task: Task, result: TrajectoryResult):
            if not self.data_inserter:
                return

            # Merge task metadata (includes expected_answer) with environment metadata
            llm_config = self.config.get_llm_config()
            merged_metadata = {"environment": self.name, "model": llm_config.model}
            if task.metadata:
                merged_metadata.update(task.metadata)

            trajectory_id = await self.data_inserter.insert_trajectory(
                result=result,
                task_id=task.id,
                env_name=self.name,
                metadata=merged_metadata
            )

            # Record sandbox session if available
            if result.session_id and result.sandbox_state:
                try:
                    await self.data_inserter.insert_sandbox_session(
                        session_state=result.sandbox_state,
                        environment=self.name,
                        trajectory_id=trajectory_id
                    )
                except Exception as e:
                    logging.getLogger("gymkhana.envs").error(f"Error recording sandbox session: {e}")

            # Insert ShareGPT dataset if enabled (only for successful rollouts with correct answers)
            # For tasks with expected answers: skip if answer_correct=False
            # For tasks without expected answers (e.g., SWE): use reward threshold
            should_export_sharegpt = False
            if self.config.dataset.output_sharegpt and result.success and result.final_answer:
                expected_answer = self.get_expected_answer(task)
                if expected_answer is not None:
                    # Task has expected answer - check correctness
                    should_export_sharegpt = result.answer_correct is not False
                else:
                    # Task has no expected answer (e.g., SWE) - use reward threshold
                    # Export if total_reward > 0.5 (indicates good quality trajectory)
                    should_export_sharegpt = result.total_reward > 0.5

            logger.info(f"ShareGPT export decision for task {task.id}: should_export={should_export_sharegpt}, "
                       f"output_sharegpt={self.config.dataset.output_sharegpt}, success={result.success}, "
                       f"final_answer={bool(result.final_answer)}, answer_correct={result.answer_correct}")

            if should_export_sharegpt:
                conversations = self.build_sharegpt_conversations(result, task)

                # Only insert if we have meaningful content
                if conversations and len(conversations) > 1:  # More than just system prompt
                    logger.info(f"Inserting ShareGPT dataset for task {task.id} with {len(conversations)} messages")
                    export_metadata = {
                        "env": self.name,
                        "success": result.success,
                        "final_answer": result.final_answer,
                        "num_code_blocks": result.num_code_blocks,
                    }
                    export_metadata.update(
                        self.build_sharegpt_metadata(result, task)
                    )
                    await self.data_inserter.insert_sharegpt_dataset(
                        task_id=task.id,
                        conversations=conversations,
                        metadata=export_metadata,
                    )
                    logger.info(f"Successfully inserted ShareGPT dataset for task {task.id}")
                else:
                    logger.warning(f"Skipping ShareGPT insertion for task {task.id}: valid={valid_conversation}, num_messages={len(conversations)}")

        # Bulk persist results (skip when num_rollouts > 1: multi-rollout already persisted all G in _run_multi_rollout_task)
        if self.data_inserter and self.num_rollouts == 1:
            persist_coros = []
            for task, result in zip(tasks, results):
                if result:
                    persist_coros.append(persist_result(task, result))
            if persist_coros:
                await asyncio.gather(*persist_coros)

        summary = EnvironmentRunSummary(
            environment=self.name,
            total_tasks=len(tasks),
            successful=sum(1 for r in valid_results if r.success),
            failed=sum(1 for r in valid_results if not r.success),
            num_errors=sum(1 for r in valid_results if not r.success),
            stats=self.stats,
            results=valid_results,
        )

        # Auto-generate rollout tracking report if rollouts were used
        if self.data_inserter and self.num_rollouts > 1:
            try:
                logger.info("Generating rollout tracking report...")
                import subprocess
                import sys
                result = subprocess.run(
                    [sys.executable, "scripts/generate_rollout_summary.py"],
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                if result.returncode == 0:
                    print("\n" + result.stdout)
                else:
                    logger.warning(f"Rollout report generation failed: {result.stderr}")
            except Exception as e:
                logger.warning(f"Failed to generate rollout report: {e}")

        return summary


def register_environment(
    *,
    name: Optional[str] = None,
    env_type: Optional[str] = None,
) -> Callable[[Type[EnvT]], Type[EnvT]]:
    """Convenience decorator that proxies to the global registry."""

    return ENVIRONMENTS.register(name=name, env_type=env_type)


__all__ = [
    "Environment",
    "EnvironmentError",
    "EnvironmentRegistry",
    "EnvironmentRunSummary",
    "Task",
    "TrajectoryState",
    "ENVIRONMENTS",
    "register_environment",
]

Environment.model_rebuild()
TrajectoryState.model_rebuild()
EnvironmentRunSummary.model_rebuild()
