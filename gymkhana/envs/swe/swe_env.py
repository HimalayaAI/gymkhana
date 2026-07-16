"""SWE (Software Engineering) environment implementation.

This environment supports code editing and bug fixing tasks with Docker containers
that provide access to repository code at /testbed.
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar, Dict, Iterable, List, Optional, Sequence

from datasets import load_dataset
from pydantic import Field

from gymkhana.core.models.tasks import TaskMetadata, TestResult, CodePatch
from .swe_prompt import SWE_SYSTEM_PROMPT

from ..config import (
    DatasetSettings,
    EnvConfig,
    EnvironmentType,
    LLMClientType,
    REPLSettings,
    RLMEnvConfig,
    RolloutTerminationPolicy,
    SubLLMSettings,
)
from ..environment import Environment, EnvironmentError, Task, register_environment
from ..workspace_utils import format_workspace_context

# Import SWE-specific reward functions to register them
from . import swe_rewards  # noqa: F401

logger = logging.getLogger(__name__)


# =============================================================================
# SWE-Specific Subclasses (extending generic environment models)
# =============================================================================


class SWETaskMetadata(TaskMetadata):
    """SWE-specific task metadata.

    Extends TaskMetadata with fields needed for SWE-bench style tasks.
    """

    repo: str = Field(
        default="",
        description="Repository name (e.g., 'django/django')"
    )

    commit: Optional[str] = Field(
        default=None,
        description="Base commit hash for the task"
    )

    image_name: Optional[str] = Field(
        default=None,
        description="Docker image name for this task"
    )

    problem_statement: str = Field(
        default="",
        description="The problem statement describing the bug/feature"
    )

    hints: Optional[str] = Field(
        default=None,
        description="Optional hints for solving the task"
    )

    patch: Optional[str] = Field(
        default=None,
        description="Reference patch (gold solution)"
    )

    test_patch: Optional[str] = Field(
        default=None,
        description="Test code to verify the fix"
    )


class SWETestResult(TestResult):
    """SWE-specific test result.

    Extends TestResult with SWE-bench evaluation metrics.
    """

    tests_passed: int = Field(
        default=0,
        description="Number of tests that passed"
    )

    tests_failed: int = Field(
        default=0,
        description="Number of tests that failed"
    )

    tests_error: int = Field(
        default=0,
        description="Number of tests with errors"
    )

    resolution_status: str = Field(
        default="",
        description="Resolution status (e.g., 'RESOLVED_FULL', 'RESOLVED_PARTIAL', 'NOT_RESOLVED')"
    )

    fail_to_pass: List[str] = Field(
        default_factory=list,
        description="Tests that went from failing to passing"
    )

    pass_to_pass: List[str] = Field(
        default_factory=list,
        description="Tests that remained passing"
    )


class SWEPatch(CodePatch):
    """SWE-specific code patch.

    Extends CodePatch with SWE-bench specific fields.
    """

    repo: str = Field(
        default="",
        description="Repository name"
    )

    commit: Optional[str] = Field(
        default=None,
        description="Commit this patch applies to"
    )

    model_name_or_path: str = Field(
        default="",
        description="Model that generated this patch"
    )


# =============================================================================
# SWE Environment Configuration
# =============================================================================


ENV_INSTRUCTIONS = (
    "## SWE Task Instructions\n"
    "- Use bash commands for file exploration and running tests.\n"
    "- Use Python for reading files and making precise edits.\n"
    "- Always verify your changes by running the relevant tests.\n"
    "- Provide a clear summary of changes in your final answer.\n"
)


DEFAULT_SWE_CONFIG = RLMEnvConfig(
    name="swe",
    main_model="Hermes-4-405B",
    main_client=LLMClientType.LITELLM,
    main_temperature=0.7,
    main_max_tokens=8192,
    repl=REPLSettings(
        server_url="http://localhost:5003",
        max_output_chars=16384,
        max_output_lines=1000,
        timeout_seconds=300,
        max_turns=20,  # SWE tasks often need more iterations
    ),
    sub_llm=SubLLMSettings(
        model="Hermes-4-70B",
        client=LLMClientType.LITELLM,
        max_parallel=4,
        timeout_seconds=60,
        max_tokens=2048,
        temperature=0.5,
    ),
    # More lenient termination policy for SWE tasks
    # SWE tasks are complex and may require trial and error
    rollout_termination_policy=RolloutTerminationPolicy(
        terminate_on_format_violation=True,  # Still enforce format
        max_format_violations=1,
        max_consecutive_errors=5,  # More lenient (default: 3)
        max_total_errors=10,  # More lenient (default: 5)
        enable_max_turns_termination=True,
        min_code_blocks_before_answer=1,
    ),
    dataset=DatasetSettings(
        environment=EnvironmentType.SWE,
        dataset_name="SWE-bench/SWE-smith-py",
        dataset_config=None,
        dataset_split="train",
        field_mapping={
            "id": "instance_id",
            "prompt": "problem_statement",
            "expected_answer": None,  # SWE uses test-based evaluation
            "context": None,
        },
        filter_repos=["swesmith/pytest-dev__iniconfig.16793ead"],
        batch_size=4,
        num_rollouts=1,
        limit=10,
        include_instructions=True,
        output_dir="outputs/gymkhana",
        output_sharegpt=True,
        mask_observations=False,
        enable_rewards=True,
        reward_function="swe_progress",  # Use SWE-specific reward function
    ),
    debug=False,
)


# =============================================================================
# SWE Environment Class
# =============================================================================


@register_environment(name="swe", env_type=EnvironmentType.SWE)
class SWEEnv(Environment):
    """SWE environment for code editing and bug fixing tasks.

    - Loads tasks from SWE-bench format datasets
    - Optionally uses Docker containers with repository code at /testbed
    - Supports both Python and bash execution
    - Evaluates based on test results

    Supports parallel rollouts per task (config.dataset.num_rollouts > 1):
    overrides repl_sessions so that Docker tasks use one container with G
    REPL sessions; non-Docker tasks use base repl_sessions with G sessions.
    """

    name: str = "swe"
    default_config: ClassVar[RLMEnvConfig] = DEFAULT_SWE_CONFIG

    _current_task: Optional[Task] = None

    def __init__(self, *, config: Optional[EnvConfig] = None, **data: Any) -> None:
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
        """Load SWE dataset."""
        cfg = self.config.dataset
        if not cfg.dataset_name:
            raise EnvironmentError("SWEEnv requires dataset.dataset_name to be set")

        try:
            if cfg.dataset_config:
                ds = load_dataset(
                    cfg.dataset_name,
                    cfg.dataset_config,
                    split=cfg.dataset_split,
                    streaming=True,
                )
            else:
                ds = load_dataset(
                    cfg.dataset_name,
                    split=cfg.dataset_split,
                    streaming=True,
                )

            if cfg.dataset_seed is not None:
                ds = ds.shuffle(seed=cfg.dataset_seed, buffer_size=10)

            # Filter by repos if specified
            if cfg.filter_repos:
                ds = ds.filter(lambda x: x.get("repo") in cfg.filter_repos)

            print(f"Using streaming mode for {cfg.dataset_name} (split={cfg.dataset_split})")
            return ds

        except ImportError as exc:
            raise EnvironmentError("datasets package is required for SWEEnv") from exc
        except Exception as exc:
            raise EnvironmentError(f"Failed to load dataset '{cfg.dataset_name}': {exc}") from exc

    def load_tasks(self, limit: Optional[int] = None) -> Sequence[Task]:
        """Load tasks from SWE dataset."""
        dataset_limit = limit or self.config.dataset.limit
        mapping = self.config.dataset.field_mapping
        records = self._load_dataset()

        tasks: List[Task] = []
        seen = 0
        id_field = mapping.get("id") or "instance_id"
        prompt_field = mapping.get("prompt") or "problem_statement"

        for record in records:
            if dataset_limit is not None and seen >= dataset_limit:
                break

            task_id = record.get(id_field, str(seen))
            prompt = record.get(prompt_field)

            if not prompt:
                continue

            # Build SWE-specific metadata
            metadata = SWETaskMetadata(
                task_id=str(task_id),
                environment="swe",
                source_dataset=self.config.dataset.dataset_name,
                repo=record.get("repo", ""),
                commit=record.get("base_commit"),
                image_name=record.get("docker_image") or record.get("image_name"),
                problem_statement=str(prompt),
                hints=record.get("hints_text") or record.get("hints"),
                patch=record.get("patch"),
                test_patch=record.get("test_patch"),
                extra={
                    k: v
                    for k, v in record.items()
                    if k not in {id_field, prompt_field, "repo", "base_commit",
                                "docker_image", "image_name", "hints_text",
                                "hints", "patch", "test_patch"}
                },
            )

            tasks.append(
                Task(
                    id=str(task_id),
                    prompt=str(prompt),
                    context=None,  # SWE tasks don't use context in the same way
                    metadata=metadata.to_dict(),
                )
            )
            seen += 1

        return tasks

    # ------------------------------------------------------------------
    # Behaviour hooks
    # ------------------------------------------------------------------

    def prepare_repl_context(self, task: Task) -> Optional[str]:
        """Return None - SWE tasks don't need context uploaded."""
        return None

    async def run_task(self, task: Task) -> Any:
        self._current_task = task
        try:
            return await super().run_task(task)
        finally:
            self._current_task = None

    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def repl_session(
        self,
        *,
        context: Optional[str] = None,
        enable_bash: bool = False,
        sub_agent_config: Optional[Dict[str, Any]] = None,
    ):
        task = self._current_task
        if (
            task
            and not self.config.dataset.skip_docker
            and task.metadata.get("image_name")
        ):
            # Use DockerSandboxService with unified config
            from gymkhana.core.services.sandboxes import DockerSandboxService
            from gymkhana.core.models.sandbox import SandboxConfig
            from ..environment import SandboxSession

            image_name = task.metadata["image_name"]
            instance_id = task.id

            # Create a dedicated configuration for this SWE task
            config = SandboxConfig.for_swe(docker_image=image_name)

            # Apply task-specific overrides from environment's REPL settings
            config.session.max_output_chars = self.config.repl.max_output_chars
            config.session.max_iterations = self.config.repl.max_turns
            config.session.enable_bash = enable_bash

            sandbox = DockerSandboxService(
                config=config,
                instance_id=instance_id
            )

            # Create session
            session_id = await sandbox.create_session(
                context=context,
                sub_agent_config=sub_agent_config,
            )
            try:
                session = SandboxSession(sandbox, session_id)
                yield session
            finally:
                await sandbox.delete_session(session_id)
        else:
            # Standard local REPL
            async with super().repl_session(
                context=context,
                enable_bash=enable_bash,
                sub_agent_config=sub_agent_config
            ) as repl:
                yield repl

    @asynccontextmanager
    async def repl_sessions(
        self,
        *,
        context: Optional[str] = None,
        enable_bash: bool = False,
        sub_agent_config: Optional[Dict[str, Any]] = None,
        num_sessions: int = 1,
    ):
        """Create multiple sessions; for Docker tasks uses one container with num_sessions REPL sessions."""
        task = self._current_task
        if (
            task
            and not self.config.dataset.skip_docker
            and task.metadata.get("image_name")
            and num_sessions >= 1
        ):
            from gymkhana.core.services.sandboxes import DockerSandboxService
            from gymkhana.core.models.sandbox import SandboxConfig
            from ..environment import SandboxSession

            image_name = task.metadata["image_name"]
            instance_id = task.id
            config = SandboxConfig.for_swe(docker_image=image_name)
            config.session.max_output_chars = self.config.repl.max_output_chars
            config.session.max_iterations = self.config.repl.max_turns
            config.session.enable_bash = enable_bash

            sandbox = DockerSandboxService(config=config, instance_id=instance_id)
            session_ids: List[str] = []
            try:
                for _ in range(num_sessions):
                    sid = await sandbox.create_session(
                        context=context,
                        sub_agent_config=sub_agent_config,
                    )
                    session_ids.append(sid)
                yield [SandboxSession(sandbox, sid) for sid in session_ids]
            finally:
                for sid in session_ids:
                    try:
                        await sandbox.delete_session(sid)
                    except Exception:
                        pass
        else:
            async with super().repl_sessions(
                context=context,
                enable_bash=enable_bash,
                sub_agent_config=sub_agent_config,
                num_sessions=num_sessions,
            ) as repls:
                yield repls

    # ------------------------------------------------------------------
    # Behaviour hooks
    # ------------------------------------------------------------------

    def get_environment_instructions(self, task: Task) -> str:
        """Return SWE-specific instructions."""
        if not self.config.dataset.include_instructions:
            return ""
        return ENV_INSTRUCTIONS

    def build_system_prompt(self, task: Task) -> str:
        """Build system prompt for SWE tasks."""
        return SWE_SYSTEM_PROMPT

    def format_initial_message(self, task: Task) -> str:
        """Format the initial user message for SWE tasks."""
        meta = task.metadata

        # Build problem description
        prompt = task.prompt

        # Add repository info if available
        repo = meta.get("repo", "")
        if repo:
            prompt += f"\n\nRepository: {repo}"

        # Add hints if available
        hints = meta.get("hints")
        if hints:
            prompt += f"\nHints: {hints}"

        return prompt

    def enable_bash_for_task(self, task: Task) -> bool:
        """SWE tasks always need bash."""
        return True

    def accept_response_without_code(
        self, response: str, *, num_code_blocks: int
    ) -> Optional[str]:
        """SWE tasks should always involve code execution."""
        if num_code_blocks > 0:
            candidates = self.extract_inline_answers(response)
            if candidates:
                return response
        return None

    def get_expected_answer(self, task: Task) -> Optional[str]:
        """SWE tasks use test-based evaluation, not string comparison."""
        # Return the reference patch if available (for analysis, not direct comparison)
        return task.metadata.get("patch")

    # ------------------------------------------------------------------
    # Docker integration placeholder
    # ------------------------------------------------------------------

    # NOTE: Docker integration will be added in Phase 2
    # The DockerSandboxService will provide container-based execution
    # with access to /testbed


__all__ = [
    "SWEEnv",
    "SWETaskMetadata",
    "SWETestResult",
    "SWEPatch",
    "DEFAULT_SWE_CONFIG",
]
