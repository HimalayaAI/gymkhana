"""Pydantic environment configuration models."""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class InteractionMode(str, Enum):
    """How the environment interacts with the model.

    RLM:                    Recursive Language Model — <python> → sandbox → <repl>
                            loop with optional sub-LLM calls. Used by existing envs.
    TOOL_CALL:              Regular tool calls: <think>…</think> then <tool_call>…
                            </tool_call> blocks, executed sequentially.
    TOOL_CALL_INTERLEAVED:  Interleaved tool calls: <tool_call> appears INSIDE
                            <think> blocks. Uses stop-execute-continue pattern.
    PLAIN_TEXT:             Text generation only, no tool use.
    """
    RLM = "rlm"
    TOOL_CALL = "tool_call"
    TOOL_CALL_INTERLEAVED = "tool_call_interleaved"
    PLAIN_TEXT = "plain_text"


class LLMClientType(Enum):
    """Supported LLM clients."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    LITELLM = "litellm"


class InferenceConfig(BaseModel):
    """Provider-neutral model configuration for Pydantic AI routing."""

    model_config = ConfigDict(validate_assignment=True)

    model: str = "openai:gpt-4.1-mini"
    client: LLMClientType = LLMClientType.OPENAI
    temperature: Optional[float] = None
    max_tokens: int = Field(default=4096, ge=1)

    @property
    def model_identifier(self) -> str:
        if ":" in self.model:
            return self.model
        return f"{self.client.value}:{self.model}"


class EnvironmentType(str, Enum):
    """Compatibility names for the environments ported from Gymkhana."""

    MATH_PYTHON = "math-python"
    OOLONG = "oolong"
    HOTPOTQA = "hotpotqa"
    SWE = "swe"
    IFEVAL = "ifeval"
    ROMANIZED_NEPALI = "romanized-nepali"
    ENGLISH_SHAREGPT_TO_NEPALI = "english-sharegpt-to-nepali"


class REPLSettings(BaseModel):
    """Configuration for the Python REPL sandbox."""

    model_config = ConfigDict(validate_assignment=True)

    server_url: str = "http://localhost:5003"
    max_output_chars: int = 8192
    max_output_lines: int = 500
    timeout_seconds: int = 120
    max_turns: int = 20


class SubLLMSettings(BaseModel):
    """Configuration for sub-LLM helper calls."""

    model_config = ConfigDict(validate_assignment=True)

    model: str = "Hermes-4-70B"
    client: LLMClientType = LLMClientType.LITELLM
    max_parallel: int = 8
    timeout_seconds: int = 60
    max_tokens: int = 4096
    temperature: float = 0.7


class ModeConfig(BaseModel):
    """Base class for mode-specific configuration settings.

    Subclass this for each interaction mode's specific settings.
    """
    model_config = ConfigDict(validate_assignment=True, arbitrary_types_allowed=True)


class RLMModeSettings(ModeConfig):
    """Configuration for RLM (Recursive Language Model) mode environments.

    Controls the RLM interaction loop with REPL sandbox and optional sub-LLM.
    """

    repl: REPLSettings = Field(
        default_factory=REPLSettings,
        description="Python REPL sandbox configuration"
    )
    sub_llm: SubLLMSettings = Field(
        default_factory=SubLLMSettings,
        description="Sub-LLM configuration for helper tasks (code generation, etc.)"
    )
    enable_correction_feedback: bool = Field(
        default=False,
        description="If True, add correction prompts when model responds without code. "
                   "If False, terminate rollout immediately (recommended for RL training with parallel rollouts)."
    )


class ToolUseModeSettings(ModeConfig):
    """Configuration for tool-use mode environments.

    Controls the tool-call interaction loop. Set max_turns=1 for single-turn
    tool use, or >1 for multi-hop reasoning with tools.
    """

    max_turns: int = Field(
        default=10,
        ge=1,
        description="Max tool-call rounds. 1 = single-turn, >1 = multi-turn."
    )
    parallel_tool_calls: bool = Field(
        default=False,
        description="Allow parallel tool calls within a single turn."
    )
    tool_choice: Optional[str] = Field(
        default=None,
        description="Force tool choice: 'auto', 'required', 'none', or a specific function name."
    )


class ChatModeSettings(ModeConfig):
    """Configuration for chat mode environments (plain text generation).

    Controls text generation. Set max_turns=1 for single-turn Q&A
    or >1 for iterative self-refinement.
    """

    max_turns: int = Field(
        default=1,
        ge=1,
        description="1 = single-turn answer, >1 = multi-turn with self-refinement."
    )
    allow_self_refinement: bool = Field(
        default=False,
        description="Enable iterative self-refinement across multiple turns."
    )


class LLMJudgeSettings(BaseModel):
    """Configuration for LLM-as-judge reward computation.

    Used when tasks lack verifiable ground-truth answers and require
    a secondary LLM to evaluate trajectory quality.
    """

    model_config = ConfigDict(validate_assignment=True)

    model: str = Field(
        ...,
        description="Judge model name (e.g. 'gpt-4o-mini')."
    )
    client: LLMClientType = Field(
        default=LLMClientType.LITELLM,
        description="LLM client for judge calls."
    )
    temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        description="Low temperature for scoring consistency."
    )
    max_tokens: int = Field(
        default=1024,
        ge=1,
        description="Max tokens for judge response."
    )
    rubric_prompt: Optional[str] = Field(
        default=None,
        description="Custom rubric/scoring prompt. If None, uses default rubric."
    )
    num_judges: int = Field(
        default=1,
        ge=1,
        description="Number of judge samples for majority vote."
    )


class RolloutTerminationPolicy(BaseModel):
    """Configurable policies for early rollout termination.

    Focus on clear, actionable rules that prevent training on erroneous trajectories.
    """

    model_config = ConfigDict(validate_assignment=True)

    # Format adherence (PRIMARY POLICY - bare minimum quality)
    terminate_on_format_violation: bool = Field(
        default=True,
        description="Terminate immediately if model violates format rules (hallucinated tags, malformed XML)"
    )
    max_format_violations: int = Field(
        default=1,
        ge=0,
        description="Maximum number of format violations before termination (if not immediate)"
    )

    # Error-based termination (PRIMARY POLICIES)
    max_consecutive_errors: int = Field(
        default=3,
        ge=1,
        description="Terminate after N consecutive code execution errors (syntax, runtime, etc.)"
    )
    max_total_errors: int = Field(
        default=5,
        ge=1,
        description="Terminate after N total errors in the rollout, regardless of recovery"
    )

    # Turn-based termination
    enable_max_turns_termination: bool = Field(
        default=True,
        description="Terminate when max_turns reached without final answer (already implicit, but tracked)"
    )

    # Code quality filters
    min_code_blocks_before_answer: int = Field(
        default=1,
        ge=0,
        description="Require at least N successful code executions before accepting final answer"
    )

    # Comparative termination (TODO - Future Enhancement)
    enable_comparative_termination: bool = Field(
        default=False,
        description="[TODO] Enable termination based on comparison with other rollouts in batch"
    )
    comparative_error_threshold: int = Field(
        default=3,
        ge=1,
        description="[TODO] If enabled, terminate if this rollout has N more errors than the best rollout"
    )
    comparative_turn_threshold: int = Field(
        default=3,
        ge=1,
        description="[TODO] If enabled, terminate if this rollout is N turns behind best completed rollout"
    )


class DatasetSettings(BaseModel):
    """Dataset ingestion and output controls."""

    model_config = ConfigDict(validate_assignment=True)

    environment: str = "custom"  # Environment name/type (e.g., "ifeval", "math-python")
    dataset_name: Optional[str] = None
    dataset_config: Optional[str] = None
    dataset_split: str = "train"
    dataset_backend: Literal[
        "auto", "local", "huggingface", "huggingface-rows"
    ] = "auto"
    dataset_offset: int = Field(default=0, ge=0)
    dataset_seed: Optional[int] = None
    field_mapping: Dict[str, Optional[str]] = Field(
        default_factory=lambda: {
            "id": "id",
            "prompt": "input",
            "expected_answer": "output",
            "context": None,
        }
    )
    context_processor: Optional[str] = None
    filter_repos: Optional[list[str]] = None
    skip_docker: bool = False
    batch_size: int = 4
    num_rollouts: int = 1
    """Number of parallel rollouts per task (for GRPO / best-of-N). Default 1 = single trajectory."""
    limit: Optional[int] = None
    include_instructions: bool = True
    output_dir: str = "outputs/gymkhana"
    output_basename: str = "sharegpt"
    output_sharegpt: bool = True
    output_audit_jsonl: bool = True
    mask_observations: bool = False
    enable_rewards: bool = True
    reward_function: str = "simple"


class EnvConfig(BaseModel):
    """Top-level environment configuration used by Environment subclasses.

    Use :class:`InferenceConfig` for provider-neutral Pydantic AI routing and
    ``mode_config`` for interaction-specific settings.
    """

    model_config = ConfigDict(validate_assignment=True, arbitrary_types_allowed=True)

    name: str

    # Primary LLM configuration
    llm: Optional[InferenceConfig] = Field(
        default=None,
        description="Primary Pydantic AI model routing configuration"
    )

    # Interaction mode
    interaction_mode: InteractionMode = Field(
        default=InteractionMode.PLAIN_TEXT,
        description="How the environment interacts with the model: RLM, TOOL_CALL, or PLAIN_TEXT."
    )

    # Unified mode-specific configuration
    mode_config: Optional[ModeConfig] = Field(
        default=None,
        description="Mode-specific settings (RLMModeSettings, ToolUseModeSettings, ChatModeSettings). "
                   "Type depends on interaction_mode."
    )

    # Common features across all modes
    enable_reasoning: bool = Field(
        default=False,
        description="Allow model to use <think>...</think> blocks for chain-of-thought reasoning"
    )

    # Optional features
    llm_judge: Optional[LLMJudgeSettings] = Field(
        default=None,
        description="Settings for LLM-as-judge reward. Used when verifiable answers are unavailable."
    )

    # Dataset and execution settings
    dataset: DatasetSettings = Field(default_factory=DatasetSettings)
    # Compatibility view for pre-mode_config RLM callers.  Keeping this typed
    # field also lets old YAML/fixtures use ``repl.max_turns`` while the runtime
    # continues to consume ``RLMModeSettings``.
    repl: REPLSettings = Field(default_factory=REPLSettings)
    rollout_termination_policy: RolloutTerminationPolicy = Field(
        default_factory=RolloutTerminationPolicy,
        description="Policy for early termination of failing rollouts"
    )
    debug: bool = False

    def get_llm_config(self) -> InferenceConfig:
        """Get the primary LLM configuration.

        Raises:
            ValueError: If no LLM configuration is set
        """
        if self.llm is not None:
            return self.llm

        raise ValueError(
            "No LLM configuration found. Set llm to an InferenceConfig."
        )

    def get_mode_config(self) -> Optional[ModeConfig]:
        """Get mode-specific configuration.

        Returns mode_config if set, otherwise returns default settings based on interaction_mode.
        """
        if self.mode_config is not None:
            return self.mode_config

        # Return default settings based on mode
        if self.interaction_mode == InteractionMode.TOOL_CALL:
            return ToolUseModeSettings()
        elif self.interaction_mode == InteractionMode.PLAIN_TEXT:
            return ChatModeSettings()
        elif self.interaction_mode == InteractionMode.RLM:
            return RLMModeSettings(repl=self.repl)

        return None

    def get_llm_client(self):
        """Return the provider identifier for the primary LLM.

        [DEPRECATED] Use get_llm_config().client instead.
        """
        return self.get_llm_config().client


class RLMEnvConfig(EnvConfig):
    """Configuration for RLM (Recursive Language Model) interaction mode.

    Extends EnvConfig with backward compatibility for legacy RLM configurations
    that use individual fields instead of mode_config.
    """

    # Override to make llm optional for backward compatibility
    llm: Optional[InferenceConfig] = Field(
        default=None,
        description="Primary Pydantic AI routing configuration. "
                   "Can be constructed from legacy main_* fields if not provided."
    )

    # Override default interaction mode for RLM
    interaction_mode: InteractionMode = Field(
        default=InteractionMode.RLM,
        description="RLM interaction mode with <python> → sandbox → <repl> loop"
    )

    # Legacy fields for backward compatibility - will construct mode_config from these
    main_model: Optional[str] = Field(
        default=None,
        description="[LEGACY] Model name. New code should use llm field."
    )
    main_client: Optional[LLMClientType] = Field(
        default=None,
        description="[LEGACY] Client type. New code should use llm field."
    )
    main_temperature: Optional[float] = Field(
        default=None,
        description="[LEGACY] Temperature. New code should use llm field."
    )
    main_max_tokens: Optional[int] = Field(
        default=None,
        description="[LEGACY] Max tokens. New code should use llm field."
    )
    repl: Optional[REPLSettings] = Field(
        default=None,
        description="[LEGACY] REPL settings. New code should use mode_config.repl."
    )
    sub_llm: Optional[SubLLMSettings] = Field(
        default=None,
        description="[LEGACY] Sub-LLM settings. New code should use mode_config.sub_llm."
    )
    enable_correction_feedback: Optional[bool] = Field(
        default=None,
        description="[LEGACY] Enable correction. New code should use mode_config.enable_correction_feedback."
    )

    def get_llm_config(self) -> InferenceConfig:
        """Get the primary LLM configuration.

        Returns the llm field if set, otherwise constructs from legacy fields.

        """
        if self.llm is not None:
            return self.llm

        if self.main_model is None:
            raise ValueError(
                "No LLM configuration found. Please set either 'llm' field or legacy 'main_model' field."
            )

        return InferenceConfig(
            client=self.main_client or LLMClientType.LITELLM,
            model=self.main_model,
            temperature=self.main_temperature if self.main_temperature is not None else 0.7,
            max_tokens=self.main_max_tokens or 8192,
        )

    def get_mode_config(self) -> RLMModeSettings:
        """Get RLM mode configuration.

        Returns mode_config if set, otherwise constructs from legacy fields.
        """
        if self.mode_config is not None:
            return self.mode_config

        # Construct from legacy fields for backward compatibility
        return RLMModeSettings(
            repl=self.repl or REPLSettings(),
            sub_llm=self.sub_llm or SubLLMSettings(),
            enable_correction_feedback=self.enable_correction_feedback if self.enable_correction_feedback is not None else False,
        )

    def get_sub_llm_client(self):
        """Return the provider-neutral sub-LLM client type."""
        return self.get_mode_config().sub_llm.client


__all__ = [
    "EnvConfig",
    "RLMEnvConfig",
    "DatasetSettings",
    "InteractionMode",
    "LLMClientType",
    "InferenceConfig",
    "LLMJudgeSettings",
    "ModeConfig",
    "RLMModeSettings",
    "ToolUseModeSettings",
    "ChatModeSettings",
    "REPLSettings",
    "RolloutTerminationPolicy",
    "SubLLMSettings",
]
