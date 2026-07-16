"""Single-turn tool-use (function calling) environment implementation.

Loads tasks from ``Salesforce/xlam-function-calling-60k`` and evaluates
model responses against ground-truth tool calls.

Uses native **tool calling** (OpenAI function calling format) where the model
generates tool_calls via the API, and we validate them against ground truth.

For models that use custom XML-based tool calling (e.g. Hermes/NousResearch),
a separate ``HermesToolCallParser`` is provided as an example of how to add
custom parsers for non-native tool-call formats.

Dataset Structure (xlam-function-calling):
    - query: User question requiring tool usage
    - answers: JSON string with ground-truth tool calls
              e.g. [{"name": "web_chain_details", "arguments": {"chain_slug": "ethereum"}}]
    - tools: JSON string with available tool definitions
              e.g. [{"name": "peers", "description": "...", "parameters": {...}}]
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
    ToolUseModeSettings,
)
from ..environment import Environment, EnvironmentError, Task, register_environment
from ..modes import ToolUseMode
from ..managers import SingleTurnManager
from ..parsers import HermesToolCallParser
from gymkhana.core.models import AnswerVerifier, TrajectoryResult
from gymkhana.core.rewards import RewardFunction, register_reward_function, TrajectoryMetrics

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tool schema conversion: xlam dataset → OpenAI function-calling format
# ---------------------------------------------------------------------------

def _xlam_type_to_json_schema_type(xlam_type: str) -> str:
    """Map xlam parameter types to JSON Schema types."""
    mapping = {
        "str": "string",
        "string": "string",
        "int": "integer",
        "integer": "integer",
        "float": "number",
        "number": "number",
        "bool": "boolean",
        "boolean": "boolean",
        "list": "array",
        "array": "array",
        "dict": "object",
        "object": "object",
    }
    return mapping.get(xlam_type.lower(), "string")


def convert_xlam_tools_to_openai(tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert xlam-format tool definitions to OpenAI function-calling format.

    xlam format:
        {"name": "fn", "description": "...", "parameters": {"param1": {"description": "...", "type": "str", "default": ""}}}

    OpenAI format:
        {"type": "function", "function": {"name": "fn", "description": "...",
         "parameters": {"type": "object", "properties": {...}, "required": [...]}}}
    """
    openai_tools = []
    for tool in tools:
        name = tool.get("name", "unknown")
        description = tool.get("description", f"Execute {name}")
        raw_params = tool.get("parameters", {})

        # Build JSON Schema properties
        properties = {}
        required = []
        for param_name, param_info in raw_params.items():
            if isinstance(param_info, dict):
                prop = {
                    "type": _xlam_type_to_json_schema_type(param_info.get("type", "string")),
                    "description": param_info.get("description", ""),
                }
                properties[param_name] = prop
                # If no default value, consider required
                if "default" not in param_info or param_info["default"] == "":
                    required.append(param_name)
            else:
                properties[param_name] = {"type": "string", "description": str(param_info)}

        openai_tools.append({
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        })
    return openai_tools


# ---------------------------------------------------------------------------
# Tool call extraction: native API response
# ---------------------------------------------------------------------------

def extract_tool_calls_from_native_response(response: str) -> List[Dict[str, Any]]:
    """Extract tool calls from a native API response.

    When the model uses native tool calling, the inference service serializes
    tool calls as a JSON array in the response content. Each element has:
        {"name": "fn", "arguments": {...}, "tool_call_id": "call_xxx"}

    We normalise to: [{"name": "fn", "arguments": {...}}]
    """
    try:
        parsed = json.loads(response)
        if isinstance(parsed, list):
            return [
                {"name": tc.get("name", ""), "arguments": tc.get("arguments", {})}
                for tc in parsed
                if isinstance(tc, dict) and "name" in tc
            ]
        if isinstance(parsed, dict) and "name" in parsed:
            return [{"name": parsed["name"], "arguments": parsed.get("arguments", {})}]
    except (json.JSONDecodeError, TypeError):
        pass
    return []


# ---------------------------------------------------------------------------
# Comparison / matching helpers
# ---------------------------------------------------------------------------

def normalise_arguments(args: Any) -> Dict[str, Any]:
    """Normalise tool call arguments to a dict."""
    if isinstance(args, str):
        try:
            return json.loads(args)
        except (json.JSONDecodeError, TypeError):
            return {"raw": args}
    if isinstance(args, dict):
        return args
    return {}


def tool_calls_match(
    predicted: List[Dict[str, Any]],
    expected: List[Dict[str, Any]],
) -> bool:
    """Order-independent check: predicted == expected (name + arguments).

    Each tool call must match on:
    - name (exact)
    - arguments (deep equality after normalisation)
    """
    if len(predicted) != len(expected):
        return False

    norm_pred = [
        {"name": tc.get("name", ""), "arguments": normalise_arguments(tc.get("arguments", {}))}
        for tc in predicted
    ]
    norm_exp = [
        {"name": tc.get("name", ""), "arguments": normalise_arguments(tc.get("arguments", {}))}
        for tc in expected
    ]

    used = [False] * len(norm_pred)
    for exp_tc in norm_exp:
        found = False
        for i, pred_tc in enumerate(norm_pred):
            if used[i]:
                continue
            if pred_tc["name"] == exp_tc["name"] and pred_tc["arguments"] == exp_tc["arguments"]:
                used[i] = True
                found = True
                break
        if not found:
            return False
    return True


# ---------------------------------------------------------------------------
# Tool Call Verifier
# ---------------------------------------------------------------------------

class ToolCallVerifier(AnswerVerifier):
    """Verifier for single-turn tool-call correctness.

    Compares predicted tool calls (from native API response) against
    ground-truth tool calls stored in task metadata.
    """

    def verify(
        self,
        *,
        expected: Optional[str],
        candidates: List[str],
        task_metadata: Optional[Dict[str, Any]] = None,
        trajectory: Optional[TrajectoryResult] = None,
    ) -> Optional[bool]:
        if not candidates or not task_metadata:
            return None

        expected_calls = task_metadata.get("expected_tool_calls", [])
        if not expected_calls:
            logger.warning("No expected_tool_calls in task metadata")
            return None

        response = candidates[0]
        # Try native format first (JSON-serialized tool calls)
        predicted_calls = extract_tool_calls_from_native_response(response)

        # Fallback: try Hermes XML format
        if not predicted_calls:
            predicted_calls = HermesToolCallParser.parse(response)

        if not predicted_calls:
            return False

        return tool_calls_match(predicted_calls, expected_calls)


# ---------------------------------------------------------------------------
# Tool Call Reward Function
# ---------------------------------------------------------------------------

@register_reward_function("tool_call_match")
class ToolCallRewardFunction(RewardFunction):
    """Reward: 1.0 if tool calls exactly match ground truth, 0.0 otherwise."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self, **data):
        super().__init__(**data)
        object.__setattr__(self, '_verifier', ToolCallVerifier())

    @property
    def verifier(self) -> ToolCallVerifier:
        return getattr(self, '_verifier', ToolCallVerifier())

    def compute(self, metrics: TrajectoryMetrics) -> Dict[str, Any]:
        passed = getattr(metrics, 'answer_correct', None)
        final_reward = 1.0 if passed else 0.0

        intermediate_rewards = getattr(metrics, 'intermediate_rewards', [])
        total = sum(intermediate_rewards) + final_reward

        return {
            "total_reward": total,
            "final_step_reward": final_reward,
            "metadata": {
                "reward_function": self.name,
                "tool_call_correct": bool(passed),
                "reason": "correct" if passed else ("incorrect" if passed is not None else "unknown"),
            },
        }


# ---------------------------------------------------------------------------
# System prompt builder
# ---------------------------------------------------------------------------

TOOL_USE_SYSTEM_PROMPT = (
    "You are a helpful assistant that can use tools to answer questions and perform tasks."
)


# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------

def _get_default_config() -> EnvConfig:
    """Create default config with env vars loaded."""
    client_str = os.getenv("LITELLM_CLIENT", "litellm").lower()
    client_map = {client.value: client for client in LLMClientType}

    return EnvConfig(
        name="tool_use_singleturn",
        llm=InferenceConfig(
            client=client_map.get(client_str, LLMClientType.LITELLM),
            model=os.getenv("LITELLM_MODEL", "gpt-4o"),
            temperature=float(os.getenv("LITELLM_TEMPERATURE", "0.7")),
            max_tokens=int(os.getenv("LITELLM_MAX_TOKENS", "4096")),
        ),
        interaction_mode=InteractionMode.TOOL_CALL,
        mode_config=ToolUseModeSettings(
            max_turns=1,  # Single-turn: one shot only
        ),
        dataset=DatasetSettings(
            environment="tool_use_singleturn",
            dataset_name="Salesforce/xlam-function-calling-60k",
            dataset_config=None,
            dataset_split="train",
            field_mapping={
                "id": None,
                "prompt": "query",
                "expected_answer": "answers",
                "context": None,
            },
            batch_size=4,
            num_rollouts=1,
            limit=int(os.getenv("TOOL_USE_LIMIT", "100")),
            include_instructions=True,
            output_dir="outputs/tool_use_singleturn",
            output_sharegpt=True,
            enable_rewards=True,
            reward_function="tool_call_match",
        ),
        debug=False,
    )


DEFAULT_CONFIG: Optional[EnvConfig] = None


# ---------------------------------------------------------------------------
# Environment class
# ---------------------------------------------------------------------------

@register_environment(name="tool-use-singleturn", env_type="tool_use")
class ToolUseSingleTurnEnv(Environment):
    """Single-turn tool-use (function calling) environment.

    Loads ``Salesforce/xlam-function-calling-60k`` and scores model
    outputs against ground-truth tool calls using exact matching.

    **Native tool calling**: Tool definitions from the dataset are converted
    to OpenAI function-calling format and passed via the ``tools`` parameter
    to the LLM API. The model generates native ``tool_calls`` in the API
    response, which are then extracted and validated against ground truth.

    For models that use Hermes-style XML tool calling (``<tool_call>`` tags),
    ``HermesToolCallParser`` is used as a fallback parser.
    """

    name: str = "tool_use_singleturn"
    default_config: ClassVar[Optional[EnvConfig]] = None

    def __init__(self, *, config: Optional[EnvConfig] = None, **data: Any) -> None:
        if ToolUseSingleTurnEnv.default_config is None:
            ToolUseSingleTurnEnv.default_config = _get_default_config()

        if config is None:
            config = ToolUseSingleTurnEnv.default_config.model_copy(deep=True)
        elif isinstance(config, dict):
            config = EnvConfig(**config)

        data["config"] = config
        super().__init__(**data)

        self._mode = ToolUseMode()
        self._manager = SingleTurnManager()

    # ------------------------------------------------------------------
    # Override run_task to use custom execute_task
    # ------------------------------------------------------------------
    async def run_task(self, task: Task) -> TrajectoryResult:
        """Execute task with custom tool-calling logic.

        Overrides base class to use execute_task() which handles
        tool extraction from task metadata and native tool calling.
        """
        G = getattr(self.config.dataset, "num_rollouts", None) or self.num_rollouts
        if G > 1:
            # Use common batch tracking with custom executor
            results = await self._execute_batch_with_tracking(
                task, num_rollouts=G,
                executor=lambda: self.execute_task(task)
            )
            return max(results, key=lambda r: (r.total_reward or 0.0, r.success))
        return await self.execute_task(task)

    # ------------------------------------------------------------------
    # Dataset handling
    # ------------------------------------------------------------------
    def _load_dataset(self) -> Iterable[Dict[str, Any]]:
        cfg = self.config.dataset
        if not cfg.dataset_name:
            raise EnvironmentError(
                "ToolUseSingleTurnEnv requires dataset.dataset_name to be set"
            )
        try:
            if cfg.dataset_config:
                ds = load_dataset(
                    cfg.dataset_name, cfg.dataset_config,
                    split=cfg.dataset_split, streaming=True,
                )
            else:
                ds = load_dataset(
                    cfg.dataset_name,
                    split=cfg.dataset_split, streaming=True,
                )

            if cfg.dataset_seed is not None:
                ds = ds.shuffle(seed=cfg.dataset_seed, buffer_size=1000)

            print(
                f"Using streaming mode for {cfg.dataset_name} "
                f"(split={cfg.dataset_split}, seed={cfg.dataset_seed})"
            )
            return ds
        except ImportError as exc:
            raise EnvironmentError(
                "datasets package is required for ToolUseSingleTurnEnv"
            ) from exc
        except Exception as exc:
            raise EnvironmentError(
                f"Failed to load dataset '{cfg.dataset_name}': {exc}"
            ) from exc

    def _parse_tools(self, raw_tools: Any) -> List[Dict[str, Any]]:
        """Parse tool definitions from dataset field."""
        if isinstance(raw_tools, str):
            try:
                return json.loads(raw_tools)
            except (json.JSONDecodeError, TypeError):
                logger.warning("Failed to parse tools JSON: %s", raw_tools[:100])
                return []
        if isinstance(raw_tools, list):
            return raw_tools
        return []

    def _parse_answers(self, raw_answers: Any) -> List[Dict[str, Any]]:
        """Parse expected tool calls from dataset 'answers' field."""
        if isinstance(raw_answers, str):
            try:
                parsed = json.loads(raw_answers)
                if isinstance(parsed, list):
                    return parsed
                if isinstance(parsed, dict):
                    return [parsed]
            except (json.JSONDecodeError, TypeError):
                logger.warning("Failed to parse answers JSON: %s", raw_answers[:100])
                return []
        if isinstance(raw_answers, list):
            return raw_answers
        return []

    def load_tasks(self, limit: Optional[int] = None) -> Sequence[Task]:
        dataset_limit = limit or self.config.dataset.limit
        records = self._load_dataset()

        tasks: List[Task] = []
        seen = 0

        for record in records:
            if dataset_limit is not None and seen >= dataset_limit:
                break

            query = record.get("query", "")
            if not query:
                continue

            # Parse tools and expected answers
            tools_raw = self._parse_tools(record.get("tools", "[]"))
            expected_calls = self._parse_answers(record.get("answers", "[]"))

            if not tools_raw or not expected_calls:
                continue

            # Convert tools to OpenAI format for native tool calling
            openai_tools = convert_xlam_tools_to_openai(tools_raw)

            metadata = {
                "tools_raw": tools_raw,           # Original xlam format
                "tools_openai": openai_tools,      # OpenAI function-calling format
                "expected_tool_calls": expected_calls,
                "dataset": "xlam-function-calling",
            }

            tasks.append(
                Task(
                    id=str(seen),
                    prompt=query,
                    metadata=metadata,
                )
            )
            seen += 1

        return tasks

    # ------------------------------------------------------------------
    # Behaviour hooks
    # ------------------------------------------------------------------
    def build_system_prompt(self, task: Task) -> str:
        """Build system prompt.

        Note: Tool definitions are NOT embedded in the system prompt when
        using native tool calling — they are passed via the ``tools`` API
        parameter. The system prompt only provides general instructions.
        """
        parts = [TOOL_USE_SYSTEM_PROMPT]

        env_instructions = self.get_environment_instructions(task)
        if env_instructions:
            parts.append(env_instructions.strip())

        return "\n\n".join(parts)

    def format_initial_message(self, task: Task) -> str:
        """The query is the user message."""
        return task.prompt

    def build_sharegpt_conversations(
        self,
        result: 'TrajectoryResult',
        task: 'Task',
    ) -> Optional[List[Dict[str, Any]]]:
        """Build Hermes-format ShareGPT conversation for tool-calling.

        Overrides the base class to use Hermes tool-calling format:
        - System prompt with tools in <tools> XML tags
        - Tool calls wrapped in <tool_call> tags
        - Reasoning wrapped in <think> tags
        """
        from gymkhana.core.utils.hermes_format import trajectory_to_hermes_sharegpt

        return trajectory_to_hermes_sharegpt(
            result=result,
            task=task,
            source_format="openai",
        )

    async def execute_task(self, task: Task) -> TrajectoryResult:
        """Execute a single tool-use task with native tool calling.

        Flow:
        1. Build system prompt (no tool definitions — tools go via API)
        2. Pass OpenAI-format tool definitions as ``tools`` parameter
        3. Model generates native tool_calls in the API response
        4. Extract tool calls from the serialized JSON response
        5. Compare against ground truth and compute reward

        Args:
            task: The task to execute

        Returns:
            TrajectoryResult with tool-call validation
        """
        from gymkhana.core.models import TrajectoryResult, Turn

        system_prompt = self.build_system_prompt(task)
        initial_message = self.format_initial_message(task)

        # Get OpenAI-format tools for native tool calling
        openai_tools = task.metadata.get("tools_openai", [])

        messages = [{"role": "user", "content": initial_message}]

        # Generate response with native tool calling
        response, reasoning_content = await self.generate_response(
            messages=messages,
            system_prompt=system_prompt,
            tools=openai_tools,   # Native OpenAI tool calling
        )

        # Build turns
        turns = [
            Turn(role="user", content=initial_message, turn_index=0),
            Turn(
                role="assistant",
                content=response,
                reasoning_content=reasoning_content,
                turn_index=1,
            ),
        ]

        # Extract tool calls from native response (JSON-serialized by inference service)
        predicted_calls = extract_tool_calls_from_native_response(response)

        # Fallback: try Hermes XML format for non-native models
        if not predicted_calls:
            predicted_calls = HermesToolCallParser.parse(response)

        expected_calls = task.metadata.get("expected_tool_calls", [])

        # Verify correctness
        passed = tool_calls_match(predicted_calls, expected_calls) if predicted_calls else False

        # Debug output
        if self.config.debug:
            CYAN = '\033[96m'
            GREEN = '\033[92m'
            YELLOW = '\033[93m'
            BOLD = '\033[1m'
            RESET = '\033[0m'

            print("═" * 100)
            print(f"{BOLD}Tool-Use Single Turn - Task {task.id}{RESET}")
            print("═" * 100)
            print(f"{CYAN}[TOOLS PASSED]{RESET} {len(openai_tools)} tools via native API")
            if reasoning_content:
                print(f"{CYAN}[REASONING]{RESET}")
                print(reasoning_content[:500])
                print()
            print(f"{GREEN}[RESPONSE]{RESET}")
            print(response[:500])
            print()
            print("─" * 100)
            print(f"{BOLD}Verification{RESET}")
            print(f"Expected:  {json.dumps(expected_calls, indent=2)}")
            print(f"Predicted: {json.dumps(predicted_calls, indent=2)}")
            print(f"Match: {'✅ CORRECT' if passed else '❌ INCORRECT'}")
            print()

        # Build result
        result = TrajectoryResult(
            success=bool(predicted_calls),
            final_answer=response,
            turns=turns,
            num_code_blocks=len(predicted_calls),
            num_errors=0,
            answer_correct=passed,
            task_id=task.id,
            environment=self.name,
            system_prompt=system_prompt,
            model_name=getattr(self.config.get_llm_config(), 'model', None),
        )

        # Compute reward
        if self.config.dataset.enable_rewards:
            reward = await self.compute_reward(result, answer_correct=passed, task=task)
            result.total_reward = reward if isinstance(reward, (int, float)) else 0.0

        return result

    # ------------------------------------------------------------------
    # Reward / scoring
    # ------------------------------------------------------------------
    async def compute_reward(
        self,
        result: TrajectoryResult,
        answer_correct: Optional[bool] = None,
        task: Optional[Task] = None,
    ) -> float:
        """Score 1.0 if tool calls match ground truth, else 0.0."""
        if not result.final_answer or task is None:
            return 0.0

        if answer_correct is not None:
            return 1.0 if answer_correct else 0.0

        # Verify from scratch
        predicted_calls = extract_tool_calls_from_native_response(result.final_answer)
        if not predicted_calls:
            predicted_calls = HermesToolCallParser.parse(result.final_answer)
        expected_calls = task.metadata.get("expected_tool_calls", [])
        return 1.0 if tool_calls_match(predicted_calls, expected_calls) else 0.0
