"""Tool use mode for function calling and tool execution."""
import json
from typing import List, Optional, TYPE_CHECKING

from gymkhana.envs.modes.base import InteractionMode

if TYPE_CHECKING:
    from gymkhana.core.models import TrajectoryResult, Turn
    from gymkhana.envs.environment import Environment, Task


class ToolUseMode(InteractionMode):
    """Regular tool call mode: Think -> Tool Call -> Execute.

    This mode supports function calling where the model:
    1. Generates a response with tool calls
    2. Tools are executed
    3. Results are fed back to the model
    4. Process repeats until final answer

    Example:
        ```python
        mode = ToolUseMode()
        result = await mode.execute_single(task, env)
        ```
    """

    async def execute_single(
        self,
        task: "Task",
        env: "Environment",
    ) -> "TrajectoryResult":
        """Execute a single tool-use rollout.

        Args:
            task: The task to execute
            env: Environment providing LLM and tools

        Returns:
            TrajectoryResult with conversation turns and final answer
        """
        from gymkhana.core.models import TrajectoryResult, Turn
        import logging

        logger = logging.getLogger(__name__)

        # Get mode settings
        from gymkhana.envs.config import ToolUseModeSettings
        settings = env.config.get_mode_config()
        if not isinstance(settings, ToolUseModeSettings):
            settings = ToolUseModeSettings()

        # Get toolkit and prompts
        toolkit = env.get_tool_executor(task)
        system_prompt = env.build_system_prompt(task)
        initial_message = env.format_initial_message(task)

        # Initialize conversation
        turns: List[Turn] = [Turn(role="user", content=initial_message, turn_index=0)]
        messages = [{"role": "user", "content": initial_message}]

        final_answer_text = None
        num_code_blocks = 0  # Track tool calls as "code blocks" for stats

        # Pydantic AI owns the model -> tool -> model loop. Its native Tool objects
        # preserve schemas and tool-call protocol across OpenAI and Anthropic.
        trace_token = toolkit.start_trace() if toolkit else None
        try:
            response, reasoning_content = await env.generate_response(
                messages=messages,
                system_prompt=system_prompt,
                tools=toolkit.pydantic_tools if toolkit else None,
            )
        finally:
            tool_calls = toolkit.finish_trace(trace_token) if toolkit and trace_token else []

        for index, call in enumerate(tool_calls):
            call_id = f"tool-{index}"
            turns.append(Turn(
                role="assistant",
                content="",
                tool_calls=[{"id": call_id, "name": call["name"], "arguments": call["arguments"]}],
                turn_index=len(turns),
            ))
            result = call["result"]
            content = json.dumps(result, ensure_ascii=False) if isinstance(result, (dict, list)) else str(result)
            turns.append(Turn(role="tool", content=content, tool_call_id=call_id, turn_index=len(turns)))
        num_code_blocks = len(tool_calls)
        turns.append(Turn(role="assistant", content=response, turn_index=len(turns)))
        final_answer_text = response

        # Build result
        return TrajectoryResult(
            success=final_answer_text is not None,
            final_answer=final_answer_text or "",
            turns=turns,
            num_code_blocks=num_code_blocks,
            num_errors=0,
            task_id=task.id,
            environment=env.name,
            system_prompt=system_prompt,
            model_name=getattr(env.config.get_llm_config(), 'model', None),
        )

    async def execute_batch(
        self,
        task: "Task",
        env: "Environment",
        num_rollouts: int,
    ) -> List["TrajectoryResult"]:
        """Execute G parallel tool-use rollouts.

        Simple parallel execution - rollout tracking is handled by Environment.

        Args:
            task: The task to execute
            env: Environment providing LLM and tools
            num_rollouts: Number of parallel rollouts (G)

        Returns:
            List of G TrajectoryResults
        """
        import asyncio

        # Execute G rollouts in parallel
        tasks = [
            self.execute_single(task, env)
            for _ in range(num_rollouts)
        ]

        results = await asyncio.gather(*tasks)
        return list(results)
