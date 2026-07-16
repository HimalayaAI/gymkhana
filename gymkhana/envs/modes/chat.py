"""Chat mode for pure text generation without tools."""
from typing import List, TYPE_CHECKING

from gymkhana.envs.modes.base import InteractionMode

if TYPE_CHECKING:
    from gymkhana.core.models import TrajectoryResult
    from gymkhana.envs.managers.base import ConversationManager
    from gymkhana.envs.environment import Environment, Task


class ChatMode(InteractionMode):
    """Pure text generation mode without code execution or tool calls.

    This mode is for tasks that only require natural language responses:
    - Question answering
    - Text classification
    - Summarization
    - General conversation

    The mode delegates conversation flow to the ConversationManager,
    which determines how many turns and what pattern to follow.

    Example:
        ```python
        mode = ChatMode()
        manager = SingleTurnManager()
        result = await mode.execute_single(task, env, manager)
        ```
    """

    async def execute_single(
        self,
        task: "Task",
        env: "Environment",
        conversation_manager: "ConversationManager",
    ) -> "TrajectoryResult":
        """Execute a single chat-based rollout.

        Delegates to the conversation manager to handle the conversation flow.
        The manager calls env.generate_response() for each turn and manages
        the conversation pattern.

        Args:
            task: The task to execute
            env: Environment providing LLM
            conversation_manager: Manages conversation turns and patterns

        Returns:
            TrajectoryResult with conversation turns and final answer
        """
        from gymkhana.core.models import TrajectoryResult

        # Get initial message and system prompt from environment
        initial_message = env.format_initial_message(task)
        system_prompt = env.build_system_prompt(task)

        # Delegate conversation flow to manager
        turns, final_answer = await conversation_manager.manage_conversation(
            initial_message=initial_message,
            system_prompt=system_prompt,
            env=env,
            mode=self,
            task=task
        )

        # Build trajectory result
        result = TrajectoryResult(
            success=final_answer is not None,
            final_answer=final_answer or "",
            turns=turns,
            num_code_blocks=0,  # No code execution in chat mode
            num_errors=0,
            task_id=task.id,
            environment=env.name,
            system_prompt=system_prompt,
            model_name=getattr(getattr(env, 'config', None), 'main_model', None),
            interaction_mode="chat",
            conversation_manager=conversation_manager.__class__.__name__.replace("Manager", "").lower(),
            max_turns=conversation_manager.max_turns,
        )

        return result

    async def execute_batch(
        self,
        task: "Task",
        env: "Environment",
        conversation_manager: "ConversationManager",
        num_rollouts: int,
    ) -> List["TrajectoryResult"]:
        """Execute G parallel chat rollouts.

        Simple parallel execution - rollout tracking is handled by Environment.

        Args:
            task: The task to execute
            env: Environment providing LLM
            conversation_manager: Manages conversation turns and patterns
            num_rollouts: Number of parallel rollouts (G)

        Returns:
            List of G TrajectoryResults
        """
        import asyncio

        # Execute G rollouts in parallel
        tasks = [
            self.execute_single(task, env, conversation_manager)
            for _ in range(num_rollouts)
        ]

        results = await asyncio.gather(*tasks)
        return list(results)
