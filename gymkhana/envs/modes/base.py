"""Base class for interaction modes."""
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from gymkhana.core.models import TrajectoryResult
    from gymkhana.envs.managers.base import ConversationManager
    from gymkhana.envs.environment import Environment, Task


class InteractionMode(ABC):
    """Base class for interaction mode execution.

    Each mode implements a specific interaction pattern:
    - RLMMode: Code execution with REPL sandbox
    - ToolUseMode: Structured tool calls (complete → parse → execute)
    - InterleavedToolUseMode: Tool calls with interleaved reasoning (stop-execute-continue)
    - ChatMode: Pure text generation without tools

    Modes are stateless and reusable across tasks. They define HOW the agent
    interacts with the environment (code execution, tool calls, or text).

    The conversation pattern (HOW MANY turns, WHAT pattern) is handled by
    ConversationManager, which is passed to execute_single().
    """

    @abstractmethod
    async def execute_single(
        self,
        task: "Task",
        env: "Environment",
        conversation_manager: "ConversationManager",
    ) -> "TrajectoryResult":
        """Execute a single rollout for the task.

        Args:
            task: The task to execute
            env: The environment providing LLM and tools
            conversation_manager: Manages conversation turns and patterns

        Returns:
            TrajectoryResult containing turns, final answer, and metrics
        """
        pass

    @abstractmethod
    async def execute_batch(
        self,
        task: "Task",
        env: "Environment",
        conversation_manager: "ConversationManager",
        num_rollouts: int,
    ) -> List["TrajectoryResult"]:
        """Execute G rollouts in parallel (no tracking).

        Simple parallel execution - rollout tracking is handled by Environment.
        Subclasses just execute G parallel rollouts and return the results.

        Args:
            task: The task to execute
            env: The environment providing LLM and tools
            conversation_manager: Manages conversation turns and patterns
            num_rollouts: Number of parallel rollouts (G)

        Returns:
            List of G TrajectoryResults
        """
        pass
