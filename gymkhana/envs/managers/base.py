"""Base class for conversation managers."""
from abc import ABC, abstractmethod
from typing import Any, TYPE_CHECKING, List, Optional, Tuple

if TYPE_CHECKING:
    from gymkhana.core.models import Turn
    from gymkhana.envs.environment import Environment, Task
    from gymkhana.envs.modes.base import InteractionMode


class ConversationManager(ABC):
    """Base class for conversation management.

    Conversation managers define the conversation pattern:
    - HOW MANY turns to execute
    - WHAT pattern to follow (single, multi-turn, sequential, conversational, self-refinement)
    - WHEN to stop (max turns, final answer, user interruption)

    They are orthogonal to InteractionMode (which defines HOW to execute each turn).
    Any manager can be used with any mode.

    Managers are stateless and reusable across tasks.
    """

    def __init__(self, max_turns: int = 10):
        """Initialize conversation manager.

        Args:
            max_turns: Maximum number of turns before stopping
        """
        self.max_turns = max_turns

    @abstractmethod
    async def manage_conversation(
        self,
        initial_message: str,
        system_prompt: str,
        env: "Environment",
        mode: "InteractionMode",
        task: "Task",
    ) -> Tuple[List["Turn"], Optional[str]]:
        """Manage the conversation flow.

        This is the main entry point called by InteractionMode.execute_single().
        The manager orchestrates the conversation pattern while delegating
        actual execution to the mode.

        Args:
            initial_message: Initial user message to start conversation
            system_prompt: System prompt for the LLM
            env: Environment providing LLM and tools
            mode: InteractionMode for executing each turn
            task: The task being executed

        Returns:
            Tuple of (turns, final_answer):
                - turns: List of Turn objects representing the conversation
                - final_answer: Extracted final answer, or None if not found
        """
        pass

    @abstractmethod
    def get_next_prompt(
        self,
        response: str,
        turn_idx: int,
        task: "Task",
        env: "Environment"
    ) -> Optional[str]:
        """Generate the next user prompt based on the assistant's response.

        This method defines the conversation pattern. Return None to stop.

        Args:
            response: The assistant's last response
            turn_idx: Current turn index (0-indexed)
            task: The task being executed
            env: Environment providing context

        Returns:
            Next user prompt, or None to stop the conversation
        """
        pass

    def _extract_final_answer(self, response: str) -> Optional[str]:
        """Extract final answer from response.

        Default implementation looks for common patterns like:
        - <answer>...</answer>
        - Final Answer: ...
        - The answer is: ...

        Subclasses can override for custom extraction logic.

        Args:
            response: The assistant's response

        Returns:
            Extracted final answer, or None if not found
        """
        import re

        # Try XML-style tags
        match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()

        # Try "Final Answer:" pattern
        match = re.search(r'Final Answer:\s*(.+?)(?:\n|$)', response, re.IGNORECASE)
        if match:
            return match.group(1).strip()

        # Try "The answer is:" pattern
        match = re.search(r'The answer is:\s*(.+?)(?:\n|$)', response, re.IGNORECASE)
        if match:
            return match.group(1).strip()

        return None

    @staticmethod
    def _normalize_response(response: Any) -> Tuple[str, Optional[str]]:
        """Normalize legacy text and current ``(text, reasoning)`` responses."""
        if isinstance(response, str):
            return response, None
        if isinstance(response, tuple) and len(response) == 2:
            text, reasoning = response
            if isinstance(text, str) and (reasoning is None or isinstance(reasoning, str)):
                return text, reasoning
        raise TypeError(
            "generate_response() must return text or a (text, reasoning) tuple"
        )

    def _is_complete(self, response: str) -> bool:
        """Check if the conversation is complete.

        Default implementation checks if a final answer was extracted.
        Subclasses can override for custom completion logic.

        Args:
            response: The assistant's response

        Returns:
            True if conversation should stop, False otherwise
        """
        return self._extract_final_answer(response) is not None
