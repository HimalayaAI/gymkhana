"""Single-turn conversation manager."""
from typing import List, Optional, Tuple, TYPE_CHECKING

from gymkhana.envs.managers.base import ConversationManager

if TYPE_CHECKING:
    from gymkhana.core.models import Turn
    from gymkhana.envs.environment import Environment, Task
    from gymkhana.envs.modes.base import InteractionMode


class SingleTurnManager(ConversationManager):
    """Single-turn conversation: one response, done.

    This manager executes exactly one turn:
    1. Send initial message
    2. Get response
    3. Extract final answer (if any)
    4. Stop

    Useful for:
    - Simple Q&A tasks
    - Classification tasks
    - Tasks that don't require reasoning chains

    Example:
        ```python
        manager = SingleTurnManager()
        mode = ChatMode()
        result = await mode.execute_single(task, env, manager)
        ```
    """

    def __init__(self):
        """Initialize single-turn manager with max_turns=1."""
        super().__init__(max_turns=1)

    async def manage_conversation(
        self,
        initial_message: str,
        system_prompt: str,
        env: "Environment",
        mode: "InteractionMode",
        task: "Task",
    ) -> Tuple[List["Turn"], Optional[str]]:
        """Execute single-turn conversation.

        Args:
            initial_message: Initial user message
            system_prompt: System prompt for LLM
            env: Environment providing LLM
            mode: InteractionMode (not used in single turn)
            task: The task being executed

        Returns:
            Tuple of (turns, final_answer):
                - turns: List with 2 turns (user, assistant)
                - final_answer: Extracted answer or full response
        """
        from gymkhana.core.models import Turn

        # Create initial user turn
        turns = [Turn(role="user", content=initial_message, turn_index=0)]

        # Generate single response
        messages = [{"role": "user", "content": initial_message}]
        generated = await env.generate_response(
            messages=messages,
            system_prompt=system_prompt
        )
        response, reasoning_content = self._normalize_response(generated)

        # Debug: Print assistant response with formatting (only in debug mode)
        if getattr(getattr(env, "config", None), "debug", False):
            # ANSI color codes
            CYAN = '\033[96m'
            GREEN = '\033[92m'
            BOLD = '\033[1m'
            RESET = '\033[0m'

            print("═" * 100)
            print(f"{BOLD}Assistant Response{RESET}")
            print("═" * 100)

            if reasoning_content:
                print(f"{CYAN}[REASONING]{RESET}")
                print(reasoning_content)
                print()

            print(f"{GREEN}[RESPONSE]{RESET}")
            print(response)
            print()

        # Create assistant turn
        turns.append(Turn(
            role="assistant",
            content=response,
            reasoning_content=reasoning_content,
            turn_index=1
        ))

        # Extract final answer (or use full response)
        final_answer = self._extract_final_answer(response)
        if final_answer is None:
            final_answer = response

        return turns, final_answer

    def get_next_prompt(
        self,
        response: str,
        turn_idx: int,
        task: "Task",
        env: "Environment"
    ) -> Optional[str]:
        """Always return None - single turn only.

        Args:
            response: Assistant's response (ignored)
            turn_idx: Turn index (ignored)
            task: Task (ignored)
            env: Environment (ignored)

        Returns:
            None (always stop after one turn)
        """
        return None
