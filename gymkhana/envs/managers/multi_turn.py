"""Multi-turn conversation manager."""
from typing import List, Optional, Tuple, TYPE_CHECKING

from gymkhana.envs.managers.base import ConversationManager

if TYPE_CHECKING:
    from gymkhana.core.models import Turn
    from gymkhana.envs.environment import Environment, Task
    from gymkhana.envs.modes.base import InteractionMode


class MultiTurnManager(ConversationManager):
    """Multi-turn conversation: loop until max_turns or final_answer.

    This manager executes multiple turns in a loop:
    1. Send initial message
    2. Get response
    3. Check for final answer → if found, stop
    4. Generate next prompt → if None, stop
    5. Repeat from step 2

    Stops when:
    - Final answer is extracted
    - get_next_prompt() returns None
    - max_turns is reached

    Useful for:
    - Reasoning tasks requiring multiple steps
    - Tasks with iterative refinement
    - Tasks requiring chain-of-thought

    Example:
        ```python
        manager = MultiTurnManager(max_turns=5)
        mode = RLMMode()
        result = await mode.execute_single(task, env, manager)
        ```
    """

    async def manage_conversation(
        self,
        initial_message: str,
        system_prompt: str,
        env: "Environment",
        mode: "InteractionMode",
        task: "Task",
    ) -> Tuple[List["Turn"], Optional[str]]:
        """Execute multi-turn conversation.

        Args:
            initial_message: Initial user message
            system_prompt: System prompt for LLM
            env: Environment providing LLM
            mode: InteractionMode (not directly used, but available)
            task: The task being executed

        Returns:
            Tuple of (turns, final_answer):
                - turns: List of all conversation turns
                - final_answer: Extracted answer, or None if not found
        """
        from gymkhana.core.models import Turn

        # Initialize conversation
        turns = [Turn(role="user", content=initial_message, turn_index=0)]
        messages = [{"role": "user", "content": initial_message}]
        final_answer = None

        # Multi-turn loop
        for turn_idx in range(self.max_turns):
            # Generate response
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
                print(f"{BOLD}Assistant Response - Turn {turn_idx + 1}{RESET}")
                print("═" * 100)

                if reasoning_content:
                    print(f"{CYAN}[REASONING]{RESET}")
                    print(reasoning_content)
                    print()

                print(f"{GREEN}[RESPONSE]{RESET}")
                print(response)
                print()

            # Record assistant turn
            turns.append(Turn(
                role="assistant",
                content=response,
                reasoning_content=reasoning_content,
                turn_index=len(turns)
            ))
            messages.append({"role": "assistant", "content": response})

            # Check for final answer
            final_answer = self._extract_final_answer(response)
            if final_answer:
                break

            # Get next prompt
            next_prompt = self.get_next_prompt(response, turn_idx, task, env)
            if next_prompt is None:
                break

            # Record user turn
            turns.append(Turn(
                role="user",
                content=next_prompt,
                turn_index=len(turns)
            ))
            messages.append({"role": "user", "content": next_prompt})

        return turns, final_answer

    def get_next_prompt(
        self,
        response: str,
        turn_idx: int,
        task: "Task",
        env: "Environment"
    ) -> Optional[str]:
        """Generate next prompt (default: no follow-up).

        Default implementation returns None (stop after each response).
        Subclasses should override to implement specific patterns:
        - SequentialToolManager: "Continue with next step"
        - ConversationalManager: Wait for user input
        - SelfRefinementManager: "Refine your answer"

        Args:
            response: Assistant's last response
            turn_idx: Current turn index
            task: The task being executed
            env: Environment providing context

        Returns:
            Next user prompt, or None to stop
        """
        return None
