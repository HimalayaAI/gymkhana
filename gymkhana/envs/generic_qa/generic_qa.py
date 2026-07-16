from typing import Optional, List, Any
import logging
from gymkhana.envs.environment import Environment
from gymkhana.core.models import Task, TrajectoryResult, Turn
from gymkhana.envs.tool_bridge import EnvironmentToolkit

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Example Tools
# ------------------------------------------------------------------

async def add(a: float, b: float) -> float:
    """Add two numbers.

    Args:
        a: First number
        b: Second number
    """
    return a + b

async def multiply(a: float, b: float) -> float:
    """Multiply two numbers.

    Args:
        a: First number
        b: Second number
    """
    return a * b

# ------------------------------------------------------------------
# GenericQA Environment
# ------------------------------------------------------------------

class GenericQAEnv(Environment):
    """
    Example environment demonstrating all interaction modes:
    1. RLM (Repl)
    2. TOOL_CALL (Regular)
    3. TOOL_CALL_INTERLEAVED (Interleaved)
    4. PLAIN_TEXT (CoT / Refinement)
    """

    def __init__(self, config=None, **kwargs):
        super().__init__(config=config, **kwargs)
        self.name = "generic_qa"

        # Register tools
        self._toolkit = EnvironmentToolkit(tools=[add, multiply])

    def get_tool_executor(self, task: Task) -> Optional[EnvironmentToolkit]:
        """Return the toolkit for TOOL_CALL / TOOL_CALL_INTERLEAVED modes."""
        return self._toolkit

    def get_refinement_prompt(self, task: Task, response: str, turn_idx: int) -> Optional[str]:
        """Return a refinement prompt for PLAIN_TEXT multi-turn mode."""
        return (
            f"Review your previous answer:\n{response}\n\n"
            "Please refine your reasoning and provide a more accurate final answer."
        )

    def evaluate_answer(self, task: Task, result: TrajectoryResult) -> bool:
        """Simple string match evaluation."""
        if not result.final_answer:
            return False

        expected = self.get_expected_answer(task)
        if not expected:
            # If no expected answer, return True (or use LLM judge reward)
            return True

        # Basic normalization and inclusion check
        pred = result.final_answer.strip().lower()
        gold = expected.strip().lower()
        return gold in pred

    async def compute_reward(self, result: TrajectoryResult, answer_correct: bool = None, task: Task = None) -> float:
        """Compute reward based on correctness."""
        # If answer_correct is provided, use it
        if answer_correct is not None:
             result.total_reward = 1.0 if answer_correct else 0.0
             return result.total_reward

        # Otherwise compute it
        is_correct = self.evaluate_answer(task, result)
        result.total_reward = 1.0 if is_correct else 0.0
        return result.total_reward

    def load_tasks(self, split: str = "train") -> List[Task]:
        """Load minimal example tasks."""
        return [
            Task(
                id="qa_001",
                question="What is 15 + 27?",
                expected_answer="42",
                metadata={"difficulty": "easy"}
            ),
            Task(
                id="qa_002",
                question="Calculate 12 * 7 then add 5.",
                expected_answer="89",
                metadata={"difficulty": "medium"}
            )
        ]
