"""Tests for ConversationManager base class."""
import pytest
from typing import List, Optional, Tuple

from gymkhana.envs.managers.base import ConversationManager
from gymkhana.core.models import Turn


class MockConversationManager(ConversationManager):
    """Mock implementation for testing base class."""

    def __init__(self, max_turns: int = 10, next_prompts: Optional[List[str]] = None):
        super().__init__(max_turns=max_turns)
        self.next_prompts = next_prompts or []
        self.next_prompt_idx = 0
        self.manage_conversation_called = False

    async def manage_conversation(
        self, initial_message, system_prompt, env, mode, task
    ) -> Tuple[List[Turn], Optional[str]]:
        self.manage_conversation_called = True
        turns = [Turn(role="user", content=initial_message, turn_index=0)]
        return turns, None

    def get_next_prompt(self, response, turn_idx, task, env) -> Optional[str]:
        if self.next_prompt_idx < len(self.next_prompts):
            prompt = self.next_prompts[self.next_prompt_idx]
            self.next_prompt_idx += 1
            return prompt
        return None


class TestConversationManagerBase:
    """Test ConversationManager base class."""

    def test_initialization(self):
        """Test manager initialization with default and custom max_turns."""
        # Default max_turns
        manager = MockConversationManager()
        assert manager.max_turns == 10

        # Custom max_turns
        manager = MockConversationManager(max_turns=5)
        assert manager.max_turns == 5

    def test_extract_final_answer_xml_tags(self):
        """Test final answer extraction with XML-style tags."""
        manager = MockConversationManager()

        # Standard case
        response = "Let me think... <answer>42</answer> That's it!"
        assert manager._extract_final_answer(response) == "42"

        # Case insensitive
        response = "Result: <ANSWER>Hello World</ANSWER>"
        assert manager._extract_final_answer(response) == "Hello World"

        # Multiline
        response = """
        <answer>
        The answer is
        multiple lines
        </answer>
        """
        result = manager._extract_final_answer(response)
        assert "multiple lines" in result

    def test_extract_final_answer_final_answer_pattern(self):
        """Test final answer extraction with 'Final Answer:' pattern."""
        manager = MockConversationManager()

        # Standard case
        response = "After calculation, Final Answer: 42"
        assert manager._extract_final_answer(response) == "42"

        # Case insensitive
        response = "final answer: Hello World"
        assert manager._extract_final_answer(response) == "Hello World"

        # With newline
        response = "Final Answer: 42\nSome other text"
        assert manager._extract_final_answer(response) == "42"

    def test_extract_final_answer_the_answer_is_pattern(self):
        """Test final answer extraction with 'The answer is:' pattern."""
        manager = MockConversationManager()

        # Standard case
        response = "After thinking, the answer is: 42"
        assert manager._extract_final_answer(response) == "42"

        # Case insensitive
        response = "The Answer Is: Hello World"
        assert manager._extract_final_answer(response) == "Hello World"

    def test_extract_final_answer_no_match(self):
        """Test final answer extraction when no pattern matches."""
        manager = MockConversationManager()

        response = "Just some text without any answer pattern"
        assert manager._extract_final_answer(response) is None

    def test_extract_final_answer_priority(self):
        """Test that XML tags have priority over other patterns."""
        manager = MockConversationManager()

        # XML should win
        response = "Final Answer: wrong <answer>correct</answer>"
        assert manager._extract_final_answer(response) == "correct"

    def test_is_complete_with_answer(self):
        """Test _is_complete returns True when answer is found."""
        manager = MockConversationManager()

        response = "<answer>42</answer>"
        assert manager._is_complete(response) is True

    def test_is_complete_without_answer(self):
        """Test _is_complete returns False when no answer is found."""
        manager = MockConversationManager()

        response = "Just thinking out loud..."
        assert manager._is_complete(response) is False

    def test_normalize_response_supports_both_inference_contracts(self):
        manager = MockConversationManager()

        assert manager._normalize_response("plain") == ("plain", None)
        assert manager._normalize_response(("answer", "reasoning")) == (
            "answer", "reasoning"
        )
        with pytest.raises(TypeError, match="generate_response"):
            manager._normalize_response(("answer", "reasoning", "extra"))

    def test_get_next_prompt_returns_none_by_default(self):
        """Test get_next_prompt returns None when no prompts configured."""
        manager = MockConversationManager()

        result = manager.get_next_prompt("response", 0, None, None)
        assert result is None

    def test_get_next_prompt_returns_configured_prompts(self):
        """Test get_next_prompt returns configured prompts in order."""
        prompts = ["prompt1", "prompt2", "prompt3"]
        manager = MockConversationManager(next_prompts=prompts)

        assert manager.get_next_prompt("", 0, None, None) == "prompt1"
        assert manager.get_next_prompt("", 1, None, None) == "prompt2"
        assert manager.get_next_prompt("", 2, None, None) == "prompt3"
        assert manager.get_next_prompt("", 3, None, None) is None

    @pytest.mark.asyncio
    async def test_manage_conversation_abstract_method(self):
        """Test that manage_conversation is called correctly."""
        manager = MockConversationManager()

        turns, answer = await manager.manage_conversation(
            "test message", "system", None, None, None
        )

        assert manager.manage_conversation_called is True
        assert len(turns) == 1
        assert turns[0].role == "user"
        assert turns[0].content == "test message"
