"""Tests for SingleTurnManager."""
import pytest
import sys
from pathlib import Path

# Add tests directory to path for conftest imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from gymkhana.envs.managers.single_turn import SingleTurnManager
from conftest import MockEnvironment


class TestSingleTurnManager:
    """Test SingleTurnManager implementation."""

    def test_initialization(self):
        """Test manager initializes with max_turns=1."""
        manager = SingleTurnManager()
        assert manager.max_turns == 1

    @pytest.mark.asyncio
    async def test_manage_conversation_single_turn(self, mock_env, mock_task, mock_mode):
        """Test single-turn conversation flow."""
        manager = SingleTurnManager()

        turns, final_answer = await manager.manage_conversation(
            initial_message="What is 2+2?",
            system_prompt="You are a helpful assistant",
            env=mock_env,
            mode=mock_mode,
            task=mock_task
        )

        # Should have exactly 2 turns (user, assistant)
        assert len(turns) == 2
        assert turns[0].role == "user"
        assert turns[0].content == "What is 2+2?"
        assert turns[0].turn_index == 0

        assert turns[1].role == "assistant"
        assert turns[1].content == "Mock response"
        assert turns[1].turn_index == 1

        # Should extract final answer or use full response
        assert final_answer == "Mock response"

        # Should have called generate_response once
        assert len(mock_env.generate_response_calls) == 1
        call = mock_env.generate_response_calls[0]
        assert call["system_prompt"] == "You are a helpful assistant"
        assert len(call["messages"]) == 1
        assert call["messages"][0]["role"] == "user"
        assert call["messages"][0]["content"] == "What is 2+2?"

    @pytest.mark.asyncio
    async def test_manage_conversation_extracts_xml_answer(self, mock_task, mock_mode):
        """Test that XML-tagged answers are extracted."""
        env = MockEnvironment(responses=["Thinking... <answer>42</answer> Done!"])
        manager = SingleTurnManager()

        turns, final_answer = await manager.manage_conversation(
            initial_message="What is the answer?",
            system_prompt="System",
            env=env,
            mode=mock_mode,
            task=mock_task
        )

        assert len(turns) == 2
        assert final_answer == "42"

    @pytest.mark.asyncio
    async def test_manage_conversation_extracts_final_answer_pattern(self, mock_task, mock_mode):
        """Test that 'Final Answer:' pattern is extracted."""
        env = MockEnvironment(responses=["Let me calculate. Final Answer: 42"])
        manager = SingleTurnManager()

        turns, final_answer = await manager.manage_conversation(
            initial_message="Calculate",
            system_prompt="System",
            env=env,
            mode=mock_mode,
            task=mock_task
        )

        assert final_answer == "42"

    @pytest.mark.asyncio
    async def test_manage_conversation_uses_full_response_if_no_pattern(self, mock_task, mock_mode):
        """Test that full response is used when no answer pattern found."""
        env = MockEnvironment(responses=["Just a plain response"])
        manager = SingleTurnManager()

        turns, final_answer = await manager.manage_conversation(
            initial_message="Question",
            system_prompt="System",
            env=env,
            mode=mock_mode,
            task=mock_task
        )

        assert final_answer == "Just a plain response"

    def test_get_next_prompt_always_returns_none(self, mock_env, mock_task):
        """Test that get_next_prompt always returns None."""
        manager = SingleTurnManager()

        # Should always return None regardless of input
        assert manager.get_next_prompt("response", 0, mock_task, mock_env) is None
        assert manager.get_next_prompt("another response", 5, mock_task, mock_env) is None
        assert manager.get_next_prompt("", 100, mock_task, mock_env) is None

    @pytest.mark.asyncio
    async def test_manage_conversation_with_empty_response(self, mock_task, mock_mode):
        """Test handling of empty response."""
        env = MockEnvironment(responses=[""])
        manager = SingleTurnManager()

        turns, final_answer = await manager.manage_conversation(
            initial_message="Question",
            system_prompt="System",
            env=env,
            mode=mock_mode,
            task=mock_task
        )

        assert len(turns) == 2
        assert turns[1].content == ""
        assert final_answer == ""

    @pytest.mark.asyncio
    async def test_manage_conversation_preserves_turn_indices(self, mock_env, mock_task, mock_mode):
        """Test that turn indices are correctly assigned."""
        manager = SingleTurnManager()

        turns, _ = await manager.manage_conversation(
            initial_message="Test",
            system_prompt="System",
            env=mock_env,
            mode=mock_mode,
            task=mock_task
        )

        assert turns[0].turn_index == 0
        assert turns[1].turn_index == 1
