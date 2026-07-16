"""Tests for MultiTurnManager."""
import pytest
import sys
from pathlib import Path

# Add tests directory to path for conftest imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from gymkhana.envs.managers.multi_turn import MultiTurnManager
from conftest import MockEnvironment


class TestMultiTurnManager:
    """Test MultiTurnManager implementation."""

    def test_initialization_default(self):
        """Test manager initializes with default max_turns."""
        manager = MultiTurnManager()
        assert manager.max_turns == 10

    def test_initialization_custom(self):
        """Test manager initializes with custom max_turns."""
        manager = MultiTurnManager(max_turns=5)
        assert manager.max_turns == 5

    @pytest.mark.asyncio
    async def test_manage_conversation_stops_on_final_answer(self, mock_task, mock_mode):
        """Test conversation stops when final answer is found."""
        env = MockEnvironment(responses=[
            "Let me think...",
            "Still thinking...",
            "<answer>42</answer>"
        ])
        manager = MultiTurnManager(max_turns=10)

        # Override get_next_prompt to always continue
        manager.get_next_prompt = lambda *args: "Continue"

        turns, final_answer = await manager.manage_conversation(
            initial_message="What is the answer?",
            system_prompt="System",
            env=env,
            mode=mock_mode,
            task=mock_task
        )

        # Should stop after finding answer (3 assistant turns + 3 user turns = 6 total)
        # Turn 0: user "What is the answer?"
        # Turn 1: assistant "Let me think..."
        # Turn 2: user "Continue"
        # Turn 3: assistant "Still thinking..."
        # Turn 4: user "Continue"
        # Turn 5: assistant "<answer>42</answer>"
        assert len(turns) == 6
        assert final_answer == "42"

        # Should have called generate_response 3 times
        assert len(env.generate_response_calls) == 3

    @pytest.mark.asyncio
    async def test_manage_conversation_stops_on_max_turns(self, mock_task, mock_mode):
        """Test conversation stops at max_turns."""
        env = MockEnvironment(responses=["Response"] * 10)
        manager = MultiTurnManager(max_turns=3)

        # Override get_next_prompt to always continue
        manager.get_next_prompt = lambda *args: "Continue"

        turns, final_answer = await manager.manage_conversation(
            initial_message="Question",
            system_prompt="System",
            env=env,
            mode=mock_mode,
            task=mock_task
        )

        # Should stop after max_turns (3) assistant responses
        # Turn 0: user "Question"
        # Turn 1: assistant "Response"
        # Turn 2: user "Continue"
        # Turn 3: assistant "Response"
        # Turn 4: user "Continue"
        # Turn 5: assistant "Response"
        # Turn 6: user "Continue" (added before loop exits)
        assert len(turns) == 7  # Updated expectation
        assert final_answer is None  # No answer found

        # Should have called generate_response exactly max_turns times
        assert len(env.generate_response_calls) == 3

    @pytest.mark.asyncio
    async def test_manage_conversation_stops_on_none_prompt(self, mock_task, mock_mode):
        """Test conversation stops when get_next_prompt returns None."""
        env = MockEnvironment(responses=["Response 1", "Response 2"])
        manager = MultiTurnManager(max_turns=10)

        # get_next_prompt returns None by default

        turns, final_answer = await manager.manage_conversation(
            initial_message="Question",
            system_prompt="System",
            env=env,
            mode=mock_mode,
            task=mock_task
        )

        # Should stop after first response (get_next_prompt returns None)
        assert len(turns) == 2  # user + assistant
        assert final_answer is None

        # Should have called generate_response once
        assert len(env.generate_response_calls) == 1

    @pytest.mark.asyncio
    async def test_manage_conversation_builds_message_history(self, mock_task, mock_mode):
        """Test that message history is built correctly."""
        env = MockEnvironment(responses=["Response 1", "Response 2", "<answer>Done</answer>"])
        manager = MultiTurnManager(max_turns=10)

        # Override get_next_prompt to continue twice
        call_count = [0]
        def get_next_prompt(*args):
            call_count[0] += 1
            return "Continue" if call_count[0] <= 2 else None
        manager.get_next_prompt = get_next_prompt

        turns, final_answer = await manager.manage_conversation(
            initial_message="Start",
            system_prompt="System",
            env=env,
            mode=mock_mode,
            task=mock_task
        )

        # Check that messages were accumulated
        assert len(env.generate_response_calls) == 3

        # First call: just initial message
        assert len(env.generate_response_calls[0]["messages"]) == 1

        # Second call: initial + response + continue
        assert len(env.generate_response_calls[1]["messages"]) == 3

        # Third call: all previous messages
        assert len(env.generate_response_calls[2]["messages"]) == 5

    @pytest.mark.asyncio
    async def test_manage_conversation_turn_indices(self, mock_task, mock_mode):
        """Test that turn indices are correctly assigned."""
        env = MockEnvironment(responses=["R1", "R2", "<answer>Done</answer>"])
        manager = MultiTurnManager(max_turns=10)

        # Continue twice
        call_count = [0]
        def get_next_prompt(*args):
            call_count[0] += 1
            return "Continue" if call_count[0] <= 2 else None
        manager.get_next_prompt = get_next_prompt

        turns, _ = await manager.manage_conversation(
            initial_message="Start",
            system_prompt="System",
            env=env,
            mode=mock_mode,
            task=mock_task
        )

        # Check turn indices are sequential
        for i, turn in enumerate(turns):
            assert turn.turn_index == i

    @pytest.mark.asyncio
    async def test_manage_conversation_alternating_roles(self, mock_task, mock_mode):
        """Test that roles alternate correctly."""
        env = MockEnvironment(responses=["R1", "R2"])
        manager = MultiTurnManager(max_turns=10)

        # Continue once
        call_count = [0]
        def get_next_prompt(*args):
            call_count[0] += 1
            return "Continue" if call_count[0] == 1 else None
        manager.get_next_prompt = get_next_prompt

        turns, _ = await manager.manage_conversation(
            initial_message="Start",
            system_prompt="System",
            env=env,
            mode=mock_mode,
            task=mock_task
        )

        # Should be: user, assistant, user, assistant
        assert turns[0].role == "user"
        assert turns[1].role == "assistant"
        assert turns[2].role == "user"
        assert turns[3].role == "assistant"

    def test_get_next_prompt_default_returns_none(self, mock_env, mock_task):
        """Test that default get_next_prompt returns None."""
        manager = MultiTurnManager()

        result = manager.get_next_prompt("response", 0, mock_task, mock_env)
        assert result is None

    @pytest.mark.asyncio
    async def test_manage_conversation_with_empty_initial_message(self, mock_env, mock_task, mock_mode):
        """Test handling of empty initial message."""
        manager = MultiTurnManager()

        turns, final_answer = await manager.manage_conversation(
            initial_message="",
            system_prompt="System",
            env=mock_env,
            mode=mock_mode,
            task=mock_task
        )

        assert len(turns) == 2
        assert turns[0].content == ""
        assert turns[0].role == "user"

    @pytest.mark.asyncio
    async def test_manage_conversation_extracts_different_answer_patterns(self, mock_task, mock_mode):
        """Test extraction of different answer patterns."""
        # Test "Final Answer:" pattern
        env = MockEnvironment(responses=["Final Answer: 42"])
        manager = MultiTurnManager()

        turns, final_answer = await manager.manage_conversation(
            initial_message="Q",
            system_prompt="S",
            env=env,
            mode=mock_mode,
            task=mock_task
        )
        assert final_answer == "42"

        # Test "The answer is:" pattern
        env = MockEnvironment(responses=["The answer is: Hello"])
        manager = MultiTurnManager()

        turns, final_answer = await manager.manage_conversation(
            initial_message="Q",
            system_prompt="S",
            env=env,
            mode=mock_mode,
            task=mock_task
        )
        assert final_answer == "Hello"
