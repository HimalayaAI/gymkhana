"""Tests for ChatMode."""
import pytest
import sys
from pathlib import Path

# Add tests directory to path for conftest imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from gymkhana.envs.modes.chat import ChatMode
from gymkhana.envs.managers.single_turn import SingleTurnManager
from gymkhana.envs.managers.multi_turn import MultiTurnManager
from conftest import MockEnvironment, MockTask


class TestChatMode:
    """Test ChatMode implementation."""

    @pytest.mark.asyncio
    async def test_execute_single_with_single_turn_manager(self, mock_task):
        """Test single chat execution with SingleTurnManager."""
        env = MockEnvironment(responses=["This is the answer"])
        env.format_initial_message = lambda task: f"Question: {task.question}"
        env.build_system_prompt = lambda task: "You are a helpful assistant"
        env.config = type('Config', (), {'main_model': 'test-model'})()

        mode = ChatMode()
        manager = SingleTurnManager()

        result = await mode.execute_single(mock_task, env, manager)

        # Check result structure
        assert result.success is True
        assert result.final_answer == "This is the answer"
        assert len(result.turns) == 2  # user + assistant
        assert result.num_code_blocks == 0  # No code in chat mode
        assert result.num_errors == 0
        assert result.task_id == mock_task.id
        assert result.environment == "mock_env"
        assert result.system_prompt == "You are a helpful assistant"
        assert result.model_name == "test-model"

        # Check turns
        assert result.turns[0].role == "user"
        assert result.turns[0].content == "Question: Test question"
        assert result.turns[1].role == "assistant"
        assert result.turns[1].content == "This is the answer"

    @pytest.mark.asyncio
    async def test_execute_single_with_multi_turn_manager(self, mock_task):
        """Test chat execution with MultiTurnManager."""
        env = MockEnvironment(responses=[
            "Let me think...",
            "Still thinking...",
            "<answer>42</answer>"
        ])
        env.format_initial_message = lambda task: "What is the answer?"
        env.build_system_prompt = lambda task: "System"
        env.config = type('Config', (), {'main_model': 'test-model'})()

        mode = ChatMode()
        manager = MultiTurnManager(max_turns=10)

        # Override get_next_prompt to continue
        manager.get_next_prompt = lambda *args: "Continue"

        result = await mode.execute_single(mock_task, env, manager)

        # Should stop when final answer is found
        assert result.success is True
        assert result.final_answer == "42"
        assert len(result.turns) == 6  # user + 3 assistants + 2 continues
        assert result.num_code_blocks == 0
        assert result.num_errors == 0

    @pytest.mark.asyncio
    async def test_execute_single_no_final_answer(self, mock_task):
        """Test chat execution when no final answer is found."""
        env = MockEnvironment(responses=["Just a response"])
        env.format_initial_message = lambda task: "Question"
        env.build_system_prompt = lambda task: "System"
        env.config = type('Config', (), {'main_model': 'test-model'})()

        mode = ChatMode()
        manager = SingleTurnManager()

        result = await mode.execute_single(mock_task, env, manager)

        # Should still succeed but with response as final answer
        assert result.success is True
        assert result.final_answer == "Just a response"

    @pytest.mark.asyncio
    async def test_execute_single_empty_response(self, mock_task):
        """Test chat execution with empty response."""
        env = MockEnvironment(responses=[""])
        env.format_initial_message = lambda task: "Question"
        env.build_system_prompt = lambda task: "System"
        env.config = type('Config', (), {'main_model': 'test-model'})()

        mode = ChatMode()
        manager = SingleTurnManager()

        result = await mode.execute_single(mock_task, env, manager)

        # Empty response means no final answer
        assert result.success is True
        assert result.final_answer == ""

    @pytest.mark.asyncio
    async def test_execute_batch(self, mock_task):
        """Test batch execution of multiple rollouts."""
        env = MockEnvironment(responses=["Answer 1", "Answer 2", "Answer 3"])
        env.format_initial_message = lambda task: "Question"
        env.build_system_prompt = lambda task: "System"
        env.config = type('Config', (), {'main_model': 'test-model'})()

        mode = ChatMode()
        manager = SingleTurnManager()

        results = await mode.execute_batch(mock_task, env, manager, num_rollouts=3)

        # Should have 3 results
        assert len(results) == 3

        # All should be successful
        for result in results:
            assert result.success is True
            assert result.num_code_blocks == 0
            assert result.num_errors == 0
            assert result.task_id == mock_task.id

    @pytest.mark.asyncio
    async def test_execute_batch_independent_rollouts(self, mock_task):
        """Test that batch rollouts are independent."""
        # Each rollout should get its own response
        env = MockEnvironment(responses=["R1", "R2", "R3", "R4", "R5"])
        env.format_initial_message = lambda task: "Question"
        env.build_system_prompt = lambda task: "System"
        env.config = type('Config', (), {'main_model': 'test-model'})()

        mode = ChatMode()
        manager = SingleTurnManager()

        results = await mode.execute_batch(mock_task, env, manager, num_rollouts=3)

        # Each result should have different response
        assert len(results) == 3
        # Note: Due to parallel execution, responses might be consumed in order
        # but each rollout is independent
        for result in results:
            assert result.final_answer in ["R1", "R2", "R3", "R4", "R5"]

    @pytest.mark.asyncio
    async def test_execute_single_uses_env_methods(self, mock_task):
        """Test that execute_single calls environment methods correctly."""
        env = MockEnvironment(responses=["Answer"])

        # Track method calls
        format_called = []
        build_called = []

        def format_initial_message(task):
            format_called.append(task)
            return "Formatted message"

        def build_system_prompt(task):
            build_called.append(task)
            return "System prompt"

        env.format_initial_message = format_initial_message
        env.build_system_prompt = build_system_prompt
        env.config = type('Config', (), {'main_model': 'test-model'})()

        mode = ChatMode()
        manager = SingleTurnManager()

        result = await mode.execute_single(mock_task, env, manager)

        # Should have called both methods with the task
        assert len(format_called) == 1
        assert format_called[0] == mock_task
        assert len(build_called) == 1
        assert build_called[0] == mock_task

        # Should use the formatted message
        assert result.turns[0].content == "Formatted message"
        assert result.system_prompt == "System prompt"

    @pytest.mark.asyncio
    async def test_execute_single_without_config(self, mock_task):
        """Test execution when env.config doesn't have main_model."""
        env = MockEnvironment(responses=["Answer"])
        env.format_initial_message = lambda task: "Question"
        env.build_system_prompt = lambda task: "System"
        # No config attribute

        mode = ChatMode()
        manager = SingleTurnManager()

        result = await mode.execute_single(mock_task, env, manager)

        # Should still work, model_name will be None
        assert result.success is True
        assert result.model_name is None

    @pytest.mark.asyncio
    async def test_execute_batch_with_multi_turn(self, mock_task):
        """Test batch execution with multi-turn conversations."""
        env = MockEnvironment(responses=[
            "R1", "<answer>A1</answer>",  # Rollout 1
            "R2", "<answer>A2</answer>",  # Rollout 2
            "R3", "<answer>A3</answer>",  # Rollout 3
        ])
        env.format_initial_message = lambda task: "Question"
        env.build_system_prompt = lambda task: "System"
        env.config = type('Config', (), {'main_model': 'test-model'})()

        mode = ChatMode()
        manager = MultiTurnManager(max_turns=5)
        manager.get_next_prompt = lambda *args: "Continue"

        results = await mode.execute_batch(mock_task, env, manager, num_rollouts=3)

        # Should have 3 results, each with multiple turns
        assert len(results) == 3

        for result in results:
            assert result.success is True
            # Each should have stopped when answer was found
            assert result.final_answer in ["A1", "A2", "A3"]
