"""Tests for IFEval environment."""
import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch

# Add tests directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from gymkhana.envs.ifeval.ifeval import IfEvalEnv, DEFAULT_IFEVAL_CONFIG
from gymkhana.envs.environment import Task
from gymkhana.core.models import TrajectoryResult


class TestIfEvalEnvInitialization:
    """Test IFEval environment initialization."""

    def test_init_with_default_config(self):
        """Test initialization with default config."""
        env = IfEvalEnv()

        assert env.name == "ifeval"
        assert env.config is not None
        assert env._mode is not None
        assert env._manager is not None

    def test_init_with_custom_config(self):
        """Test initialization with custom config."""
        config = DEFAULT_IFEVAL_CONFIG.model_copy(deep=True)
        config.llm.model = "custom-model"

        env = IfEvalEnv(config=config)

        assert env.config.llm.model == "custom-model"

    def test_init_with_dict_config(self):
        """Test initialization with dict config."""
        config_dict = DEFAULT_IFEVAL_CONFIG.model_dump()
        config_dict["llm"]["model"] = "dict-model"

        env = IfEvalEnv(config=config_dict)

        assert env.config.llm.model == "dict-model"


class TestIfEvalEnvTaskLoading:
    """Test task loading from dataset."""

    @patch('gymkhana.envs.ifeval.ifeval.load_dataset')
    def test_load_tasks_basic(self, mock_load_dataset):
        """Test basic task loading."""
        # Mock dataset
        mock_dataset = [
            {
                "messages": [{"content": "Test prompt 1"}],
                "ground_truth": '{"func_name": "validate_lowercase"}',
                "constraint": "all lowercase",
                "constraint_type": "casing",
                "dataset": "ifeval"
            },
            {
                "messages": [{"content": "Test prompt 2"}],
                "ground_truth": '{"func_name": "validate_no_commas"}',
                "constraint": "no commas",
                "constraint_type": "punctuation",
                "dataset": "ifeval"
            }
        ]
        mock_load_dataset.return_value = iter(mock_dataset)

        env = IfEvalEnv()
        tasks = env.load_tasks(limit=2)

        assert len(tasks) == 2
        assert tasks[0].prompt == "Test prompt 1"
        assert tasks[1].prompt == "Test prompt 2"
        assert tasks[0].metadata["constraint"] == "all lowercase"
        assert tasks[1].metadata["constraint"] == "no commas"

    @patch('gymkhana.envs.ifeval.ifeval.load_dataset')
    def test_load_tasks_with_limit(self, mock_load_dataset):
        """Test task loading respects limit."""
        mock_dataset = [
            {"messages": [{"content": f"Prompt {i}"}], "ground_truth": "{}"}
            for i in range(10)
        ]
        mock_load_dataset.return_value = iter(mock_dataset)

        env = IfEvalEnv()
        tasks = env.load_tasks(limit=3)

        assert len(tasks) == 3

    @patch('gymkhana.envs.ifeval.ifeval.load_dataset')
    def test_load_tasks_parses_ground_truth(self, mock_load_dataset):
        """Test that ground_truth JSON is parsed correctly."""
        mock_dataset = [{
            "messages": [{"content": "Test"}],
            "ground_truth": '{"func_name": "validate_lowercase", "extra": "data"}',
        }]
        mock_load_dataset.return_value = iter(mock_dataset)

        env = IfEvalEnv()
        tasks = env.load_tasks(limit=1)

        assert len(tasks) == 1
        gt = tasks[0].metadata["ground_truth"]
        assert gt["func_name"] == "validate_lowercase"
        assert gt["extra"] == "data"

    @patch('gymkhana.envs.ifeval.ifeval.load_dataset')
    def test_load_tasks_handles_invalid_json(self, mock_load_dataset):
        """Test that invalid JSON in ground_truth is handled."""
        mock_dataset = [{
            "messages": [{"content": "Test"}],
            "ground_truth": 'invalid json',
        }]
        mock_load_dataset.return_value = iter(mock_dataset)

        env = IfEvalEnv()
        tasks = env.load_tasks(limit=1)

        assert len(tasks) == 1
        assert tasks[0].metadata["ground_truth"] == {}


class TestIfEvalEnvBehaviorHooks:
    """Test environment behavior hooks."""

    def test_get_environment_instructions_with_flag(self):
        """Test environment instructions when flag is True."""
        env = IfEvalEnv()
        env.config.dataset.include_instructions = True

        task = Task(id="1", prompt="Test")
        instructions = env.get_environment_instructions(task)

        assert len(instructions) > 0
        assert "Follow" in instructions or "instructions" in instructions.lower()

    def test_get_environment_instructions_without_flag(self):
        """Test environment instructions when flag is False."""
        env = IfEvalEnv()
        env.config.dataset.include_instructions = False

        task = Task(id="1", prompt="Test")
        instructions = env.get_environment_instructions(task)

        assert instructions == ""

    def test_format_initial_message(self):
        """Test that format_initial_message returns the prompt."""
        env = IfEvalEnv()
        task = Task(id="1", prompt="Test prompt")

        message = env.format_initial_message(task)

        assert message == "Test prompt"


class TestIfEvalEnvRewardComputation:
    """Test reward computation based on constraints."""

    @pytest.mark.asyncio
    async def test_compute_reward_lowercase_pass(self):
        """Test reward for passing lowercase constraint."""
        env = IfEvalEnv()

        task = Task(
            id="1",
            prompt="Test",
            metadata={
                "ground_truth": {"func_name": "validate_lowercase"}
            }
        )

        result = TrajectoryResult(
            success=True,
            final_answer="all lowercase text",
            turns=[],
            task_id="1"
        )

        reward = await env.compute_reward(result, task=task)
        assert reward == 1.0

    @pytest.mark.asyncio
    async def test_compute_reward_lowercase_fail(self):
        """Test reward for failing lowercase constraint."""
        env = IfEvalEnv()

        task = Task(
            id="1",
            prompt="Test",
            metadata={
                "ground_truth": {"func_name": "validate_lowercase"}
            }
        )

        result = TrajectoryResult(
            success=True,
            final_answer="Has Capital Letters",
            turns=[],
            task_id="1"
        )

        reward = await env.compute_reward(result, task=task)
        assert reward == 0.0

    @pytest.mark.asyncio
    async def test_compute_reward_no_commas_pass(self):
        """Test reward for passing no commas constraint."""
        env = IfEvalEnv()

        task = Task(
            id="1",
            prompt="Test",
            metadata={
                "ground_truth": {"func_name": "validate_no_commas"}
            }
        )

        result = TrajectoryResult(
            success=True,
            final_answer="No commas here",
            turns=[],
            task_id="1"
        )

        reward = await env.compute_reward(result, task=task)
        assert reward == 1.0

    @pytest.mark.asyncio
    async def test_compute_reward_no_func_name(self):
        """Test reward when no func_name in ground_truth."""
        env = IfEvalEnv()

        task = Task(
            id="1",
            prompt="Test",
            metadata={"ground_truth": {}}
        )

        result = TrajectoryResult(
            success=True,
            final_answer="test",
            turns=[],
            task_id="1"
        )

        reward = await env.compute_reward(result, task=task)
        assert reward == 0.0

    @pytest.mark.asyncio
    async def test_compute_reward_unknown_validator(self):
        """Test reward when validator doesn't exist."""
        env = IfEvalEnv()

        task = Task(
            id="1",
            prompt="Test",
            metadata={
                "ground_truth": {"func_name": "nonexistent_validator"}
            }
        )

        result = TrajectoryResult(
            success=True,
            final_answer="test",
            turns=[],
            task_id="1"
        )

        reward = await env.compute_reward(result, task=task)
        assert reward == 0.0

    @pytest.mark.asyncio
    async def test_compute_reward_empty_answer(self):
        """Test reward with empty final answer."""
        env = IfEvalEnv()

        task = Task(
            id="1",
            prompt="Test",
            metadata={
                "ground_truth": {"func_name": "validate_lowercase"}
            }
        )

        result = TrajectoryResult(
            success=False,
            final_answer="",
            turns=[],
            task_id="1"
        )

        reward = await env.compute_reward(result, task=task)
        assert reward == 0.0


class TestIfEvalEnvExecuteTask:
    """Test task execution with ChatMode."""

    @pytest.mark.asyncio
    async def test_execute_task_basic(self):
        """Test basic task execution."""
        env = IfEvalEnv()

        # Mock the mode's execute_single method
        mock_result = TrajectoryResult(
            success=True,
            final_answer="test response",
            turns=[],
            task_id="1"
        )
        env._mode.execute_single = AsyncMock(return_value=mock_result)

        task = Task(
            id="1",
            prompt="Test prompt",
            metadata={"ground_truth": {"func_name": "validate_lowercase"}}
        )

        result = await env.execute_task(task)

        assert result.success is True
        assert result.final_answer == "test response"
        env._mode.execute_single.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_task_with_rewards(self):
        """Test task execution computes rewards."""
        env = IfEvalEnv()
        env.config.dataset.enable_rewards = True

        # Mock the mode's execute_single method
        mock_result = TrajectoryResult(
            success=True,
            final_answer="all lowercase",
            turns=[],
            task_id="1"
        )
        env._mode.execute_single = AsyncMock(return_value=mock_result)

        task = Task(
            id="1",
            prompt="Test",
            metadata={"ground_truth": {"func_name": "validate_lowercase"}}
        )

        result = await env.execute_task(task)

        assert result.total_reward == 1.0
        assert result.step_rewards == [1.0]

    @pytest.mark.asyncio
    async def test_execute_task_without_rewards(self):
        """Test task execution without reward computation."""
        env = IfEvalEnv()
        env.config.dataset.enable_rewards = False

        # Mock the mode's execute_single method
        mock_result = TrajectoryResult(
            success=True,
            final_answer="test",
            turns=[],
            task_id="1"
        )
        env._mode.execute_single = AsyncMock(return_value=mock_result)

        task = Task(id="1", prompt="Test", metadata={})

        result = await env.execute_task(task)

        # Reward should not be computed
        assert result.total_reward == 0.0



class TestIfEvalDatasetStructure:
    """Test that we correctly parse the actual dataset structure."""

    @patch('gymkhana.envs.ifeval.ifeval.load_dataset')
    def test_parse_real_dataset_example(self, mock_load_dataset):
        """Test parsing a real example from allenai/RLVR-IFeval."""
        # Real example from the dataset
        mock_dataset = [{
            "messages": [{
                "content": "Answer a question about this article:\n"
                          "Increasingly, more modern games such as video games and slot machines "
                          "are provided. Pubs hold special events, from tournaments of the "
                          "aforementioned games to karaoke nights to pub quizzes. Some play pop "
                          "music and hip-hop (dance bar), or show football and rugby union on big "
                          "screen televisions (sports bar). Shove ha'penny and Bat and trap were "
                          "also popular in pubs south of London.\n"
                          "Along with slot machines, what is a modern game that is increasingly "
                          "present in pubs? In your response, the word nonsensorial should appear "
                          "17 times.",
                "role": "user"
            }],
            "constraint": "In your response, the word {word} should appear {N} times.",
            "ground_truth": '{"func_name": "verify_keyword_frequency", "N": 17, '
                          '"quantifier": null, "end_phrase": null, "keyword_list": null, '
                          '"word": "nonsensorial", "forbidden_words": null, "letter": null, '
                          '"i": null, "first_word": null, "postscript_marker": null, '
                          '"options": null, "section_splitter": null, "original_prompt": null}',
            "constraint_type": "Keyword Frequency",
            "dataset": "ifeval"
        }]
        mock_load_dataset.return_value = iter(mock_dataset)

        env = IfEvalEnv()
        tasks = env.load_tasks(limit=1)

        # Verify task was created correctly
        assert len(tasks) == 1
        task = tasks[0]

        # Check prompt extraction
        assert "nonsensorial should appear 17 times" in task.prompt
        assert "video games and slot machines" in task.prompt

        # Check ground_truth parsing
        gt = task.metadata["ground_truth"]
        assert gt["func_name"] == "verify_keyword_frequency"
        assert gt["N"] == 17
        assert gt["word"] == "nonsensorial"

        # Check metadata
        assert task.metadata["constraint_type"] == "Keyword Frequency"
        assert task.metadata["constraint"] == "In your response, the word {word} should appear {N} times."

    @pytest.mark.asyncio
    async def test_validate_real_example_pass(self):
        """Test validation with a response that satisfies the constraint."""
        env = IfEvalEnv()

        task = Task(
            id="1",
            prompt="Test",
            metadata={
                "ground_truth": {
                    "func_name": "verify_keyword_frequency",
                    "N": 17,
                    "word": "nonsensorial",
                    "keyword_list": ["nonsensorial"]
                }
            }
        )

        # Create a response with "nonsensorial" appearing exactly 17 times
        response = " ".join(["nonsensorial"] * 17)
        result = TrajectoryResult(
            success=True,
            final_answer=response,
            turns=[],
            task_id="1"
        )

        reward = await env.compute_reward(result, task=task)
        assert reward == 1.0

    @pytest.mark.asyncio
    async def test_validate_real_example_fail(self):
        """Test validation with a response that fails the constraint."""
        env = IfEvalEnv()

        task = Task(
            id="1",
            prompt="Test",
            metadata={
                "ground_truth": {
                    "func_name": "verify_keyword_frequency",
                    "N": 17,
                    "word": "nonsensorial",
                    "keyword_list": ["nonsensorial"]
                }
            }
        )

        # Create a response with "nonsensorial" appearing only 5 times
        response = " ".join(["nonsensorial"] * 5)
        result = TrajectoryResult(
            success=True,
            final_answer=response,
            turns=[],
            task_id="1"
        )

        reward = await env.compute_reward(result, task=task)
        assert reward == 0.0

    def test_single_turn_manager_used(self):
        """Verify that IFEval uses SingleTurnManager (single-turn task)."""
        env = IfEvalEnv()

        # Should use SingleTurnManager for single-turn tasks
        from gymkhana.envs.managers.single_turn import SingleTurnManager
        assert isinstance(env._manager, SingleTurnManager)
        assert env._manager.max_turns == 1

    def test_chat_mode_used(self):
        """Verify that IFEval uses ChatMode (text generation only)."""
        env = IfEvalEnv()

        # Should use ChatMode for pure text generation
        from gymkhana.envs.modes.chat import ChatMode
        assert isinstance(env._mode, ChatMode)
