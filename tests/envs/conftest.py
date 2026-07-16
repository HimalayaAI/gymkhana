"""Shared fixtures for environment tests."""
import pytest
from typing import List, Dict, Any, Optional
from unittest.mock import AsyncMock, Mock

from gymkhana.core.models import Turn


class MockEnvironment:
    """Mock environment for testing."""

    def __init__(self, responses: Optional[List[str]] = None):
        """Initialize mock environment.

        Args:
            responses: List of responses to return in order
        """
        self.responses = responses or ["Mock response"]
        self.response_idx = 0
        self.generate_response_calls = []
        self.name = "mock_env"
        self.data_inserter = None

    async def generate_response(
        self,
        messages: List[Dict[str, str]],
        system_prompt: str,
        **kwargs
    ) -> str:
        """Mock generate_response method."""
        # Store a copy of messages to avoid reference issues
        self.generate_response_calls.append({
            "messages": list(messages),  # Make a copy
            "system_prompt": system_prompt,
            "kwargs": kwargs
        })

        if self.response_idx < len(self.responses):
            response = self.responses[self.response_idx]
            self.response_idx += 1
            return response

        return "No more responses"


class MockTask:
    """Mock task for testing."""

    def __init__(self, task_id: str = "test_task"):
        self.id = task_id
        self.question = "Test question"
        self.expected_answer = "42"


class MockInteractionMode:
    """Mock interaction mode for testing."""

    def __init__(self):
        self.execute_single_called = False
        self.execute_batch_called = False

    async def execute_single(self, task, env, conversation_manager):
        """Mock execute_single."""
        self.execute_single_called = True
        from gymkhana.core.models import TrajectoryResult
        return TrajectoryResult(
            success=True,
            final_answer="42",
            turns=[],
            task_id=task.id
        )

    async def execute_batch(self, task, env, conversation_manager, num_rollouts):
        """Mock execute_batch."""
        self.execute_batch_called = True
        from gymkhana.core.models import TrajectoryResult
        return [
            TrajectoryResult(
                success=True,
                final_answer="42",
                turns=[],
                task_id=task.id,
                total_reward=0.8 + i * 0.1
            )
            for i in range(num_rollouts)
        ]


@pytest.fixture
def mock_env():
    """Fixture providing a mock environment."""
    return MockEnvironment()


@pytest.fixture
def mock_task():
    """Fixture providing a mock task."""
    return MockTask()


@pytest.fixture
def mock_mode():
    """Fixture providing a mock interaction mode."""
    return MockInteractionMode()
