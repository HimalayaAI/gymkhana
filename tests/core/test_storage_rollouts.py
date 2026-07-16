"""Unit tests for rollout tracking storage methods."""

import asyncio
from datetime import datetime, timezone
from uuid import uuid4

import pytest

from gymkhana.core.models.trajectory import (
    RolloutGroup,
    RolloutState,
    RolloutStatus,
    TrajectoryResult,
)
from gymkhana.core.services.storage.env_storage import EnvStorageService


# Mock database for testing (in-memory)
class MockAsyncDatabase:
    """Mock database for testing without actual PostgreSQL."""

    def __init__(self):
        self.rollout_groups = {}
        self.rollouts = {}
        self.trajectories = {}
        self.is_initialized = False

    async def initialize(self):
        self.is_initialized = True

    async def close(self):
        self.is_initialized = False

    async def execute(self, query, *args):
        # Mock execute - just store data
        if "INSERT INTO rollout_groups" in query:
            group_id = args[0]
            self.rollout_groups[group_id] = {
                'id': args[0],
                'task_id': args[1],
                'environment': args[2],
                'num_rollouts': args[3],
            }
        elif "INSERT INTO rollouts" in query:
            rollout_id = args[0]
            self.rollouts[rollout_id] = {
                'id': args[0],
                'rollout_group_id': args[1],
                'rollout_index': args[3],
                'status': args[4],
            }
        elif "UPDATE rollouts" in query:
            rollout_id = args[0]
            if rollout_id in self.rollouts:
                self.rollouts[rollout_id]['status'] = args[1]
        return "OK"

    async def fetch_one(self, query, *args):
        if "SELECT * FROM rollout_groups" in query:
            group_id = args[0]
            if group_id in self.rollout_groups:
                return self.rollout_groups[group_id]
        elif "SELECT id FROM rollouts" in query and "ORDER BY total_reward DESC" in query:
            # Return first rollout as best
            if self.rollouts:
                return {'id': list(self.rollouts.keys())[0]}
        return None

    async def fetch(self, query, *args):
        if "SELECT * FROM rollouts" in query:
            group_id = args[0]
            return [r for r in self.rollouts.values() if r.get('rollout_group_id') == group_id]
        return []


@pytest.fixture
def mock_storage():
    """Create a mock storage service for testing."""
    storage = EnvStorageService(
        host="localhost",
        port=5432,
        user="test",
        password="test",
        db_name="test"
    )
    # Replace with mock database
    storage._db = MockAsyncDatabase()
    storage._is_initialized = False
    return storage


class TestRolloutGroupStorage:
    """Tests for rollout group storage methods."""

    @pytest.mark.asyncio
    async def test_insert_rollout_group(self, mock_storage):
        """Test inserting a rollout group."""
        await mock_storage.initialize()

        group = RolloutGroup(
            task_id="test_task_123",
            environment="math_reasoning",
            num_rollouts=8
        )

        group_id = await mock_storage.insert_rollout_group(group)

        assert group_id == group.id
        assert group.id in mock_storage._db.rollout_groups

        stored = mock_storage._db.rollout_groups[group.id]
        assert stored['task_id'] == "test_task_123"
        assert stored['environment'] == "math_reasoning"
        assert stored['num_rollouts'] == 8

    @pytest.mark.asyncio
    async def test_get_rollout_group(self, mock_storage):
        """Test retrieving a rollout group."""
        await mock_storage.initialize()

        # Insert a group
        group = RolloutGroup(
            task_id="test_task",
            environment="test_env",
            num_rollouts=4
        )
        await mock_storage.insert_rollout_group(group)

        # Retrieve it
        retrieved = await mock_storage.get_rollout_group(group.id)

        assert retrieved is not None
        assert retrieved.id == group.id
        assert retrieved.task_id == "test_task"
        assert retrieved.environment == "test_env"
        assert retrieved.num_rollouts == 4


class TestRolloutStorage:
    """Tests for rollout state storage methods."""

    @pytest.mark.asyncio
    async def test_insert_rollout(self, mock_storage):
        """Test inserting a rollout state."""
        await mock_storage.initialize()

        # Create group first
        group = RolloutGroup(
            task_id="test_task",
            environment="test_env",
            num_rollouts=4
        )
        group_id = await mock_storage.insert_rollout_group(group)

        # Create rollout
        rollout = RolloutState(
            rollout_id=0,
            status=RolloutStatus.ACTIVE
        )

        rollout_id = await mock_storage.insert_rollout(rollout, group_id)

        assert rollout_id == rollout.id
        assert rollout.id in mock_storage._db.rollouts

        stored = mock_storage._db.rollouts[rollout.id]
        assert stored['rollout_group_id'] == group_id
        assert stored['rollout_index'] == 0
        assert stored['status'] == "active"

    @pytest.mark.asyncio
    async def test_update_rollout(self, mock_storage):
        """Test updating a rollout state."""
        await mock_storage.initialize()

        # Create group and rollout
        group = RolloutGroup(
            task_id="test_task",
            environment="test_env",
            num_rollouts=4
        )
        group_id = await mock_storage.insert_rollout_group(group)

        rollout = RolloutState(
            rollout_id=0,
            status=RolloutStatus.ACTIVE
        )
        await mock_storage.insert_rollout(rollout, group_id)

        # Update rollout
        rollout.mark_completed()
        rollout.num_turns = 5
        rollout.total_reward = 0.85

        await mock_storage.update_rollout(rollout)

        # Verify update
        stored = mock_storage._db.rollouts[rollout.id]
        assert stored['status'] == "completed"

    @pytest.mark.asyncio
    async def test_get_rollouts_by_group(self, mock_storage):
        """Test retrieving all rollouts for a group."""
        await mock_storage.initialize()

        # Create group
        group = RolloutGroup(
            task_id="test_task",
            environment="test_env",
            num_rollouts=3
        )
        group_id = await mock_storage.insert_rollout_group(group)

        # Create multiple rollouts
        rollouts = []
        for i in range(3):
            rollout = RolloutState(
                rollout_id=i,
                status=RolloutStatus.ACTIVE
            )
            await mock_storage.insert_rollout(rollout, group_id)
            rollouts.append(rollout)

        # Retrieve all
        retrieved = await mock_storage.get_rollouts_by_group(group_id)

        assert len(retrieved) == 3


class TestRolloutGroupStatistics:
    """Tests for rollout group statistics methods."""

    @pytest.mark.asyncio
    async def test_update_rollout_group_statistics(self, mock_storage):
        """Test updating rollout group statistics."""
        await mock_storage.initialize()

        # Create group
        group = RolloutGroup(
            task_id="test_task",
            environment="test_env",
            num_rollouts=5
        )
        group_id = await mock_storage.insert_rollout_group(group)

        # Create rollouts with different rewards
        for i, reward in enumerate([0.5, 0.7, 0.9, 0.6, 0.8]):
            rollout = RolloutState(
                rollout_id=i,
                status=RolloutStatus.COMPLETED,
                total_reward=reward
            )
            await mock_storage.insert_rollout(rollout, group_id)

        # Update statistics
        rewards = [0.5, 0.7, 0.9, 0.6, 0.8]
        status_counts = {'completed': 5, 'failed': 0, 'error': 0, 'timeout': 0}

        await mock_storage.update_rollout_group_statistics(
            group_id, rewards, status_counts
        )

        # Note: In real implementation, we'd verify the statistics were updated
        # For mock, we just verify it doesn't crash


class TestTrajectoryWithRollouts:
    """Tests for trajectory storage with rollout tracking."""

    @pytest.mark.asyncio
    async def test_store_trajectory_with_rollout_refs(self, mock_storage):
        """Test storing trajectory with rollout references."""
        await mock_storage.initialize()

        # Create group and rollout
        group = RolloutGroup(
            task_id="test_task",
            environment="test_env",
            num_rollouts=4
        )
        group_id = await mock_storage.insert_rollout_group(group)

        rollout = RolloutState(
            rollout_id=0,
            status=RolloutStatus.ACTIVE
        )
        rollout_id = await mock_storage.insert_rollout(rollout, group_id)

        # Create trajectory
        trajectory = TrajectoryResult(
            success=True,
            final_answer="42",
            rollout_id=rollout_id,
            rollout_group_id=group_id,
            rollout_index=0
        )

        # Store trajectory
        traj_id = await mock_storage.store_trajectory(
            trajectory=trajectory,
            task_id="test_task",
            environment="test_env",
            rollout_id=rollout_id,
            rollout_group_id=group_id
        )

        assert traj_id == trajectory.id


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
