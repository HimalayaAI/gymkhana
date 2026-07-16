"""Integration tests for rollout tracking storage with real database."""

import asyncio
import os
from datetime import datetime, timezone
from uuid import uuid4

import pytest

from gymkhana.core.models.trajectory import (
    RolloutGroup,
    RolloutState,
    RolloutStatus,
    TrajectoryResult,
    Turn,
)
from gymkhana.core.services.storage.env_storage import EnvStorageService


# Skip if database not available
pytestmark = pytest.mark.skipif(
    not os.getenv("DB_HOST"),
    reason="Database not configured (set DB_HOST env var)"
)


@pytest.fixture
async def storage():
    """Create storage service connected to test database."""
    service = EnvStorageService(
        host=os.getenv("DB_HOST", "localhost"),
        port=int(os.getenv("DB_PORT", "5433")),
        user=os.getenv("DB_USER", "db_user"),
        password=os.getenv("DB_PASSWORD", "db_pwd@123"),
        db_name=os.getenv("DB_NAME", "gymkhana"),
    )
    service.schema_path = None  # Skip schema application for tests

    await service.initialize()
    yield service
    await service.close()


@pytest.mark.asyncio
async def test_full_rollout_lifecycle(storage):
    """Test complete rollout lifecycle: create group, rollouts, trajectories, link, update stats."""

    # 1. Create rollout group
    group = RolloutGroup(
        task_id="integration_test_task_001",
        environment="test_env",
        num_rollouts=3,
        config={
            "termination_policy": {"max_consecutive_errors": 3},
            "reward_function": "simple"
        }
    )

    group_id = await storage.insert_rollout_group(group)
    assert group_id == group.id
    print(f"✓ Created rollout group: {group_id}")

    # 2. Create rollouts
    rollout_states = []
    rollout_ids = []

    for i in range(3):
        rollout = RolloutState(
            rollout_id=i,
            status=RolloutStatus.ACTIVE,
            session_id=f"test_session_{i}"
        )

        rollout_id = await storage.insert_rollout(rollout, group_id)
        assert rollout_id == rollout.id

        rollout_states.append(rollout)
        rollout_ids.append(rollout_id)
        print(f"✓ Created rollout {i}: {rollout_id}")

    # 3. Simulate execution and update rollouts
    rewards = []
    for i, rollout in enumerate(rollout_states):
        # Simulate some execution
        rollout.record_execution(success=True, reward=0.5)
        rollout.num_turns = 3 + i
        rollout.mark_completed()

        await storage.update_rollout(rollout)
        rewards.append(rollout.total_reward)
        print(f"✓ Updated rollout {i}: reward={rollout.total_reward}, turns={rollout.num_turns}")

    # 4. Create trajectories for each rollout
    trajectory_ids = []

    for i, rollout in enumerate(rollout_states):
        trajectory = TrajectoryResult(
            success=True,
            final_answer=f"Answer {i}",
            turns=[
                Turn(role="user", content="Question", turn_index=0),
                Turn(role="assistant", content="Response", turn_index=1),
            ],
            num_code_blocks=2,
            total_reward=rollout.total_reward,
            rollout_id=rollout.id,
            rollout_group_id=group_id,
            rollout_index=i
        )

        traj_id = await storage.store_trajectory(
            trajectory=trajectory,
            task_id="integration_test_task_001",
            environment="test_env",
            rollout_id=rollout.id,
            rollout_group_id=group_id
        )

        trajectory_ids.append(traj_id)
        print(f"✓ Created trajectory {i}: {traj_id}")

        # 5. Link rollout to trajectory
        await storage.link_rollout_to_trajectory(rollout.id, traj_id)
        print(f"✓ Linked rollout {i} to trajectory")

    # 6. Update group statistics
    status_counts = {
        'completed': 3,
        'failed': 0,
        'error': 0,
        'timeout': 0
    }

    await storage.update_rollout_group_statistics(
        group_id,
        rewards,
        status_counts
    )
    print(f"✓ Updated group statistics")

    # 7. Verify retrieval
    retrieved_group = await storage.get_rollout_group(group_id)
    assert retrieved_group is not None
    assert retrieved_group.task_id == "integration_test_task_001"
    assert retrieved_group.num_completed == 3
    assert retrieved_group.best_reward == max(rewards)
    print(f"✓ Retrieved group: mean_reward={retrieved_group.reward_mean:.3f}")

    retrieved_rollouts = await storage.get_rollouts_by_group(group_id)
    assert len(retrieved_rollouts) == 3
    print(f"✓ Retrieved {len(retrieved_rollouts)} rollouts")

    # 8. Verify trajectories were stored
    for traj_id in trajectory_ids:
        traj = await storage.get_trajectory(traj_id)
        assert traj is not None
        assert traj.rollout_group_id == group_id
        print(f"✓ Verified trajectory: {traj_id}")

    print("\n✅ Full rollout lifecycle test passed!")


@pytest.mark.asyncio
async def test_rollout_with_format_violations(storage):
    """Test rollout with format violations tracking."""

    # Create group
    group = RolloutGroup(
        task_id="format_test_task",
        environment="test_env",
        num_rollouts=1
    )
    group_id = await storage.insert_rollout_group(group)

    # Create rollout with format violations
    rollout = RolloutState(
        rollout_id=0,
        status=RolloutStatus.ACTIVE
    )

    # Record format violations
    rollout.record_execution(
        success=True,
        reward=0.3,
        has_format_violation=True,
        format_violation_type="hallucinated_tags"
    )
    rollout.record_execution(
        success=True,
        reward=0.2,
        has_format_violation=True,
        format_violation_type="malformed_xml"
    )

    rollout.mark_failed("Format violations: 2")

    rollout_id = await storage.insert_rollout(rollout, group_id)

    # Retrieve and verify
    retrieved = await storage.get_rollouts_by_group(group_id)
    assert len(retrieved) == 1
    assert retrieved[0].num_format_violations == 2
    assert retrieved[0].status == RolloutStatus.FAILED
    assert "hallucinated_tags" in retrieved[0].format_violation_history
    assert "malformed_xml" in retrieved[0].format_violation_history

    print("✅ Format violation tracking test passed!")


@pytest.mark.asyncio
async def test_rollout_termination_reasons(storage):
    """Test different termination reasons."""

    group = RolloutGroup(
        task_id="termination_test_task",
        environment="test_env",
        num_rollouts=4
    )
    group_id = await storage.insert_rollout_group(group)

    # Create rollouts with different termination reasons
    termination_cases = [
        (RolloutStatus.COMPLETED, None),
        (RolloutStatus.FAILED, "Consecutive errors: 3/3"),
        (RolloutStatus.ERROR, "Exception during execution"),
        (RolloutStatus.TIMEOUT, "Max turns reached without completion"),
    ]

    for i, (status, reason) in enumerate(termination_cases):
        rollout = RolloutState(rollout_id=i, status=RolloutStatus.ACTIVE)

        if status == RolloutStatus.COMPLETED:
            rollout.mark_completed()
        elif status == RolloutStatus.FAILED:
            rollout.mark_failed(reason)
        elif status == RolloutStatus.ERROR:
            rollout.mark_error(reason)
        elif status == RolloutStatus.TIMEOUT:
            rollout.mark_timeout()

        await storage.insert_rollout(rollout, group_id)

    # Retrieve and verify
    rollouts = await storage.get_rollouts_by_group(group_id)
    assert len(rollouts) == 4

    for rollout, (expected_status, expected_reason) in zip(rollouts, termination_cases):
        assert rollout.status == expected_status
        if expected_reason:
            assert rollout.termination_reason == expected_reason

    print("✅ Termination reasons test passed!")


if __name__ == "__main__":
    # Run tests manually
    async def main():
        storage = EnvStorageService(
            host=os.getenv("DB_HOST", "localhost"),
            port=int(os.getenv("DB_PORT", "5433")),
            user=os.getenv("DB_USER", "db_user"),
            password=os.getenv("DB_PASSWORD", "db_pwd@123"),
            db_name=os.getenv("DB_NAME", "gymkhana"),
        )
        storage.schema_path = None  # Skip schema application for tests

        await storage.initialize()

        try:
            print("Running integration tests...\n")
            await test_full_rollout_lifecycle(storage)
            print()
            await test_rollout_with_format_violations(storage)
            print()
            await test_rollout_termination_reasons(storage)
            print("\n✅ All integration tests passed!")
        finally:
            await storage.close()

    asyncio.run(main())
