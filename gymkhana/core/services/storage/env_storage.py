"""SQL implementation of storage services for Gymkhana environments.

Provides persistence using PostgreSQL via AsyncDatabase and SQLModel.
Includes support for automatic database provisioning via Docker.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import ConfigDict

# Handle optional dependencies for database implementation
# Import local database wrapper
from gymkhana.core.services.storage.db import AsyncDatabase, HAS_ASYNCPG
STORAGE_AVAILABLE = HAS_ASYNCPG

from gymkhana.core.models.trajectory import (
    TrajectoryResult,
    Turn,
    RolloutGroup,
    RolloutState,
    RolloutStatus
)
from gymkhana.core.models.execution import ExecutionResult
from gymkhana.core.models.sandbox import SessionState
# DB models are imported lazily to avoid triggering SQLAlchemy at import time
from gymkhana.core.services.storage.storage import StorageService, StorageSession


class SQLStorageService(StorageService):
    """Base SQL implementation using AsyncDatabase."""

    host: str = "localhost"
    port: int = 5432
    user: str = "postgres"
    password: str = ""
    db_name: str = "gymkhana"
    pool_min: int = 2
    pool_max: int = 10

    schema_path: Optional[str] = None

    # Auto-provisioning settings
    auto_provision: bool = False
    docker_compose_path: Optional[str] = None

    _db: Optional[Any] = None
    _is_initialized: bool = False

    def model_post_init(self, __context: Any) -> None:
        if STORAGE_AVAILABLE and self._db is None:
            # Create a mock config object compatible with AsyncDatabase
            class DbConfig:
                def __init__(self, **kwargs):
                    for k, v in kwargs.items():
                        setattr(self, k, v)
                    # Use provided values or defaults for AsyncDatabase expectations
                    self.retry_delay = 1.0
                    self.retry_max_delay = 10.0
                    self.retry_backoff_factor = 2.0
                    self.retry_jitter = 0.1

            config = DbConfig(
                host=self.host,
                port=self.port,
                user=self.user,
                password=self.password,
                db_name=self.db_name,
                pool_min=self.pool_min,
                pool_max=self.pool_max
            )
            self._db = AsyncDatabase(config)

        if not self.docker_compose_path:
            self.docker_compose_path = os.path.join(
                os.path.dirname(__file__), "docker", "docker-compose.yaml"
            )

    async def initialize(self) -> None:
        """Initialize connection pool and apply schema.

        If auto_provision is True, will attempt to start the database
        via Docker if connection fails initially.
        """
        if self._is_initialized:
            return

        if self._db is None:
            if not STORAGE_AVAILABLE:
                raise RuntimeError("Database storage is unavailable: asyncpg not installed.")
            raise RuntimeError("Database instance (_db) is None even though storage is available.")

        try:
            await self._db.initialize()
        except Exception as e:
            if self.auto_provision:
                logging.getLogger("gymkhana.storage").info(
                    "Connection failed, attempting to auto-provision database via Docker..."
                )
                if self._provision_via_docker():
                    # Wait a bit for DB to be truly ready (healthchecks help but extra margin is good)
                    time.sleep(2)
                    await self._db.initialize()
                else:
                    raise RuntimeError(f"Auto-provisioning failed: {e}") from e
            else:
                raise

        # Apply schema only if schema_path is set and file exists
        if self.schema_path and os.path.exists(self.schema_path):
            with open(self.schema_path, "r") as f:
                schema_sql = f.read()

            try:
                async with self._db.safe_transaction() as conn:
                    await conn.execute(schema_sql)
            except Exception as e:
                # Log but don't fail if schema already exists
                logging.getLogger("gymkhana.storage").debug(
                    f"Schema application skipped (may already exist): {e}"
                )

        self._is_initialized = True

    def _provision_via_docker(self) -> bool:
        """Attempt to run docker compose up -d for the database."""
        if not os.path.exists(self.docker_compose_path):
            logging.getLogger("gymkhana.storage").error(
                f"Docker compose file not found at {self.docker_compose_path}"
            )
            return False

        try:
            # Try 'docker compose' (V2) then 'docker-compose' (V1)
            commands = [
                ["docker", "compose", "-f", self.docker_compose_path, "up", "-d"],
                ["docker-compose", "-f", self.docker_compose_path, "up", "-d"]
            ]

            success = False
            for cmd in commands:
                try:
                    subprocess.run(
                        cmd, check=True, capture_output=True, env=os.environ.copy()
                    )
                    success = True
                    break
                except (subprocess.CalledProcessError, FileNotFoundError):
                    continue

            if success:
                logging.getLogger("gymkhana.storage").info("Database provisioned successfully.")
                return True
            else:
                logging.getLogger("gymkhana.storage").error("Docker not found or failed to start containers.")
                return False

        except Exception as e:
            logging.getLogger("gymkhana.storage").error(f"Error during provisioning: {e}")
            return False

    async def close(self) -> None:
        if self._db:
            await self._db.close()
        self._is_initialized = False

    def create_session(self, trajectory_id: Optional[UUID] = None) -> "SQLEnvStorageSession":
        return SQLEnvStorageSession(service=self, trajectory_id=trajectory_id)


class SQLEnvStorageSession(StorageSession):
    """Concrete SQL session for environment data."""
    service: SQLStorageService

    async def store_turn(self, turn: Turn, trajectory_id: Optional[UUID] = None) -> UUID:
        # Lazy import to avoid triggering SQLAlchemy at module import time
        from gymkhana.core.models.db import TurnDB

        tid = trajectory_id or self.trajectory_id
        if not tid:
            raise ValueError("No trajectory ID provided")

        if self.service._db is None:
            raise RuntimeError("Database unavailable")

        db_turn = TurnDB(
            id=turn.id,
            trajectory_id=tid,
            turn_index=turn.turn_index,
            role=turn.role,
            content=turn.content,
            code=turn.code,
            reasoning_content=turn.reasoning_content
        )

        query = """
            INSERT INTO turns (id, trajectory_id, turn_index, role, content, code, reasoning_content)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
        """
        await self.service._db.execute(
            query,
            db_turn.id,
            db_turn.trajectory_id,
            db_turn.turn_index,
            db_turn.role,
            db_turn.content,
            db_turn.code,
            db_turn.reasoning_content
        )

        # Store associated execution if present
        if turn.execution:
            logging.getLogger("gymkhana.storage").info(f"Storing execution for turn {turn.id}")
            await self.store_execution(execution=turn.execution, turn_id=turn.id)
        else:
            logging.getLogger("gymkhana.storage").debug(f"No execution for turn {turn.id} (role={turn.role})")

        return turn.id

    async def store_execution(self, execution: ExecutionResult, turn_id: UUID) -> UUID:
        # Lazy import to avoid triggering SQLAlchemy at module import time
        from gymkhana.core.models.db import ExecutionResultDB

        if self.service._db is None:
            raise RuntimeError("Database unavailable")

        db_exec = ExecutionResultDB(
            id=execution.id,
            turn_id=turn_id,
            success=execution.success,
            output=execution.output,
            error=execution.error,
            truncated=execution.truncated,
            execution_time_ms=execution.execution_time_ms,
            files_created_json=json.dumps(execution.files_created),
            variables_json=json.dumps(execution.variables),
            state_json=json.dumps(execution.state),
            state_formatted=execution.state_formatted,
            done=execution.done,
            final_answer=execution.final_answer,
            iteration=execution.iteration,
            reward=execution.reward,
            episode_state_json=json.dumps(execution.episode_state)
        )

        query = """
            INSERT INTO executions (
                id, turn_id, success, output, error, truncated,
                execution_time_ms, files_created_json, variables_json,
                state_json, state_formatted, done, final_answer,
                iteration, reward, episode_state_json
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16)
        """
        await self.service._db.execute(
            query,
            db_exec.id, db_exec.turn_id, db_exec.success, db_exec.output,
            db_exec.error, db_exec.truncated, db_exec.execution_time_ms,
            db_exec.files_created_json, db_exec.variables_json,
            db_exec.state_json, db_exec.state_formatted,
            db_exec.done, db_exec.final_answer,
            db_exec.iteration, db_exec.reward,
            db_exec.episode_state_json
        )

        logging.getLogger("gymkhana.storage").info(f"Successfully stored execution {execution.id}")

        # Store sub-agent calls if present
        if execution.sub_agent_calls:
            for call in execution.sub_agent_calls:
                call_query = """
                    INSERT INTO sub_agent_calls (execution_id, task, system_prompt, response)
                    VALUES ($1, $2, $3, $4)
                """
                await self.service._db.execute(
                    call_query,
                    execution.id,
                    call.task,
                    call.system_prompt,
                    call.response
                )

        return execution.id


class EnvStorageService(SQLStorageService):
    """Specific implementation for Gymkhana environment data."""

    _skip_schema: bool = False  # Internal flag to skip schema application

    def model_post_init(self, __context: Any) -> None:
        super().model_post_init(__context)
        # Set default schema path if not provided
        if self.schema_path is None and not self._skip_schema:
            self.schema_path = os.path.join(
                os.path.dirname(__file__),
                "schema.sql"
            )

    async def store_trajectory(
        self,
        trajectory: TrajectoryResult,
        task_id: str,
        environment: str,
        model_name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        rollout_id: Optional[UUID] = None,
        rollout_group_id: Optional[UUID] = None,
    ) -> UUID:
        """Store a trajectory with optional rollout tracking.

        Args:
            trajectory: TrajectoryResult instance
            task_id: Task ID from dataset
            environment: Environment name
            model_name: LLM model name (optional)
            metadata: Additional metadata (optional)
            rollout_id: Parent rollout ID (optional, for GRPO)
            rollout_group_id: Parent rollout group ID (optional, for GRPO)

        Returns:
            UUID of the inserted trajectory
        """
        # Lazy import to avoid triggering SQLAlchemy at module import time
        from gymkhana.core.models.db import TrajectoryResultDB

        db_traj = TrajectoryResultDB(
            id=trajectory.id,
            rollout_id=rollout_id or trajectory.rollout_id,
            rollout_group_id=rollout_group_id or trajectory.rollout_group_id,
            rollout_index=trajectory.rollout_index,
            task_id=task_id,
            environment=environment,
            interaction_mode=trajectory.interaction_mode,  # NEW
            conversation_manager=trajectory.conversation_manager,  # NEW
            max_turns=trajectory.max_turns,  # NEW
            success=trajectory.success,
            final_answer=trajectory.final_answer,
            expected_answer=metadata.get("expected_answer") if metadata else None,
            answer_correct=trajectory.answer_correct,
            num_code_blocks=trajectory.num_code_blocks,
            num_turns=trajectory.num_turns,  # NEW
            num_errors=trajectory.num_errors,
            total_reward=trajectory.total_reward,
            step_rewards_json=json.dumps(trajectory.step_rewards),
            reward_function=trajectory.reward_function,
            efficiency_score=trajectory.efficiency_score,
            quality_score=trajectory.quality_score,
            system_prompt=trajectory.system_prompt,
            model_name=model_name or trajectory.model_name,
            metadata_json=json.dumps(metadata or trajectory.metadata)
        )

        query = """
            INSERT INTO trajectories (
                id, rollout_id, rollout_group_id, rollout_index,
                task_id, environment, interaction_mode, conversation_manager, max_turns,
                success, final_answer, expected_answer, answer_correct,
                num_code_blocks, num_turns, num_errors, total_reward, step_rewards_json,
                reward_function, efficiency_score, quality_score,
                system_prompt, model_name, metadata_json
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24)
        """
        await self._db.execute(
            query,
            db_traj.id, db_traj.rollout_id, db_traj.rollout_group_id, db_traj.rollout_index,
            db_traj.task_id, db_traj.environment, db_traj.interaction_mode, db_traj.conversation_manager, db_traj.max_turns,
            db_traj.success, db_traj.final_answer, db_traj.expected_answer, db_traj.answer_correct,
            db_traj.num_code_blocks, db_traj.num_turns, db_traj.num_errors, db_traj.total_reward,
            db_traj.step_rewards_json, db_traj.reward_function,
            db_traj.efficiency_score, db_traj.quality_score,
            db_traj.system_prompt, db_traj.model_name, db_traj.metadata_json
        )

        # Optionally create a session and store turns
        session = self.create_session(trajectory.id)
        for turn in trajectory.turns:
            await session.store_turn(turn)

        logging.getLogger("gymkhana.storage").info(
            f"Stored trajectory {trajectory.id} for task {task_id} "
            f"(mode={trajectory.interaction_mode}, manager={trajectory.conversation_manager})"
        )

        return trajectory.id

    async def get_trajectory(self, trajectory_id: UUID) -> Optional[TrajectoryResult]:
        if self._db is None:
            return None

        row = await self._db.fetch_one("SELECT * FROM trajectories WHERE id = $1", trajectory_id)
        if not row:
            return None

        turn_rows = await self._db.fetch(
            "SELECT * FROM turns WHERE trajectory_id = $1 ORDER BY turn_index",
            trajectory_id
        )

        turns = [
            Turn(
                id=r["id"],
                role=r["role"],
                content=r["content"],
                code=r["code"],
                turn_index=r["turn_index"]
            )
            for r in turn_rows
        ]

        return TrajectoryResult(
            id=row["id"],
            task_id=row["task_id"],
            environment=row["environment"],
            success=row["success"],
            final_answer=row["final_answer"],
            turns=turns,
            num_code_blocks=row["num_code_blocks"],
            num_errors=row["num_errors"],
            total_reward=row["total_reward"],
            step_rewards=json.loads(row["step_rewards_json"] or "[]"),
            system_prompt=row["system_prompt"],
            model_name=row["model_name"],
            metadata=json.loads(row["metadata_json"] or "{}"),
            # Rollout tracking fields
            rollout_id=row.get("rollout_id"),
            rollout_group_id=row.get("rollout_group_id"),
            rollout_index=row.get("rollout_index"),
            answer_correct=row.get("answer_correct"),
            reward_function=row.get("reward_function"),
            efficiency_score=row.get("efficiency_score"),
            quality_score=row.get("quality_score"),
        )

    async def insert_trajectory(
        self,
        result: TrajectoryResult,
        task_id: str,
        env_name: str,
        metadata: Optional[Dict[str, Any]] = None,
        rollout_id: Optional[UUID] = None,
        rollout_group_id: Optional[UUID] = None,
    ) -> UUID:
        """Legacy wrapper for store_trajectory with rollout support."""
        return await self.store_trajectory(
            trajectory=result,
            task_id=task_id,
            environment=env_name,
            metadata=metadata,
            rollout_id=rollout_id,
            rollout_group_id=rollout_group_id
        )

    async def insert_turn(
        self,
        trajectory_id: UUID,
        round_num: int,
        turn: Turn,
        reward: float = 0.0
    ) -> UUID:
        """Legacy helper to insert a turn."""
        session = self.create_session(trajectory_id)
        # turn_index is already in Turn model?
        # Environment.generate passes turn_index manually in legacy?
        # Actually it passes round_num.
        turn.turn_index = round_num - 1
        return await session.store_turn(turn)

    async def insert_request(self, req_data: Dict[str, Any]) -> None:
        """Log an LLM request."""
        if self._db is None:
            return

        query = """
            INSERT INTO requests (
                prompt_context_id, start_time, end_time, total_time,
                model, max_tokens, temperature, messages, system,
                raw_response, completion_tokens, prompt_tokens, total_tokens,
                reasoning_tokens, reasoning_content, raw_request
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16)
        """

        # Convert messages and raw_response to JSON (not string)
        messages = req_data.get("messages", [])
        raw_response = req_data.get("raw_response", {})
        raw_request = req_data.get("raw_request", {})

        # If they're already dicts, use them directly; if strings, parse them
        if isinstance(messages, str):
            messages = json.loads(messages)
        if isinstance(raw_response, str):
            try:
                raw_response = json.loads(raw_response)
            except json.JSONDecodeError:
                raw_response = {"error": "Failed to parse raw_response"}
        if isinstance(raw_request, str):
            try:
                raw_request = json.loads(raw_request)
            except json.JSONDecodeError:
                raw_request = {"error": "Failed to parse raw_request"}

        await self._db.execute(
            query,
            req_data.get("prompt_context_id"),
            req_data.get("start_time"),
            req_data.get("end_time"),
            req_data.get("total_time"),
            req_data.get("model"),
            req_data.get("max_tokens"),
            req_data.get("temperature"),
            json.dumps(messages),  # Still need to dump for asyncpg
            req_data.get("system"),
            json.dumps(raw_response),  # Still need to dump for asyncpg
            req_data.get("completion_tokens"),
            req_data.get("prompt_tokens"),
            req_data.get("total_tokens"),
            req_data.get("reasoning_tokens"),
            req_data.get("reasoning_content"),
            json.dumps(raw_request),  # Full API request payload
        )

    async def insert_environment_state(
        self,
        trajectory_id: UUID,
        turn_index: int,
        state: Dict[str, Any],
        state_formatted: str
    ) -> None:
        """Legacy state logging (mapped to executions or similar if needed)."""
        # For now, we don't have a direct table for states only,
        # they are usually part of executions.
        pass

    async def insert_sharegpt_dataset(
        self,
        task_id: str,
        conversations: List[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Insert ShareGPT formatted dataset into sharegpt_datasets."""
        if self._db is None:
            return False

        query = """
            INSERT INTO sharegpt_datasets (task_id, conversations, metadata)
            VALUES ($1, $2, $3)
        """
        try:
            await self._db.execute(
                query,
                task_id,
                json.dumps(conversations),
                json.dumps(metadata or {})
            )
            return True
        except Exception as e:
            logging.getLogger("gymkhana.storage").error(f"Error inserting sharegpt dataset: {e}")
            return False

    async def insert_sandbox_session(
        self,
        session_state: SessionState,
        environment: Optional[str] = None,
        trajectory_id: Optional[UUID] = None,
        rollout_id: Optional[UUID] = None,
        rollout_group_id: Optional[UUID] = None
    ) -> UUID:
        """Insert or update a sandbox session record."""
        # Lazy import to avoid triggering SQLAlchemy at module import time
        from gymkhana.core.models.db import SandboxSessionDB

        if self._db is None:
            raise RuntimeError("Database unavailable")

        db_session = SandboxSessionDB(
            session_id=str(session_state.session_id),
            status=session_state.status.value,
            created_at=session_state.created_at,
            ready_at=session_state.ready_at,
            last_execution_at=session_state.last_execution_at,
            completed_at=session_state.completed_at,
            total_reward=session_state.metrics.total_reward,
            total_executions=session_state.metrics.total_executions,
            successful_executions=session_state.metrics.successful_executions,
            failed_executions=session_state.metrics.failed_executions,
            total_execution_time_ms=session_state.metrics.total_execution_time_ms,
            environment=environment,
            trajectory_id=trajectory_id
        )

        # Set JSON fields via property setters
        db_session.interpreter = session_state.interpreter.model_dump()
        db_session.episode = session_state.episode.model_dump()
        # config is not directly in SessionState but we can pass it if needed,
        # for now use empty dict or extracted if available
        db_session.config = {}

        query = """
            INSERT INTO sandbox_sessions (
                session_id, status, created_at, ready_at,
                last_execution_at, completed_at, total_reward,
                total_executions, successful_executions, failed_executions,
                total_execution_time_ms, interpreter_json, episode_json,
                config_json, environment, trajectory_id, rollout_id, rollout_group_id
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18)
            ON CONFLICT (id) DO UPDATE SET
                status = EXCLUDED.status,
                ready_at = EXCLUDED.ready_at,
                last_execution_at = EXCLUDED.last_execution_at,
                completed_at = EXCLUDED.completed_at,
                total_reward = EXCLUDED.total_reward,
                total_executions = EXCLUDED.total_executions,
                successful_executions = EXCLUDED.successful_executions,
                failed_executions = EXCLUDED.failed_executions,
                total_execution_time_ms = EXCLUDED.total_execution_time_ms,
                interpreter_json = EXCLUDED.interpreter_json,
                episode_json = EXCLUDED.episode_json,
                rollout_id = EXCLUDED.rollout_id,
                rollout_group_id = EXCLUDED.rollout_group_id,
                updated_at = CURRENT_TIMESTAMP
            RETURNING id
        """

        result = await self._db.fetch_one(
            query,
            db_session.session_id, db_session.status, db_session.created_at,
            db_session.ready_at, db_session.last_execution_at,
            db_session.completed_at, db_session.total_reward,
            db_session.total_executions, db_session.successful_executions,
            db_session.failed_executions, db_session.total_execution_time_ms,
            db_session.interpreter_json, db_session.episode_json,
            db_session.config_json, db_session.environment,
            db_session.trajectory_id, rollout_id, rollout_group_id
        )
        return result["id"]

    # =============================================================================
    # Rollout Tracking Methods (NEW for GRPO)
    # =============================================================================

    async def insert_rollout_group(self, group: RolloutGroup) -> UUID:
        """Insert a rollout group for tracking parallel rollouts.

        Args:
            group: RolloutGroup instance with task_id, environment, num_rollouts

        Returns:
            UUID of the inserted rollout group
        """
        # Lazy import to avoid triggering SQLAlchemy at module import time
        from gymkhana.core.models.db import RolloutGroupDB

        if self._db is None:
            raise RuntimeError("Database unavailable")

        db_group = RolloutGroupDB(
            id=group.id,
            task_id=group.task_id,
            environment=group.environment,
            num_rollouts=group.num_rollouts,
            num_completed=group.num_completed,
            num_failed=group.num_failed,
            num_error=group.num_error,
            num_timeout=group.num_timeout,
            best_rollout_id=group.best_rollout_id,
            best_reward=group.best_reward,
            reward_mean=group.reward_mean,
            reward_std=group.reward_std,
            reward_min=group.reward_min,
            completed_at=group.completed_at
        )

        # Set config via property
        db_group.config = group.config

        query = """
            INSERT INTO rollout_groups (
                id, task_id, environment, num_rollouts, num_completed,
                num_failed, num_error, num_timeout, best_rollout_id,
                best_reward, reward_mean, reward_std, reward_min,
                config_json, completed_at
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15)
        """

        await self._db.execute(
            query,
            db_group.id, db_group.task_id, db_group.environment,
            db_group.num_rollouts, db_group.num_completed,
            db_group.num_failed, db_group.num_error, db_group.num_timeout,
            db_group.best_rollout_id, db_group.best_reward,
            db_group.reward_mean, db_group.reward_std, db_group.reward_min,
            db_group.config_json, db_group.completed_at
        )

        logging.getLogger("gymkhana.storage").info(
            f"Inserted rollout group {group.id} for task {group.task_id}"
        )

        return group.id

    async def insert_rollout(self, rollout: RolloutState, group_id: UUID) -> UUID:
        """Insert a rollout state for tracking individual rollout execution.

        Args:
            rollout: RolloutState instance with execution metrics
            group_id: UUID of the parent rollout group

        Returns:
            UUID of the inserted rollout
        """
        # Lazy import to avoid triggering SQLAlchemy at module import time
        from gymkhana.core.models.db import RolloutStateDB

        if self._db is None:
            raise RuntimeError("Database unavailable")

        db_rollout = RolloutStateDB(
            id=rollout.id,
            rollout_group_id=group_id,
            trajectory_id=None,  # Set later when trajectory is created
            rollout_index=rollout.rollout_id,  # rollout_id is the index
            status=rollout.status.value if isinstance(rollout.status, RolloutStatus) else rollout.status,
            termination_reason=rollout.termination_reason,
            num_turns=rollout.num_turns,
            num_code_blocks=rollout.num_code_blocks,
            num_errors=rollout.num_errors,
            consecutive_errors=rollout.consecutive_errors,
            num_format_violations=rollout.num_format_violations,
            consecutive_format_violations=rollout.consecutive_format_violations,
            last_format_violation_type=rollout.last_format_violation_type,
            total_reward=rollout.total_reward,
            started_at=rollout.started_at,
            completed_at=rollout.completed_at,
            duration_ms=rollout.duration_ms(),
            session_id=rollout.session_id
        )

        # Set format violation history via property
        db_rollout.format_violation_history = rollout.format_violation_history

        query = """
            INSERT INTO rollouts (
                id, rollout_group_id, trajectory_id, rollout_index,
                status, termination_reason, num_turns, num_code_blocks,
                num_errors, consecutive_errors, num_format_violations,
                consecutive_format_violations, last_format_violation_type,
                format_violation_history_json, total_reward, started_at,
                completed_at, duration_ms, session_id
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19)
        """

        await self._db.execute(
            query,
            db_rollout.id, db_rollout.rollout_group_id, db_rollout.trajectory_id,
            db_rollout.rollout_index, db_rollout.status, db_rollout.termination_reason,
            db_rollout.num_turns, db_rollout.num_code_blocks, db_rollout.num_errors,
            db_rollout.consecutive_errors, db_rollout.num_format_violations,
            db_rollout.consecutive_format_violations, db_rollout.last_format_violation_type,
            db_rollout.format_violation_history_json, db_rollout.total_reward,
            db_rollout.started_at, db_rollout.completed_at, db_rollout.duration_ms,
            db_rollout.session_id
        )

        logging.getLogger("gymkhana.storage").info(
            f"Inserted rollout {rollout.id} (index {rollout.rollout_id}) for group {group_id}"
        )

        return rollout.id

    async def update_rollout(self, rollout: RolloutState) -> None:
        """Update a rollout state with current execution metrics.

        Args:
            rollout: RolloutState instance with updated metrics
        """
        if self._db is None:
            raise RuntimeError("Database unavailable")

        query = """
            UPDATE rollouts SET
                status = $2,
                termination_reason = $3,
                num_turns = $4,
                num_code_blocks = $5,
                num_errors = $6,
                consecutive_errors = $7,
                num_format_violations = $8,
                consecutive_format_violations = $9,
                last_format_violation_type = $10,
                format_violation_history_json = $11,
                total_reward = $12,
                completed_at = $13,
                duration_ms = $14,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = $1
        """

        await self._db.execute(
            query,
            rollout.id,
            rollout.status.value if isinstance(rollout.status, RolloutStatus) else rollout.status,
            rollout.termination_reason,
            rollout.num_turns,
            rollout.num_code_blocks,
            rollout.num_errors,
            rollout.consecutive_errors,
            rollout.num_format_violations,
            rollout.consecutive_format_violations,
            rollout.last_format_violation_type,
            json.dumps(rollout.format_violation_history),
            rollout.total_reward,
            rollout.completed_at,
            rollout.duration_ms()
        )

        logging.getLogger("gymkhana.storage").debug(
            f"Updated rollout {rollout.id} (status: {rollout.status})"
        )

    async def link_rollout_to_trajectory(self, rollout_id: UUID, trajectory_id: UUID) -> None:
        """Link a rollout to its trajectory after trajectory is created.

        Args:
            rollout_id: UUID of the rollout
            trajectory_id: UUID of the trajectory
        """
        if self._db is None:
            raise RuntimeError("Database unavailable")

        query = """
            UPDATE rollouts SET
                trajectory_id = $2,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = $1
        """

        await self._db.execute(query, rollout_id, trajectory_id)

        logging.getLogger("gymkhana.storage").debug(
            f"Linked rollout {rollout_id} to trajectory {trajectory_id}"
        )

    async def update_rollout_group_statistics(
        self,
        group_id: UUID,
        rollout_rewards: List[float],
        status_counts: Dict[str, int]
    ) -> None:
        """Update rollout group with aggregate statistics.

        Args:
            group_id: UUID of the rollout group
            rollout_rewards: List of total_reward values from all rollouts
            status_counts: Dict mapping status to count (e.g., {'completed': 5, 'failed': 3})
        """
        if self._db is None:
            raise RuntimeError("Database unavailable")

        # Calculate statistics
        import statistics
        reward_mean = statistics.mean(rollout_rewards) if rollout_rewards else 0.0
        reward_std = statistics.stdev(rollout_rewards) if len(rollout_rewards) > 1 else 0.0
        reward_min = min(rollout_rewards) if rollout_rewards else 0.0
        best_reward = max(rollout_rewards) if rollout_rewards else 0.0

        # Get best rollout ID
        best_rollout_query = """
            SELECT id FROM rollouts
            WHERE rollout_group_id = $1
            ORDER BY total_reward DESC
            LIMIT 1
        """
        best_rollout_row = await self._db.fetch_one(best_rollout_query, group_id)
        best_rollout_id = best_rollout_row["id"] if best_rollout_row else None

        query = """
            UPDATE rollout_groups SET
                num_completed = $2,
                num_failed = $3,
                num_error = $4,
                num_timeout = $5,
                best_rollout_id = $6,
                best_reward = $7,
                reward_mean = $8,
                reward_std = $9,
                reward_min = $10,
                completed_at = CURRENT_TIMESTAMP,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = $1
        """

        await self._db.execute(
            query,
            group_id,
            status_counts.get('completed', 0),
            status_counts.get('failed', 0),
            status_counts.get('error', 0),
            status_counts.get('timeout', 0),
            best_rollout_id,
            best_reward,
            reward_mean,
            reward_std,
            reward_min
        )

        logging.getLogger("gymkhana.storage").info(
            f"Updated rollout group {group_id} statistics: "
            f"mean={reward_mean:.3f}, std={reward_std:.3f}, best={best_reward:.3f}"
        )

    async def get_rollout_group(self, group_id: UUID) -> Optional[RolloutGroup]:
        """Retrieve a rollout group by ID.

        Args:
            group_id: UUID of the rollout group

        Returns:
            RolloutGroup instance or None if not found
        """
        if self._db is None:
            return None

        row = await self._db.fetch_one(
            "SELECT * FROM rollout_groups WHERE id = $1",
            group_id
        )

        if not row:
            return None

        return RolloutGroup(
            id=row["id"],
            task_id=row["task_id"],
            environment=row["environment"],
            num_rollouts=row["num_rollouts"],
            num_completed=row.get("num_completed", 0),
            num_failed=row.get("num_failed", 0),
            num_error=row.get("num_error", 0),
            num_timeout=row.get("num_timeout", 0),
            best_rollout_id=row.get("best_rollout_id"),
            best_reward=row.get("best_reward", 0.0),
            reward_mean=row.get("reward_mean", 0.0),
            reward_std=row.get("reward_std", 0.0),
            reward_min=row.get("reward_min", 0.0),
            config=json.loads(row.get("config_json") or "{}"),
            completed_at=row.get("completed_at"),
            created_at=row.get("created_at", datetime.now(timezone.utc))
        )

    async def get_rollouts_by_group(self, group_id: UUID) -> List[RolloutState]:
        """Retrieve all rollouts for a group.

        Args:
            group_id: UUID of the rollout group

        Returns:
            List of RolloutState instances
        """
        if self._db is None:
            return []

        rows = await self._db.fetch(
            "SELECT * FROM rollouts WHERE rollout_group_id = $1 ORDER BY rollout_index",
            group_id
        )

        rollouts = []
        for row in rows:
            rollout = RolloutState(
                id=row["id"],
                rollout_id=row["rollout_index"],
                status=RolloutStatus(row["status"]),
                session_id=row.get("session_id"),
                num_turns=row.get("num_turns", 0),
                num_code_blocks=row.get("num_code_blocks", 0),
                num_errors=row.get("num_errors", 0),
                consecutive_errors=row.get("consecutive_errors", 0),
                num_format_violations=row.get("num_format_violations", 0),
                consecutive_format_violations=row.get("consecutive_format_violations", 0),
                last_format_violation_type=row.get("last_format_violation_type"),
                format_violation_history=json.loads(row.get("format_violation_history_json") or "[]"),
                total_reward=row.get("total_reward", 0.0),
                termination_reason=row.get("termination_reason"),
                started_at=row.get("started_at", datetime.now(timezone.utc)),
                completed_at=row.get("completed_at"),
                created_at=row.get("created_at", datetime.now(timezone.utc))
            )
            rollouts.append(rollout)

        return rollouts


__all__ = [
    "SQLStorageService",
    "EnvStorageService",
    "SQLEnvStorageSession",
    "STORAGE_AVAILABLE"
]
