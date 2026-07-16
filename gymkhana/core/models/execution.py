"""Execution models for sandbox interactions.

These models capture the results of code execution and sub-agent calls.
All models use Pydantic for business logic.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import Field

from gymkhana.core.models.entity import Entity


class SubAgentCall(Entity):
    """Record of a sub-agent invocation."""
    task: str = Field(..., description="Task description for sub-agent")
    system_prompt: Optional[str] = Field(default=None, description="System prompt used")
    response: str = Field(default="", description="Sub-agent response")


class ExecutionResult(Entity):
    """Result from code execution.

    Contains output, errors, state snapshots, and metadata about the execution.
    """
    success: bool = Field(..., description="Whether execution succeeded")
    output: str = Field(default="", description="Standard output")
    error: Optional[str] = Field(default=None, description="Error message if any")
    truncated: bool = Field(default=False, description="Whether output was truncated")
    execution_time_ms: int = Field(default=0, description="Execution time in milliseconds")
    answer: Optional[Dict[str, Any]] = Field(default=None, description="Extracted answer")
    files_created: List[str] = Field(default_factory=list, description="Files created")
    variables: Dict[str, str] = Field(default_factory=dict, description="Variable snapshot")
    state: Dict[str, Any] = Field(
        default_factory=lambda: {
            "variables": {},
            "functions": {},
            "classes": {},
            "modules": [],
        },
        description="Complete interpreter state"
    )
    state_formatted: str = Field(default="(empty state)", description="Formatted state string")
    sub_agent_calls: List[SubAgentCall] = Field(
        default_factory=list, description="Sub-agent invocations"
    )

    # RLM-style fields for reinforcement learning / multi-turn agents
    done: bool = Field(default=False, description="Whether episode is complete")
    final_answer: Optional[str] = Field(default=None, description="Final answer if done")
    iteration: int = Field(default=0, description="Current iteration count")
    reward: float = Field(default=0.0, description="Reward for this step")
    episode_state: Dict[str, Any] = Field(
        default_factory=dict, description="Persistent episode state"
    )


__all__ = [
    "SubAgentCall",
    "ExecutionResult",
]
