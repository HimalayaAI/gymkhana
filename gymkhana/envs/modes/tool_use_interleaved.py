"""Interleaved tool use backed by Pydantic AI's native agent loop."""

from __future__ import annotations

from typing import TYPE_CHECKING, List

from gymkhana.envs.modes.tool_use import ToolUseMode

if TYPE_CHECKING:
    from gymkhana.core.models import TrajectoryResult
    from gymkhana.envs.environment import Environment, Task


class ToolUseInterleavedMode(ToolUseMode):
    """Native provider tool calling with execution handled by Pydantic AI.

    Pydantic AI preserves provider-specific interleaved tool-call semantics while
    exposing the same provider-neutral result to Gymkhana, so this mode shares the
    rollout implementation with :class:`ToolUseMode`.
    """

    async def execute_batch(
        self, task: "Task", env: "Environment", num_rollouts: int
    ) -> List["TrajectoryResult"]:
        return await super().execute_batch(task, env, num_rollouts)


__all__ = ["ToolUseInterleavedMode"]
