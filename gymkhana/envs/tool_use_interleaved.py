"""Compatibility exports for interleaved Pydantic AI tool use."""

from gymkhana.envs.modes.tool_use_interleaved import ToolUseInterleavedMode

InterleavedToolUsePipeline = ToolUseInterleavedMode

__all__ = ["InterleavedToolUsePipeline", "ToolUseInterleavedMode"]
