"""Compatibility exports for Gymkhana's Pydantic AI tool-use mode."""

from gymkhana.envs.modes.tool_use import ToolUseMode

# DeepGym exposed the implementation as ``ToolUsePipeline``. Keep the import
# path while routing all execution through the provider-neutral mode.
ToolUsePipeline = ToolUseMode

__all__ = ["ToolUseMode", "ToolUsePipeline"]
