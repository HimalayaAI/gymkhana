"""Interaction modes for Gymkhana environments.

This module provides different interaction patterns for agent-environment communication:
- ChatMode: Pure text generation without tools
- ToolUseMode: Regular tool calls (Think -> Tool Call -> Execute)
- ToolUseInterleavedMode: Interleaved tool calls with native execution
- RLMMode: Code execution with REPL sandbox (in rlm.py module)
"""

from gymkhana.envs.modes.base import InteractionMode
from gymkhana.envs.modes.chat import ChatMode
from gymkhana.envs.modes.tool_use import ToolUseMode
from gymkhana.envs.modes.tool_use_interleaved import ToolUseInterleavedMode

__all__ = [
    "InteractionMode",
    "ChatMode",
    "ToolUseMode",
    "ToolUseInterleavedMode",
]
