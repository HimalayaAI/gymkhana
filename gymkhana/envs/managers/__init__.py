"""Conversation managers for Gymkhana environments.

This module provides different conversation patterns:
- SingleTurnManager: One turn, done
- MultiTurnManager: Generic multi-turn loop
- SequentialToolManager: Sequential tool chains
- ConversationalManager: Interactive with user interruptions
- SelfRefinementManager: Iterative improvement
"""

from gymkhana.envs.managers.base import ConversationManager
from gymkhana.envs.managers.single_turn import SingleTurnManager
from gymkhana.envs.managers.multi_turn import MultiTurnManager

__all__ = [
    "ConversationManager",
    "SingleTurnManager",
    "MultiTurnManager",
]
