"""Base abstractions for reward functions using Pydantic models.

Defines the interface for trajectory evaluation.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, TYPE_CHECKING, Type

from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    from gymkhana.core.models.trajectory import TrajectoryMetrics


# Global registry for reward functions
_REWARD_REGISTRY: Dict[str, Type["RewardFunction"]] = {}


def register_reward_function(name: str):
    """Decorator to register a reward function.

    Usage:
        @register_reward_function("my_reward")
        class MyReward(RewardFunction):
            ...
    """
    def decorator(cls: Type[RewardFunction]) -> Type[RewardFunction]:
        _REWARD_REGISTRY[name] = cls
        return cls
    return decorator


def get_reward_function(name: str, **kwargs) -> "RewardFunction":
    """Get a reward function by name."""
    if name not in _REWARD_REGISTRY:
        raise ValueError(
            f"Unknown reward function: {name}. "
            f"Available: {list(_REWARD_REGISTRY.keys())}"
        )
    return _REWARD_REGISTRY[name](**kwargs)


def list_reward_functions() -> List[str]:
    """List all registered reward functions."""
    return list(_REWARD_REGISTRY.keys())


class RewardFunction(BaseModel, ABC):
    """Abstract base class for reward functions.

    Allows for configurable reward logic that can be serialized and
    passed between services.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @abstractmethod
    def compute(self, metrics: "TrajectoryMetrics") -> Dict[str, Any]:
        """
        Compute reward based on trajectory metrics.

        Args:
            metrics: Trajectory metrics (from core.models)

        Returns:
            Dict with:
                - total_reward: float
                - final_step_reward: float
                - metadata: dict
        """
        ...

    @property
    def name(self) -> str:
        """Name of the reward function."""
        return self.__class__.__name__
