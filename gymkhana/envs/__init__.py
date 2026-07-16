"""Environment package for Gymkhana."""

from .environment import (
    ENVIRONMENTS,
    Environment,
    EnvironmentError,
    EnvironmentRegistry,
    EnvironmentRunSummary,
    Task,
    TrajectoryState,
    get_environment,
    register_environment,
)

# Import built-in environments so they register with the global registry
from .math_python import MathPythonEnv  # noqa: F401
from .oolong import OolongEnv  # noqa: F401
from .hotpotqa import HotpotQAEnv  # noqa: F401
from .swe import SWEEnv  # noqa: F401
from .ifeval import IfEvalEnv  # noqa: F401
from .tool_use_singleturn.tool_use_singleturn import ToolUseSingleTurnEnv  # noqa: F401
from .romanized_nepali import RomanizedNepaliEnv  # noqa: F401

__all__ = [
    "ENVIRONMENTS",
    "Environment",
    "EnvironmentError",
    "EnvironmentRegistry",
    "EnvironmentRunSummary",
    "Task",
    "TrajectoryState",
    "get_environment",
    "register_environment",
    "MathPythonEnv",
    "OolongEnv",
    "HotpotQAEnv",
    "RomanizedNepaliEnv",
]
