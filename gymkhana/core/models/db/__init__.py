"""Database models for gymkhana.

This module should ONLY be imported by storage/migration code, never at
the package level. This prevents SQLAlchemy from being triggered at import time.

Usage:
    # In storage code only:
    from gymkhana.core.models.db import TrajectoryResultDB, TurnDB
"""

from gymkhana.core.models.db.trajectory_db import (
    TurnDB,
    TrajectoryResultDB,
    RolloutGroupDB,
    RolloutStateDB,
)
from gymkhana.core.models.db.execution_db import (
    ExecutionResultDB,
    SubAgentCallDB,
)
from gymkhana.core.models.db.sandbox_db import (
    SandboxSessionDB,
)

__all__ = [
    "TurnDB",
    "TrajectoryResultDB",
    "RolloutGroupDB",
    "RolloutStateDB",
    "ExecutionResultDB",
    "SubAgentCallDB",
    "SandboxSessionDB",
]
