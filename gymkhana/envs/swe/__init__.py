"""SWE (Software Engineering) utilities for Gymkhana.

This module contains utilities for working with SWE-bench datasets:
- Docker container management for isolated repository environments
- Repository metadata and image size information
"""

from .swe_env import (
    SWEEnv,
    SWETaskMetadata,
    SWETestResult,
    SWEPatch,
    DEFAULT_SWE_CONFIG,
)

__all__ = [
    "SWEEnv",
    "SWETaskMetadata",
    "SWETestResult",
    "SWEPatch",
    "DEFAULT_SWE_CONFIG",
]
