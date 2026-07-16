"""Generic environment models for Gymkhana.

This module provides base models that can be extended by specific environments:
- TaskMetadata: Base metadata for all environment tasks
- TestResult: Generic test/evaluation result
- CodePatch: Represents a code edit/patch

These models extend Entity and can be subclassed for environment-specific needs.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import Field

from gymkhana.core.models.entity import Entity


class TaskMetadata(Entity):
    """Base metadata for environment tasks.

    Provides common fields that most environments need.
    Subclass for environment-specific metadata (e.g., SWETaskMetadata).
    """

    task_id: str = Field(
        ...,
        description="Unique identifier for the task"
    )

    environment: str = Field(
        default="",
        description="Environment type (e.g., 'math-python', 'swe', 'hotpotqa')"
    )

    source_dataset: Optional[str] = Field(
        default=None,
        description="Name of the source dataset"
    )

    extra: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional task-specific metadata"
    )


class TestResult(Entity):
    """Generic test/evaluation result.

    Represents the outcome of running tests or evaluation.
    Subclass for specific result types (e.g., SWETestResult).
    """

    task_id: str = Field(
        ...,
        description="ID of the task this result belongs to"
    )

    passed: bool = Field(
        default=False,
        description="Whether the test/evaluation passed"
    )

    score: Optional[float] = Field(
        default=None,
        description="Numeric score (if applicable)"
    )

    message: str = Field(
        default="",
        description="Human-readable result message"
    )

    logs: str = Field(
        default="",
        description="Raw logs from test execution"
    )

    execution_time_ms: int = Field(
        default=0,
        description="Execution time in milliseconds"
    )

    extra: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional result data"
    )


class CodePatch(Entity):
    """Represents a code edit/patch.

    Generic representation of code changes made by an agent.
    Subclass for specific patch formats (e.g., SWEPatch).
    """

    task_id: str = Field(
        ...,
        description="ID of the task this patch belongs to"
    )

    file_path: str = Field(
        default="",
        description="Path to the file being modified"
    )

    original_content: str = Field(
        default="",
        description="Original content before patch"
    )

    patched_content: str = Field(
        default="",
        description="Content after patch is applied"
    )

    diff: str = Field(
        default="",
        description="Unified diff representation"
    )

    line_start: Optional[int] = Field(
        default=None,
        description="Starting line number of the change"
    )

    line_end: Optional[int] = Field(
        default=None,
        description="Ending line number of the change"
    )


__all__ = ["TaskMetadata", "TestResult", "CodePatch"]
