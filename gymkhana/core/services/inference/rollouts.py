"""GRPO-style grouped rollout generation."""

from __future__ import annotations

from typing import List, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from .base import InferenceService


class RolloutRequest(BaseModel):
    task_id: str
    prompt: str
    group_size: int = Field(default=4, ge=1)
    system_prompt: Optional[str] = None
    model: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: int = Field(default=4096, ge=1)
    seed: Optional[int] = None


class RolloutCandidate(BaseModel):
    group_id: UUID
    task_id: str
    index: int
    output: str


class RolloutGroupResult(BaseModel):
    group_id: UUID = Field(default_factory=uuid4)
    task_id: str
    candidates: List[RolloutCandidate] = Field(default_factory=list)


async def generate_rollout_group(
    service: InferenceService, request: RolloutRequest
) -> RolloutGroupResult:
    """Generate an order-preserving group of independent candidates."""
    group = RolloutGroupResult(task_id=request.task_id)
    outputs = await service.batch_generate_identical(
        prompt=request.prompt,
        system_prompt=request.system_prompt,
        n=request.group_size,
        model=request.model,
        temperature=request.temperature,
        max_tokens=request.max_tokens,
        seed=request.seed,
    )
    group.candidates = [
        RolloutCandidate(
            group_id=group.group_id,
            task_id=request.task_id,
            index=index,
            output=output,
        )
        for index, output in enumerate(outputs)
    ]
    return group
