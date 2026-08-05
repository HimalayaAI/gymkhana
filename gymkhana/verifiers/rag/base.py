"""Shared infrastructure for external RAG verifiers."""

from __future__ import annotations

import json
import logging
from typing import Any, TypeVar

from pydantic import BaseModel

from gymkhana.core.services.inference import InferenceService
from gymkhana.envs.config import LLMJudgeSettings
from gymkhana.verifiers.rag.models import RAGMetricResult

logger = logging.getLogger(__name__)

JudgmentT = TypeVar("JudgmentT", bound=BaseModel)


class BaseRAGVerifier:
    """Call a fixed external judge and normalize verifier failure behavior."""

    metric_name = "rag_metric"

    def __init__(
        self,
        *,
        settings: LLMJudgeSettings,
        inference_service: InferenceService,
        threshold: float,
    ) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0 and 1")
        self.settings = settings
        self.inference_service = inference_service
        self.threshold = threshold

    async def _generate_structured(
        self,
        *,
        payload: dict[str, Any],
        output_type: type[JudgmentT],
        system_prompt: str,
    ) -> JudgmentT:
        return await self.inference_service.generate_structured(
            messages=[
                {
                    "role": "user",
                    "content": json.dumps(payload, ensure_ascii=False),
                }
            ],
            output_type=output_type,
            system_prompt=system_prompt,
            model=self.settings.model,
            temperature=self.settings.temperature,
            max_tokens=self.settings.max_tokens,
        )

    def _result(
        self,
        score: float,
        *,
        details: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> RAGMetricResult:
        normalized = min(max(float(score), 0.0), 1.0)
        return RAGMetricResult(
            metric=self.metric_name,
            score=normalized,
            passed=error is None and normalized >= self.threshold,
            threshold=self.threshold,
            details=details or {},
            error=error,
        )

    def _error(self, error: Exception | str) -> RAGMetricResult:
        message = error if isinstance(error, str) else f"{type(error).__name__}: {error}"
        logger.warning("%s verifier failed: %s", self.metric_name, message)
        return self._result(0.0, error=message)


__all__ = ["BaseRAGVerifier"]
