#!/usr/bin/env python3
"""Run a small paid Anthropic smoke test for the Romanized Nepali environment."""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv

from gymkhana.core.models import TrajectoryResult
from gymkhana.core.services.inference import (
    PydanticAIInferenceService,
    RolloutRequest,
    generate_rollout_group,
)
from gymkhana.envs.romanized_nepali import RomanizedNepaliEnv


async def run(model: str, group_size: int) -> int:
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise SystemExit(
            "ANTHROPIC_API_KEY is missing. Copy .env.example to .env and add the key locally."
        )

    environment = RomanizedNepaliEnv()
    task = environment.load_tasks(limit=1)[0]
    service = PydanticAIInferenceService(default_model=model, max_concurrency=group_size)
    group = await generate_rollout_group(
        service,
        RolloutRequest(
            task_id=task.id,
            prompt=environment.format_initial_message(task),
            system_prompt=environment.get_environment_instructions(task),
            model=model,
            group_size=group_size,
            max_tokens=128,
        ),
    )

    print(f"model={model} task={task.id} group={group.group_id}")
    print(f"reference={task.metadata['reference']!r}")
    for candidate in group.candidates:
        result = TrajectoryResult(success=True, final_answer=candidate.output)
        reward = await environment.compute_reward(result, task=task)
        print(f"candidate={candidate.index} reward={reward:.4f} output={candidate.output!r}")
    return 0


def main() -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default=os.getenv("ANTHROPIC_MODEL", "anthropic:claude-sonnet-4-5"),
    )
    parser.add_argument("--group-size", type=int, default=2)
    args = parser.parse_args()
    if args.group_size < 1:
        parser.error("--group-size must be at least 1")
    return asyncio.run(run(args.model, args.group_size))


if __name__ == "__main__":
    raise SystemExit(main())
