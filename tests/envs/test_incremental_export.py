"""Incremental export: rows hit disk as tasks finish, and runs resume."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from gymkhana.core.models import TrajectoryResult, Turn
from gymkhana.envs.config import DatasetSettings, EnvConfig, InferenceConfig
from gymkhana.envs.environment import Environment, Task


class SimulatedInterrupt(BaseException):
    """Stands in for Ctrl-C: a BaseException, so `except Exception` does not swallow it."""


class RecordingEnv(Environment):
    """Minimal environment whose tasks are scripted, with a hook per task."""

    name: str = "incremental-test"

    def __init__(self, *, config: EnvConfig, task_ids: List[str], **data: Any) -> None:
        data["config"] = config
        super().__init__(**data)
        self._task_ids = task_ids
        self._seen: List[str] = []
        self._fail_on: Optional[str] = None
        self._observer = None

    def load_tasks(self, limit: Optional[int] = None) -> List[Task]:
        return [Task(id=tid, prompt=f"prompt {tid}", metadata={}) for tid in self._task_ids]

    async def run_task(self, task: Task) -> TrajectoryResult:
        self._seen.append(task.id)
        if self._observer is not None:
            self._observer(task.id)
        if self._fail_on == task.id:
            raise SimulatedInterrupt("simulated Ctrl-C")
        return TrajectoryResult(
            success=True,
            final_answer=f"answer {task.id}",
            turns=[
                Turn(role="user", content=task.prompt, turn_index=0),
                Turn(role="assistant", content=f"answer {task.id}", turn_index=1),
            ],
            answer_correct=True,
            task_id=task.id,
            environment=self.name,
            total_reward=1.0,
        )

    def build_sharegpt_conversations(
        self, result: TrajectoryResult, task: Task
    ) -> Optional[List[Dict[str, Any]]]:
        return [
            {"from": "human", "value": task.prompt},
            {"from": "gpt", "value": result.final_answer},
        ]

    def build_sharegpt_metadata(self, result: TrajectoryResult, task: Task) -> Dict[str, Any]:
        return {"reward": result.total_reward}


def make_env(tmp_path: Path, task_ids: List[str], **dataset_overrides: Any) -> RecordingEnv:
    settings = DatasetSettings(
        environment="incremental-test",
        output_dir=str(tmp_path),
        output_basename="export",
        output_sharegpt=True,
        output_audit_jsonl=True,
        **dataset_overrides,
    )
    config = EnvConfig(name="incremental-test", llm=InferenceConfig(), dataset=settings)
    env = RecordingEnv(config=config, task_ids=task_ids)
    env.max_parallel_rollouts = 1
    return env


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


@pytest.mark.asyncio
async def test_rows_are_on_disk_before_the_run_finishes(tmp_path: Path) -> None:
    env = make_env(tmp_path, ["a", "b", "c"])
    export = tmp_path / "export.jsonl"
    seen_counts: List[int] = []

    def observe(_task_id: str) -> None:
        seen_counts.append(len(read_jsonl(export)) if export.exists() else 0)

    env._observer = observe

    await env.run()

    # Task 2 sees task 1's row already durable, task 3 sees two.
    assert seen_counts == [0, 1, 2]
    assert [r["id"] for r in read_jsonl(export)] == ["a", "b", "c"]
    assert [r["id"] for r in read_jsonl(tmp_path / "export_audit.jsonl")] == ["a", "b", "c"]


@pytest.mark.asyncio
async def test_interrupted_run_keeps_finished_rows_and_resumes(tmp_path: Path) -> None:
    env = make_env(tmp_path, ["a", "b", "c", "d"])
    env._fail_on = "c"  # a Ctrl-C partway through the run

    with pytest.raises(SimulatedInterrupt):
        await env.run()

    export = tmp_path / "export.jsonl"
    written = [r["id"] for r in read_jsonl(export)]
    # Tasks that finished before the interrupt are durable; the interrupted one is not.
    # (Siblings already scheduled by asyncio.gather may also finish; a real Ctrl-C
    # kills the process instead.)
    assert {"a", "b"} <= set(written)
    assert "c" not in written

    # Resuming re-runs only what is missing and appends it.
    resumed = make_env(tmp_path, ["a", "b", "c", "d"])
    summary = await resumed.run()

    assert "c" in resumed._seen
    assert not ({"a", "b"} & set(resumed._seen)), "already-exported tasks must not be re-run"
    final = [r["id"] for r in read_jsonl(export)]
    assert sorted(final) == ["a", "b", "c", "d"]
    assert len(final) == len(set(final)), "resume must not duplicate rows"
    # The audit is an append-only log: the interrupted attempt at "c" is kept
    # (errored=True) alongside the successful retry.
    audit = read_jsonl(tmp_path / "export_audit.jsonl")
    assert sorted({r["id"] for r in audit}) == ["a", "b", "c", "d"]
    c_rows = [r for r in audit if r["id"] == "c"]
    assert [r["errored"] for r in c_rows] == [True, False]
    assert summary.accepted == len(resumed._seen)


@pytest.mark.asyncio
async def test_resume_disabled_rewrites_from_scratch(tmp_path: Path) -> None:
    first = make_env(tmp_path, ["a", "b"], resume=False)
    await first.run()

    second = make_env(tmp_path, ["a", "b"], resume=False)
    await second.run()

    assert second._seen == ["a", "b"]
    assert [r["id"] for r in read_jsonl(tmp_path / "export.jsonl")] == ["a", "b"]


@pytest.mark.asyncio
async def test_legacy_write_once_path_still_works(tmp_path: Path) -> None:
    env = make_env(tmp_path, ["a", "b"], incremental_export=False)
    export = tmp_path / "export.jsonl"
    during: List[bool] = []
    env._observer = lambda _tid: during.append(export.exists())

    summary = await env.run()

    assert during == [False, False], "write-once must not create the file mid-run"
    assert [r["id"] for r in read_jsonl(export)] == ["a", "b"]
    assert summary.accepted == 2


@pytest.mark.asyncio
async def test_summary_counts_match_written_rows(tmp_path: Path) -> None:
    env = make_env(tmp_path, ["a", "b", "c"])
    summary = await env.run()

    payload = json.loads((tmp_path / "export_summary.json").read_text(encoding="utf-8"))
    assert payload["accepted"] == summary.accepted == 3
    assert payload["processed"] == 3
    assert payload["mean_reward"] == 1.0
    assert payload["artifacts"]["sharegpt_jsonl"].endswith("export.jsonl")
