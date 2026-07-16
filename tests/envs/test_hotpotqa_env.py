"""Unit tests for HotpotQAEnv."""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from typing import Any, Dict, List

import pytest

from gymkhana.envs.config import EnvironmentType
from gymkhana.envs.hotpotqa import HotpotQAEnv
from gymkhana.envs import Task
from gymkhana.envs.environment import Environment


@pytest.fixture
def hotpot_env() -> HotpotQAEnv:
    config = HotpotQAEnv.default_config.model_copy(deep=True)
    config.dataset.environment = EnvironmentType.HOTPOTQA
    config.dataset.limit = 1
    return HotpotQAEnv(config=config)


@pytest.mark.asyncio
async def test_hotpotqa_load_tasks_normalizes_documents(monkeypatch: pytest.MonkeyPatch, hotpot_env: HotpotQAEnv):
    dataset_records: List[Dict[str, Any]] = [
        {
            "id": "hp-1",
            "question": "Who directed the movie and what is their nationality?",
            "answer": "Christopher Nolan is British.",
            "context": [
                [
                    "Inception",
                    [
                        "Inception is a 2010 science fiction film.",
                        "It was written and directed by Christopher Nolan.",
                    ],
                ],
                [
                    "Christopher Nolan",
                    [
                        "Christopher Nolan is a British-American film director.",
                    ],
                ],
            ],
        }
    ]

    monkeypatch.setattr(hotpot_env, "_load_dataset", lambda *_: dataset_records)

    tasks = hotpot_env.load_tasks(limit=1)
    assert len(tasks) == 1
    task = tasks[0]

    # Context should contain serialized documents
    documents = json.loads(task.context)
    assert len(documents) == 2
    assert documents[0]["filename"].startswith("doc_01_")
    assert "Christopher Nolan" in documents[1]["content"]

    # Metadata mirrors documents and expected answer
    assert "documents" in task.metadata
    assert task.metadata["expected_answer"] == "Christopher Nolan is British."


def test_hotpotqa_format_initial_message_lists_documents(hotpot_env: HotpotQAEnv):
    task = Task(
        id="hp-1",
        prompt="Who directed the movie and what is their nationality?",
        context=json.dumps(
            [
                {
                    "filename": "doc_01_Inception.txt",
                    "title": "Inception",
                    "content": "Director: Christopher Nolan",
                },
                {
                    "filename": "doc_02_Christopher_Nolan.txt",
                    "title": "Christopher Nolan",
                    "content": "Nationality: British",
                },
            ]
        ),
        metadata={},
    )

    message = hotpot_env.format_initial_message(task)
    assert "doc_01_Inception.txt" in message
    assert "doc_02_Christopher_Nolan.txt" in message


def test_hotpotqa_instructions_toggle():
    config = HotpotQAEnv.default_config.model_copy(deep=True)
    env = HotpotQAEnv(config=config)
    task = Task(id="hp-1", prompt="Question", context=None)
    instructions = env.get_environment_instructions(task)
    assert "HotpotQA" in instructions

    config_without = HotpotQAEnv.default_config.model_copy(deep=True)
    config_without.dataset.include_instructions = False
    env_without = HotpotQAEnv(config=config_without)
    assert env_without.get_environment_instructions(task) == ""


@pytest.mark.asyncio
async def test_hotpotqa_repl_session_uploads_documents(monkeypatch: pytest.MonkeyPatch, hotpot_env: HotpotQAEnv):
    executed: List[str] = []

    class DummyRepl:
        async def execute(self, code: str):
            executed.append(code)
            return type("Execution", (), {"error": None})()

    @asynccontextmanager
    async def fake_repl_session(self, **_: Any):
        yield DummyRepl()

    monkeypatch.setattr(Environment, "repl_session", fake_repl_session, raising=False)

    documents = [
        {
            "filename": "doc_01_Inception.txt",
            "content": "Inception is directed by Christopher Nolan.",
        },
        {
            "filename": "doc_02_Christopher_Nolan.txt",
            "content": "Christopher Nolan is British.",
        },
    ]

    async with hotpot_env.repl_session(context=json.dumps(documents)):
        pass

    assert any("doc_01_Inception.txt" in snippet for snippet in executed)
    assert any("doc_02_Christopher_Nolan.txt" in snippet for snippet in executed)
