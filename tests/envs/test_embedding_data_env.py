from pathlib import Path

import pytest
from pydantic import PrivateAttr

from gymkhana.envs import ENVIRONMENTS
from gymkhana.envs.config import EnvironmentType
from gymkhana.envs.embedding_data import EmbeddingDataEnv
from gymkhana.run import load_environment_config


def test_environment_is_registered_and_configurable() -> None:
    assert ENVIRONMENTS.get("embedding-data") is EmbeddingDataEnv
    assert EnvironmentType.EMBEDDING_DATA.value == "embedding-data"


def test_seed_schema_is_swappable_via_field_mapping() -> None:
    env = EmbeddingDataEnv(records=[
        {
            "question_text": "कर्मचारीले कति बिदा पाउँछ?",
            "answer_text": "कर्मचारीले नियमानुसार बिदा पाउँछ।",
            "wrong_answers": ["यो अर्को विषय हो।"],
            "law": "श्रम ऐन",
        }
    ])
    env.config.dataset.field_mapping = {
        "id": None,
        "query": "question_text",
        "positive": "answer_text",
        "negatives": "wrong_answers",
        "source": "law",
        "document": None,
        "license": None,
        "script": None,
    }
    task = env.load_tasks()[0]
    assert task.prompt == "कर्मचारीले कति बिदा पाउँछ?"
    assert task.metadata["positive"] == "कर्मचारीले नियमानुसार बिदा पाउँछ।"
    assert task.metadata["negatives"] == ["यो अर्को विषय हो।"]
    assert task.metadata["source_provenance"]["source"] == "श्रम ऐन"


def test_rows_from_same_document_share_deterministic_split() -> None:
    env = EmbeddingDataEnv(records=[
        {"id": "a", "query": "प्रश्न एक?", "positive": "उत्तर एक।", "doc": "act-a"},
        {"id": "b", "query": "प्रश्न दुई?", "positive": "उत्तर दुई।", "doc": "act-a"},
    ])
    env.config.dataset.field_mapping.update({"document": "doc"})
    tasks = env.load_tasks()
    assert tasks[0].metadata["split"] == tasks[1].metadata["split"]


def test_duplicate_queries_join_document_groups_before_splitting() -> None:
    env = EmbeddingDataEnv(records=[
        {"id": "a", "query": "एक साझा प्रश्न?", "positive": "उत्तर एक।", "doc": "act-a"},
        {"id": "b", "query": "दोस्रो प्रश्न?", "positive": "उत्तर दुई।", "doc": "act-a"},
        {"id": "c", "query": "एक साझा प्रश्न?", "positive": "उत्तर तीन।", "doc": "act-b"},
        {"id": "d", "query": "चौथो प्रश्न?", "positive": "उत्तर चार।", "doc": "act-b"},
    ])
    env.config.dataset.field_mapping.update({"document": "doc"})

    tasks = env.load_tasks()

    assert len({task.metadata["split"] for task in tasks}) == 1


def test_same_query_with_distinct_positives_is_preserved_and_ids_are_unique() -> None:
    env = EmbeddingDataEnv(records=[
        {"id": "duplicate-id", "query": "एक प्रश्न?", "positive": "पहिलो उत्तर।"},
        {"id": "duplicate-id", "query": "एक प्रश्न?", "positive": "अर्को सही उत्तर।"},
    ])
    tasks = env.load_tasks()
    assert len(tasks) == 2
    assert tasks[0].id != tasks[1].id
    assert {task.metadata["positive"] for task in tasks} == {"पहिलो उत्तर।", "अर्को सही उत्तर।"}


def test_bad_seed_negatives_are_removed_and_zero_limit_is_honored() -> None:
    env = EmbeddingDataEnv(records=[{
        "id": "seed-3",
        "query": "प्रश्न?",
        "positive": "सही उत्तर।",
        "negatives": ["सही उत्तर।", "  ", "गलत उत्तर।", "गलत उत्तर।", None],
    }])
    task = env.load_tasks()[0]
    assert task.metadata["negatives"] == ["गलत उत्तर।"]
    assert env.load_tasks(limit=0) == []


def test_malformed_seed_fields_are_skipped_and_offsets_are_audited() -> None:
    env = EmbeddingDataEnv(records=[
        *({"query": f"छोडिने प्रश्न {index}?", "positive": "उत्तर।"} for index in range(7)),
        {"query": 42, "positive": "यो पाठ होइन"},
        {"query": "प्रश्न?", "positive": "उत्तर।", "negatives": [{"text": "invalid"}]},
    ])
    env.config.dataset.dataset_offset = 7

    tasks = env.load_tasks()

    assert len(tasks) == 1
    assert tasks[0].metadata["negatives"] == []
    assert tasks[0].metadata["source_row_index"] == 8


@pytest.mark.asyncio
async def test_wrong_script_candidate_is_audited_without_semantic_judge() -> None:
    class ScriptedEmbeddingEnv(EmbeddingDataEnv):
        _responses = PrivateAttr()

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self._responses = iter(['{"queries": ["How much leave is allowed?"]}'])

        async def generate_response(self, **kwargs):
            return next(self._responses), None

    env = ScriptedEmbeddingEnv(records=[{
        "id": "seed-script",
        "query": "कर्मचारीले कति बिदा पाउँछ?",
        "positive": "कर्मचारीले नियमानुसार बिदा पाउँछ।",
    }])
    env._inference_service = object()
    result = await env.run_task(env.load_tasks()[0])

    assert result.metadata["generation_error"] is None
    assert result.metadata["candidate_evaluations"][0]["accepted"] is False
    assert "script ratio" in result.metadata["candidate_evaluations"][0]["reason"]
    assert len(result.metadata["retrieval_rows"]) == 1


def test_resume_uses_configured_audit_basename_and_keeps_failed_rows_retryable(tmp_path: Path) -> None:
    env = EmbeddingDataEnv(records=[
        {"id": "retry-me", "query": "प्रश्न?", "positive": "उत्तर।"},
    ])
    env.config.dataset.output_dir = str(tmp_path)
    env.config.dataset.output_basename = "custom_pairs"
    audit_path = tmp_path / "custom_pairs_audit.jsonl"
    audit_path.write_text(
        '{"id":"retry-me","errored":true}\n', encoding="utf-8"
    )
    assert [task.id for task in env.load_tasks()] == ["retry-me"]
    audit_path.write_text(
        '{"id":"retry-me","errored":false}\n', encoding="utf-8"
    )
    assert env.load_tasks() == []


def test_nepali_seed_config_loads() -> None:
    _, config = load_environment_config(
        Path("configs/embedding_data/nepali_legal_seed.yaml")
    )
    assert config.dataset.dataset_name == "Somtharu181coder/jiban4_law_qa_clean"
    assert config.dataset.field_mapping["query"] == "query"
    assert config.embedding.domain == "Nepali legal retrieval"


@pytest.mark.asyncio
async def test_generated_queries_keep_seed_labels_and_are_judge_filtered() -> None:
    class ScriptedEmbeddingEnv(EmbeddingDataEnv):
        _responses = PrivateAttr()

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self._responses = iter([
                '{"queries": ["कर्मचारीले कति बिदा प्राप्त गर्न सक्छ?"]}',
                '{"score": 9, "reason": "Same question intent."}',
            ])

        async def generate_response(self, **kwargs):
            return next(self._responses), None

    env = ScriptedEmbeddingEnv(records=[{
        "id": "seed-1",
        "query": "कर्मचारीले कति बिदा पाउँछ?",
        "positive": "कर्मचारीले नियमानुसार बिदा पाउँछ।",
        "negatives": ["अर्को विषयको पाठ।"],
    }])
    env._inference_service = object()
    task = env.load_tasks()[0]
    result = await env.run_task(task)
    rows = result.metadata["retrieval_rows"]

    assert len(rows) == 2
    assert rows[0]["label_source"] == "seed"
    assert rows[1]["label_source"] == "generated_paraphrase"
    assert rows[1]["positive"] == task.metadata["positive"]
    assert rows[1]["negatives"] == task.metadata["negatives"]
    assert rows[1]["semantic_score"] == 0.9


@pytest.mark.asyncio
async def test_seed_only_run_writes_retrieval_jsonl(tmp_path: Path) -> None:
    import json

    env = EmbeddingDataEnv(records=[{
        "id": "seed-2",
        "query": "कर्मचारीले कति बिदा पाउँछ?",
        "positive": "कर्मचारीले नियमानुसार बिदा पाउँछ।",
        "negatives": ["अर्को विषयको पाठ।"],
        "doc_name": "श्रम ऐन",
    }])
    env.config.dataset.output_dir = str(tmp_path)
    env.config.dataset.incremental_export = False
    env.config.dataset.field_mapping["document"] = "doc_name"
    env.embedding_config.embedding.generate_paraphrases = False

    summary = await env.run()
    pair_path = Path(summary.artifacts["retrieval_pairs_jsonl"])
    rows = [json.loads(line) for line in pair_path.read_text(encoding="utf-8").splitlines()]

    assert summary.accepted == 1
    assert rows[0]["query"] == "कर्मचारीले कति बिदा पाउँछ?"
    assert rows[0]["positive"] == "कर्मचारीले नियमानुसार बिदा पाउँछ।"
    assert rows[0]["negative_types"] == ["seed_unreviewed"]
    assert rows[0]["split"] in {"train", "validation", "test"}
    assert rows[0]["source_provenance"]["document"] == "श्रम ऐन"
    audit_path = Path(summary.artifacts["retrieval_audit_jsonl"])
    audit_row = json.loads(audit_path.read_text(encoding="utf-8").splitlines()[0])
    assert "seed_row" not in audit_row
    assert "retrieval_summary_json" in summary.artifacts


def _verifying_env(judge_replies, records):
    class JudgedEmbeddingEnv(EmbeddingDataEnv):
        _replies = PrivateAttr()

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self._replies = iter(judge_replies)

        async def generate_response(self, **kwargs):
            return next(self._replies), None

    def retrieve_all(queries, passages, top_k):
        return [list(range(len(passages))) for _ in queries]

    env = JudgedEmbeddingEnv(records=records, negative_retriever=retrieve_all)
    env._inference_service = object()
    settings = env.embedding_config.embedding
    settings.generate_paraphrases = False
    settings.verify_negatives = True
    settings.negative_target = 2
    return env


@pytest.mark.asyncio
async def test_verify_stage_drops_false_negatives_and_audits_them() -> None:
    records = [
        {"id": "a", "query": "क्वेरी एक?", "positive": "उत्तर एक।", "negatives": ["बिल्कुल फरक विषय।"]},
        {"id": "b", "query": "क्वेरी दुई?", "positive": "उत्तर दुई।"},
        {"id": "c", "query": "क्वेरी तीन?", "positive": "उत्तर तीन।"},
    ]
    # Judge order for task a: positive, seed negative, then pool candidates "उत्तर दुई।", "उत्तर तीन।".
    env = _verifying_env(['{"scores": [9, 0, 8, 1]}'], records)
    task = env.load_tasks()[0]
    result = await env.run_task(task)
    row = result.metadata["retrieval_rows"][0]

    assert row["negatives"] == ["बिल्कुल फरक विषय।", "उत्तर तीन।"]
    assert row["negative_types"] == ["seed_verified", "mined_verified"]
    assert row["below_target"] is False
    audit = result.metadata["negative_audit"]
    assert [item["text"] for item in audit["rejected_false_negatives"]] == ["उत्तर दुई।"]


@pytest.mark.asyncio
async def test_verify_stage_flags_short_rows_and_never_exports_unverified_negatives_on_judge_failure() -> None:
    records = [
        {"id": "a", "query": "क्वेरी एक?", "positive": "उत्तर एक।", "negatives": ["बिल्कुल फरक विषय।"]},
        {"id": "b", "query": "क्वेरी दुई?", "positive": "उत्तर दुई।"},
    ]
    env = _verifying_env(['{"scores": [9, 0, 9]}', "not json at all"], records)
    first, second = env.load_tasks()
    short = await env.run_task(first)
    assert short.metadata["retrieval_rows"][0]["negatives"] == ["बिल्कुल फरक विषय।"]
    assert short.metadata["retrieval_rows"][0]["below_target"] is True

    failed = await env.run_task(second)
    assert failed.metadata["retrieval_rows"][0]["negatives"] == []
    assert failed.metadata["retrieval_rows"][0]["below_target"] is True
    assert failed.metadata["verification_errors"]


def test_default_retriever_gets_e5_prefixes_from_settings(monkeypatch) -> None:
    import gymkhana.envs.embedding_data.negatives as negatives

    seen = {}

    def fake_factory(model_name, query_prefix="", doc_prefix=""):
        seen.update(model=model_name, query=query_prefix, doc=doc_prefix)
        return lambda queries, passages, top_k: [list(range(len(passages))) for _ in queries]

    monkeypatch.setattr(negatives, "sentence_transformer_retriever", fake_factory)
    env = EmbeddingDataEnv(records=[{"id": "a", "query": "क्वेरी एक?", "positive": "उत्तर एक।"}])
    env.embedding_config.embedding.verify_negatives = True
    env.load_tasks()
    assert seen == {"model": "intfloat/multilingual-e5-small", "query": "query: ", "doc": "passage: "}


def test_missing_sentence_transformers_stops_with_install_hint(monkeypatch) -> None:
    import sys

    monkeypatch.setitem(sys.modules, "sentence_transformers", None)  # simulates a clean install
    env = EmbeddingDataEnv(records=[{"id": "a", "query": "क्वेरी एक?", "positive": "उत्तर एक।"}])
    env.embedding_config.embedding.verify_negatives = True
    with pytest.raises(ImportError, match=r"gymkhana\[embedding\]"):
        env.load_tasks()
