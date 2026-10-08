from gymkhana.envs.embedding_data.decontam import filter_rows, overlap, build_index
from gymkhana.envs.embedding_data.negatives import mine_verified_negatives


def _retrieve_all(queries, passages, top_k):
    return [list(range(min(top_k, len(passages)))) for _ in queries]


def test_mining_rejects_false_negatives_and_same_query_positives_and_flags_short_rows() -> None:
    rows = [
        {"query": "q1", "positive": "alpha answer"},
        {"query": "q2", "positive": "beta other"},
        {"query": "q1", "positive": "alpha duplicate question answer"},  # same query as row 0
    ]
    answers = {("q1", "alpha answer"): 0.95, ("q2", "beta other"): 0.9, ("q1", "alpha duplicate question answer"): 0.9}

    def rerank(pairs):
        return [answers.get(p, 0.02) for p in pairs]

    out, stats = mine_verified_negatives(
        rows, retrieve=_retrieve_all, rerank=rerank, extra_passages=["gamma", "delta"], n_negatives=3
    )
    first = out[0]
    assert "alpha answer" not in first["negatives"]
    assert "alpha duplicate question answer" not in first["negatives"]  # positive of a same-query row
    assert set(first["negatives"]) == {"beta other", "gamma", "delta"}
    assert first["below_target"] is False and first["negative_types"] == ["mined_verified"] * 3
    assert stats["rows"] == 3


def test_mining_flags_row_below_target_instead_of_padding() -> None:
    rows = [{"query": "q", "positive": "p"}]
    out, stats = mine_verified_negatives(
        rows, retrieve=_retrieve_all, rerank=lambda pairs: [0.9] + [0.8] * (len(pairs) - 1), extra_passages=["x", "y"]
    )
    assert out[0]["negatives"] == [] and out[0]["below_target"] is True
    assert stats["below_target"] == 1 and stats["rejected_false_negative"] == 2


def test_devanagari_overlap_is_not_hidden_by_matras() -> None:
    eval_text = "नेपालको संविधानले प्रत्येक नागरिकलाई शिक्षाको हक दिएको छ।"
    index = build_index([eval_text])
    assert overlap("नेपालको संविधानले प्रत्येक नागरिकलाई शिक्षाको हक दिएको छ", index) == 1.0
    assert overlap("बिल्कुल फरक वाक्य यहाँ छ भन्ने कुरा हो", index) == 0.0


def test_filter_rows_drops_overlapping_rows_and_prunes_negatives() -> None:
    eval_text = "नेपालको संविधानले प्रत्येक नागरिकलाई शिक्षाको हक दिएको छ।"
    rows = [
        {"query": "के हो?", "positive": eval_text},
        {"query": "के हो?", "positive": "बिल्कुल फरक वाक्य यहाँ छ भन्ने कुरा हो", "negatives": [eval_text, "अर्कै कुरा हो यो पाँच शब्दभन्दा लामो"], "negative_types": ["a", "b"]},
    ]
    kept, dropped = filter_rows(rows, [eval_text])
    assert len(dropped) == 1 and dropped[0]["decontam"]["field"] == "positive"
    assert kept[0]["negatives"] == ["अर्कै कुरा हो यो पाँच शब्दभन्दा लामो"] and kept[0]["negative_types"] == ["b"]


def test_below_target_is_recomputed_after_pruning_negatives() -> None:
    eval_text = "नेपालको संविधानले प्रत्येक नागरिकलाई शिक्षाको हक दिएको छ।"
    row = {
        "query": "के हो?",
        "positive": "बिल्कुल फरक वाक्य यहाँ छ भन्ने कुरा हो",
        "negatives": [eval_text, "एक अर्को लामो नकारात्मक वाक्य यहाँ छ"],
        "negative_types": ["mined_verified", "mined_verified"],
        "below_target": False,
    }
    kept, _ = filter_rows([row], [eval_text], negative_target=2)
    assert len(kept[0]["negatives"]) == 1
    assert kept[0]["below_target"] is True and kept[0]["negatives_pruned"] == 1


def test_cli_decontam_indexes_passages_from_a_separate_eval_corpus(tmp_path) -> None:
    import json

    from gymkhana.envs.embedding_data.postprocess import main

    eval_passage = "नेपालको संविधानले प्रत्येक नागरिकलाई शिक्षाको हक दिएको छ।"
    (tmp_path / "queries.jsonl").write_text(
        json.dumps({"qid": "q1", "query": "शिक्षाको हक के हो", "positive_ids": ["p1"]}, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (tmp_path / "corpus.jsonl").write_text(json.dumps({"pid": "p1", "text": eval_passage}, ensure_ascii=False) + "\n", encoding="utf-8")
    (tmp_path / "train.jsonl").write_text(
        json.dumps({"query": "कुनै अर्को प्रश्न हो यो", "positive": eval_passage, "negatives": []}, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    out = tmp_path / "clean.jsonl"
    args = ["decontam", str(tmp_path / "train.jsonl"), str(out), "--eval-jsonl", str(tmp_path / "queries.jsonl")]
    main(args)  # query table alone does not hold the passage text
    assert len(out.read_text(encoding="utf-8").splitlines()) == 1
    main(args + ["--eval-corpus", str(tmp_path / "corpus.jsonl")])
    assert out.read_text(encoding="utf-8").strip() == ""
    assert len((tmp_path / "clean.dropped.jsonl").read_text(encoding="utf-8").splitlines()) == 1
