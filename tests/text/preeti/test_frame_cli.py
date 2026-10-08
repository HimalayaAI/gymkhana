import json

import pandas as pd

from gymkhana.text.preeti import PreetiFixer, fix_frame
from gymkhana.text.preeti.cli import main


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "chunk_id": ["a", "b"],
            "doc_type": ["pdf", "html"],
            "text": ["Ao'l6 पार्लर सञ्चालन", "शुद्ध पाठ"],
        }
    )


def test_fix_frame_keeps_original_and_reports() -> None:
    df = _frame()
    res = fix_frame(df, PreetiFixer(), n_pairs=5)
    assert res.frame["text"].tolist() == ["ब्युटि पार्लर सञ्चालन", "शुद्ध पाठ"]
    assert res.frame["text_original"].tolist() == df["text"].tolist()
    assert res.frame["preeti_touched"].tolist() == [True, False]
    assert res.report["rows_touched"] == 1
    assert res.pairs[0]["id"] == "a"


def test_cli_build_vocab_and_fix(tmp_path, capsys) -> None:
    src = tmp_path / "in.parquet"
    _frame().to_parquet(src, index=False)
    vocab = tmp_path / "vocab"
    assert main(["build-vocab", "--input", str(src), "--nepali-where", "doc_type=html",
                 "--min-freq", "1", "--out", str(vocab)]) == 0
    assert (vocab / "nepali.txt").read_text(encoding="utf-8").split() == ["पाठ", "शुद्ध"]

    out = tmp_path / "out.parquet"
    assert main(["fix", "--input", str(src), "--vocab-dir", str(vocab), "--output", str(out)]) == 0
    fixed = pd.read_parquet(out)
    assert fixed["text"].iloc[0] == "ब्युटि पार्लर सञ्चालन"
    report = json.loads(out.with_suffix(".report.json").read_text(encoding="utf-8"))
    assert report["rows_touched"] == 1
    assert out.with_suffix(".pairs.jsonl").exists()


def test_cli_convert(capsys) -> None:
    assert main(["convert", "--whole", "g]kfn"]) == 0
    assert capsys.readouterr().out.strip() == "नेपाल"
