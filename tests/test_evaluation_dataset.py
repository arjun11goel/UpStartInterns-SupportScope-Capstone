import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_dataset_is_valid_jsonl_with_unique_ids():
    rows = [json.loads(line) for line in (ROOT / "evals" / "evaluation_set.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) >= 20
    assert len({row["id"] for row in rows}) == len(rows)
    assert {row["expected_status"] for row in rows} == {"answered", "out_of_scope", "unsafe_refusal"}
    assert all(row["query"].strip() for row in rows)
