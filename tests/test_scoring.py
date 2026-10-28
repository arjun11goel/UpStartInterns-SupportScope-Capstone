import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from evaluate import score_case  # noqa: E402


def test_answer_scoring_checks_status_facts_and_sources():
    case = {
        "id": "T01",
        "query": "refund?",
        "expected_status": "answered",
        "required_facts": ["5-10 business days", "original payment method"],
        "expected_sources": ["REFUND-01"],
    }
    response = {
        "answer": "Approved refunds go to the original payment method within 5–10 business days. Sources: [REFUND-01]",
        "status": "answered",
        "source_ids": ["REFUND-01"],
        "input_tokens": 10,
        "output_tokens": 10,
        "total_tokens": 20,
        "estimated_cost_usd": 0.0001,
        "latency_seconds": 1.0,
        "max_retrieval_score": 0.5,
        "model": "test",
        "error": None,
    }
    result = score_case(case, response)
    assert result["pass"] is True
