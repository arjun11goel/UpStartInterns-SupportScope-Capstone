"""Run the JSONL evaluation set and save scored results and usage telemetry."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
sys.path.insert(0, str(ROOT / "src"))

from supportscope.assistant import answer_question  # noqa: E402


def normalise(text: str) -> str:
    """Normalise punctuation so 5–10 and 5-10 match in benchmark checks."""
    return re.sub(r"[^a-z0-9@.]+", " ", text.lower()).strip()


def load_cases(path: Path) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                case = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc
            for key in ("id", "query", "expected_status", "required_facts", "expected_sources"):
                if key not in case:
                    raise ValueError(f"Evaluation case on line {line_number} lacks '{key}'.")
            cases.append(case)
    if not cases:
        raise ValueError("Evaluation set is empty.")
    if len({case["id"] for case in cases}) != len(cases):
        raise ValueError("Evaluation case IDs must be unique.")
    return cases


def score_case(case: dict[str, Any], response: dict[str, Any]) -> dict[str, Any]:
    answer_norm = normalise(response["answer"])
    required_facts = case["required_facts"]
    facts_found = [fact for fact in required_facts if normalise(fact) in answer_norm]
    expected_sources = set(case["expected_sources"])
    actual_sources = set(response["source_ids"])
    source_check = expected_sources.issubset(actual_sources)
    status_check = response["status"] == case["expected_status"]
    facts_check = len(facts_found) == len(required_facts)
    passed = status_check and facts_check and source_check
    return {
        "id": case["id"],
        "category": case["expected_status"],
        "query": case["query"],
        "expected_status": case["expected_status"],
        "actual_status": response["status"],
        "status_pass": status_check,
        "facts_pass": facts_check,
        "facts_found": facts_found,
        "facts_required": required_facts,
        "source_pass": source_check,
        "expected_sources": sorted(expected_sources),
        "actual_sources": response["source_ids"],
        "pass": passed,
        "answer": response["answer"],
        "input_tokens": response["input_tokens"],
        "output_tokens": response["output_tokens"],
        "total_tokens": response["total_tokens"],
        "estimated_cost_usd": response["estimated_cost_usd"],
        "latency_seconds": response["latency_seconds"],
        "max_retrieval_score": response["max_retrieval_score"],
        "model": response["model"],
        "error": response["error"],
    }


def summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_category[row["category"]].append(row)
    category_metrics = {}
    for category, group in sorted(by_category.items()):
        category_metrics[category] = {
            "passed": sum(bool(row["pass"]) for row in group),
            "total": len(group),
            "pass_rate": round(sum(bool(row["pass"]) for row in group) / len(group), 4),
        }
    latencies = [row["latency_seconds"] for row in rows]
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "total_cases": len(rows),
        "passed_cases": sum(bool(row["pass"]) for row in rows),
        "failed_cases": sum(not bool(row["pass"]) for row in rows),
        "overall_pass_rate": round(sum(bool(row["pass"]) for row in rows) / len(rows), 4),
        "category_metrics": category_metrics,
        "source_check_rate": round(sum(bool(row["source_pass"]) for row in rows) / len(rows), 4),
        "input_tokens": sum(int(row["input_tokens"]) for row in rows),
        "output_tokens": sum(int(row["output_tokens"]) for row in rows),
        "total_tokens": sum(int(row["total_tokens"]) for row in rows),
        "estimated_cost_usd": round(sum(float(row["estimated_cost_usd"]) for row in rows), 8),
        "mean_latency_seconds": round(statistics.mean(latencies), 4) if latencies else 0,
        "p50_latency_seconds": round(statistics.median(latencies), 4) if latencies else 0,
        "cost_note": "Cost is an estimate at configured published per-token reference rates, not a billing statement. Free-plan usage is subject to provider limits.",
    }


def write_reports(output_dir: Path, rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "evaluation_results.json").write_text(
        json.dumps({"summary": summary, "cases": rows}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    if rows:
        with (output_dir / "evaluation_results.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    lines = [
        "# SupportScope AI — Evaluation Results",
        "",
        f"- Run time (UTC): {summary['created_at_utc']}",
        f"- Overall: {summary['passed_cases']}/{summary['total_cases']} passed ({summary['overall_pass_rate']:.1%})",
        f"- Total tokens: {summary['total_tokens']} ({summary['input_tokens']} input, {summary['output_tokens']} output)",
        f"- Estimated reference cost: ${summary['estimated_cost_usd']:.8f}",
        f"- Mean latency: {summary['mean_latency_seconds']:.3f}s; P50: {summary['p50_latency_seconds']:.3f}s",
        "",
        "## Results by expected behavior",
        "",
        "| Behavior | Passed | Total | Pass rate |",
        "|---|---:|---:|---:|",
    ]
    for category, metric in summary["category_metrics"].items():
        lines.append(f"| {category} | {metric['passed']} | {metric['total']} | {metric['pass_rate']:.1%} |")
    lines.extend(["", summary["cost_note"], "", "See `evaluation_results.csv` for case-level scores and `evaluation_results.json` for full answers and telemetry.", ""])
    (output_dir / "evaluation_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "evals" / "evaluation_set.jsonl")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports")
    args = parser.parse_args()
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("ERROR: GROQ_API_KEY is missing. Copy .env.example to .env and add your free Groq API key.", file=sys.stderr)
        return 2

    cases = load_cases(args.dataset)
    rows: list[dict[str, Any]] = []
    print(f"Running {len(cases)} evaluation cases. In-scope cases call the API; blocked cases should not.")
    for index, case in enumerate(cases, start=1):
        response = answer_question(case["query"], api_key=api_key).as_dict()
        row = score_case(case, response)
        rows.append(row)
        symbol = "PASS" if row["pass"] else "FAIL"
        print(f"[{index:02d}/{len(cases):02d}] {symbol} {case['id']} status={row['actual_status']} tokens={row['total_tokens']}")
    summary = summarise(rows)
    write_reports(args.output_dir, rows, summary)
    print("\n=== Summary ===")
    print(json.dumps(summary, indent=2))
    print(f"\nSaved reports to: {args.output_dir.resolve()}")
    return 0 if summary["failed_cases"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
