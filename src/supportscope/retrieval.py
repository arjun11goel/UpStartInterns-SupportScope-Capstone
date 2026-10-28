"""TF-IDF retrieval over the project's small, fixed policy collection."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_KB_PATH = PROJECT_ROOT / "knowledge_base" / "policies.json"
RELEVANCE_THRESHOLD = 0.055


@dataclass(frozen=True)
class RetrievedPolicy:
    source_id: str
    title: str
    content: str
    score: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "title": self.title,
            "content": self.content,
            "score": round(self.score, 4),
        }


def _normalise_search_text(text: str) -> str:
    # Keep the literal email in source text, but avoid tokenising its "example.com"
    # domain as a misleading lexical match for unrelated "example" questions.
    return re.sub(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b", "supportaddress", text)


def _load_policies(path: Path = DEFAULT_KB_PATH) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        records = json.load(handle)
    if not isinstance(records, list) or not records:
        raise ValueError("The policy knowledge base must be a non-empty JSON list.")
    required = {"source_id", "title", "content", "keywords"}
    for record in records:
        missing = required.difference(record)
        if missing:
            raise ValueError(f"Policy entry is missing required keys: {sorted(missing)}")
    return records


class PolicyRetriever:
    """Fit a lightweight TF-IDF index once and return ranked policy entries."""

    def __init__(self, path: Path = DEFAULT_KB_PATH) -> None:
        self.policies = _load_policies(path)
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            sublinear_tf=True,
        )
        searchable_text = [
            _normalise_search_text(f"{item['title']} {item['content']} {' '.join(item['keywords'])}")
            for item in self.policies
        ]
        self.matrix = self.vectorizer.fit_transform(searchable_text)

    def retrieve(self, query: str, top_k: int = 2) -> list[RetrievedPolicy]:
        if not query.strip():
            return []
        query_vector = self.vectorizer.transform([_normalise_search_text(query)])
        scores = cosine_similarity(query_vector, self.matrix).flatten()
        ranked_indices = scores.argsort()[::-1][: max(1, top_k)]
        results = [
            RetrievedPolicy(
                source_id=self.policies[int(index)]["source_id"],
                title=self.policies[int(index)]["title"],
                content=self.policies[int(index)]["content"],
                score=float(scores[index]),
            )
            for index in ranked_indices
            if float(scores[index]) >= RELEVANCE_THRESHOLD
        ]
        return results
