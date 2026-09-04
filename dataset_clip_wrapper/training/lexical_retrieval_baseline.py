#!/usr/bin/env python3
"""BM25 reference point for L2 candidate reranking.

The learned reranker is worth its cost only if it beats matching the question
against the clip text directly.  Oracle selection over the same candidates
reaches far above what the reranker scores, so before adding training signal it
is worth knowing whether a non-learned baseline already closes part of that gap.

Emits the same report shape as ``evaluate_l2_pointwise_adapter`` so the official
metric scripts consume it unchanged.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

from trainer.grpo.l2_dataset_rewards import _text as candidate_text

# Question words carry no retrieval signal and appear in every query.
_STOPWORDS = frozenset(
    "a an the is are was were be been being am of to in on at for with and or but from that this "
    "it its he she they them his her their as by not no do does did have has had will would can "
    "could i you we what who how why when where which whom there here then than so if while during "
    "about into over under after before between".split()
)


def tokenize(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9]+", (text or "").lower()) if w not in _STOPWORDS and len(w) > 2]


class BM25:
    """Textbook BM25 over a single example's candidate descriptions."""

    def __init__(self, documents: Sequence[Sequence[str]], *, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.documents = [Counter(document) for document in documents]
        self.lengths = [sum(counts.values()) for counts in self.documents]
        self.average_length = (sum(self.lengths) / len(self.lengths)) if self.lengths else 0.0
        document_frequency: Counter[str] = Counter()
        for counts in self.documents:
            document_frequency.update(counts.keys())
        total = max(1, len(self.documents))
        self.idf = {
            term: math.log(1.0 + (total - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in document_frequency.items()
        }

    def score(self, index: int, query: Iterable[str]) -> float:
        counts = self.documents[index]
        length = self.lengths[index]
        if not length or not self.average_length:
            return 0.0
        total = 0.0
        for term in query:
            frequency = counts.get(term, 0)
            if not frequency:
                continue
            denominator = frequency + self.k1 * (1 - self.b + self.b * length / self.average_length)
            total += self.idf.get(term, 0.0) * frequency * (self.k1 + 1) / denominator
        return total


def question_text(payload: dict[str, Any]) -> str:
    question = payload.get("question") or {}
    parts = [question.get("question_text") or ""]
    parts.extend(str(option.get("text") or "") for option in question.get("options") or [])
    return " ".join(parts)


def build_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_example: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        metadata = row.get("metadata") or {}
        by_example.setdefault(str(metadata.get("source_example_id") or ""), []).append(row)
    results = []
    for example_id, group in sorted(by_example.items()):
        group.sort(key=lambda row: int((row.get("metadata") or {})["candidate_index"]))
        documents = [tokenize(candidate_text((row.get("metadata") or {}).get("candidate_entry") or {})) for row in group]
        payload = json.loads(group[0]["messages"][1]["content"])
        query = tokenize(question_text(payload.get("state_t") or {}))
        bm25 = BM25(documents)
        ranking = [
            {
                "candidate_index": int((row.get("metadata") or {})["candidate_index"]),
                "score": bm25.score(position, query),
                "retrieval_rank": int((row.get("metadata") or {}).get("retrieval_rank") or 0),
            }
            for position, row in enumerate(group)
        ]
        gold = sorted({int(v) for row in group for v in ((row.get("metadata") or {}).get("gold_indices") or [])})
        results.append({"example_id": example_id, "gold": gold, "ranking": ranking})
    return {
        "schema_version": "video-skills/l2-lexical-baseline-v1",
        "adapter": "bm25-question-vs-clip-text",
        "adapter_weight_sha256": None,
        "results": results,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pointwise-jsonl", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    rows = [json.loads(line) for line in args.pointwise_jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
    report = build_report(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"examples": len(report["results"]), "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
