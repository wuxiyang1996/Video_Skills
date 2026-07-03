"""Evaluate whether saved L1 clue graphs can support query-time L2 memory.

This is an offline diagnostic. It may compare against hidden gold answers for
evaluation, but it never feeds gold labels back into retrieval or scoring.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


_TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
_STOPWORDS = {
    "a",
    "an",
    "the",
    "is",
    "are",
    "was",
    "were",
    "what",
    "when",
    "where",
    "who",
    "why",
    "how",
    "which",
    "does",
    "do",
    "did",
    "in",
    "on",
    "at",
    "to",
    "of",
    "and",
    "or",
    "for",
    "with",
    "from",
    "that",
    "this",
    "it",
    "its",
    "their",
    "they",
    "he",
    "she",
    "his",
    "her",
    "video",
}


@dataclass(frozen=True)
class MemoryItem:
    source: str
    node_id: str
    node_type: str
    text: str
    time_span: dict[str, Any] | None
    hidden_supervision: bool


def _tokens(text: str) -> set[str]:
    return {tok.lower() for tok in _TOKEN_RE.findall(text) if len(tok) > 1 and tok.lower() not in _STOPWORDS}


def _text_of_node(node: dict[str, Any]) -> str:
    parts: list[str] = []
    for key in ("text", "description", "label", "content", "scene_description"):
        value = node.get(key)
        if isinstance(value, str) and value.strip():
            parts.append(value.strip())
    for key in ("observable_facts", "events", "entities", "dialogue", "ocr_text"):
        value = node.get(key)
        if isinstance(value, str) and value.strip():
            parts.append(value.strip())
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, str) and item.strip():
                    parts.append(item.strip())
                elif isinstance(item, dict):
                    parts.append(_text_of_node(item))
    return " ".join(part for part in parts if part)


def _is_hidden(node: dict[str, Any]) -> bool:
    visibility = node.get("visibility") or {}
    return bool(visibility.get("hidden_supervision") or node.get("hidden_supervision"))


def _iter_memory_items(example: dict[str, Any]) -> Iterable[MemoryItem]:
    graph = ((example.get("metadata") or {}).get("clue_memory_graph") or {})
    for node in graph.get("nodes") or []:
        text = _text_of_node(node)
        if text:
            yield MemoryItem(
                source="clue_memory_graph",
                node_id=str(node.get("node_id") or ""),
                node_type=str(node.get("node_type") or ""),
                text=text,
                time_span=node.get("time_span"),
                hidden_supervision=_is_hidden(node),
            )

    coarse_fine = ((example.get("metadata") or {}).get("coarse_fine_graph") or {})
    for node in ((coarse_fine.get("coarse_graph") or {}).get("nodes") or []):
        text = _text_of_node(node)
        if text:
            yield MemoryItem(
                source="coarse_fine_graph.coarse",
                node_id=str(node.get("node_id") or ""),
                node_type=str(node.get("node_type") or ""),
                text=text,
                time_span=node.get("time_span"),
                hidden_supervision=_is_hidden(node),
            )

    for node in ((coarse_fine.get("fine_graph") or {}).get("nodes") or []):
        text = _text_of_node(node)
        if text:
            yield MemoryItem(
                source="coarse_fine_graph.fine",
                node_id=str(node.get("node_id") or ""),
                node_type=str(node.get("node_type") or ""),
                text=text,
                time_span=node.get("time_span"),
                hidden_supervision=_is_hidden(node),
            )

    evidence_index = example.get("evidence_index") or {}
    for node in evidence_index.get("nodes") or []:
        text = _text_of_node(node)
        if text:
            yield MemoryItem(
                source="evidence_index",
                node_id=str(node.get("node_id") or node.get("evidence_id") or ""),
                node_type=str(node.get("node_type") or node.get("source_type") or ""),
                text=text,
                time_span=node.get("time_span"),
                hidden_supervision=_is_hidden(node),
            )


def _score(query: str, item: MemoryItem) -> float:
    query_tokens = _tokens(query)
    item_tokens = _tokens(item.text)
    if not query_tokens or not item_tokens:
        return 0.0
    overlap = query_tokens & item_tokens
    if not overlap:
        return 0.0
    return len(overlap) / max(1.0, len(query_tokens) ** 0.5)


def _top_items(query: str, items: list[MemoryItem], *, topk: int) -> list[tuple[float, MemoryItem]]:
    scored = [(_score(query, item), item) for item in items if not item.hidden_supervision]
    scored = [entry for entry in scored if entry[0] > 0]
    scored.sort(key=lambda entry: entry[0], reverse=True)
    return scored[:topk]


def _option_scores(question_text: str, options: list[dict[str, Any]], items: list[MemoryItem], *, topk: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for option in options:
        label = str(option.get("label") or "")
        text = str(option.get("text") or "")
        query = f"{question_text} {text}".strip()
        hits = _top_items(query, items, topk=topk)
        rows.append(
            {
                "label": label,
                "text": text,
                "score": round(sum(score for score, _ in hits), 4),
                "top_refs": [item.node_id for _, item in hits[:3]],
            }
        )
    rows.sort(key=lambda row: row["score"], reverse=True)
    return rows


def _compact_text(text: str, limit: int = 180) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _gold_label(example: dict[str, Any]) -> str | None:
    answer = (example.get("question") or {}).get("answer") or {}
    label = answer.get("label")
    if label:
        return str(label)
    text = answer.get("text")
    if not text:
        return None
    for option in (example.get("question") or {}).get("options") or []:
        if str(option.get("text") or "").strip().lower() == str(text).strip().lower():
            return str(option.get("label") or "")
    return None


def evaluate_example(example: dict[str, Any], *, topk: int) -> dict[str, Any]:
    question = example.get("question") or {}
    question_text = str(question.get("question_text") or "")
    options = question.get("options") or []
    items = list(_iter_memory_items(example))
    graph = ((example.get("metadata") or {}).get("clue_memory_graph") or {})
    coarse_fine = ((example.get("metadata") or {}).get("coarse_fine_graph") or {})
    top = _top_items(question_text, items, topk=topk)
    option_rows = _option_scores(question_text, options, items, topk=topk) if options else []
    predicted = option_rows[0]["label"] if option_rows else None
    gold = _gold_label(example)
    l2 = ((example.get("metadata") or {}).get("reasoning_rollout") or {})
    l2_final = l2.get("final_answer") or {}
    gold_text = ((question.get("answer") or {}).get("text") or "").strip()
    l2_text = str(l2_final.get("text") or "").strip()

    return {
        "example_id": example.get("example_id"),
        "dataset": example.get("dataset"),
        "question": question_text,
        "graph_nodes": len(graph.get("nodes") or []),
        "graph_edges": len(graph.get("edges") or []),
        "memory_items": len(items),
        "hidden_memory_items": sum(1 for item in items if item.hidden_supervision),
        "coarse_fine_counts": coarse_fine.get("counts") or {},
        "coarse_coverage": ((coarse_fine.get("coarse_graph") or {}).get("coverage")),
        "fine_coverage": ((coarse_fine.get("fine_graph") or {}).get("coverage")),
        "selected_coarse_indices": coarse_fine.get("selected_coarse_indices") or [],
        "top_question_hits": [
            {
                "score": round(score, 4),
                "source": item.source,
                "node_id": item.node_id,
                "node_type": item.node_type,
                "time_span": item.time_span,
                "text": _compact_text(item.text),
            }
            for score, item in top
        ],
        "option_scores": option_rows,
        "predicted_label_by_l1_memory": predicted,
        "gold_label_eval_only": gold,
        "correct_eval_only": bool(predicted and gold and predicted == gold),
        "l2_rollout_source": l2.get("rollout_source"),
        "l2_final_answer": l2_final,
        "l2_uses_gold_text_warning": bool(gold_text and l2_text == gold_text),
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    reports: list[dict[str, Any]] = []
    for path in args.paths:
        for example in _read_jsonl(path):
            report = evaluate_example(example, topk=args.topk)
            report["source_path"] = str(path)
            reports.append(report)

    correct = sum(1 for row in reports if row["correct_eval_only"])
    summary = {
        "examples": len(reports),
        "correct_eval_only": correct,
        "accuracy_eval_only": round(correct / len(reports), 4) if reports else 0.0,
        "datasets": sorted({str(row["dataset"]) for row in reports}),
    }
    payload = {"summary": summary, "reports": reports}

    text = json.dumps(payload, indent=2, ensure_ascii=False)
    if args.output:
        args.output.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
