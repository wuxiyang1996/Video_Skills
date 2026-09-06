"""Verifiable rewards for the reader, derived from the atomic-skill structure.

  correct              option letter equals gold
  citation precision   share of cited clip ranks whose time span overlaps an annotated
                       evidence span (Video-Holmes inference shots / VRBench steps / CG clues)
  time-order check     for timeline questions, cited ranks appear in increasing time order

reward = 1.0*correct + 0.5*citation_precision*correct + 0.1*format - 0.05*length_penalty
The process terms only pay when the answer is right, so the policy cannot farm
citations; the outcome-only ablation sets the process weights to 0.
"""
from __future__ import annotations

from typing import Any

from scripts.eval.measure_answer_chain import oracle_gold_spans, retrieval_catalog
from trainer.reader.prompting import cited_ranks, parse_reader_output


def _hit(a: dict[str, Any], b: dict[str, Any]) -> bool:
    return float(a.get("start_s") or 0) < float(b.get("end_s") or 0) and float(b.get("start_s") or 0) < float(a.get("end_s") or 0)


def citation_precision(example: dict[str, Any], ranks: list[int]) -> float | None:
    """None when the example carries no annotated evidence spans (no process signal)."""
    gold = oracle_gold_spans(example) or []
    if not gold:
        return None
    schemas, _ = retrieval_catalog(example)
    spans = [(s.get("time_span") or {}) for s in schemas]
    cited = [spans[r - 1] for r in ranks if 0 < r <= len(spans)]
    if not cited:
        return 0.0
    return sum(any(_hit(c, g) for g in gold) for c in cited) / len(cited)


def time_order_ok(example: dict[str, Any], ranks: list[int]) -> bool:
    schemas, _ = retrieval_catalog(example)
    starts = [float((schemas[r - 1].get("time_span") or {}).get("start_s") or 0) for r in ranks if 0 < r <= len(schemas)]
    return all(x <= y for x, y in zip(starts, starts[1:]))


def reader_reward(example: dict[str, Any], completion: str, gold_label: str, *, process_weight: float = 0.5,
                  max_chars: int = 2500) -> dict[str, Any]:
    label, reasoning = parse_reader_output(completion)
    correct = float(label is not None and str(label).strip().upper() == str(gold_label).strip().upper())
    fmt = float(label is not None and bool(reasoning))
    ranks = cited_ranks(reasoning)
    prec = citation_precision(example, ranks)
    length_pen = max(0.0, (len(completion) - max_chars) / max_chars)
    r = 1.0 * correct + process_weight * (prec or 0.0) * correct + 0.1 * fmt - 0.05 * length_pen
    return {"reward": r, "correct": bool(correct), "format": bool(fmt), "citation_precision": prec,
            "cited": len(ranks), "time_order_ok": time_order_ok(example, ranks) if ranks else None}
