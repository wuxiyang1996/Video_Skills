"""OPD KL/JSD warm-up over complete-action candidate sets.

Smoke mode validates distill rows and computes teacher/student KL with a
length-normalized character score proxy (no GPU). Full LoRA training can wrap
the same loss once a student scorer is plugged in.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .opd_action_distill_adapter import load_opd_rows

StudentScoreFn = Callable[[Mapping[str, Any], Sequence[Mapping[str, Any]]], list[float]]


def _softmax(scores: Sequence[float]) -> list[float]:
    if not scores:
        return []
    max_s = max(scores)
    exps = [math.exp(s - max_s) for s in scores]
    total = sum(exps) or 1.0
    return [e / total for e in exps]


def length_normalized_char_score(action: Mapping[str, Any]) -> float:
    """Cheap proxy score: prefer shorter valid JSON dumps (smoke only)."""
    blob = json.dumps(action, sort_keys=True, ensure_ascii=False)
    return -math.log(max(len(blob), 1))


def default_student_scores(
    state: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
) -> list[float]:
    del state
    return [length_normalized_char_score(c.get("action") or {}) for c in candidates]


def kl_divergence(p: Sequence[float], q: Sequence[float], eps: float = 1e-12) -> float:
    total = 0.0
    for pi, qi in zip(p, q):
        pi = max(float(pi), eps)
        qi = max(float(qi), eps)
        total += pi * math.log(pi / qi)
    return total


def jsd_divergence(p: Sequence[float], q: Sequence[float]) -> float:
    m = [(a + b) / 2.0 for a, b in zip(p, q)]
    return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)


def evaluate_opd_row(
    row: Mapping[str, Any],
    *,
    student_score_fn: StudentScoreFn = default_student_scores,
    sft_replay_mix: float = 0.1,
) -> dict[str, Any]:
    state = row.get("state") or {}
    cand_payload = row.get("candidates") or {}
    teacher = row.get("teacher") or {}
    candidates = list(cand_payload.get("candidates") or [])
    action_ids = [str(c.get("action_id")) for c in candidates]
    teacher_probs_map = teacher.get("action_probs") or {}
    teacher_probs = [float(teacher_probs_map.get(aid, 0.0)) for aid in action_ids]
    if sum(teacher_probs) <= 0:
        teacher_probs = [1.0 / max(len(action_ids), 1)] * len(action_ids)
    else:
        z = sum(teacher_probs)
        teacher_probs = [p / z for p in teacher_probs]

    student_scores = student_score_fn(state, candidates)
    student_probs = _softmax(student_scores)
    # Optional tiny uniform SFT-replay mix to avoid collapse in smoke.
    if sft_replay_mix > 0 and student_probs:
        u = 1.0 / len(student_probs)
        student_probs = [
            (1.0 - sft_replay_mix) * p + sft_replay_mix * u for p in student_probs
        ]
        z = sum(student_probs) or 1.0
        student_probs = [p / z for p in student_probs]

    return {
        "state_id": state.get("state_id") or cand_payload.get("state_id"),
        "n_candidates": len(candidates),
        "kl_teacher_to_student": kl_divergence(teacher_probs, student_probs),
        "jsd": jsd_divergence(teacher_probs, student_probs),
        "teacher_entropy": -sum(p * math.log(max(p, 1e-12)) for p in teacher_probs),
        "precheck_passed": bool((row.get("precheck") or {}).get("passed")),
    }


def run_opd_smoke(
    distill_path: str | Path,
    *,
    output_path: str | Path | None = None,
    sft_replay_mix: float = 0.1,
) -> dict[str, Any]:
    rows = load_opd_rows(distill_path)
    metrics = [
        evaluate_opd_row(row, sft_replay_mix=sft_replay_mix)
        for row in rows
    ]
    report = {
        "schema_version": "video-skills/opd-kl-smoke-v1",
        "n_rows": len(rows),
        "n_precheck_passed": sum(1 for m in metrics if m.get("precheck_passed")),
        "mean_kl": (
            sum(float(m["kl_teacher_to_student"]) for m in metrics) / len(metrics)
            if metrics
            else None
        ),
        "mean_jsd": (
            sum(float(m["jsd"]) for m in metrics) / len(metrics) if metrics else None
        ),
        "rows": metrics,
    }
    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="OPD KL smoke over distill JSONL")
    parser.add_argument("--distill", required=True, help="OPD distill JSONL path")
    parser.add_argument("--output", default=None, help="Write smoke report JSON")
    parser.add_argument("--sft-replay-mix", type=float, default=0.1)
    args = parser.parse_args(argv)
    report = run_opd_smoke(
        args.distill,
        output_path=args.output,
        sft_replay_mix=float(args.sft_replay_mix),
    )
    print(json.dumps({k: report[k] for k in ("n_rows", "n_precheck_passed", "mean_kl", "mean_jsd")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
