#!/usr/bin/env python3
"""Paired Motif on/off QA diagnostic over frozen L1 examples.

Does not promote motifs; only measures attempt/fallback/accuracy deltas.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from motif import MotifRecord, MotifTransferAdapter, MotifTransferExample  # noqa: E402
from motif.transfer import MotifEvalResult  # noqa: E402


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _normalize_label(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, dict):
        for key in ("label", "answer", "text"):
            if value.get(key) is not None and str(value.get(key)).strip():
                # Prefer short MCQ labels over free text when both exist.
                if key in ("label", "answer"):
                    return str(value.get(key)).strip()
        text = value.get("text")
        return str(text).strip() if text is not None and str(text).strip() else None
    text = str(value).strip()
    return text or None


def _gold_label(example: dict[str, Any]) -> str | None:
    question = example.get("question") or {}
    for key in ("answer", "label", "gold_label"):
        label = _normalize_label(question.get(key))
        if label:
            return label
    hidden = (example.get("metadata") or {}).get("hidden") or {}
    return _normalize_label(hidden.get("answer") or hidden.get("label"))


def _eval_from_rollout(example: dict[str, Any], rollout: dict[str, Any]) -> MotifEvalResult:
    final = rollout.get("final_answer") or {}
    pred = str(final.get("label") or "").strip()
    gold = _gold_label(example)
    answer_correct = bool(gold is not None and pred and pred == gold)
    runtime = ((rollout.get("metadata") or {}).get("runtime_verifier") or {})
    verifier_passed = bool(runtime.get("passed"))
    pack = rollout.get("verified_evidence_pack") or {}
    evidence_valid = bool(pack.get("support_refs")) or bool(rollout.get("answer_support_chain"))
    summary = rollout.get("verifier_summary") or {}
    no_leakage = bool(summary.get("no_old_video_fact_leakage", True))
    return MotifEvalResult(
        answer_correct=answer_correct,
        verifier_passed=verifier_passed,
        evidence_valid=evidence_valid,
        no_hidden_leakage=no_leakage,
    )


def build_run_fn(
    *,
    rollout_fn,
    motif_bank_path: str | None,
):
    def run_fn(transfer_example: MotifTransferExample, motif: MotifRecord | None) -> MotifEvalResult:
        example = dict(transfer_example.payload)
        meta = dict(example.get("metadata") or {})
        meta["motif_enabled"] = motif is not None
        if motif_bank_path:
            meta["motif_bank_path"] = motif_bank_path
        if motif is not None:
            meta["forced_motif_id"] = motif.motif_id
        else:
            meta.pop("forced_motif_id", None)
        example["metadata"] = meta
        clue = meta.get("clue_memory_graph") or {}
        rollout = rollout_fn(example, clue)
        return _eval_from_rollout(example, rollout)

    return run_fn


def summarize_motif_logs(rollouts: list[dict[str, Any]]) -> dict[str, Any]:
    attempts = 0
    fallbacks = 0
    expansions = 0
    for rollout in rollouts:
        motif = ((rollout.get("metadata") or {}).get("motif_online") or {})
        if motif.get("motif_retrieval_attempted"):
            attempts += 1
        if motif.get("fallback_reason"):
            fallbacks += 1
        if motif.get("expansion_valid"):
            expansions += 1
    n = max(len(rollouts), 1)
    return {
        "n": len(rollouts),
        "retrieval_attempt_rate": attempts / n,
        "fallback_rate": fallbacks / n,
        "expansion_valid_rate": expansions / n,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--frozen-l1",
        nargs="+",
        required=True,
        help="One or more staged 04_l1_example.json paths",
    )
    parser.add_argument("--motif-bank", required=True, help="Motif bank JSONL")
    parser.add_argument("--motif-id", required=True, help="Motif id for paired evaluation")
    parser.add_argument("--output", default=None, help="Write JSON report")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not call LLM; synthesize motif_online logs for plumbing check",
    )
    args = parser.parse_args(argv)

    from motif import MotifBank

    bank = MotifBank.load_jsonl(args.motif_bank)
    motif = bank.require(args.motif_id)

    examples: list[MotifTransferExample] = []
    for path in args.frozen_l1:
        payload = _load_json(Path(path))
        examples.append(
            MotifTransferExample(
                dataset=str(payload.get("dataset") or ""),
                example_id=str(payload.get("example_id") or Path(path).stem),
                task_family=str(payload.get("task_family") or ""),
                payload=payload,
            )
        )

    if args.dry_run:
        def rollout_fn(example: dict[str, Any], clue: dict[str, Any]) -> dict[str, Any]:
            del clue
            enabled = bool((example.get("metadata") or {}).get("motif_enabled"))
            return {
                "final_answer": {"label": "A"},
                "acceptance_status": "accepted_strong",
                "verified_evidence_pack": {"support_refs": ["n1"]},
                "answer_support_chain": [{"evidence_refs": ["n1"]}],
                "verifier_summary": {"no_old_video_fact_leakage": True},
                "metadata": {
                    "runtime_verifier": {"passed": True},
                    "motif_online": {
                        "motif_retrieval_attempted": enabled,
                        "candidate_ids": [args.motif_id] if enabled else [],
                        "selected_motif_id": args.motif_id if enabled else None,
                        "bank_version": Path(args.motif_bank).name,
                        "expansion_valid": enabled,
                        "fallback_reason": None if enabled else "motif_disabled",
                        "downstream_verified_success": True,
                    },
                },
            }
    else:
        from dataset_clip_wrapper.l2_reasoning_graph.reasoning_planner import (  # noqa: E402
            build_llm_reasoning_rollout,
        )
        from dataset_clip_wrapper.perception.openrouter_client import (  # noqa: E402
            OpenRouterClient,
            load_openrouter_api_key,
        )

        api_key = load_openrouter_api_key(
            keys_py_path="/fs/gamma-projects/vlm-robot/keys.py"
        )
        client = OpenRouterClient(
            model="openai/gpt-oss-120b",
            api_key=api_key,
            max_tokens=1800,
            reasoning={"effort": "minimal", "exclude": True},
        )

        def rollout_fn(example: dict[str, Any], clue: dict[str, Any]) -> dict[str, Any]:
            return build_llm_reasoning_rollout(
                example,
                clue,
                client=client,
                skill_executor=None,
                motif_enabled=bool((example.get("metadata") or {}).get("motif_enabled")),
                motif_bank_path=args.motif_bank,
                forced_motif_id=(example.get("metadata") or {}).get("forced_motif_id"),
            )

    adapter = MotifTransferAdapter(build_run_fn(rollout_fn=rollout_fn, motif_bank_path=args.motif_bank))
    report = adapter.evaluate(motif, examples)
    # Collect synthetic motif logs via one forced pass for rates.
    logs = []
    for ex in examples:
        logs.append(rollout_fn({**ex.payload, "metadata": {**(ex.payload.get("metadata") or {}), "motif_enabled": True}}, {}))
    motif_rates = summarize_motif_logs(logs)
    out = {
        "transfer": report.to_dict(),
        "motif_online_rates": motif_rates,
        "n_examples": len(examples),
        "motif_id": args.motif_id,
        "dry_run": bool(args.dry_run),
    }
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
