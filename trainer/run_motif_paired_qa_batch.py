#!/usr/bin/env python3
"""Batch Motif on/off QA over frozen L1 caches (real L2 rollouts)."""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset_clip_wrapper.l2_reasoning_graph.reasoning_planner import (  # noqa: E402
    build_llm_reasoning_rollout,
)
from dataset_clip_wrapper.perception.openrouter_client import (  # noqa: E402
    OpenRouterClient,
    load_openrouter_api_key,
)
from dataset_clip_wrapper.schemas import (  # noqa: E402
    SkillExecutionConfig,
    WrapperConfig,
)
from dataset_clip_wrapper.runners.llm_pipeline import _build_skill_executor  # noqa: E402
from motif import MotifBank  # noqa: E402
from trainer.run_motif_paired_qa import _gold_label, _normalize_label  # noqa: E402


def _pred_label(rollout: dict[str, Any]) -> str | None:
    return _normalize_label((rollout.get("final_answer") or {}).get("label"))


def _score_row(example: dict[str, Any], rollout: dict[str, Any]) -> dict[str, Any]:
    gold = _gold_label(example)
    pred = _pred_label(rollout)
    motif = ((rollout.get("metadata") or {}).get("motif_online") or {})
    runtime = ((rollout.get("metadata") or {}).get("runtime_verifier") or {})
    acceptance = str(rollout.get("acceptance_status") or "")
    return {
        "example_id": example.get("example_id"),
        "dataset": example.get("dataset"),
        "task_family": example.get("task_family"),
        "gold": gold,
        "pred": pred,
        "correct": bool(gold is not None and pred is not None and gold == pred),
        "acceptance_status": acceptance,
        "accepted_strong": acceptance.startswith("accepted_strong")
        or acceptance.startswith("resolved_strong"),
        "verifier_passed": bool(runtime.get("passed")),
        "motif_online": {
            "motif_retrieval_attempted": motif.get("motif_retrieval_attempted"),
            "selected_motif_id": motif.get("selected_motif_id"),
            "candidate_ids": motif.get("candidate_ids"),
            "expansion_valid": motif.get("expansion_valid"),
            "fallback_reason": motif.get("fallback_reason"),
            "downstream_verified_success": motif.get("downstream_verified_success"),
        },
        "planner": ((rollout.get("metadata") or {}).get("llm_plan") or {}).get("planner"),
    }


def _run_one(
    *,
    example: dict[str, Any],
    client: OpenRouterClient,
    skill_executor: Any | None,
    motif_enabled: bool,
    motif_bank_path: str,
    forced_motif_id: str | None,
    include_shadow: bool,
) -> dict[str, Any]:
    clue = ((example.get("metadata") or {}).get("clue_memory_graph") or {})
    meta = dict(example.get("metadata") or {})
    meta["motif_enabled"] = motif_enabled
    meta["motif_bank_path"] = motif_bank_path
    if forced_motif_id and motif_enabled:
        meta["forced_motif_id"] = forced_motif_id
    else:
        meta.pop("forced_motif_id", None)
    example = {**example, "metadata": meta}
    t0 = time.time()
    try:
        rollout = build_llm_reasoning_rollout(
            example,
            clue,
            client=client,
            skill_executor=skill_executor,
            motif_enabled=motif_enabled,
            motif_bank_path=motif_bank_path,
            forced_motif_id=forced_motif_id if motif_enabled else None,
            include_shadow_motifs=include_shadow,
        )
        row = _score_row(example, rollout)
        row["ok"] = True
        row["error"] = None
        row["forced_motif_id"] = forced_motif_id if motif_enabled else None
    except Exception as exc:
        row = {
            "example_id": example.get("example_id"),
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(limit=8),
            "correct": False,
            "accepted_strong": False,
            "verifier_passed": False,
            "motif_online": {"motif_retrieval_attempted": motif_enabled},
            "forced_motif_id": forced_motif_id if motif_enabled else None,
        }
    row["elapsed_s"] = round(time.time() - t0, 3)
    row["motif_enabled"] = motif_enabled
    return row


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows) or 1
    ok_rows = [r for r in rows if r.get("ok")]
    motif_ids = [
        ((r.get("motif_online") or {}).get("selected_motif_id"))
        for r in rows
        if ((r.get("motif_online") or {}).get("selected_motif_id"))
    ]
    return {
        "n": len(rows),
        "n_ok": len(ok_rows),
        "correct_rate": sum(1 for r in ok_rows if r.get("correct")) / max(len(ok_rows), 1),
        "accepted_strong_rate": sum(1 for r in ok_rows if r.get("accepted_strong"))
        / max(len(ok_rows), 1),
        "verifier_pass_rate": sum(1 for r in ok_rows if r.get("verifier_passed"))
        / max(len(ok_rows), 1),
        "retrieval_attempt_rate": sum(
            1 for r in rows if ((r.get("motif_online") or {}).get("motif_retrieval_attempted"))
        )
        / n,
        "expansion_valid_rate": sum(
            1 for r in rows if ((r.get("motif_online") or {}).get("expansion_valid"))
        )
        / n,
        "fallback_rate": sum(
            1
            for r in rows
            if ((r.get("motif_online") or {}).get("fallback_reason"))
            not in (None, "", "motif_disabled")
        )
        / n,
        "n_unique_motifs": len(set(motif_ids)),
        "motif_counts": dict(Counter(motif_ids)),
    }


def _resolve_paths(pattern: str, limit: int) -> list[Path]:
    from glob import glob

    paths = [Path(p) for p in sorted(glob(pattern, recursive=True))]
    if not paths and "/**/" in pattern:
        root_s, _, suffix = pattern.partition("/**/")
        root = Path(root_s)
        paths = sorted(root.rglob(suffix)) if root.exists() else []
    return paths[:limit]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-l1-glob", required=True, help="Glob of 04_l1_example.json")
    parser.add_argument("--motif-bank", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--model", default="openai/gpt-oss-120b")
    parser.add_argument("--skill-model", default="qwen/qwen3.5-9b")
    parser.add_argument("--keys-py", default="/fs/gamma-projects/vlm-robot/keys.py")
    parser.add_argument("--include-shadow-motifs", action="store_true")
    parser.add_argument(
        "--with-skill-executor",
        action="store_true",
        help="Use SkillExecutor on BOTH on/off arms (fair execution).",
    )
    parser.add_argument(
        "--skill-scope",
        default="verifier",
        choices=["all", "verifier"],
        help="LLM skill scope when --with-skill-executor (default verifier for cost).",
    )
    parser.add_argument(
        "--rotate-motifs",
        action="store_true",
        help="Force-rotate ACTIVE motifs from the bank across ON runs (>=3 ids).",
    )
    args = parser.parse_args(argv)

    paths = _resolve_paths(args.frozen_l1_glob, int(args.limit))
    if not paths:
        raise SystemExit(f"No frozen L1 files matched: {args.frozen_l1_glob}")

    bank = MotifBank.load_jsonl(args.motif_bank)
    motif_ids = [r.motif_id for r in bank.active_records()] or bank.motif_ids
    if args.rotate_motifs and len(motif_ids) < 3:
        raise SystemExit(f"Need >=3 active motifs for rotation; got {motif_ids}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    api_key = load_openrouter_api_key(keys_py_path=args.keys_py)
    client = OpenRouterClient(
        model=args.model,
        api_key=api_key,
        max_tokens=1800,
        reasoning={"effort": "minimal", "exclude": True},
        timeout_s=180,
    )

    skill_executor = None
    if args.with_skill_executor:
        # Minimal WrapperConfig carrying skill_execution only.
        config = WrapperConfig(
            dataset_root="/fs/gamma-projects/vlm-robot/datasets",
            dataset="cg_bench",  # type: ignore[arg-type]
            skill_execution=SkillExecutionConfig(
                skill_model=args.skill_model,
                llm_skill_scope=args.skill_scope,  # type: ignore[arg-type]
                enable_llm_skills=True,
                enable_vlm_skills=False,
            ),
        )
        skill_executor = _build_skill_executor(api_key, config)

    off_rows: list[dict[str, Any]] = []
    on_rows: list[dict[str, Any]] = []
    report: dict[str, Any] = {}
    for idx, path in enumerate(paths):
        example = json.loads(path.read_text(encoding="utf-8"))
        forced = motif_ids[idx % len(motif_ids)] if args.rotate_motifs else None

        print(f"[{idx+1}/{len(paths)}] OFF {example.get('example_id')}", flush=True)
        off = _run_one(
            example=example,
            client=client,
            skill_executor=skill_executor,
            motif_enabled=False,
            motif_bank_path=args.motif_bank,
            forced_motif_id=None,
            include_shadow=bool(args.include_shadow_motifs),
        )
        off["source_path"] = str(path)
        off_rows.append(off)
        print(
            f"  off correct={off.get('correct')} accept={off.get('acceptance_status')} "
            f"err={off.get('error')} t={off.get('elapsed_s')}",
            flush=True,
        )

        print(
            f"[{idx+1}/{len(paths)}] ON  {example.get('example_id')} forced={forced}",
            flush=True,
        )
        on = _run_one(
            example=example,
            client=client,
            skill_executor=skill_executor,
            motif_enabled=True,
            motif_bank_path=args.motif_bank,
            forced_motif_id=forced,
            include_shadow=bool(args.include_shadow_motifs),
        )
        on["source_path"] = str(path)
        on_rows.append(on)
        print(
            f"  on  correct={on.get('correct')} motif={((on.get('motif_online') or {}).get('selected_motif_id'))} "
            f"expand={((on.get('motif_online') or {}).get('expansion_valid'))} "
            f"fallback={((on.get('motif_online') or {}).get('fallback_reason'))} "
            f"err={on.get('error')} t={on.get('elapsed_s')}",
            flush=True,
        )

        report = {
            "motif_bank": args.motif_bank,
            "model": args.model,
            "skill_model": args.skill_model if args.with_skill_executor else None,
            "skill_scope": args.skill_scope if args.with_skill_executor else None,
            "with_skill_executor": bool(args.with_skill_executor),
            "rotate_motifs": bool(args.rotate_motifs),
            "motif_ids": motif_ids,
            "n_examples": len(paths),
            "completed_pairs": idx + 1,
            "off": _summarize(off_rows),
            "on": _summarize(on_rows),
            "delta_correct": _summarize(on_rows)["correct_rate"] - _summarize(off_rows)["correct_rate"],
            "off_rows": off_rows,
            "on_rows": on_rows,
        }
        (out_dir / "paired_qa_report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

    print(
        json.dumps(
            {
                "off": report["off"],
                "on": report["on"],
                "delta_correct": report["delta_correct"],
                "n_unique_motifs_on": report["on"]["n_unique_motifs"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
