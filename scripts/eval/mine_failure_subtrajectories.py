"""Feasibility probe for atomic skills as reusable sub-trajectories learned from reasoning failures.

For every question the direct reader got wrong, fit the benchmark's own gold
reasoning (Video-Holmes: Explanation + Inference Shots; VRBench: the annotated
reasoning_process) to a FIXED skill ontology, together with the reader's wrong
rationale, and record (a) a failure code from a fixed list, (b) the gold
sub-trajectory as a short typed program over the ontology, (c) for every step,
what executing it would need: re-reading the text the reader already had, a
deterministic computation, looking at pixels, listening to dialogue, or outside
knowledge.  The aggregate answers the feasibility question before any GPU is
spent: do failures cluster into a few reusable sub-trajectories, and do those
sub-trajectories contain steps a text re-reader cannot perform?

The fitter may not invent skills (unknown ids are mapped to `other`).
"""
from __future__ import annotations

import argparse
import ast
import collections
import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from dataset_clip_wrapper.perception.openrouter_client import OpenRouterClient, load_openrouter_api_key

ONTOLOGY = [
    "parse_question_target", "propose_evidence_roles", "retrieve_by_event", "retrieve_by_entity", "retrieve_by_time",
    "retrieve_by_relation", "localize_clue", "extract_claim", "assign_evidence_role", "compose_evidence_chain",
    "detect_missing_role", "search_counterevidence", "infer_temporal_relation", "infer_state_change",
    "infer_causal_relation", "infer_intention_or_motive", "infer_social_contradiction", "verify_claim_support",
    "commit_answer", "resolve_entity_coreference", "extract_dialogue_span", "extract_observation", "count_events",
    "order_events_by_time", "compare_options_by_explanatory_power", "other",
]
FAILURE_CODES = [
    "missed_clue_not_in_text", "clue_in_text_but_ignored", "wrong_temporal_order", "identity_confusion",
    "dialogue_needed", "literal_over_explanatory_option", "film_grammar_or_symbolism", "over_reading",
    "causal_link_missing", "counting_or_quantity", "other",
]
NEEDS = ["text_reread", "deterministic", "look_pixels", "listen_dialogue", "outside_knowledge"]

SYSTEM = (
    "You analyse why a video-QA reader answered wrongly and express the GOLD reasoning as a short program over a "
    "fixed skill ontology. Use only the listed skill ids and failure codes. For each step say what executing it "
    "would require beyond the clip descriptions the reader already read: 'text_reread' (the fact is in the text, "
    "just re-read/re-organise), 'deterministic' (a computation such as ordering by timestamps or counting), "
    "'look_pixels' (must see frames), 'listen_dialogue' (must hear speech), 'outside_knowledge' (film grammar, "
    "world knowledge). Be strict: mark 'text_reread' only if the reader's own text contains the needed fact. "
    "Reply with JSON only: {\"failure_code\": ..., \"subtrajectory\": [{\"skill\": ..., \"needs\": ..., "
    "\"what\": \"<10 words>\"}], \"pattern\": \"<one-line reusable template of the gold reasoning>\"}"
)


def load_vh(test_json: Path, ann_dir: Path) -> dict[str, dict[str, Any]]:
    gold = {}
    for row in json.load(test_json.open()):
        eid = f"video_holmes:test:{row['video ID']}:q{row['Question ID']}"
        gold[eid] = {"question": row["Question"], "options": row["Options"], "answer": row["Answer"],
                     "type": row["Question Type"], "explanation": row.get("Explanation") or "", "video": row["video ID"]}
    for eid, g in gold.items():
        p = ann_dir / f"{g['video']}.json"
        if p.exists():
            try:
                ann = json.load(p.open())
                ann = ann[0] if isinstance(ann, list) else ann
                g["inference_shots"] = ann.get("Inference Shots") or ann.get("InferenceScenes") or []
            except Exception:
                g["inference_shots"] = []
    return gold


def load_vrbench(eval_jsonl: Path) -> dict[str, dict[str, Any]]:
    gold = {}
    for line in eval_jsonl.open():
        row = json.loads(line)
        for key, qa in (row.get("mcq") or {}).items():
            gold[f"vrbench:{row['video_id']}:{key}"] = {"question": qa.get("question"), "options": qa.get("options") or qa.get("choices"),
                                                       "answer": qa.get("answer"), "type": qa.get("reasoning_type") or qa.get("type"),
                                                       "explanation": qa.get("reasoning_process"), "video": row["video_id"]}
    return gold


def fit_one(client: Any, g: dict[str, Any], wrong: dict[str, Any]) -> dict[str, Any]:
    payload = {"question": g["question"], "options": g["options"], "gold_answer": g["answer"],
               "gold_reasoning": g.get("explanation"), "gold_inference_shots": (g.get("inference_shots") or [])[:6],
               "reader_answer": wrong.get("label"), "reader_rationale": (wrong.get("thinking") or "")[:2500],
               "skill_ontology": ONTOLOGY, "failure_codes": FAILURE_CODES, "needs_values": NEEDS}
    text = client.chat([{"role": "system", "content": SYSTEM}, {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}])
    try:
        out = json.loads(re.search(r"\{.*\}", text or "", re.S).group(0))
    except Exception:
        return {"failure_code": "other", "subtrajectory": [], "pattern": "", "parse_error": True}
    out["failure_code"] = out.get("failure_code") if out.get("failure_code") in FAILURE_CODES else "other"
    steps = []
    for s in out.get("subtrajectory") or []:
        if isinstance(s, dict):
            steps.append({"skill": s.get("skill") if s.get("skill") in ONTOLOGY else "other",
                          "needs": s.get("needs") if s.get("needs") in NEEDS else "text_reread", "what": str(s.get("what") or "")[:80]})
    out["subtrajectory"] = steps
    out["pattern"] = str(out.get("pattern") or "")[:200]
    return out


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(records)
    by_code = collections.Counter(r["fit"]["failure_code"] for r in records)
    by_seq = collections.Counter(" > ".join(s["skill"] for s in r["fit"]["subtrajectory"]) for r in records)
    steps = [s for r in records for s in r["fit"]["subtrajectory"]]
    needs = collections.Counter(s["needs"] for s in steps)
    non_text_q = sum(1 for r in records if any(s["needs"] != "text_reread" for s in r["fit"]["subtrajectory"]))
    deterministic_q = sum(1 for r in records if any(s["needs"] == "deterministic" for s in r["fit"]["subtrajectory"]))
    top_seq = by_seq.most_common(8)
    cover_top5 = sum(c for _, c in by_seq.most_common(5))
    return {"failures": n, "failure_codes": by_code.most_common(), "steps": len(steps), "step_needs": needs.most_common(),
            "failures_with_any_non_text_step": non_text_q, "failures_with_deterministic_step": deterministic_q,
            "distinct_subtrajectories": len(by_seq), "top5_subtrajectory_coverage": cover_top5, "top_subtrajectories": top_seq}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", choices=["vh", "vrbench"], required=True)
    ap.add_argument("--rollouts", type=Path, required=True, help="direct --rationale rollouts (thinking + final_answer)")
    ap.add_argument("--rows", type=Path, help="rows.jsonl with per-question correct flags (default: derive from rollouts vs gold)")
    ap.add_argument("--vh-test-json", type=Path, default=Path("/fs/gamma-projects/vlm-robot/datasets/Video-Holmes/Benchmark/test_Video-Holmes.json"))
    ap.add_argument("--vh-ann-dir", type=Path, default=Path("/fs/gamma-projects/vlm-robot/datasets/Video-Holmes/Benchmark/annotations"))
    ap.add_argument("--vrbench-eval", type=Path, default=Path("/fs/gamma-projects/vlm-robot/datasets/VRBench/VRBench_eval.jsonl"))
    ap.add_argument("--model", default="openai/gpt-oss-120b")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--keys-py", type=Path, default=Path("/fs/gamma-projects/vlm-robot/keys.py"))
    args = ap.parse_args(argv)

    gold = load_vh(args.vh_test_json, args.vh_ann_dir) if args.dataset == "vh" else load_vrbench(args.vrbench_eval)
    correct_flag = {}
    if args.rows:
        for line in args.rows.open():
            r = json.loads(line); correct_flag[r["example_id"]] = bool(r.get("correct"))
    wrong: list[tuple[str, dict[str, Any]]] = []
    for line in args.rollouts.open():
        rec = json.loads(line); eid = rec["example_id"]
        if eid not in gold:
            continue
        label = (rec["rollout"].get("final_answer") or {}).get("label")
        is_correct = correct_flag.get(eid, str(label) == str(rec.get("gold_label") or gold[eid]["answer"]))
        if not is_correct:
            wrong.append((eid, {"label": label, "thinking": rec["rollout"].get("thinking")}))
    if args.limit:
        wrong = wrong[: args.limit]
    client = OpenRouterClient(model=args.model, api_key=load_openrouter_api_key(keys_py_path=args.keys_py),
                              max_tokens=1200, temperature=0.0, reasoning={"effort": "low", "exclude": True}, timeout_s=240)
    done: dict[str, dict[str, Any]] = {}
    if args.output.exists():
        for line in args.output.open():
            r = json.loads(line); done[r["example_id"]] = r
    todo = [(e, w) for e, w in wrong if e not in done]

    def run(item):
        eid, w = item
        return {"example_id": eid, "type": gold[eid].get("type"), "reader_label": w["label"], "gold": gold[eid]["answer"],
                "fit": fit_one(client, gold[eid], w)}

    with ThreadPoolExecutor(max_workers=args.workers) as pool, args.output.open("a") as out:
        for i, rec in enumerate(pool.map(run, todo), 1):
            done[rec["example_id"]] = rec
            out.write(json.dumps(rec, ensure_ascii=False) + "\n"); out.flush()
            if i % 25 == 0:
                print(f"[{i}/{len(todo)}]", flush=True)
    records = [done[e] for e, _ in wrong if e in done]
    summary = summarize(records)
    by_type = {}
    for t in sorted({r["type"] for r in records}, key=str):
        by_type[str(t)] = summarize([r for r in records if r["type"] == t])
    summary["by_type"] = {t: {k: v for k, v in s.items() if k in ("failures", "failures_with_any_non_text_step",
                                                                    "failures_with_deterministic_step", "failure_codes")} for t, s in by_type.items()}
    args.output.with_suffix(".summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
