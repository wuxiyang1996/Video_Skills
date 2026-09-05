#!/usr/bin/env python3
"""Measure the Video-Holmes answer chain on a reserved slice of the heldout set.

End-to-end accuracy on these benchmarks is (completion rate) x (accuracy when
answered), and an abstention scores as wrong.  On the cached GRPO rollouts the
system abstained 60.6% of the time while being right on 76.4% of what it did
answer -- but that cache is a biased sample, selected for reward variance over 25
examples.  This re-measures on heldout examples the policy never trained on.

Two retrieval conditions separate the answer chain from the retriever:
``model`` uses the ranking a checkpoint produced, ``oracle`` uses the candidates
that actually overlap the gold spans.  If accuracy barely moves between them,
retrieval quality is not what limits answers.

Only a seeded subsample is consumed so the rest of the heldout stays unread.
"""

from __future__ import annotations

import argparse
import collections
import glob
import base64
import json
import threading
import random
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from dataset_clip_wrapper.l2_reasoning_graph.reasoning_planner import build_llm_reasoning_rollout
from atomic_skills.skill_executor import SkillExecutor
from atomic_skills.skill_model_client import SkillModelClient
from dataset_clip_wrapper.perception.openrouter_client import OpenRouterClient
from trainer.grpo.live_rollout import _grpo_skill_backend_config, _GRPO_LLM_SKILLS
from trainer.closed_loop_harness import load_frozen_l1_examples
from trainer.grpo.l2_dataset_rewards import temporal_hit
from dataset_clip_wrapper.training.lexical_retrieval_baseline import BM25  # noqa: F401  (shared tokeniser deps)
from trainer.build_l2_dataset_opd import load_dataset_reward_supervision, supervision_key
from trainer.grpo.l2_dataset_rewards import _text as clip_schema_text
from trainer.grpo.train_l2_terminal_on_policy import filter_example_for_retrieval, retrieval_catalog


class _LazyExamples:
    """Mapping from example_id to a freshly loaded frozen L1 example."""

    def __init__(self, path_by_id: dict[str, Path]) -> None:
        self._paths = path_by_id

    def __contains__(self, example_id: object) -> bool:
        return example_id in self._paths

    def __iter__(self):
        return iter(self._paths)

    def __len__(self) -> int:
        return len(self._paths)

    def __getitem__(self, example_id: str) -> dict[str, Any]:
        loaded = load_frozen_l1_examples([self._paths[example_id]])
        if not loaded:
            raise KeyError(example_id)
        return loaded[0]


def _with_rate_limit_retry(call, *, attempts: int = 4, base_sleep_s: float = 10.0, sleep=None):
    """Retry a call on HTTP 429 with exponential backoff; re-raise anything else.

    A rate-limited rollout used to come back as an error row and drop out of the
    sample with no retry.  On a multi-hour run that silently shrinks n.
    """
    import time as _time

    sleep = sleep or _time.sleep
    for attempt in range(attempts):
        try:
            return call()
        except Exception as exc:  # noqa: BLE001 -- only 429 is retried below
            text = f"{type(exc).__name__}: {exc}"
            # Transient transport faults: rate limit, upstream 5xx, or a non-JSON
            # body (an HTML error page makes requests' .json() raise ValueError,
            # which escaped as an error row on 10 of 1,837 questions).
            transient = "429" in text or re.search(r"\b5\d\d\b", text) is not None or isinstance(exc, ValueError)
            if not transient or attempt == attempts - 1:
                raise
            sleep(base_sleep_s * (2 ** attempt))


def oracle_gold_spans(supervision: dict[str, Any]) -> list[dict[str, Any]]:
    """The spans an oracle may use: per-question clue intervals on CG-Bench,
    per-video Inference-Shot timestamps on Video-Holmes.

    Video-Holmes ``segment_spans`` are the Segment-Description rows and cover
    ~95% of a video, so 'overlaps a gold span' held for a median 52 of ~61
    clips and a first-hits cap returned the video's opening four clips for
    every question.  Those are never an oracle.
    """
    supervision = supervision or {}
    if supervision.get("dataset") == "cg_bench":
        return list(supervision.get("clue_spans") or [])
    return list(supervision.get("inference_spans") or [])


def oracle_indices(example: dict[str, Any], supervision: dict[str, Any], top_k: int) -> list[int]:
    """Candidates overlapping the oracle gold spans, most overlap first, up to top_k.

    Returns [] when the example has no usable gold (the caller records it as a
    missing-index error rather than silently falling back to another source).
    """
    schemas, _ = retrieval_catalog(example)
    gold = oracle_gold_spans(supervision)
    if not gold:
        return []
    scored = []
    for index, schema in enumerate(schemas):
        if not (isinstance(schema, dict) and isinstance(schema.get("time_span"), dict)):
            continue
        span = schema["time_span"]
        start, end = float(span.get("start_s") or 0.0), float(span.get("end_s") or 0.0)
        overlap = sum(
            max(0.0, min(end, float(g.get("end_s") or 0.0)) - max(start, float(g.get("start_s") or 0.0)))
            for g in gold
            if isinstance(g, dict)
        )
        hit = any(temporal_hit(span, g) for g in gold if isinstance(g, dict))
        if hit:
            # point timestamps have zero overlap seconds; keep the hit, rank by overlap
            scored.append((-overlap, index))
    scored.sort()
    return [index for _, index in scored[:top_k]]


def answer_model_budget(effort: str, max_tokens: int | None) -> int:
    """Completion budget: hidden reasoning is billed against max_tokens, so a
    larger effort needs headroom or the answer JSON never arrives."""
    if max_tokens is not None:
        return max_tokens
    return 1800 if effort == "minimal" else 8000


def sample_clip_frames(video_path: str | Path, span: dict[str, Any], count: int, width: int = 448) -> list[str]:
    """Evenly spaced JPEG frames (base64) from one clip span, downscaled to `width`."""
    if count <= 0 or not video_path:
        return []
    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise RuntimeError("frame sampling needs opencv-python") from exc
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []
    start = float(span.get("start_s") or 0.0)
    end = float(span.get("end_s") or start)
    if count == 1:
        times = [(start + end) / 2.0]
    else:
        step = max(end - start, 0.1) / (count - 1)
        times = [start + i * step for i in range(count)]
    frames: list[str] = []
    for t in times:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ok, frame = cap.read()
        if not ok:
            continue
        h, w = frame.shape[:2]
        if width and w > width:
            frame = cv2.resize(frame, (width, max(1, int(h * width / w))))
        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if ok:
            frames.append(base64.b64encode(buf.tobytes()).decode("ascii"))
    cap.release()
    return frames


def clip_frames_for_answer(
    example: dict[str, Any], indices: list[int], focus: list[int] | None, per_clip: int, max_clips: int, width: int = 448
) -> dict[int, list[str]]:
    """Frames for the clips the answer model should *watch*: the highlighted
    (pointer) clips when there are any, else the first `max_clips` retrieved."""
    if per_clip <= 0:
        return {}
    schemas, _ = retrieval_catalog(example)
    video_path = ((example.get("video") or {}).get("primary_path")) or ""
    chosen = [i for i in (focus or indices) if 0 <= i < len(schemas)][:max_clips]
    out: dict[int, list[str]] = {}
    for index in chosen:
        schema = schemas[index] if isinstance(schemas[index], dict) else {}
        span = schema.get("time_span") or {}
        frames = sample_clip_frames(video_path, span, per_clip, width)
        if frames:
            out[index] = frames
    return out


def multimodal_user_content(payload: dict[str, Any], indices: list[int], frames: dict[int, list[str]]) -> Any:
    """OpenAI-style content parts: the JSON payload, then each watched clip's frames."""
    if not frames:
        return json.dumps(payload, ensure_ascii=False)
    position = {index: rank for rank, index in enumerate(indices, start=1)}
    parts: list[dict[str, Any]] = [{"type": "text", "text": json.dumps(payload, ensure_ascii=False)}]
    for index, jpegs in frames.items():
        parts.append({"type": "text", "text": f"frames of clip rank {position.get(index, '?')}, in time order:"})
        for jpeg in jpegs:
            parts.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{jpeg}"}})
    return parts


def majority_label(labels: list[str | None]) -> str | None:
    """Most frequent non-empty label; ties go to the earliest sample."""
    counts: dict[str, int] = {}
    for label in labels:
        if label:
            counts[label] = counts.get(label, 0) + 1
    if not counts:
        return None
    best = max(counts.values())
    for label in labels:
        if label and counts[label] == best:
            return label
    return None


def strip_verdicts(findings: dict[str, Any]) -> dict[str, Any]:
    """Drop the graph's per-option scores and its vote, keep the observations.

    With scores and a vote attached, the notes drag the answer toward the
    graph's own (mostly wrong) pick: hybrid scored -8.7 [-16.0, -1.3] against
    plain direct answering over the same clips.  This variant tests whether the
    per-option *verdicts* are the poison or the analysis itself is worthless.
    """
    notes = []
    for note in findings.get("notes") or []:
        text = str(note.get("note") or "").strip()
        if text:
            notes.append({"observation": text})
    return {"notes": notes, "vote": None}


def extract_findings(rollout: dict[str, Any], max_chars: int = 6000) -> dict[str, Any]:
    """Turn a skill-graph rollout into notes a single answer call can read.

    The graph scores each option separately against a few observations and
    picks the max, which is where it loses to direct answering (3/24 vs ~10/24
    on the same evidence).  As *notes* next to the whole catalog the same
    per-option reasoning can only add information; the answer call still
    weighs the options against each other over all the clips.
    """
    notes: list[dict[str, Any]] = []
    used = 0
    diagnostics = ((rollout.get("metadata") or {}).get("answer_step_diagnostics") or {})
    for step, payload in diagnostics.items():
        if not isinstance(payload, dict):
            continue
        for key in ("scored_hypothesis", "best_hypothesis"):
            item = payload.get(key)
            if not isinstance(item, dict) or not item.get("llm_reasoning"):
                continue
            note = {
                "step": step,
                "option_label": item.get("option_label"),
                "support_score": item.get("support_score"),
                "contradiction_score": item.get("contradiction_score"),
                "note": str(item["llm_reasoning"])[:800],
            }
            size = len(json.dumps(note, ensure_ascii=False))
            if used + size > max_chars:
                break
            notes.append(note)
            used += size
    final = rollout.get("final_answer") or {}
    vote = {
        "label": final.get("label") if isinstance(final, dict) else None,
        "confidence": final.get("confidence") if isinstance(final, dict) else None,
        "acceptance_status": rollout.get("acceptance_status"),
    }
    return {"notes": notes, "vote": vote}


def dump_rollout(handle: Any, lock: Any, record: dict[str, Any]) -> None:
    """Append one full rollout as a JSON line; workers share the handle under a lock."""
    line = json.dumps(record, ensure_ascii=False, default=str)
    with lock:
        handle.write(line + "\n")
        handle.flush()


def control_indices(example: dict[str, Any], source: str) -> list[int]:
    """The two retrieval controls: 'none' = no clips at all, 'all' = the whole catalog.

    Together with 'oracle' they bracket what retrieval can contribute: if 'none'
    is close to the system, the clip descriptions are not carrying the answer; if
    'all' is well above the system, recall at top-k is the limit.
    """
    if source == "none":
        return []
    if source == "all":
        schemas, _ = retrieval_catalog(example)
        return list(range(len(schemas)))
    raise ValueError(f"not a control source: {source}")


def model_indices(example_id: str, rankings: dict[str, list[int]], top_k: int) -> list[int]:
    return (rankings.get(example_id) or [])[:top_k]


def bm25_indices(example: dict[str, Any], top_k: int) -> list[int]:
    """Question-aware retrieval with no learning: BM25 of the question against
    each clip's generic caption.  Used when no reranker ranking exists for an
    example (derived sibling questions are absent from the eval report), so the
    run covers the whole benchmark instead of silently skipping them."""
    from dataset_clip_wrapper.training.lexical_retrieval_baseline import BM25, question_text, tokenize

    schemas, _ = retrieval_catalog(example)
    docs = [tokenize(clip_schema_text(s if isinstance(s, dict) else {})) for s in schemas]
    query = tokenize(question_text({"question": example.get("question") or {}}))
    if not docs or not query:
        return []
    bm25 = BM25(docs)
    order = sorted(range(len(docs)), key=lambda i: (-bm25.score(i, query), i))
    return order[:top_k] if top_k > 0 else order


def apply_temporal_nms(example: dict[str, Any], ranked: list[int], top_k: int) -> list[int]:
    """Greedy top-k over an already-ranked list, skipping candidates that overlap a
    chosen pick.  No free parameter; on Video-Holmes dev it lifted segment_recall
    for every reranker (OPD 59.76 -> 65.32) with precision flat."""
    from dataset_clip_wrapper.training.evaluate_l2_pointwise_adapter import _topk_temporal_nms

    schemas, _ = retrieval_catalog(example)
    spans = {
        i: s["time_span"] for i, s in enumerate(schemas)
        if isinstance(s, dict) and isinstance(s.get("time_span"), dict)
    }
    return _topk_temporal_nms(ranked, spans, top_k=top_k)


def retrieval_rank_indices(example: dict[str, Any], top_k: int) -> list[int]:
    """Question-blind visual-retrieval order stored on the catalog."""
    schemas, _ = retrieval_catalog(example)
    ranked = sorted(
        range(len(schemas)),
        key=lambda i: (int((schemas[i] or {}).get("retrieval_rank") or 10**6) if isinstance(schemas[i], dict) else 10**6, i),
    )
    return ranked[:top_k]


DIRECT_SYSTEM_MULTIMODAL = (
    "You answer multiple-choice questions about a video using the clip descriptions "
    "provided and the attached frames of the clips most likely to matter. Reply with JSON only."
)
DIRECT_SYSTEM = (
    "You answer multiple-choice questions about a video using only the clip "
    "descriptions provided. Reply with JSON only."
)


def direct_answer(
    client: OpenRouterClient,
    example: dict[str, Any],
    indices: list[int],
    highlight: list[int] | None = None,
    findings: dict[str, Any] | None = None,
    frames: dict[int, list[str]] | None = None,
) -> dict[str, Any]:
    """Ask the same model the same question over the same clips, with no skill graph.

    This is the control for the atomic-skill decomposition: it holds the model,
    the evidence and the budget fixed and removes only the plan-and-execute
    structure, so any difference is attributable to the decomposition itself.
    """
    schemas, _ = retrieval_catalog(example)
    question = example.get("question") or {}
    clips = []
    for rank, index in enumerate(indices, start=1):
        if not (0 <= index < len(schemas)):
            continue
        schema = schemas[index] if isinstance(schemas[index], dict) else {}
        span = schema.get("time_span") or {}
        clips.append({
            "rank": rank,
            "time_span": span,
            "description": clip_schema_text(schema)[:1200],
        })
    payload = {
        "question": question.get("question_text") or "",
        "options": [
            {"label": o.get("label"), "text": o.get("text")}
            for o in question.get("options") or []
        ],
        "clips": clips,
        "answer_format": 'reply exactly {"label": "<option letter>"}',
    }
    if highlight:
        # Ranks (1-based positions in `clips`) the retriever flagged as most likely
        # to matter -- the whole catalog is still given, so retrieval can only
        # steer attention, never remove evidence.
        position = {index: rank for rank, index in enumerate(indices, start=1)}
        payload["likely_key_clips"] = [position[i] for i in highlight if i in position]
        payload["likely_key_clips_note"] = (
            "ranks of clips a retriever flagged as most likely to matter; it may be wrong"
        )
    if findings:
        payload["skill_findings"] = findings.get("notes") or []
        if findings.get("vote"):
            payload["graph_vote"] = findings["vote"]
        payload["skill_findings_note"] = (
            "notes from an analysis pass by atomic skills over the same clips, one per "
            "candidate option, plus that pass's own vote; they may be wrong -- decide "
            "from the clips"
        )
    system = DIRECT_SYSTEM
    if frames:
        payload["watched_clips_note"] = (
            "frames of the clips listed after this JSON are attached; use them together "
            "with the descriptions"
        )
        system = DIRECT_SYSTEM_MULTIMODAL
    text = client.chat([
        {"role": "system", "content": system},
        {"role": "user", "content": multimodal_user_content(payload, indices, frames or {})},
    ])
    label = None
    try:
        label = (json.loads(re.search(r"\{.*\}", text, re.S).group(0)) or {}).get("label")
    except Exception:
        match = re.search(r"\b([A-H])\b", text or "")
        label = match.group(1) if match else None
    return {
        "final_answer": {"label": label},
        "failure_reasons": [] if label else ["no_label_parsed"],
        "frames_attached": sum(len(v) for v in (frames or {}).values()),
    }


def degradation(rollout: dict[str, Any]) -> dict[str, Any]:
    """Markers for the two silent fallbacks: planner -> deterministic, skill -> rule.

    Both leave ok=True and errors=0.  The planner fallback shows only as
    llm_plan.fallback_reason; an LLM-executed skill shows only as
    metadata.answer_step_diagnostics[<step_id>].backend == "llm" (rollout nodes
    carry no backend field), and a skill that fell back to its rule carries a
    `*_fallback_to_rule` message there.  A rollout whose planner fell back is not
    a measurement of the system and is excluded from accuracy.
    """
    meta = rollout.get("metadata") or {}
    llm_plan = meta.get("llm_plan") or {}
    planner_fell_back = bool(llm_plan.get("fallback_reason") or llm_plan.get("planner_error"))
    diagnostics = meta.get("answer_step_diagnostics") or {}

    def step_backend(step: dict[str, Any]) -> str | None:
        if step.get("backend"):
            return str(step["backend"])
        for key in ("scored_hypothesis", "best_hypothesis", "verified_claim"):
            nested = step.get(key)
            if isinstance(nested, dict) and nested.get("backend"):
                return str(nested["backend"])
        return None

    llm_nodes = 0
    critical_on_llm = 0
    rule_fallbacks = 0
    for node in rollout.get("nodes") or []:
        step = diagnostics.get(str(node.get("step_id") or ""))
        if not isinstance(step, dict):
            continue
        backend = step_backend(step)
        messages = " ".join(str(m) for m in (step.get("messages") or []))
        if backend == "llm":
            llm_nodes += 1
            if node.get("skill_id") in _GRPO_LLM_SKILLS:
                critical_on_llm += 1
        if "fallback_to_rule" in messages:
            rule_fallbacks += 1
    return {
        "planner_fell_back": planner_fell_back,
        "llm_skill_nodes": llm_nodes,
        "critical_skills_on_llm": critical_on_llm,
        "rule_fallbacks": rule_fallbacks,
    }


def score(rollout: dict[str, Any], gold_label: str) -> dict[str, Any]:
    label = (rollout.get("final_answer") or {}).get("label")
    committed = bool(label)
    correct = committed and str(label).strip().upper() == str(gold_label).strip().upper()
    return {
        "committed": committed,
        "correct": bool(correct),
        "acceptance_status": rollout.get("acceptance_status"),
        "failure_reasons": list(rollout.get("failure_reasons") or [])[:3],
        **degradation(rollout),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--l1-glob", required=True)
    parser.add_argument("--eval-report", type=Path, help="ranking source for the 'model' condition")
    parser.add_argument(
        "--indices-from",
        choices=("report", "bm25", "retrieval_rank", "oracle", "none", "all"),
        default="report",
        help=(
            "Where the retrieved clip indices come from.  'none' passes NO clips (question "
            "and options only -- the no-evidence control, 'direct' condition only); 'all' "
            "passes the whole catalog (the no-retrieval control); 'report' uses the reranker "
            "ranking in --eval-report and SKIPS examples it does not cover; 'bm25' ranks "
            "each example's own captions against its question (no learning, covers "
            "every example); 'retrieval_rank' uses the catalog's question-blind order; "
            "'oracle' uses the candidates overlapping the gold spans -- a retrieval CEILING, "
            "never a system result."
        ),
    )
    parser.add_argument(
        "--highlight-from",
        choices=("report", "bm25", "retrieval_rank", "oracle"),
        help=(
            "Direct condition only: also flag the top-k clips from this source as "
            "'likely_key_clips' inside the prompt.  Meant with --indices-from all, so the "
            "retriever steers attention over the whole catalog instead of cutting it."
        ),
    )
    parser.add_argument("--findings-mode", choices=("full", "observations_only"), default="full",
                        help="hybrid: 'full' passes the skills' per-option scores and vote; "
                             "'observations_only' passes just their observations.")
    parser.add_argument("--answer-model", default=None,
                        help="Model for the direct/hybrid answer call when different from --planner-model (e.g. a VLM).")
    parser.add_argument("--frames-per-clip", type=int, default=0,
                        help="Attach this many frames per watched clip to the answer call (0 = text only).")
    parser.add_argument("--frame-max-clips", type=int, default=4,
                        help="Watch at most this many clips: the --highlight-from picks, else the first retrieved.")
    parser.add_argument("--frame-width", type=int, default=448)
    parser.add_argument("--votes", type=int, default=1,
                        help="Self-consistency: sample the answer call this many times at --vote-temperature and take the majority.")
    parser.add_argument("--vote-temperature", type=float, default=0.7)
    parser.add_argument("--dump-rollouts", type=Path,
                        help="Also append every full rollout (plan, skill outputs, commit) to this jsonl for diagnosis.")
    parser.add_argument("--sample", type=int, default=40)
    parser.add_argument("--example-ids", type=Path,
                        help="Newline-separated example ids to run instead of a seeded sample (for re-running a hung tail).")
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--planner-model", default="openai/gpt-oss-120b")
    parser.add_argument(
        "--reasoning-effort",
        choices=("minimal", "low", "medium", "high"),
        default="minimal",
        help=(
            "Hidden-reasoning budget for the planner/direct model on OpenRouter.  The "
            "default 'minimal' is what every number so far was measured with; Video-Holmes "
            "is a reasoning benchmark, so this is a lever on the answer model itself."
        ),
    )
    parser.add_argument("--max-tokens", type=int, default=None,
                        help="Completion budget (reasoning tokens count against it); default 1800 for minimal, 8000 otherwise.")
    parser.add_argument("--skill-model", default="qwen/qwen3.5-9b",
                        help="LLM that executes answer-critical skills; mirrors the GRPO trainer's --skill-model.")
    parser.add_argument("--skill-timeout-s", type=int, default=90)
    parser.add_argument("--no-skill-executor", action="store_true",
                        help="Run skills as rules only (the degraded mode this script used to run in silently).")
    parser.add_argument("--keys-py", type=Path, default=Path("/fs/gamma-projects/vlm-robot/keys.py"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout-s", type=int, default=240)
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=["model", "oracle", "direct"],
        help=("'model'/'oracle' run the skill graph; 'direct' is the no-decomposition control; "
              "'hybrid' runs the graph, then answers once over all the clips plus the graph's notes."),
    )
    parser.add_argument(
        "--always-commit-mcq",
        action="store_true",
        help="Commit the best hypothesis even when verification fails (abstention scores as wrong on MCQ).",
    )
    parser.add_argument(
        "--temporal-nms",
        action="store_true",
        help="Apply temporal non-max suppression when cutting the retrieval ranking to top-k (bm25/report/retrieval_rank sources).",
    )
    parser.add_argument("--dataset-root", type=Path, default=Path("/fs/gamma-projects/vlm-robot/datasets"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.highlight_from and any(c not in ("direct", "hybrid") for c in args.conditions):
        parser.error("--highlight-from only applies to the direct and hybrid conditions")
    if args.indices_from == "none" and any(c != "direct" for c in args.conditions):
        parser.error("--indices-from none is the no-evidence control; it only makes sense with --conditions direct")

    from trainer.grpo.train_l2_terminal_on_policy import load_openrouter_api_key

    # Index example ids to paths without holding every example resident: each
    # frozen L1 carries a ~2,600-node clue graph, and loading all 1,837 kept
    # ~25 GB per process on a shared login node.  Examples are re-read per job.
    paths = [Path(p) for p in sorted(glob.glob(args.l1_glob, recursive=True))]
    path_by_id: dict[str, Path] = {}
    for path in paths:
        try:
            head = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        example_id = str(head.get("example_id") or "")
        if example_id and example_id not in path_by_id:
            path_by_id[example_id] = path
        del head
    examples = _LazyExamples(path_by_id)
    if args.example_ids:
        wanted = [line.strip() for line in args.example_ids.read_text(encoding="utf-8").splitlines() if line.strip()]
        chosen = [e for e in wanted if e in examples]
    else:
        chosen = sorted(path_by_id)
        random.Random(args.seed).shuffle(chosen)
        chosen = chosen[: max(1, args.sample)]

    # Hidden supervision is loaded evaluator-side; it never enters a prompt.
    supervision_index = load_dataset_reward_supervision(args.dataset_root)

    rankings: dict[str, list[int]] = {}
    if args.eval_report and args.eval_report.exists():
        report = json.loads(args.eval_report.read_text(encoding="utf-8"))
        for result in report.get("results") or []:
            rankings[str(result["example_id"])] = [
                int(row["candidate_index"])
                for row in sorted(
                    result["ranking"], key=lambda r: (-float(r["score"]), int(r["candidate_index"]))
                )
            ]

    client = OpenRouterClient(
        model=args.planner_model,
        api_key=load_openrouter_api_key(keys_py_path=args.keys_py),
        max_tokens=answer_model_budget(args.reasoning_effort, args.max_tokens),
        temperature=0.0,
        reasoning={"effort": args.reasoning_effort, "exclude": True},
        timeout_s=args.timeout_s,
    )

    # The trainer executes answer-critical skills with a separate LLM; without it
    # every skill degrades to a lexical rule and the chain scores below chance.
    executor = None
    if not args.no_skill_executor:
        executor = SkillExecutor(
            llm_client=SkillModelClient(
                model=args.skill_model,
                api_key=load_openrouter_api_key(keys_py_path=args.keys_py),
                max_tokens=768,
                temperature=0.0,
                timeout_s=args.skill_timeout_s,
            ),
            vlm_client=None,
            config=_grpo_skill_backend_config(),
        )

    answer_client = client
    if args.answer_model or args.votes > 1:
        answer_client = OpenRouterClient(
            model=args.answer_model or args.planner_model,
            api_key=load_openrouter_api_key(keys_py_path=args.keys_py),
            max_tokens=answer_model_budget(args.reasoning_effort, args.max_tokens),
            temperature=args.vote_temperature if args.votes > 1 else 0.0,
            reasoning={"effort": args.reasoning_effort, "exclude": True},
            timeout_s=args.timeout_s,
        )

    def answer(example: dict[str, Any], indices: list[int], highlight: list[int] | None, findings: dict[str, Any] | None = None) -> dict[str, Any]:
        frames = clip_frames_for_answer(example, indices, highlight, args.frames_per_clip, args.frame_max_clips, args.frame_width)
        if args.votes <= 1:
            return direct_answer(answer_client, example, indices, highlight, findings, frames)
        samples = [direct_answer(answer_client, example, indices, highlight, findings, frames) for _ in range(args.votes)]
        label = majority_label([(s.get("final_answer") or {}).get("label") for s in samples])
        return {
            "final_answer": {"label": label},
            "failure_reasons": [] if label else ["no_label_parsed"],
            "frames_attached": samples[0].get("frames_attached", 0),
            "votes": [(s.get("final_answer") or {}).get("label") for s in samples],
        }

    dump_lock = threading.Lock()
    dump_handle = None
    if args.dump_rollouts:
        args.dump_rollouts.parent.mkdir(parents=True, exist_ok=True)
        dump_handle = args.dump_rollouts.open("w", encoding="utf-8")

    def run(example_id: str, condition: str) -> dict[str, Any] | None:
        example = examples[example_id]
        question = example.get("question") or {}
        gold_label = (question.get("answer") or {}).get("label")
        if not gold_label:
            return None
        supervision = supervision_index.get(supervision_key(example)) or {}

        def pick(source: str) -> list[int]:
            if source in ("none", "all"):
                return control_indices(example, source)
            if condition == "oracle" or source == "oracle":
                return oracle_indices(example, supervision, args.top_k)
            slate = 0 if args.temporal_nms else args.top_k   # 0 -> full ranking
            if source == "bm25":
                ranked = bm25_indices(example, slate)
            elif source == "retrieval_rank":
                ranked = retrieval_rank_indices(example, slate) if slate else retrieval_rank_indices(example, 10**6)
            else:
                ranked = model_indices(example_id, rankings, slate) if slate else (rankings.get(example_id) or [])
            return apply_temporal_nms(example, ranked, args.top_k) if args.temporal_nms else ranked[: args.top_k]

        indices = pick(args.indices_from)
        highlight = pick(args.highlight_from) if args.highlight_from else None
        if not indices and args.indices_from != "none":
            # Skipping silently hid 1,574 of 1,837 questions once; count it.
            return {"example_id": example_id, "condition": condition, "error": "no_retrieval_indices"}
        try:
            graph_rollout = None
            if condition == "direct":
                rollout = _with_rate_limit_retry(lambda: answer(example, indices, highlight))
            else:
                isolated, graph = filter_example_for_retrieval(example, indices)
                rollout = _with_rate_limit_retry(lambda: build_llm_reasoning_rollout(
                    isolated, graph, client=client, skill_executor=executor, motif_enabled=False,
                    commit_policy={"always_commit_mcq": True} if args.always_commit_mcq else None,
                ))
                if condition == "hybrid":
                    # skills as an analysis pass; one answer call over all the evidence
                    graph_rollout = rollout
                    findings = extract_findings(graph_rollout)
                    if args.findings_mode == "observations_only":
                        findings = strip_verdicts(findings)
                    rollout = _with_rate_limit_retry(lambda: answer(example, indices, highlight, findings))
        except Exception as exc:  # a transport failure is not an abstention
            return {"example_id": example_id, "condition": condition, "error": type(exc).__name__}
        if dump_handle is not None:
            dump_rollout(dump_handle, dump_lock, {
                "example_id": example_id, "condition": condition, "gold_label": gold_label,
                "indices": indices, "rollout": graph_rollout if graph_rollout is not None else rollout,
            })
        row = {"example_id": example_id, "condition": condition, **score(rollout, gold_label)}
        if "frames_attached" in rollout:
            row["frames_attached"] = rollout["frames_attached"]
        if "votes" in rollout:
            row["votes"] = rollout["votes"]
        if graph_rollout is not None:
            graph_scored = score(graph_rollout, gold_label)
            row.update({
                "graph_committed": graph_scored["committed"],
                "graph_correct": graph_scored["correct"],
                "graph_acceptance_status": graph_scored["acceptance_status"],
                "planner_fell_back": graph_scored["planner_fell_back"],
                "critical_skills_on_llm": graph_scored["critical_skills_on_llm"],
                "skill_notes": len(extract_findings(graph_rollout)["notes"]),
            })
        return row

    rows: list[dict[str, Any]] = []
    jobs = [(e, c) for e in chosen for c in args.conditions]
    # Rows are appended to a sidecar as they finish.  A rollout that hangs on a
    # slow API call must not forfeit the ones already completed: the final JSON
    # is only written at the end, and one full run was lost that way.
    sidecar = args.output.with_suffix(".rows.jsonl")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sidecar.write_text("", encoding="utf-8")
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(run, e, c): (e, c) for e, c in jobs}
        for done in as_completed(futures):
            row = done.result()
            if row:
                rows.append(row)
                with sidecar.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row) + "\n")
            print(f"[{len(rows)}/{len(jobs)}]", flush=True)

    summary: dict[str, Any] = {}
    for condition in args.conditions:
        all_rows = [r for r in rows if r.get("condition") == condition and "error" not in r]
        degraded = [r for r in all_rows if r.get("planner_fell_back")]
        subset = [r for r in all_rows if not r.get("planner_fell_back")]
        n = max(1, len(subset))
        skill_rows = [r for r in subset if condition != "direct"]
        no_llm_skills = sum(1 for r in skill_rows if r.get("critical_skills_on_llm", 0) == 0)
        committed = sum(bool(r["committed"]) for r in subset)
        correct = sum(bool(r["correct"]) for r in subset)
        reasons: collections.Counter[str] = collections.Counter()
        for r in subset:
            if not r["committed"]:
                reasons.update(str(x)[:60] for x in r["failure_reasons"])
        summary[condition] = {
            "examples": len(subset),
            "excluded_planner_fallback": len(degraded),
            "rows_with_no_llm_critical_skill": no_llm_skills,
            "rows_with_rule_fallbacks": sum(1 for r in skill_rows if r.get("rule_fallbacks", 0) > 0),
            "errors": sum(1 for r in rows if r.get("condition") == condition and "error" in r),
            "completion_rate": 100.0 * committed / n,
            "accuracy_end_to_end": 100.0 * correct / n,
            "accuracy_when_answered": (100.0 * correct / committed) if committed else 0.0,
            "top_abstention_reasons": reasons.most_common(3),
        }
    invalid = [
        c for c, v in summary.items()
        if v["excluded_planner_fallback"] or (c != "direct" and not args.no_skill_executor and v["rows_with_no_llm_critical_skill"] > 0)
    ]
    if invalid:
        print(f"WARNING: degraded execution detected in conditions {invalid}; see excluded_planner_fallback / rows_with_no_llm_critical_skill", flush=True)
    payload = {
        "schema_version": "video-skills/answer-chain-heldout-v2",
        "valid": not invalid,
        "skill_model": None if args.no_skill_executor else args.skill_model,
        "sample": len(chosen),
        "seed": args.seed,
        "top_k": args.top_k,
        "planner_model": args.planner_model,
        "always_commit_mcq": bool(args.always_commit_mcq),
        "indices_from": args.indices_from,
        "highlight_from": args.highlight_from,
        "reasoning_effort": args.reasoning_effort,
        "answer_model": args.answer_model or args.planner_model,
        "frames_per_clip": args.frames_per_clip,
        "frame_max_clips": args.frame_max_clips,
        "votes": args.votes,
        "findings_mode": args.findings_mode,
        "max_tokens": answer_model_budget(args.reasoning_effort, args.max_tokens),
        "temporal_nms": bool(args.temporal_nms),
        "note": "seeded subsample of heldout_test; the rest stays unread",
        "conditions": summary,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in payload.items() if k != "rows"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
