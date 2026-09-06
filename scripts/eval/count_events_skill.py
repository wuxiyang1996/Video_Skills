"""A deterministic atomic skill for counting questions: enumerate, then count, then map to the option.

VRBench "Counting Problems" are a fifth of the benchmark and the direct reader
answers 32% of them (71-93% on every other type).  Counting is exactly the
kind of step an LLM does unreliably in one pass over a long catalog and a
program does reliably: the model is only asked to *enumerate* instances of the
target inside small, time-ordered chunks (with the running list in view so it
can mark repeats); the count itself, the de-duplication by time and identity,
and the mapping of the count onto the answer options are code.

Modes on the same enumeration:
  count      the option whose number equals the merged count (nearest if none matches;
             LLM fallback only when no option carries a number)
  assist     the direct reader answers with the enumerated, time-stamped list added as evidence

Per-question output rows are compatible with measure_answer_chain rows (example_id, correct, ...).
"""
from __future__ import annotations

import argparse
import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from dataset_clip_wrapper.perception.openrouter_client import OpenRouterClient, load_openrouter_api_key
from scripts.eval.measure_answer_chain import clip_schema_text, retrieval_catalog

PARSE_SYSTEM = (
    "You read a counting question about a video and say precisely what must be counted. Reply with JSON only: "
    '{"target": "<what one instance is, in a few words>", "unit": "occurrences|distinct_entities|distinct_scenes|other", '
    '"identity_key": "<what makes two instances the same, e.g. same person, same location, same event>"}'
)
ENUM_SYSTEM = (
    "You enumerate instances of a target inside a time-ordered chunk of video descriptions. You are given the "
    "target, what makes two instances the same, the instances already found in earlier chunks, and the chunk. "
    "List every instance visible in THIS chunk with its time span and a short identity descriptor; if an instance "
    "is the same as one already found (a continuation, or the same entity/scene), say so by giving its id. Do not "
    "count; do not answer the question. Reply with JSON only: "
    '{"instances": [{"start_s": <num>, "end_s": <num>, "identity": "<descriptor>", "same_as": "<id or null>", "evidence": "<10 words>"}]}'
)
ANSWER_SYSTEM = (
    "Answer a multiple-choice counting question about a video. You are given the question, the options, and an "
    "enumerated, time-stamped list of the instances found in the video with their identities. Reply with JSON "
    'only: {"label": "<option letter>", "count_used": <int>, "reason": "<one sentence>"}'
)

_WORDS = {"zero": 0, "no": 0, "none": 0, "once": 1, "one": 1, "single": 1, "a single": 1, "twice": 2, "two": 2, "three": 3,
          "thrice": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12}


def option_number(text: str) -> int | None:
    """The count an option states ('three times', 'twice', 'There are 4 shots'), or None."""
    t = str(text or "").lower()
    m = re.search(r"\b(\d{1,3})\b", t)
    if m:
        return int(m.group(1))
    for word in sorted(_WORDS, key=len, reverse=True):
        if re.search(rf"\b{re.escape(word)}\b", t):
            return _WORDS[word]
    return None


def merge_instances(instances: list[dict[str, Any]], gap_s: float = 15.0) -> list[dict[str, Any]]:
    """Deterministic de-duplication: explicit same_as links, then time adjacency of the same identity."""
    kept: list[dict[str, Any]] = []
    ids: dict[str, dict[str, Any]] = {}
    for k, inst in enumerate(sorted(instances, key=lambda i: float(i.get("start_s") or 0.0))):
        inst = dict(inst)
        inst.setdefault("id", f"i{k + 1}")
        same = inst.get("same_as")
        if same and same in ids:
            ids[same]["end_s"] = max(float(ids[same].get("end_s") or 0.0), float(inst.get("end_s") or 0.0))
            ids[inst["id"]] = ids[same]
            continue
        ident = str(inst.get("identity") or "").strip().lower()
        prev = kept[-1] if kept else None
        if prev and ident and ident == str(prev.get("identity") or "").strip().lower() \
                and float(inst.get("start_s") or 0.0) - float(prev.get("end_s") or 0.0) <= gap_s:
            prev["end_s"] = max(float(prev.get("end_s") or 0.0), float(inst.get("end_s") or 0.0))
            ids[inst["id"]] = prev
            continue
        kept.append(inst)
        ids[inst["id"]] = inst
    return kept


def choose_option(options: dict[str, str], count: int) -> tuple[str | None, str]:
    """Option whose stated number equals the count; else the nearest; None if no option carries a number."""
    numbered = {label: option_number(text) for label, text in options.items()}
    numbered = {k: v for k, v in numbered.items() if v is not None}
    if not numbered:
        return None, "no_numeric_options"
    exact = [k for k, v in numbered.items() if v == count]
    if exact:
        return exact[0], "exact"
    label = min(numbered, key=lambda k: (abs(numbered[k] - count), k))
    return label, "nearest"


def _json(text: str) -> dict[str, Any]:
    try:
        return json.loads(re.search(r"\{.*\}", text or "", re.S).group(0))
    except Exception:
        return {}


def enumerate_instances(client: Any, example: dict[str, Any], parsed: dict[str, Any], chunk_rows: int = 15,
                        per_row_chars: int = 700) -> list[dict[str, Any]]:
    schemas, _ = retrieval_catalog(example)
    rows = [(s.get("time_span") or {}, clip_schema_text(s)[:per_row_chars]) for s in schemas if isinstance(s, dict)]
    rows.sort(key=lambda r: float(r[0].get("start_s") or 0.0))
    found: list[dict[str, Any]] = []
    for c in range(0, len(rows), chunk_rows):
        chunk = [{"time_span": ts, "description": d} for ts, d in rows[c:c + chunk_rows]]
        payload = {"target": parsed.get("target"), "identity_key": parsed.get("identity_key"), "unit": parsed.get("unit"),
                   "already_found": [{"id": f["id"], "start_s": f.get("start_s"), "identity": f.get("identity")} for f in found][-40:],
                   "chunk": chunk}
        out = _json(client.chat([{"role": "system", "content": ENUM_SYSTEM}, {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]))
        for inst in out.get("instances") or []:
            if not isinstance(inst, dict):
                continue
            try:
                inst["start_s"] = float(inst.get("start_s") or 0.0); inst["end_s"] = float(inst.get("end_s") or inst["start_s"])
            except (TypeError, ValueError):
                continue
            inst["id"] = f"i{len(found) + 1}"
            found.append(inst)
    return found


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--l1-index", type=Path, required=True)
    ap.add_argument("--example-ids", type=Path, required=True)
    ap.add_argument("--mode", choices=["count", "assist"], default="count")
    ap.add_argument("--model", default="qwen/qwen3-vl-235b-a22b-instruct")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--keys-py", type=Path, default=Path("/fs/gamma-projects/vlm-robot/keys.py"))
    args = ap.parse_args(argv)
    index = json.load(args.l1_index.open())
    ids = [l.strip() for l in args.example_ids.read_text().splitlines() if l.strip() and l.strip() in index]
    client = OpenRouterClient(model=args.model, api_key=load_openrouter_api_key(keys_py_path=args.keys_py),
                              max_tokens=1500, temperature=0.0, reasoning={"effort": "minimal", "exclude": True}, timeout_s=240)
    done = {}
    if args.output.exists():
        for line in args.output.open():
            r = json.loads(line); done[r["example_id"]] = r

    def run(eid: str) -> dict[str, Any]:
        example = json.loads(Path(index[eid]["path"]).read_text())
        q = example.get("question") or {}
        options = {o.get("label"): o.get("text") for o in (q.get("options") or []) if isinstance(o, dict)} or (q.get("options") or {})
        gold = (q.get("answer") or {}).get("label") if isinstance(q.get("answer"), dict) else q.get("answer") or q.get("gold_label")
        parsed = _json(client.chat([{"role": "system", "content": PARSE_SYSTEM},
                                    {"role": "user", "content": json.dumps({"question": q.get("question_text"), "options": options}, ensure_ascii=False)}]))
        found = enumerate_instances(client, example, parsed)
        merged = merge_instances(found)
        count = len(merged)
        label, how = choose_option(options, count)
        if args.mode == "assist" or label is None:
            payload = {"question": q.get("question_text"), "options": options, "target": parsed.get("target"),
                       "instances": [{"start_s": m.get("start_s"), "end_s": m.get("end_s"), "identity": m.get("identity"), "evidence": m.get("evidence")} for m in merged],
                       "merged_count": count}
            out = _json(client.chat([{"role": "system", "content": ANSWER_SYSTEM}, {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]))
            label = out.get("label") or label; how = f"llm_{how}"
        return {"example_id": eid, "condition": f"count_{args.mode}", "label": label, "gold_label": gold, "correct": bool(label) and str(label) == str(gold),
                "count": count, "raw_instances": len(found), "mapping": how, "target": parsed.get("target"), "unit": parsed.get("unit"),
                "instances": [{"start_s": m.get("start_s"), "end_s": m.get("end_s"), "identity": m.get("identity")} for m in merged][:40]}

    todo = [e for e in ids if e not in done]
    with ThreadPoolExecutor(max_workers=args.workers) as pool, args.output.open("a") as out:
        for i, rec in enumerate(pool.map(run, todo), 1):
            done[rec["example_id"]] = rec; out.write(json.dumps(rec, ensure_ascii=False) + "\n"); out.flush()
            if i % 10 == 0:
                print(f"[{i}/{len(todo)}]", flush=True)
    rows = [done[e] for e in ids if e in done]
    acc = 100 * sum(1 for r in rows if r["correct"]) / max(len(rows), 1)
    print(json.dumps({"mode": args.mode, "questions": len(rows), "accuracy": round(acc, 1),
                      "mapping": {k: sum(1 for r in rows if r["mapping"] == k) for k in sorted({r["mapping"] for r in rows})}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
