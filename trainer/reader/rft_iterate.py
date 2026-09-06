"""Rejection-sampling fine-tuning (verifiable-reward iteration) for the reader.

One round: for every training question, sample K completions from the current
policy (an OpenAI-compatible endpoint, e.g. vLLM serving the base model with the
current LoRA adapter), score each with trainer.reader.rewards.reader_reward, and
keep the best completion of each question whose reward reaches --keep-threshold
(right answer, well-cited).  The kept set is the next SFT round's data.  This is
the RLVR-lite loop (expert iteration / STaR with verifiable rewards); GRPO can
replace it without changing the reward.
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from dataset_clip_wrapper.perception.openrouter_client import OpenRouterClient
from trainer.reader.prompting import reader_messages
from trainer.reader.rewards import reader_reward


def sample_and_score(client: Any, example: dict[str, Any], gold: str, k: int, process_weight: float) -> list[dict[str, Any]]:
    msgs = reader_messages(example, rationale=True)
    out = []
    for _ in range(k):
        text = client.chat(msgs)
        score = reader_reward(example, text or "", gold, process_weight=process_weight)
        out.append({"completion": text or "", **score})
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--example-index", type=Path, required=True)
    ap.add_argument("--example-ids", type=Path, default=None)
    ap.add_argument("--api-base", required=True, help="chat-completions URL of the current policy")
    ap.add_argument("--model", required=True)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--process-weight", type=float, default=0.5)
    ap.add_argument("--keep-threshold", type=float, default=1.0)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args(argv)
    index = json.loads(args.example_index.read_text(encoding="utf-8"))
    ids = [l.strip() for l in args.example_ids.read_text().splitlines() if l.strip()] if args.example_ids else list(index)
    client = OpenRouterClient(model=args.model, api_key="local", api_base=args.api_base, temperature=args.temperature,
                              max_tokens=1500, timeout_s=600)
    done: dict[str, Any] = {}
    if args.output.exists():
        for l in args.output.open(encoding="utf-8"):
            r = json.loads(l); done[r["example_id"]] = r

    def run(eid: str) -> dict[str, Any]:
        example = json.loads(Path(index[eid]["path"]).read_text(encoding="utf-8"))
        gold = ((example.get("question") or {}).get("answer") or {}).get("label") or ""
        samples = sample_and_score(client, example, str(gold), args.k, args.process_weight)
        best = max(samples, key=lambda s: s["reward"])
        return {"example_id": eid, "gold": gold, "k": args.k, "mean_reward": sum(s["reward"] for s in samples) / len(samples),
                "pass_rate": sum(1 for s in samples if s["correct"]) / len(samples), "best": best,
                "messages": reader_messages(example, rationale=True), "completion": best["completion"], "reward": best["reward"],
                "keep": best["reward"] >= args.keep_threshold}

    todo = [e for e in ids if e not in done and e in index]
    with ThreadPoolExecutor(max_workers=args.workers) as pool, args.output.open("a", encoding="utf-8") as f:
        for i, rec in enumerate(pool.map(run, todo), 1):
            done[rec["example_id"]] = rec; f.write(json.dumps(rec, ensure_ascii=False) + "\n"); f.flush()
            if i % 50 == 0:
                print(f"[{i}/{len(todo)}]", flush=True)
    rows = [done[e] for e in ids if e in done]
    print(json.dumps({"questions": len(rows), "kept": sum(1 for r in rows if r["keep"]),
                      "mean_pass_rate": round(sum(r["pass_rate"] for r in rows) / max(len(rows), 1), 3),
                      "mean_reward": round(sum(r["mean_reward"] for r in rows) / max(len(rows), 1), 3)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
