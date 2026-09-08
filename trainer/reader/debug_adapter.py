"""Diagnose a trained reader adapter with plain HF (no vLLM): losses and greedy generations on training rows.

Prints, for --n rows of the SFT data: the completion-only loss used in training, the standard
full-logits loss (same positions, as a cross-check of the alignment), and greedy generations with
and without the adapter from the exact training prompt.  Separates "the adapter is broken" from
"the adapter is fine but vLLM serves it wrongly".
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--adapter", type=Path, required=True)
    ap.add_argument("--base-model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--max-len", type=int, default=9000)
    ap.add_argument("--new-tokens", type=int, default=220)
    args = ap.parse_args(argv)
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trainer.reader.sft_lora import completion_only_loss, encode_example

    tok = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.bfloat16, device_map="cuda")
    model = PeftModel.from_pretrained(model, str(args.adapter))
    model.eval()
    rows = [json.loads(l) for l in args.data.open() if l.strip()]
    picked = []
    for r in rows:
        e = encode_example(tok, r, args.max_len)
        if e:
            picked.append((r, e))
        if len(picked) >= args.n:
            break
    for r, e in picked:
        ids = torch.tensor([e["input_ids"]], device="cuda"); labels = torch.tensor([e["labels"]], device="cuda"); att = torch.ones_like(ids)
        with torch.no_grad():
            mine = completion_only_loss(model, {"input_ids": ids, "attention_mask": att, "labels": labels})
            out = model(input_ids=ids, attention_mask=att)
            logits = out.logits[:, :-1, :].float(); tgt = labels[:, 1:]
            full = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1), ignore_index=-100)
            with model.disable_adapter():
                base_loss = completion_only_loss(model, {"input_ids": ids, "attention_mask": att, "labels": labels})
        n_sup = int((labels != -100).sum())
        prompt_len = int((labels == -100).sum())
        print(json.dumps({"example_id": r["example_id"], "prompt_tokens": prompt_len, "supervised_tokens": n_sup,
                          "loss_completion_only": round(float(mine), 3), "loss_full_logits_same_positions": round(float(full), 3),
                          "loss_base_no_adapter": round(float(base_loss), 3)}), flush=True)
        prompt_ids = ids[:, :prompt_len]
        for tag, ctx in (("adapter", None), ("base", model.disable_adapter)):
            with torch.no_grad():
                if ctx is None:
                    gen = model.generate(input_ids=prompt_ids, attention_mask=torch.ones_like(prompt_ids), max_new_tokens=args.new_tokens, do_sample=False)
                else:
                    with ctx():
                        gen = model.generate(input_ids=prompt_ids, attention_mask=torch.ones_like(prompt_ids), max_new_tokens=args.new_tokens, do_sample=False)
            text = tok.decode(gen[0, prompt_len:], skip_special_tokens=True)
            print(f"[{tag}] {text[:500]!r}", flush=True)
        print("[target]", r["completion"][:300], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
