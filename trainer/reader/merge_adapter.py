"""Merge a LoRA adapter into Qwen3.5-9B and save a full-repo checkpoint that vLLM can serve.

vLLM's LoRA path for Qwen3.5 mis-applies reader adapters (the same checkpoint generates clean
JSON under HF/PEFT but collapses when served with --enable-lora), so we merge.  Merging through
`AutoModelForCausalLM` yields the language-model weights under the base repo's names
(`model.language_model.*`, `lm_head.weight`) but a text-only config (`qwen3_5_text`) that vLLM
rejects.  This writes the merged text weights together with the base repo's vision weights,
config, preprocessors and tokenizer, so the result loads exactly like the original repo.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
from pathlib import Path


def base_snapshot(model_id: str) -> Path:
    home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    snaps = sorted(glob.glob(f"{home}/hub/models--{model_id.replace('/', '--')}/snapshots/*"))
    if not snaps:
        raise FileNotFoundError(f"no local snapshot for {model_id} under {home}")
    return Path(snaps[-1])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--adapter", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)
    import torch
    from peft import PeftModel
    from safetensors import safe_open
    from safetensors.torch import save_file
    from transformers import AutoModelForCausalLM

    snap = base_snapshot(args.base_model)
    model = AutoModelForCausalLM.from_pretrained(args.base_model, dtype=torch.bfloat16, device_map=args.device)
    model = PeftModel.from_pretrained(model, str(args.adapter)).merge_and_unload()
    # save_pretrained applies HF's checkpoint key mapping (in-memory `model.layers.*` -> on-disk
    # `model.language_model.layers.*`); read the saved tensors back so names match the base repo.
    tmp = args.out.parent / (args.out.name + ".tmp_text")
    if tmp.exists():
        shutil.rmtree(tmp, ignore_errors=True)   # NFS keeps .nfs* handles briefly; the whole tree is deleted after evaluation
    model.save_pretrained(str(tmp), safe_serialization=True, max_shard_size="4GB")
    del model
    merged = {}
    for shard in sorted(glob.glob(str(tmp / "*.safetensors"))):
        with safe_open(shard, "pt") as f:
            for key in f.keys():
                merged[key] = f.get_tensor(key)
    shutil.rmtree(tmp, ignore_errors=True)
    # vision tower and anything else the text-only class does not carry, verbatim from the base shards
    index = json.load((snap / "model.safetensors.index.json").open())["weight_map"]
    extra = {}
    for shard in sorted(set(index.values())):
        with safe_open(str(snap / shard), "pt") as f:
            for key in f.keys():
                if key not in merged:
                    extra[key] = f.get_tensor(key)
    args.out.mkdir(parents=True, exist_ok=True)
    tensors = {**merged, **extra}
    # write ~4 GB shards with an index, like the base repo
    keys = sorted(tensors)
    shards, cur, size = [], [], 0
    for k in keys:
        n = tensors[k].numel() * tensors[k].element_size()
        if cur and size + n > 4_000_000_000:
            shards.append(cur); cur, size = [], 0
        cur.append(k); size += n
    if cur:
        shards.append(cur)
    weight_map = {}
    for i, ks in enumerate(shards, start=1):
        name = f"model-{i:05d}-of-{len(shards):05d}.safetensors"
        save_file({k: tensors[k] for k in ks}, str(args.out / name), metadata={"format": "pt"})
        for k in ks:
            weight_map[k] = name
    (args.out / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": weight_map}, indent=1))
    for fn in ["config.json", "generation_config.json", "preprocessor_config.json", "video_preprocessor_config.json",
               "tokenizer_config.json", "tokenizer.json", "vocab.json", "merges.txt", "chat_template.jinja"]:
        if (snap / fn).exists():
            shutil.copy2(snap / fn, args.out / fn)
    unexpected = sorted(set(tensors) - set(index))
    print(json.dumps({"merged": str(args.out), "text_keys": len(merged), "copied_base_keys": len(extra), "total_keys": len(tensors),
                      "shards": len(shards), "missing_vs_base": len(set(index) - set(tensors)), "unexpected_vs_base": len(unexpected),
                      "unexpected_examples": unexpected[:3]}))
    if unexpected or len(tensors) != len(index):
        raise SystemExit("merged checkpoint does not match the base repo's parameter names")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
