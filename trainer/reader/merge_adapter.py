"""Merge a LoRA adapter into the base weights and save a standalone model for vLLM.

vLLM's LoRA path for Qwen3.5 (served through the ConditionalGeneration wrapper) applied the
reader adapters wrongly: the same checkpoint that generates clean JSON under HF/PEFT collapsed to
option "A" with 6k-character prose when served with --enable-lora.  Merging sidesteps LoRA serving.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--adapter", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(args.base_model, dtype=torch.bfloat16, device_map=args.device)
    model = PeftModel.from_pretrained(model, str(args.adapter))
    merged = model.merge_and_unload()
    args.out.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(args.out), safe_serialization=True)
    AutoTokenizer.from_pretrained(args.base_model).save_pretrained(str(args.out))
    print({"merged": str(args.out), "architectures": merged.config.architectures, "files": len(list(args.out.iterdir()))})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
