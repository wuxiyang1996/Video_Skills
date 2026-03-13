#!/usr/bin/env python
"""
Shared ALFWorld-7B policy wrapper (HF checkpoints).

This module exposes a simple text-in / action-list-out interface so we can
reuse the same ALFWorld-7B checkpoint as a baseline across different cold_start
scripts (LMGame-Bench, AgentEvolver, Orak, Pokemon Red).

It does NOT depend on any specific environment; callers are responsible for:
  - Constructing a textual observation string.
  - Providing a list of admissible action strings for the current step.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import torch

try:
    from transformers import (  # type: ignore
        AutoConfig,
        AutoModelForCausalLM,
        AutoTokenizer,
    )
except Exception:  # pragma: no cover
    AutoConfig = None  # type: ignore
    AutoModelForCausalLM = None  # type: ignore
    AutoTokenizer = None  # type: ignore


BASE_TOKENIZER_REPO = "Qwen/Qwen2.5-7B-Instruct"

KNOWN_SUBFOLDERS = {
    "Jianwen/Alfworld-7B-SFT": "checkpoint-140",
    "Jianwen/Webshop-7B-SFT": "checkpoint-140",
    "Jianwen/Webshop-7B-RL": "checkpoint-140",
    "Jianwen/Search-7B-SFT": "checkpoint-140",
    "Jianwen/Search-7B-RL": "checkpoint-140",
}

# Repos that use FSDP-style shards (model_world_size_*_rank_*.pt) inside a subfolder.
# We load rank 0 (which contains the full model) manually.
FSDP_SHARD_REPOS = {
    "Jianwen/Alfworld-7B-RL": {
        "subfolder": "actor",
        "rank_file": "actor/model_world_size_4_rank_0.pt",
    },
}


@dataclass
class Alfworld7BConfig:
    model_path: str
    checkpoint_type: str = "sft"  # "sft" or "rl"
    temperature: float = 0.8
    max_new_tokens: int = 64
    subfolder: Optional[str] = None  # auto-detected for known repos


class Alfworld7BPolicy:
    """
    Lightweight HF policy that selects exactly one action from a discrete set.
    """

    def __init__(self, cfg: Alfworld7BConfig, device: Optional[str] = None):
        if AutoModelForCausalLM is None or AutoTokenizer is None:
            raise ImportError(
                "transformers is not installed. Install transformers to use "
                "ALFWORLD-7B checkpoints as a baseline."
            )
        self.cfg = cfg
        fsdp_info = FSDP_SHARD_REPOS.get(cfg.model_path)

        if fsdp_info is not None:
            self._load_fsdp_model(cfg, fsdp_info, device)
        else:
            self._load_standard_model(cfg, device)

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    # ------------------------------------------------------------------
    def _load_standard_model(self, cfg: Alfworld7BConfig, device: Optional[str]):
        """Load a repo that uses standard HF weight files (possibly in a subfolder)."""
        subfolder = cfg.subfolder or KNOWN_SUBFOLDERS.get(cfg.model_path)
        sf_kw = {"subfolder": subfolder} if subfolder else {}

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                cfg.model_path, trust_remote_code=True, **sf_kw,
            )
        except Exception:
            print(f"[INFO] Tokenizer from {cfg.model_path} failed; "
                  f"falling back to base tokenizer {BASE_TOKENIZER_REPO}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                BASE_TOKENIZER_REPO, trust_remote_code=True,
            )

        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.model_path,
            device_map="auto" if device is None else device,
            torch_dtype="auto",
            trust_remote_code=True,
            **sf_kw,
        )

    # ------------------------------------------------------------------
    def _load_fsdp_model(self, cfg: Alfworld7BConfig, fsdp_info: dict, device: Optional[str]):
        """Load a repo whose weights are FSDP shards (rank 0 = full model)."""
        from huggingface_hub import hf_hub_download

        subfolder = cfg.subfolder or fsdp_info["subfolder"]
        rank_file = fsdp_info["rank_file"]

        # --- tokenizer ---
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                cfg.model_path, subfolder=subfolder, trust_remote_code=True,
            )
        except Exception:
            print(f"[INFO] Tokenizer from {cfg.model_path}/{subfolder} failed; "
                  f"falling back to base tokenizer {BASE_TOKENIZER_REPO}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                BASE_TOKENIZER_REPO, trust_remote_code=True,
            )

        # --- config ---
        config = AutoConfig.from_pretrained(
            cfg.model_path, subfolder=subfolder, trust_remote_code=True,
        )

        # --- model weights from rank-0 shard ---
        print(f"[INFO] Downloading FSDP rank-0 shard: {rank_file} ...")
        shard_path = hf_hub_download(cfg.model_path, rank_file)
        print(f"[INFO] Loading state dict from {shard_path} ...")
        state_dict = torch.load(shard_path, map_location="cpu", weights_only=False)

        # Instantiate model from config and load weights
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
        model.load_state_dict(state_dict, strict=True)

        # Move to device / cast dtype
        if device is None:
            model = model.half().cuda()
        else:
            model = model.half().to(device)

        self.model = model
        print(f"[INFO] FSDP model loaded successfully on {self.model.device}")

    def _build_prompt(self, obs: str, action_names: List[str]) -> str:
        actions_str = ", ".join(action_names)
        return (
            "You are a game-playing agent powered by an ALFWorld-7B checkpoint. "
            "You must choose EXACTLY one next action from the allowed action list.\n\n"
            f"Observation:\n{obs}\n\n"
            f"Allowed actions: {actions_str}\n\n"
            "Respond with ONLY the chosen action string, nothing else.\n"
        )

    def choose_action(self, obs: str, action_names: List[str]) -> str:
        if not action_names:
            # Fallback to a generic 'noop' if nothing is provided.
            return "noop"

        prompt = self._build_prompt(obs, action_names)
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.cfg.max_new_tokens,
            do_sample=True,
            temperature=self.cfg.temperature,
            pad_token_id=self.tokenizer.eos_token_id,
        )[0]

        full_text = self.tokenizer.decode(output_ids, skip_special_tokens=True)
        generated = full_text[len(prompt) :].strip()

        # First, try to match any allowed action that appears in the generated text.
        for a in action_names:
            if a.lower() in generated.lower():
                return a

        # Second, try to interpret the first line as an exact action.
        first_line = generated.splitlines()[0].strip()
        for a in action_names:
            if first_line.lower() == a.lower():
                return a

        # Fallback: default to the first action.
        return action_names[0]

