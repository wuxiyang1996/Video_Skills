"""High-level motif extraction and management agent."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dataset_clip_wrapper.perception.openrouter_client import load_openrouter_api_key

from .llm_agent import DEFAULT_CURATOR_MODEL, DEFAULT_EXTRACTOR_MODEL, LLMMotifAgent, LLMMotifAgentConfig
from .instance_miner import _rows_from_path, mine_motif_instances, mine_motif_instances_from_path
from .promotion import PromotionConfig, apply_promotion_gates
from .registry import MotifBank


@dataclass(frozen=True)
class MotifAgentConfig:
    input_paths: tuple[Path, ...]
    output_bank: Path
    summary_output: Path | None = None
    agent_mode: str = "hybrid"
    extractor_model: str = DEFAULT_EXTRACTOR_MODEL
    curator_model: str = DEFAULT_CURATOR_MODEL
    keys_py_path: str | None = None
    llm_timeout_s: int = 180
    max_rows: int | None = None
    min_support_count: int = 2
    min_verifier_pass_rate: float = 0.8
    min_dataset_coverage: int = 1


class MotifAgent:
    """Build and maintain a reusable L1/L2 motif bank.

    The main path mirrors the old skill-bank-agent pipeline: an LLM extractor
    proposes motif candidates, an LLM curator approves/defer/vetoes them, and
    the bank stores approved/deferred expandable templates. Deterministic mining
    remains available as seed/fallback/audit.
    """

    def __init__(self, config: MotifAgentConfig) -> None:
        self.config = config

    def run(self) -> dict[str, Any]:
        bank = MotifBank()
        input_counts: dict[str, int] = {}
        instance_count = 0
        llm_error_count = 0
        deterministic_fallback_count = 0
        llm_startup_error = ""
        llm_agent = None
        if self.config.agent_mode in {"llm", "hybrid"}:
            try:
                llm_agent = self._make_llm_agent()
            except Exception as exc:
                llm_startup_error = f"{type(exc).__name__}: {exc}"
                if self.config.agent_mode == "llm":
                    raise
        for path in self.config.input_paths:
            if self.config.agent_mode == "deterministic":
                instances = mine_motif_instances_from_path(path)
                input_counts[str(path)] = len(instances)
                for motif_id, instance in instances:
                    bank.add_instance(motif_id, instance)
                    instance_count += 1
                continue

            path_count = 0
            rows = _rows_from_path(path)
            if self.config.max_rows is not None:
                rows = rows[: self.config.max_rows]
            for row in rows:
                instances: list[tuple[str, Any]] = []
                try:
                    if llm_agent is None:
                        raise RuntimeError("LLM motif agent is unavailable")
                    instances = llm_agent.propose_and_curate(row, source_path=path, bank=bank)
                except Exception as exc:
                    llm_error_count += 1
                    if self.config.agent_mode == "llm":
                        raise
                    fallback = mine_motif_instances(row, path)
                    for _, instance in fallback:
                        instance.proposal_source = "deterministic_fallback"
                        instance.agent_backend = f"llm_unavailable:{type(exc).__name__}"
                    instances = fallback
                    deterministic_fallback_count += len(instances)
                for motif_id, instance in instances:
                    bank.add_instance(motif_id, instance)
                    instance_count += 1
                    path_count += 1
            input_counts[str(path)] = path_count

        promotion = apply_promotion_gates(
            bank,
            PromotionConfig(
                min_support_count=self.config.min_support_count,
                min_verifier_pass_rate=self.config.min_verifier_pass_rate,
                min_dataset_coverage=self.config.min_dataset_coverage,
            ),
        )
        bank.save_jsonl(self.config.output_bank)
        summary = {
            "schema_version": "video-skills-relaunch/motif-bank-summary-v0.1",
            "agent_mode": self.config.agent_mode,
            "extractor_model": self.config.extractor_model,
            "curator_model": self.config.curator_model,
            "input_counts": input_counts,
            "instance_count": instance_count,
            "llm_error_count": llm_error_count,
            "llm_startup_error": llm_startup_error,
            "deterministic_fallback_count": deterministic_fallback_count,
            "promotion": promotion,
            "bank": bank.summary(),
            "output_bank": str(self.config.output_bank),
        }
        if self.config.summary_output:
            self.config.summary_output.parent.mkdir(parents=True, exist_ok=True)
            self.config.summary_output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return summary

    def _make_llm_agent(self) -> LLMMotifAgent:
        api_key = load_openrouter_api_key(keys_py_path=self.config.keys_py_path)
        return LLMMotifAgent(
            LLMMotifAgentConfig(
                extractor_model=self.config.extractor_model,
                curator_model=self.config.curator_model,
                api_key=api_key,
                timeout_s=self.config.llm_timeout_s,
            )
        )
