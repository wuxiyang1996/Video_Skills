"""Qwen/GPT-OSS motif proposal and curation agent.

This ports the old skill-bank-agent shape into the L1/L2 motif layer:
propose reusable graph motifs, curate proposed bank mutations, then store only
expandable templates in the motif bank.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dataset_clip_wrapper.perception.openrouter_client import OpenRouterClient

from .canonicalize import motif_id, motif_signature
from .instance_miner import ACCEPTED_FINAL_STATUSES, mine_motif_instances
from .registry import MotifBank, MotifInstance


DEFAULT_EXTRACTOR_MODEL = "qwen/qwen3.5"
DEFAULT_CURATOR_MODEL = "openai/gpt-oss-120b"


EXTRACTOR_SYSTEM_PROMPT = """You are a motif extraction agent for a video L1/L2 reasoning system.
Your job is to find reusable L1/L2 graph motifs from accepted rollouts.

Rules:
- A motif is a reusable graph or repair template, not a new atomic skill.
- Never copy answer text, hidden ground truth, option labels, names, or timestamps as a motif.
- Motifs must expand into ordinary L1/L2 nodes before execution.
- Prefer patterns that help future L1 graph construction, L2 recursive repair, or evidence verification.
- Return strict JSON only."""


CURATOR_SYSTEM_PROMPT = """You are a motif bank maintenance curator.
Review proposed motif candidates and decide approve, veto, or defer.

Approve only if the candidate is reusable, evidence-grounded, and expandable.
Veto candidates that leak answers, encode dataset-specific facts, or behave like black-box skills.
Defer if the candidate may be useful but needs more rollout support.
Return strict JSON only."""


@dataclass(frozen=True)
class LLMMotifAgentConfig:
    extractor_model: str = DEFAULT_EXTRACTOR_MODEL
    curator_model: str = DEFAULT_CURATOR_MODEL
    api_key: str = ""
    timeout_s: int = 180
    max_tokens: int = 1800
    max_candidates_per_row: int = 4


class LLMMotifAgent:
    """LLM-backed motif proposal and curator adapter."""

    def __init__(
        self,
        config: LLMMotifAgentConfig,
        *,
        extractor_client: OpenRouterClient | None = None,
        curator_client: OpenRouterClient | None = None,
    ) -> None:
        self.config = config
        if extractor_client is not None:
            self.extractor_client = extractor_client
        else:
            self.extractor_client = OpenRouterClient(
                model=config.extractor_model,
                api_key=config.api_key,
                temperature=0.1,
                max_tokens=config.max_tokens,
                timeout_s=config.timeout_s,
            )
        if curator_client is not None:
            self.curator_client = curator_client
        else:
            self.curator_client = OpenRouterClient(
                model=config.curator_model,
                api_key=config.api_key,
                temperature=0.0,
                max_tokens=config.max_tokens,
                timeout_s=config.timeout_s,
            )

    def propose_and_curate(
        self,
        row: dict[str, Any],
        *,
        source_path: Path,
        bank: MotifBank,
    ) -> list[tuple[str, MotifInstance]]:
        if _final_status(row) not in ACCEPTED_FINAL_STATUSES:
            return []
        seed_instances = [instance.to_dict() for _, instance in mine_motif_instances(row, source_path)]
        compact = _compact_rollout(row, seed_instances=seed_instances)
        proposals = self._propose(compact)
        decisions = self._curate(proposals, bank.summary())
        return self._materialize(row, source_path, proposals, decisions)

    def _propose(self, compact: dict[str, Any]) -> list[dict[str, Any]]:
        payload = self.extractor_client.chat_json(
            [
                {"role": "system", "content": EXTRACTOR_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        "Extract up to "
                        f"{self.config.max_candidates_per_row} reusable motif candidates from this accepted rollout. "
                        "Use this JSON schema: "
                        '{"motif_candidates":[{"name":"short stable name","motif_type":"trajectory_round_path|repair_subgraph_path|l1_evidence_template|l2_recursive_reasoning_template","trigger_signature":{},"graph_template":{},"expansion_template":{"must_expand_before_execution":true},"confidence":0.0,"reason":"brief"}]}\n\n'
                        f"Rollout:\n{compact}"
                    ),
                },
            ]
        )
        candidates = payload.get("motif_candidates") or payload.get("candidates") or []
        if not isinstance(candidates, list):
            return []
        return [item for item in candidates[: self.config.max_candidates_per_row] if isinstance(item, dict)]

    def _curate(self, proposals: list[dict[str, Any]], bank_summary: dict[str, Any]) -> dict[int, dict[str, str]]:
        if not proposals:
            return {}
        payload = self.curator_client.chat_json(
            [
                {"role": "system", "content": CURATOR_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        "Curate these motif candidates against the current bank summary. "
                        'Return {"decisions":[{"idx":0,"verdict":"approve|veto|defer","reason":"brief"}]}.\n\n'
                        f"Bank summary:\n{bank_summary}\n\nCandidates:\n{proposals}"
                    ),
                },
            ]
        )
        out: dict[int, dict[str, str]] = {}
        for item in payload.get("decisions") or []:
            if not isinstance(item, dict):
                continue
            try:
                idx = int(item.get("idx"))
            except Exception:
                continue
            verdict = str(item.get("verdict") or "defer").lower()
            if verdict not in {"approve", "veto", "defer"}:
                verdict = "defer"
            out[idx] = {"verdict": verdict, "reason": str(item.get("reason") or "")}
        return out

    def _materialize(
        self,
        row: dict[str, Any],
        source_path: Path,
        proposals: list[dict[str, Any]],
        decisions: dict[int, dict[str, str]],
    ) -> list[tuple[str, MotifInstance]]:
        out: list[tuple[str, MotifInstance]] = []
        meta = _metadata(row, source_path)
        final_status = _final_status(row)
        for idx, proposal in enumerate(proposals):
            decision = decisions.get(idx, {"verdict": "defer", "reason": "curator_missing_decision"})
            if decision["verdict"] == "veto":
                continue
            motif_type = _clean_type(proposal.get("motif_type"))
            graph_template = _dict_or_empty(proposal.get("graph_template"))
            trigger_signature = _dict_or_empty(proposal.get("trigger_signature"))
            expansion_template = _dict_or_empty(proposal.get("expansion_template"))
            expansion_template["must_expand_before_execution"] = True
            signature = motif_signature(
                "llm",
                motif_type,
                proposal.get("name") or "",
                graph_template.get("node_types") or graph_template.get("round_types") or graph_template,
                trigger_signature,
            )
            instance = MotifInstance(
                motif_type=motif_type,
                signature=signature,
                final_status=final_status,
                verifier_passed=final_status in ACCEPTED_FINAL_STATUSES,
                graph_template=graph_template,
                trigger_signature=trigger_signature,
                expansion_template=expansion_template,
                proposal_source="llm_agent",
                agent_backend=f"extractor={self.config.extractor_model};curator={self.config.curator_model}",
                curator_verdict=decision["verdict"],
                curator_reason=decision["reason"],
                confidence=_float_or_none(proposal.get("confidence")),
                **meta,
            )
            out.append((motif_id(instance.motif_type, instance.signature), instance))
        return out


def _dict_or_empty(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except Exception:
        return None


def _clean_type(value: Any) -> str:
    text = str(value or "l2_recursive_reasoning_template").strip()
    allowed = {
        "trajectory_round_path",
        "repair_subgraph_path",
        "l1_evidence_template",
        "l2_recursive_reasoning_template",
    }
    return text if text in allowed else "l2_recursive_reasoning_template"


def _final_status(row: dict[str, Any]) -> str:
    if "l2" in row and isinstance(row["l2"], dict):
        return str(row["l2"].get("final_acceptance_status") or row.get("demo_type") or "")
    return str(row.get("final_acceptance_status") or "")


def _metadata(row: dict[str, Any], source_path: Path) -> dict[str, str]:
    return {
        "dataset": str(row.get("dataset") or ""),
        "example_id": str(row.get("example_id") or ""),
        "task_family": str(row.get("task_family") or ""),
        "video_regime": str(row.get("video_regime") or ""),
        "source_path": str(source_path),
    }


def _l2_payload(row: dict[str, Any]) -> dict[str, Any]:
    return row.get("l2") if isinstance(row.get("l2"), dict) else row


def _compact_rollout(row: dict[str, Any], *, seed_instances: list[dict[str, Any]]) -> dict[str, Any]:
    l2 = _l2_payload(row)
    trajectory = l2.get("trajectory") or l2.get("l2_trajectory") or row.get("l2_trajectory") or {}
    rounds = []
    for item in trajectory.get("rounds") or []:
        if not isinstance(item, dict):
            continue
        rounds.append(
            {
                "round_type": item.get("round_type"),
                "action_type": (item.get("action") or {}).get("action_type"),
                "terminal_status": item.get("terminal_status"),
                "verifier_status": (item.get("verifier") or {}).get("status"),
            }
        )
    repair = l2.get("repair_subgraph") or row.get("repair_subgraph") or {}
    nodes = repair.get("nodes") or []
    edges = repair.get("edges") or []
    return {
        "dataset": row.get("dataset"),
        "example_id": row.get("example_id"),
        "task_family": row.get("task_family"),
        "video_regime": row.get("video_regime"),
        "final_status": _final_status(row),
        "trajectory_rounds": rounds[:8],
        "repair_node_types": [node.get("node_type") for node in nodes[:12] if isinstance(node, dict)],
        "repair_edge_types": [edge.get("edge_type") for edge in edges[:12] if isinstance(edge, dict)],
        "deterministic_seed_motifs": seed_instances[:4],
    }
