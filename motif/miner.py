"""Deterministic motif mining from existing L1/L2 rollout files."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .bank import MotifBank
from .schemas import MotifEvidenceRef, MotifLifecycleStatus, MotifRecord


@dataclass(frozen=True)
class MiningResult:
    bank: MotifBank
    input_rows: int
    rows_with_l1: int
    rows_with_l2: int
    motif_instances: int
    motif_type_counts: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_rows": self.input_rows,
            "rows_with_l1": self.rows_with_l1,
            "rows_with_l2": self.rows_with_l2,
            "motif_instances": self.motif_instances,
            "motif_type_counts": self.motif_type_counts,
            "bank": self.bank.summary(),
        }


def mine_paths(paths: Iterable[Path | str], min_support: int = 2) -> MiningResult:
    """Mine reusable motif candidates from JSON/JSONL L1/L2 outputs."""

    grouped: dict[str, MotifRecord] = {}
    support_refs: dict[str, list[MotifEvidenceRef]] = defaultdict(list)
    input_rows = 0
    rows_with_l1 = 0
    rows_with_l2 = 0
    motif_instances = 0
    motif_type_counts: Counter[str] = Counter()

    for row, source_path in _iter_rows(paths):
        input_rows += 1
        l1_graph = _get_l1_graph(row)
        l2_rollout = _get_l2_rollout(row)
        if l1_graph:
            rows_with_l1 += 1
        if l2_rollout:
            rows_with_l2 += 1
        for record, evidence_ref in _mine_row(row, source_path, l1_graph, l2_rollout):
            motif_instances += 1
            motif_type_counts[record.motif_type] += 1
            existing = grouped.get(record.motif_id)
            if existing is None:
                grouped[record.motif_id] = record
            support_refs[record.motif_id].append(evidence_ref)

    bank = MotifBank()
    for motif_id, record in grouped.items():
        for ref in support_refs[motif_id]:
            record.add_evidence(ref)
        if record.support_count >= min_support:
            record.status = MotifLifecycleStatus.CANDIDATE
        else:
            record.status = MotifLifecycleStatus.SHADOW
        bank.add(record)

    return MiningResult(
        bank=bank,
        input_rows=input_rows,
        rows_with_l1=rows_with_l1,
        rows_with_l2=rows_with_l2,
        motif_instances=motif_instances,
        motif_type_counts=dict(sorted(motif_type_counts.items())),
    )


def _iter_rows(paths: Iterable[Path | str]):
    for raw_path in paths:
        path = Path(raw_path)
        if not path.exists():
            continue
        if path.suffix == ".jsonl":
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        yield json.loads(line), path
        elif path.suffix == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, list):
                for item in payload:
                    if isinstance(item, dict):
                        yield item, path
            elif isinstance(payload, dict):
                yield payload, path


def _mine_row(
    row: dict[str, Any],
    source_path: Path,
    l1_graph: dict[str, Any] | None,
    l2_rollout: dict[str, Any] | None,
):
    dataset = str(row.get("dataset") or "")
    example_id = str(row.get("example_id") or "")
    task_family = str(row.get("task_family") or "")
    video_regime = str((row.get("metadata") or {}).get("video_regime") or "")

    if l2_rollout:
        node_ids = tuple(str(node.get("node_id") or "") for node in l2_rollout.get("nodes") or ())
        evidence_ref = MotifEvidenceRef(
            dataset=dataset,
            example_id=example_id,
            task_family=task_family,
            source_path=str(source_path),
            l2_node_ids=tuple(item for item in node_ids if item),
            final_answer_correct=_answer_correct(row),
            verifier_passed=_verifier_passed(l2_rollout),
            evidence_valid=_evidence_valid(l2_rollout),
            no_hidden_leakage=_no_hidden_leakage(l2_rollout),
        )
        skill_ids = _skill_sequence(l2_rollout)
        if skill_ids:
            yield _sequence_motif(row, l2_rollout, skill_ids), evidence_ref
        fanout = _hypothesis_fanout_signature(skill_ids)
        if fanout:
            yield _fanout_motif(row, l2_rollout, fanout), evidence_ref
        support_chain = _support_chain_signature(l2_rollout)
        if support_chain:
            yield _support_chain_motif(row, l2_rollout, support_chain), evidence_ref

    if l1_graph:
        l1_node_ids = tuple(
            str(node.get("node_id") or "")
            for node in l1_graph.get("nodes") or ()
            if str(node.get("node_id") or "").startswith("evidence.")
        )
        evidence_ref = MotifEvidenceRef(
            dataset=dataset,
            example_id=example_id,
            task_family=task_family,
            source_path=str(source_path),
            l1_node_ids=tuple(item for item in l1_node_ids if item),
        )
        profile = _l1_profile_signature(l1_graph, video_regime=video_regime)
        if profile:
            yield _l1_profile_motif(row, l1_graph, profile), evidence_ref


def _get_l1_graph(row: dict[str, Any]) -> dict[str, Any] | None:
    metadata = row.get("metadata") or {}
    graph = metadata.get("clue_memory_graph") or row.get("evidence_index")
    return graph if isinstance(graph, dict) and graph.get("nodes") else None


def _get_l2_rollout(row: dict[str, Any]) -> dict[str, Any] | None:
    metadata = row.get("metadata") or {}
    rollout = metadata.get("reasoning_rollout") or metadata.get("reasoning_rollout_shell")
    if not rollout:
        rollout = row.get("repair_subgraph")
    if not rollout and isinstance(row.get("l2_trajectory"), dict):
        rollout = row["l2_trajectory"].get("repair_subgraph")
    return rollout if isinstance(rollout, dict) and rollout.get("nodes") else None


def _skill_sequence(rollout: dict[str, Any]) -> tuple[str, ...]:
    return tuple(
        str(node.get("skill_id") or node.get("node_type") or "")
        for node in rollout.get("nodes") or ()
        if node.get("skill_id") or node.get("node_type")
    )


def _sequence_motif(
    row: dict[str, Any],
    rollout: dict[str, Any],
    skill_ids: tuple[str, ...],
) -> MotifRecord:
    compact_sequence = _compress_repeats(skill_ids)
    motif_id = _motif_id("l2_sequence", compact_sequence)
    return MotifRecord(
        motif_id=motif_id,
        name=f"L2 sequence: {' -> '.join(compact_sequence[:4])}",
        description="Reusable L2 reasoning skill order observed in existing rollouts.",
        motif_type="l2_skill_sequence",
        status=MotifLifecycleStatus.SHADOW,
        trigger_signature={
            "task_family": row.get("task_family"),
            "answer_format": (rollout.get("question") or {}).get("answer_format"),
            "sequence_length": len(skill_ids),
        },
        l2_template={
            "skill_sequence": list(skill_ids),
            "compressed_skill_sequence": list(compact_sequence),
            "edge_pattern": _edge_type_counts(rollout),
        },
        proposal_source="deterministic_l1_l2_miner",
    )


def _hypothesis_fanout_signature(skill_ids: tuple[str, ...]) -> dict[str, Any] | None:
    retrieve_count = sum(1 for skill in skill_ids if skill == "retrieve_evidence_for_hypothesis")
    score_count = sum(1 for skill in skill_ids if skill == "score_hypothesis_support")
    has_compare = "compare_hypotheses" in skill_ids
    if retrieve_count < 2 or score_count < 2 or not has_compare:
        return None
    return {
        "retrieve_branches": retrieve_count,
        "score_branches": score_count,
        "has_compare": has_compare,
        "has_commit": "commit_answer" in skill_ids,
    }


def _fanout_motif(
    row: dict[str, Any],
    rollout: dict[str, Any],
    fanout: dict[str, Any],
) -> MotifRecord:
    capped = {
        "retrieve_branches": min(int(fanout["retrieve_branches"]), 8),
        "score_branches": min(int(fanout["score_branches"]), 8),
        "has_compare": bool(fanout["has_compare"]),
        "has_commit": bool(fanout["has_commit"]),
    }
    motif_id = _motif_id("hypothesis_fanout", tuple(capped.items()))
    return MotifRecord(
        motif_id=motif_id,
        name="Hypothesis Fanout And Compare",
        description="Generate multiple answer hypotheses, retrieve evidence per branch, score support, then compare.",
        motif_type="l2_hypothesis_fanout",
        status=MotifLifecycleStatus.SHADOW,
        trigger_signature={
            "task_family": row.get("task_family"),
            "answer_format": (rollout.get("question") or {}).get("answer_format"),
            **capped,
        },
        l2_template={
            "branching_pattern": capped,
            "required_skills": [
                "generate_answer_hypotheses",
                "retrieve_evidence_for_hypothesis",
                "score_hypothesis_support",
                "compare_hypotheses",
            ],
            "optional_skills": ["verify_claim_support", "commit_answer"],
        },
        proposal_source="deterministic_l1_l2_miner",
    )


def _support_chain_signature(rollout: dict[str, Any]) -> dict[str, Any] | None:
    claims = rollout.get("claims") or []
    final_answer = rollout.get("final_answer") or {}
    supported_claims = sum(1 for claim in claims if claim.get("supported_by_refs"))
    node_types = {str(node.get("node_type") or "") for node in rollout.get("nodes") or []}
    if "option_verifier" in node_types and "final_commit_or_abstain" in node_types:
        return {
            "supported_claims": supported_claims,
            "has_final_answer": "final_commit_or_abstain" in node_types,
            "acceptance_status": rollout.get("acceptance_status") or rollout.get("repair_status"),
            "repair_subgraph": True,
        }
    if not final_answer or supported_claims <= 0:
        return None
    return {
        "supported_claims": supported_claims,
        "has_final_answer": True,
        "acceptance_status": rollout.get("acceptance_status"),
    }


def _support_chain_motif(
    row: dict[str, Any],
    rollout: dict[str, Any],
    signature: dict[str, Any],
) -> MotifRecord:
    motif_id = _motif_id("claim_support_chain", tuple(sorted(signature.items())))
    return MotifRecord(
        motif_id=motif_id,
        name="Claim Support To Commit",
        description="Bind evidence-backed claims to a final answer through verifier support.",
        motif_type="l2_claim_support_chain",
        status=MotifLifecycleStatus.SHADOW,
        trigger_signature={
            "task_family": row.get("task_family"),
            **signature,
        },
        l2_template={
            "required_skills": ["verify_claim_support", "commit_answer"],
            "claim_count": len(rollout.get("claims") or []),
            "answer_support_chain_len": len(rollout.get("answer_support_chain") or []),
        },
        proposal_source="deterministic_l1_l2_miner",
    )


def _l1_profile_signature(
    l1_graph: dict[str, Any],
    video_regime: str,
) -> dict[str, Any] | None:
    nodes = l1_graph.get("nodes") or []
    if not nodes:
        return None
    node_types = Counter(str(node.get("node_type") or "unknown") for node in nodes)
    modalities = Counter(
        str(node.get("modality") or "unknown")
        for node in nodes
        if node.get("node_type") in {"observation", "event", "entity_mention"}
    )
    edge_types = Counter(str(edge.get("edge_type") or "unknown") for edge in l1_graph.get("edges") or [])
    return {
        "video_regime": video_regime,
        "dominant_node_type": node_types.most_common(1)[0][0],
        "has_observations": node_types.get("observation", 0) > 0,
        "has_events": node_types.get("event", 0) > 0,
        "has_entity_mentions": node_types.get("entity_mention", 0) > 0,
        "dominant_modality": modalities.most_common(1)[0][0] if modalities else "unknown",
        "edge_type_count": len(edge_types),
    }


def _l1_profile_motif(
    row: dict[str, Any],
    l1_graph: dict[str, Any],
    profile: dict[str, Any],
) -> MotifRecord:
    motif_id = _motif_id("l1_profile", tuple(sorted(profile.items())))
    node_types = Counter(str(node.get("node_type") or "unknown") for node in l1_graph.get("nodes") or [])
    return MotifRecord(
        motif_id=motif_id,
        name=f"L1 profile: {profile['dominant_modality']} {profile['dominant_node_type']}",
        description="Reusable L1 evidence graph profile mined from clue memory graphs.",
        motif_type="l1_evidence_profile",
        status=MotifLifecycleStatus.SHADOW,
        trigger_signature={
            "task_family": row.get("task_family"),
            **profile,
        },
        l1_template={
            "node_type_counts": dict(sorted(node_types.items())),
            "index_stats": l1_graph.get("index_stats") or {},
            "retrieval": l1_graph.get("retrieval") or {},
        },
        proposal_source="deterministic_l1_l2_miner",
    )


def _answer_correct(row: dict[str, Any]) -> bool | None:
    metadata = row.get("metadata") or {}
    for key in ("eval", "evaluation", "final_acceptance", "acceptance"):
        value = metadata.get(key) or row.get(key)
        if isinstance(value, dict):
            for field in ("answer_correct", "correct", "is_correct"):
                if field in value:
                    return bool(value[field])
    return None


def _verifier_passed(rollout: dict[str, Any]) -> bool | None:
    verifier = rollout.get("verifier_summary") or {}
    if not verifier:
        return None
    hard_flags = [
        verifier.get("no_hidden_supervision_leakage"),
        verifier.get("no_old_video_fact_leakage"),
    ]
    if any(flag is False for flag in hard_flags):
        return False
    acceptance = str(rollout.get("acceptance_status") or "")
    if not acceptance:
        for node in rollout.get("nodes") or []:
            if node.get("node_type") == "final_commit_or_abstain":
                acceptance = str(node.get("terminal_status") or "")
                break
    if acceptance in {"resolved_strong", "accepted_bridge"}:
        return True
    if acceptance.startswith("accepted"):
        return True
    return None


def _evidence_valid(rollout: dict[str, Any]) -> bool | None:
    nodes = rollout.get("nodes") or []
    commit_nodes = [node for node in nodes if node.get("skill_id") == "commit_answer"]
    if not commit_nodes:
        return None
    return all(bool(node.get("evidence_refs")) for node in commit_nodes)


def _no_hidden_leakage(rollout: dict[str, Any]) -> bool | None:
    verifier = rollout.get("verifier_summary") or {}
    value = verifier.get("no_hidden_supervision_leakage")
    return bool(value) if value is not None else None


def _edge_type_counts(rollout: dict[str, Any]) -> dict[str, int]:
    counts = Counter(str(edge.get("edge_type") or "unknown") for edge in rollout.get("edges") or [])
    return dict(sorted(counts.items()))


def _compress_repeats(items: tuple[str, ...]) -> tuple[str, ...]:
    compressed = []
    for item in items:
        if not compressed or compressed[-1] != item:
            compressed.append(item)
    return tuple(compressed)


def _motif_id(prefix: str, signature: Any) -> str:
    digest = hashlib.sha1(json.dumps(signature, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:10]
    return f"{prefix}:{digest}"
