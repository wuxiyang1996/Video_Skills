"""Mine reusable motif candidates from accepted L1/L2 rollout artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from .canonicalize import motif_id, motif_signature
from .expansion import repair_subgraph_expansion, trajectory_round_expansion
from .registry import MotifInstance


ACCEPTED_FINAL_STATUSES = {"accepted_strong", "accepted_bridge", "resolved_strong"}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _rows_from_path(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        return _read_jsonl(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict):
        for key in ("reports", "demos", "examples", "rows"):
            rows = payload.get(key)
            if isinstance(rows, list):
                return [row for row in rows if isinstance(row, dict)]
        return [payload]
    return []


def _final_status(row: dict[str, Any]) -> str:
    if "l2" in row and isinstance(row["l2"], dict):
        return str(row["l2"].get("final_acceptance_status") or row.get("demo_type") or "")
    return str(row.get("final_acceptance_status") or "")


def _l2_payload(row: dict[str, Any]) -> dict[str, Any]:
    return row.get("l2") if isinstance(row.get("l2"), dict) else row


def _trajectory(row: dict[str, Any]) -> dict[str, Any]:
    l2 = _l2_payload(row)
    return (l2.get("trajectory") or l2.get("l2_trajectory") or row.get("l2_trajectory") or {})


def _repair_subgraph(row: dict[str, Any]) -> dict[str, Any]:
    l2 = _l2_payload(row)
    return (l2.get("repair_subgraph") or row.get("repair_subgraph") or {})


def _metadata(row: dict[str, Any], source_path: Path) -> dict[str, str]:
    return {
        "dataset": str(row.get("dataset") or ""),
        "example_id": str(row.get("example_id") or ""),
        "task_family": str(row.get("task_family") or ""),
        "video_regime": str(row.get("video_regime") or ""),
        "source_path": str(source_path),
    }


def _trajectory_instances(row: dict[str, Any], source_path: Path) -> Iterable[MotifInstance]:
    trajectory = _trajectory(row)
    rounds = [round_row for round_row in trajectory.get("rounds") or [] if isinstance(round_row, dict)]
    if not rounds:
        return []
    round_types = [str(item.get("round_type") or "unknown") for item in rounds]
    action_types = [str((item.get("action") or {}).get("action_type") or "unknown") for item in rounds]
    terminal_statuses = [str(item.get("terminal_status") or "") for item in rounds]
    signature = motif_signature("trajectory", "->".join(round_types), "->".join(action_types), "->".join(terminal_statuses))
    final_status = _final_status(row)
    meta = _metadata(row, source_path)
    return [
        MotifInstance(
            motif_type="trajectory_round_path",
            signature=signature,
            final_status=final_status,
            verifier_passed=final_status in ACCEPTED_FINAL_STATUSES,
            graph_template={
                "round_types": round_types,
                "action_types": action_types,
                "terminal_statuses": terminal_statuses,
            },
            trigger_signature={
                "task_family": meta["task_family"],
                "video_regime": meta["video_regime"],
                "round_count": len(rounds),
            },
            expansion_template=trajectory_round_expansion(round_types, action_types),
            **meta,
        )
    ]


def _linear_subgraph_path(nodes: list[dict[str, Any]], edges: list[dict[str, Any]]) -> tuple[list[str], list[str]]:
    node_by_id = {str(node.get("node_id")): node for node in nodes if node.get("node_id")}
    incoming = {str(edge.get("dst")) for edge in edges if edge.get("dst")}
    starts = [node_id for node_id in node_by_id if node_id not in incoming]
    current = starts[0] if starts else (next(iter(node_by_id), ""))
    node_types: list[str] = []
    edge_types: list[str] = []
    seen: set[str] = set()
    while current and current not in seen:
        seen.add(current)
        node_types.append(str(node_by_id.get(current, {}).get("node_type") or "unknown"))
        next_edge = next((edge for edge in edges if str(edge.get("src")) == current), None)
        if not next_edge:
            break
        edge_types.append(str(next_edge.get("edge_type") or "unknown"))
        current = str(next_edge.get("dst") or "")
    return node_types, edge_types


def _repair_subgraph_instances(row: dict[str, Any], source_path: Path) -> Iterable[MotifInstance]:
    subgraph = _repair_subgraph(row)
    nodes = [node for node in subgraph.get("nodes") or [] if isinstance(node, dict)]
    edges = [edge for edge in subgraph.get("edges") or [] if isinstance(edge, dict)]
    if len(nodes) < 2:
        return []
    node_types, edge_types = _linear_subgraph_path(nodes, edges)
    if len(node_types) < 2:
        return []
    final_status = _final_status(row)
    meta = _metadata(row, source_path)
    signature = motif_signature("repair_subgraph", "->".join(node_types), "->".join(edge_types), final_status)
    return [
        MotifInstance(
            motif_type="repair_subgraph_path",
            signature=signature,
            final_status=final_status,
            verifier_passed=final_status in ACCEPTED_FINAL_STATUSES,
            graph_template={
                "node_types": node_types,
                "edge_types": edge_types,
                "node_count": len(nodes),
                "edge_count": len(edges),
            },
            trigger_signature={
                "task_family": meta["task_family"],
                "video_regime": meta["video_regime"],
                "requires_repair": True,
            },
            expansion_template=repair_subgraph_expansion(node_types, edge_types),
            **meta,
        )
    ]


def mine_motif_instances(row: dict[str, Any], source_path: Path) -> list[tuple[str, MotifInstance]]:
    final_status = _final_status(row)
    if final_status not in ACCEPTED_FINAL_STATUSES:
        return []
    instances: list[MotifInstance] = []
    instances.extend(_trajectory_instances(row, source_path))
    instances.extend(_repair_subgraph_instances(row, source_path))
    return [(motif_id(instance.motif_type, instance.signature), instance) for instance in instances]


def mine_motif_instances_from_path(path: Path) -> list[tuple[str, MotifInstance]]:
    out: list[tuple[str, MotifInstance]] = []
    for row in _rows_from_path(path):
        out.extend(mine_motif_instances(row, path))
    return out
