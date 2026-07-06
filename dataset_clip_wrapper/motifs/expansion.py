"""Motif expansion helpers.

The runtime must never execute a motif as a black box. Expansion templates are
planning priors that future L1/L2 agents can instantiate as ordinary atomic
graph nodes or repair-stage nodes.
"""

from __future__ import annotations

from typing import Any


def repair_subgraph_expansion(node_types: list[str], edge_types: list[str]) -> dict[str, Any]:
    return {
        "expansion_kind": "repair_subgraph_template",
        "node_types": node_types,
        "edge_types": edge_types,
        "must_expand_before_execution": True,
    }


def trajectory_round_expansion(round_types: list[str], action_types: list[str]) -> dict[str, Any]:
    return {
        "expansion_kind": "trajectory_round_template",
        "round_types": round_types,
        "action_types": action_types,
        "must_expand_before_execution": True,
    }


def motif_record_to_planning_prior(record: dict[str, Any]) -> dict[str, Any]:
    """Return the small object a future L1/L2 agent may retrieve."""
    return {
        "motif_id": record.get("motif_id"),
        "status": record.get("status"),
        "trigger_signature": record.get("trigger_signature") or {},
        "expansion_template": record.get("expansion_template") or {},
        "constraints": record.get("constraints") or [],
    }
