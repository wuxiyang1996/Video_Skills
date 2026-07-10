#!/usr/bin/env python3
"""Export compact expert demos as MDP-style L2/repair controller SFT rows."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


GOLDISH_KEYS = {
    "answer",
    "correct",
    "correct_answer",
    "final_answer",
    "gold",
    "gold_answer",
    "gold_label",
    "hidden_supervision",
    "official_answer",
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _l1_snapshot_from_round(round_row: dict[str, Any]) -> dict[str, Any]:
    state_snapshot = round_row.get("state_snapshot")
    if not isinstance(state_snapshot, dict):
        return {}
    l1 = state_snapshot.get("l1")
    if not isinstance(l1, dict):
        return {}
    allowed = {
        "dataset",
        "edge_count",
        "example_id",
        "graph_id",
        "index_stats",
        "layer",
        "node_count",
        "observation_end_s",
        "video_regime",
    }
    return {key: value for key, value in l1.items() if key in allowed}


def _contains_forbidden_prompt_key(payload: Any) -> bool:
    if isinstance(payload, dict):
        return any(str(key) in GOLDISH_KEYS or _contains_forbidden_prompt_key(value) for key, value in payload.items())
    if isinstance(payload, list):
        return any(_contains_forbidden_prompt_key(value) for value in payload)
    return False


def _eligible_demo(demo: dict[str, Any]) -> bool:
    flags = demo.get("quality_flags")
    if not isinstance(flags, dict):
        return False
    return bool(flags.get("training_candidate") or flags.get("abstain_candidate"))


def _state_for_round(
    demo: dict[str, Any],
    trajectory: dict[str, Any],
    round_row: dict[str, Any],
    prior_rounds: list[dict[str, Any]],
    round_index: int,
) -> dict[str, Any]:
    return {
        "schema_version": "video-skills/mdp-round-state-v0.2",
        "process_model": "pomdp_compatible_l2_repair_controller",
        "transition_granularity": "l2_or_repair_round",
        "known_limitation": "round-level cold-start data; not atomic-skill pre-state logging",
        "demo_id": demo.get("demo_id"),
        "dataset": demo.get("dataset"),
        "example_id": demo.get("example_id"),
        "video_regime": demo.get("video_regime"),
        "task_family": demo.get("task_family"),
        "visible_demo_inputs": demo.get("visible_demo_inputs"),
        "l1_compact": demo.get("l1"),
        "l1_round_snapshot": _l1_snapshot_from_round(round_row),
        "prior_round_summaries": prior_rounds,
        "budget_state": {
            "round_index": round_row.get("round_index", round_index),
            "max_repair_rounds": trajectory.get("max_repair_rounds"),
        },
    }


def _action_for_round(round_row: dict[str, Any], round_type: str) -> dict[str, Any]:
    action = round_row.get("action")
    return {
        "schema_version": "video-skills/controller-action-v0.1",
        "round_type": round_type,
        "action": action if isinstance(action, dict) else {},
        "target_policy": "choose_next_l2_or_repair_tool_action",
    }


def _observation_for_round(round_row: dict[str, Any]) -> dict[str, Any]:
    return {
        "observation_summary": round_row.get("observation_summary") or {},
        "graph_delta": round_row.get("graph_delta") or {},
        "verifier_signal": round_row.get("verifier_signal") or {},
    }


def _transition_to_sft_chat(transition: dict[str, Any], round_type: str, terminal_status: str | None) -> dict[str, Any]:
    user_payload = {
        "task": "choose_next_controller_action",
        "state_t": transition["state_t"],
    }
    return {
        "schema_version": "video-skills/mdp-round-sft-chat-v0.2",
        "transition_id": transition["transition_id"],
        "demo_id": transition.get("demo_id"),
        "dataset": transition.get("dataset"),
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are the Video_Skills L2/repair controller. Given the current graph state, "
                    "choose the next verifier-aware tool/action as JSON. Do not use hidden supervision."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(user_payload, ensure_ascii=False, separators=(",", ":")),
            },
            {
                "role": "assistant",
                "content": json.dumps(transition["action_t"], ensure_ascii=False, separators=(",", ":")),
            },
        ],
        "metadata": {
            "controller": "l2_repair",
            "transition_granularity": "l2_or_repair_round",
            "round_type": round_type,
            "terminal_status": terminal_status,
            "reward_proxy": transition.get("reward_proxy_t") or {},
            "quality_flags": transition.get("quality_flags") or {},
        },
    }


def build_stepwise_exports(demos: list[dict[str, Any]], *, source_path: str = "") -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    transitions: list[dict[str, Any]] = []
    sft_chats: list[dict[str, Any]] = []
    round_type_counts: Counter[str] = Counter()
    terminal_status_counts: Counter[str] = Counter()
    prompt_forbidden_key_hits = 0

    for demo in demos:
        if not _eligible_demo(demo):
            continue
        l2 = demo.get("l2") if isinstance(demo.get("l2"), dict) else {}
        trajectory = l2.get("trajectory") if isinstance(l2.get("trajectory"), dict) else {}
        rounds = trajectory.get("rounds") if isinstance(trajectory.get("rounds"), list) else []
        prior_rounds: list[dict[str, Any]] = []
        for index, round_row in enumerate(rounds):
            if not isinstance(round_row, dict):
                continue
            round_type = str(round_row.get("round_type") or "unknown")
            terminal_status = round_row.get("terminal_status")
            verifier = round_row.get("verifier_signal") if isinstance(round_row.get("verifier_signal"), dict) else {}
            status = str(terminal_status or verifier.get("status") or "unknown")
            state = _state_for_round(demo, trajectory, round_row, prior_rounds, index)
            action = _action_for_round(round_row, round_type)
            transition = {
                "schema_version": "video-skills/mdp-round-transition-v0.2",
                "transition_id": f"{demo.get('demo_id')}::round:{round_row.get('round_index', index)}",
                "demo_id": demo.get("demo_id"),
                "dataset": demo.get("dataset"),
                "example_id": demo.get("example_id"),
                "split_role": "cold_start_sft",
                "controller": "l2_repair",
                "state_t": state,
                "action_t": action,
                "observation_t": _observation_for_round(round_row),
                "state_t_plus_1_summary": {
                    "terminal_status": terminal_status,
                    "acceptance_status": status,
                    "graph_delta": round_row.get("graph_delta") or {},
                },
                "reward_proxy_t": round_row.get("reward_proxy") or {},
                "done": index == len(rounds) - 1 or bool(terminal_status),
                "quality_flags": demo.get("quality_flags") or {},
            }
            user_payload = {"task": "choose_next_controller_action", "state_t": state}
            if _contains_forbidden_prompt_key(user_payload):
                prompt_forbidden_key_hits += 1
            transitions.append(transition)
            sft_chats.append(_transition_to_sft_chat(transition, round_type, terminal_status))
            round_type_counts[round_type] += 1
            terminal_status_counts[status] += 1
            prior_rounds.append(
                {
                    "round_index": round_row.get("round_index", index),
                    "round_type": round_type,
                    "action_type": action["action"].get("action_type"),
                    "terminal_status": terminal_status,
                    "verifier_status": verifier.get("status"),
                }
            )

    report = {
        "schema_version": "video-skills/mdp-round-sft-report-v0.2",
        "source_expert_demos": source_path,
        "input_demos": len(demos),
        "eligible_demos": sum(1 for demo in demos if _eligible_demo(demo)),
        "exported_transitions": len(transitions),
        "exported_sft_chats": len(sft_chats),
        "round_type_counts": dict(round_type_counts),
        "terminal_status_counts": dict(terminal_status_counts),
        "prompt_forbidden_key_hits": prompt_forbidden_key_hits,
        "granularity": "l2_or_repair_round",
        "intended_use": "cold-start behavior cloning for L2/repair controller; not final atomic-skill RL data",
    }
    return transitions, sft_chats, report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export compact expert demos as MDP-style L2/repair SFT rows.")
    parser.add_argument("--expert-demos", type=Path, required=True)
    parser.add_argument("--transition-output-jsonl", type=Path, required=True)
    parser.add_argument("--sft-output-jsonl", type=Path, required=True)
    parser.add_argument("--quality-report-output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    demos = _read_jsonl(args.expert_demos)
    transitions, sft_chats, report = build_stepwise_exports(demos, source_path=str(args.expert_demos))
    _write_jsonl(args.transition_output_jsonl, transitions)
    _write_jsonl(args.sft_output_jsonl, sft_chats)
    _write_json(args.quality_report_output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
