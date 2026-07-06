#!/usr/bin/env python3
"""Smoke test for motif mining and bank promotion."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from dataset_clip_wrapper.motifs.llm_agent import LLMMotifAgent, LLMMotifAgentConfig
from dataset_clip_wrapper.motifs.registry import MotifBank
from dataset_clip_wrapper.motifs.agent import MotifAgent, MotifAgentConfig


class _FakeExtractorClient:
    last_response_metadata = {}

    def chat_json(self, messages, *, response_format=None):  # noqa: ANN001
        return {
            "motif_candidates": [
                {
                    "name": "repair then verify options",
                    "motif_type": "l2_recursive_reasoning_template",
                    "trigger_signature": {"requires_repair": True, "video_regime": "short"},
                    "graph_template": {"node_types": ["gap_diagnosis", "l1_patch", "option_verifier"]},
                    "expansion_template": {"expansion_kind": "l2_repair_template"},
                    "confidence": 0.87,
                    "reason": "recurring repair path",
                }
            ]
        }


class _FakeCuratorClient:
    last_response_metadata = {}

    def chat_json(self, messages, *, response_format=None):  # noqa: ANN001
        return {"decisions": [{"idx": 0, "verdict": "approve", "reason": "expandable and reusable"}]}


def _write_fixture(path: Path) -> None:
    rows = [
        {
            "dataset": "video_holmes",
            "example_id": "toy:q1",
            "task_family": "short_social",
            "video_regime": "short",
            "final_acceptance_status": "accepted_strong",
            "l2_trajectory": {
                "rounds": [
                    {
                        "round_type": "initial_l2_reasoning",
                        "action": {"action_type": "call_gptoss_reasoning_planner"},
                        "terminal_status": "accepted_strong",
                    }
                ]
            },
        },
        {
            "dataset": "video_holmes",
            "example_id": "toy:q2",
            "task_family": "short_social",
            "video_regime": "short",
            "final_acceptance_status": "accepted_strong",
            "l2_trajectory": {
                "rounds": [
                    {
                        "round_type": "initial_l2_reasoning",
                        "action": {"action_type": "call_gptoss_reasoning_planner"},
                        "terminal_status": "repair_requested",
                    },
                    {
                        "round_type": "repair_l2_reasoning",
                        "action": {"action_type": "bounded_recursive_repair"},
                        "terminal_status": "resolved_strong",
                    },
                ]
            },
            "repair_subgraph": {
                "nodes": [
                    {"node_id": "n1", "node_type": "l2_gap_diagnosis"},
                    {"node_id": "n2", "node_type": "repair_plan"},
                    {"node_id": "n3", "node_type": "l1_patch"},
                    {"node_id": "n4", "node_type": "option_evidence_selector"},
                    {"node_id": "n5", "node_type": "option_verifier"},
                    {"node_id": "n6", "node_type": "final_commit_or_abstain"},
                ],
                "edges": [
                    {"src": "n1", "dst": "n2", "edge_type": "requests_repair"},
                    {"src": "n2", "dst": "n3", "edge_type": "patches_l1"},
                    {"src": "n3", "dst": "n4", "edge_type": "selects_evidence"},
                    {"src": "n4", "dst": "n5", "edge_type": "verifies_options"},
                    {"src": "n5", "dst": "n6", "edge_type": "commits_or_abstains"},
                ],
            },
        },
    ]
    path.write_text(json.dumps({"reports": rows}, indent=2), encoding="utf-8")


def test_motif_agent() -> None:
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        fixture = root / "final.json"
        bank_path = root / "motif_bank.jsonl"
        summary_path = root / "summary.json"
        _write_fixture(fixture)
        summary = MotifAgent(
            MotifAgentConfig(
                input_paths=(fixture,),
                output_bank=bank_path,
                summary_output=summary_path,
                agent_mode="deterministic",
                min_support_count=1,
            )
        ).run()
        assert summary["instance_count"] == 3
        assert summary["bank"]["motif_count"] == 3
        assert bank_path.exists()
        records = [json.loads(line) for line in bank_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        assert any(record["motif_type"] == "repair_subgraph_path" for record in records)
        assert all(record["status"] == "promoted" for record in records)
        assert summary_path.exists()


def test_llm_motif_agent_with_mock_clients() -> None:
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        fixture = root / "final.json"
        _write_fixture(fixture)
        row = json.loads(fixture.read_text(encoding="utf-8"))["reports"][1]
        agent = LLMMotifAgent(
            LLMMotifAgentConfig(
                extractor_model="qwen/qwen3.5",
                curator_model="openai/gpt-oss-120b",
                api_key="test",
            ),
            extractor_client=_FakeExtractorClient(),
            curator_client=_FakeCuratorClient(),
        )
        instances = agent.propose_and_curate(row, source_path=fixture, bank=MotifBank())
        assert len(instances) == 1
        _, instance = instances[0]
        assert instance.proposal_source == "llm_agent"
        assert "qwen/qwen3.5" in instance.agent_backend
        assert "openai/gpt-oss-120b" in instance.agent_backend
        assert instance.curator_verdict == "approve"
        assert instance.expansion_template["must_expand_before_execution"] is True


if __name__ == "__main__":
    test_motif_agent()
    test_llm_motif_agent_with_mock_clients()
    print("motif agent smoke test passed")
