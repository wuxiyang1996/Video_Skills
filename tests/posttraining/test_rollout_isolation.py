from __future__ import annotations

import pytest

from trainer.grpo.isolation import assert_rollout_isolation, deep_isolate, fingerprint_example


def test_deep_isolate_breaks_shared_identity() -> None:
    parent = {"metadata": {"clue_memory_graph": {"nodes": [{"id": 1}]}, "x": 1}}
    a = deep_isolate(parent)
    b = deep_isolate(parent)
    a["metadata"]["x"] = 99
    a["metadata"]["clue_memory_graph"]["nodes"].append({"id": 2})
    assert parent["metadata"]["x"] == 1
    assert len(parent["metadata"]["clue_memory_graph"]["nodes"]) == 1
    assert b["metadata"]["x"] == 1
    assert a is not b
    assert a["metadata"] is not b["metadata"]


def test_assert_rollout_isolation_detects_shared_metadata() -> None:
    shared_meta = {"motif_online": {"motif_retrieval_attempted": True}}
    rollouts = [
        {"metadata": shared_meta, "final_answer": {"label": "A"}},
        {"metadata": shared_meta, "final_answer": {"label": "B"}},
    ]
    with pytest.raises(AssertionError):
        assert_rollout_isolation(rollouts)


def test_fingerprint_stable() -> None:
    ex = {"example_id": "e1", "metadata": {"a": 1}}
    assert fingerprint_example(ex) == fingerprint_example(dict(ex))
