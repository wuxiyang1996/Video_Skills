import json

import pytest

from dataset_clip_wrapper.adapters.video_holmes import parse_time_range
from dataset_clip_wrapper.training.build_sft_splits import build_splits


def _chat(transition_id, controller, dataset, example_id, video_id=None):
    state = {"dataset": dataset, "example_id": example_id}
    if video_id:
        state["video_state"] = {"video_id": video_id}
    return {
        "transition_id": transition_id,
        "messages": [
            {"role": "system", "content": "Choose the next action."},
            {"role": "user", "content": json.dumps({"state_t": state})},
            {"role": "assistant", "content": json.dumps({"action": controller})},
        ],
        "metadata": {"controller": controller, "dataset": dataset},
    }


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_video_holmes_time_range_normalizes_fullwidth_punctuation():
    assert parse_time_range("02；36") == {"start_s": 156.0, "end_s": 157.0}
    assert parse_time_range("00：10–00：12") == {"start_s": 10.0, "end_s": 12.0}
    assert parse_time_range("malformed") is None


def test_video_level_split_has_no_group_overlap(tmp_path):
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    rows = [
        _chat(f"vh:a:{index}", "l1_builder", "video_holmes", f"video_holmes:train:a:q{index}", "a")
        for index in range(5)
    ] + [
        _chat(f"vh:b:{index}", "l1_builder", "video_holmes", f"video_holmes:train:b:q{index}", "b")
        for index in range(5)
    ]
    _write_jsonl(snapshot / "l1_builder_sft.jsonl", rows)

    report = build_splits(snapshot, tmp_path / "splits", dev_percent=50, salt="test", patterns=["l1_builder_sft.jsonl"])

    assert report["group_overlap_count"] == 0
    train = {json.loads(line)["split_group_id"] for line in (tmp_path / "splits" / "train_sft.jsonl").read_text().splitlines()}
    dev = {json.loads(line)["split_group_id"] for line in (tmp_path / "splits" / "dev_sft.jsonl").read_text().splitlines()}
    assert not train & dev


def test_balanced_pilot_uses_requested_controller_mixture(tmp_path):
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    families = {
        "l1_builder": "l1_builder_sft.jsonl",
        "l2_repair": "l2_repair_from_reports_sft.jsonl",
        "auxiliary_verifier": "verifier_sft.jsonl",
        "motif_lifecycle": "motif_lifecycle_sft.jsonl",
    }
    for controller, filename in families.items():
        _write_jsonl(snapshot / filename, [
            _chat(f"{controller}:{index}", controller, "cg_bench", f"cg_bench:{controller}:{index}")
            for index in range(100)
        ])

    report = build_splits(
        snapshot,
        tmp_path / "splits",
        dev_percent=5,
        salt="test",
        patterns=list(families.values()),
        target_total=100,
    )

    assert report["controller_family_counts_total"] == {"l1": 35, "l2": 35, "motif": 10, "verifier": 20}
    assert report["rows_total"] == 100


def test_external_example_map_groups_controllers_by_source_video(tmp_path):
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    rows = [
        _chat("repair:q1", "l2_repair", "cg_bench", "cg_bench:q1"),
        _chat("verify:q2", "auxiliary_verifier", "cg_bench", "cg_bench:q2"),
    ]
    _write_jsonl(snapshot / "l2_repair_from_reports_sft.jsonl", rows[:1])
    _write_jsonl(snapshot / "verifier_sft.jsonl", rows[1:])
    mapping = {
        "cg_bench:q1": "cg_bench:video:shared",
        "cg_bench:q2": "cg_bench:video:shared",
    }

    report = build_splits(
        snapshot,
        tmp_path / "splits",
        dev_percent=50,
        salt="test",
        patterns=["l2_repair_from_reports_sft.jsonl", "verifier_sft.jsonl"],
        example_video_map=mapping,
    )

    assert report["group_counts"]["total"] == 1
    assert report["group_overlap_count"] == 0


def test_excludes_evaluation_only_datasets_and_passes_strict_gates(tmp_path):
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    rows = [
        _chat("cg:1", "l1_builder", "cg_bench", "cg_bench:1"),
        _chat("vr:1", "l1_builder", "vrbench", "vrbench:1"),
        _chat("mme:1", "l1_builder", "videomme", "videomme:1"),
    ]
    _write_jsonl(snapshot / "l1_builder_sft.jsonl", rows)

    report = build_splits(
        snapshot,
        tmp_path / "splits",
        dev_percent=0,
        salt="test",
        patterns=["l1_builder_sft.jsonl"],
        exclude_datasets={"vrbench", "videomme"},
        strict=True,
    )

    assert report["rows_total"] == 1
    assert report["dataset_counts_total"] == {"cg_bench": 1}
    assert report["excluded_dataset_counts"] == {"videomme": 1, "vrbench": 1}
    assert report["hard_gates_passed"] is True


def test_colliding_record_ids_are_rewritten_without_dropping_distinct_rows(tmp_path):
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    first = _chat("same", "l2_repair", "cg_bench", "cg_bench:1")
    second = _chat("same", "l2_repair", "cg_bench", "cg_bench:1")
    second["messages"][-1]["content"] = json.dumps({"action": "different"})
    _write_jsonl(snapshot / "l2_repair_from_reports_sft.jsonl", [first, second])

    report = build_splits(
        snapshot,
        tmp_path / "splits",
        dev_percent=0,
        salt="test",
        patterns=["l2_repair_from_reports_sft.jsonl"],
        strict=True,
    )
    rows = [json.loads(line) for line in (tmp_path / "splits" / "train_sft.jsonl").read_text().splitlines()]

    assert len(rows) == 2
    assert len({row["transition_id"] for row in rows}) == 2
    assert report["id_normalization"]["record_id_collision_groups"] == 1
    assert report["id_normalization"]["record_ids_rewritten"] == 2


def test_controller_minimum_reserves_rows_within_family_quota(tmp_path):
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    repair = [_chat(f"repair:{i}", "l2_repair", "cg_bench", f"cg_bench:r:{i}") for i in range(100)]
    retrieval = [_chat(f"retrieval:{i}", "l2_retrieval", "cg_bench", f"cg_bench:t:{i}") for i in range(10)]
    _write_jsonl(snapshot / "l2_repair_from_reports_sft.jsonl", repair)
    _write_jsonl(snapshot / "l2_retrieval_sft.jsonl", retrieval)

    report = build_splits(
        snapshot,
        tmp_path / "splits",
        dev_percent=0,
        salt="test",
        patterns=["l2_repair_from_reports_sft.jsonl", "l2_retrieval_sft.jsonl"],
        target_total=20,
        mixture={"l2": 100},
        controller_minimums={"l2_retrieval": 8},
        strict=True,
    )

    assert report["controller_counts_total"]["l2_retrieval"] >= 8


def test_character_cap_filters_oversized_rows_before_sampling(tmp_path):
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    short = _chat("short", "l1_builder", "cg_bench", "cg_bench:short")
    long = _chat("long", "l1_builder", "cg_bench", "cg_bench:long")
    long["messages"][1]["content"] += "x" * 5000
    _write_jsonl(snapshot / "l1_builder_sft.jsonl", [short, long])

    report = build_splits(
        snapshot,
        tmp_path / "splits",
        dev_percent=0,
        salt="test",
        patterns=["l1_builder_sft.jsonl"],
        max_characters=1000,
        strict=True,
    )

    assert report["rows_total"] == 1
    assert report["rows_excluded_too_long"] == 1
    assert report["audit"]["character_lengths"]["max"] <= 1000


def test_strict_mode_rejects_unmet_controller_minimum(tmp_path):
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    _write_jsonl(snapshot / "l2_retrieval_sft.jsonl", [
        _chat("retrieval:1", "l2_retrieval", "cg_bench", "cg_bench:1")
    ])

    with pytest.raises(ValueError, match="controller_minimum_shortfall"):
        build_splits(
            snapshot,
            tmp_path / "splits",
            dev_percent=0,
            salt="test",
            patterns=["l2_retrieval_sft.jsonl"],
            controller_minimums={"l2_retrieval": 2},
            strict=True,
        )
