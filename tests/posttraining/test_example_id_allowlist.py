"""Targeted repair: process only listed example ids within a shard."""

from types import SimpleNamespace

from dataset_clip_wrapper.runners.run_staged_llm_pipeline import _apply_example_id_allowlist


def _items(*ids):
    return [SimpleNamespace(example_id=i) for i in ids]


def test_no_allowlist_keeps_everything() -> None:
    items = _items("a", "b")
    assert _apply_example_id_allowlist(items, None) is items


def test_allowlist_filters_and_preserves_order(tmp_path) -> None:
    f = tmp_path / "ids.txt"; f.write_text("cg_bench:513\n\ncg_bench:468\n")
    kept = _apply_example_id_allowlist(_items("cg_bench:468", "cg_bench:500", "cg_bench:513"), f)
    assert [i.example_id for i in kept] == ["cg_bench:468", "cg_bench:513"]


def test_allowlist_with_no_matches_yields_nothing(tmp_path) -> None:
    f = tmp_path / "ids.txt"; f.write_text("cg_bench:999\n")
    assert _apply_example_id_allowlist(_items("cg_bench:468"), f) == []
