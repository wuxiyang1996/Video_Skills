"""Examples are loaded per job, not all at once.

Holding all 1,837 derived Video-Holmes examples resident cost ~25 GB per
process on a shared login node with two runs and a pointwise build alive.
"""

import json
from pathlib import Path

from scripts.eval.measure_answer_chain import _LazyExamples


def test_lazy_mapping_reads_on_access_only(tmp_path, monkeypatch) -> None:
    reads = []
    import scripts.eval.measure_answer_chain as m
    def fake_load(paths):
        reads.extend(paths)
        return [{"example_id": Path(paths[0]).stem, "payload": "x"}]
    monkeypatch.setattr(m, "load_frozen_l1_examples", fake_load)
    lazy = _LazyExamples({"a": tmp_path / "a.json", "b": tmp_path / "b.json"})
    assert len(lazy) == 2 and "a" in lazy and "z" not in lazy
    assert reads == []                       # nothing loaded yet
    assert lazy["a"]["payload"] == "x"
    assert reads == [tmp_path / "a.json"]   # exactly one file, on demand
    assert sorted(lazy) == ["a", "b"]


def test_missing_example_raises_keyerror(tmp_path, monkeypatch) -> None:
    import scripts.eval.measure_answer_chain as m
    monkeypatch.setattr(m, "load_frozen_l1_examples", lambda paths: [])
    lazy = _LazyExamples({"a": tmp_path / "a.json"})
    try:
        lazy["a"]; assert False
    except KeyError:
        pass
