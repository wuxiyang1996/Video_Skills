import pytest

from scripts.eval.replay_l2_typedplan_affected_rollouts import (
    needs_typedplan_replay,
    replace_affected_rows,
    row_key,
)


def _row(dataset="cg_bench", group=1, sample=2, process=True, code="invalid_skill_args"):
    return {
        "dataset": dataset, "group": group, "sample": sample,
        "process_supported": process,
        "rollout_diagnostic": {"failed_skill_codes": [
            {"skill_id": "localize_clue", "failure_code": code}
        ]},
    }


def test_replay_targets_only_process_supported_cg_typed_binding_failures() -> None:
    assert needs_typedplan_replay(_row()) is True
    assert needs_typedplan_replay(_row(dataset="video_holmes")) is False
    assert needs_typedplan_replay(_row(process=False)) is False
    assert needs_typedplan_replay(_row(code="no_clue_candidate")) is False


def test_replacement_preserves_unaffected_rows_and_order() -> None:
    first, second = _row(group=1), _row(group=2)
    replacement = {**first, "terminal_success": True}
    result = replace_affected_rows([first, second], {row_key(first): replacement})
    assert result == [replacement, second]


def test_replacement_rejects_unknown_or_duplicate_keys() -> None:
    row = _row()
    with pytest.raises(ValueError, match="duplicate"):
        replace_affected_rows([row, dict(row)], {})
    with pytest.raises(ValueError, match="absent"):
        replace_affected_rows([row], {("cg_bench", 99, 0): row})
