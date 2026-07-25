from pathlib import Path

from motif import (
    MotifBank,
    MotifEvidenceRef,
    MotifLifecycleManager,
    MotifLifecycleStatus,
    MotifQueryEngine,
    MotifRecord,
    MotifTransferAdapter,
    MotifTransferExample,
)
from motif.transfer import MotifEvalResult


def test_motif_bank_roundtrip(tmp_path: Path) -> None:
    record = MotifRecord(
        motif_id="motif_repeat_change",
        name="Repeat Change",
        description="Compare repeated visual states and infer the changed object.",
        trigger_signature={"question_type": "temporal_comparison"},
    )
    record.add_evidence(MotifEvidenceRef(
        dataset="video_holmes",
        example_id="vh_001",
        task_family="temporal_comparison",
        final_answer_correct=True,
        verifier_passed=True,
        evidence_valid=True,
        no_hidden_leakage=True,
    ))

    bank = MotifBank([record])
    path = tmp_path / "motifs.jsonl"
    bank.save_jsonl(path)

    loaded = MotifBank.load_jsonl(path)
    restored = loaded.require("motif_repeat_change")
    assert restored.support_count == 1
    assert restored.trigger_signature["question_type"] == "temporal_comparison"


def test_lifecycle_promotes_after_transfer_gate() -> None:
    record = MotifRecord(
        motif_id="motif_evidence_bridge",
        name="Evidence Bridge",
        description="Bind clue evidence to a final reasoning claim.",
    )
    record.add_evidence(MotifEvidenceRef(dataset="cg_bench", example_id="cg_001"))
    record.add_evidence(MotifEvidenceRef(dataset="cg_bench", example_id="cg_002"))

    examples = [
        MotifTransferExample("cg_bench", "heldout_1", "causal", {}),
        MotifTransferExample("cg_bench", "heldout_2", "causal", {}),
    ]

    def run_fn(example, motif):
        if motif is None:
            return MotifEvalResult(False, True, True, True)
        return MotifEvalResult(True, True, True, True)

    report = MotifTransferAdapter(run_fn).evaluate(record, examples)
    record.add_transfer_report(report)

    bank = MotifBank()
    lifecycle = MotifLifecycleManager(bank)
    lifecycle.add_candidate(record)
    lifecycle.apply_transfer_gates(record.motif_id)

    assert bank.require(record.motif_id).status == MotifLifecycleStatus.VERIFIED


def test_query_engine_only_returns_visible_motifs() -> None:
    active = MotifRecord(
        motif_id="motif_temporal_order",
        name="Temporal Order",
        description="Use before and after evidence to answer temporal questions.",
        status=MotifLifecycleStatus.ACTIVE,
        trigger_signature={"question_type": "before after temporal order"},
    )
    draft = MotifRecord(
        motif_id="motif_draft",
        name="Draft",
        description="Unverified draft motif.",
        status=MotifLifecycleStatus.DRAFT,
    )
    bank = MotifBank([active, draft])

    results = MotifQueryEngine(bank).select("before after temporal order", top_k=5)

    assert [item.motif_id for item in results] == ["motif_temporal_order"]

    # Non-empty ACTIVE bank still returns a candidate even with an unrelated query.
    unrelated = MotifQueryEngine(bank).select("zzzz unrelated tokens", top_k=5)
    assert [item.motif_id for item in unrelated] == ["motif_temporal_order"]
    assert unrelated[0].score == 0.0
