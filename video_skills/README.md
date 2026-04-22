# `video_skills/` — Phase-1 reasoning runtime

Minimal, deterministic Python implementation of the runtime described in
[`infra_plans/`](../infra_plans/). It is the **Phase-1 substrate**: a runnable
end-to-end loop with stable contracts that later phases (8B-controller training,
72B-VLM tool calls, replay-based skill mining) plug into without breaking.

> **Status:** all 9 substrate todos are complete; 58/58 unit + smoke tests pass
> via `pytest tests/video_skills`. See
> [`infra_plans/99_meta/plan_docs_implementation_checklist.md`](../infra_plans/99_meta/plan_docs_implementation_checklist.md)
> for the doc-vs-code traceability matrix.

---

## What this gives you

| Subsystem | File | Phase-1 behaviour |
|---|---|---|
| Canonical contracts | `contracts.py` | 7 typed dataclasses (`GroundedWindow`, `EvidenceBundle`, `HopGoal`, `AtomicStepResult`, `VerificationResult`, `AbstainDecision`, `ReasoningTrace`) + supporting refs |
| Memory stores | `memory/stores.py` | In-memory Episodic, Semantic, State (Belief + Spatial), Evidence, EntityProfile registry (union-find aliases) |
| Memory procedures | `memory/procedures.py` | All 9 fixed procedures from `agentic_memory_design.md`, every call audit-logged |
| Skill bank | `skills/bank.py` + `skills/atomics.py` | `SkillRecord` schema + `ReasoningSkillBank` + 14 curated v1 atomics |
| Retriever | `retriever.py` | Lexical + entity/time/perspective filters, dedup, counter-retrieval, broaden-ladder |
| Verifier | `verifier.py` | 6 named checks (`claim_evidence_alignment`, `evidence_sufficiency`, `counterevidence`, `temporal_consistency`, `perspective_consistency`, `entity_consistency`) with `support_threshold` / `abstain_threshold` gates |
| Harness | `harness.py` | Deterministic hop interpreter, atomic-step iteration, evidence binding, all writes routed through `MemoryProcedureRegistry` |
| Controller (v0) | `controller.py` | Rule-based question analyser, hop planner, skill router, answer composer (placeholder for the trainable 8B) |
| Online loop | `loop.py` | The §2D control flow (`controller.analyze → next_hop → harness.run_hop → verifier.verify_hop → ...`) |

What is **deliberately stubbed** in Phase 1:

- The controller is rule-based, not the trained 8B model.
- Skills are deterministic Python; no calls into the 72B / 32B grounding tools.
- Retrieval is lexical; no learned reranker or vector store.
- Skill mining (synthesis from successful traces) is not wired yet.

---

## Quickstart

```python
from video_skills import build_runtime, run_question
from tests.video_skills.synthetic import make_alice_bob_key_window

rt = build_runtime()

rt.memory_procedures.update_entity_profile(
    entity_id="alice", canonical_name="Alice", seen_at=10.0,
)
rt.memory_procedures.update_entity_profile(
    entity_id="bob", canonical_name="Bob", seen_at=10.0,
)
rt.memory_procedures.append_grounded_event(window=make_alice_bob_key_window())

trace = run_question(
    rt,
    "Did Alice pick up the key before Bob entered?",
    target_entities=["alice", "bob"],
)

print(trace.answer)
print(trace.final_verification.passed, trace.final_verification.score)
```

Expected output:

```
Order: a_before_b.
True 0.92
```

The trace contains the `QuestionAnalysis`, every `HopRecord` (with all
`AtomicStepResult` and `VerificationResult` objects), the final verification,
and any `AbstainDecision`.

---

## Tests

```
pytest tests/video_skills -q
```

- `test_contracts.py` — schema versions, `next_action` aggregation, AtomicStepResult validation
- `test_memory_procedures.py` — all 9 procedures + audit log
- `test_skills_bank.py` — bank registration, schema, lookup
- `test_retriever.py` — filters, dedup, broaden ladder, counter-retrieval
- `test_verifier.py` — every check + threshold behaviour
- `test_harness.py` — atomic chain execution, step caps, write routing, schema validation
- `test_end_to_end.py` — full loop on synthetic windows, multiple question types

---

## How this maps onto `infra_plans/`

| Plan doc | Code |
|---|---|
| `03_controller/actors_reasoning_model.md` — canonical runtime data contracts | `video_skills/contracts.py` |
| `03_controller/actors_reasoning_model.md` — retriever/verifier as first-class subsystems | `video_skills/{retriever,verifier}.py` |
| `02_memory/agentic_memory_design.md` — episodic / semantic / state stores, lifecycle procedures, entity-centric indexing | `video_skills/memory/{stores,procedures}.py` |
| `04_harness/atomic_skills_hop_refactor_execution_plan.md` — hop interpreter, evidence binding, trace logging | `video_skills/harness.py` |
| `05_skills/skill_extraction_bank.md` — formal `SkillRecord`, atomic starter set | `video_skills/skills/{bank,atomics}.py` |
| `03_controller/actors_reasoning_model.md` §2D — online serving loop | `video_skills/loop.py` |

Phase-2 deltas (training, live VLM tools, learned retrieval, skill mining) are
intentionally out of scope here; the contracts in `contracts.py` are the
freeze-line that those phases must respect.
