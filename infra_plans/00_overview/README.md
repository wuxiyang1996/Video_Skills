# 00_overview — Cross-Cutting Orientation

> **Layer purpose:** Anchor every other plan in the project. Files here are **system-level**, not subsystem-level: they define the build order, the typed objects every module exchanges, and the integrative narrative that ties grounding, memory, controller, harness, skills, training, and evaluation together.
>
> **Read first** before any other folder.

---

## Files in this folder

| File | What it is | When you read it |
|------|------------|------------------|
| [`system_overview.md`](system_overview.md) | One-page integrative description of the system, module roles, offline vs online flow, and the open-work backlog grouped into "middle-layer glue" vs "later-phase capability". | First. Gives you the mental model before diving into any subsystem. |
| [`mvp_build_order.md`](mvp_build_order.md) | The **authoritative phasing reference**. Phase 1 (stable substrate), Phase 2 (limited reuse), Phase 3 (controlled evolution). Defines the **MVP success criterion**. Every other plan points back here for the v1 scope boundary. | Second. Whenever someone proposes work, check whether it is in Phase 1 / 2 / 3. |
| [`runtime_contracts.md`](runtime_contracts.md) | The **canonical typed objects** every major module passes to every other major module: `GroundedWindow`, `EvidenceBundle`, `HopGoal`, `AtomicStepResult`, `VerificationResult`, `AbstainDecision`, `ReasoningTrace`. Plus contract rules (evidence-or-abstain, no silent enrichment, etc.). | Third. Implementing any subsystem? It must produce / consume these objects and nothing else at the wire. |

---

## Why these are grouped

These three documents do not belong to any single layer because **every layer depends on them**. Moving them into `01_grounding` or `03_controller` would create false ownership. They sit at the top so any reader (controller engineer, memory engineer, eval lead) starts from the same shared vocabulary.

## What is *not* here

- Subsystem-specific designs (those live in `01_grounding` through `07_evaluation`).
- Implementation checklists (those live in [`../99_meta/plan_docs_implementation_checklist.md`](../99_meta/plan_docs_implementation_checklist.md)).
- Training recipes (those live in [`../06_training/`](../06_training/README.md)).
