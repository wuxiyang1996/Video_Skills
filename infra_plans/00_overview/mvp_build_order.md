# MVP Build Order

> Goal: Define the **minimum viable implementation order** for the Video_Skills project. This document is the authoritative phasing reference; per-subsystem files (actors / memory / harness / bank / synthetics) point back here for the v1 scope boundary.
>
> **Related plans:**
> - [Actors / Reasoning Model](../03_controller/actors_reasoning_model.md) — 8B controller, role split, training focus
> - [Agentic Memory](../02_memory/agentic_memory_design.md) — fixed memory procedures, three stores + evidence
> - [Atomic Skills & Hop Refactor](../04_harness/atomic_skills_hop_refactor_execution_plan.md) — harness runtime spec, MVP failure handling
> - [Skill Extraction / Bank](../05_skills/skill_extraction_bank.md) — reasoning bank scope, separate registries
> - [Skill Synthetics Agents](../05_skills/skill_synthetics_agents.md) — synthesis pipeline (later phases)
> - [Grounding Pipeline Execution Plan](../01_grounding/grounding_pipeline_execution_plan.md) — perception + entity stack
> - [Video Benchmarks & Grounding](../01_grounding/video_benchmarks_grounding.md) — benchmarks, memory graph, adapters
> - [Evaluation & Ablation Plan](../07_evaluation/evaluation_ablation_plan.md) — eval matrix and ablations

---

## 0. Cross-Plan Principle

This file operationalizes the **Stable Memory, Evolving Reasoning** principle ([Actors §0](../03_controller/actors_reasoning_model.md#0-design-principle-stable-memory-evolving-reasoning)):

- Memory construction and maintenance are handled by **fixed procedures** and **fixed memory-management skills**.
- The evolving skill bank is reserved for **reasoning skills**.
- 72B is a **frozen visual grounding / proposal specialist**, not the orchestrator.
- The 8B model is the **central controller / orchestrator**.
- The first milestone is a **robust 8B controller + structured memory + hop-based reasoning + 72B grounding**, not a self-evolving system.

The phases below sequence the build so the substrate stabilizes **before** any adaptive layer is allowed to write back to it.

---

## Phase 1: Stable Substrate

Implement first, in this order. Nothing in phase 2 starts until each item below is testable end-to-end.

1. **Canonical runtime schemas / data contracts.** Materialize the objects in [Actors §2A](../03_controller/actors_reasoning_model.md#2a-canonical-runtime-data-contracts): `GroundedWindow`, `EvidenceBundle`, `HopGoal`, `AtomicStepResult`, `VerificationResult`, `AbstainDecision`, `ReasoningTrace`, `HopRecord`. These are the only allowed wire format between major modules.
2. **Visual grounding layer.** Implement the perception + entity stack from the [Grounding Pipeline Execution Plan](../01_grounding/grounding_pipeline_execution_plan.md) so it produces well-typed `GroundedWindow`s. The 72B grounding tools (Observer-A / Observer-B / Reasoner) are wired here as **frozen specialists**, callable by the harness.
3. **Fixed memory lifecycle and fixed memory procedures.** Build the **Memory Procedure Registry** from [Agentic Memory §0.2](../02_memory/agentic_memory_design.md#02-memory-management-skills-vs-reasoning-skills): `open_episode_thread`, `append_grounded_event`, `update_entity_profile`, `refresh_state_memory`, `compress_episode_cluster`, `attach_evidence_ref`, `resolve_entity_alias`, `revise_belief_state`, `mark_memory_conflict`. Implement the lifecycle table in [Agentic Memory — Lifecycle implementation table](../02_memory/agentic_memory_design.md#lifecycle-implementation-table). All write triggers, decay constants, and refresh modes are **fixed** in v1.
4. **Retriever / verifier basic subsystem.** Build the retriever ([Actors §2B](../03_controller/actors_reasoning_model.md#2b-retriever-as-a-first-class-subsystem)) with query rewriting, entity / time / perspective filters, and the broaden ladder; build the verifier ([Actors §2C](../03_controller/actors_reasoning_model.md#2c-verifier-as-a-first-class-subsystem)) with the standard check catalog and the two threshold gates.
5. **Harness runtime.** Implement the harness from [Atomic Skills & Hop Refactor — Harness runtime specification](../04_harness/atomic_skills_hop_refactor_execution_plan.md#harness-runtime-specification), including hop length / termination rules, atomic-step I/O contract, composite expansion, trace logging format, and **MVP Failure Handling**.
6. **Curated atomic reasoning skill set.** Ship the **Reasoning Skill Bank** with the starter inventory in [Skill Extraction / Bank §4.8 / §4.9](../05_skills/skill_extraction_bank.md#48-minimal-starter-set-implementation-v1). The bank is **closed** in v1 — no online additions, no synthesis-driven writes.
7. **8B controller over hops / memory / evidence.** Train the 8B controller's adapters on the four phase-1 behaviors ([Actors §0.3](../03_controller/actors_reasoning_model.md#03-first-phase-training-focus)): hop planning, skill routing, answer-vs-abstain, evidence-aware control. Use GRPO with the `ReasoningTrace`-based reward shaping from [Actors §2E](../03_controller/actors_reasoning_model.md#2e-training-signals-for-the-controller).

**Phase 1 explicitly does not include:**

- a free self-evolving bank
- adaptive memory policy
- aggressive skill synthesis
- 72B-driven orchestration
- automatic patch / split / retire loops
- bank growth from runtime traces

---

## Phase 2: Limited Reuse

Begin only after phase 1 is stable end-to-end and the phase-1 evaluation matrix ([Evaluation & Ablation Plan](../07_evaluation/evaluation_ablation_plan.md)) is reproducible.

1. **Conservative composite reasoning skills.** Allow the synthesizer to surface composite candidates from repeated successful atomic chains. **Activation requires a human reviewer** in phase 2; no auto-activation.
2. **Limited promotion from repeated successful atomic chains.** Use the doubled phase-1 thresholds from [Skill Synthetics §0.2](../05_skills/skill_synthetics_agents.md#02-phase-1-conservative-synthesis-policy) (`N_repeat = 10`, `τ_success = 0.8`, transfer ≥ 3 task families). Promotion is versioned and shadow-deployed before activation.
3. **Improved reflection / trace export.** Refine the harness's *Reflection Update Hooks* and the synthesizer's *Trace Localization Procedure* so failure buckets are assigned with high agreement on a held-out audit sample.
4. **Stronger abstention and verifier policies.** Tighten the verifier's threshold gates and the controller's abstention policy based on phase-1 calibration data; introduce per-task-family thresholds where evidence supports them.

Memory remains **fixed** through phase 2. The Memory Procedure Registry is unchanged.

---

## Phase 3: Controlled Evolution

Begin only after the **Preconditions for Later Self-Evolution** in [Skill Synthetics §0.3](../05_skills/skill_synthetics_agents.md#03-preconditions-for-later-self-evolution) are demonstrably in place.

1. **Trace-based skill synthesis.** Enable the full trace-first synthesis pipeline ([Skill Synthetics §1](../05_skills/skill_synthetics_agents.md#1-primary-synthesis-unit-successful-reasoning-traces)), including verifiability and non-leakiness gates and the 8B-as-judge quality control.
2. **Patch / split / retire policies.** Enable the bank maintenance operations from [Skill Extraction / Bank §8](../05_skills/skill_extraction_bank.md#8-skill-promotion-split-merge-retire) under the synthesizer's gates. Aggressive operations remain feature-flagged and monitored.
3. **Bank versioning / rollback.** Operationalize the [Bank Versioning and Rollback](../05_skills/skill_synthetics_agents.md#bank-versioning-and-rollback) policy: every operation writes a new version; rollback is automatic when a promoted composite regresses.
4. **Broader cross-benchmark transfer testing.** Extend the evaluation matrix to test transferability of the evolved bank across benchmark families; add cross-task-family success rate to promotion thresholds.

Memory **may** become adaptive in a future phase only after a separate principled design pass; it is not in scope for phase 3 of this MVP plan.

---

## MVP Success Criterion

The first real success criterion for the project is:

> Demonstrating that the **8B controller over structured memory and hop-based reasoning** outperforms **direct large-VLM QA** and **naive retrieval baselines** on **evidence-grounded multi-hop video reasoning**.

Operationally, that means:

- The **8B + curated bank + fixed memory + 72B grounding** configuration beats:
  - the **72B-only** direct QA configuration (large-VLM baseline), and
  - a **naive retrieval + 72B answer** configuration without hop planning, perspective handling, or verifier-driven abstention,
- on the long-video, multi-hop, perspective-bearing benchmarks listed in [Video Benchmarks & Grounding](../01_grounding/video_benchmarks_grounding.md), measured per the [Evaluation & Ablation Plan](../07_evaluation/evaluation_ablation_plan.md),
- with **reviewable traces**: every answer carries a `ReasoningTrace` whose hops, evidence bundles, and verifier results are inspectable.

A self-evolving bank is **not** part of the v1 success criterion. It is a phase-2 / phase-3 capability whose purpose is to extend, not to define, the system's first proof of value.
