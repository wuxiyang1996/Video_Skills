# System Overview and Open Work

> Purpose: Pull the current `infra_plans` together into one system-level view and enumerate the work that still remains after the MVP-staging pass ([MVP Build Order](mvp_build_order.md)). This file is **descriptive and integrative**; the per-subsystem files are the normative source of truth.
>
> **Related plans:**
> - [MVP Build Order](mvp_build_order.md) — phase 1 / 2 / 3 sequencing
> - [Actors / Reasoning Model](actors_reasoning_model.md) — 8B controller, role split, runtime contracts
> - [Agentic Memory](agentic_memory_design.md) — three stores + evidence + fixed memory procedures
> - [Atomic Skills & Hop Refactor](atomic_skills_hop_refactor_execution_plan.md) — harness runtime
> - [Skill Extraction / Bank](skill_extraction_bank.md) — reasoning skill bank, separate registries
> - [Skill Synthetics Agents](skill_synthetics_agents.md) — synthesis pipeline (later phases)
> - [Grounding Pipeline Execution Plan](grounding_pipeline_execution_plan.md) — perception + entity stack
> - [Video Benchmarks & Grounding](video_benchmarks_grounding.md) — benchmarks, memory graph, adapters
> - [Evaluation & Ablation Plan](evaluation_ablation_plan.md) — eval matrix and ablations
> - [Plan Docs Implementation Checklist](plan_docs_implementation_checklist.md) — checklist this file's "open work" section is anchored to

---

## 1. How the system is set up now

At a high level, the repo has converged on a **three-layer system**.

### 1.1 Infrastructure / grounding / memory layer

Raw video is converted into structured grounded outputs and stored in a compact memory design with **episodic**, **semantic**, and **state** memory ([Agentic Memory](agentic_memory_design.md)). Short and long videos share the same grounding schema; the difference is that long videos additionally need persistent storage and retrieval ([Grounding Pipeline](grounding_pipeline_execution_plan.md)).

### 1.2 8B controller layer

The center of the system is a **trainable 8B controller / orchestrator**. It does not process raw pixels. It operates over structured observer outputs, memory nodes, perspective threads, evidence chains, and skill protocols. Its responsibilities include question decomposition, hop planning, skill selection, memory retrieval, evidence sufficiency judgment, verification, reflection, and (later phases) skill-bank updates ([Actors §0.1 — Role Split](actors_reasoning_model.md#01-role-split-72b-vs-8b-vs-harness)).

### 1.3 Reasoning skill / reflection layer

Skills are defined as **reasoning operators**, not infrastructure steps. The bank stores **atomic** skills and **composite** skills, where one reasoning hop is a short, verifiable composition of several atomic skills ([Skill Extraction / Bank](skill_extraction_bank.md)). Reflection and evolution act on **traces of these hops**, not on raw segments or pipeline primitives ([Skill Synthetics Agents](skill_synthetics_agents.md)).

---

## 2. The current role split

| Role | What it is | What it does |
|---|---|---|
| **72B models** | Frozen large grounding / proposal tools | Social extraction, spatial extraction, ambiguity resolution, evidence refinement, evidence-to-answer generation. Called on demand; **not** the main policy. |
| **8B controller** | The central orchestrator (only learnable component in v1) | Hop planning, skill selection, memory-aware retrieval, evidence sufficiency, verification, abstention, trace export. |
| **Harness** | Runtime layer (not a model) | Executes hop plans, expands composites, calls fixed memory procedures and 72B tools, performs local verification, logs canonical traces. |

The memory backbone is the **`SocialVideoGraph` + three-store** design: episodic memory for grounded events over time, semantic memory for slowly changing abstractions, state memory for the current social/spatial world state ([Agentic Memory](agentic_memory_design.md)).

Conceptually, the system now looks like:

```
72B grounding / proposal tools
        │
        ▼
structured grounding (GroundedWindow)
        │
        ▼
episodic / semantic / state memory  (fixed memory procedures)
        │
        ▼
8B controller chooses hop goals and skills
        │
        ▼
hop executes over memory + evidence (harness)
        │
        ▼
verification (per step + per hop + final)
        │
        ▼
answer or abstain
        │
        ▼
trace exported for later reflection / evolution
```

---

## 3. How reasoning is supposed to run

The current design has two execution regimes (with an auto-routing default):

- **Direct mode** (short videos) — grounded local evidence stays in context; hops still use atomic reasoning steps internally.
- **Retrieval mode** (long videos) — the controller issues memory searches and chains retrieved evidence across hops.
- **Auto mode** — switches between the two based on video duration ([Actors §3](actors_reasoning_model.md#3-reasoning-core--think--search--answer-loop)).

Within a regime, the **reasoning loop** is:

1. parse the question
2. choose the next hop goal
3. choose an atomic chain or composite skill
4. execute over memory / evidence
5. verify intermediate outputs
6. continue, retrieve more evidence, answer, or abstain
7. on failure, emit a trace for reflection and (later phases) skill-bank updates

---

## 4. The most important architectural improvement in the updated plan

The MVP staging pass ([MVP Build Order](mvp_build_order.md)) makes the system more realistic by separating **stable infrastructure** from the **adaptive reasoning** layer:

- Memory is treated as a **stable substrate**.
- Memory construction and maintenance are handled by **fixed procedures** ([Agentic Memory §0.2 — Memory-Management Skills vs Reasoning Skills](agentic_memory_design.md#02-memory-management-skills-vs-reasoning-skills)).
- The **evolving bank** is reserved for **reasoning skills** ([Skill Extraction / Bank §0.3 — Separate Registries](skill_extraction_bank.md#03-separate-registries)).
- **72B remains a specialist tool**, not the orchestrator.
- **8B remains the central controller.**

This is a better setup than the earlier "everything co-evolves" version because it sequences the system: substrate stabilizes first, then the reasoning layer is allowed to adapt over it ([Skill Synthetics Agents §0.3 — Preconditions for Later Self-Evolution](skill_synthetics_agents.md#03-preconditions-for-later-self-evolution)).

---

## 5. After this, what still needs to be worked out

After the current MVP-staging pass, the remaining work falls into two groups. Each item is anchored to the open issues in [`plan_docs_implementation_checklist.md`](plan_docs_implementation_checklist.md).

### Group A — Middle-layer glue that must be nailed down before implementation

This is the most urgent group. Without it, the system devolves into module-specific dicts and prompt strings.

#### A.1 Canonical runtime schemas

Define one canonical end-to-end set of objects that flow between modules. The drafted schemas in [Actors §2A — Canonical Runtime Data Contracts](actors_reasoning_model.md#2a-canonical-runtime-data-contracts) — `GroundedWindow`, `EvidenceBundle`, `HopGoal`, `AtomicStepResult`, `VerificationResult`, `AbstainDecision`, `ReasoningTrace` — must be locked down and adopted as the **only** allowed wire format between major runtime components. Open work: per-module migration to these objects; freeze `schema_version`; add serializer/validator.

#### A.2 Memory lifecycle rules

The memory design is strong on top-level structure but still thin on lifecycle semantics. Now that v1 treats memory as **fixed and stable** ([Agentic Memory §0.1](agentic_memory_design.md#01-fixed-memory-procedures-in-phase-1)), the lifecycle table in [Agentic Memory — Lifecycle implementation table](agentic_memory_design.md#lifecycle-implementation-table) must be implemented end-to-end:

- write triggers per store
- contradiction handling (`contradicts` edges, no silent overwrite)
- confidence decay
- stale-state rules
- semantic refresh (versioned regenerate)
- entity-centric indexing and identity persistence
- compression / eviction with reversible archive

Open work: implement the [Memory Procedure Registry](agentic_memory_design.md#02-memory-management-skills-vs-reasoning-skills) entries against this table, with deterministic tests per row.

#### A.3 Retriever + verifier as real subsystems

Plans already mention `search_memory` and per-atomic `verification_rule`, but retrieval and verification are still under-specified as subsystems. The drafted specs in [Actors §2B — Retriever](actors_reasoning_model.md#2b-retriever-as-a-first-class-subsystem) and [Actors §2C — Verifier](actors_reasoning_model.md#2c-verifier-as-a-first-class-subsystem) must become real components, with explicit policies for:

- query rewriting
- entity- and time-conditioned retrieval
- perspective-conditioned retrieval
- counterevidence retrieval
- top-k fusion across episodic / semantic / state stores
- contradiction-aware retrieval and dedup
- the broaden ladder
- support / abstain thresholds and the verifier check catalog

Open work: implementation, calibration of thresholds against the eval set, and harness wiring through the standard `next_action` codes.

#### A.4 Harness runtime details

The hop/atomic refactor doc gives the right philosophy; what remains is making it a **fully executable runtime spec**. The drafted spec in [Atomic Skills & Hop Refactor — Harness runtime specification](atomic_skills_hop_refactor_execution_plan.md#harness-runtime-specification) covers hop length and termination, atomic I/O contract, composite expansion rules, trace logging format, MVP failure handling, failure localization, and reflection update hooks. Open work:

- implement `Harness.run_hop` / `retry_last_step` / `replay_step` against the canonical objects
- enforce the `max_hops` and `max_atomic_steps_per_hop` caps
- emit hooks consumed by the (later-phase) synthesizer
- add the audit harness that checks "no-op iterations are forbidden"

### Group B — Later-phase capability that should come after the base runtime works

These are gated by Group A. They should not be started until the base runtime is producing trustworthy traces.

#### B.1 Formal skill-bank spec

The bank direction is cleaner, but it still needs the **formal `SkillRecord`** schema, a **canonical starter atomic inventory**, and **formal trigger / verification formats**. The drafted versions live in [Skill Extraction / Bank §6 — Skill schema](skill_extraction_bank.md#6-skill-schema) (`SkillRecord`, `SkillUsage`, `SkillVersion`, `TriggerSpec`, `VerificationCheckSpec`) and [§4.9 — Canonical Starter Atomic Skill Inventory](skill_extraction_bank.md#49-canonical-starter-atomic-skill-inventory). Open work: serializer (`skill_bank.jsonl`), bank loader / validator, RAG-style selection layer that consumes triggers + embeddings.

#### B.2 Conservative skill synthesis rules

Synthesis must be rewritten **trace-first** rather than tag-first. The drafted pipeline in [Skill Synthetics Agents §1](skill_synthetics_agents.md#1-primary-synthesis-unit-successful-reasoning-traces) plus the verifiability / non-leakiness gates, **promotion thresholds** ([§ Promotion Thresholds](skill_synthetics_agents.md#promotion-thresholds)), and **bank versioning / rollback** ([§ Bank Versioning and Rollback](skill_synthetics_agents.md#bank-versioning-and-rollback)) describe the target. Open work: build the synthesizer behind a feature flag, with the **doubled phase-1 thresholds** ([§0.2](skill_synthetics_agents.md#02-phase-1-conservative-synthesis-policy)) and human-in-the-loop activation. Do **not** turn this on until A.4 is solid.

#### B.3 Controller training signal design

The actor doc says the controller is trainable and exposes traces, but a concrete reward/supervision table is needed. The drafted version is in [Actors §2E — Training Signals for the Controller](actors_reasoning_model.md#2e-training-signals-for-the-controller), covering decomposition, retrieval recall, evidence precision, perspective consistency, temporal consistency, abstention correctness, final-answer correctness, and **anti-hacking constraints** (over-retrieving, over-abstaining, no-progress hops, shortcut exploitation, verifier collusion, trace padding). Open work: implement the reward computation against `ReasoningTrace`, build the GRPO training loop, and validate the anti-hacking caps on a small probe set before scaling.

#### B.4 Evaluation and ablation plan

A dedicated benchmark + ablation plan must exist so the system's working pieces can be isolated. The drafted version is [`evaluation_ablation_plan.md`](evaluation_ablation_plan.md). Open work: lock the eval matrix (per benchmark family), the ablations (memory on/off, verifier on/off, abstention on/off, bank on/off, atomic-only vs composite), and the reporting format. This is what makes the [MVP success criterion](mvp_build_order.md#mvp-success-criterion) testable.

---

## 6. One-sentence summaries

**The system in one sentence.** A stable grounded-memory substrate plus an 8B hop-based reasoning controller, with frozen 72B specialists for difficult grounding and answer generation, and a reasoning-only skill bank that is meant to evolve later from verified traces rather than from infrastructure procedures.

**The remaining work in one sentence.** Mostly not a new big idea but the **execution glue**: canonical schemas, memory lifecycle, retriever/verifier policy, harness runtime spec, formal skill-bank records, conservative trace-based synthesis, and evaluation/ablation.

---

## 7. Next deliverable (suggested)

Turn this overview into:

- a single **architecture diagram** showing the three layers, the role split, and the canonical-object flow between modules; and
- a **module dependency order** that mirrors the phasing in [MVP Build Order](mvp_build_order.md), so the implementation team can pick up tasks in a topologically valid order without re-reading every plan file.
