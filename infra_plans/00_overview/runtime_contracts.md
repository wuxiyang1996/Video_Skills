# Runtime Contracts

> Goal: Define the **single canonical set of typed objects** that all major modules in the system pass to one another at runtime — grounding, memory, controller, harness, retriever, verifier, and skill bank. This file is the normative contract that the rest of `infra_plans` references.
>
> The detailed dataclass schemas live in [Actors §2A — Canonical Runtime Data Contracts](../03_controller/actors_reasoning_model.md#2a-canonical-runtime-data-contracts). This file is the **system-wide view**: which object each module produces, which it consumes, what every evidence-bearing object must carry, and what is forbidden.
>
> **Related plans:**
> - [Actors / Reasoning Model](../03_controller/actors_reasoning_model.md) — canonical dataclasses, retriever, verifier
> - [Agentic Memory](../02_memory/agentic_memory_design.md) — three stores + evidence layer that consume / emit these objects
> - [Atomic Skills & Hop Refactor](../04_harness/atomic_skills_hop_refactor_execution_plan.md) — harness runtime that produces `AtomicStepResult` / `HopRecord`
> - [Skill Extraction / Bank](../05_skills/skill_extraction_bank.md) — `SkillRecord` and skill I/O schemas built on these objects
> - [Skill Synthetics Agents](../05_skills/skill_synthetics_agents.md) — synthesizer that consumes `ReasoningTrace`
> - [Grounding Pipeline Execution Plan](../01_grounding/grounding_pipeline_execution_plan.md) — perception stack that emits `GroundedWindow`
> - [Training Plan: SFT to GRPO](../06_training/training_plan_sft_grpo.md) — SFT/GRPO consumes `ReasoningTrace`
> - [System Overview](system_overview.md) — integrative view

---

## 1. Core Principle

- **All major runtime modules communicate through canonical typed objects.** No ad hoc dict passing between subsystems.
- **Evidence-bearing objects must include `refs` and `confidence`.** Any output that makes a claim must point at the evidence it is based on or be explicitly marked unsupported.
- **Inferred objects must be explicitly marked.** Any value not directly read from grounded perception or stored memory carries `inferred=True` and (where applicable) populates `alternative_hypotheses`.
- **Stable enough for logging, SFT, GRPO, ablation.** The shape of these objects is what training and evaluation depend on; their fields are versioned, not improvised per run.

These rules apply uniformly. A module that wants to enrich its outputs with extra information must do so via a documented optional field or via the `meta: dict` slot — never by adding new top-level keys to canonical objects.

---

## 2. Object Definitions

The seven canonical objects below cover every cross-module hand-off in the system. Field-level dataclass definitions are normative in [Actors §2A](../03_controller/actors_reasoning_model.md#2a-canonical-runtime-data-contracts); the table here is the **summary contract**.

### 2.1 `GroundedWindow`

| Aspect | Contract |
|---|---|
| **Purpose** | A grounded slice of video / dialogue / state, the primary consumable for memory writers and atomic grounding skills |
| **Required fields** | `window_id`, `clip_id`, `time_span`, `entities`, `events`, `dialogue`, `spatial_state`, `keyframes`, `provenance`, `confidence` |
| **Optional fields** | `meta` |
| **Provenance / evidence** | Carries detector ids and model versions in `provenance`; `keyframes` and per-event references are the evidence pointers |
| **Confidence** | Window-level `confidence ∈ [0, 1]`; episodic write requires `confidence ≥ τ_grounding` |
| **Observed vs inferred** | `inferred` is **always `False`** for raw `GroundedWindow`s |
| **Producer** | Grounding pipeline ([Grounding Pipeline](../01_grounding/grounding_pipeline_execution_plan.md)) |
| **Consumer** | Memory writers (`build_episodic`, `update_state`), grounding atomic skills (`ground_event_span`, `ground_entity_reference`) |

### 2.2 `EvidenceBundle`

| Aspect | Contract |
|---|---|
| **Purpose** | The unit returned by the retriever and consumed by every reasoning skill that makes a claim — the **only** way evidence is passed between modules |
| **Required fields** | `bundle_id`, `refs: list[EvidenceRef]`, `query: RetrievalQuery`, `coverage`, `contradictions`, `sufficiency_hint`, `confidence` |
| **Optional fields** | `meta` (e.g. `meta.broaden_level`) |
| **Provenance / evidence** | Each `EvidenceRef` carries `source_id`, `time_span`, `entities`, `provenance ∈ {observed, inferred}`, `confidence` |
| **Confidence** | Aggregate `confidence` over refs; `sufficiency_hint` is the retriever's prior on sufficiency |
| **Observed vs inferred** | The bundle itself is `inferred=False`; individual refs may be inferred |
| **Producer** | Retriever ([Actors §2B](../03_controller/actors_reasoning_model.md#2b-retriever-as-a-first-class-subsystem)) |
| **Consumer** | Every reasoning skill that emits a claim; verifier; controller's answer composer |

### 2.3 `HopGoal`

| Aspect | Contract |
|---|---|
| **Purpose** | One hop's contract: what it must establish, with retrieval and termination hints |
| **Required fields** | `hop_id`, `parent_question_id`, `goal_text`, `target_claim_type`, `required_entities`, `success_predicate`, `max_atomic_steps` |
| **Optional fields** | `required_time_scope`, `perspective_anchor`, `retrieval_hints`, `meta` |
| **Provenance / evidence** | Not evidence-bearing itself; it specifies what evidence the hop must collect |
| **Confidence** | N/A — this is a planning artifact |
| **Observed vs inferred** | N/A |
| **Producer** | 8B controller / planner |
| **Consumer** | Harness; retriever (uses `retrieval_hints`) |

### 2.4 `AtomicStepResult`

| Aspect | Contract |
|---|---|
| **Purpose** | Output of every atomic skill invocation; the harness logs one per atomic call |
| **Required fields** | `step_id`, `hop_id`, `skill_id`, `inputs`, `output`, `output_type`, `verification`, `confidence`, `inferred`, `latency_ms` |
| **Optional fields** | `evidence: EvidenceBundle | None`, `failure_mode`, `meta` |
| **Provenance / evidence** | If the output is a claim, `evidence` must be a non-empty `EvidenceBundle` **or** `verification.next_action="abstain"` — the *evidence-or-abstain* rule |
| **Confidence** | Step-level `confidence ∈ [0, 1]` |
| **Observed vs inferred** | `inferred=True` if the output is inferred (vs a grounded read) |
| **Producer** | Harness (per atomic invocation) |
| **Consumer** | Verifier; `HopRecord` aggregator; reflection / synthesizer |

### 2.5 `VerificationResult`

| Aspect | Contract |
|---|---|
| **Purpose** | Local, per-step (or per-hop or per-final) verification record |
| **Required fields** | `passed`, `checks: list[VerificationCheck]`, `score`, `counterevidence`, `reasons`, `next_action` |
| **Optional fields** | `meta` |
| **Provenance / evidence** | `counterevidence` is a list of `EvidenceRef`; `checks[i].evidence_refs` cite what each check inspected |
| **Confidence** | `score ∈ [0, 1]` is the aggregate |
| **Observed vs inferred** | N/A — verification operates over already-tagged inputs |
| **Producer** | Verifier ([Actors §2C](../03_controller/actors_reasoning_model.md#2c-verifier-as-a-first-class-subsystem)) — and **only** the verifier sets `next_action` |
| **Consumer** | Harness (decides what to do next: continue / retry / broaden / switch_skill / abstain); controller |

### 2.6 `AbstainDecision`

| Aspect | Contract |
|---|---|
| **Purpose** | Emitted when the controller declines to answer; every abstention must point to which check failed |
| **Required fields** | `abstain`, `reason`, `blocking_checks`, `confidence_ceiling` |
| **Optional fields** | `last_evidence: EvidenceBundle | None`, `meta` |
| **Provenance / evidence** | `last_evidence` is the most informative bundle reached; `blocking_checks` cite specific `VerificationCheck.name` values |
| **Confidence** | `confidence_ceiling` is the best confidence reached before abstaining |
| **Observed vs inferred** | N/A |
| **Producer** | Verifier (`decide_abstain`) and / or controller (`compose_answer` opt-out) |
| **Consumer** | Final answer pipeline; eval harness; trace export |

### 2.7 `ReasoningTrace`

| Aspect | Contract |
|---|---|
| **Purpose** | The end-to-end record of a question's execution; the unit consumed by logging, GRPO, reflection, and (later) bank synthesis |
| **Required fields** | `trace_id`, `question_id`, `question_analysis`, `hops: list[HopRecord]`, `final_verification`, `bank_skill_ids_used`, `cost` |
| **Optional fields** | `final_claim`, `final_evidence`, `abstain`, `answer`, `meta` |
| **Provenance / evidence** | `final_evidence` is the assembled bundle for the answer; per-hop evidence is reachable via `hops[i].steps[j].evidence` |
| **Confidence** | Per-hop and per-step confidence are aggregated from child records |
| **Observed vs inferred** | Inheritance from constituent objects; the trace itself is the audit log |
| **Producer** | Harness (assembles); controller (closes with `answer` or `abstain`) |
| **Consumer** | Eval harness; SFT / GRPO ([Training Plan](../06_training/training_plan_sft_grpo.md)); reflection / synthesizer ([Skill Synthetics](../05_skills/skill_synthetics_agents.md)) |

`HopRecord` (per-hop record nested inside `ReasoningTrace.hops`) carries `hop_goal`, `steps: list[AtomicStepResult]`, `hop_verification: VerificationResult`, `outcome ∈ {resolved, blocked, abstain}`, `cost`, `meta` (including `composite_id` when relevant).

---

## 3. Object Flow

The canonical objects flow through the system in one direction per loop iteration. Each arrow is the only allowed wire format for that hand-off.

```
[grounding pipeline]
        │  GroundedWindow
        ▼
[memory layer: episodic / semantic / state]
        │  EvidenceRef (stored on records); read API returns EvidenceBundle
        ▼
[8B controller]
        │  HopGoal  (per hop)
        ▼
[harness]
        │  per atomic call: invokes skill with bound inputs;
        │  emits AtomicStepResult
        ▼
[verifier]
        │  VerificationResult (per step / per hop / per final)
        ▼
[8B controller]   ──► next_action: continue / retry / broaden / switch_skill / abstain
        │
        ▼
[final answer  OR  AbstainDecision]
        │
        ▼
[ReasoningTrace export]
        │
        ▼
[reflection / synthesizer]   (later phases)
```

Module-by-module summary:

| Hand-off | Object | Producer → Consumer |
|---|---|---|
| grounding → memory | `GroundedWindow` | Grounding pipeline → memory writers (fixed memory procedures) |
| memory → controller | `EvidenceBundle` (via retriever) | Memory + retriever → controller |
| controller → harness | `HopGoal`, selected `skill_id` | Controller → harness |
| harness → verifier | `AtomicStepResult` (per step), `HopRecord` (per hop) | Harness → verifier |
| verifier → controller | `VerificationResult` with `next_action` | Verifier → controller |
| controller → final answer / abstain | answer string + `final_evidence` **or** `AbstainDecision` | Controller → caller |
| trace → reflection / synthesis | `ReasoningTrace` | Harness/controller → reflection / synthesizer (offline) |

The retriever sits between memory and controller; the verifier sits between harness and controller. Neither talks to the bank or to memory writers directly — they only produce/consume canonical objects.

---

## 4. Contract Rules

These rules are normative and enforced at the harness boundary. A module that violates them is rejected at load / runtime.

1. **Canonical objects only.** No module may depend on raw free-form text blobs as its primary API. Free-form text appears only inside object fields (e.g. `goal_text`, `answer`) — never as the cross-module wire format.
2. **Evidence-or-abstain.** Every `AtomicStepResult` whose output is a claim must include either an `EvidenceBundle` with non-empty `refs` **or** a `VerificationResult.next_action="abstain"`.
3. **Inferred tagging is mandatory.** Any object whose value was not directly read from grounded perception or stored memory must set `inferred=True` and (where applicable) populate `alternative_hypotheses`.
4. **Versioning.** Each object carries an implicit `schema_version` (set by the serializer); the harness rejects unknown future versions and logs a downgrade path for older ones.
5. **Idempotent reads.** Memory retrievers return the same `EvidenceBundle` shape for the same `RetrievalQuery` within a session; perturbations require a new query id.
6. **No silent enrichment.** Modules may not add new top-level keys to canonical objects. Optional fields are documented per object; everything else goes into `meta: dict`.
7. **Single producer per object kind per loop.** One retriever produces `EvidenceBundle`s; one verifier produces `VerificationResult`s; one harness produces `AtomicStepResult`s. Cross-module duplication of producer roles is forbidden in v1.
8. **Stable for logging, SFT, GRPO, ablation.** The shapes above are what training and evaluation depend on. Schema changes require a release bump and a backfill plan; they do not happen mid-experiment.

These rules — together with the dataclass definitions in [Actors §2A](../03_controller/actors_reasoning_model.md#2a-canonical-runtime-data-contracts) — are what let the rest of the project be implemented as independent subsystems without producing module-specific dicts that break logging, SFT, GRPO, or ablation.
