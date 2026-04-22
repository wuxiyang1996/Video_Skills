# Atomic skills, hop composition & layer split — execution plan

> **Purpose:** Cursor-ready checklist and file-by-file edit plan (English) for refactoring `infra_plans` so reasoning skills stay **reusable inference operators**, coarse skills split into **atomic** + **composite** forms, **one hop** is a short verifiable chain of atomics, and **infrastructure** (observation, tracking, memory build, retrieval) is not stored as bank skills.
>
> **Related plans (apply together):**
> - [Skill Extraction & Skill Bank](../05_skills/skill_extraction_bank.md) — canonical bank design
> - [Skill Synthetics Agents](../05_skills/skill_synthetics_agents.md) — evolution & reflection
> - [Actors / Reasoning Model](../03_controller/actors_reasoning_model.md) — controller & hop execution
> - [Agentic Memory](../02_memory/agentic_memory_design.md) — three stores + evidence
> - [MVP Build Order](../00_overview/mvp_build_order.md) — phased implementation plan

---

## 0. Design Principle: Stable Memory, Evolving Reasoning

This file is the **execution-runtime spec** for the principle stated in [Actors §0](../03_controller/actors_reasoning_model.md#0-design-principle-stable-memory-evolving-reasoning):

- The harness executes hops over **fixed memory procedures** ([Agentic Memory §0.2](../02_memory/agentic_memory_design.md#02-memory-management-skills-vs-reasoning-skills)) and **frozen 72B grounding tools**.
- The harness **does not** evolve memory policies and **does not** edit the bank. It executes, verifies, and logs.
- The reasoning skills it expands and runs come from the **curated reasoning bank** ([Skill Extraction / Bank](../05_skills/skill_extraction_bank.md)). In phase 1 that bank does not grow during runtime.

---

## MVP Harness Mission

The harness is the **execution runtime** for hop-based reasoning. It is **not** the controller and **not** the skill bank.

- It executes short reasoning hops over structured memory and grounded evidence.
- It expands composite reasoning skills into atomic steps; composites never run as opaque blocks.
- It calls **fixed memory procedures** (write / update / refresh / compress) and **frozen large-model grounding tools** (72B) when the controller requests them.
- It logs **step-level traces** and **local verification results** through the canonical `AtomicStepResult` / `HopRecord` / `ReasoningTrace` objects.
- It does not select skills, modify memory policies, or write to the bank.

Tightening for MVP execution:

- **One hop = one short, verifiable reasoning segment.** A hop establishes one local subgoal toward the final answer.
- Each hop **may include a few atomic reasoning skills** chained together; it does not have to be a single atomic.
- **Memory procedures are callable runtime procedures**, exposed to atomic skills via the harness, **not entries in the evolving bank**.
- **Infrastructure calls** (retrieval, memory writes, 72B grounding) remain outside the evolving skill bank. The harness is their dispatcher.

---

## Overall objective

Refactor the plan so the system has three clean layers:

1. **Infrastructure / observer pipeline** — segmentation, entity tracking, episodic / semantic / state memory construction, grounding outputs.
2. **Reasoning skill bank** — atomic skills, composite skills, hop composition, verification skills.
3. **Reflection / evolution loop** — failure diagnosis, atomic-step localization, patch / split / promote / retire.

This aligns with the compact three-store memory design and the 8B controller as orchestrator over memory and perspective threads.

---

## Conceptual center (use in bank + actors)

One reasoning hop may involve a few atomic skills rather than a single monolithic reasoning skill. Atomic skills are the minimal reusable reasoning operators, while composite skills are stable, reusable short chains of atomic skills that can still be expanded for diagnosis and revision.

We treat one reasoning hop as a short, verifiable composition of several atomic skills rather than a monolithic reasoning step. Infrastructure operations such as observation, tracking, storage, and retrieval are not first-class reasoning skills; they are pipeline primitives that reasoning skills may call.

---

## Infrastructure primitives (not bank skills)

These remain callable system primitives but are **not** stored as reasoning skills in the bank:

- `observe_segment`
- `detect_entities`
- `build_episodic`
- `build_semantic`
- `update_state`
- `search_memory`

They belong to observation, memory, and retrieval infrastructure.

---

## File-by-file goals

### 1) `skill_extraction_bank.md` (main rewrite)

- Separate infrastructure primitives from reasoning skills.
- Introduce atomic skills, composite skills, and task-level policies.
- Define one reasoning hop as a short composition of atomic skills.
- Add atomic skill families (parsing, retrieval/grounding, temporal, causal, social, verification, reflection).
- Replace coarse composition examples with atomic-chain / hop examples.
- Use layered skill schema: `level`, `output_type`, `child_skills`, `verification_rule`, `failure_modes`, `repair_strategies`, etc.

### 2) `skill_synthetics_agents.md`

- Evolution unit: **atomic chain / hop / composite skill** (not only coarse skills).
- **Reflection unit:** atomic-step failure localization (observation vs reasoning vs verification; localize failed atomic skill(s) within a hop).
- Failure categories (observation-side, reasoning-side, verification-side).
- Update rules: patch, split, promote, merge, retire.
- Promotion criteria for atomic chain → composite (tie back to bank design).

### 3) `actors_reasoning_model.md`

- Controller operates over: atomic skills, composite skills, hop traces, reflection outputs.
- Skill execution granularity: hops = short atomic compositions; composites expandable to atomic traces.
- Reasoning loop with verification, abstain, and reflection trace emission.
- Explicit controller outputs during reasoning.

### 4) `agentic_memory_design.md`

- Short clarification: reasoning skills operate **over** memory outputs; they do not replace episodic / semantic / state functions.

---

## Constraints

- Keep the system **compact**; do not add extra top-level memory types.
- Do not treat generic tool calls as first-class reasoning skills.
- Do not make dataset-specific templates the main bank content.
- Emphasize reusable, verifiable, evolvable reasoning operators.

---

## Minimal atomic skill set (implementation v1)

**First pass (12):**

- `identify_question_target`
- `decompose_into_subgoals`
- `retrieve_relevant_episode`
- `ground_entity_reference`
- `ground_event_span`
- `infer_observation_access`
- `order_two_events`
- `check_state_change`
- `check_causal_support`
- `update_belief_state`
- `check_evidence_sufficiency`
- `decide_answer_or_abstain`

**Second pass add:** `check_alternative_hypothesis`, `locate_counterevidence`, `classify_failure_type`, `localize_failed_step`.

---

## Suggested Cursor task list

1. **skill_extraction_bank.md:** infrastructure vs skills; atomic/composite/hops; families; hop examples; layered schema.
2. **skill_synthetics_agents.md:** localization; failure categories; patch/split/promote/merge/retire; promotion criteria.
3. **actors_reasoning_model.md:** hop planning; composite expansion; reasoning loop; traces and outputs.
4. **agentic_memory_design.md:** skills vs memory substrate paragraph.

---

## Harness runtime specification

The remainder of this document specifies the **Harness** module — the runtime that executes hops over atomic skills, talks to the retriever and verifier, and emits the canonical `ReasoningTrace`. It is implementation-ready: an engineer should be able to build the `Harness` class directly from the sections below.

The harness consumes the canonical objects defined in [Actors §2A](../03_controller/actors_reasoning_model.md#2a-canonical-runtime-data-contracts): `HopGoal`, `AtomicStepResult`, `EvidenceBundle`, `VerificationResult`, `ReasoningTrace`. It does not invent new wire formats.

### Definition of a Hop

- A **hop** is a short, verifiable reasoning segment that establishes one local subgoal toward the final answer.
- A hop is **not** the final answer. Composing hops into an answer is the controller's job, not the harness's.
- A hop **may include a few atomic skills** chained together; it does not have to be a single atomic.
- A hop **ends** when one of: (a) its `success_predicate` is satisfied with verifier `passed=True`, (b) the verifier returns `next_action="abstain"`, (c) it is blocked (max steps reached, or skill repeatedly fails verification), or (d) the controller revokes it via `switch_skill`.

A hop owns: one `HopGoal`, an ordered list of `AtomicStepResult`s, a `hop_verification: VerificationResult`, and an `outcome ∈ {resolved, blocked, abstain}`.

### Allowed Hop Length and Termination

| Quantity | Default | Hard cap | Notes |
|---|---|---|---|
| Atomic steps per hop | 3 (typical) | `max_atomic_steps_per_hop = 6` | Hops longer than 4 steps should be candidates for splitting |
| Hops per question | 2–4 (typical) | `max_hops = 6` | Above 6 → automatic abstain |
| Verifier retries per step | 1 | 2 | Each retry counts as an atomic step against the cap |
| Broaden escalations per hop | 1 | 2 | Each broaden invokes the retriever ladder ([Actors §2B.3](../03_controller/actors_reasoning_model.md#2b3-broaden-ladder)) |

Termination triggers:

- **Split-when-too-long.** If a hop exceeds 4 atomic steps without progress (`VerificationResult.score` not improving by `δ_hop`), the harness aborts the hop and asks the controller to either decompose it into two hops or switch skill.
- **Stop-and-retrieve.** If an atomic step emits `output_type="claim"` with empty or stale `EvidenceBundle`, the harness blocks the hop and routes back through the retriever before continuing.
- **Fail-fast.** If two consecutive atomic steps in the same hop fail verification, the hop is marked `blocked` and the controller is asked to switch skill or abstain.

### Atomic-Step Input/Output Contract

Every atomic skill must declare the following five attributes (see [Skill Bank — Formal SkillRecord Schema](../05_skills/skill_extraction_bank.md#formal-skillrecord-schema)):

| Field | Required | Description |
|---|---|---|
| `input_schema` | yes | Named, typed inputs the harness must populate before invoking the skill |
| `output_schema` | yes | Named, typed outputs the skill must produce; matches `AtomicStepResult.output` keys |
| `required_memory_fields` | yes | Memory subfields the skill reads (e.g. `state.social.belief`, `episodic.events`); harness pre-fetches these |
| `retrieval_hints` | optional | Default `RetrievalQuery` templates the harness should issue if no upstream `EvidenceBundle` is provided |
| `verification_rule` | yes | Local, deterministic checks the verifier runs on this skill's output (see *Local Verification Format*) |

The harness is responsible for:

1. building the `inputs` dict from the running trace + memory + retriever output, conforming to `input_schema`;
2. calling the skill's executable;
3. validating that the returned `output` matches `output_schema` (else: `failure_mode="schema_violation"`);
4. wrapping the result in an `AtomicStepResult` and handing it to the verifier.

#### Failure modes (harness-recognized)

Every skill enumerates its `failure_modes`. The harness recognizes the following standard codes in addition to skill-specific ones:

- `schema_violation` — output did not match `output_schema`
- `missing_input` — required input field was unresolved at call time
- `empty_evidence` — required `EvidenceBundle` was empty after retrieval + broaden
- `verification_failed` — verifier returned `passed=False`
- `timeout` — skill exceeded its per-call latency budget (default 5s for atomics)
- `exception` — skill raised; output is `None` and the trace is recoverable

### Local Verification Format

`verification_rule` is a list of named, deterministic checks. Each check has the form:

```python
@dataclass
class VerificationCheckSpec:
    name: str                           # one of the catalog in Actors §2C.1, or skill-specific
    inputs: list[str]                   # AtomicStepResult.output keys this check reads
    predicate: str                      # symbolic / textual rule (e.g. "evidence.refs ⊇ output.claim.entities")
    threshold: float | None             # numeric gate if applicable
    on_fail: str                        # "retry" | "broaden" | "switch_skill" | "abstain" | "continue"
```

The verifier evaluates each check in order, short-circuiting on `passed=False` only when `on_fail != "continue"`. The aggregated `VerificationResult.next_action` is the most-severe `on_fail` triggered (severity: `continue < retry < broaden < switch_skill < abstain`).

A skill **must** provide at least one check whose `name` is one of `{claim_evidence_alignment, evidence_sufficiency, temporal_consistency, perspective_consistency, entity_consistency}`, depending on its `output_type`.

### Composite Expansion Rules

A composite skill is a stable named macro over an ordered list of child skill ids (`child_skills`). At execution time, the harness expands the composite into an explicit atomic chain — the composite **never** executes as an opaque block.

Expansion rules:

1. **Each child step emits its own `AtomicStepResult`.** The composite's record in the trace is a `HopRecord` whose `steps` are the child results, plus a synthetic `composite_id` field on `HopRecord.meta`.
2. **Each child runs its own `verification_rule`.** A composite is *not* allowed to suppress child-level verification.
3. **Composite success.** A composite is judged successful when (a) every child step's verifier returned `passed=True`, **and** (b) the composite's own top-level check (declared in its own `verification_rule`) passes against the final child's output.
4. **Composite failure.** Failure of any child step propagates as a failure of the composite. The synthesizer's failure-localization step (below) identifies which child caused the failure.
5. **Re-expansion on bank update.** If the bank revises the composite's `child_skills` (see [Skill Synthetics — Promotion Thresholds](../05_skills/skill_synthetics_agents.md#promotion-thresholds)), already-running hops finish under the old expansion; new hops use the new expansion.

### Trace Logging Format

The harness writes every hop into the canonical `ReasoningTrace`. The minimum logged content per hop:

```python
@dataclass
class HopRecord:
    hop_goal: HopGoal
    steps: list[AtomicStepResult]       # each carries inputs, output, evidence, verification, confidence
    hop_verification: VerificationResult
    outcome: str                        # "resolved" | "blocked" | "abstain"
    cost: dict                          # {atomic_steps, retrieval_calls, broaden_levels, latency_ms}
    meta: dict                          # composite_id (if any), broaden history, skill switch history
```

Logging requirements:

- **Evidence attachments** are referenced by id (`EvidenceRef`) on each `AtomicStepResult`. Raw payloads are not duplicated in the trace; the trace is a thin record over the index.
- **Confidence logging** is mandatory at the step, hop, and final levels (`AtomicStepResult.confidence`, `VerificationResult.score`, `ReasoningTrace.cost`).
- **Failure-type logging** uses the standard `failure_mode` codes above plus any skill-declared codes; reflection consumes these directly.
- **Provenance.** Every `AtomicStepResult.inputs` records which trace keys / memory ids it pulled from, so reflection can reconstruct the read pattern.

### Failure Localization Protocol

When a hop's `outcome ∈ {blocked, abstain}` or the final answer is wrong, the harness (or the offline reflection job) must classify the failure into exactly one of the following buckets, in priority order:

| Bucket | Detection signal |
|---|---|
| **Wrong atomic reasoning** | An atomic step's `VerificationResult.passed=False` with `failure_mode="verification_failed"` and `evidence` non-empty (the evidence existed; the step misused it) |
| **Missing retrieval** | Step's `failure_mode="empty_evidence"` after broaden ladder exhausted; or evidence found later by reflection that the retriever missed |
| **Bad grounding** | Cited evidence has `provenance="observed"` but the underlying `GroundedWindow` had `confidence < τ_grounding` (perception was wrong) |
| **Unsupported final claim despite correct intermediates** | Every hop's `hop_verification.passed=True`, but `verify_final` flips to `passed=False` with `claim_evidence_alignment` failing on the answer |
| **Perspective mismatch** | Any `perspective_consistency` check failed; or final answer used global state where the question was perspective-bound |
| **Premature answer** | Trace skipped a required atomic family (e.g. answered a temporal question without an `order_two_events` step), or `confidence > 0.8` while `hop_verification.score < abstain_threshold` |

The detected bucket determines which side updates apply (see *Reflection Update Hooks* below and the failure-to-update mapping in [Skill Synthetics §6](../05_skills/skill_synthetics_agents.md#6-failure--update-mapping)).

### MVP Failure Handling

In the MVP phase the harness does **not** attempt autonomous repair. It detects failure, classifies it, and routes control back to the 8B controller (or, after the run, to the offline reflection job). The five failure shapes the MVP harness must handle explicitly are:

| MVP failure shape | What the harness does | What it does NOT do in v1 |
|---|---|---|
| **Retrieval failure** (`empty_evidence` after broaden ladder) | Mark hop `blocked`; return control to controller for `switch_skill` or `abstain`; log query trace | Auto-rewrite retrieval policy; auto-edit retriever config |
| **Grounding failure** (cited evidence below `τ_grounding`, or 72B tool returned low-confidence span) | Mark step `failure_mode="bad_grounding"`; offer one re-ground via 72B if budget permits; otherwise route to controller | Re-train the 72B; auto-promote a "better grounding" skill |
| **Unsupported claim** (`claim_evidence_alignment` failed on a step or final answer) | Block emission of the claim; surface the failed check to the controller; verifier may force `abstain` | Silently emit the claim with a confidence penalty |
| **Perspective mismatch** (`perspective_consistency` failed) | Force a re-run with the perspective anchor bound, or route to `switch_skill` if no perspective-aware atomic exists | Auto-add a new perspective skill to the bank |
| **Premature answer** (required atomic family skipped, or `confidence > 0.8` while `hop_verification.score < abstain_threshold`) | Block answer emission; require the missing atomic family or trigger `decide_answer_or_abstain` | Auto-patch the controller's policy from a single example |

Phase-1 scope discipline: the harness is intentionally narrow. It is **focused on execution, logging, and verification**. It does not implement full autonomous evolution; the [synthesizer](../05_skills/skill_synthetics_agents.md) consumes the traces it emits and decides what (if anything) to change about the bank, under the v1 conservative-promotion policy.

### Reflection Update Hooks

The harness exposes hooks that the [Skill Synthesizer / Crafter](../05_skills/skill_synthetics_agents.md) consumes. Hooks fire on hop completion (success and failure) and on final verification.

| Hook | Trigger | Trace slice exported | Suggested actions for the synthesizer |
|---|---|---|---|
| `on_hop_success` | `outcome="resolved"` and `hop_verification.passed=True` | The full `HopRecord` plus the upstream `EvidenceBundle` and `QuestionAnalysis` | Increment success counters on used skills; consider promotion if pattern repeats |
| `on_hop_failure` | `outcome ∈ {blocked, abstain}` or any step `failure_mode != None` | The `HopRecord`, the failed steps, and the immediate upstream/downstream context | `patch` (revise verification rule), `split` (carve narrower variants), or `retire` (consistently failing) |
| `on_composite_failure` | Composite expansion failed at a specific child | `HopRecord` + identified failing child id + the child's `inputs` | `split` the composite, or `patch` the child step's `verification_rule` |
| `on_final_failure` | `verify_final.passed=False` despite all hops passing | Full `ReasoningTrace` (no truncation) | Tighten cross-hop consistency checks; introduce a verification atomic between the last hop and answer |
| `on_promotion_candidate` | A previously-unseen atomic chain succeeded `≥ N_promote` times across distinct hops | The recurring chain + per-occurrence `HopRecord`s | `promote` to composite (subject to promotion thresholds in [Skill Synthetics](../05_skills/skill_synthetics_agents.md#promotion-thresholds)) |

Hooks are pure read-side: they emit events; the synthesizer decides what to write back to the bank, and bank writes go through the versioning policy ([Skill Synthetics — Bank Versioning and Rollback](../05_skills/skill_synthetics_agents.md#bank-versioning-and-rollback)).

### Harness module sketch

```python
class Harness:
    def __init__(self, bank: SkillBank, retriever: Retriever, verifier: Verifier,
                 memory: MemoryStores, max_hops: int = 6,
                 max_atomic_steps_per_hop: int = 6): ...

    def run_hop(self, hop_goal: HopGoal, skill: Skill) -> HopRecord: ...
    def retry_last_step(self, hop: HopRecord) -> AtomicStepResult: ...
    def replay_step(self, hop: HopRecord) -> AtomicStepResult: ...

    def emit_hooks(self, trace: ReasoningTrace) -> list[ReflectionEvent]: ...
```

`run_hop` is the inner loop: build inputs → optionally retrieve → call skill → verify → loop until terminator. It is the **only** caller of atomic skills inside the runtime.

---

*This document is now both the refactor reference and the normative spec for the harness; the canonical wire formats live in `actors_reasoning_model.md`, the bank schema in `skill_extraction_bank.md`, and the synthesis side in `skill_synthetics_agents.md`.*
