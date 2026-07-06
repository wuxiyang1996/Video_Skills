# Skill Extraction & Skill Bank — Design Plan

> Goal: Define the **skill bank infrastructure** — how **reasoning** skills are represented, stored, composed, and evolved. Skills are **reusable inference operators** over memory outputs and perspective threads, not generic retrieval routines or offline pipeline steps.
>
> **Related plans:**
> - [Atomic skills & hop refactor — execution checklist](../04_harness/atomic_skills_hop_refactor_execution_plan.md) — layer split, constraints, minimal atomic set
> - [Agentic Memory](../02_memory/agentic_memory_design.md) — episodic / semantic / state stores + evidence layer
> - [Video Benchmarks & Grounding](../01_grounding/video_benchmarks_grounding.md) — benchmarks, memory graph, adapters
> - [Actors / Reasoning Model](../03_controller/actors_reasoning_model.md) — 8B controller, hops, orchestration
> - [Skill Synthetics Agents](skill_synthetics_agents.md) — synthesis, evolution, reflection
> - [MVP Build Order](../00_overview/mvp_build_order.md) — phased implementation plan

---

## 0. Design Principle: Stable Memory, Evolving Reasoning

This file defines the **evolving reasoning** half of the cross-plan principle ([Actors §0](../03_controller/actors_reasoning_model.md#0-design-principle-stable-memory-evolving-reasoning)). The bank evolves; memory does not.

### 0.1 Bank Scope in Phase 1

- The evolving bank stores **reasoning skills only**, not infrastructure primitives.
- **Observation, tracking, storage, memory write / update, and retrieval** are **runtime procedures** (or, for memory, **fixed memory procedures** in the [Memory Procedure Registry](../02_memory/agentic_memory_design.md#02-memory-management-skills-vs-reasoning-skills)). They are **not** entries in this bank.
- The primary bank contents are **atomic and composite reasoning operators** that consume memory + evidence and emit verifiable claims.

### 0.2 Phase-1 Bank Policy

- Start with a **curated starter set** of atomic reasoning skills (see §4.8 / §4.9).
- **Do not allow unconstrained free growth** of the bank in v1. New atomics are added only via human-authored release, not via online synthesis.
- Composite skills may be added **conservatively** in phase 2 from repeated successful atomic chains, gated by the synthesizer's promotion thresholds.
- Promotion is **limited and high-threshold** in the MVP phase. The defaults in §6 (`N_repeat`, `τ_stable`, `σ_stable`, transfer requirement) are minimums, not aspirations.
- **Bank evolution is not a v1 milestone.** It is sequenced after retrieval, verifier, and harness are stable ([MVP Build Order](../00_overview/mvp_build_order.md)).

### 0.3 Separate Registries

The system maintains **two distinct registries**. They share neither schema nor lifecycle, and they are not collapsed into one undifferentiated "skills" store:

| Registry | Contents | Lifecycle |
|---|---|---|
| **Memory Procedure Registry** ([Agentic Memory §0.2](../02_memory/agentic_memory_design.md#02-memory-management-skills-vs-reasoning-skills)) | Fixed memory-management procedures: `open_episode_thread`, `append_grounded_event`, `update_entity_profile`, `refresh_state_memory`, `compress_episode_cluster`, `attach_evidence_ref`, `resolve_entity_alias`, `revise_belief_state`, `mark_memory_conflict`, … | Stable; manually versioned between releases; never modified by trace-driven synthesis |
| **Reasoning Skill Bank** (this document) | Atomic and composite reasoning operators: `identify_question_target`, `retrieve_relevant_episode`, `order_two_events`, `infer_observation_access`, `update_belief_state`, `check_evidence_sufficiency`, `decide_answer_or_abstain`, … | Curated in v1; conservative promotion in phase 2; broader synthesis in phase 3, all under versioned, gated synthesis rules |

A reasoning skill **never writes memory directly**: when it needs to mutate state, it requests a Memory Procedure Registry entry via the harness. This boundary is what lets reasoning skills evolve without destabilizing the substrate.

---

## 1. Core principle: skills are reasoning operators, not infrastructure steps

Skills in this system are **reusable inference operators** — each encodes a verifiable reasoning pattern over episodic / semantic / state memory outputs, evidence attachments, and perspective threads.

**Infrastructure** (observation, entity tracking, memory construction, index search) exposes **primitives** the controller calls; those primitives are **not** first-class entries in the reasoning skill bank.

One reasoning hop may involve a few atomic skills rather than a single monolithic reasoning skill. Atomic skills are the minimal reusable reasoning operators, while composite skills are stable, reusable short chains of atomic skills that can still be expanded for diagnosis and revision.

We treat **one reasoning hop** as a **short, verifiable composition** of several atomic skills rather than a monolithic reasoning step. Composite skills **emerge** from frequently successful atomic chains (see §8).

### 1.1 Infrastructure primitives (callable, not bank skills)

These operations remain first-class **system** capabilities but are **not** stored as reasoning skills in the bank:

| Primitive | Role |
|-----------|------|
| `observe_segment` | Run observation on one clip; produce structured observations |
| `detect_entities` | Face/voice/ID detection; entity graph updates |
| `build_episodic` | Write timestamped episodic records with entity links and evidence |
| `build_semantic` | Distill episodic clusters into semantic summaries |
| `update_state` | Refresh query-time **state memory** (social + spatial subfields) |
| `search_memory` | Embed a query and return top-k matches from the graph / stores |

Reasoning skills **consume** the outputs of these primitives; they are not duplicates of them.

**Do not** treat generic tool calls or dataset-specific answer templates as the main bank content. The bank emphasizes **reusable, verifiable, evolvable** reasoning operators.

---

## 2. Skill hierarchy

### 2.1 Atomic skills

**Atomic skills** are the **minimal** reusable reasoning operators: single-purpose steps with clear inputs, outputs, and verification hooks (e.g. `order_two_events`, `ground_entity_reference`).

### 2.2 Composite skills

**Composite skills** bundle a **stable** sequence of atomic skills under one `skill_id` for efficient invocation. They must remain **expandable** into an explicit atomic trace for debugging, localization, and bank maintenance.

Legacy names such as `who_saw_what` or `infer_belief_update` are best modeled as **composite** macros over atomics (see §5.2), not as opaque monoliths.

### 2.3 Task-level reasoning policies

**Task-level policies** map question families / benchmarks to **preferred hop sequences** or **skill family** emphasis (e.g. more temporal atomics for ordering questions). Policies are **routing hints** for the controller, not additional top-level memory types.

---

## 3. One reasoning hop as a composition of atomic skills

A **hop** is one unit of progress toward an answer: a small goal (e.g. “establish order of A and B”, “determine whether Alice could know X”) realized by **chaining atomics**, each producing an intermediate claim or structured result that the next step can check.

Multi-hop QA is **multiple hops**, each with its own atomic chain and verification boundary.

---

## 4. Atomic skill families

### 4.1 Question parsing skills

- `identify_question_target`
- `identify_question_type`
- `extract_time_anchor`
- `decompose_into_subgoals`
- `identify_missing_information`

### 4.2 Retrieval and grounding skills

- `retrieve_relevant_episode`
- `retrieve_entity_thread`
- `retrieve_state_snapshot`
- `ground_event_span`
- `ground_entity_reference`
- `check_visibility`
- `check_co_presence`
- `locate_supporting_evidence`
- `locate_counterevidence`

### 4.3 Temporal skills

- `order_two_events`
- `trace_event_sequence`
- `check_state_change`
- `align_cause_before_effect`

### 4.4 Causal skills

- `propose_candidate_causes`
- `check_causal_support`
- `compare_candidate_causes`
- `link_action_to_outcome`

### 4.5 Social / belief skills

- `infer_observation_access`
- `update_belief_state`
- `compare_beliefs_between_agents`
- `detect_belief_conflict`
- `infer_intention_from_behavior`
- `infer_deception_possibility`
- `trace_social_state_transition`

### 4.6 Verification skills

- `check_evidence_sufficiency`
- `check_reasoning_gap`
- `check_alternative_hypothesis`
- `calibrate_answer_confidence`
- `decide_answer_or_abstain`

### 4.7 Reflection skills (for evolution / repair; may run post-hoc)

- `classify_failure_type`
- `localize_failed_step`
- `propose_repair_action`
- `promote_trace_to_composite_skill`
- `merge_redundant_skills`
- `retire_unreliable_skill`

### 4.8 Minimal starter set (implementation v1)

For a practical first version, implement **12** atomics:

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

**Second pass:** add `check_alternative_hypothesis`, `locate_counterevidence`, `classify_failure_type`, `localize_failed_step`.

### 4.9 Canonical Starter Atomic Skill Inventory

The starter inventory below is the **shipping v1 bank**: a small but representative atomic set, grouped by reasoning purpose, that covers all benchmark families targeted by [Video Benchmarks](../01_grounding/video_benchmarks_grounding.md). Every entry must conform to the `SkillRecord` schema (§6).

| Category | Skill | `output_type` | One-line role |
|---|---|---|---|
| **Entity grounding support** | `ground_entity_reference` | `entity_ref` | Resolve a question's entity mention to a `character_id` |
| | `check_co_presence` | `claim` | Decide whether two entities share a time / space window |
| | `check_visibility` | `claim` | Decide whether entity X could have observed event E |
| **Temporal linking** | `order_two_events` | `ordering` | Establish before/after between two grounded events |
| | `align_cause_before_effect` | `claim` | Verify that a candidate cause precedes its effect |
| | `check_state_change` | `claim` | Detect whether a state predicate flipped over a span |
| **Causal linking** | `propose_candidate_causes` | `set[claim]` | Enumerate plausible causes for an observed effect |
| | `check_causal_support` | `claim` | Score evidence supporting one cause→effect link |
| **Belief update** | `infer_observation_access` | `claim` | Decide whether agent had perceptual access to event |
| | `update_belief_state` | `belief` | Apply an evidence-justified update to an agent's belief |
| **Perspective check** | `check_perspective_anchor` | `meta` | Confirm the question is bound to a specific viewpoint and tag it |
| | `compare_beliefs_between_agents` | `claim` | Check whether two agents hold matching/diverging beliefs |
| **Contradiction check** | `detect_belief_conflict` | `claim` | Surface contradictions between belief states or evidence |
| | `locate_counterevidence` | `evidence_set` | Retrieve refuting evidence for a candidate claim |
| **Evidence sufficiency** | `check_evidence_sufficiency` | `claim` | Decide whether the cited bundle covers the claim |
| | `locate_supporting_evidence` | `evidence_set` | Retrieve refs that support a candidate claim |
| **Alternative hypothesis check** | `check_alternative_hypothesis` | `claim` | Compare the chosen claim against the strongest alternative |
| **Answer vs abstain decision** | `calibrate_answer_confidence` | `meta` | Aggregate per-hop confidences into an answer-level score |
| | `decide_answer_or_abstain` | `decision` | Emit final answer or `AbstainDecision` based on thresholds |

This inventory is small enough to be hand-curated in v1 and rich enough that the synthesizer can build composites from successful chains over it. New atomics are added only after a sustained pattern emerges in failure logs.

---

## 5. Composite skill examples and worked hops

### 5.1 One reasoning hop as a composition of atomic skills — examples

**Example A: temporal hop**

*Question:* Did event A happen before event B?

*Hop:*

```
identify_question_target
  -> ground_event_span(A)
  -> ground_event_span(B)
  -> order_two_events
  -> check_evidence_sufficiency
```

**Example B: belief hop**

*Question:* Did John know the key was moved?

*Hop:*

```
ground_entity_reference(John)
  -> retrieve_entity_thread
  -> ground_event_span(key moved)
  -> infer_observation_access
  -> update_belief_state
  -> check_evidence_sufficiency
```

**Example C: deception hop**

*Question:* Why does Alice think Bob is lying?

*Hop:*

```
identify_question_target
  -> decompose_into_subgoals
  -> retrieve_relevant_episode
  -> compare_beliefs_between_agents
  -> detect_belief_conflict
  -> locate_supporting_evidence
  -> check_alternative_hypothesis
```

### 5.2 Social composites as macros over atomics

Earlier **coarse** social operators can be expressed as composites, e.g.:

- `who_saw_what` → `ground_event_span` → `check_visibility` / `check_co_presence` → `retrieve_entity_thread` fragments as needed
- `infer_belief_update` → `infer_observation_access` → `update_belief_state` → `check_evidence_sufficiency`
- `deception_hypothesis_check` → `compare_beliefs_between_agents` → `locate_counterevidence` → `check_alternative_hypothesis`

---

## 6. Skill schema

### Formal SkillRecord Schema

`SkillRecord` is the **canonical, serializable, versioned** record for every entry in the bank. It is the unit the harness ([Atomic Skills & Hop Plan — Harness](../04_harness/atomic_skills_hop_refactor_execution_plan.md#harness-runtime-specification)) loads at startup, the synthesizer ([Skill Synthetics](skill_synthetics_agents.md)) writes back to, and the retriever's RAG layer indexes.

```python
@dataclass
class SkillRecord:
    skill_id: str                       # globally unique, stable across versions
    name: str                           # human-readable, snake_case
    type: Literal["atomic", "composite"]
    family: str                         # one of §4.x families
    trigger_conditions: list[TriggerSpec]    # see Trigger and Verification Formats
    input_schema: dict                  # name -> {type, required, source_hint}
    output_schema: dict                 # name -> {type, required}
    output_type: str                    # claim | span | ordering | belief | presence | abstain | meta
    verification_rule: list[VerificationCheckSpec]
    failure_modes: list[str]            # standard codes + skill-specific codes
    required_memory_fields: list[str]   # e.g. "state.social.belief", "episodic.events"
    retrieval_hints: list[RetrievalQuery]    # default queries when no upstream evidence
    required_primitives: list[str]      # e.g. "search_memory" (infra, not bank skills)
    protocol_steps: list[str]           # textual / pseudo-code steps, COS-PLAY-compatible
    child_links: list[str]              # ordered child skill_ids (composite only; empty for atomic)
    parent_links: list[str]             # composites this atomic/composite is part of
    usage_stats: SkillUsage             # invocations, success_rate, avg_confidence, transfer_rate
    version: SkillVersion               # see Bank Versioning in skill_synthetics_agents.md
    examples: list[dict]                # illustrative HopRecord excerpts
    meta: dict                          # free-form, non-canonical
```

`SkillUsage` carries `n_invocations`, `n_success`, `n_failure_by_mode: dict[str,int]`, `avg_confidence`, `transfer_rate` (success rate on tasks outside the skill's training family), and `last_updated`.

`SkillVersion` carries `version_id`, `parent_version_id`, `created_at`, `created_by` (`crafted | promoted | merged | split | patched`), `status` (`active | shadow | retired`).

**Required fields for every skill:** `skill_id`, `name`, `type`, `family`, `trigger_conditions`, `input_schema`, `output_schema`, `verification_rule`, `failure_modes`, `required_memory_fields`, `usage_stats`, `version`. All others are optional with sensible defaults.

**Field roles (summary):**

| Field | Role |
|-------|------|
| `type` | Atomic vs composite |
| `output_type` | Structured output expectation (claim, span, ordering, belief snapshot, abstain, …) |
| `child_links` | Ordered atomic (or nested composite) IDs for expansion |
| `parent_links` | Reverse pointer for impact analysis when patching/retiring |
| `required_primitives` | Infrastructure / retrieval calls the skill assumes |
| `verification_rule` | How to check the step locally (see *Trigger and Verification Formats*) |
| `failure_modes` | For reflection and bank maintenance |
| `version` | Enables rollback, shadow deployment, and audit |

Optional compatibility shim: map `protocol_steps` ↔ COS-PLAY `Protocol.steps`, and keep `contract` / `SkillEffectsContract` for existing code paths (`skill_agents/stage3_mvp/schemas.py`). The previous `Skill(...)` shape is preserved as a backwards-compatible projection of `SkillRecord`.

### Trigger and Verification Formats

`TriggerSpec` and `VerificationCheckSpec` are formal so that the controller's selection layer and the verifier can both read them without skill-specific glue.

```python
@dataclass
class TriggerSpec:
    kind: Literal["question_type", "entity_present", "time_anchor",
                  "perspective_required", "predicate", "embedding_match"]
    value: str | dict                   # e.g. "TCI" for question_type, or {"k":3,"sim":0.6}
    weight: float = 1.0                 # contributes to skill selection score
```

A trigger fires when its `kind`-specific predicate evaluates true against the current `QuestionAnalysis` and `ReasoningTrace`. The selection layer aggregates trigger weights with embedding similarity (see §9.3).

```python
@dataclass
class VerificationCheckSpec:
    name: str                           # one of Actors §2C.1 catalog or skill-specific
    inputs: list[str]                   # AtomicStepResult.output keys consumed
    predicate: str                      # symbolic / textual rule
    threshold: float | None
    on_fail: Literal["retry", "broaden", "switch_skill", "abstain", "continue"]
```

Every atomic skill must declare at least one check whose `name` belongs to the verifier catalog ([Actors §2C.1](../03_controller/actors_reasoning_model.md#2c1-check-catalog)) and whose `inputs` reference real keys in `output_schema`. The harness rejects skills that violate this at load time.

### Composite Skill Formation Rules

Composites are not authored by hand; they are **promoted** from observed atomic chains by the synthesizer. The bank enforces:

| Rule | Threshold (default) | Notes |
|---|---|---|
| **Minimum repetition** | An atomic chain must appear ≥ `N_repeat = 5` times across distinct hops | Counts only chains where the trailing step's `verification.passed=True` |
| **Verification stability** | Mean `hop_verification.score` ≥ `τ_stable = 0.7` over the repetitions | Variance must also be < `σ_stable = 0.15` |
| **Transfer threshold** | Chain succeeded on ≥ 2 distinct task families (§7 mapping) | Prevents promotion of benchmark-specific shortcuts |
| **Verification rule inheritance** | Promoted composite inherits the union of its children's `verification_rule`s, plus a top-level check on the final child's output | Required before the composite can be activated |
| **Rollback rule** | If a promoted composite's success rate drops below `τ_stable - 0.15` for ≥ 20 invocations, it is rolled back to `status="shadow"` and its children are re-exposed for selection | Rollback is a versioned operation (see [Skill Synthetics — Bank Versioning](skill_synthetics_agents.md#bank-versioning-and-rollback)) |

Promotion always produces a new `SkillRecord` with `version.created_by="promoted"`; the children retain `parent_links` pointing to the new composite.

### Reasoning Skills vs Scene/Action Tags

Scene tags, action tags, and intention tags (e.g. `OBSERVE | INTERACT | NAVIGATE | ...` from the COS-PLAY video intention taxonomy) are **auxiliary metadata** attached to grounded segments by the perception pipeline. They are useful for retrieval filtering and for clustering during synthesis, but they are **not** the primary skill ontology.

The primary content of the bank is **reasoning skills** as defined above: atomic operators that consume memory + evidence and emit verifiable claims, plus composites built from them. A reasoning skill may *use* a scene or intention tag as one of its `trigger_conditions` (e.g. an `OBSERVE`-tagged segment is a stronger trigger for `infer_observation_access`), but a tag alone is never a skill, and the bank must not be browsable as a tag taxonomy. Any synthesis pipeline that produces "skills" that are merely tag classifiers is rejected by the synthesizer's verifiability check ([Skill Synthetics — Verifiability and Non-Leakiness Checks](skill_synthetics_agents.md#verifiability-and-non-leakiness-checks)).

---

## 7. Composition patterns

**Direct mode** (short video; raw frames in context): hops still use **atomic** reasoning steps internally; there may be no `search_memory` primitive, but atomics like `ground_event_span` may operate on the provided visual/text context.

**Retrieval mode** (long video; graph available):

```
[offline pipeline — infrastructure, not bank skills]
  for each clip:
      detect_entities -> observe_segment -> build_episodic
  build_semantic
  update_state

[online reasoning — bank skills over memory outputs]
  Question -> parse / policy
         -> Hop 1: atomic chain + verify
         -> Hop 2: ...
         -> decide_answer_or_abstain
```

Legacy **Think / Search / Answer** loops remain the **surface protocol** for the frozen reasoner; **internally**, the controller logs **atomic traces** per hop. `[Search]` invokes **`search_memory`** (primitive), not a reasoning skill.

### Prompt / task mapping (unchanged intent)

| Question type | Skill family emphasis | Memory emphasis |
|---------------|----------------------|-----------------|
| Social Relationship (SR) | Social / belief atomics | Entities, perspective threads |
| Temporal Causal (TCI) | Temporal + causal atomics | Episodic chain, edges |
| Hidden Reasoning (MHR) | Verification + counterevidence | Semantic + episodic |
| Temporal Arrangement (TA) | Temporal atomics | Timestamped nodes |
| Core Theme (CTI) | Parsing + retrieval | Semantic summaries |

---

## 8. Skill promotion, split, merge, retire

Aligned with [Skill Synthetics Agents](skill_synthetics_agents.md):

| Operation | When |
|-----------|------|
| **Patch** | Add or revise preconditions, `verification_rule`, or intermediate checks |
| **Split** | A composite shows divergent failure clusters; carve narrower variants |
| **Promote** | A frequently successful **atomic chain** stabilizes with a clear verification rule |
| **Merge** | Near-duplicate composites with similar `child_skills` and triggers |
| **Retire** | Consistently unreliable or unused skills |

**Promotion criteria (atomic chain → composite):** succeeds repeatedly across multiple examples; has a **stable** `verification_rule`; appears in more than one reasoning context or task family; remains interpretable as a reusable procedure.

---

## 9. Skill bank infrastructure

### 9.1 SkillBank class

```python
class SkillBank:
    skills: Dict[str, Skill]  # atomic + composite
    composition_edges: Dict[str, List[str]]  # parent composite -> child skills
    performance: Dict[str, SkillPerformance]
    co_occurrence: Dict[Tuple[str, str], int]

    def select(self, question_analysis: Dict) -> List[Skill]: ...
    def compose(self, skill_ids: List[str]) -> ComposedStrategy: ...
    def update_performance(self, skill_id: str, trace: ReasoningTrace) -> None: ...
    def craft_new_skill(self, failure_analysis: FailureAnalysis) -> Optional[Skill]: ...
    def maintain(self) -> MaintenanceReport: ...  # merge, split, retire, promote
```

### 9.2 Skill DAG

- **Composition edges** link composite strategies to child atomics (or nested composites).
- **Performance** and **co-occurrence** inform promotion and merge.

### 9.3 Skill retrieval

Skill retrieval uses RAG-style scoring (e.g. from `decision_agents/agent_helper.py`):

1. **Relevance** — embedding similarity between question and skill description  
2. **Applicability** — trigger match against question analysis  
3. **Pass rate** — historical success  

### 9.4 Bank storage

- `skill_bank.jsonl` — serialized `Skill` records compatible with `SkillBankMVP` (`skill_agents/skill_bank/bank.py`) where applicable.

---

## 10. Integration with existing components

| Component | Connection |
|-----------|------------|
| `skill_agents/stage3_mvp/schemas.py` | `Skill`, `Protocol`, `SkillEffectsContract` — map layered fields onto existing structs or extend |
| `skill_agents/skill_bank/bank.py` | `SkillBankMVP` / JSONL |
| `skill_agents/skill_evaluation/` | LLM judge dimensions for crafted skills |
| `decision_agents/agent_helper.py` | `select_skill_from_bank()` — RAG scoring |
| `rag/retrieval.py`, `rag/embedding/` | Memory and skill embeddings |
| `data_structure/experience.py` | Package traces as `Experience` / `Episode` |

---

## 11. Failure modes (example)

Atomics fail in **localized** ways; composites inherit failure from child steps. Examples:

| Family | Example failure modes |
|--------|------------------------|
| Grounding | Wrong entity; wrong span; missed evidence |
| Temporal | Order unsupported by timestamps / narrative |
| Causal | Competing causes; weak support |
| Social | Perspective confusion; belief update without access |
| Verification | Premature answer; ignored alternatives |

Full reflection taxonomy: [Skill Synthetics Agents — failure categories](skill_synthetics_agents.md).
