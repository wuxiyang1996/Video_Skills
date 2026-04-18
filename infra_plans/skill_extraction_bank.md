# Skill Extraction & Skill Bank — Design Plan

> Goal: Define the **skill bank infrastructure** — how **reasoning** skills are represented, stored, composed, and evolved. Skills are **reusable inference operators** over memory outputs and perspective threads, not generic retrieval routines or offline pipeline steps.
>
> **Related plans:**
> - [Atomic skills & hop refactor — execution checklist](atomic_skills_hop_refactor_execution_plan.md) — layer split, constraints, minimal atomic set
> - [Agentic Memory](agentic_memory_design.md) — episodic / semantic / state stores + evidence layer
> - [Video Benchmarks & Grounding](video_benchmarks_grounding.md) — benchmarks, memory graph, adapters
> - [Actors / Reasoning Model](actors_reasoning_model.md) — 8B controller, hops, orchestration
> - [Skill Synthetics Agents](skill_synthetics_agents.md) — synthesis, evolution, reflection

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

Skills conform to a **layered** schema (COS-PLAY-compatible fields can map into `protocol_steps`, `tags`, and contracts). Important fields for **evolvability**:

```python
Skill(
    skill_id: str,
    level: Literal["atomic", "composite"],
    family: str,
    trigger_conditions: list[str],
    inputs: list[str],
    output_type: str,
    protocol_steps: list[str],
    child_skills: list[str] | None,   # composite only
    required_primitives: list[str],    # e.g. search_memory, not "skills"
    verification_rule: list[str],
    failure_modes: list[str],
    repair_strategies: list[str],
    reuse_stats: dict,
    examples: list[dict],
)
```

**Field roles:**

| Field | Role |
|-------|------|
| `level` | Atomic vs composite |
| `output_type` | Structured output expectation (claim, span, ordering, belief snapshot, abstain, …) |
| `child_skills` | Ordered atomic (or nested composite) IDs for expansion |
| `required_primitives` | Infrastructure / retrieval calls the skill assumes |
| `verification_rule` | How to check the step locally |
| `failure_modes` / `repair_strategies` | For reflection and bank maintenance |

Optional compatibility shim: map `protocol_steps` ↔ COS-PLAY `Protocol.steps`, and keep `contract` / `SkillEffectsContract` for existing code paths (`skill_agents/stage3_mvp/schemas.py`).

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
