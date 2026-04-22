# Skill Synthetics Agents — Design Plan

> Goal: Define how skills are **synthesized, crafted, evaluated, and evolved**
> over time. This covers the skill crafter pipeline, quality control (8B as
> judge), cross-video accumulation, failure-driven self-evolution, and the
> continuous skill maintenance loop.
>
> **Evolution unit:** updates apply to **atomic skill chains**, **reasoning hops**, and **composite skills** — not only coarse monolithic skills. See [Skill Extraction / Bank](skill_extraction_bank.md) for atomic families and hop composition.
>
> **Phase scope.** The synthesis pipeline described here is the **target design**. Phase 1 only exercises trace collection and (optionally) human-supervised promotion — see §0.1–0.3 below.
>
> **Related plans:**
> - [Atomic skills & hop refactor — execution checklist](../04_harness/atomic_skills_hop_refactor_execution_plan.md)
> - [Agentic Memory](../02_memory/agentic_memory_design.md) — three memory stores + evidence layer
> - [Video Benchmarks & Grounding](../01_grounding/video_benchmarks_grounding.md) — benchmarks, memory graph, adapters
> - [Actors / Reasoning Model](../03_controller/actors_reasoning_model.md) — 8B controller, reasoning core, orchestrator
> - [Skill Extraction / Bank](skill_extraction_bank.md) — atomic/composite skills and bank infrastructure
> - [MVP Build Order](../00_overview/mvp_build_order.md) — phased implementation plan

---

## 0. Design Principle: Stable Memory, Evolving Reasoning

This file describes the **adaptive layer** of the system. It must be read with two cross-plan rules in mind:

- Memory construction and maintenance are **fixed** in v1 ([Agentic Memory §0](../02_memory/agentic_memory_design.md#0-design-principle-stable-memory-evolving-reasoning)). Synthesis never edits memory procedures.
- The evolving bank holds **reasoning skills only** ([Skill Extraction / Bank §0](skill_extraction_bank.md#0-design-principle-stable-memory-evolving-reasoning)). Synthesis writes only to the Reasoning Skill Bank, never to the Memory Procedure Registry.

---

## 0.1 Role of Skill Synthesis in the MVP

- **Full automatic skill synthesis is not the first milestone.**
- The first milestone is **robust reasoning trace collection and failure localization** by the harness ([Atomic Skills & Hop Plan — Harness](../04_harness/atomic_skills_hop_refactor_execution_plan.md#harness-runtime-specification)) plus a curated starter bank ([Skill Bank §0.2](skill_extraction_bank.md#02-phase-1-bank-policy)).
- **Synthesis is introduced only after** traces, retrieval, verifier, and hop execution are stable. The MVP success criterion ([MVP Build Order](../00_overview/mvp_build_order.md)) is defined entirely on a curated bank — it does not require this section to be implemented at v1 ship time.

The pipeline described later in this document (recurring-chain mining, verifiability gates, promotion thresholds, versioning, rollback) is the **target design** for phase 2 / phase 3. In phase 1 only the trace ingestion side is exercised; the actions side is restricted as below.

## 0.2 Phase-1 Conservative Synthesis Policy

In phase 1, synthesis is intentionally minimal:

- Allow only **limited / manual or semi-automatic** promotion of repeated successful reasoning traces. A human reviewer signs off on the first composites; the synthesizer surfaces candidates but does not auto-activate them.
- **Do not** enable aggressive `patch` / `split` / `retire` loops in the first implementation phase. These operations exist in the spec for design completeness but are gated off behind a feature flag in v1.
- **Do not** let noisy traces freely rewrite the bank. Promotion thresholds (§ Promotion Thresholds) are minimums; in v1 they are doubled (`N_repeat = 10`, `τ_success = 0.8`, `transfer ≥ 3 distinct task families`) before activation.
- All bank writes go through the versioning + shadow path (§ Bank Versioning and Rollback). v1 does not bypass this for any operation.

## 0.3 Preconditions for Later Self-Evolution

Automatic synthesis is unsafe without the following invariants. Phase 2 / phase 3 may not enable broader synthesis until these are demonstrably in place:

| Precondition | What it means concretely |
|---|---|
| **Stable runtime schemas** | The canonical objects in [Actors §2A](../03_controller/actors_reasoning_model.md#2a-canonical-runtime-data-contracts) are versioned and unchanged across the synthesis window |
| **Reliable retrieval** | The retriever ([Actors §2B](../03_controller/actors_reasoning_model.md#2b-retriever-as-a-first-class-subsystem)) meets target recall on the eval set; broaden ladder behavior is monotone and reproducible |
| **Reliable verifier** | The verifier ([Actors §2C](../03_controller/actors_reasoning_model.md#2c-verifier-as-a-first-class-subsystem)) check catalog is stable; per-check pass rates are calibrated and not gameable by the controller |
| **Clean step / hop traces** | `AtomicStepResult` and `HopRecord` are populated for every run; no silent enrichment; failure modes use the standard codes |
| **Trustworthy failure localization** | The harness's *Failure Localization Protocol* assigns exactly one bucket per failure with high inter-rater agreement on a held-out audit sample |

If any of these regress, synthesis activations are paused until the invariant is restored. This is the safety counterpart to "memory is the stable substrate, reasoning is the adaptive layer."

---

## 1. Primary Synthesis Unit: Successful Reasoning Traces

The skill bank is grown **trace-first**, not tag-first. The main object the synthesizer consumes is a **successful reasoning trajectory** — a `ReasoningTrace` (or `HopRecord` slice) whose final and per-hop verifications passed, captured by the harness ([Atomic Skills & Hop Plan — Harness](../04_harness/atomic_skills_hop_refactor_execution_plan.md#harness-runtime-specification)).

| Statement | Implication |
|---|---|
| Successful **hop traces** are the main synthesis unit. | The synthesizer indexes recurring atomic chains within `HopRecord.steps`, not segment intention tags. |
| Reusable skills come from **verified reasoning traces**. | Every promoted skill is traceable back to ≥ `N_repeat` `HopRecord`s with `passed=True`. |
| Grounded segments and intention tags are **support metadata**, not the ontology. | They influence retrieval filtering and trigger conditions, but they cannot be promoted into bank skills on their own. |

### Synthesis pipeline (trace-first)

```
ReasoningTrace stream  (from harness on_hop_success / on_promotion_candidate hooks)
  │
  ├─ 1. Trace ingestion
  │     • Filter to HopRecords with hop_verification.passed=True
  │     • Normalize each step's (skill_id, output_type) pair into a chain signature
  │
  ├─ 2. Recurring chain mining
  │     • Group hops by chain signature; require ≥ N_repeat occurrences across distinct
  │       traces (and ≥ 2 distinct task families — see Promotion Thresholds)
  │     • Per group, compute mean and variance of hop_verification.score
  │
  ├─ 3. Composite candidate construction
  │     • For each qualifying group, build a SkillRecord with
  │       type="composite", child_links = chain, version.created_by="promoted"
  │     • Inherit verification rules per Composite Skill Formation Rules
  │       (see skill_extraction_bank.md §6)
  │
  └─ 4. Verifiability + non-leakiness gate (§ below)
        • Reject candidates that fail verifiability or leakage checks
        • Surviving candidates enter the bank as version.status="shadow"
          and are activated after a shadow evaluation pass
```

The COS-PLAY-style temporal segmentation + intention tagging + contract extraction pipeline is **retained as auxiliary input**, but its role is now to enrich `trigger_conditions` (e.g. attach an `intention_tag` trigger) and to help the retriever index segments — not to define skills.

### Output

- `skill_bank.jsonl` — `SkillRecord` entries (see [Skill Bank §6](skill_extraction_bank.md#6-skill-schema)).
- Each record carries `skill_id`, `name`, `type`, `family`, `child_links`, `verification_rule`, `usage_stats`, `version`, plus the COS-PLAY-compatible `protocol_steps` view for legacy code paths.

---

## 2. Role of Segment Tags as Side Metadata

Segment intention tags from the COS-PLAY taxonomy (`OBSERVE | INTERACT | NAVIGATE | COMMUNICATE | MANIPULATE | INVESTIGATE | REACT | WAIT | APPROACH | RETREAT | DELIVER | RECEIVE`) are **demoted** from primary ontology to side metadata:

- They are stored on grounded segments / episodic records, not on skills directly.
- They may appear in a skill's `trigger_conditions` as `TriggerSpec(kind="predicate", value={"intention_tag": "OBSERVE"})`.
- They are used by the synthesizer's clustering step to seed candidate groupings, but a tag-only cluster is **never** sufficient for promotion.
- They are **not** indexed as searchable skill identities; the bank cannot be browsed as a tag taxonomy.

This keeps the bank centered on reasoning operators rather than on a perception ontology.

### Verifiability and Non-Leakiness Checks

Every synthesis candidate must pass the following gates before becoming a `shadow` bank entry:

| Check | Rule |
|---|---|
| **Verifiable output** | The candidate must have a non-empty `verification_rule` whose checks reference real keys in `output_schema`; tag-classifier "skills" with no claim output are rejected |
| **Evidence-grounded** | Every example trace cited by the candidate must have non-empty `EvidenceBundle.refs`; candidates whose only support is the answer label are rejected |
| **No benchmark-specific shortcut** | The candidate's `trigger_conditions` and `protocol_steps` must not pattern-match a single benchmark's question template (detected by an n-gram overlap > τ_template against held-out template fixtures) |
| **No hidden label leakage** | The candidate's `input_schema` must not include the gold answer or any field that is a 1:1 function of the gold answer; static check by the synthesizer |
| **No question-style overfit** | If the candidate's trigger conditions only fire on questions from one benchmark, it is rejected; it must fire on at least 2 distinct task families |

A rejection is a first-class event — logged with a reason and the offending field — so the synthesizer's decisions are auditable.

### Skill Quality Control (8B as Judge)

After verifiability and non-leakiness gates pass, the 8B judge scores the candidate on six dimensions (reusing `skill_agents/skill_evaluation/`). These dimensions remain useful, now applied **on top of** the trace-first pipeline rather than as the primary filter:

| Dimension | 8B Model Check |
|-----------|----------------|
| **Coherence** | "Does this skill's `protocol_steps` make logical sense for its `family` and child chain?" |
| **Discriminability** | "Is this skill distinct from existing skills in the bank (compared by `child_links` and `trigger_conditions`)?" |
| **Composability** | "Can this skill chain with other skills in a reasoning plan?" |
| **Generalization** | "Would this skill apply to videos beyond the source traces?" |
| **Utility** | "Would invoking this skill on a fresh trace help close a hop's `success_predicate`?" |
| **Granularity** | "Is this skill at the right level of abstraction (atomic vs composite of length 2-5)?" |

Below-threshold candidates are returned to the synthesizer for revision; persistent failures are dropped, not merged blindly.

### Trace Localization Procedure

When a hop or final answer fails, the synthesizer must localize the failure before proposing any bank update. The procedure mirrors the harness's *Failure Localization Protocol* ([Atomic Skills & Hop Plan](../04_harness/atomic_skills_hop_refactor_execution_plan.md#failure-localization-protocol)) and assigns exactly one bucket per failure:

| Bucket | Where in trace | Bank-side action |
|---|---|---|
| **Wrong atomic step** | Step's `verification_failed` with non-empty evidence | `patch` the offending atomic's `verification_rule` or `protocol_steps`; consider `split` if multiple variants emerge |
| **Missing retrieval** | `failure_mode="empty_evidence"` after broaden | No bank patch; route fix to retriever config; only patch the atomic if it failed to declare a needed `retrieval_hint` |
| **Unsupported final answer** | All hops `passed=True` but `verify_final` failed `claim_evidence_alignment` | Add a cross-hop verification atomic (e.g. `check_evidence_sufficiency` over the full trace); patch the answer-composer policy |
| **Perspective mismatch** | `perspective_consistency` check failed | Patch involved skills' `verification_rule` to require `check_perspective_anchor`; promote a perspective-bound composite if pattern recurs |
| **Bad evidence selection** | Cited evidence has low `confidence` or contradicting refs ignored | Patch the responsible retrieval-style atomic to consult `EvidenceBundle.contradictions`; log to retriever for fusion-rule review |

### Promotion Thresholds

A candidate atomic chain is promoted to a composite skill only when **all** of the following thresholds are met:

| Threshold | Default | Meaning |
|---|---|---|
| **Repetition** | `N_repeat = 5` | Distinct hop occurrences with the same chain signature and `passed=True` |
| **Success rate** | `τ_success = 0.7` | Fraction of occurrences with `hop_verification.passed=True` |
| **Score stability** | mean ≥ `τ_stable = 0.7`, variance < `σ_stable = 0.15` | Verification scores are high and consistent |
| **Transfer** | ≥ 2 distinct task families | Demonstrates cross-task reuse, not benchmark overfit |
| **Evidence verifiability** | 100% of cited example traces have non-empty `EvidenceBundle.refs` | No promotion based on label-only success |

Promotion produces a new `SkillRecord` with `version.created_by="promoted"` and `version.status="shadow"`; activation requires a successful shadow pass (success rate ≥ `τ_success` over `N_shadow = 30` invocations).

### Bank Versioning and Rollback

Bank evolution is **always versioned**; nothing in the bank is ever silently overwritten.

- Every bank operation (`patch`, `split`, `promote`, `merge`, `retire`) writes a **new version** of the affected `SkillRecord`(s) with `parent_version_id` pointing at the previous version.
- Previous **stable versions are retained** with `status="archived"`; archived versions are read-only and queryable for rollback and audit.
- A promoted composite that regresses (success rate < `τ_stable - 0.15` over ≥ 20 invocations, or 3 consecutive shadow failures after re-activation) is **rolled back**: the latest version is set to `status="retired"`, and the most recent `status="archived"` version of the same `skill_id` family is reactivated.
- Rollback is recorded as its own version (`created_by="rolled_back"`) so the history is fully reconstructible.
- Bank snapshots are taken every `K_snapshot = 100` operations; a snapshot is a manifest of `(skill_id, version_id)` pairs that defines the "state of the bank" for a given evaluation run.

This versioning is what makes the synthesis story safe: **successful reasoning trajectories → localized atomic patterns → promoted reusable skills**, with full ability to undo when a promotion proves wrong.

---

## 3. Cross-Video Skill Accumulation

When processing multiple videos, the skill crafter:

1. Loads the existing skill bank
2. Attempts to **merge** new segments into existing skills (by embedding
   similarity + contract overlap)
3. Creates **new** skills only when no existing skill covers the pattern
4. **Retires** skills that lose all supporting evidence

This mirrors COS-PLAY's `bank_maintenance` (split/merge/refine/retire)
but adapted for video reasoning skills rather than game strategies.

---

## 4. Evolution unit: atomic chains, hops, and composite skills

Bank maintenance and learning signals target **atomic traces** and **hops** (short compositions of atomics), not only whole-answer or whole-skill failures. A **composite skill** is a promoted macro; evolution must still recover the **underlying atomic chain** for diagnosis (see [Skill Extraction / Bank §8](skill_extraction_bank.md)).

### 4.1 Reflection unit: atomic-step failure localization

Failures are **not** attributed only at the whole-skill or whole-answer level. The system first localizes failure to:

1. **Observation / grounding failure** — wrong spans, wrong entities, missed evidence, visibility or access misread.
2. **Reasoning failure** — atomic skill misapplied or wrong output within a hop (temporal, causal, belief, perspective).
3. **Verification failure** — sufficiency, alternatives, or confidence not checked before answering.

For reasoning failures, the controller further localizes the failed **atomic skill(s)** within the hop trace before proposing a patch, split, or promotion.

### 4.2 Failure categories (by subsystem)

**Observation-side failures**

- Wrong entity grounding  
- Wrong event span grounding  
- Missed relevant evidence  
- Visibility / access misread  

**Reasoning-side failures**

- Temporal ordering error  
- Causal linkage error  
- Belief update error  
- Perspective confusion  
- Unsupported social inference  

**Verification-side failures**

- Evidence insufficiency not detected  
- Alternative hypothesis ignored  
- Overconfident answer  

These categories refine the finer-grained taxonomy in §5 and drive which bank fields to patch (`verification_rule`, `protocol_steps`, `repair_strategies`).

### 4.3 Update rules

| Rule | Action |
|------|--------|
| **Patch** | Add or revise a precondition, `verification_rule`, or intermediate check on an atomic or composite skill |
| **Split** | Split a broad composite into narrower variants when **failure clusters** differ by child step or trigger |
| **Promote** | Promote a frequently successful **atomic chain** into a composite skill (criteria below) |
| **Merge** | Merge near-duplicate composites with similar `child_skills` and triggers |
| **Retire** | Retire consistently unreliable or unused skills |

### 4.4 Promotion criteria (atomic chain → composite)

A candidate atomic chain is promoted to a composite skill only if it:

- Succeeds **repeatedly** across multiple examples  
- Has a **stable** `verification_rule` (see [Skill Extraction / Bank §6](skill_extraction_bank.md))  
- Appears in **more than one** reasoning context or task family  
- Remains **interpretable** as a reusable procedure (expandable to atomics for diagnosis)  

---

## 5. Failure Taxonomy

When the controller produces a wrong answer, the failure is classified to
drive targeted skill updates.

| Failure Type | Description | Example |
|-------------|-------------|---------|
| **Missed evidence** | Relevant memory node exists but was not retrieved | Question about a conversation; retrieval query missed the right episode |
| **Wrong temporal linkage** | Events connected incorrectly in the reasoning chain | Cause and effect reversed; wrong temporal ordering |
| **Wrong entity grounding** | Confused two characters or misidentified an entity | "The woman" resolved to wrong face_id |
| **Perspective confusion** | Answered based on global truth instead of character's local view | System knows Bob stole the key; incorrectly claims Alice also knows |
| **False-belief reasoning error** | Failed to model that a character holds an incorrect belief | Character was told a lie; system treated the lie as truth |
| **Overconfident inference** | Drew strong conclusion from weak/ambiguous evidence | Single facial expression interpreted as definitive proof of deception |
| **Insufficient evidence, forced answer** | Not enough evidence existed but system answered anyway | Memory graph lacked the relevant segment; system hallucinated |

---

## 6. Failure → Update Mapping

Each failure type triggers a targeted update. The controller identifies the
structural cause and patches the specific component.

| Failure Type | Update Target | Specific Action |
|-------------|---------------|-----------------|
| Missed evidence | **Retrieval strategy** | Add alternative query patterns to the skill; increase retrieval breadth |
| Wrong temporal linkage | **Memory schema** | Strengthen `precedes`/`causes` edges; add temporal verification step |
| Wrong entity grounding | **Entity resolver** | Add disambiguation step; refine entity matching thresholds |
| Perspective confusion | **Perspective thread** | Add explicit "check perspective" step to social skills; create `check_character_access` skill if none exists |
| False-belief reasoning error | **Skill refinement** | Refine `infer_belief_update` to handle lie propagation; add false-belief substep |
| Overconfident inference | **Confidence calibration** | Lower confidence thresholds; add "require N supporting events" constraint |
| Insufficient evidence | **Verifier rule** | Strengthen evidence sufficiency checker; add abstention option |

---

## 7. Skill Evolution Mechanisms

| Mechanism | Trigger | Action |
|-----------|---------|--------|
| **Reinforce** | Skill used in correct answer | Bump success rate, update average evidence quality |
| **Refine** | Skill used in wrong answer with identified fix | Modify execution steps; add/remove preconditions |
| **Split** | Skill has high variance (works for some subtypes, fails for others) | Create two specialized skills from the original |
| **Merge** | Two skills have >80% step overlap and similar performance | Combine into single skill with broader trigger conditions |
| **Craft new** | Failure type has no matching skill | 8B controller generates a new skill from failure analysis + successful counter-example |
| **Retire** | Skill has <20% success rate over 20+ invocations | Remove from active bank; archive for reference |

---

## 8. Evolution Loop

```
For each evaluation batch:
  1. Run questions through controller (with skill bank)
  2. Score against ground truth
  3. For correct answers:
     → Reinforce skills used
     → Extract new skill patterns from novel compositions
  4. For wrong answers:
     → Classify failure type (§5; subsystem: §4.2)
     → Apply targeted update (§6)
     → If no existing skill addresses the failure pattern → craft new skill
  5. Every K batches:
     → Run bank maintenance (merge, split, retire)
     → Log skill bank statistics for analysis
```

### Evolution metrics

| Metric | What It Measures |
|--------|-----------------|
| **Skill reuse rate** | Unique skills / total skill invocations |
| **Skill refinement rate** | Success rate before/after refinement |
| **Bank growth rate** | New skills per evaluation batch |
| **Retirement rate** | Skills retired per K batches |
| **Composition stability** | How often top compositions change across batches |

---

## 9. Integration with Existing Components

| Existing component | How it connects |
|---|---|
| `skill_agents/stage3_mvp/schemas.py` → `Skill`, `Protocol`, `SkillEffectsContract` | Crafted skills use these schemas |
| `skill_agents/skill_bank/bank.py` → `SkillBankMVP` | Bank stored as SkillBankMVP-compatible JSONL |
| `skill_agents/skill_evaluation/` → LLM judge | Quality control reuses evaluation dimensions |
| `dataset_examples/video_skill_pipeline_design.md` | Offline pipeline extends that design with memory-graph-driven discovery |
| `data_structure/experience.py` → `Experience`, `Episode` | Reasoning traces stored as Episodes for replay/analysis |

---

## 10. Implementation Notes

### Skill Crafter Module (`skill_crafter.py`)

Lives in `Video_Skills/small_model_orchestrator/skill_crafter.py`.

```python
class SkillCrafter:
    def __init__(self, vlm_fn, embedder, existing_bank_path=None): ...
    def craft_from_graph(self, graph: SocialVideoGraph) -> List[Skill]: ...
    def evaluate_skill(self, skill: Skill) -> SkillEvaluation: ...
    def merge_into_bank(self, new_skills: List[Skill], bank: SkillBank) -> SkillBank: ...
```

### Expected effort

| Phase | Task | Effort |
|-------|------|--------|
| Crafter pipeline | Temporal segmentation + contract extraction + protocol gen | 3-4 days |
| Quality control | 8B judge + evaluation dimensions | 1-2 days |
| Cross-video accumulation | Merge/dedup logic | 1-2 days |
| Evolution loop | Failure classification + targeted updates | 2-3 days |
| Bank maintenance | Split/merge/retire automation | 1-2 days |
| **Total** | | **~8-13 days** |
