# Plan docs: implementation checklist (missing parts)

This is an implementation checklist organized by file. Gaps are phrased as concrete edits rather than abstract criticism. Focus: what to add directly to the plan documents.

> **Status update:** The grounding-related items (§2 entity schema, §3 entire checklist) are addressed by the new [`grounding_pipeline_execution_plan.md`](grounding_pipeline_execution_plan.md) plus additions to [`video_benchmarks_grounding.md`](video_benchmarks_grounding.md) §2.6 / §2.7 / §6.1 / §11 and the "Entity-centric indexing" section in [`agentic_memory_design.md`](agentic_memory_design.md). Checkboxes below are ticked accordingly; remaining unchecked items are still open.

---

## 1) `infra_plans/actors_reasoning_model.md`

**Role today:** Strong on the 8B controller, two-phase operation, hop execution, and direct vs retrieval modes.

**Gap:** It does not yet define one canonical end-to-end schema for what flows between modules.

### Add: section — **Canonical Runtime Data Contracts**

Define typed objects for at least:

- `GroundedWindow`
- `EvidenceBundle`
- `HopGoal`
- `AtomicStepResult`
- `VerificationResult`
- `AbstainDecision`
- `ReasoningTrace`

The file already says the controller should log hop goals, atomic traces, intermediate claims, evidence pointers, and confidence — this section formalizes those objects.

### Add: section — **Retriever and Verifier as first-class subsystems**

Move beyond `graph.search(query)` and conceptual verification. Specify explicit policies for:

- query rewriting
- entity-conditioned retrieval
- time-conditioned retrieval
- counterevidence retrieval
- top-k fusion
- contradiction handling
- evidence sufficiency thresholds

The loop already distinguishes `search_memory` as a primitive and says each atomic has a `verification_rule` — specify what those rules actually check.

### Add: section — **Training signals for the controller**

The file says the controller is trainable and outputs are exposed for logging, GRPO, and evolution, but not how supervision is generated.

Add a **table** for rewards or supervision targets, e.g.:

- decomposition quality
- retrieval recall
- evidence precision
- perspective correctness
- abstention correctness
- final answer correctness

Add a short **anti-hacking** note: the controller must not win by over-retrieving or abstaining too often.

### Checklist (what to add)

- [ ] Canonical object schemas
- [ ] Retrieval policy
- [ ] Verification rubric
- [ ] Abstention policy
- [ ] Training/reward table
- [ ] Online serving loop with retry/fallback behavior

---

## 2) `infra_plans/agentic_memory_design.md`

**Role today:** Right top-level split (episodic, semantic, state) and “skills operate over memory.”

**Gap:** Too thin for lifecycle semantics — it stops before defining when and how memory changes.

### Add: **Memory write policy**

- When an observation becomes an episodic entry
- When episodic clusters become semantic summaries
- When state is updated
- How conflicts are resolved on write

### Add: **Memory revision policy**

- Contradiction handling
- Confidence decay
- Stale state
- Semantic-summary refresh
- Long-video social cases: “old belief later disproven,” “identity uncertain,” “speaker attribution revised”

### Add: **Entity-centric indexing** (subsection)

- Character profiles, aliases
- Face/voice IDs
- Cross-episode identity persistence
- Local vs global knowledge state  

(Rationale: state memory already covers who knows what, beliefs, trust, stance, spatial facts.)

### Add: **Compression and eviction**

- What stays episodic at full granularity
- What gets compressed
- What gets discarded or archived  

(Rationale: 8B controller needs compact, interpretable memory.)

### Checklist (what to add)

- [ ] Write/update triggers
- [ ] Contradiction and revision rules
- [ ] Confidence fields and decay
- [x] Entity profile schema — `agentic_memory_design.md` → "Entity-centric indexing" section; backing implementation in `grounding_pipeline_execution_plan.md` Phase 2
- [ ] Compression/eviction policy
- [ ] Semantic refresh policy

---

## 3) `infra_plans/video_benchmarks_grounding.md`

**Role today:** Shared grounding schema idea, tiers, grounding on vs persistence conditional.

**Gap:** Reads more like a survey than an implementation plan; missing wire format and entity policy.

### Add: section — **Grounded output schema**

Exact fields per window or clip, e.g.:

- entities, actions, dialogue spans
- object states, interactions
- inferred social cues
- timestamps, uncertainty
- evidence pointers  

Then: **mapping from grounded outputs → episodic memory writes.**

### Add: **Entity resolution / re-identification**

Policy for:

- re-identification
- alias mapping
- occlusion handling
- confidence-based identity repair  

(M3-Bench and similar already imply face/voice/person tracking — this is a high-risk gap if unspecified.)

### Add: **Benchmark-to-capability mapping table**

Per benchmark, which submodules are supervised, stressed, or evaluation-only, e.g.:

- local grounding, temporal ordering, retrieval
- evidence attribution, perspective tracking, belief modeling, entity resolution

### Checklist (what to add)

- [x] Grounded window schema — `video_benchmarks_grounding.md` §2.6 (normative wire format with typed dataclasses + invariants)
- [x] Grounding confidence and uncertainty fields — `video_benchmarks_grounding.md` §2.6 (`confidence`, `provenance`, `supporting_evidence`, `contradicting_evidence`, `identity_status`, `low_confidence_reason` metadata)
- [x] Entity resolution/re-identification policy — `video_benchmarks_grounding.md` §2.7 (three-stage resolver) + `grounding_pipeline_execution_plan.md` Phase 2
- [x] Benchmark-to-capability mapping — `video_benchmarks_grounding.md` §6.1 (S / G / E / — matrix across 17 capabilities × 6 benchmarks)
- [x] Adapter definitions per benchmark — `grounding_pipeline_execution_plan.md` Phase 5 (six adapters behind `BaseAdapter`) ; design in `video_benchmarks_grounding.md` §5
- [x] Grounding error taxonomy — `video_benchmarks_grounding.md` §11 (E1–E8 with detection signal + repair action)

---

## 4) `infra_plans/skill_extraction_bank.md`

**Role today:** Solid stance — skills as reasoning operators; atomic vs composite.

**Gap:** No full “bank specification” or canonical starter set; boundary with synthesis doc unclear.

### Add: formal **SkillRecord** schema

Fields to pin down, e.g.:

- `skill_id`, `name`, `type` (atomic | composite)
- `trigger_conditions`
- `input_schema`, `output_schema`
- `verification_rule`, `failure_modes`
- `required_memory_fields`, `retrieval_hints`
- `usage_stats`
- parent/child links

### Add: **minimal atomic skill inventory** (canonical starter set)

Group explicitly, e.g.:

- entity grounding
- temporal linking
- causal linking
- belief update
- perspective check
- contradiction check
- evidence sufficiency
- alternative hypothesis check
- answer/abstain decision

### Add: **composite formation rules** and formats

- trigger-condition format
- verification-rule format

### Add: **one explicit paragraph** — reasoning skills vs scene/action tags

The bank’s primary content is **reasoning skills**. Scene/action/intention patterns (e.g. NAVIGATE, MANIPULATE) are **auxiliary metadata or triggers only**, not the main skill definition — aligns with `skill_synthetics_agents.md` cleanup (see §5).

### Checklist (what to add)

- [ ] Formal SkillRecord schema
- [ ] Canonical starter atomic skill set
- [ ] Composite skill formation rules
- [ ] Trigger-condition format
- [ ] Verification-rule format
- [ ] Clear boundary between reasoning skills and scene/action tags

---

## 5) `infra_plans/skill_synthetics_agents.md`

**Role today:** Quality control, failure taxonomy, failure-to-update mapping, evolution loop — keep.

**Gap:** Front half still leans on segment intention tags (OBSERVE, INTERACT, NAVIGATE, …) as if they were the skill ontology — closer to video-behavior taxonomy than reasoning-skill bank.

### Revise narrative: **reasoning-skill synthesis from reasoning traces**

- **Primary synthesis unit:** successful hop traces over memory (atomic reasoning chains).
- **Segments:** produce grounded evidence and latent situation patterns; keep as **support signal**, not the main synthesis unit.
- **Bank:** synthesized primarily from **repeated successful reasoning chains**.

Keep segment tags only as **side metadata** where useful.

### Add to quality control

- **Verifiability** — can each skill output be checked against explicit evidence?
- **Non-leakiness** — does the skill avoid benchmark-specific answer templates or spurious shortcuts?

### Add: **trace-localization procedure**

On failure, distinguish:

- wrong atomic step vs missing retrieval vs final answer unsupported despite correct intermediates

### Add

- promotion thresholds: atomic → composite
- bank versioning and rollback rules

### Checklist (what to change / add)

- [ ] Replace “skill synthesis from intention-tagged segments” with “skill synthesis from successful reasoning traces”
- [ ] Keep segment tags only as side metadata
- [ ] Verifiability and shortcut checks
- [ ] Trace-localization procedure
- [ ] Promotion thresholds for atomic → composite
- [ ] Bank versioning and rollback rules

---

## 6) `infra_plans/atomic_skills_hop_refactor_execution_plan.md`

**Status:** This file **exists** in `infra_plans/`; several other docs reference it. Treat it as the **operational bridge** between controller and skill bank — **verify it is complete** and expand if it is thin or outdated.

**Suggested contents** (create or fill gaps):

- definition of a hop
- allowed hop length
- atomic-step input/output contract
- local verification rule format
- composite expansion rules
- trace logging format
- failure localization protocol
- reflection update hooks

This should reduce ambiguity currently spread across other docs.

---

## 7) New file: `infra_plans/evaluation_ablation_plan.md`

**Purpose:** Benchmark and synthesis docs name tasks and failures; this doc **proves each subsystem matters** (paper-ready ablations).

### Include

**Ablations** (examples):

- no memory
- no state memory
- no entity resolver
- no verifier
- no abstention
- no skill bank
- atomic-only
- composite-only

**Metrics per subsystem** (examples):

- retrieval recall, evidence precision
- entity resolution accuracy
- perspective accuracy
- abstention F1
- final QA accuracy

**Design:**

- error buckets aligned to failure taxonomy
- benchmark-to-metric table
- cost/latency reporting

---

## Highest-priority edit order

Suggested fastest path (updated — grounding layer done, reasoning + skills layer still pending):

1. ~~Verify / expand `atomic_skills_hop_refactor_execution_plan.md`~~ *(exists; already referenced by the grounding execution plan for the infrastructure-primitive contract)*
2. Expand `agentic_memory_design.md` — entity-centric indexing added; **still open:** write/update triggers, contradiction rules, confidence decay, compression/eviction, semantic refresh policy
3. Add canonical runtime schemas to `actors_reasoning_model.md`
4. Rewrite the front half of `skill_synthetics_agents.md` around reasoning traces
5. ~~Add entity-resolution and benchmark-capability mapping to `video_benchmarks_grounding.md`~~ *(done: §2.7 + §6.1 + §11 wire-format / error taxonomy)*
6. Add formal SkillRecord schema to `skill_extraction_bank.md`
7. Add `evaluation_ablation_plan.md`
8. Execute `grounding_pipeline_execution_plan.md` Phase 0 → Phase 6 (replaces `out/claude_grounding/` with `out/grounding_v1/`)

---

## Blunt overall verdict

The repo is not missing a new big idea. It is missing:

- a shared schema
- a precise memory lifecycle
- a retrieval/verification design
- a clean reasoning-skill definition
- an evaluation plan

Those are fixable by **tightening the plans** rather than changing direction.

---

## Optional next step

Turn this into a **copy-paste TODO list** in Cursor with one block per file and **exact section titles** to add — this file is the source list; section titles above can be copied verbatim into each target doc.

For the grounding layer specifically, see [`grounding_pipeline_execution_plan.md`](grounding_pipeline_execution_plan.md) — it is already the Cursor-ready checklist for §3 of this file and for the entity-indexing addition to §2.
