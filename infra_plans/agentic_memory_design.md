# Agentic Memory — Design Plan

> Sub-folder: `Video_Skills/memory_manage/` (with retrieval / graph implementation)
>
> Goal: Define **what** the 8B controller persists and retrieves — a small, function-based memory model that stays manageable for an 8B policy and avoids redundant top-level stores.
>
> **Related plans:**
>
> - [Actors / Reasoning Model](actors_reasoning_model.md) — controller, perspective threads, orchestration
> - [Skill Extraction / Bank](skill_extraction_bank.md) — reasoning skill bank (atomic / composite) over memory outputs
> - [Video Benchmarks & Social Grounding](video_benchmarks_grounding.md) — grounding pipeline, `SocialVideoGraph`, adapters
> - [Grounding Pipeline Execution Plan](grounding_pipeline_execution_plan.md) — perception + entity stack implementation backing the entity-centric index below
> - [MVP Build Order](mvp_build_order.md) — phased implementation plan

---

## 0. Design Principle: Stable Memory, Evolving Reasoning

This file defines the **stable substrate** half of that principle.

- Memory construction and maintenance are handled by **fixed procedures** and **fixed memory-management skills**.
- These fixed procedures guarantee stable storage, revision, compression, and evidence attachment.
- The evolving skill bank ([Skill Extraction / Bank](skill_extraction_bank.md)) is reserved for **reasoning skills**; memory procedures are **not** entries in that bank in v1.
- **Memory is the stable substrate; reasoning is the adaptive layer.**
- In phase 1 the memory policy is fixed; it is **not** auto-evolved by reflection or synthesis.

---

## 0.1 Fixed Memory Procedures in Phase 1

Phase 1 treats memory as **infrastructure**, not as a learned policy:

- Memory construction is performed by a fixed set of procedures, invoked by the harness on behalf of the 8B controller.
- Memory **write / update / compression / refresh** rules are not automatically evolved in v1; they are versioned manually between releases.
- The set of available memory-management skills is **closed** for v1: new procedures are introduced only by an explicit human-authored release, never by trace-driven synthesis.
- The retriever and verifier read these stable structures and assume their invariants hold.

This is what makes the controller learnable in v1: it learns to plan and route over a substrate whose semantics do not shift underneath it.

---

## 0.2 Memory-Management Skills vs Reasoning Skills

The system maintains a clean separation between two registries (see also [Skill Extraction / Bank §0](skill_extraction_bank.md#0-bank-scope-in-phase-1)). They are not collapsed into one undifferentiated "skills" store.

### Fixed memory-management procedures (this document)

These belong to the **Memory Procedure Registry** — a stable, versioned-by-hand catalog of infrastructure operations. They are the only writers / mutators of memory state. Examples:

| Procedure | Role |
|---|---|
| `open_episode_thread` | Create a new episodic thread for a clip / window |
| `append_grounded_event` | Append a `GroundedWindow`-derived event to episodic memory |
| `update_entity_profile` | Apply a detection / equivalence update to an `EntityProfile` |
| `refresh_state_memory` | Recompute social / spatial state for a query at time `t` |
| `compress_episode_cluster` | Produce a per-episode summary, retaining `source_ids` |
| `attach_evidence_ref` | Bind an `EvidenceRef` to an episodic / state record |
| `resolve_entity_alias` | Move an alias from `aliases_pending` to a bound `EntityProfile.aliases` |
| `revise_belief_state` | Apply a verifier-passed belief update with `supersedes` linkage |
| `mark_memory_conflict` | Record a `contradicts` edge between two memory records |

Properties of memory-management procedures:

- **stable** — same input → same effect across releases
- **versioned manually or by rule**, not by trace-driven promotion
- **infrastructure / procedure layer** — they encapsulate write / update / refresh policy
- **not** primary content of the evolving reasoning bank
- callable from atomic reasoning skills via the harness, but never confused with them

### Evolving reasoning skills (the bank)

Reasoning skills live in the **Reasoning Skill Bank** ([Skill Extraction / Bank](skill_extraction_bank.md)). Examples:

- `identify_question_target`
- `retrieve_relevant_episode`
- `order_two_events`
- `check_character_access` / `infer_observation_access`
- `infer_belief_update` / `update_belief_state`
- `locate_counterevidence`
- `check_evidence_sufficiency`
- `compare_hypotheses` / `check_alternative_hypothesis`
- `decide_answer_or_abstain` (abstain-if-unsupported)

Properties of reasoning skills:

- **trace-derived** (curated in phase 1; conservatively promoted in phase 2; synthesized in phase 3)
- **reusable reasoning operators** over memory + evidence outputs
- **bank-managed** and eligible for promotion / split / patch / retire under the synthesizer's gates ([Skill Synthetics](skill_synthetics_agents.md))

Boundary rule: a reasoning skill **never writes memory directly**; it requests a memory-management procedure via the harness, which then invokes the appropriate fixed procedure. This keeps the write path auditable and prevents bank churn from corrupting the substrate.

---

## Reasoning skills vs memory functions

**Reasoning skills** operate **over** memory outputs; they do **not** replace memory functions. Episodic, semantic, and state memory remain the **storage and retrieval substrates**. Atomic and composite skills **consume** retrieved memory entries, evidence attachments, and perspective-thread state to produce **verifiable intermediate reasoning outputs** ([Skill Extraction / Bank](skill_extraction_bank.md)). Keeping this boundary explicit avoids drifting toward duplicate “skill-shaped” memory modules or a fourth top-level store.

---

## Principle: separate by function, not by every content type

The design favors **evidence-grounded reasoning**, **compact interpretable memory**, and **few modules** so the controller does not juggle overlapping categories (e.g. episodic vs. visual vs. social-state as four competing stores).

**Rule of thumb:** three functional roles cover the needs:

| Function | Question it answers |
|----------|---------------------|
| **What happened?** | Episodic memory |
| **What stays true over time?** | Semantic memory |
| **What is true *now* for reasoning at query time?** | State memory |

Visual material is **not** a fourth top-level memory store.

---

## Recommendation: three core memory types

### 1. Episodic memory

Stores **grounded events over time** — the main memory backbone.

Includes:

- clips / timestamps  
- involved characters (entity IDs)  
- dialogue / subtitle spans  
- actions and events  
- **evidence links** (pointers into the evidence layer; see below)

### 2. Semantic memory

Stores **slowly changing abstractions** distilled from episodic memory — compressed long-horizon memory.

Includes:

- character attributes  
- stable relationship summaries  
- recurring behavior patterns  
- long-term factual summaries  

### 3. State memory

Stores the **current inferred world state** needed for reasoning at query time — a **single** memory type with two subfields (not two separate top-level memories).

**Social-state subfield**

- who knows what  
- beliefs, trust, stance  
- concealment, probing, avoidance, deception risk  

**Spatial-state subfield**

- where people and objects are  
- visibility, layout  
- movement and scene layout changes  

Updates to state should be **evidence-linked** back to episodic entries or attached evidence.

---

## Evidence layer (not a top-level memory store)

**Do not** treat **visual memory** as its own first-class store alongside episodic / semantic / state.

**Why**

- Visual evidence already belongs on **episodic** (and sometimes **state**) entries as attachments: keyframes, regions, appearances, trajectories.  
- A separate “visual memory” duplicates indexing and retrieval and forces the controller to choose among episodic vs. visual vs. social vs. spatial — too heavy for an 8B policy.

**Instead:** **visual = evidence substrate / attachment**, plus modality-specific refs:

- **Visual evidence:** key frames, region crops, appearance embeddings, trajectory snippets  
- **Subtitle / dialogue evidence:** spans and alignment refs  
- **Audio evidence:** segment refs, speaker IDs when used  

Retrieval features (embeddings over crops, etc.) index **through** episodic and state records, not as a parallel memory product.

---

## Entity-centric indexing

Entity identity is the **primary index** across all three memory stores. Episodic entries are keyed by who was involved; semantic summaries roll up per character; state memory answers "what does *this character* believe / know / do right now." This subsection fixes the schema of that identity layer.

### Entity profile schema

Every persistent character in a video is represented by a single `EntityProfile` that aggregates face / voice / subtitle aliases and is the anchor for all memory reads and writes. The profile is materialized by [`SocialVideoGraph.refresh_equivalences`](video_benchmarks_grounding.md#27-entity-resolution--re-identification) after grounding completes.

```python
@dataclass
class EntityProfile:
    character_id: str                 # "character_12" (video-global)
    face_nodes: list[NodeId]          # <face_N> nodes unified into this character
    voice_nodes: list[NodeId]         # <voice_N> nodes unified into this character
    aliases: list[str]                # subtitle names ("Alice"), script labels, role tags
    role: str | None                  # e.g. "manager", "suspect", "robot_operator"
    appearance: dict                  # stable visual attrs: hair, clothing, glasses, ...
    voice_traits: dict                # stable audio attrs: pitch band, accent, language
    first_seen: tuple[str, float]     # (clip_id, time)
    last_seen: tuple[str, float]
    confidence: float                 # identity-resolution confidence
    active_clips: list[str]           # clips where the character appears
    aliases_pending: list[str]        # unresolved subtitle names awaiting equivalence
```

`EntityProfile` is not a fourth top-level store — it is an **index view** built from `img` + `voice` + `semantic` nodes in `SocialVideoGraph`. Updates propagate back to the three stores as described below.

### Face / voice IDs and identity persistence

| Identity layer | Stored on | When assigned | Persistence scope |
|---|---|---|---|
| `<face_N>` | `img` node | First time a face cluster matches or is created (Stage B of the resolver) | Video-global once the node exists |
| `<voice_N>` | `voice` node | First time a speaker-diarized segment matches or is created | Video-global once the node exists |
| `character_N` | `character_mappings` | At `refresh_equivalences()` (end of video, or incrementally after a new equivalence assertion) | Video-global; stable across reloads of the graph `.pkl` |
| Subtitle alias (`"Alice"`) | `aliases` on `EntityProfile` + `aliases_pending` until bound | On first subtitle/ASR mention | Bound permanently when an equivalence assertion ties the alias to a `<voice_N>` or `<face_N>` |

Identity persistence rules:

1. **Append-only within a video.** `<face_N>` and `<voice_N>` never get renamed or recycled; downstream memory entries stay pointing at stable ids.
2. **Merges via equivalence, not rewrite.** Two `img` nodes found to be the same person are **not** collapsed into one; instead the union-find in `refresh_equivalences` places them under the same `character_id`. Existing episodic links stay intact.
3. **Cross-episode persistence** (across different videos in a multi-video corpus) is **out of scope for v1**. Recorded as future work in the execution plan.
4. **Confidence-weighted retrieval.** When two candidate profiles have overlapping faces/voices but conflicting aliases, retrieval returns both with their identity-resolution confidences; the reasoning core must decide.

### Local vs global knowledge state

State memory is partitioned **per character** to preserve perspective:

- **Global state**: facts that all observers in the video share (scene layout, objects present).
- **Local state (per character)**: what *this* character knows, believes, or intends — keyed by `character_id`. Two characters in the same scene may hold contradictory social-state subfields (`trust`, `suspicion`, `belief`).

When a new observation arrives, the writer decides which `character_id`'s local state to update based on:

- who was looking at / listening to the evidence (gaze / speaking flags in `EntityAttributes`),
- whether the evidence is marked `inferred_from_behavior` vs `directly_observed` (see provenance enum in [§2.6 wire format](video_benchmarks_grounding.md#26-grounded-window-wire-format-normative)).

This is what lets benchmarks like SIV-Bench and MA-EgoQA evaluate ToM questions without the pipeline collapsing everyone's beliefs into one worldview (error class E7 in [grounding error taxonomy](video_benchmarks_grounding.md#11-grounding-error-taxonomy)).

### Episodic / semantic / state writes pivoting on entities

All three stores index by entity:

- **Episodic** — each episodic record lists participating `character_id`s (plus `<face_N>`/`<voice_N>` raw references for traceability); `graph.get_timeline(character_id)` returns them in order.
- **Semantic** — Level-3 and Level-4 distillations (see [grounding pipeline §4](video_benchmarks_grounding.md#4-hierarchical-memory-for-long-videos)) are written **per character** or **per character-pair**, not globally.
- **State** — keyed by `character_id` as described above; retrieval for ToM questions walks `EntityProfile.character_id → state entries`.

Retrieval APIs on the graph that honor this indexing:

- `SocialVideoGraph.get_timeline(character_id)` — episodic nodes involving a character.
- `SocialVideoGraph.get_relations(character_id)` — interactions / events linked to a character.
- `SocialVideoGraph.get_entity_info(character_id)` — dedup-filtered semantic context attached to the character (vendored from m3-agent).
- `SocialVideoGraph.back_translate(question_text)` / `translate(answer_text)` — name ↔ identity round-trip for benchmarks like M3-Bench.

### Alias resolution policy

1. A subtitle or ASR name that is not yet bound to any entity is recorded in the current window as a `aliases_pending` entry on the **most likely speaker**, picked by: (a) voice node active at the subtitle's time span, else (b) the face node currently marked `speaking=True`, else (c) the most-recent speaker.
2. When the grounding pipeline later emits an equivalence assertion that resolves the name (e.g. via Stage-1 LLM output: `"equivalence: Alice = <voice_5>"`), the pending alias is moved onto the entity profile's `aliases` list and removed from `aliases_pending`.
3. An alias must not be bound to more than one character; if conflicting assertions arrive, the resolver defers to the assertion with the highest edge weight (`fix_collisions(mode="eq_only")`), same mechanism used for face↔voice equivalence.
4. Aliases left in `aliases_pending` at end-of-video are surfaced as warnings but do **not** block retrieval (the character remains addressable by `character_id`).

### Confidence, decay, and stale identities

- Each entity node carries a confidence score rolled up from its detections (face detection/quality scores, voice segment durations). Retrieval thresholding uses this score.
- An entity not seen for more than `N` clips has its state-memory entries confidence-discounted per the revision policy (see checklist item "Confidence decay"). The profile itself is not deleted.
- Compression / eviction (checklist item) preserves `EntityProfile` and high-edge-weight episodic entries; low-edge-weight semantic entries are first to be pruned via `prune_memory_by_node_type` (vendored).

---

## Memory lifecycle policies

The three stores plus the evidence layer are not write-once buffers. They are governed by explicit **write**, **revision**, **compression**, and **refresh** policies so that an 8B controller can rely on stable invariants instead of ad hoc heuristics. All policies operate over the canonical objects defined in [Actors §2A](actors_reasoning_model.md#2a-canonical-runtime-data-contracts) — `GroundedWindow`, `EvidenceBundle`, `AtomicStepResult`.

> **Phase-1 stance.** The triggers, thresholds, decay constants, and refresh modes below are **fixed** for v1. They are tuned manually between releases. They are **not** auto-evolved by the synthesizer or learned by the controller. The lifecycle implementation table at the end of this section is the **normative contract** that the fixed memory-management procedures (§0.2) must implement.

### Memory Write Policy

Defines when an observation becomes a persistent record and what evidence must be attached.

| Trigger | Store | Becomes a record when | Evidence required |
|---|---|---|---|
| New `GroundedWindow` from perception | **Episodic** | Window passes local verification: `confidence ≥ τ_grounding` (default 0.5) and contains at least one `EventRef` or `DialogueSpan` | `keyframes` + `provenance` from window; `EvidenceRef` per derived event |
| Episodic cluster of ≥ N similar events involving same entities | **Semantic** | Cluster passes the *Semantic Refresh Policy* below | Pointers (not copies) to all source episodic node ids |
| New `AtomicStepResult` with `output_type ∈ {belief, intention, trust, suspicion, presence}` | **State** (social or spatial subfield, per `output_type`) | Step's `VerificationResult.passed=True`; updates only the local state of the perspective anchor | The originating `EvidenceBundle.refs` and the `AtomicStepResult.step_id` |
| Equivalence assertion (alias ↔ identity) | **EntityProfile** index view | Resolver edge weight exceeds `fix_collisions` threshold | Source assertion id + the unifying detection ids |

Write timing rules:

1. **Local verification before persistence.** No write happens straight from a `GroundedWindow` if the window's `confidence` is below `τ_grounding`; the window goes to a staging buffer and is dropped if not confirmed within the next clip.
2. **Atomic-step writes are deferred.** A reasoning step writes to state memory only after the verifier returns `passed=True` for that step; failed or retried steps never write.
3. **Inferred = inferred.** Records derived from inference (rather than direct observation) carry `provenance="inferred"` and a non-empty `alternative_hypotheses` list when applicable.
4. **No silent overwrite.** Write of a record whose `(entity_ids, time_span, type)` collides with an existing record triggers the *Memory Revision Policy* instead of a blind overwrite.

### Memory Revision Policy

Defines how the system updates beliefs, identities, and stale state when new evidence arrives.

| Situation | Rule |
|---|---|
| **Contradiction** between new and existing record | Keep both; mark them with reciprocal `contradicts` edges; downstream reads return both with confidences and let the verifier resolve |
| **Confidence decay** | State entries' confidence multiplied by `decay(Δt)` per clip without supporting re-observation; `decay(Δt) = exp(-λ * Δt)` with `λ = 0.05` per clip (default) |
| **Stale state** | A state entry whose decayed confidence falls below `τ_state_active` (default 0.3) is marked `is_active=False`; it remains queryable for history but is excluded from default state retrieval |
| **Belief revision** | A new `update_belief_state` result with higher confidence supersedes the prior entry; superseded entries set `is_active=False` and link via `supersedes` |
| **Identity revision** | If the resolver produces a new equivalence that conflicts with prior bindings, apply `fix_collisions(mode="eq_only")`; never rename existing `<face_N>` / `<voice_N>`, only re-bind aliases on the `EntityProfile` |
| **Speaker attribution revision** | When ASR / diarization changes the most-likely speaker for a dialogue span, move the span's reference; preserve original attribution under `meta.prior_attribution` for audit |

Revision is always evidence-linked: the operation must cite the `EvidenceBundle` (or assertion id) that motivated it.

### Entity-Centric Indexing (lifecycle view)

The `EntityProfile` schema in the section above defines the static shape. The lifecycle additions below specify how the index evolves:

- **Aliases** are append-only on the profile; pending aliases live on `aliases_pending` until bound (per *Alias resolution policy*) and never silently disappear.
- **Face / voice IDs** persist video-globally once allocated; merges happen only via union-find under a `character_id` (no rename).
- **Cross-episode identity persistence** (within the same video) is provided by the union-find; cross-video persistence remains v2 work.
- **Local vs global state** is partitioned by `character_id` for the social subfield and shared for the spatial subfield, exactly as described above; the lifecycle rules guarantee that local state writes never leak into another character's slice.

### Compression and Eviction

Long videos generate more episodic records than the 8B controller can usefully retrieve. Compression and eviction shape the index without losing auditability.

| Material | Policy |
|---|---|
| Recent episodic windows (last K min) | **Full episodic resolution** — kept verbatim with all evidence |
| Mid-age episodic windows | **Compressed** into per-episode summaries; raw events still pointed to via `source_ids` |
| High-edge-weight episodic nodes (high entity centrality, witnessed by many) | Retained at full resolution regardless of age |
| Low-edge-weight episodic nodes older than the eviction horizon | **Archived**: moved out of the active index but kept on disk for replay; not returned by default retrieval |
| Stale state entries (`is_active=False` and decayed confidence < `τ_evict`) | **Discarded** unless they were ever cited by a final answer; cited ones are archived |
| Raw frame crops not referenced by any retained record | **Discarded** after the eviction horizon |

Invariants:

- **Compressed summaries retain pointers to raw evidence.** A compressed episode's `source_ids` field always lists its source event ids; if the underlying events have been archived, the pointer resolves to the archive shard.
- **Eviction is reversible.** Archived material can be re-loaded on demand by the retriever's broaden ladder ([Actors §2B.3](actors_reasoning_model.md#2b3-broaden-ladder)).
- **EntityProfiles are never evicted.** Even for entities not seen recently, the profile stays in the index (with confidence-discounted state).

### Semantic Refresh Policy

Semantic memory is the place that most easily drifts; this policy makes refresh deterministic.

| Trigger | Action |
|---|---|
| Every K episodic writes for a given entity (default `K=5`) | Recompute that entity's semantic summary |
| New episode whose contracts (`eff_add` / `eff_del`) differ from the existing summary's predicates above threshold τ_drift | Mark the existing summary stale and regenerate |
| Verifier reports `temporal_consistency` or `entity_consistency` failure citing a semantic summary | Mark stale and regenerate from current episodic nodes |
| Confidence-decayed below `τ_semantic_active` | Mark stale; re-derive on next read |

Refresh modes:

- **Versioned update (default).** Each refresh writes a new `version` of the summary; previous versions are kept (read-only) and pointed to from the latest one. This is what enables rollback ([Skill Synthetics — Bank Versioning and Rollback](skill_synthetics_agents.md#bank-versioning-and-rollback)).
- **Overwrite.** Allowed only for typo-level corrections (no semantic change); requires `overwrite=True` flag and is logged.

A "stale summary" is detected by any of: (a) source episodic nodes archived/removed, (b) drift trigger above, (c) explicit verifier flag. Stale summaries do not block retrieval; they are returned with a `meta.stale=True` marker so the verifier can downweight them.

When new evidence reopens an old summary, the system creates a new version that links to both the old version and the new evidence; downstream reasoners must consume the latest version unless they explicitly request history.

### Lifecycle implementation table

| Memory type | Write trigger | Update trigger | Evidence requirement | Revision rule | Compression policy |
|---|---|---|---|---|---|
| **Episodic** | `GroundedWindow` confidence ≥ τ_grounding with ≥1 event | New event with same `(entities, time_span)` collides → contradicts edge | `keyframes` + per-event `EvidenceRef` | Append + `contradicts` edges, never overwrite | Recent: verbatim. Older: per-episode summary with `source_ids` |
| **Semantic** | ≥N clustered episodic items per entity / pair | Drift τ exceeded; verifier flag; periodic K-write counter | Pointers to source episodic ids | Versioned regenerate; previous versions retained | Always compressed; old versions archived |
| **State (social)** | `update_belief_state` / `infer_observation_access` step verified | New verified step with higher confidence; decay over time | Originating `EvidenceBundle` + `step_id` | Supersede via `supersedes` edge; decayed → `is_active=False` | Inactive + decayed → discarded unless cited |
| **State (spatial)** | `GroundedWindow.spatial_state` change vs prior | New spatial observation overrides at later timestamp | `keyframes` from window | Time-keyed overwrite (kept history) | Down-sample positions older than horizon |
| **Evidence layer** | Created with each episodic / state write that references it | Re-attached when revision adds new evidence | The detector / model output ids | Never modified in place; replaced by new ref | Frame crops without references → discarded |
| **EntityProfile** | First detection of face / voice / alias | Equivalence assertion; new detection appended | Source detection + assertion ids | Union-find merge, no rename; aliases append-only | Never evicted |

This table is the **normative implementation contract** for `memory_manage/`. Any module that writes memory must declare which row it implements.

---

## Mapping from a “five-way” split (avoid)

| Earlier notion | Canonical placement |
|----------------|---------------------|
| Episodic | **Episodic memory** (unchanged) |
| Semantic | **Semantic memory** (unchanged) |
| Social-state + Spatial-state (separate stores) | **State memory** — one store, `social` + `spatial` subfields |
| Visual (top-level) | **Demoted:** evidence attachments + retrieval features on episodic / state entries |

---

## Minimal spec (copy-paste summary)

**Agentic memory (three stores + evidence):**

1. **Episodic memory** — Event-centric grounded records over time: clips, timestamps, characters, dialogue, actions/events, and linked visual/audio/subtitle evidence.

2. **Semantic memory** — Long-term distilled knowledge: attributes, relationship summaries, recurring patterns, stable facts.

3. **State memory** — Query-time world state for reasoning:
   - **Social:** beliefs, knowledge boundaries, trust/stance, concealment/probing/avoidance, etc.  
   - **Spatial:** locations, visibility, placement, movement, layout changes.

**Note:** Visual information is **not** maintained as a separate top-level memory store. Keyframes, regions, appearances, layouts, and trajectories are **attached as evidence** to episodic/state entries and used for retrieval.

---

## Relation to `SocialVideoGraph` and benchmarks

- The **graph** (e.g. `SocialVideoGraph`) is the **index / schema** over grounded content: entities, interactions, events, and links. It is **not** the same as “five memory RAM banks.”  
- **Episodic** rows (or nodes) carry the timeline backbone; **semantic** nodes compress spans of episodic material; **state** holds the rolling social+spatial snapshot for the current question.  
- Short-video **direct** mode may keep episodic + state **in context** without building a large persistent index; long-video **retrieval** mode persists all three stores plus evidence-backed retrieval features.
