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
