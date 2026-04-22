# Video Benchmarks & Social Visual Grounding — Design Plan

> Sub-folder: `Video_Skills/memory_manage/`
>
> Goal: Define the benchmark landscape, unified social visual grounding infrastructure, and benchmark-specific evaluation adapters for the COS-PLAY (`Video_Skills`) framework.
>
> **Related plans:**
>
> - [Agentic Memory](agentic_memory_design.md) — three memory stores (episodic, semantic, state) + evidence layer
> - [Actors / Reasoning Model](actors_reasoning_model.md) — reasoning core, 8B controller, orchestrator
> - [Skill Extraction / Bank](skill_extraction_bank.md) — atomic/composite reasoning skills, hop composition, bank infrastructure
> - [Skill Synthetics Agents](skill_synthetics_agents.md) — skill crafting, evolution, quality control
> - [Grounding Pipeline Execution Plan](grounding_pipeline_execution_plan.md) — m3-agent-based implementation plan that realizes the schema and adapters defined here

---

## 0. Key Insight — Shared Reasoning, Shared Grounding, Different Context Regimes

After auditing the target benchmarks, the updated finding is:

> The core reasoning capability is the same across short and long videos, and the **core visual grounding schema should also be the same**. What differs is not the semantics of reasoning, but **how grounded observations are stored, compressed, and retrieved**.

### Short vs. long video

| Aspect | Short context-rich video | Long video |
|--------|--------------------------|------------|
| **Representative benchmarks** | Video-Holmes, SIV-Bench | VRBench, LongVideoBench, CG-Bench, M3-Bench |
| **Video length** | seconds to minutes | tens of minutes to hours |
| **Fits in VLM context** | usually yes | usually no |
| **Needs grounding** | yes | yes |
| **Needs persistent memory / retrieval** | usually no | yes |
| **Reasoning style** | multi-step reasoning over grounded local evidence | multi-step reasoning over retrieved grounded evidence |

### Architectural implication

The system should always include three conceptual layers:

1. **A reasoning core (always active)**  
   Performs multi-step causal, temporal, and social reasoning over grounded evidence.

2. **A social grounding layer (always active)**  
   Converts raw video into structured entity-, interaction-, event-, and social-state representations.

3. **A persistence / retrieval layer (conditionally active)**  
   Activated only when the grounded video state exceeds the context budget. Short videos keep grounded states in-context; long videos store them in a hierarchical memory index.

### Design consequence

Short videos do **not** mean “no grounding.”  
They mean **grounding without persistent retrieval**.

Long videos mean **grounding plus hierarchical storage and retrieval**.

---

## 1. Benchmark Landscape

### Tier 1 — Direct reasoning over grounded short videos

#### Video-Holmes — Multi-hop Visual Deduction

- **Video length**: 1–5 min
- **QA format**: MC with `<redacted_thinking>` reasoning trace
- **What it tests**: compositional, clue-linking reasoning across multiple moments in a clip
- **Why it matters**: strongest short-video reasoning benchmark for direct mode; `<redacted_thinking>` maps to our `[Think]` protocol
- **Grounding demand**: medium — local event/clue grounding, temporal ordering, evidence linking
- **Persistence demand**: low — grounded windows can remain in-context

#### SIV-Bench — Social Interaction & Mental-State Reasoning

- **Video length**: short clips (~tens of seconds; ~32 s average in corpus)
- **QA format**: MC with multiple subtitle conditions (original / added / removed)
- **What it tests**: intentions, beliefs, emotions, relationship inference, counterfactual social reasoning
- **Why it matters**: primary benchmark for `social_reason`
- **Grounding demand**: high — social interaction grounding, social-state hypotheses, subtitle-aware evidence attribution
- **Persistence demand**: low — grounded states can remain in-context, but the representation must still be structured

### Tier 2 — Retrieval-based reasoning over grounded long videos

#### VRBench — Multi-step Narrative Reasoning

- **What it tests**: reasoning across hour-long narratives with **timestamped intermediate reasoning steps** (gold supervision for multi-step chains)
- **Grounding demand**: high
- **Persistence demand**: high — must retrieve events and evidence across long time spans

#### LongVideoBench — Referring Reasoning under Context Pressure

- **What it tests**: finding and reasoning over relevant windows in long interleaved video/subtitle streams
- **Grounding demand**: high
- **Persistence demand**: high — cross-modal retrieval over video + subtitles

#### CG-Bench — Clue-Grounded QA

- **What it tests**: answer correctness plus retrieval of supporting clues
- **Grounding demand**: high
- **Persistence demand**: high — evidence attribution down to clips / windows / frames

#### M3-Bench — Entity-Grounded Factual Recall

- **What it tests**: long-range entity tracking, speaker-aware recall, factual retrieval; open-ended + `[Search]/[Answer]` style protocol
- **Grounding demand**: highest — face/voice/person tracking and entity grounding
- **Persistence demand**: highest — entity-centric memory and name/ID translation

### Tier 3 — Extended social benchmarks (from self-evolving design)

| Benchmark | Focus | Why It Tests Our System |
|-----------|-------|------------------------|
| **MA-EgoQA** | Multi-agent social interaction, ToM, task coordination | Perspective-aware reasoning across multiple agents |
| **EgoLife** | Long-term ego-centric daily life | Long-range memory + social tracking at scale |
| **LongVidSearch** | Multi-hop evidence retrieval | Evidence chaining quality |

### Summary

All benchmarks need grounded representations.  
Only long-video benchmarks require persistent indexing and retrieval.

---

## 2. SocialVideoGraph — Unified Visual Grounding Data Model

**Persistence model (canonical):** the controller uses **three memory stores** — episodic, semantic, and state (social + spatial subfields) — plus an **evidence attachment layer** for visual/audio/subtitle refs. See [Agentic Memory](agentic_memory_design.md).  
This section describes the **grounding graph / index schema** (`SocialVideoGraph`): structural node kinds for retrieval and APIs, **not** five independent top-level “memory products.” In particular, **visual evidence is not a separate store**; it attaches to episodic and state records.

The graph combines:

- **Structural nodes** — entities, interactions, events (for indexing and traversal).  
- **Backbone nodes** — episodic windows and semantic summaries (aligned with episodic / semantic stores).  
- **Hypotheses and stance** — materialize primarily under **state memory** at query time; the graph may still expose `social_state`-style nodes for search, but they should not duplicate a parallel “social memory bank” outside the three-store model.

### 2.1 Node types

```python
@dataclass
class GroundingNode:
    node_id: str
    node_type: str  # "entity" | "interaction" | "event" | "social_hypothesis" | "episodic" | "semantic"
    text: str
    timestamp: Tuple[float, float]
    clip_id: Optional[str]
    entity_ids: List[str]
    confidence: float
    evidence_refs: List[str]
    metadata: Dict[str, Any]
```

### 2.2 Core node semantics

**Entity nodes** — Represent people, objects, groups, and optionally speakers.

Examples: `person_12`, `group_A`, `object_phone_3`

Attributes may include: appearance, pose, gaze target, speaking state, role, location in scene.

**Interaction nodes** — Represent socially meaningful relations between entities.

Examples: `talking_to`, `looking_at`, `following`, `helping`, `blocking`, `yielding`, `ignoring`, `confronting`, `handing_over`.

**Event nodes** — Represent temporally localized social or physical events.

Examples: `enters_room`, `joins_group`, `starts_argument`, `passes_object`, `leaves_scene`, `group_splits`, `person_returns`.

**Social-hypothesis nodes** — Latent interpretations as **hypotheses** (not hard facts), typically rolled into **state memory** for query-time reasoning while remaining evidence-linked in the graph.

Examples: intention, belief, uncertainty, emotion, alliance, conflict, deception risk, trust shift.

Each hypothesis should include: target entity / pair / group, confidence, evidence refs, optional revision history.

**Episodic nodes** — Local grounded descriptions of short temporal windows.

**Semantic nodes** — Compressed summaries aggregated from multiple episodic / event nodes.

### 2.3 Core graph API

```python
class SocialVideoGraph:
    def add_entity(...)
    def add_interaction(...)
    def add_event(...)
    def add_social_hypothesis(...)  # indexes hypotheses; state memory holds query-time snapshot
    def add_episodic(...)
    def add_semantic(...)

    def search(query, top_k=5, clip_filter=None, entity_filter=None, time_range=None):
        ...

    def get_timeline(entity_id):
        ...

    def get_relations(entity_id):
        ...

    def get_evidence(node_id):
        ...

    def translate(text):
        ...

    def back_translate(text):
        ...

    def save(path):
        ...

    def load(path):
        ...
```

### 2.4 Design principle

All high-level social conclusions must be **evidence-linked**.  
No social claim should exist without timestamps, clip refs, subtitle refs, or frame refs.

### 2.5 Integration point

- Wraps a `MemoryStore` (from `rag/retrieval.py`) for the embedding index when operating in retrieval mode.
- Accepts either a `TextEmbedder` or `MultimodalEmbedder` at construction time.
- Entity nodes store face/voice embeddings matched via cosine similarity where applicable (e.g. M3-Bench).

### 2.6 Grounded-window wire format (normative)

The per-window JSON emitted by the grounding pipeline is the normative contract between observer and reasoner. Every field below is mandatory unless explicitly marked optional. Confidence and uncertainty fields are first-class: any consumer (retrieval, verification, controller) may discard or down-weight items below a configurable threshold.

```python
# Scalars and identifiers
NodeId        = str          # e.g. "win_ae4d4b6d80", "int_e29a54e2b1", "char_12"
EntityRef     = str          # "<face_3>" | "<voice_5>" | "character_12" after refresh_equivalences
EvidenceId    = str          # "frm_81de6284e4" | "sub_000f2" | "vseg_voice7_s3"
Modality      = Literal["frame", "subtitle", "voice", "clip"]
Provenance    = Literal["directly_observed", "inferred_from_behavior",
                        "derived_from_subtitle", "revised_from_prior"]

@dataclass
class EvidenceRef:
    ref_id: EvidenceId
    modality: Modality
    timestamp: tuple[float, float]   # seconds, inclusive
    locator: dict                    # {path, frame_index} | {srt_index, text} | {voice_node, seg_index}
    text: str | None                 # subtitle / ASR text, if applicable

@dataclass
class EntityAttributes:
    emotion: str | None
    gaze: str | None                 # target entity_ref or "forward"/"off-screen"
    speaking: bool
    pose: str | None                 # optional: standing/sitting/walking/...
    role: str | None                 # optional: scene role tag
    location: str | None             # optional: scene-local place label

@dataclass
class EntityMention:
    id: EntityRef                    # window-local if pre-equivalence, character_N after
    type: Literal["person", "group", "object", "speaker"]
    attributes: EntityAttributes
    confidence: float                # 0.0–1.0
    identity_status: Literal["resolved", "unresolved", "alias", "conflict"]

@dataclass
class Interaction:
    src: EntityRef
    rel: str                         # vocabulary: see §3.5
    dst: EntityRef
    confidence: float
    evidence_refs: list[EvidenceId]
    metadata: dict                   # free-form

@dataclass
class Event:
    type: str                        # e.g. "enters_room", "relationship_shift"
    agents: list[EntityRef]
    confidence: float
    description: str
    evidence_refs: list[EvidenceId]
    metadata: dict                   # may contain {"revised_from": <EventId>, "level": 2|3}

@dataclass
class SocialHypothesis:
    type: Literal["intention", "belief", "emotion", "alliance", "conflict",
                  "trust", "suspicion", "goal", "deception_risk"]
    target: EntityRef                # or pair / group (comma-joined)
    value: str
    confidence: float
    provenance: Provenance
    supporting_evidence: list[EvidenceId]
    contradicting_evidence: list[EvidenceId]
    revised_from: NodeId | None      # previous hypothesis this supersedes

@dataclass
class SubtitleSpan:
    ref_id: EvidenceId               # also appears in EvidenceRef
    time_span: tuple[float, float]
    text: str
    speaker: EntityRef | None        # resolved via voice node when possible

@dataclass
class GroundedWindow:
    window_id: NodeId
    time_span: tuple[float, float]
    scene: str
    subtitle_mode: Literal["origin", "added", "removed", "none"]
    entities: list[EntityMention]
    interactions: list[Interaction]
    events: list[Event]
    social_hypotheses: list[SocialHypothesis]
    subtitle_spans: list[SubtitleSpan]
    evidence: list[EvidenceRef]      # superset referenced by all *_refs above
    frame_indices: list[int]
    confidence: float                # window-level aggregate
    metadata: dict                   # e.g. {"cut_boundary": True, "level": 1}
```

**Invariants that pipeline consumers may assume:**

1. Every `EntityRef` appearing in `interactions`, `events`, `social_hypotheses`, or `subtitle_spans.speaker` is either (a) present in `entities`, or (b) resolvable via `SocialVideoGraph.character_mappings` to an `img`/`voice` node.
2. Every `EvidenceId` in any `*_refs` list appears in `evidence` (closed reference set).
3. `supporting_evidence` is non-empty for any `social_hypothesis` with `confidence ≥ 0.5`.
4. `time_span` on every sub-structure lies within the window's `time_span`.
5. `confidence` fields are calibrated to the range `[0, 1]`; a node with `confidence < 0.2` **must** carry an explanation in `metadata["low_confidence_reason"]`.

Mapping to episodic memory writes (see [Agentic Memory](agentic_memory_design.md)): one `GroundedWindow` materializes as one `episodic` entry carrying links to its `entities`, attached `interactions`/`events`, and pointers into the evidence store. `social_hypothesis` entries flow into **state memory** at query time; they remain indexed in the graph for retrieval but are not copied into episodic.

### 2.7 Entity resolution & re-identification

Entity identity persistence is the load-bearing capability that separates "window-local schema" from a usable grounding graph. The pipeline uses a three-stage resolver, realized by the [Grounding Pipeline Execution Plan](grounding_pipeline_execution_plan.md) Phase 2.

**Stage A — per-clip clustering.**

- **Faces:** InsightFace `buffalo_l` (RetinaFace + ArcFace) extracts face embeddings per sampled frame; within-clip DBSCAN (`eps=0.5`, `min_samples=2`) groups detections into face tracks. Tracks below a quality / detection-score threshold are dropped (see m3-agent `processing_config.json`: `face_detection_score_threshold`, `face_quality_score_threshold`).
- **Voices:** Speaker diarization segments the clip (Gemini-1.5-pro by default; Whisper diarizer as fallback); ERes2NetV2 produces per-segment embeddings; segments below `min_duration_for_audio` are dropped.

**Stage B — cross-clip matching into graph nodes.**

- For each new face cluster, search the graph for existing `img` nodes via cosine similarity; merge when similarity ≥ `img_matching_threshold` (default 0.3), otherwise create a new `img` node. Same for `voice` nodes with `audio_matching_threshold` (default 0.6).
- Each `img`/`voice` node stores up to `max_img_embeddings` / `max_audio_embeddings` embeddings; overflow is reservoir-sampled (m3-agent `update_node`).
- The graph records **per-clip appearance evidence** on each entity node (`metadata.contents`: base64 face crops, ASR text, start/end times).

**Stage C — cross-modal equivalence (face ↔ voice → character).**

- Stage-1 caption prompt (`prompt_generate_memory_with_ids_sft`) is required to emit explicit equivalence assertions whenever a face is observed to be the speaker of a voice, e.g. `"equivalence: <face_3> = <voice_5>"`.
- After the full video is processed, `SocialVideoGraph.refresh_equivalences()` runs union-find over all equivalence assertions attached to `img`/`voice` nodes, producing `character_mappings: {character_N: [face_i, voice_j, ...]}` and the reverse map.
- Downstream consumers use `character_N` (video-global) rather than per-clip `<face_N>`.

**Failure handling.**

| Failure mode | Detection signal | Repair |
|---|---|---|
| Identity drift (same person, new `img` node) | Two `img` nodes with cosine similarity ≥ `img_matching_threshold` but not merged because they were created in the same clip | Post-processing merge pass at end of video |
| Voice attribution conflict | Stage-1 emits two equivalence assertions `<voice_5> = <face_3>` and `<voice_5> = <face_7>` | `fix_collisions(mode="eq_only")` keeps the assertion with the highest edge weight |
| Occlusion / missing face | Person clearly speaks (voice node active) but no face embedding for the clip | Graph records only `voice_N`; entity_ref falls back to the voice tag until a future clip adds face evidence |
| Alias mismatch | Subtitle text names a character that has no resolved identity (`"Alice said"`) | Store as candidate alias on the voice node's metadata; resolved when equivalence is later asserted |
| Confidence decay for stale identities | Character unseen for more than N clips | Its hypotheses in state memory are confidence-discounted (see [Agentic Memory](agentic_memory_design.md) revision policy) |

**M3-Bench name round-trip.** `SocialVideoGraph.translate(text)` and `back_translate(text)` are thin wrappers over `character_mappings` and `reverse_character_mappings`; the M3-Bench adapter calls `back_translate` on the question before retrieval and `translate` on the answer before returning it. See [Grounding Pipeline Execution Plan §2.3, Phase 2].

---

## 3. Unified Grounding Pipeline

This pipeline is used for **both** short and long videos.  
The difference is whether outputs stay in-context or are persisted into a retrieval index.

### 3.1 Pipeline overview

```
Raw video
  │
  ├─ 1. adaptive_segment(video) -> windows / clips
  │
  ├─ 2. perceptual grounding
  │     • frame sampling
  │     • object / person / face detection
  │     • optional voice diarization
  │     • subtitle / ASR alignment
  │
  ├─ 3. local social grounding
  │     • entities
  │     • actions
  │     • spatial relations
  │     • gaze / speaking / turn-taking
  │     • interactions
  │     • local event hypotheses
  │
  ├─ 4. temporal consolidation
  │     • merge adjacent windows
  │     • track entities across clips
  │     • build event nodes
  │     • update relationship / intention hypotheses
  │
  ├─ 5A. direct mode buffer (short videos)
  │     • keep grounded windows in-context
  │
  └─ 5B. retrieval mode memory (long videos)
        • build hierarchical graph / index
        • semantic distillation
        • entity-centric timeline memory
        • query-time retrieval
```

### 3.2 Per-window grounding output

Each window should produce structured JSON, for example:

```json
{
  "time_span": [12.4, 16.8],
  "scene": "kitchen conversation",
  "entities": [
    {
      "id": "p1",
      "type": "person",
      "attributes": {
        "emotion": "tense",
        "gaze": "p2",
        "speaking": true
      }
    }
  ],
  "interactions": [
    {
      "src": "p1",
      "rel": "talking_to",
      "dst": "p2",
      "confidence": 0.84
    }
  ],
  "events": [
    {
      "type": "confrontation_start",
      "agents": ["p1", "p2"],
      "confidence": 0.66
    }
  ],
  "social_hypotheses": [
    {
      "type": "intention",
      "target": "p1",
      "value": "seeking explanation",
      "confidence": 0.61
    }
  ],
  "evidence": {
    "frames": [188, 196, 204],
    "subtitle_spans": ["Why didn’t you tell me?"]
  }
}
```

### 3.3 Direct mode for short videos

**Used for:** Video-Holmes, SIV-Bench.

**Procedure:**

- Sample windows densely (or frame-sample + window as needed).
- Run local social grounding on each window.
- Keep all grounded windows in an **in-context buffer**.
- Reason directly over raw frames/clips, subtitles/transcripts, and grounded structured states.

No persistent retrieval index is required.

### 3.4 Retrieval mode for long videos

**Used for:** VRBench, LongVideoBench, CG-Bench, M3-Bench.

**Procedure:**

- Generate local grounded windows.
- Merge windows into event-level nodes.
- Build entity timelines and relationship memory.
- Distill semantic summaries for long-range compression.
- Support retrieval by: entity, relation, event type, time range, subtitle phrase, social-state hypothesis, evidence chain.

### 3.5 Social reasoning requirements

A social-first pipeline must support at least:

- person tracking across time  
- group tracking  
- turn-taking / speaking state  
- gaze / attention cues  
- interaction typing  
- relationship change over time  
- intention / belief / emotion hypotheses with confidence  
- evidence attribution for every high-level claim  

### 3.6 8B-driven builder variant (orchestrator-aligned)

For long videos, an implementation can follow the adaptive sampling + batched VLM pattern:

```
Video
  │
  ├─ 1. Adaptive frame sampling
  │     • Base rate: 1 frame / 2s (short), 1 frame / 5s (long)
  │     • + keyframes at scene-change boundaries (pixel diff > threshold)
  │     • + subtitle-aligned frames (e.g. M3-Bench robot)
  │
  ├─ 2. Batch VLM grounding  (e.g. Qwen3-VL-8B)
  │     • Send frames in batches of 4-8 (multi-image input)
  │     • Per-frame or per-window output: structured JSON
  │       {objects, actions, spatial, scene, emotions, predicates, interactions}
  │     • Cross-window: entity tracking via face/voice embeddings
  │
  ├─ 3. Episodic / window nodes -> SocialVideoGraph
  │     • Temporal aggregation: merge adjacent windows when predicate delta is low
  │     • Entity linking: face_0, voice_1, etc.
  │
  ├─ 4. Semantic distillation  (second pass)
  │     • Summarize clusters into semantic nodes
  │     • Each semantic node carries: text, linked entity_ids, confidence, evidence_refs
  │
  └─ 5. Entity resolution
        • Merge face/voice detections across clips
        • Build entity profile nodes (appearance, role, relationships)
        • Map <face_N> / <voice_N> IDs to consistent entity_ids
```

### 3.7 Compute budget (indicative)

| Video length | Frames sampled | 8B inference calls | Wall time (1× A100) |
|-------------|----------------|-------------------|---------------------|
| 2 min (Video-Holmes) | ~60 | ~15 batches + 1 semantic pass | ~45 sec |
| 30 min (M3-Bench robot) | ~360 | ~90 batches + 5 semantic passes | ~8 min |
| 2 hr (long-form) | ~1400 | ~350 batches + 15 semantic passes | ~30 min |

---

## 4. Hierarchical Memory for Long Videos

Long videos should be stored hierarchically.

| Level | Role |
|-------|------|
| **Level 1 — grounded windows** | Short local clips with entities, interactions, and evidence. |
| **Level 2 — event memory** | Merged event nodes: greeting, argument, alliance formation, pursuit, help request, refusal, deception cue, group split / merge, … |
| **Level 3 — entity / relationship memory** | Long-range summaries: e.g. p1 repeatedly avoids p2; p3 aligns with group_A; speaker_2 likely same as face_7. |
| **Level 4 — semantic summaries** | Compressed cross-scene summaries for retrieval efficiency. |
| **Level 5 — query-time reasoning trace** | At inference: retrieve relevant events, entity timelines, subtitle/audio evidence, social-state hypotheses; **final answers come from retrieved evidence**, not ungrounded free-form summarization. |

---

## 5. Benchmark Evaluation Adapters

Thin wrappers that map each benchmark’s I/O format to `build_grounded_context(...)` and the unified `reason(...)` entry point.

### 5.1 Direct mode adapters

```python
class VideoHolmesAdapter:
    def evaluate(self, video_path, question, options) -> dict:
        grounded = build_grounded_context(video_path, mode="direct")
        return reason(question, options, video_context=grounded, mode="direct")

class SIVBenchAdapter:
    def evaluate(self, video_path, question, options, subtitle_mode="origin") -> dict:
        grounded = build_grounded_context(
            video_path,
            mode="direct",
            subtitle_mode=subtitle_mode,
            social_grounding=True,
        )
        return reason(question, options, video_context=grounded, mode="direct")
```

### 5.2 Retrieval mode adapters

```python
class VRBenchAdapter:
    def evaluate(self, video_path, question, options=None, graph=None) -> dict:
        graph = graph or build_grounded_context(video_path, mode="retrieval")
        return reason(question, options, video_context=graph, mode="retrieval")

class LongVideoBenchAdapter:
    def evaluate(self, video_path, question, options, graph=None) -> dict:
        graph = graph or build_grounded_context(
            video_path,
            mode="retrieval",
            include_subtitles=True,
        )
        return reason(question, options, video_context=graph, mode="retrieval")

class CGBenchAdapter:
    def evaluate(self, video_path, question, options=None, graph=None) -> dict:
        graph = graph or build_grounded_context(video_path, mode="retrieval")
        return reason(question, options, video_context=graph, mode="retrieval")

class M3BenchAdapter:
    def evaluate(self, video_path, question, graph=None) -> dict:
        graph = graph or build_grounded_context(
            video_path,
            mode="retrieval",
            include_subtitles=True,
            entity_tracking=True,
            voice_tracking=True,
        )
        question = graph.back_translate(question)
        result = reason(question, None, video_context=graph, mode="retrieval")
        result.answer = graph.translate(result.answer)
        return result
```

### 5.3 Shared interface

```python
def evaluate(self, video_path: str, question: str, **kwargs) -> dict
```

Returns at minimum:

```json
{
  "answer": "…",
  "reasoning": "…",
  "mode": "direct | retrieval",
  "evidence": []
}
```

---

## 6. Per-Benchmark Configuration Cheat Sheet

| Benchmark | Mode | Grounding | Subtitles | Entity tracking | Social-state hypotheses | Retrieval |
|-----------|------|-----------|-----------|-----------------|---------------------------|-----------|
| Video-Holmes | direct | local clue/event grounding | optional | no | low | no |
| SIV-Bench | direct | local social grounding | yes | light | high | no |
| VRBench | retrieval | event/narrative grounding | optional | no | medium | yes |
| LongVideoBench | retrieval | video + subtitle grounding | yes | no | low–medium | yes |
| CG-Bench | retrieval | clue/evidence grounding | optional | no | low | yes |
| M3-Bench | retrieval | entity-aware grounding | yes | high | medium | yes |

### 6.1 Benchmark-to-capability mapping

Each benchmark stresses a different subset of the grounding, memory, retrieval, and reasoning capabilities. This table is the source of truth for "which subsystem must work for this benchmark to move" and drives the ablation design in [`evaluation_ablation_plan.md`](plan_docs_implementation_checklist.md#7-new-file-infra_plansevaluation_ablation_planmd) (pending).

Columns:

- **Stressed (S):** the capability is load-bearing for accuracy on this benchmark; degrading it should visibly drop the score.
- **Supervised (G):** the benchmark provides gold labels that can directly train or evaluate this capability.
- **Evaluation-only (E):** the capability is exercised but not directly graded.
- **—:** not exercised.

| Capability | Video-Holmes | SIV-Bench | VRBench | LongVideoBench | CG-Bench | M3-Bench |
|---|---|---|---|---|---|---|
| Adaptive sampling (§3.1) | S | S | S | S | S | S |
| Subtitle / ASR ingestion (§3.1) | E | **S** (mode ablation) | E | **S** | E | **S** |
| Face detection + cross-clip track (§2.7) | E | S | E | E | E | **S** + G (answers cite names) |
| Voice diarization + embedding (§2.7) | — | S | E | E | — | **S** |
| Equivalence (face ↔ voice → character) (§2.7) | — | E | E | E | — | **S** + G |
| Entity-level episodic memory (§3.3 / §3.4) | S | S | S | S | S | **S** |
| Interaction grounding (§2.2) | S | **S** + G | S | E | E | E |
| Event grounding (§2.2) | **S** + G (multi-hop clues) | S | **S** + G (narrative steps) | E | **S** + G (clue spans) | E |
| Social-hypothesis grounding (§2.2) | S | **S** + G (belief/intention) | E | — | — | E |
| Level-3 relationship memory (§4) | — | E | S | E | — | **S** |
| Level-4 semantic distillation (§4) | — | — | S | **S** | E | **S** |
| Retrieval recall@k (§7.4) | — | — | **S** + G | **S** + G | **S** + G | **S** + G |
| Evidence-chain retrieval (§7.4) | E | E | **S** + G | E | **S** + G (clue spans) | S |
| Entity-centric retrieval (§2.5) | — | E | — | — | — | **S** |
| Name round-trip (`translate` / `back_translate`) (§2.7) | — | — | — | — | — | **S** + G |
| Perspective / ToM reasoning | E | **S** (ToM questions) | E | — | — | E |
| Reasoning-trace supervision (§7.1) | **S** + G (`<redacted_thinking>`) | E | **S** + G (timestamped steps) | — | E | E |

How to read the table during development:

- An **S** cell that fails evaluation implicates the corresponding subsystem directly.
- A **G** cell tells you where direct supervision / fine-tuning signal exists; prioritize these cells for atomic-skill training (see [`skill_extraction_bank.md`](skill_extraction_bank.md)).
- Absence of **S** marks (all `E`/`—`) in a capability column means no benchmark currently stresses that capability; either add one or de-prioritize the capability.

---

## 7. Evaluation Plan

### 7.1 End-task metrics

- answer accuracy  
- open-ended answer quality  
- reasoning trace quality  
- evidence grounding correctness  

### 7.2 Grounding metrics

- entity grounding accuracy  
- interaction grounding accuracy  
- event grounding accuracy  
- evidence attribution correctness  
- temporal consistency  
- entity identity consistency  
- subtitle–visual alignment quality  

### 7.3 Social reasoning metrics

- intention / belief inference accuracy  
- relationship classification accuracy  
- counterfactual support quality  
- consistency of social-state hypotheses over time  

### 7.4 Retrieval metrics for long videos

- retrieval recall@k  
- evidence chain correctness  
- cross-segment linkage accuracy  
- answer correctness conditioned on retrieved evidence  

### 7.5 Self-evolving system benchmarks (extended)

| Benchmark | Focus | Why It Tests Our System |
|-----------|-------|------------------------|
| **Video-Holmes** | Deep causal/temporal/social reasoning (short films) | Reasoning depth over social dynamics |
| **MA-EgoQA** | Multi-agent social interaction, ToM, task coordination | Perspective-aware reasoning across multiple agents |
| **EgoLife** | Long-term ego-centric daily life | Long-range memory + social tracking at scale |
| **LongVidSearch** | Multi-hop evidence retrieval | Evidence chaining quality |

### 7.6 Additional cross-cutting metrics

| Metric | What It Measures | How |
|--------|-----------------|-----|
| **Multi-hop retrieval quality** | Evidence chained across distant segments | Count correct cross-segment links |
| **Memory efficiency** | Compactness vs. video length | Nodes per minute; retrieval precision |
| **Tool call count** | Frozen-model calls per question | Count observer/reasoner invocations |
| **Skill reuse rate** | Skills reused across questions | Unique skills / total invocations |
| **Long-range dependency handling** | Evidence >5 min apart | Stratify accuracy by temporal span |

### 7.7 Orchestrator evaluation commands

```bash
# Video-Holmes
python -m small_model_orchestrator.run_offline \
    --dataset video_holmes --output output/vh_graphs/ --build_skills
python -m small_model_orchestrator.run_eval \
    --dataset video_holmes --graphs output/vh_graphs/ \
    --skill_bank output/vh_skill_bank.jsonl \
    --large_model Qwen/Qwen3-VL-72B-Instruct --output output/vh_results.json

# M3-Bench
python -m small_model_orchestrator.run_offline \
    --dataset m3_bench --output output/m3_graphs/
python -m small_model_orchestrator.run_eval \
    --dataset m3_bench --graphs output/m3_graphs/ \
    --large_model Qwen/Qwen3-VL-72B-Instruct

# SIV-Bench
python -m small_model_orchestrator.run_offline \
    --dataset siv_bench --subset caregiver-recipient --output output/siv_graphs/
python -m small_model_orchestrator.run_eval \
    --dataset siv_bench --graphs output/siv_graphs/ \
    --large_model Qwen/Qwen3-VL-72B-Instruct
```

### 7.8 Expected improvements (reference targets)

| Benchmark | Baseline (72B raw) | With orchestrator | Why |
|-----------|--------------------|--------------------|-----|
| Video-Holmes | ~65% acc | ~72–75% acc | Skills provide reasoning scaffolds for causal/temporal questions |
| M3-Bench | ~55% acc | ~65–70% acc | Memory graph handles long-range entity tracking |
| SIV-Bench | ~60% acc | ~68–72% acc | Entity relationship skills transfer well to social interaction |

### 7.9 Ablation designs

**Self-evolving system ablations:**

| Variant | What's Removed | What It Tests |
|---------|---------------|---------------|
| **No social-state memory** | Remove social-state entries | Value of tracking beliefs/intentions |
| **No perspective threads** | Remove per-character tracking | Value of perspective-aware reasoning |
| **No hierarchical memory** | Flat event-level only | Value of multi-timescale organization |
| **No uncertainty fields** | Remove confidence, provenance | Value of calibrated inference |
| **No skill evolution** | Freeze skill bank after cold start | Value of continual improvement |
| **Static skill bank** | Hand-written skills only | Value of learned vs. designed skills |
| **No 8B controller** | Direct 72B QA on raw video | Full system contribution vs. raw VLM |
| **8B without skills** | Controller manages memory only | Value of skill bank specifically |

**Orchestrator ablations:**

| Variant | Memory | Skills | Keyframes | Self-reflect | Tests |
|---------|--------|--------|-----------|-------------|-------|
| **Full** | Yes | Yes | Yes | Yes | Main result |
| No skills | Yes | No | Yes | Yes | Isolates skill contribution |
| No memory | No | Yes | Yes (random) | No | Isolates memory contribution |
| No keyframes | Yes | Yes | No (text only) | Yes | Isolates visual grounding |
| No reflection | Yes | Yes | Yes | No | Isolates verification value |
| 8B only | Yes | Yes | Yes | Yes | Difficulty routing sends all to 8B |
| 72B raw | No | No | All frames | No | Baseline: large model on raw video |

---

## 8. Final Design Rule

The system should never treat visual grounding as only “objects in frames.”

For this project, **visual grounding** means:

**grounding socially meaningful state transitions with traceable evidence over time.**

That definition is what makes the same pipeline usable for short context-rich clips, long videos, social reasoning benchmarks, and future skill-bank / memory-agent integration.

---

## 9. Integration with Existing Components

| Existing component | How it connects |
|-------------------|-----------------|
| `rag/retrieval.py` → `MemoryStore` | `SocialVideoGraph` wraps a `MemoryStore` in retrieval mode |
| `rag/embedding/` → `TextEmbedder`, `MultimodalEmbedder` | Used by graph embedding index and grounding pipeline |
| `data_structure/experience.py` → `Experience`, `Episode` | Each reasoning trace can be packaged as an `Episode` |
| `skill_agents/stage3_mvp/schemas.py` → `Skill`, `Protocol` | Visual skills use these schemas |
| `decision_agents/agent.py` → `VLMDecisionAgent` | Future `VideoQAAgent` subclass uses `select_skill` |

---

## 10. Implementation Order

| Phase | Files | Depends on | Priority |
|-------|-------|------------|----------|
| **Phase 1** | `reasoning.py` | VLM callable only | **Highest** — unlocks all benchmarks |
| **Phase 2** | `adapters.py` (Tier 1: direct + `build_grounded_context`) | Phase 1 | **High** — Video-Holmes + SIV-Bench |
| **Phase 3** | `social_video_graph.py` (or evolve `video_memory.py`) | `rag/retrieval.py`, `rag/embedding/` | **High** — retrieval mode |
| **Phase 4** | `observe.py` / unified grounding | Phase 3, face/voice utils | **High** — all retrieval-mode adapters |
| **Phase 5** | `adapters.py` (Tier 2) | Phase 3–4 | **High** |
| **Phase 6** | `skills.py` | Phase 1–5 | Lower — skill bank integration |
| **Phase 7** | `__init__.py`, integration tests | All above | Final |

The **operational** phasing (perception stack vendoring, adaptive sampler, two-stage extraction, consolidation, adapters, validation) is tracked in [`grounding_pipeline_execution_plan.md`](grounding_pipeline_execution_plan.md) and complements this design-level order.

---

## 11. Grounding error taxonomy

Six canonical failure modes observed on the current `Video_Skills/out/claude_grounding/*.json` baseline, each with a detection signal and a repair action. Pipeline runs must emit per-failure counters into `out/<run_id>/grounding_errors.jsonl` so the `evaluation_ablation_plan.md` (pending) and reflection loop ([Skill Synthetics Agents](skill_synthetics_agents.md)) can diagnose regressions.

| # | Error class | Symptom in output | Detection signal | Repair |
|---|---|---|---|---|
| **E1 — Identity drift** | Same person receives different `entity_id` across windows (current Claude baseline: `p1` is a different person in each scene) | Post-hoc cosine similarity between `img`/`voice` nodes exceeds `img_matching_threshold` yet no edge exists; or `character_mappings` has >> expected cluster count | End-of-video re-clustering pass; raise thresholds; add equivalence assertion if stage-1 missed it |
| **E2 — Missing entity** | A clearly visible / audible person produces no `img` or `voice` node | Frames with face detections below quality/detection thresholds but above a relaxed floor; voice segments shorter than `min_duration_for_audio` | Lower per-benchmark thresholds; relax `max_faces_per_character`; run secondary face detector |
| **E3 — Wrong speaker attribution** | `subtitle_span.speaker` points at a face that isn't speaking; or equivalence assertion maps `<voice_N>` to the wrong `<face_M>` | Mouth-movement keyframe disagrees with active voice segment; two conflicting equivalence assertions | `fix_collisions(mode="eq_only")`; require lip-movement confirmation before writing equivalence |
| **E4 — Subtitle / frame misalignment** | Subtitle text references event not visible in the sampled frames for that time range | SRT timestamp lies outside any sampled frame's `time_span`; ASR text entities have no matching entity in the window | Add subtitle-aligned keyframes to sampler; widen window boundary to include adjacent cut |
| **E5 — Hypothesis without evidence** | `social_hypothesis` node has empty `supporting_evidence` despite `confidence ≥ 0.5` (current Claude baseline: many hypotheses reference only `"frame_0"` which isn't in the evidence list) | Wire-format invariant #3 (see §2.6) violated; or `supporting_evidence` ids not present in `window.evidence` | Reject the hypothesis at stage-2 parse; log to `pipeline_errors.jsonl`; require the stage-2 prompt to re-emit with evidence |
| **E6 — Semantic over-compression** | Distilled semantic node subsumes contradictory episodic claims (e.g. alliance + conflict between same pair) without a `relationship_shift` event | Level-4 summary cosine-similar (≥ `semantic_drop_threshold`) to two semantic nodes with disjoint supporting entity sets | Split the summary; emit a `relationship_shift` event (§4); re-run distillation on the split clusters |

Two additional classes are tracked but not yet auto-detected:

- **E7 — Perspective collapse**: different characters' beliefs merged into one "world view" (blocks ToM benchmarks). Requires per-character perspective threads (see [Actors / Reasoning Model](actors_reasoning_model.md)).
- **E8 — Retrieval shortcut**: adapter returns an answer without any `evidence_refs` in the returned trace. Detected by verifier in the reasoning loop, not by the grounding layer itself.

Each error class is a first-class metric in the Phase 6 exit criteria of [`grounding_pipeline_execution_plan.md`](grounding_pipeline_execution_plan.md); a pipeline run is considered regression-free only when E1–E6 counts are ≤ baseline.
