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
