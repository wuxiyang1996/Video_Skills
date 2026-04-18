# Video Benchmarks & Visual Grounding — Design Plan

> Sub-folder: `Video_Skills/memory_manage/`
>
> Goal: Define the benchmark landscape, visual grounding infrastructure
> (VideoMemoryGraph, observation pipeline), and benchmark-specific
> evaluation adapters for the COS-PLAY (`Video_Skills`) framework.
>
> **Related plans:**
> - [Actors / Reasoning Model](actors_reasoning_model.md) — reasoning core, 8B controller, orchestrator
> - [Skill Extraction / Bank](skill_extraction_bank.md) — skill definitions and bank infrastructure
> - [Skill Synthetics Agents](skill_synthetics_agents.md) — skill crafting, evolution, quality control

---

## 0. Key Insight — Reasoning vs. Context Management

After auditing all target benchmarks, the core finding is:

> **The reasoning capability needed is the same regardless of video length.
> What differs is only context management — whether the full video fits inside
> the model's context window.**

| | Short video | Long video |
|---|---|---|
| **Benchmarks** | Video-Holmes, SIV-Bench | VRBench, LongVideoBench, CG-Bench, M3-Bench |
| **Video length** | 5 s – 5 min | 10 min – hours |
| **Fits in VLM context?** | Yes | No |
| **Needs explicit memory?** | No — feed raw frames/clip directly | Yes — must segment, index, and retrieve |
| **Core reasoning** | Chain-of-thought over visual evidence | Same chain-of-thought, but over *retrieved* evidence |

The memory graph, entity tracking, and segmentation pipeline from M3-Agent
exist because **the video does not fit in context**. They are a
compression/indexing strategy, not a fundamentally different reasoning paradigm.

**Implication for architecture:**

1. **A reasoning core** (always active) — chain-of-thought with optional
   retrieval, applicable to every benchmark. *(see [Actors plan](actors_reasoning_model.md))*
2. **A memory/retrieval layer** (conditionally active) — builds an indexed
   graph only when the video exceeds a configurable context budget.

---

## 1. Benchmark Landscape

All datasets in the workspace, classified by what they actually test:

### Tier 1 — Direct reasoning (short video, no memory needed)

#### Video-Holmes — Multi-hop Visual Deduction

| | |
|---|---|
| **Video length** | 1–5 min (suspense film clips) |
| **QA format** | MC (A/B/C/D) with `<think>` reasoning trace |
| **Scale** | Curated suspense films |

**What it tests.** Compositional, multi-hop deduction: the model must actively
locate and connect multiple subtle visual cues scattered across different
moments in the clip. Perception alone is insufficient — the cues are easy to
see but hard to *connect*.

**Why it matters for us.** Video-Holmes is the strongest short-video reasoning
benchmark in the set. Although the video fits in context (no memory layer
needed), it demands the same chain-of-thought discipline as the long-video
benchmarks. Its `<think>` tag maps directly to our `[Think]` protocol, making
it the primary testbed for the reasoning core in direct mode.

**Memory-layer demands.** None. The full clip fits in the VLM context window.

---

#### SIV-Bench — Social Interaction & Mental-State Reasoning

| | |
|---|---|
| **Video length** | ~32 s average |
| **QA format** | MC (A–N, up to 14 options) |
| **Scale** | 8.4k QA pairs; 14 relationship types; 3 subtitle conditions |

**What it tests.** Social cognition: inferring unobservable mental states
(beliefs, intentions, emotions), counterfactual reasoning ("what would happen
if..."), and relationship classification — all from short video clips of social
interactions. Three subtitle conditions (original, added, removed) test
whether the model relies on visual vs. textual cues.

**Why it matters for us.** SIV-Bench stresses the `social_reason` skill and
counterfactual inference, which are reasoning flavors not covered by other
benchmarks. The subtitle-condition ablation also lets us measure how robust the
reasoning core is when modality availability changes.

**Memory-layer demands.** None. Short clips fit entirely in context.

### Tier 2 — Memory-dependent (long video)

#### VRBench — Multi-step Narrative Reasoning

| | |
|---|---|
| **Video length** | ~1.6 h average (long narrative films) |
| **QA format** | MC + open-ended; each question includes **timestamped intermediate reasoning steps** |
| **Scale** | Feature-length narrative videos |

**What it tests.** Whether a model can perform genuine multi-step reasoning
across hour-long narratives. Questions require connecting evidence scattered
across different segments of the film.

**Why it matters for us.** VRBench provides **gold intermediate reasoning steps
with timestamps**, not just a final answer. This maps directly to our
`[Think] → [Search] → [Think] → [Answer]` protocol.

**Memory-layer demands.** High. The memory graph must index narrative events
over ~1.6 h and support temporal queries.

---

#### LongVideoBench — Referring Reasoning under Context Pressure

| | |
|---|---|
| **Video length** | Up to 1 hour |
| **QA format** | MC (referring reasoning) |
| **Scale** | Large-scale; diverse video types |

**What it tests.** The ability to locate a relevant context window inside a
long interleaved stream of video frames and subtitles, then reason over that
retrieved context.

**Why it matters for us.** LongVideoBench puts maximum pressure on the
**retrieval** side of the pipeline. It complements VRBench (multi-step
reasoning) and CG-Bench (clue grounding).

**Memory-layer demands.** High. Subtitles are interleaved with visual content,
so the memory graph should store both modalities and support cross-modal search.

---

#### CG-Bench — Clue-Grounded QA

| | |
|---|---|
| **Video length** | Long videos (varies) |
| **QA format** | MC + open-ended (clue-grounded) |
| **Scale** | 1,219 long videos; 12,129 QA pairs |

**What it tests.** Clue-grounded question answering: the model must not only
produce a correct answer but also **retrieve the key visual clues** that
support it.

**Why it matters for us.** CG-Bench is the closest existing benchmark to a
pure memory-retrieval evaluation. It directly measures whether retrieved
memories are the *right* clues.

**Memory-layer demands.** High. The graph must store fine-grained visual
observations so that retrieved clues can be traced back to specific frames.

---

#### M3-Bench — Entity-Grounded Factual Recall

| | |
|---|---|
| **Video length** | 10 min – hours |
| **QA format** | Open-ended + `[Search]/[Answer]` protocol |
| **Scale** | Robot FPV + web videos |

**What it tests.** Entity tracking and factual recall across long timelines.
Questions reference specific people (faces) and speakers (voices).

**Why it matters for us.** M3-Bench is the only benchmark that requires the
full **entity-tracking** stack: face detection/embedding, voice diarization,
entity-ID assignment, and ID ↔ name translation.

**Memory-layer demands.** Highest. Requires entity nodes (face/voice
embeddings), entity-aware retrieval, and bidirectional entity-name translation.

### Tier 3 — Extended social benchmarks (from self-evolving design)

| Benchmark | Focus | Why It Tests Our System |
|-----------|-------|------------------------|
| **MA-EgoQA** | Multi-agent social interaction, ToM, task coordination | Perspective-aware reasoning across multiple agents |
| **EgoLife** | Long-term ego-centric daily life | Long-range memory + social tracking at scale |
| **LongVidSearch** | Multi-hop evidence retrieval | Evidence chaining quality |

### What this means

- **2 out of 6 core benchmarks need zero memory management.** Video-Holmes and
  SIV-Bench need strong reasoning, good perception, and chain-of-thought — but
  the full video fits in context.
- **4 benchmarks require the memory/retrieval layer**, each stressing it
  differently:
  - **VRBench** — explicit multi-step reasoning chains over hour-long narratives
  - **LongVideoBench** — referring reasoning under heavy long-context pressure
  - **CG-Bench** — clue retrieval + reasoning
  - **M3-Bench** — entity-grounded factual recall with face/voice tracking

---

## 2. VideoMemoryGraph — Visual Grounding Data Model

### Data model

```python
@dataclass
class MemoryNode:
    node_id: str                # e.g. "episodic_003", "semantic_007"
    node_type: str              # "episodic" | "semantic" | "entity" | "relational"
    text: str                   # natural-language description
    clip_id: Optional[str]      # which video segment produced this
    timestamp: Tuple[float, float]  # (start_sec, end_sec)
    entity_ids: List[str]       # linked entities, e.g. ["face_0", "voice_1"]
    confidence: float           # reinforcement weight (semantic nodes decay/grow)
    metadata: Dict[str, Any]    # arbitrary extra info (frames, crops, etc.)
```

### Class: `VideoMemoryGraph`

| Responsibility | Method |
|---|---|
| Insert episodic memories | `add_episodic(text, clip_id, timestamp, entity_ids, embedding)` |
| Insert semantic memories | `add_semantic(text, clip_id, entity_ids, embedding)` |
| Track entities (face/voice) | `add_or_update_entity(entity_type, embedding, metadata)` |
| Reinforce / decay semantics | `reinforce(node_id, delta)` |
| Retrieve by query | `search(query, top_k, clip_filter) -> List[MemoryNode]` |
| Retrieve by entity | `get_by_entity(entity_id) -> List[MemoryNode]` |
| Translate entity IDs ↔ names | `translate(text) / back_translate(text)` |
| Serialize | `save(path) / load(path)` |

### Integration point

- Wraps a `MemoryStore` (from `rag/retrieval.py`) for the embedding index.
- Accepts either a `TextEmbedder` or `MultimodalEmbedder` at construction time.
- Entity nodes store face/voice embeddings matched via cosine similarity.

---

## 3. Observation Pipeline (long-video only)

Converts raw long video into structured `MemoryNode` entries. **Skipped
entirely for short-video benchmarks.**

### Pipeline stages

```
Long video file (> context budget)
  │
  ├─ 1. segment_video(path, interval_sec) -> List[Clip]
  │
  ├─ 2. extract_entities(clip) -> List[EntityDetection]
  │     Face detection + embedding (InsightFace/ArcFace).
  │     Voice diarization + embedding (ERes2NetV2 or Gemini).
  │
  ├─ 3. generate_observations(clip, entities, vlm) -> ObservationResult
  │     VLM generates episodic + semantic descriptions with entity IDs.
  │
  └─ 4. store_observations(observations, graph: VideoMemoryGraph)
        Embed texts, insert nodes, link entities.
```

### 8B-driven memory builder variant (from orchestrator plan)

```
Video
  │
  ├─ 1. Adaptive frame sampling
  │     • Base rate: 1 frame / 2s (short), 1 frame / 5s (long)
  │     • + keyframes at scene-change boundaries (pixel diff > threshold)
  │     • + subtitle-aligned frames (M3-Bench robot)
  │
  ├─ 2. Batch VLM grounding  (Qwen3-VL-8B)
  │     • Send frames in batches of 4-8 (multi-image input)
  │     • Per-frame output: structured JSON
  │       {objects, actions, spatial, scene, emotions, predicates}
  │     • Cross-frame: entity tracking via face/voice embeddings
  │
  ├─ 3. Episodic memory construction
  │     • Each frame → MemoryNode(type="episodic")
  │     • Temporal window aggregation: merge 3-5 consecutive frames
  │       into a single episodic node when predicate delta is low
  │     • Entity linking: face_0, voice_1, etc.
  │
  ├─ 4. Semantic memory distillation  (Qwen3-VL-8B, second pass)
  │     • Summarize clusters of episodic nodes into semantic nodes
  │     • Each semantic node carries: text, linked entity_ids, confidence
  │
  └─ 5. Entity resolution
        • Merge face/voice detections across clips
        • Build entity profile nodes (appearance, role, relationships)
        • Map <face_N> / <voice_N> IDs to consistent entity_ids
```

### Compute budget

| Video length | Frames sampled | 8B inference calls | Wall time (1× A100) |
|-------------|----------------|-------------------|---------------------|
| 2 min (Video-Holmes) | ~60 | ~15 batches + 1 semantic pass | ~45 sec |
| 30 min (M3-Bench robot) | ~360 | ~90 batches + 5 semantic passes | ~8 min |
| 2 hr (long-form) | ~1400 | ~350 batches + 15 semantic passes | ~30 min |

---

## 4. Benchmark Evaluation Adapters

Thin wrappers that map each benchmark's I/O format to the unified pipeline.

### Tier 1 — Direct mode adapters

```python
class VideoHolmesAdapter:
    """Video-Holmes: short suspense films, compositional reasoning."""
    def evaluate(self, video_path, question, options) -> dict:
        frames = sample_frames(video_path, n=32)
        return reason(question, options, video_context=frames,
                      vlm_fn=self.vlm, mode="direct")

class SIVBenchAdapter:
    """SIV-Bench: social interaction understanding + reasoning."""
    def evaluate(self, video_path, question, options,
                 subtitle_mode="origin") -> dict:
        frames = sample_frames(video_path, n=16)
        subtitle_text = load_subtitles(video_path, subtitle_mode)
        context = frames + [subtitle_text] if subtitle_text else frames
        return reason(question, options, video_context=context,
                      vlm_fn=self.vlm, mode="direct")
```

### Tier 2 — Retrieval mode adapters

```python
class VRBenchAdapter:
    """VRBench: long narrative video, multi-step reasoning."""
    def evaluate(self, video_path, question, options=None, graph=None) -> dict:
        if graph is None:
            graph = build_memory_graph(video_path)
        return reason(question, options, video_context=graph,
                      vlm_fn=self.vlm, mode="retrieval")

class LongVideoBenchAdapter:
    """LongVideoBench: long video + subtitle referring reasoning."""
    def evaluate(self, video_path, question, options, graph=None) -> dict:
        if graph is None:
            graph = build_memory_graph(video_path, include_subtitles=True)
        return reason(question, options, video_context=graph,
                      vlm_fn=self.vlm, mode="retrieval")

class CGBenchAdapter:
    """CG-Bench: clue-grounded long-video QA."""
    def evaluate(self, video_path, question, options=None, graph=None) -> dict:
        if graph is None:
            graph = build_memory_graph(video_path)
        return reason(question, options, video_context=graph,
                      vlm_fn=self.vlm, mode="retrieval")

class M3BenchAdapter:
    """M3-Bench: long-video entity-grounded QA."""
    def evaluate(self, video_path, question, graph=None) -> dict:
        if graph is None:
            graph = build_memory_graph(video_path)
        question = graph.back_translate(question)
        result = reason(question, None, video_context=graph,
                        vlm_fn=self.vlm, mode="retrieval")
        result.answer = graph.translate(result.answer)
        return result
```

### Shared interface

All adapters expose:
```python
def evaluate(self, video_path: str, question: str, **kwargs) -> dict
```
returning at minimum `{"answer": str, "reasoning": str, "mode": str}`.

---

## 5. Per-Benchmark Configuration Cheat Sheet

| Benchmark | Mode | `sample_frames` | Subtitles? | Entity tracking? | Special prompt |
|---|---|---|---|---|---|
| **Video-Holmes** | direct | 32 | no | no | Compositional deduction prompt |
| **SIV-Bench** | direct | 16 | yes (3 conditions) | no | Social cognition prompt; relationship types |
| **VRBench** | retrieval | N/A (graph) | no | no | Multi-step narrative reasoning prompt |
| **LongVideoBench** | retrieval | N/A (graph) | yes (interleaved) | no | Referring reasoning prompt |
| **CG-Bench** | retrieval | N/A (graph) | no | no | Clue-grounded prompt |
| **M3-Bench** | retrieval | N/A (graph) | yes (voice transcripts) | yes (face + voice) | Entity-aware prompt with `<face_N>/<voice_N>` |

---

## 6. Evaluation Plans

### 6.1 Self-Evolving System Benchmarks

| Benchmark | Focus | Why It Tests Our System |
|-----------|-------|------------------------|
| **Video-Holmes** | Deep causal/temporal/social reasoning (short films) | Reasoning depth over social dynamics |
| **MA-EgoQA** | Multi-agent social interaction, ToM, task coordination | Perspective-aware reasoning across multiple agents |
| **EgoLife** | Long-term ego-centric daily life | Long-range memory + social tracking at scale |
| **LongVidSearch** | Multi-hop evidence retrieval | Evidence chaining quality |

### 6.2 Metrics

| Metric | What It Measures | How |
|--------|-----------------|-----|
| **Answer accuracy** | Final answer correctness | Standard MCQ / open-ended eval |
| **Evidence grounding quality** | Are cited timestamps and facts correct? | Compare cited evidence against ground truth |
| **Multi-hop retrieval quality** | Does the system chain evidence across distant segments? | Count correct cross-segment links |
| **Social consistency** | Are social-state inferences consistent across time? | Check for contradictions in perspective threads |
| **Memory efficiency** | How compact is the memory graph relative to video length? | Nodes per minute of video; retrieval precision |
| **Tool call count** | How many frozen-model calls per question? | Count observer/reasoner invocations |
| **Skill reuse rate** | Are skills being reused across questions? | Unique skills / total skill invocations |
| **Long-range dependency handling** | Can the system answer questions requiring evidence from >5 min apart? | Stratify accuracy by temporal span |

### 6.3 Orchestrator Evaluation Commands

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

### 6.4 Expected Improvements

| Benchmark | Baseline (72B raw) | With orchestrator | Why |
|-----------|--------------------|--------------------|-----|
| Video-Holmes | ~65% acc | ~72-75% acc | Skills provide reasoning scaffolds for causal/temporal questions |
| M3-Bench | ~55% acc | ~65-70% acc | Memory graph handles long-range entity tracking |
| SIV-Bench | ~60% acc | ~68-72% acc | Entity relationship skills transfer well to social interaction |

### 6.5 Ablation Designs

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

## 7. Integration with Existing Components

| Existing component | How it connects |
|---|---|
| `rag/retrieval.py` → `MemoryStore` | `VideoMemoryGraph` wraps a `MemoryStore` internally |
| `rag/embedding/` → `TextEmbedder`, `MultimodalEmbedder` | Used by `VideoMemoryGraph` and observation pipeline |
| `data_structure/experience.py` → `Experience`, `Episode` | Each reasoning trace can be packaged as an `Episode` |
| `skill_agents/stage3_mvp/schemas.py` → `Skill`, `Protocol` | All visual skills use these schemas directly |
| `decision_agents/agent.py` → `VLMDecisionAgent` | Future `VideoQAAgent` subclass uses `select_skill` |

---

## 8. Implementation Order

| Phase | Files | Depends on | Priority |
|---|---|---|---|
| **Phase 1** | `reasoning.py` | VLM callable only | **Highest** — unlocks all 6 benchmarks |
| **Phase 2** | `adapters.py` (Tier 1: direct mode) | Phase 1 | **High** — Video-Holmes + SIV-Bench immediately usable |
| **Phase 3** | `video_memory.py` | `rag/retrieval.py`, `rag/embedding/` | **High** — unlocks 4 long-video benchmarks |
| **Phase 4** | `observe.py` | Phase 3, face/voice utils | **High** — needed by all retrieval-mode adapters |
| **Phase 5** | `adapters.py` (Tier 2) | Phase 3–4 | **High** |
| **Phase 6** | `skills.py` | Phase 1–5 | Lower — skill bank integration |
| **Phase 7** | `__init__.py`, integration tests | All above | Final |
