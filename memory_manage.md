# Memory Management Module — Design Plan

> Sub-folder: `Video_Skills/memory_manage/`
>
> Goal: Unify the memory-centric pipeline from **M3-Bench** (entity-grounded
> episodic/semantic graph) with the reasoning pipeline from **Video-Holmes**
> (compositional chain-of-thought), and expose both as reusable **visual skills**
> within the COS-PLAY (`Video_Skills`) framework.

---

## 1. Module Layout

```
Video_Skills/memory_manage/
├── __init__.py            # public API re-exports
├── video_memory.py        # VideoMemoryGraph — core data structure
├── observe.py             # segment → observe → store pipeline
├── reasoning.py           # Think / Search / Answer reasoning loop
├── skills.py              # Skill definitions (Perception, Memory, Reasoning)
└── adapters.py            # benchmark-specific evaluation adapters
```

---

## 2. `video_memory.py` — VideoMemoryGraph

The central data structure that bridges M3-Bench's `VideoGraph` concept with
`MemoryStore` from `rag/retrieval.py`.

### Data model

```python
@dataclass
class MemoryNode:
    node_id: str                # e.g. "episodic_003", "semantic_007"
    node_type: str              # "episodic" | "semantic" | "entity"
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
| Reinforce / decay semantics | `reinforce(node_id, delta)` — bump confidence; called when new evidence supports an existing conclusion |
| Retrieve by query | `search(query, top_k, clip_filter) -> List[MemoryNode]` — delegates to internal `MemoryStore` |
| Retrieve by entity | `get_by_entity(entity_id) -> List[MemoryNode]` |
| Translate entity IDs ↔ names | `translate(text) / back_translate(text)` — mirrors M3-Agent's character mapping |
| Serialize | `save(path) / load(path)` — pickle or JSON |

### Integration point

- Wraps a `MemoryStore` (from `rag/retrieval.py`) for the embedding index.
- Accepts either a `TextEmbedder` or `MultimodalEmbedder` at construction time.
- Entity nodes store face/voice embeddings and are matched via cosine similarity
  with configurable thresholds (same approach as `mmagent/face_processing.py`
  and `mmagent/voice_processing.py`).

---

## 3. `observe.py` — Observation Pipeline

Converts raw video into structured `MemoryNode` entries.

### Pipeline stages

```
Video file
  │
  ├─ 1. segment_video(path, interval_sec) -> List[Clip]
  │     Uniform temporal segmentation (configurable interval).
  │
  ├─ 2. extract_entities(clip) -> List[EntityDetection]
  │     Face detection + embedding (InsightFace/ArcFace).
  │     Voice diarization + embedding (ERes2NetV2 or Gemini).
  │     Each detection carries a tentative entity_id.
  │
  ├─ 3. generate_observations(clip, entities, vlm) -> ObservationResult
  │     Send clip frames + entity crops + voice transcripts to a VLM.
  │     Prompt mirrors M3-Agent's entity-ID-aware generation:
  │       - episodic descriptions (what happened, who, when)
  │       - semantic conclusions (high-level inferences)
  │     Returns structured text with <face_N> / <voice_N> references.
  │
  └─ 4. store_observations(observations, graph: VideoMemoryGraph)
        Embed texts, insert episodic + semantic nodes, link entities.
```

### Key design decisions

| Decision | Rationale |
|---|---|
| Entity IDs are assigned at detection time and referenced in VLM prompts | Follows M3-Agent's grounding strategy — the VLM learns to use `<face_0>` tokens |
| Observations are generated per-clip, not per-frame | Balances cost vs. coverage; interval is configurable |
| Semantic nodes can be generated in a second pass | Allows a "summarize across clips" step for higher-level conclusions |

### Depends on

- `VideoMemoryGraph` from `video_memory.py`
- VLM inference (abstracted behind a callable `vlm_fn(prompt, images) -> str`)
- Face/voice processing utilities (can import from `m3-agent/mmagent/` or
  reimplement lightweight versions)

---

## 4. `reasoning.py` — Think / Search / Answer Loop

Implements the ReAct-style iterative reasoning used at **question-answering time**.

### Core function

```python
def reason(
    question: str,
    options: Optional[List[str]],
    graph: VideoMemoryGraph,
    vlm_fn: Callable,
    max_iterations: int = 5,
) -> ReasoningResult:
```

### Iteration protocol

Each iteration the VLM produces **exactly one** of three tagged outputs:

| Tag | Meaning | System action |
|---|---|---|
| `[Think] <reflection>` | Internal reasoning step | Append to chain-of-thought; continue |
| `[Search] <query>` | Request information from memory | Run `graph.search(query)`, inject results as context; continue |
| `[Answer] <answer>` | Final answer | Stop; return answer + full reasoning chain |

### Output

```python
@dataclass
class ReasoningResult:
    answer: str
    thinking_chain: List[str]       # ordered [Think] / [Search] steps
    retrieved_contexts: List[str]   # memories returned by [Search] steps
    n_iterations: int
    confidence: Optional[float]
```

### Design notes

- The `[Think]` tag maps to Video-Holmes's `<think>` tag, making the reasoning
  chain directly usable for reasoning evaluation.
- The `[Search]` tag maps to M3-Agent's retrieval mechanism
  (`retrieve_from_videograph`), but through the unified `VideoMemoryGraph.search`.
- For **short videos** (Video-Holmes), the graph may be small or even empty if
  the VLM can answer from raw frames alone. In that case the loop naturally
  converges to `[Think] → [Answer]` without any `[Search]` steps.
- For **long videos** (M3-Bench), multiple `[Search]` steps gather scattered
  evidence before converging on `[Answer]`.
- `max_iterations` prevents runaway loops. A hard-stop fallback forces
  `[Answer]` after the budget is exhausted.

---

## 5. `skills.py` — Visual Skill Definitions

Each capability is packaged as a `Skill` (from
`skill_agents/stage3_mvp/schemas.py`) so it can be managed by the skill bank
and selected by `VLMDecisionAgent`.

### Skill taxonomy

#### A. Perception Skills

| skill_id | Name | What it does |
|---|---|---|
| `observe_segment` | Observe Video Segment | Run the observation pipeline on a single clip; produce episodic descriptions |
| `detect_entities` | Detect & Track Entities | Run face/voice detection and update entity nodes in the graph |

#### B. Memory Skills

| skill_id | Name | What it does |
|---|---|---|
| `build_episodic` | Build Episodic Memory | Store timestamped observations with entity links into the graph |
| `build_semantic` | Build Semantic Memory | Aggregate episodic memories into higher-level conclusions |
| `search_memory` | Retrieve from Memory | Embed a query and return top-k matches from the graph |

#### C. Reasoning Skills

| skill_id | Name | What it does |
|---|---|---|
| `reason_chain` | Chain-of-Thought Reasoning | Run the Think/Search/Answer loop from `reasoning.py` |
| `temporal_reason` | Temporal Ordering | Reason about event ordering using clip timestamps |
| `causal_reason` | Causal Inference | Identify cause-effect from retrieved evidence |

### Example skill definition

```python
Skill(
    skill_id="search_memory",
    name="Retrieve from Memory",
    strategic_description=(
        "Embed a natural-language query and retrieve the most relevant "
        "stored observations via cosine similarity."
    ),
    tags=["RETRIEVE", "MEMORY", "REASONING"],
    protocol=Protocol(
        preconditions=["memory_populated=true", "question_received=true"],
        steps=[
            "Embed the query using the text embedder",
            "Compute cosine similarity against the memory store",
            "Return top-k results with clip_id and score",
            "Format results as context for reasoning",
        ],
        success_criteria=["knowledge_retrieved=true", "results_non_empty"],
        abort_criteria=["memory_empty", "embedding_failed"],
        expected_duration=1,
    ),
    contract=SkillEffectsContract(
        skill_id="search_memory",
        eff_add={"knowledge_retrieved"},
        eff_event={"memory_queried"},
    ),
)
```

### Composition patterns

**Video-Holmes (short video, reasoning-heavy):**
```
observe_segment → build_episodic → reason_chain
                                      ├─ [Think]
                                      ├─ [Search] → search_memory
                                      ├─ [Think]
                                      └─ [Answer]
```

**M3-Bench (long video, memory-heavy):**
```
for each clip:
    detect_entities → observe_segment → build_episodic
                                              │
build_semantic  (aggregate across clips)      │
         │                                    │
         └────────────────────────────────────┘
                        │
                  reason_chain
                    ├─ [Think]
                    ├─ [Search] → search_memory (+ entity translate)
                    ├─ [Search] → search_memory (different query)
                    ├─ [Think]
                    └─ [Answer]
```

---

## 6. `adapters.py` — Benchmark Evaluation Adapters

Thin wrappers that map each benchmark's I/O format to the unified pipeline.

### `VideoHolmesAdapter`

```python
class VideoHolmesAdapter:
    """Adapter for Video-Holmes evaluation.

    Input:  video path, question, options (A/B/C/D)
    Output: answer letter, <think> block, <answer> block
    """

    def evaluate(self, video_path, question, options) -> dict:
        # 1. Build graph from video (lightweight — few clips)
        # 2. Run reason_chain(question, options, graph)
        # 3. Format output as <think>...</think><answer>X</answer>
        ...
```

### `M3BenchAdapter`

```python
class M3BenchAdapter:
    """Adapter for M3-Bench evaluation.

    Input:  video path, question, prebuilt memory graph (optional)
    Output: answer text, retrieval trace
    """

    def evaluate(self, video_path, question, graph=None) -> dict:
        # 1. Load or build full VideoMemoryGraph (entity-aware)
        # 2. Translate question entities via back_translate
        # 3. Run reason_chain with [Search] enabled
        # 4. Translate answer entities back to names
        ...
```

### Shared interface

Both adapters expose:
```python
def evaluate(self, video_path: str, question: str, **kwargs) -> dict
```
returning at minimum `{"answer": str, "reasoning": str}`.

---

## 7. Integration with Existing `Video_Skills` Components

| Existing component | How `memory_manage` connects |
|---|---|
| `rag/retrieval.py` → `MemoryStore` | `VideoMemoryGraph` wraps a `MemoryStore` internally for embedding search |
| `rag/embedding/` → `TextEmbedder`, `MultimodalEmbedder` | Passed into `VideoMemoryGraph` at construction; used by `observe.py` for embedding observations |
| `data_structure/experience.py` → `Experience`, `Episode` | Each reasoning trace can be packaged as an `Episode` of `Experience` objects for replay/training |
| `skill_agents/stage3_mvp/schemas.py` → `Skill`, `Protocol`, `SkillEffectsContract` | All visual skills defined in `skills.py` use these schemas directly |
| `decision_agents/agent.py` → `VLMDecisionAgent` | A future `VideoQAAgent` subclass would use `select_skill` to orchestrate the visual skills |

---

## 8. Implementation Order

| Phase | Files | Depends on |
|---|---|---|
| **Phase 1** | `video_memory.py` | `rag/retrieval.py`, `rag/embedding/` |
| **Phase 2** | `observe.py` | Phase 1, VLM callable, face/voice utils |
| **Phase 3** | `reasoning.py` | Phase 1 |
| **Phase 4** | `skills.py` | Phase 1–3, `skill_agents/stage3_mvp/schemas.py` |
| **Phase 5** | `adapters.py` | Phase 1–4 |
| **Phase 6** | `__init__.py`, integration tests | All above |
