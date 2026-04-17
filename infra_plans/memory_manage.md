# Memory Management Module — Design Plan (v2)

> Sub-folder: `Video_Skills/memory_manage/`
>
> Goal: Provide a **reasoning-first** video QA pipeline with an **optional
> memory layer** that activates only when videos exceed the VLM's context
> budget. Expose capabilities as reusable visual skills within the COS-PLAY
> (`Video_Skills`) framework.

---

## 0. Key Insight — Reasoning vs. Context Management

After auditing all target benchmarks, the core finding is:

> **The reasoning capability needed is the same regardless of video length.
> What differs is only context management — whether the full video fits inside
> the model's context window.**

| | Short video | Long video |
|---|---|---|
| **Benchmarks** | Video-Holmes, SIV-Bench, MVBench, VideoHallu, VideoPhy2 | M3-Bench |
| **Video length** | 5 s – 5 min | 10 min – hours |
| **Fits in VLM context?** | Yes | No |
| **Needs explicit memory?** | No — feed raw frames/clip directly | Yes — must segment, index, and retrieve |
| **Core reasoning** | Chain-of-thought over visual evidence | Same chain-of-thought, but over *retrieved* evidence |

The memory graph, entity tracking, and segmentation pipeline from M3-Agent
exist because **the video does not fit in context**. They are a
compression/indexing strategy, not a fundamentally different reasoning paradigm.
If a VLM had an infinite context window, the memory layer would be unnecessary.

**Implication for architecture:** The module should be structured as:

1. **A reasoning core** (always active) — chain-of-thought with optional
   retrieval, applicable to every benchmark.
2. **A memory/retrieval layer** (conditionally active) — builds an indexed
   graph only when the video exceeds a configurable context budget. This is
   M3-Bench-specific engineering that other benchmarks skip entirely.

---

## 1. Benchmark Landscape

All datasets in the workspace, classified by what they actually test:

### Tier 1 — Direct reasoning (short video, no memory needed)

| Benchmark | Video len | QA format | Primary challenge | Notes |
|---|---|---|---|---|
| **Video-Holmes** | 1–5 min | MC (A/B/C/D) + `<think>` | Compositional deduction, connecting subtle cues | Suspense films; reasoning is hard, perception is easy |
| **SIV-Bench** | ~32 s avg | MC (A–N) | Social cognition: mental-state inference, counterfactual reasoning, relationship classification | 14 relationship types; 3 subtitle conditions (origin/+sub/-sub); 8.4k QAs |
| **MVBench** | 5–20 s | MC (3–4 options) | 20 fine-grained temporal tasks: action counting, sequence, prediction, localization, etc. | Short clips; questions are perception-heavy not reasoning-heavy |
| **VideoHallu** | gen. video | open-ended | Hallucination detection: alignment, physics, spatial-temporal consistency | AI-generated video (CogVideo, Kling, etc.); checks factual grounding |
| **VideoPhy2** | gen. video | scoring | Physics rule compliance: does the video obey real-world physics? | Human-annotated physics rules per video; evaluates VLM physical reasoning |
| **MMVU** | varies | open-ended | Expert knowledge QA: art, science, medicine — requires domain knowledge beyond the video | 1k validation; each Q links textbook + Wikipedia rationale |

### Tier 2 — Memory-dependent (long video)

| Benchmark | Video len | QA format | Primary challenge | Notes |
|---|---|---|---|---|
| **M3-Bench** | 10 min–hours | open-ended + `[Search]/[Answer]` | Entity tracking + factual recall across long timelines | Robot FPV + web videos; needs face/voice grounding; VideoGraph memory |

### What this means

- **6 out of 7 benchmarks need zero memory management.** They need strong
  reasoning, good perception, and the ability to produce chain-of-thought.
- **Only M3-Bench** needs the full memory pipeline (segmentation, entity
  detection, graph construction, retrieval).
- The reasoning core (`[Think]` → `[Answer]`, possibly with `[Search]`) is
  universal. The memory layer is an optional add-on.

---

## 2. Module Layout (revised)

```
Video_Skills/memory_manage/
├── __init__.py            # public API re-exports
├── reasoning.py           # CORE: Think / Search / Answer loop (all benchmarks)
├── video_memory.py        # OPTIONAL: VideoMemoryGraph (long-video only)
├── observe.py             # OPTIONAL: segment → observe → store (long-video only)
├── skills.py              # Skill definitions (universal + long-video)
└── adapters.py            # benchmark-specific evaluation adapters
```

The key change: `reasoning.py` is the primary module. `video_memory.py` and
`observe.py` are *optional* extensions that activate only when the video exceeds
the context budget.

---

## 3. `reasoning.py` — Reasoning Core (universal)

The single module that all benchmarks share.

### Core function

```python
def reason(
    question: str,
    options: Optional[List[str]],
    video_context: Union[str, List[str], "VideoMemoryGraph"],
    vlm_fn: Callable,
    max_iterations: int = 5,
    mode: str = "auto",  # "direct" | "retrieval" | "auto"
) -> ReasoningResult:
```

### Two execution paths

**Direct mode** (short video — Video-Holmes, SIV-Bench, MVBench, etc.):

`video_context` is raw frames/transcript fed directly to the VLM. No graph,
no retrieval. The loop is just `[Think]` → `[Think]` → `[Answer]` — pure
chain-of-thought reasoning.

```
VLM receives: [video frames] + [question + options]
         ├─ [Think] reason about what is observed
         ├─ [Think] connect cues / infer mental states / apply physics
         └─ [Answer] final answer
```

**Retrieval mode** (long video — M3-Bench):

`video_context` is a `VideoMemoryGraph`. The VLM cannot see the full video, so
it issues `[Search]` queries to retrieve relevant memories.

```
VLM receives: [question] + [memory context so far]
         ├─ [Think] what do I need to know?
         ├─ [Search] "who was in the kitchen at 3:00?"
         │     → system retrieves from graph, injects results
         ├─ [Think] connect retrieved evidence
         ├─ [Search] "what did face_0 say about the cup?"
         │     → system retrieves, injects
         └─ [Answer] final answer
```

**Auto mode** (default): checks video duration against a configurable
threshold (e.g. 5 min). Below threshold → direct. Above → retrieval.

### Iteration protocol

Each iteration the VLM produces **exactly one** tagged output:

| Tag | Meaning | System action |
|---|---|---|
| `[Think] <reflection>` | Internal reasoning step | Append to chain-of-thought; continue |
| `[Search] <query>` | Request information from memory | Run `graph.search(query)`, inject results; continue. **Only available in retrieval mode.** |
| `[Answer] <answer>` | Final answer | Stop; return answer + full reasoning chain |

### Output

```python
@dataclass
class ReasoningResult:
    answer: str
    thinking_chain: List[str]       # ordered [Think] / [Search] steps
    retrieved_contexts: List[str]   # memories from [Search] (empty in direct mode)
    n_iterations: int
    mode_used: str                  # "direct" | "retrieval"
    confidence: Optional[float]
```

### Design notes

- `[Think]` maps to Video-Holmes's `<think>` tag and SIV-Bench's reasoning
  chain, making traces directly usable for reasoning evaluation.
- In **direct mode**, the prompt tells the VLM that `[Search]` is unavailable,
  so it reasons purely from what it sees. This is the natural path for 6/7
  benchmarks.
- In **retrieval mode**, `[Search]` delegates to `VideoMemoryGraph.search()`
  with entity translation. This is the M3-Bench path.
- `max_iterations` prevents runaway loops. A hard-stop fallback forces
  `[Answer]` after the budget is exhausted.

---

## 4. `video_memory.py` — VideoMemoryGraph (long-video only)

Activated only when the video exceeds the context budget. Not used by
Video-Holmes, SIV-Bench, MVBench, VideoHallu, VideoPhy2, or MMVU.

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
| Retrieve by query | `search(query, top_k, clip_filter) -> List[MemoryNode]` — delegates to internal `MemoryStore` |
| Retrieve by entity | `get_by_entity(entity_id) -> List[MemoryNode]` |
| Translate entity IDs ↔ names | `translate(text) / back_translate(text)` |
| Serialize | `save(path) / load(path)` |

### Integration point

- Wraps a `MemoryStore` (from `rag/retrieval.py`) for the embedding index.
- Accepts either a `TextEmbedder` or `MultimodalEmbedder` at construction time.
- Entity nodes store face/voice embeddings matched via cosine similarity with
  configurable thresholds (same approach as `mmagent/face_processing.py` and
  `mmagent/voice_processing.py`).

---

## 5. `observe.py` — Observation Pipeline (long-video only)

Converts raw long video into structured `MemoryNode` entries. **Skipped
entirely for short-video benchmarks** — they feed the video directly to the
VLM.

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

### When is this used?

Only when `mode="retrieval"` — i.e., only for M3-Bench or any future
long-video benchmark where the full video cannot be fed to the VLM.

---

## 6. `skills.py` — Visual Skill Definitions

Skills are split into **universal** (all benchmarks) and **long-video-only**.

### Universal skills (always available)

| skill_id | Name | What it does |
|---|---|---|
| `reason_chain` | Chain-of-Thought Reasoning | The core `[Think]/[Answer]` loop. Every benchmark uses this. |
| `temporal_reason` | Temporal Ordering | Reason about event ordering using visual evidence or timestamps |
| `causal_reason` | Causal / Physical Inference | Infer cause-effect; applies to Video-Holmes deduction, VideoPhy2 physics, SIV-Bench counterfactuals |
| `social_reason` | Social State Inference | Infer unobservable mental states (emotions, intentions, attitudes, relationships) from observable cues. SIV-Bench SSR. |

### Long-video-only skills (activated when memory layer is active)

| skill_id | Name | What it does |
|---|---|---|
| `observe_segment` | Observe Video Segment | Run observation pipeline on one clip; produce episodic descriptions |
| `detect_entities` | Detect & Track Entities | Face/voice detection and entity graph updates |
| `build_episodic` | Build Episodic Memory | Store timestamped observations with entity links |
| `build_semantic` | Build Semantic Memory | Aggregate episodic memories into high-level conclusions |
| `search_memory` | Retrieve from Memory | Embed a query and return top-k matches from the graph |

### Example skill definition

```python
Skill(
    skill_id="reason_chain",
    name="Chain-of-Thought Reasoning",
    strategic_description=(
        "Iterative Think/Answer reasoning over video evidence. "
        "Works in direct mode (raw video in context) or retrieval mode "
        "(search memory graph). Universal across all benchmarks."
    ),
    tags=["REASONING", "UNIVERSAL"],
    protocol=Protocol(
        preconditions=["question_received=true"],
        steps=[
            "Determine mode: direct (video in context) or retrieval (memory graph)",
            "Generate [Think] step: reason about available evidence",
            "If evidence insufficient and retrieval available: [Search] for more",
            "Repeat Think/Search until confident",
            "Generate [Answer] with final response",
        ],
        success_criteria=["answer_produced=true"],
        abort_criteria=["max_iterations_exceeded"],
        expected_duration=5,
    ),
    contract=SkillEffectsContract(
        skill_id="reason_chain",
        eff_add={"answer_produced", "reasoning_chain_generated"},
        eff_event={"reasoning_completed"},
    ),
)
```

### Composition patterns

**Direct mode** (Video-Holmes, SIV-Bench, MVBench, VideoHallu, VideoPhy2, MMVU):
```
[raw video + question] → reason_chain
                           ├─ [Think] (perceive + reason)
                           ├─ [Think] (connect / infer)
                           └─ [Answer]
```

**Retrieval mode** (M3-Bench):
```
[offline] for each clip:
              detect_entities → observe_segment → build_episodic
          build_semantic (aggregate)

[online]  [question] → reason_chain
                         ├─ [Think]
                         ├─ [Search] → search_memory (+ entity translate)
                         ├─ [Think]
                         └─ [Answer]
```

---

## 7. `adapters.py` — Benchmark Evaluation Adapters

Thin wrappers that map each benchmark's I/O format to the unified pipeline.
All adapters share the same `evaluate()` interface.

### Tier 1 — Direct mode adapters

```python
class VideoHolmesAdapter:
    """Video-Holmes: short suspense films, compositional reasoning.
    Input:  video path, question, options (A/B/C/D)
    Output: answer letter, <think>...</think><answer>X</answer>
    """
    def evaluate(self, video_path, question, options) -> dict:
        frames = sample_frames(video_path, n=32)
        return reason(question, options, video_context=frames,
                      vlm_fn=self.vlm, mode="direct")

class SIVBenchAdapter:
    """SIV-Bench: social interaction understanding + reasoning.
    Input:  video path, question, options (A–N), subtitle_mode
    Output: answer letter, reasoning chain
    """
    def evaluate(self, video_path, question, options,
                 subtitle_mode="origin") -> dict:
        frames = sample_frames(video_path, n=16)
        subtitle_text = load_subtitles(video_path, subtitle_mode)
        context = frames + [subtitle_text] if subtitle_text else frames
        return reason(question, options, video_context=context,
                      vlm_fn=self.vlm, mode="direct")

class MVBenchAdapter:
    """MVBench: 20 fine-grained temporal perception tasks.
    Input:  video path, question, candidates
    Output: answer, reasoning chain
    """
    def evaluate(self, video_path, question, candidates) -> dict:
        frames = sample_frames(video_path, n=16)
        return reason(question, candidates, video_context=frames,
                      vlm_fn=self.vlm, mode="direct")

class VideoHalluAdapter:
    """VideoHallu: hallucination detection on AI-generated video.
    Input:  video path, question, generation prompt
    Output: answer, reasoning chain
    """
    def evaluate(self, video_path, question, prompt=None) -> dict:
        frames = sample_frames(video_path, n=16)
        ctx = frames + [f"Generation prompt: {prompt}"] if prompt else frames
        return reason(question, None, video_context=ctx,
                      vlm_fn=self.vlm, mode="direct")

class VideoPhy2Adapter:
    """VideoPhy2: physics rule compliance scoring.
    Input:  video path, physics rules, caption
    Output: compliance assessment, reasoning chain
    """
    def evaluate(self, video_path, rules, caption=None) -> dict:
        frames = sample_frames(video_path, n=16)
        question = f"Does this video obey the following physics rules? {rules}"
        ctx = frames + [f"Video description: {caption}"] if caption else frames
        return reason(question, None, video_context=ctx,
                      vlm_fn=self.vlm, mode="direct")

class MMVUAdapter:
    """MMVU: expert-knowledge video QA (art, science, medicine).
    Input:  video path/url, question, knowledge hints
    Output: answer, reasoning chain
    """
    def evaluate(self, video_path, question, knowledge=None) -> dict:
        frames = sample_frames(video_path, n=16)
        ctx = frames + [f"Domain knowledge: {knowledge}"] if knowledge else frames
        return reason(question, None, video_context=ctx,
                      vlm_fn=self.vlm, mode="direct")
```

### Tier 2 — Retrieval mode adapter

```python
class M3BenchAdapter:
    """M3-Bench: long-video entity-grounded QA.
    Input:  video path, question, prebuilt memory graph (optional)
    Output: answer text, retrieval trace
    """
    def evaluate(self, video_path, question, graph=None) -> dict:
        if graph is None:
            graph = build_memory_graph(video_path)  # observe.py pipeline
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

## 8. Integration with Existing `Video_Skills` Components

| Existing component | How `memory_manage` connects |
|---|---|
| `rag/retrieval.py` → `MemoryStore` | `VideoMemoryGraph` wraps a `MemoryStore` internally (retrieval mode only) |
| `rag/embedding/` → `TextEmbedder`, `MultimodalEmbedder` | Used by `VideoMemoryGraph` and `observe.py` (retrieval mode only) |
| `data_structure/experience.py` → `Experience`, `Episode` | Each reasoning trace can be packaged as an `Episode` for replay/training |
| `skill_agents/stage3_mvp/schemas.py` → `Skill`, `Protocol`, `SkillEffectsContract` | All visual skills defined in `skills.py` use these schemas directly |
| `decision_agents/agent.py` → `VLMDecisionAgent` | A future `VideoQAAgent` subclass would use `select_skill` to orchestrate skills |

---

## 9. Per-Benchmark Configuration Cheat Sheet

| Benchmark | Mode | `sample_frames` | Subtitles? | Entity tracking? | Special prompt |
|---|---|---|---|---|---|
| **Video-Holmes** | direct | 32 | no | no | Compositional deduction prompt |
| **SIV-Bench** | direct | 16 | yes (3 conditions) | no | Social cognition prompt; relationship types |
| **MVBench** | direct | 16 | no | no | Temporal perception prompt; task-type-specific |
| **VideoHallu** | direct | 16 | no | no | Hallucination detection prompt; include gen. prompt |
| **VideoPhy2** | direct | 16 | no | no | Physics rules prompt; include annotated rules |
| **MMVU** | direct | 16 | no | no | Expert knowledge prompt; include knowledge links |
| **M3-Bench** | retrieval | N/A (graph) | yes (voice transcripts) | yes (face + voice) | Entity-aware prompt with `<face_N>/<voice_N>` |

---

## 10. Implementation Order (revised)

| Phase | Files | Depends on | Priority |
|---|---|---|---|
| **Phase 1** | `reasoning.py` | VLM callable only | **Highest** — unlocks all 7 benchmarks |
| **Phase 2** | `adapters.py` (Tier 1: direct mode) | Phase 1 | **High** — 6 benchmarks immediately usable |
| **Phase 3** | `video_memory.py` | `rag/retrieval.py`, `rag/embedding/` | Medium — M3-Bench only |
| **Phase 4** | `observe.py` | Phase 3, face/voice utils | Medium — M3-Bench only |
| **Phase 5** | `adapters.py` (Tier 2: M3BenchAdapter) | Phase 3–4 | Medium |
| **Phase 6** | `skills.py` | Phase 1–5, `skill_agents/stage3_mvp/schemas.py` | Lower — skill bank integration |
| **Phase 7** | `__init__.py`, integration tests | All above | Final |

The critical change from v1: **Phase 1 is now the reasoning core, not the
memory graph.** This means 6 out of 7 benchmarks become usable after just
Phase 1 + Phase 2, without any memory infrastructure.
