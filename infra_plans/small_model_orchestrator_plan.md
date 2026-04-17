# Small-Model Orchestrator — Using Qwen3-VL-8B to Drive Memory, Skills, and Large-VLM Prompting

> **Location:** `Video_Skills/small_model_orchestrator/`
>
> **Core idea:** A lightweight VLM (Qwen3-VL-8B, ~16 GB VRAM) runs as an
> always-on orchestrator that **(1)** builds and maintains a
> `VideoMemoryGraph`, **(2)** discovers and curates reusable skills in a
> `SkillBank`, and **(3)** composes rich, context-aware prompts so that a
> large VLM (Qwen3-VL-72B, InternVL-78B, etc.) can answer complex video
> questions in a single forward pass — without ever touching raw video
> frames itself.

---

## 0. Motivation

| Problem | How the small model solves it |
|---------|-------------------------------|
| Large VLMs are expensive to run on every frame | Qwen3-VL-8B processes all frames; the 72B model only sees curated text + a few key frames |
| Long videos overflow context windows | The 8B model distills hours of video into a compact memory graph and skill bank |
| Reasoning from scratch produces hallucinations | Pre-extracted skills give the 72B model step-by-step scaffolds grounded in visual evidence |
| No reuse across videos | The skill bank accumulates transferable reasoning patterns across datasets |
| Prompt engineering is manual and brittle | The 8B model dynamically composes prompts with retrieved memory + skills tailored to each question |

### Analogy to COS-PLAY

In COS-PLAY, a **Qwen3-8B** decision agent retrieves skills from a learned
skill bank to guide game actions. Here, a **Qwen3-VL-8B** orchestrator
retrieves visual skills and memory to compose prompts for a **72B VLM** — the
small model plays the role of "executive assistant" while the large model
is the "expert" that only speaks when given well-prepared context.

---

## 1. Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────┐
│                         OFFLINE PHASE                                │
│                  (Qwen3-VL-8B, runs once per video)                  │
│                                                                      │
│   Video ──► Frame Sampler ──► Qwen3-VL-8B ──┬──► VideoMemoryGraph   │
│                                              │      (episodic +      │
│                                              │       semantic +      │
│                                              │       entity nodes)   │
│                                              │                       │
│                                              └──► SkillBank          │
│                                                    (protocols +      │
│                                                     contracts)       │
└──────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────────┐
│                        ONLINE PHASE                                  │
│                 (Qwen3-VL-8B + large VLM 72B)                        │
│                                                                      │
│   Question ──► Qwen3-VL-8B ──┬──► Memory retrieval (graph.search)   │
│                               │                                      │
│                               ├──► Skill retrieval (RAG)             │
│                               │                                      │
│                               ├──► Keyframe selection                │
│                               │    (pick 3-5 most relevant frames)   │
│                               │                                      │
│                               └──► Prompt Composer                   │
│                                       │                              │
│                                       ▼                              │
│                              ┌─────────────────┐                     │
│                              │  Composed Prompt │                     │
│                              │  ┌─────────────┐│                     │
│                              │  │ System      ││                     │
│                              │  │ Skills (x3) ││                     │
│                              │  │ Memory (x5) ││                     │
│                              │  │ Frames (x5) ││                     │
│                              │  │ Question    ││                     │
│                              │  └─────────────┘│                     │
│                              └────────┬────────┘                     │
│                                       │                              │
│                                       ▼                              │
│                              Large VLM (72B)                         │
│                              ──► Answer                              │
└──────────────────────────────────────────────────────────────────────┘
```

### Why Two Phases?

- **Offline:** The 8B model is cheap enough to process every frame of every
  video. It builds persistent artifacts (memory graph + skill bank) that
  amortize across all future questions about those videos.
- **Online:** When a question arrives, the 8B model quickly retrieves
  relevant context and assembles a prompt. The 72B model is called once
  with a compact, high-quality prompt rather than raw video.

---

## 2. Module Layout

```
Video_Skills/small_model_orchestrator/
├── __init__.py                   # Public API re-exports
├── config.py                     # Model paths, thresholds, prompt templates
├── orchestrator.py               # SmallModelOrchestrator — top-level controller
├── memory_builder.py             # Video → VideoMemoryGraph (offline)
├── skill_crafter.py              # VideoMemoryGraph → SkillBank (offline)
├── prompt_composer.py            # Question → composed prompt for large VLM (online)
├── keyframe_selector.py          # Pick the N most question-relevant frames
├── question_analyzer.py          # Decompose question into retrieval signals
├── self_reflection.py            # 8B model verifies/refines the 72B answer
└── adapters/
    ├── __init__.py
    ├── video_holmes.py           # Video-Holmes benchmark adapter
    ├── m3_bench.py               # M3-Bench benchmark adapter
    └── siv_bench.py              # SIV-Bench adapter (caregiver-recipient, etc.)
```

---

## 3. Memory Builder (`memory_builder.py`)

The 8B model watches the video and constructs a `VideoMemoryGraph` (from
`memory_manage/video_memory.py`). This module owns the **observe** pipeline.

### Pipeline

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
  │     • Prompt: "Given these observations, what high-level conclusions
  │       can be drawn about [entity/relationship/causality]?"
  │     • Each semantic node carries: text, linked entity_ids, confidence
  │
  └─ 5. Entity resolution
        • Merge face/voice detections across clips
        • Build entity profile nodes (appearance, role, relationships)
        • Map <face_N> / <voice_N> IDs to consistent entity_ids
```

### Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Batch multi-image input | Qwen3-VL natively handles multi-image; batching 4-8 frames amortizes KV-cache overhead |
| Two-pass (episodic → semantic) | Episodic nodes capture "what happened"; semantic nodes capture "what it means" — both are needed for different question types |
| Adaptive merging | Low-delta regions (e.g., static camera) collapse into single nodes, keeping the graph compact for long videos |
| Entity-aware prompting | Following M3-Agent's grounding strategy: the VLM generates text with `<face_0>` tokens that link to entity nodes |

### Integration

- **Input:** Raw video file (mp4), optional subtitles (srt)
- **Output:** `VideoMemoryGraph` (serializable to JSON/pickle)
- **Depends on:** `memory_manage/video_memory.py`, `rag/embedding/` for
  embeddings, VLM callable wrapping Qwen3-VL-8B via vLLM or transformers

### Compute Budget

| Video length | Frames sampled | 8B inference calls | Wall time (1× A100) |
|-------------|----------------|-------------------|---------------------|
| 2 min (Video-Holmes) | ~60 | ~15 batches + 1 semantic pass | ~45 sec |
| 30 min (M3-Bench robot) | ~360 | ~90 batches + 5 semantic passes | ~8 min |
| 2 hr (long-form) | ~1400 | ~350 batches + 15 semantic passes | ~30 min |

---

## 4. Skill Crafter (`skill_crafter.py`)

After the memory graph is built, the 8B model analyzes the graph to
discover reusable reasoning skills. This follows the same
three-stage pipeline as `video_skill_pipeline_design.md` but is driven
entirely by the small model and operates on the memory graph rather than
raw frames.

### Pipeline

```
VideoMemoryGraph
  │
  ├─ 1. Temporal segmentation
  │     • Boundary detection via predicate delta between consecutive
  │       episodic nodes (reuses COS-PLAY's ScoredBoundary scoring)
  │     • Intention tagging: Qwen3-VL-8B classifies each segment
  │       with a [TAG] from the video intention taxonomy:
  │       OBSERVE | INTERACT | NAVIGATE | COMMUNICATE | MANIPULATE |
  │       INVESTIGATE | REACT | WAIT | APPROACH | RETREAT | DELIVER | RECEIVE
  │
  ├─ 2. Contract extraction  (Qwen3-VL-8B)
  │     • For each segment: compute eff_add / eff_del from predicate
  │       changes across the segment boundary
  │     • Aggregate across similar segments (same intention + high
  │       predicate overlap) to build robust contracts
  │
  └─ 3. Protocol generation  (Qwen3-VL-8B)
        • For each skill cluster, synthesize a step-by-step Protocol
        • Prompt: "You are creating a reusable reasoning skill from
          these video segments. Write a protocol that a reasoning agent
          can follow to analyze similar scenes in new videos."
        • Output: preconditions, steps[], success_criteria, abort_criteria
```

### Skill Quality Control (8B as Judge)

The 8B model also runs quality evaluation on crafted skills, scoring each
on six dimensions (reusing `skill_agents/skill_evaluation/`):

| Dimension | 8B Model Check |
|-----------|----------------|
| **Coherence** | "Does this skill's protocol make logical sense for its intention tag?" |
| **Discriminability** | "Is this skill distinct from existing skills in the bank?" |
| **Composability** | "Can this skill chain with other skills in a reasoning plan?" |
| **Generalization** | "Would this skill apply to videos beyond the source?" |
| **Utility** | "Would following this protocol help answer a question?" |
| **Granularity** | "Is this skill at the right level of abstraction?" |

Skills scoring below threshold on any dimension are sent back for
refinement or merged with higher-quality neighbors.

### Output

- `skill_bank.jsonl` — COS-PLAY-compatible `Skill` objects
- Each skill carries: `skill_id`, `name`, `strategic_description`, `tags`,
  `protocol`, `contract`, `sub_episodes` (evidence pointers), `n_instances`

### Cross-Video Skill Accumulation

When processing multiple videos, the skill crafter:
1. Loads the existing skill bank
2. Attempts to merge new segments into existing skills (by embedding
   similarity + contract overlap)
3. Creates new skills only when no existing skill covers the pattern
4. Retires skills that lose all supporting evidence

This mirrors COS-PLAY's `bank_maintenance` (split/merge/refine/retire)
but adapted for video reasoning skills rather than game strategies.

---

## 5. Question Analyzer (`question_analyzer.py`)

When a question arrives at inference time, the 8B model first decomposes
it into structured retrieval signals before touching the memory graph or
skill bank.

### Decomposition

```python
@dataclass
class QuestionAnalysis:
    question_type: str          # e.g., "SR", "TCI", "MHR", "PAR", "CTI"
    target_entities: List[str]  # mentioned entities to look up
    temporal_scope: str         # "full_video" | "segment" | "point_in_time"
    temporal_hint: Optional[Tuple[float, float]]  # approximate time range if mentioned
    required_evidence: List[str]  # what kind of memory nodes are needed
    reasoning_type: str         # "causal" | "temporal" | "relational" | "descriptive"
    retrieval_queries: List[str]  # generated sub-queries for memory search
    skill_tags: List[str]       # suggested intention tags for skill retrieval
    difficulty_estimate: str    # "simple" (skip 72B?) | "moderate" | "hard"
```

### 8B Model Prompt for Analysis

```
You are analyzing a question about a video. Decompose it into retrieval
signals. Return JSON:

Question: "What caused the woman to become frightened in the elevator?"

{
  "question_type": "TCI",
  "target_entities": ["woman", "elevator"],
  "temporal_scope": "segment",
  "reasoning_type": "causal",
  "retrieval_queries": [
    "woman in elevator emotional state",
    "threatening event near elevator",
    "cause of fear reaction"
  ],
  "skill_tags": ["REACT", "INVESTIGATE", "OBSERVE"]
}
```

### Adaptive Routing

Based on `difficulty_estimate`, the orchestrator can skip the 72B model
entirely for simple factual queries:

| Difficulty | Route |
|-----------|-------|
| **simple** | 8B answers directly from memory graph (e.g., "How many people are in the video?") |
| **moderate** | 8B composes prompt, 72B answers (e.g., "What is the relationship between X and Y?") |
| **hard** | 8B composes prompt with extra skills + more memory, 72B answers with chain-of-thought (e.g., "What would happen if X had not entered the room?") |

This routing saves 72B inference cost on ~30-40% of Video-Holmes questions
that are factual or descriptive.

---

## 6. Keyframe Selector (`keyframe_selector.py`)

Even though the 72B model primarily reasons over text (memory + skills),
a small number of visual frames dramatically improve grounding. The 8B
model selects the most question-relevant frames.

### Selection Strategy

```python
def select_keyframes(
    question_analysis: QuestionAnalysis,
    memory_graph: VideoMemoryGraph,
    retrieved_memories: List[MemoryNode],
    max_frames: int = 5,
) -> List[KeyFrame]:
```

1. **Temporal anchoring:** frames from timestamps referenced by retrieved
   memory nodes
2. **Entity anchoring:** frames containing the target entities from
   question analysis
3. **Diversity sampling:** ensure selected frames span different segments
   (avoid redundancy)
4. **Saliency scoring:** the 8B model scores candidate frames by how
   relevant they are to the question (single batch VLM call with
   question + candidate frames)

### Frame Budget

| Scenario | Max frames | Rationale |
|----------|-----------|-----------|
| Simple factual | 1-2 | Just verification |
| Relationship / causal | 3-5 | Need multiple moments |
| Temporal ordering | 5-8 | Need representative moments from each segment |
| Full video summary | 8-12 | Coverage |

The frame budget is constrained by the 72B model's context window. At
~1000 tokens per frame (Qwen3-VL image encoding), 5 frames ≈ 5K tokens
of visual context — leaving ample room for text context.

---

## 7. Prompt Composer (`prompt_composer.py`)

The heart of the online phase. The 8B model assembles a structured prompt
for the 72B model from all retrieved context.

### Prompt Template

```
╔══════════════════════════════════════════════════════════╗
║  SYSTEM                                                  ║
║  You are an expert video analyst. Use the provided        ║
║  reasoning skills and memory context to answer the        ║
║  question. Follow the skill protocols step by step.       ║
║  Ground every claim in the provided visual evidence.      ║
╠══════════════════════════════════════════════════════════╣
║  REASONING SKILLS  (from skill bank, top-k retrieved)    ║
║                                                          ║
║  --- Skill 1: {skill.name} ---                           ║
║  When to apply: {skill.protocol.preconditions}           ║
║  Steps:                                                  ║
║    1. {step_1}                                           ║
║    2. {step_2}                                           ║
║    ...                                                   ║
║  Expect: {skill.contract.eff_add}                        ║
║                                                          ║
║  --- Skill 2: ... ---                                    ║
║  --- Skill 3: ... ---                                    ║
╠══════════════════════════════════════════════════════════╣
║  MEMORY CONTEXT  (from VideoMemoryGraph, top-k search)   ║
║                                                          ║
║  [Episodic] 00:12-00:24: Woman enters elevator, notices  ║
║    a person with a plastic bag on their head in the       ║
║    corridor. Entity: <face_0>=woman, <face_1>=masked     ║
║                                                          ║
║  [Episodic] 00:24-00:45: Elevator door closes. Woman     ║
║    shows signs of distress. <face_1> does not follow.     ║
║                                                          ║
║  [Semantic] The masked figure appears to be a threat      ║
║    that triggers fear in the woman. Confidence: 0.87      ║
║                                                          ║
║  [Entity] <face_0>: woman, ~30yo, appears in segments     ║
║    1-4, primary character                                 ║
║  [Entity] <face_1>: masked person, appears in segment 1   ║
║    only, threatening presence                             ║
╠══════════════════════════════════════════════════════════╣
║  VISUAL FRAMES  (keyframes selected by 8B model)         ║
║                                                          ║
║  [Frame @ 00:15] <image_1>                               ║
║  [Frame @ 00:30] <image_2>                               ║
║  [Frame @ 00:42] <image_3>                               ║
╠══════════════════════════════════════════════════════════╣
║  QUESTION                                                 ║
║                                                          ║
║  What caused the woman to become frightened?              ║
║  Options: (A) ... (B) ... (C) ... (D) ...                ║
╠══════════════════════════════════════════════════════════╣
║  INSTRUCTIONS                                             ║
║                                                          ║
║  Think step by step. Reference specific memory entries    ║
║  and frames. Output:                                     ║
║  <think>your reasoning chain</think>                     ║
║  <answer>X</answer>                                      ║
╚══════════════════════════════════════════════════════════╝
```

### Token Budget Management

The 8B model actively manages the total prompt size to fit within the
72B model's effective context:

| Section | Token budget | Priority |
|---------|-------------|----------|
| System + instructions | ~300 | Fixed |
| Skills (top-k) | ~800 (k=3) | High — reasoning scaffold |
| Memory context | ~1200 (top-5 nodes) | High — factual grounding |
| Visual frames | ~5000 (5 frames) | Medium — visual grounding |
| Question + options | ~200 | Fixed |
| **Total** | **~7500** | Fits well within 32K context |

When the budget is tight (many relevant memories, complex skills), the
8B model summarizes memory nodes and truncates skill protocols to fit.

### Prompt Variants by Question Type

| Question type | Skill emphasis | Memory emphasis | Frame emphasis |
|---------------|---------------|-----------------|----------------|
| Social Relationship (SR) | Character analysis, interaction tracking | Entity nodes, interaction episodes | Frames with multiple people |
| Temporal Causal (TCI) | Causal reasoning, temporal tracking | Chronological episodic chain | Frames at cause and effect points |
| Hidden Reasoning (MHR) | Threat detection, pattern recognition | Semantic inferences, anomaly nodes | Frames with subtle cues |
| Temporal Arrangement (TA) | Temporal ordering | Timestamped episodic nodes | Spread across timeline |
| Core Theme (CTI) | Theme extraction, pattern recognition | Semantic summary nodes | Representative frames |
| M3-Bench Memory QA | Memory recall, dialogue management | Long-range episodic + entity | Dialogue-aligned frames |

---

## 8. Self-Reflection Loop (`self_reflection.py`)

After the 72B model produces an answer, the 8B model optionally verifies
it against the memory graph. This catches hallucinations cheaply.

### Verification Steps

```python
def verify_answer(
    answer: str,
    reasoning_chain: str,
    memory_graph: VideoMemoryGraph,
    question_analysis: QuestionAnalysis,
) -> VerificationResult:
```

1. **Claim extraction:** 8B model extracts factual claims from the 72B
   answer (e.g., "the woman was frightened", "the masked person entered
   the elevator")
2. **Memory grounding:** For each claim, search the memory graph. Flag
   claims with no supporting memory node (potential hallucination)
3. **Temporal consistency:** Check that the reasoning chain respects the
   temporal order of events in the memory graph
4. **Entity consistency:** Verify entity references match the entity
   nodes (e.g., "the man" should correspond to a tracked entity)

### Feedback Actions

| Verification outcome | Action |
|---------------------|--------|
| All claims grounded | Accept answer |
| 1-2 ungrounded claims | Re-prompt 72B with targeted corrections: "Note: no evidence for X in the video" |
| Major contradiction | Re-compose prompt with additional memory context and re-run 72B |
| Temporal error | Re-compose with explicit timeline and re-run 72B |

The self-reflection loop runs at most **2 iterations** to bound cost.

---

## 9. Orchestrator (`orchestrator.py`)

The top-level controller that ties everything together.

### Class: `SmallModelOrchestrator`

```python
class SmallModelOrchestrator:
    """Uses Qwen3-VL-8B to manage memory, craft skills, and compose
    prompts for a large VLM."""

    def __init__(
        self,
        small_model: str = "Qwen/Qwen3-VL-8B",
        large_model: str = "Qwen/Qwen3-VL-72B-Instruct",
        embedder: str = "Qwen/Qwen3-Embedding-0.6B",
        mm_embedder: str = "Qwen/Qwen3-VL-Embedding-2B",
        skill_bank_path: Optional[str] = None,
        max_keyframes: int = 5,
        top_k_skills: int = 3,
        top_k_memories: int = 5,
        enable_self_reflection: bool = True,
        device_small: str = "cuda:0",     # 8B fits on 1 GPU (16GB+)
        device_large: str = "cuda:1",     # 72B needs multi-GPU
    ):
        ...

    # --- Offline API ---

    def process_video(
        self, video_path: str, subtitle_path: Optional[str] = None
    ) -> VideoMemoryGraph:
        """Build memory graph from video using the 8B model."""
        ...

    def craft_skills(
        self, graph: VideoMemoryGraph, existing_bank: Optional[str] = None
    ) -> SkillBank:
        """Discover and curate skills from a memory graph."""
        ...

    # --- Online API ---

    def answer(
        self, question: str, options: Optional[List[str]] = None,
        video_path: Optional[str] = None,
        graph: Optional[VideoMemoryGraph] = None,
    ) -> AnswerResult:
        """Full pipeline: analyze question → retrieve → compose → call 72B → verify."""
        ...

    def compose_prompt(
        self, question_analysis: QuestionAnalysis,
        graph: VideoMemoryGraph, skill_bank: SkillBank,
    ) -> str:
        """Compose a prompt for the large VLM (exposed for debugging/customization)."""
        ...
```

### End-to-End Flow

```python
from small_model_orchestrator import SmallModelOrchestrator

orch = SmallModelOrchestrator(
    small_model="Qwen/Qwen3-VL-8B",
    large_model="Qwen/Qwen3-VL-72B-Instruct",
)

# Offline: process a video
graph = orch.process_video("videos/elevator_scene.mp4")
bank = orch.craft_skills(graph)

# Online: answer a question
result = orch.answer(
    question="What caused the woman to become frightened?",
    options=["A) The elevator stopped", "B) She saw a masked figure", ...],
    graph=graph,
)

print(result.answer)        # "B"
print(result.reasoning)     # Full chain-of-thought
print(result.confidence)    # 0.92
print(result.grounded)      # True (all claims verified)
print(result.small_model_calls)  # 4 (analyze + retrieve + compose + verify)
print(result.large_model_calls)  # 1 (single inference)
```

---

## 10. Model Choices and Alternatives

The design targets **Qwen3-VL-8B** as the default small model but is
model-agnostic. Any VLM that supports multi-image input and structured
output can serve as the orchestrator.

### Recommended Small Models

| Model | Params | VRAM | Strength | Limitation |
|-------|--------|------|----------|------------|
| **Qwen3-VL-8B** (default) | 8B | ~16 GB | Strong structured output, native multi-image, good at following protocols | Weaker on subtle social reasoning |
| Qwen2.5-VL-7B | 7B | ~14 GB | Proven in video_skill_pipeline_design.md, available locally | Older generation, weaker instruction following |
| InternVL3-8B | 8B | ~16 GB | Strong visual grounding | Different API/tokenizer |
| Phi-4-multimodal | 5.6B | ~12 GB | Very efficient, good reasoning | Smaller capacity for long contexts |
| LLaVA-Video-7B | 7B | ~14 GB | Video-native (temporal tokens) | Less structured output |

### Recommended Large Models

| Model | Params | VRAM | When to use |
|-------|--------|------|-------------|
| **Qwen3-VL-72B-Instruct** (default) | 72B | ~144 GB (4×A100) | Best accuracy, same family as 8B |
| InternVL2.5-78B | 78B | ~156 GB | Alternative, strong on M3-Bench |
| Qwen3-VL-32B | 32B | ~64 GB (2×A100) | Budget option, still strong |
| GPT-5.4 / Claude 4.6 | API | API | When local GPU is unavailable |

### Serving Configuration

```yaml
# Small model: single GPU, always running
small_model:
  model: Qwen/Qwen3-VL-8B
  backend: vllm
  tensor_parallel: 1
  gpu_memory_utilization: 0.85
  max_model_len: 32768

# Large model: multi-GPU, on-demand
large_model:
  model: Qwen/Qwen3-VL-72B-Instruct
  backend: vllm
  tensor_parallel: 4
  gpu_memory_utilization: 0.90
  max_model_len: 32768
```

---

## 11. Inference Latency Analysis

A multi-model pipeline raises fair concerns about speed. This section
gives concrete latency estimates based on published vLLM benchmarks
(Qwen3-VL-8B: ~120 tok/s on A100; 72B: ~25-35 tok/s on 4×A100) and
practical optimization strategies.

### 11.1 Offline Phase — One-Time Cost, Not Per-Question

The offline phase processes each video **once**. The resulting memory
graph and skill bank are persisted to disk and reused across all
future questions. This cost is fully amortized.

| Step | What runs | Video-Holmes (2 min) | M3-Bench robot (30 min) |
|------|----------|---------------------|------------------------|
| Frame sampling | CPU (decord) | ~0.5 s | ~2 s |
| Batch VLM grounding | 8B, ~15 batches of 4 frames | ~20 s | ~3 min |
| Semantic distillation | 8B, 1 summarization pass | ~4 s | ~30 s |
| Entity resolution | Embedding cosine ops | ~0.5 s | ~3 s |
| Skill crafting | 8B, segmentation + protocols | ~12 s | ~1.5 min |
| **Total offline** | | **~37 s** | **~5.5 min** |

This runs once. After that, answering 100 questions about the same video
costs zero additional offline time.

### 11.2 Online Phase — Per-Question Latency

This is the user-facing latency. The critical path is:

```
Question ──► 8B calls (parallel where possible) ──► 72B call ──► Answer
             ~3-5 s                                  ~15-25 s
```

**Detailed breakdown (2 min Video-Holmes video, 1 question):**

| Step | Model | Input tokens | Output tokens | Latency |
|------|-------|-------------|---------------|---------|
| Question analysis | 8B | ~300 | ~150 | ~1.2 s |
| Memory retrieval | Embedding search (CPU) | — | — | ~50 ms |
| Skill retrieval | Embedding search (CPU) | — | — | ~50 ms |
| Keyframe selection | 8B, 1 batch | ~2K (frames + Q) | ~100 | ~1.5 s |
| Prompt composition | CPU string ops | — | — | ~10 ms |
| **72B inference** | 72B on 4×A100 | **~7.5K** | **~400** | **~18 s** |
| Self-reflection (opt.) | 8B | ~800 | ~200 | ~2 s |
| **Total per question** | | | | **~23 s** |

### 11.3 Comparison: Orchestrator vs Raw 72B

| Approach | 72B input tokens | 72B TTFT | 72B generation | Total |
|----------|-----------------|---------|---------------|-------|
| 72B on all 60 frames (raw) | ~60K | ~20-30 s | ~15-20 s | **40-50 s** |
| 72B on 15 sampled frames | ~15K | ~8-12 s | ~12-18 s | **22-30 s** |
| **Orchestrator (8B + 72B)** | **~7.5K** | **~4-6 s** | **~12-16 s** | **~20-25 s** |
| Orchestrator (simple Q, 8B only) | 0 | — | — | **~2-3 s** |

**Key insight:** The 72B model's time-to-first-token (TTFT) scales
roughly linearly with input length for vision tokens. Cutting input from
60K to 7.5K saves ~15-25 seconds of prefill time. The 8B overhead
(~3-5 s) is smaller than the TTFT savings.

For **long videos** (30 min M3-Bench), the difference is more dramatic:

| Approach | 72B input | Total per Q |
|----------|----------|-------------|
| 72B on 360 frames | ~360K (exceeds context) | **Impossible** |
| 72B on 30 sampled frames | ~30K | **35-50 s** |
| **Orchestrator (8B + 72B)** | ~7.5K (same — memory is text) | **~23 s** |

The orchestrator's online latency is **constant** regardless of video
length because the memory graph compresses the video into a fixed-size
text representation.

### 11.4 Optimization Strategies

**Already built into the design:**

| Optimization | Effect | Section |
|---|---|---|
| Difficulty-based routing | ~30-40% of questions answered by 8B alone (2-3 s) | §5 Question Analyzer |
| Compact text-only prompts | 72B sees 7.5K tokens instead of 60K+ | §7 Prompt Composer |
| Pre-computed embeddings | Retrieval is CPU cosine search, ~50 ms | §3 Memory Builder |
| Offline amortization | Video processing cost is zero at query time | §11.1 above |

**Additional optimizations if latency is critical:**

| Optimization | Latency reduction | Trade-off |
|---|---|---|
| **Parallel 8B calls** — run question analysis + keyframe selection simultaneously on the same vLLM instance via batching | 8B time: 3-5 s → ~2 s | Minor: need async orchestration |
| **Skip self-reflection** for high-confidence answers (8B confidence > 0.9) | Save ~2 s on ~60% of questions | Slight accuracy drop on hard questions |
| **Use Qwen3-VL-32B** instead of 72B | 72B time 18 s → ~8-10 s | ~2-4% accuracy drop |
| **Speculative decoding** on vLLM (8B drafts, 72B verifies) | 72B generation 12-16 s → ~6-8 s | Requires same-family models (Qwen3 ✓) |
| **KV-cache reuse** — for multiple questions on same video, the system prompt + skills + memory section is shared | 72B TTFT: 4-6 s → ~1-2 s for Q2+ | Requires vLLM prefix caching (supported) |
| **Quantization** (AWQ/GPTQ) — run 72B in 4-bit on 2×A100 | Same speed, half the GPUs | ~1-2% accuracy drop |

### 11.5 Realistic Deployment Scenarios

| Scenario | Config | Per-Q latency | Hardware |
|----------|--------|---------------|----------|
| **Research eval** (accuracy first) | 8B + 72B + reflection | ~23 s | 1 + 4 A100s |
| **Fast eval** (balanced) | 8B + 72B, no reflection, parallel 8B | ~18 s | 1 + 4 A100s |
| **Budget** | 8B + 32B, no reflection | ~10-12 s | 1 + 2 A100s |
| **API** | 8B local + GPT-5.4 API | ~8-15 s (network dependent) | 1 A100 + API |
| **8B only** (simple Qs) | 8B with routing | ~2-3 s | 1 A100 |

### 11.6 Throughput (Batch Processing)

For benchmark evaluation where you process hundreds of questions, **batch
throughput** matters more than single-query latency:

- vLLM supports continuous batching — multiple questions share GPU time
- The 8B model can process 8-16 questions simultaneously (small model,
  fits in memory alongside large batch)
- The 72B model can batch 2-4 prompts at ~7.5K tokens each within 32K
  context budget

**Estimated throughput (4×A100 for 72B + 1×A100 for 8B):**

| Batch size | Per-question amortized | Questions/hour |
|-----------|----------------------|----------------|
| 1 (sequential) | ~23 s | ~156 |
| 4 (concurrent) | ~8-10 s | ~360-450 |
| 8 (concurrent, 8B batched) | ~6-8 s | ~450-600 |

For a full Video-Holmes eval (270 videos × ~10 Qs = 2700 questions):
- Offline: 270 × 37 s = ~2.8 hours
- Online at batch-8: 2700 / 500 = ~5.4 hours
- **Total: ~8 hours** (vs ~18-24 hours for raw 72B on all frames)

---

## 12. Integration with Existing Components

| Existing component | How the orchestrator connects |
|-------------------|-------------------------------|
| `memory_manage/video_memory.py` → `VideoMemoryGraph` | `memory_builder.py` produces the graph; `prompt_composer.py` reads from it |
| `memory_manage/reasoning.py` → Think/Search/Answer | The 72B model's chain-of-thought follows the same `[Think]/[Search]/[Answer]` protocol; `[Search]` queries are answered by the 8B model's memory retrieval |
| `memory_manage/skills.py` → Skill definitions | `skill_crafter.py` outputs the same `Skill` schema; memory/perception/reasoning skills are all representable |
| `rag/retrieval.py` → `MemoryStore` | `VideoMemoryGraph` wraps `MemoryStore` for embedding search; skill bank retrieval uses `SkillQueryEngine` |
| `rag/embedding/` → embedders | Both `TextEmbedder` (0.6B) and `MultimodalEmbedder` (2B) are used for retrieval |
| `skill_agents/stage3_mvp/schemas.py` → `Skill`, `Protocol`, `SkillEffectsContract` | `skill_crafter.py` outputs skills in this exact schema |
| `skill_agents/skill_bank/bank.py` → `SkillBankMVP` | The crafted skill bank is stored as `SkillBankMVP`-compatible JSONL |
| `skill_agents/skill_evaluation/` → LLM judge | `skill_crafter.py` reuses evaluation dimensions for quality control |
| `decision_agents/agent_helper.py` → `select_skill_from_bank()` | Prompt composer uses the same RAG scoring (relevance + applicability + pass_rate) |
| `dataset_examples/video_skill_pipeline_design.md` | The offline pipeline here extends that design with memory-graph-driven skill discovery |
| `data_structure/experience.py` → `Experience`, `Episode` | Each Q&A trace can be packaged as an `Episode` for replay/analysis |

---

## 13. Benchmark Evaluation Plan

### Video-Holmes (270 films, complex reasoning)

```bash
# Step 1: Build memory graphs for all videos (8B model, offline)
python -m small_model_orchestrator.run_offline \
    --dataset video_holmes \
    --output output/vh_graphs/ \
    --build_skills

# Step 2: Answer all questions (8B + 72B, online)
python -m small_model_orchestrator.run_eval \
    --dataset video_holmes \
    --graphs output/vh_graphs/ \
    --skill_bank output/vh_skill_bank.jsonl \
    --large_model Qwen/Qwen3-VL-72B-Instruct \
    --output output/vh_results.json
```

### M3-Bench (100 robot + 920 web, memory-heavy)

```bash
python -m small_model_orchestrator.run_offline \
    --dataset m3_bench \
    --output output/m3_graphs/

python -m small_model_orchestrator.run_eval \
    --dataset m3_bench \
    --graphs output/m3_graphs/ \
    --large_model Qwen/Qwen3-VL-72B-Instruct
```

### SIV-Bench (social interaction, caregiver-recipient)

```bash
python -m small_model_orchestrator.run_offline \
    --dataset siv_bench \
    --subset caregiver-recipient \
    --output output/siv_graphs/

python -m small_model_orchestrator.run_eval \
    --dataset siv_bench \
    --graphs output/siv_graphs/ \
    --large_model Qwen/Qwen3-VL-72B-Instruct
```

### Expected Improvements

| Benchmark | Baseline (72B raw) | With orchestrator | Why |
|-----------|--------------------|--------------------|-----|
| Video-Holmes | ~65% acc | ~72-75% acc | Skills provide reasoning scaffolds for causal/temporal questions |
| M3-Bench | ~55% acc | ~65-70% acc | Memory graph handles long-range entity tracking that raw frames miss |
| SIV-Bench | ~60% acc | ~68-72% acc | Entity relationship skills transfer well to social interaction |

### Cost Comparison

| Approach | 72B calls per question | Total tokens per question | Relative cost |
|----------|----------------------|--------------------------|---------------|
| 72B raw (all frames) | 1 | ~50K (frames dominate) | 1.0× |
| 72B raw (sampled frames) | 1 | ~15K | 0.3× |
| **Orchestrator (8B + 72B)** | 1 | ~7.5K (72B) + ~2K (8B) | **0.18×** |
| Orchestrator (simple Q, 8B only) | 0 | ~2K (8B only) | **0.02×** |

---

## 14. Implementation Order

| Phase | Files | Depends on | Effort |
|-------|-------|------------|--------|
| **Phase 0** | `config.py` | — | 1 day |
| **Phase 1** | `memory_builder.py` | `memory_manage/video_memory.py`, VLM callable | 3-4 days |
| **Phase 2** | `question_analyzer.py` | VLM callable | 1-2 days |
| **Phase 3** | `keyframe_selector.py` | Phase 1, Phase 2 | 1-2 days |
| **Phase 4** | `skill_crafter.py` | Phase 1, `skill_agents/stage3_mvp/schemas.py` | 3-4 days |
| **Phase 5** | `prompt_composer.py` | Phase 1-4 | 2-3 days |
| **Phase 6** | `self_reflection.py` | Phase 1, Phase 5 | 1-2 days |
| **Phase 7** | `orchestrator.py` | Phase 1-6 | 2-3 days |
| **Phase 8** | `adapters/` (Video-Holmes, M3-Bench, SIV-Bench) | Phase 7 | 2-3 days |
| **Phase 9** | Evaluation runs + ablations | Phase 8 | 3-5 days |
| **Total** | | | **~20-28 days** |

### Phase 1 Milestone (Memory-Only Baseline)

After Phase 1 + 5 + 7, we can run a memory-only version (no skills) to
get a baseline: the 8B model builds the memory graph, composes a prompt
with retrieved memories + keyframes, and the 72B model answers. This
should already outperform raw-frame baselines on long videos.

### Phase 4 Milestone (Full Pipeline)

After Phase 4, the skill bank is online. This is where the biggest
accuracy gains are expected, especially on reasoning-heavy benchmarks
(Video-Holmes causal/temporal questions).

---

## 15. Ablation Design

| Variant | Memory | Skills | Keyframes | Self-reflect | Tests |
|---------|--------|--------|-----------|-------------|-------|
| **Full** | Yes | Yes | Yes | Yes | Main result |
| No skills | Yes | No | Yes | Yes | Isolates skill contribution |
| No memory | No | Yes | Yes (random) | No | Isolates memory contribution |
| No keyframes | Yes | Yes | No (text only) | Yes | Isolates visual grounding |
| No reflection | Yes | Yes | Yes | No | Isolates verification value |
| 8B only | Yes | Yes | Yes | Yes | Difficulty routing sends all to 8B |
| 72B raw | No | No | All frames | No | Baseline: large model on raw video |
| 72B sampled | No | No | Uniform sample | No | Baseline: large model on sampled frames |

---

## 16. Future Extensions

### A. Online Skill Learning

The current design builds skills offline. A natural extension is **online
skill learning**: when the 72B model discovers a novel reasoning pattern
(identified by the 8B model's self-reflection), the 8B model packages it
as a new skill and adds it to the bank. This creates a COS-PLAY-style
co-evolution loop between the small and large models.

### B. Multi-Turn Dialogue

For interactive video QA (e.g., "Tell me more about what happened next"),
the 8B model maintains a conversation memory alongside the video memory
graph, allowing the prompt composer to include dialogue history.

### C. Real-Time Video Processing

For streaming video applications, the 8B model runs the observe pipeline
in real time (1 frame / second), continuously updating the memory graph.
When a question arrives, the graph is already populated.

### D. LoRA Fine-Tuning the 8B Orchestrator

The 8B model's orchestration capabilities (question analysis, skill
crafting, prompt composition) can be improved by fine-tuning with LoRA
on curated examples — similar to COS-PLAY's multi-LoRA approach but
with orchestration-specific adapters:

| Adapter | Task |
|---------|------|
| `memory_builder` | Video frame → structured observation JSON |
| `skill_crafter` | Segment cluster → Protocol + Contract |
| `question_analyzer` | Question → QuestionAnalysis JSON |
| `prompt_composer` | Context bundle → optimized prompt |
| `self_reflector` | Answer + memory → verification judgement |

### E. Hierarchical Multi-Model Cascade

For extremely long videos (10+ hours), add a third tier:

```
Qwen3-VL-3B (frame-level triage, ~6 GB)
    → Qwen3-VL-8B (segment-level reasoning, ~16 GB)
        → Qwen3-VL-72B (question answering, ~144 GB)
```

The 3B model handles initial frame filtering and scene detection,
the 8B model handles memory/skill construction, and the 72B model
handles final reasoning.
