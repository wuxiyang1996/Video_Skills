# Benchmark Discussion: Memory Graphs & Multi-Hop Reasoning

> Comparison of **M3-Bench**, **Video-Holmes**, **VRBench**, **LongVideoBench**,
> and **CG-Bench** with respect to memory graph construction and multi-hop
> reasoning in long-video understanding.

---

## 1. Memory Graph Construction (M3-Bench)

M3-Bench ships prebuilt memory graphs as pickled `VideoGraph` objects. The
construction pipeline lives in `m3-agent/` and proceeds as follows:

### Pipeline

```
Long video
  │
  ├─ 1. Segment into ~30 s clips (ffmpeg)
  │
  ├─ 2. For each clip, run process_segment:
  │       ├─ process_voices  →  voice nodes (speaker embeddings)
  │       ├─ process_faces   →  image nodes (face embeddings)
  │       ├─ generate_memories (VLM)
  │       │     ├─ episodic: video_descriptions  (what happened)
  │       │     └─ semantic: high_level_conclusions (inferred facts)
  │       └─ process_memories  →  text nodes + edges to entity nodes
  │
  ├─ 3. refresh_equivalences()  →  union-find over face/voice IDs
  │
  └─ 4. pickle.dump(video_graph)
```

### VideoGraph data structure (`mmagent/videograph.py`)

| Component | Description |
|-----------|-------------|
| **nodes** | `node_id → Node` with type ∈ {`img`, `voice`, `episodic`, `semantic`}, embeddings, and metadata |
| **edges** | `(node_id1, node_id2) → weight`; bidirectional; same-type text↔text edges disallowed |
| **text_nodes** | Ordered list of episodic/semantic node IDs |
| **text_nodes_by_clip** | Index from clip ID → text node IDs |
| **character_mappings** | Built by `refresh_equivalences()` via union-find over entity embeddings |

### Key files

| File | Role |
|------|------|
| `m3-agent/m3_agent/memorization_memory_graphs.py` | Entry script — iterates clips, calls `process_segment`, pickles graph |
| `m3-agent/mmagent/videograph.py` | `VideoGraph` class |
| `m3-agent/mmagent/face_processing.py` | Face detection + node matching (InsightFace) |
| `m3-agent/mmagent/voice_processing.py` | Speaker diarization + node matching |
| `m3-agent/mmagent/memory_processing_qwen.py` | VLM-based memory generation + graph insertion |
| `m3-agent/mmagent/retrieve.py` | QA-time retrieval over the graph |
| `m3-agent/configs/memory_config.json` | Thresholds and hyperparameters |

### How raw video becomes a graph

1. **`process_video_clip`** reads each `.mp4`, yields base64-encoded full clip,
   frames at 5 fps, and 16 kHz WAV audio.
2. **`process_voices` / `process_faces`** create or merge voice and face nodes
   via embedding similarity against configurable thresholds.
3. **`generate_memories`** builds a multimodal prompt (frames + face crops
   tagged `<face_id>` + diarized text keyed by `<voice_id>`) and calls the VLM
   to produce episodic descriptions and semantic conclusions.
4. **`process_memories`** embeds each string, adds text nodes, parses entity
   references like `<face_3>`, and creates edges from text nodes to entity
   nodes. Semantic memories can reinforce or weaken existing nodes by embedding
   similarity.
5. **`refresh_equivalences()`** consolidates face/voice identities via
   union-find for downstream `translate` / `back_translate` in retrieval.

---

## 2. Multi-Hop Reasoning — Definition

> "Multi-hop reasoning" means the model must perform **multiple dependent
> reasoning steps** during the video reasoning process, where each step's
> conclusion is a necessary input to the next step. This is distinct from
> multi-evidence retrieval (gathering facts from scattered locations) —
> multi-hop is about the **depth of the reasoning chain**, not the breadth of
> evidence collection.
>
> ```
> Multi-evidence (broad):   Fact A + Fact B + Fact C  →  Answer
> Multi-hop (deep):         Fact A  →  Infer X  →  Use X + Fact B  →  Infer Y  →  Answer
> ```

---

## 3. Benchmark-by-Benchmark Analysis

### 3.1 VRBench — Long narrative video + explicit multi-step reasoning

**Paper:** [arXiv:2506.10857](https://arxiv.org/abs/2506.10857) (ICCV 2025)
**Source:** [github.com/OpenGVLab/VRBench](https://github.com/OpenGVLab/VRBench)

The closest benchmark to "long video + multi-step reasoning." Videos average
~1.6 hours across 960 long narrative videos (8 languages, 7 categories).

| Aspect | Detail |
|--------|--------|
| **Scale** | 960 videos, 8,243 QA pairs, **25,106 reasoning steps with timestamps** |
| **Multi-hop** | **Yes, explicit and annotated.** Each QA pair includes a full reasoning chain of temporally grounded intermediate steps. |
| **Reasoning types** | Event attribution, implicit inference, counting, hypothetical reasoning, information synopsis, event prediction, logical linkage |
| **Unique strength** | Not just final-answer evaluation — includes **progress-level scoring** that evaluates reasoning chain quality, so you can tell whether a model reached the right answer via correct intermediate steps or by shortcutting. |
| **Video length** | ~1.6 hours average — genuine long-form narrative |

VRBench is the strongest benchmark for evaluating whether a model can produce
correct **intermediate reasoning** (not just the final answer) over long video.
The temporally grounded step annotations make it possible to measure both
outcome accuracy and process faithfulness.

### 3.2 LongVideoBench — Long video + referring reasoning

**Paper:** [arXiv:2407.15754](https://arxiv.org/abs/2407.15754) (NeurIPS 2024)
**Source:** [longvideobench.github.io](https://longvideobench.github.io/)

A more classic long-video understanding benchmark focused on **referring
reasoning** — questions contain a referring query that points to specific video
contexts, and the model must first locate the relevant context, then reason
over it.

| Aspect | Detail |
|--------|--------|
| **Scale** | 3,763 videos, 6,678 MCQ questions, 17 fine-grained categories |
| **Multi-hop** | **Moderate.** The "referring" step (find the relevant context) + "reasoning" step (answer based on it) forms a 2-step chain. Deeper chains are not explicitly annotated. |
| **Unique strength** | Video-language **interleaved inputs** up to 1 hour; subtitles woven into context. Tests long-context memory pressure more than reasoning depth. |
| **Video length** | Up to ~1 hour |
| **Key finding** | Unlike many video benchmarks, performance genuinely improves when models process more frames — no "single-frame bias." |

LongVideoBench does not emphasize explicit reasoning chains like VRBench, but
it applies strong **long-context memory pressure**: the model must sift through
an hour of interleaved video + subtitle context to find and reason over the
right segment. This makes it a good complement — it tests the retrieval /
memory side more than the reasoning-chain side.

### 3.3 CG-Bench — Long video + clue retrieval + reasoning

**Paper:** [arXiv:2412.12075](https://arxiv.org/abs/2412.12075) (ICLR 2025)
**Source:** [cg-bench.github.io](https://cg-bench.github.io/leaderboard/)

Bridges the gap between memory retrieval and reasoning by requiring models to
not only answer questions but also **retrieve the key clues** that support the
answer.

| Aspect | Detail |
|--------|--------|
| **Scale** | 1,219 long videos, 12,129 QA pairs, 14 primary / 171 secondary / 638 tertiary categories |
| **Multi-hop** | **Moderate to high.** Questions span perception, reasoning, and hallucination types. The clue-grounded evaluation forces models to justify answers with retrieved evidence, which implicitly requires multi-step reasoning (find clue → interpret clue → answer). |
| **Unique strength** | **Clue-grounded evaluation** — white-box and black-box methods that check whether the model's answer is based on actual video understanding, not MCQ elimination strategies. |
| **Video length** | Long-form (benchmark is described as the largest for long video analysis) |

CG-Bench is the most relevant benchmark for evaluating the **memory retrieval
→ reasoning** pipeline that `memory_manage` implements. The clue-grounded
evaluation directly maps to the `[Search] → [Think] → [Answer]` loop: the
model must retrieve relevant clues (Search), interpret them (Think), and
produce a grounded answer (Answer).

### 3.4 M3-Bench — Long video + memory graph + multi-hop

**Paper:** [arXiv:2508.09736](https://arxiv.org/abs/2508.09736)

Already detailed in Section 1. From a multi-hop perspective:

| Aspect | Detail |
|--------|--------|
| **Scale** | 100 robot + 920 web videos with prebuilt memory graphs |
| **Multi-hop** | **Yes, explicit.** "Multi-hop Reasoning" is a named question type alongside "Multi-evidence Reasoning." |
| **Reasoning types** | Multi-hop reasoning, multi-evidence reasoning, + 3 other types |
| **Unique strength** | Prebuilt **entity-grounded memory graphs** — the only benchmark that ships a structured graph representation, not just raw video + QA. |
| **Video length** | Long-form (robot interactions, web videos) |

Multi-hop in M3-Bench typically involves 2–3 dependent retrieval + reasoning
steps: e.g., "You visited Ding Cha; what is the *next* bubble tea shop?"
requires identifying visited shops (step 1), ordering visits (step 2),
identifying the successor (step 3).

### 3.5 Video-Holmes — Short video + deep multi-step reasoning

**Paper:** [arXiv:2505.21374](https://arxiv.org/abs/2505.21374)

Targets complex reasoning over suspense short films (1–5 min). Not long-video,
but has the **deepest reasoning chains** among the benchmarks here.

| Aspect | Detail |
|--------|--------|
| **Scale** | 270 videos, 1,837 questions, 7 task types |
| **Multi-hop** | **Yes, in substance.** IMC, TCI, and TA tasks require 3+ dependent reasoning steps. |
| **Reasoning types** | SR, IMC, TCI, TA, MHR, PAR, CTI |
| **Unique strength** | Questions require **connecting clues scattered across segments** with deep causal/temporal chains — the reasoning itself is the hard part, not the retrieval. |
| **Video length** | 1–5 min (short) |

| Code | Task | Reasoning chain depth |
|------|------|-----------------------|
| SR | Social Relationship | 1–2 steps |
| **IMC** | **Intention and Motive Chaining** | **3+ steps**: action → intent → motive → behavior |
| **TCI** | **Temporal Causal Inference** | **3+ steps**: event A → consequence B → final outcome C |
| **TA** | **Timeline Analysis** | **3+ steps**: temporal cues → ordering → contradiction resolution |
| MHR | Multimodal Hint Reasoning | 2–3 steps |
| PAR | Pattern Recognition | 2 steps |
| CTI | Core Theme Identification | 2–3 steps |

---

## 4. Summary Table

| Benchmark | Video length | Multi-hop? | Chain depth | Primary challenge | Venue |
|-----------|-------------|-----------|-------------|-------------------|-------|
| **VRBench** | ~1.6 h avg | **Yes, annotated steps** | 3+ (with timestamps) | Long narrative + multi-step reasoning with process evaluation | ICCV 2025 |
| **LongVideoBench** | Up to 1 h | Moderate (2-step referring reasoning) | 2 | Long-context memory pressure + referring reasoning | NeurIPS 2024 |
| **CG-Bench** | Long-form | Moderate–High | 2–3 | Clue retrieval + grounded reasoning; anti-shortcut evaluation | ICLR 2025 |
| **M3-Bench** | Long-form | **Yes, explicit type** | 2–3 | Memory graph + retrieval + multi-hop over long video | — |
| **Video-Holmes** | 1–5 min | **Yes, in substance** | 3+ (IMC/TCI/TA) | Deep causal/temporal reasoning over short films | — |

### Two-axis view: reasoning depth vs. evidence breadth

```
                        Reasoning depth (multi-hop)
                        ────────────────────────────►
                    Low                              High
                ┌──────────────────┬──────────────────────────┐
         Low    │                  │  Video-Holmes             │
Evidence        │  (trivial)       │  (short video,            │
breadth         │                  │   deep causal chains)     │
(# sources /    ├──────────────────┼──────────────────────────┤
 video length)  │  LongVideoBench  │  VRBench                  │
         High   │  (long context,  │  (1.6 h + annotated       │
                │   refer+reason)  │   multi-step chains)      │
                │                  │                            │
                │  CG-Bench        │  M3-Bench                  │
                │  (clue retrieval │  (memory graph +            │
                │   + grounding)   │   retrieval + chains)      │
                └──────────────────┴──────────────────────────┘
```

---

## 5. Implications for the Unified Pipeline

The `memory_manage` module (see `memory_manage.md`) must handle both axes —
**memory/retrieval** (evidence breadth) and **reasoning chains** (reasoning
depth) — as independent, configurable dimensions.

### Where each benchmark pushes the pipeline

- **VRBench** is the most demanding overall: long video requires heavy memory,
  and the annotated reasoning steps demand deep `[Think]` chains. The
  progress-level scoring can directly evaluate whether `memory_manage`'s
  Think/Search/Answer loop produces faithful intermediate reasoning.

- **LongVideoBench** stresses the **retrieval** side: hour-long interleaved
  video+subtitle context means the `[Search]` mechanism must be precise under
  high memory load, but reasoning chains are shallow (refer → reason).

- **CG-Bench** bridges retrieval and reasoning: the clue-grounded evaluation
  maps directly to `[Search] → [Think] → [Answer]`, and the anti-shortcut
  design (white-box / black-box eval) tests whether the model genuinely uses
  retrieved clues rather than guessing.

- **M3-Bench** exercises the full entity-grounded memory pipeline: face/voice
  tracking → episodic/semantic storage → graph-based retrieval → multi-hop
  answer. The `[Search] → [Think] → [Search]` pattern is essential.

- **Video-Holmes** exercises pure reasoning depth: the graph is small (short
  video), but `[Think]` chains must be 3+ steps deep. IMC/TCI/TA questions
  produce multi-iteration `Think → Think → Search → Think → Answer` traces.

### Pipeline configuration by benchmark

| Parameter | VRBench | LongVideoBench | CG-Bench | M3-Bench | Video-Holmes |
|-----------|---------|----------------|----------|----------|--------------|
| Video length | ~1.6 h | Up to 1 h | Long | Long | 1–5 min |
| Clip interval | 30 s | 30 s | 30 s | 30 s | Full or 2–3 segments |
| Entity tracking | Heavy | Light | Moderate | Heavy (face + voice) | Moderate (face) |
| Semantic memory | Essential | Moderate | Moderate | Essential | Optional |
| `max_iterations` | 5–10 | 3–5 | 4–6 | 5–8 | 3–5 |
| Bottleneck | Both retrieval + reasoning | Retrieval precision | Clue grounding | Retrieval quality | Reasoning chain depth |
| Primary pattern | search → think → search → think → answer | search → think → answer | search → think → answer (grounded) | search → think → search → answer | think → think → think → answer |

### Recommended evaluation order

1. **Video-Holmes** — fast iteration (short videos), validates reasoning depth
2. **CG-Bench** — validates clue retrieval + grounding pipeline
3. **M3-Bench** — validates full entity-grounded memory pipeline
4. **LongVideoBench** — stress-tests retrieval under long-context pressure
5. **VRBench** — full evaluation of both axes with process-level scoring
