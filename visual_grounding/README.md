# `visual_grounding/` — unified grounding for short and long videos

This package implements the **visual grounding layer** described in
`Video_Skills/infra_plans/`. Its one job is to take a video (optionally
with subtitles) and produce a **structured, evidence-linked social
representation** that both the 8B controller and the frozen reasoner can
consume.

The design obeys one key insight from
[`video_benchmarks_grounding.md`](../infra_plans/01_grounding/video_benchmarks_grounding.md) §0:

> Short and long videos use the **same** grounding schema. What differs
> is whether grounded state stays in-context (short) or is persisted in
> a hierarchical retrieval index (long).

Short videos → [`DirectContext`](#directcontext-short-videos).
Long videos → [`SocialVideoGraph`](#socialvideograph-long-videos).

---

## 1. Quick start

```python
from visual_grounding import build_grounded_context

# Short video (auto-dispatches to direct mode)
ctx = build_grounded_context(
    "clip.mp4",
    mode="auto",
    vlm_fn=my_vlm_callable,          # optional — see §6
)
print(ctx.as_reasoner_text())        # feed to [Think]/[Answer] loop

# Long video (auto-dispatches to retrieval mode)
graph = build_grounded_context(
    "movie.mp4",
    subtitle_path="movie.srt",
    mode="auto",
    vlm_fn=my_vlm_callable,
    embedder=my_text_embedder,       # optional — see §5
)
for node, score in graph.search("who left the room first?"):
    print(score, node.node_type, node.text)
```

Six benchmark-specific presets mirror §5 of the plan:

```python
from visual_grounding import (
    build_for_video_holmes,      # short, direct, no subs
    build_for_siv_bench,         # short, direct, subtitle-aware social
    build_for_vrbench,           # long,  retrieval, event/narrative
    build_for_long_video_bench,  # long,  retrieval, + subtitles
    build_for_cg_bench,          # long,  retrieval, clue/evidence
    build_for_m3_bench,          # long,  retrieval, entity + voice tracking
)
```

---

## 2. Module layout

| File | Role | Infra-plan anchor |
|---|---|---|
| [`schemas.py`](schemas.py) | Data contracts: `GroundedWindow`, `GroundingNode`, `DirectContext`, `Entity`, `Interaction`, `Event`, `SocialHypothesis`, `EvidenceRef`, `EntityProfile` | `video_benchmarks_grounding.md` §2, §3.2; `plan_docs_implementation_checklist.md` §1 |
| [`segmenter.py`](segmenter.py) | Adaptive temporal segmentation + scene-change detection + subtitle-aligned frame injection | `video_benchmarks_grounding.md` §3.1 step 1, §3.6 |
| [`perception.py`](perception.py) | Frame sampling, entity-detection hook, subtitle parsing / alignment | `video_benchmarks_grounding.md` §3.1 step 2 |
| [`local_grounder.py`](local_grounder.py) | Per-window social grounding via a pluggable VLM callable | `video_benchmarks_grounding.md` §3.2; `actors_reasoning_model.md` §2.4 |
| [`consolidator.py`](consolidator.py) | Temporal consolidation, cross-window entity resolution, semantic distillation | `video_benchmarks_grounding.md` §3.1 step 4, §3.6; `plan_docs_implementation_checklist.md` §3 entity-resolution gap |
| [`social_video_graph.py`](social_video_graph.py) | `SocialVideoGraph`: graph API + retrieval (wraps `rag.MemoryStore`) + name translation + save/load | `video_benchmarks_grounding.md` §2.3 |
| [`pipeline.py`](pipeline.py) | `build_grounded_context` entry point + six `build_for_*` benchmark presets | `video_benchmarks_grounding.md` §3 / §5 |

Every module exposes *infrastructure primitives*, never first-class
reasoning skills — consistent with
[`atomic_skills_hop_refactor_execution_plan.md`](../infra_plans/04_harness/atomic_skills_hop_refactor_execution_plan.md).

---

## 3. Pipeline

The 5-step pipeline from §3.1 of the plan:

```
Raw video (+ subtitles)
  │
  ├─ 1. segmenter.adaptive_segment
  │       → List[Window]   (frame schedules + scene-change / subtitle anchors)
  │
  ├─ 2. perception.sample_frames / parse_subtitle_file / detect_entities
  │       → SampledFrame[], EvidenceRef[], detection hints
  │
  ├─ 3. local_grounder.ground_window
  │       → GroundedWindow (entities, interactions, events, hypotheses,
  │                          evidence refs — matches §3.2 JSON)
  │
  ├─ 4. consolidator.merge_adjacent_windows
  │            + resolve_entities
  │            + windows_to_nodes + distill_semantic_summaries
  │       → List[GroundingNode]       (retrieval mode only)
  │
  ├─ 5A. (short)  DirectContext(windows=[…])         ← stays in-context
  │
  └─ 5B. (long)   SocialVideoGraph.add_nodes(…)       ← persisted + searchable
```

Steps 1–3 run **identically** for short and long videos. Only step 4/5
differs, which is exactly what the plan demands.

---

## 4. Output schemas

### `GroundedWindow` (per-window)

Emitted by the local grounder. One-to-one with §3.2 of the plan.

```python
GroundedWindow(
    window_id="win_abc",
    time_span=(12.4, 16.8),
    scene="kitchen conversation",
    entities=[Entity(id="p1", type="person",
                     attributes={"emotion": "tense", "gaze": "p2",
                                 "speaking": True})],
    interactions=[Interaction(src="p1", rel="talking_to",
                              dst="p2", confidence=0.84)],
    events=[Event(type="confrontation_start", agents=["p1","p2"],
                  confidence=0.66)],
    social_hypotheses=[SocialHypothesis(type="intention", target="p1",
                                        value="seeking explanation",
                                        confidence=0.61,
                                        provenance="inferred_from_behavior")],
    evidence=[EvidenceRef(ref_id="frm_00188", modality="frame",
                          timestamp=(15.3, 15.3)),
              EvidenceRef(ref_id="sub_0007", modality="subtitle",
                          timestamp=(14.9, 16.2),
                          text="Why didn’t you tell me?")],
)
```

### `DirectContext` (short videos)

```python
DirectContext(
    video_path="clip.mp4",
    duration=93.0,
    mode=<GroundingMode.direct>,
    windows=[GroundedWindow, …],
    subtitle_mode="origin",   # or "added" / "removed" / "none"
    subtitles=[EvidenceRef, …],
)
```

`DirectContext.as_reasoner_text()` renders a compact prompt-ready view
so the shared `reason(...)` loop in
[`actors_reasoning_model.md`](../infra_plans/03_controller/actors_reasoning_model.md) §3
can ingest it without retrieval.

### `SocialVideoGraph` (long videos)

Implements the graph API from §2.3 of the plan. Node kinds:

| `node_type` | What it carries |
|---|---|
| `entity` | Stable person / object / group / speaker with aliases + profile |
| `interaction` | `(src, rel, dst)` with confidence and evidence refs |
| `event` | Temporally localised social/physical event |
| `social_hypothesis` | Intention / belief / emotion / trust / … with provenance |
| `episodic` | Per-window backbone node (maps to episodic memory) |
| `semantic` | Compressed summary across a cluster of windows |

Query methods:

```python
graph.search("who hid the envelope?", top_k=5,
             clip_filter=…, entity_filter=…, time_range=…, node_types=["event"])
graph.get_timeline(entity_id="person_07")
graph.get_relations(entity_id="person_07")
graph.get_evidence(node_id)        # resolves evidence_refs -> EvidenceRef
graph.translate("person_07 left")  # -> "Alice left"
graph.back_translate("Did Alice leave?")  # -> "Did person_07 leave?"
graph.save("graph.json"); SocialVideoGraph.load("graph.json")
```

`evidence` is **not** a separate top-level memory store. It is an
attachment layer on episodic / state records, matching
[`agentic_memory_design.md`](../infra_plans/02_memory/agentic_memory_design.md).

---

## 5. Retrieval backend

The graph uses `rag.retrieval.MemoryStore` when you pass an embedder:

```python
from rag.embedding.text_embedder import get_text_embedder
from visual_grounding import build_for_m3_bench

graph = build_for_m3_bench(
    video_path, subtitle_path=srt_path,
    vlm_fn=my_vlm_callable,
    embedder=get_text_embedder("Qwen/Qwen3-Embedding-0.6B"),
)
```

Without an embedder the graph falls back to a keyword-overlap score —
useful for tests and offline smoke runs, not for production retrieval.

---

## 6. Plugging in models

All external dependencies are **injected** so the package stays light:

| Parameter | Type | Default |
|---|---|---|
| `vlm_fn` | `Callable[[str, frames=...], str]` returning §3.2 JSON | `API_func.ask_model` (text-only fallback) |
| `entity_detector` | `Callable[[frames], list[dict]]` | `None` (entity tracking becomes VLM-only) |
| `entity_resolver` | `AttributeEntityResolver` / `EmbeddingEntityResolver` / your own | `AttributeEntityResolver` |
| `embedder` | `rag.embedding.base.TextEmbedderBase` or `MultimodalEmbedderBase` | keyword fallback |
| `semantic_summarizer` | `Callable[[list[GroundedWindow]], str]` | concatenate scene phrases |

An `EmbeddingEntityResolver(embedding_fn, threshold)` is provided for
M3-Bench-style face/voice re-identification — see
`consolidator.py`.

---

## 7. Benchmark presets (§5 of the plan)

| Preset | Mode | Subtitles | Entity tracking | Voice tracking |
|---|---|---|---|---|
| `build_for_video_holmes` | direct | no | no | no |
| `build_for_siv_bench` | direct | yes (`origin/added/removed`) | light | no |
| `build_for_vrbench` | retrieval | optional | no | no |
| `build_for_long_video_bench` | retrieval | yes | no | no |
| `build_for_cg_bench` | retrieval | optional | no | no |
| `build_for_m3_bench` | retrieval | yes | **high** | yes |

Each preset returns the shared contract
(`DirectContext | SocialVideoGraph`) so downstream adapters just forward
the object to `reason(question, video_context=…)`.

---

## 8. Tests

Benchmark-level schema tests live in
`tests/visual_grounding/test_benchmarks_schema.py`. They feed a **real
sample video from each benchmark** through the matching preset, assert
the return type and schema population, and round-trip the graph through
`save()` / `load()` for M3-Bench.

Videos that are not on disk are `pytest.skip`-ed, so the suite stays
portable. From the repo root (`Video_Skills/`):

```bash
pip install opencv-python-headless pytest
python -m pytest tests/visual_grounding/ -v
```

Expected: `13 passed in ~3-5s` on the dataset mirror at
`/fs/gamma-projects/vlm-robot/datasets/` + `Video_Skills/dataset_examples/`.

The tests rely on a deterministic stub VLM that emits the §3.2 JSON
envelope, so they never hit a real model — only I/O + schema logic is
exercised.

---

## 9. Integration surface

| Consumer | How it uses this package |
|---|---|
| **8B controller** (`actors_reasoning_model.md`) | Calls `build_grounded_context(...)` during offline memory construction; calls `graph.search(...)` / `graph.get_timeline(...)` online. |
| **Skill bank** (`skill_extraction_bank.md`) | Atomic skills consume `GroundingNode` / `EvidenceRef` outputs; infra primitives like `observe_segment`, `detect_entities`, `build_episodic`, `build_semantic`, `search_memory` are exposed here as functions, not as bank skills. |
| **Reasoner** (`reason(...)` in `actors_reasoning_model.md` §3) | Receives `DirectContext` or `SocialVideoGraph` directly as `video_context=`; the direct branch uses `as_reasoner_text()`, the retrieval branch issues `[Search]` queries. |
| **Benchmark adapters** (`video_benchmarks_grounding.md` §5) | Thin wrappers around the `build_for_*` presets. |

---

## 10. Design invariants (do not break)

1. **One schema.** Short and long videos share `GroundedWindow` /
   `GroundingNode`. Never fork a "short-only" or "long-only" type.
2. **Evidence is an attachment, not a store.** Visual, subtitle, and
   audio refs go through `EvidenceRef`. No "visual memory" top-level
   product exists or may be added here.
3. **Infra primitives are not bank skills.** Anything in this package
   (`observe_segment`, `detect_entities`, `build_*`, `search_memory`,
   `translate`, `back_translate`) stays callable but must not be
   serialized into the reasoning skill bank.
4. **VLM-agnostic.** The grounder must accept any `vlm_fn` conforming to
   `(prompt: str, frames=list[dict]) -> str`.
5. **Graceful degradation.** If OpenCV / ffprobe / the embedder / the
   VLM are unavailable, the pipeline must still return an (impoverished
   but well-formed) `DirectContext` or `SocialVideoGraph`, not raise.
