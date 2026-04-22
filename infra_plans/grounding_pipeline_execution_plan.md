# Grounding Pipeline — Execution Plan (m3-agent based)

> **Purpose:** Cursor-ready, file-by-file execution plan to turn `Video_Skills/out/claude_grounding/` (schema-only smoke test) into the full grounding infrastructure described in [`video_benchmarks_grounding.md`](video_benchmarks_grounding.md). Strategy: vendor the perception / entity-resolution / memory stack from the upstream `m3-agent/` repo and layer the `SocialVideoGraph` semantic model (interactions, events, social hypotheses) on top.
>
> **Related plans (apply together):**
>
> - [Video Benchmarks & Social Grounding](video_benchmarks_grounding.md) — canonical schema, adapters, per-benchmark cheat sheet
> - [Agentic Memory](agentic_memory_design.md) — three stores + evidence + entity-centric indexing
> - [Actors / Reasoning Model](actors_reasoning_model.md) — controller consumes the grounded context built here
> - [Atomic skills / hop refactor](atomic_skills_hop_refactor_execution_plan.md) — infrastructure primitives (`observe_segment`, `detect_entities`, `build_episodic`, …) are implemented by this pipeline
> - [Plan docs checklist](plan_docs_implementation_checklist.md) §3 — this doc closes "Grounded window schema", "Entity resolution", "Benchmark-to-capability mapping", "Adapter definitions", "Grounding error taxonomy"

---

## 0. Motivation — what is broken today

Current `Video_Skills/out/claude_grounding/*.json` is produced by a single frontier-VLM pass per benchmark. It proves the prompt contract is feasible but fails the `SocialVideoGraph` requirements in eight concrete ways:

| # | Gap observed in `claude_grounding/*.json` | Impact |
|---|-------------------------------------------|--------|
| G1 | Entity IDs are window-local (`p1` in "forest" and "elevator" are different people but share the same ID) | Breaks cross-window reasoning, evidence chaining, relationship tracking |
| G2 | Long-video `entities: 0` for `vrbench`, `long_video_bench`, `cg_bench`; `m3_bench` only 3 | Retrieval-mode benchmarks have no entity index to query |
| G3 | No face / voice embeddings, no speaker diarization | Cannot match same person across clips; M3-Bench name translation impossible |
| G4 | `"subtitles": []` in every output despite `subtitle_mode: "origin"` | SIV-Bench subtitle ablations and LongVideoBench subtitle-aware retrieval can't run |
| G5 | Fixed 20s × N windows, no scene-cut or subtitle-aligned keyframes | Video-Holmes clue-linking questions lose temporal alignment; `§3.1/§3.6` of the grounding plan violated |
| G6 | `semantic: 1` for all long videos; no second-pass distillation | Level-3/4 memory missing → hierarchical retrieval degenerates to window search |
| G7 | Evidence `modality` is always `frame` — no clip/subtitle/voice refs | Violates `§2.4` evidence-linked design rule |
| G8 | No `translate` / `back_translate` for M3-Bench name ↔ `<face_N>/<voice_N>` | M3-Bench adapter cannot round-trip names |

## 1. Strategy — vendor m3-agent, layer SocialVideoGraph on top

`m3-agent/` (Bytedance, Apache-2.0, already checked into `/fs/gamma-projects/vlm-robot/m3-agent/`) implements the exact perception / entity / memory substrate we are missing. Its `VideoGraph` matches our `SocialVideoGraph` for `{img, voice, episodic, semantic}`. We inherit its mechanics and extend the node type vocabulary to `{interaction, event, social_hypothesis, state}` — the social-semantic layer the grounding plan calls for.

| `infra_plans` requirement | m3-agent module reused | What we add |
|---|---|---|
| Adaptive sampling + scene cuts + subtitle alignment | `utils/video_processing.process_video_clip`, `has_static_segment` | PySceneDetect integration; subtitle/ASR-aligned frames |
| Face detection + clustering + cross-clip matching | `face_processing.process_faces` (InsightFace `buffalo_l` + DBSCAN + cosine match to graph img nodes) | Thresholds tuned per benchmark |
| Voice diarization + speaker embeddings | `voice_processing.process_voices` (Gemini-driven segmentation + ERes2NetV2) | Optional Whisper fallback when Gemini unavailable |
| Entity identity resolution across the whole video | `videograph.refresh_equivalences` (union-find over face↔voice equivalence assertions) → `character_mappings` | None — used as-is |
| Name ↔ `<face_N>/<voice_N>` translation (M3-Bench) | `retrieve.translate` / `retrieve.back_translate` | Expose on `SocialVideoGraph` |
| Per-clip VLM caption with entity IDs | `memory_processing_qwen.generate_memories` (+ `prompt_generate_memory_with_ids_sft`) | Stage-1 of two-stage extraction |
| Episodic / semantic text node insertion, dedup, edge building | `memory_processing[_qwen].process_memories` + `parse_video_caption` | Generalize to `interaction / event / social_hypothesis` node types |
| Interaction / event / social-hypothesis grounding | — | **Stage-2** structured-extraction prompt (new) |
| Relationship-change detection, Level-3/4 summarization | — | Periodic consolidation pass (new) |
| Benchmark adapters | — | Six adapters behind `BaseAdapter.evaluate(video_path, question, **kwargs)` |

## 2. Vendor plan

### 2.1 Directory layout

```
Video_Skills/grounding/
├── __init__.py
├── perception/                 # vendored from m3-agent/mmagent, Apache-2.0 preserved
│   ├── face.py                 # from face_processing.py
│   ├── voice.py                # from voice_processing.py
│   ├── video_io.py             # from utils/video_processing.py
│   ├── face_clustering.py      # from src/face_clustering.py
│   └── face_extraction.py      # from src/face_extraction.py
├── graph/
│   ├── social_video_graph.py   # subclasses / extends m3-agent VideoGraph
│   └── node_types.py           # adds interaction / event / social_hypothesis / state
├── sampling/
│   └── adaptive.py             # scene-cut + subtitle-aligned sampler
├── pipeline/
│   ├── stage1_caption.py       # m3-agent-style caption with <face_N>/<voice_N>
│   ├── stage2_extract.py       # typed extraction → SocialVideoGraph nodes
│   └── build.py                # build_grounded_context(video_path, mode=...)
├── memory/
│   └── consolidation.py        # Level-3 relationship + Level-4 semantic passes
├── adapters/
│   ├── base.py
│   ├── video_holmes.py
│   ├── siv_bench.py
│   ├── vrbench.py
│   ├── long_video_bench.py
│   ├── cg_bench.py
│   └── m3_bench.py
└── configs/
    ├── processing_config.json  # copied from m3-agent/configs, thresholds tunable
    └── memory_config.json
```

### 2.2 Vendoring rules

- Copy files verbatim where possible; preserve the `Copyright (2025) Bytedance Ltd.` / Apache-2.0 headers.
- Add `VENDORED_FROM.md` at `Video_Skills/grounding/perception/` citing the source commit hash of `m3-agent/`.
- Do **not** modify vendored modules in-place for logic changes; extend via subclass / wrapper in `graph/`, `pipeline/`, or `sampling/`.
- New code (ours) under `grounding/{graph,sampling,pipeline,memory,adapters}` uses the project's default license header, not Apache-2.0.

### 2.3 External dependencies

| Package | Notes |
|---|---|
| `insightface` | Face detection / embedding, model name `buffalo_l` (RetinaFace + ArcFace) |
| `onnxruntime` / `onnxruntime-gpu` | Backend for InsightFace |
| `torchaudio` + `speakerlab` (3D-Speaker) | Voice embedding via ERes2NetV2 |
| `moviepy`, `pydub`, `ffmpeg` | Video/audio I/O |
| `scikit-learn` | DBSCAN + cosine similarity (already a repo dep) |
| `scenedetect>=0.6` | Adaptive cut detection |
| `openai-whisper` *(optional)* | ASR fallback when a benchmark has no SRT |

Weights stored under `Video_Skills/grounding/models/`:

- `buffalo_l/` — downloaded on first use by InsightFace
- `pretrained_eres2netv2.ckpt` — from ModelScope (same URL as m3-agent `README.md` §4)

### 2.4 Config surface

`configs/processing_config.json` keeps m3-agent keys (`max_retries`, `cluster_size`, `face_detection_score_threshold`, `face_quality_score_threshold`, `max_faces_per_character`, `min_duration_for_audio`, `logging`) and adds:

```json
{
  "sampling": {
    "short_base_fps": 1.0,
    "long_base_fps": 0.2,
    "cut_threshold": 27.0,
    "subtitle_align": true,
    "clip_seconds": 30
  },
  "extraction": {
    "stage1_model": "qwen2.5-omni-7b | claude-sonnet-4-5 | gpt-4o",
    "stage2_model": "claude-sonnet-4-5 | gpt-4o",
    "require_equivalence_assertions": true
  },
  "consolidation": {
    "relationship_diff_threshold": 0.35,
    "level4_every_n_clips": 8,
    "semantic_drop_threshold": 0.9
  }
}
```

---

## 3. Phase-by-phase execution

Effort estimates assume one engineer with API + single-GPU access.

### Phase 0 — Foundation (0.5 day)

**Deliverable:** `Video_Skills/grounding/` directory skeleton, dependencies installed, smoke test that imports vendored modules.

- [ ] Copy `m3-agent/mmagent/{face_processing.py, voice_processing.py, videograph.py, src/, utils/video_processing.py, utils/general.py, prompts.py}` → `Video_Skills/grounding/perception/` and `grounding/graph/`.
- [ ] Add `VENDORED_FROM.md` referencing the upstream commit.
- [ ] Create `setup.sh` that installs deps and downloads `pretrained_eres2netv2.ckpt`.
- [ ] Write `tests/test_vendor_import.py`: import `SocialVideoGraph`, `process_faces`, `process_voices` without running them.

**Exit criteria:** `pytest tests/test_vendor_import.py` passes in a fresh env.

### Phase 1 — Adaptive sampler (0.5–1 day)

**Fixes G5.**

**Deliverable:** `grounding/sampling/adaptive.py` with `sample_video(video_path, config) -> List[Clip]`.

- [ ] `Clip` dataclass: `clip_id`, `time_span`, `frames: list[str]` (base64), `frame_timestamps: list[float]`, `subtitle_spans: list[SubtitleSpan]`, `audio_b64: bytes`, `video_b64: bytes`.
- [ ] Base-rate sampling at `short_base_fps` or `long_base_fps` depending on duration threshold (≤ 5 min → short).
- [ ] Cut detection via `scenedetect.ContentDetector(cut_threshold)`; inject one frame at t-0.05 s and one at t+0.05 s around each cut.
- [ ] Subtitle loading: prefer SRT sidecar → fall back to Whisper (gated by `subtitle_align`). Emit `SubtitleSpan(start, end, text)` and add a frame at each span midpoint.
- [ ] Aggregate frames into 30 s clips (aligned with m3-agent's processing unit).
- [ ] `tests/test_sampler.py`: for `dataset_examples/video_holmes/0at001QMutY.mp4`, assert ≥ 1 cut-triggered frame per scene change and non-empty `subtitle_spans` when ASR enabled.

### Phase 2 — Entity stack (1 day)

**Fixes G1, G2, G3, G8.**

**Deliverable:** `grounding/graph/social_video_graph.py` with entity identity persistence.

- [ ] `class SocialVideoGraph(VideoGraph)` — inherits `add_img_node`, `add_voice_node`, `search_img_nodes`, `search_voice_nodes`, `refresh_equivalences`, `character_mappings`.
- [ ] Override `add_text_node` to accept `text_type ∈ {episodic, semantic, interaction, event, social_hypothesis, state}`.
- [ ] Expose `translate(text)` / `back_translate(text)` by porting `mmagent/retrieve.py:translate`, `back_translate` to graph methods.
- [ ] Per-clip driver in `pipeline/build.py`:
  ```python
  for clip in sample_video(video_path, cfg):
      process_faces(graph, clip.frames, save_path=...)        # vendored
      process_voices(graph, clip.audio_b64, clip.video_b64)   # vendored
  graph.refresh_equivalences()
  ```
- [ ] `tests/test_entity_persistence.py`: run on `bedroom_01.mp4`; assert `graph.character_mappings` has ≥ 3 characters, each referenced by ≥ 2 clips.

**Exit criteria:** long-video entity count > 0 (fixes G2); `graph.translate("character_0")` returns at least one `<face_N>` or `<voice_N>` tag (fixes G8).

### Phase 3 — Two-stage grounded extraction (1.5–2 days)

**Fixes G4, G6 (first half), G7, and upgrades G1 from "ID" to "semantically rich ID-bound interaction set".**

**Deliverable:** `grounding/pipeline/{stage1_caption.py, stage2_extract.py, build.py}`.

#### Stage 1 — caption with IDs (vendor-parity)

- [ ] Reuse `prompt_generate_memory_with_ids_sft` (m3-agent `prompts.py:308`).
- [ ] Input builder = m3-agent `generate_video_context` with these inputs:
  - clip video base64 (or sampled frames for API models)
  - face thumbnails with `<face_N>` labels
  - voice ASR segments with `<voice_N>` labels
  - previous clip's episodic tail (1–2 sentences, for continuity)
- [ ] Output: `episodic_memories: list[str]`, `semantic_memories: list[str]` (includes `equivalence: <face_3> = <voice_5>` assertions — required for `refresh_equivalences`).
- [ ] Embed via `text-embedding-3-large`, insert via existing `process_memories` (vendored).

#### Stage 2 — typed extraction (new)

Prompt input = stage-1 caption + clip frames + subtitle spans + current entity profiles. Prompt requires strict JSON conforming to the wire format in `video_benchmarks_grounding.md` §2.6 (added in parallel PR).

- [ ] Template at `grounding/pipeline/prompts/stage2_extract.txt`.
- [ ] Parser uses `parse_video_caption` (vendored) to resolve every `<face_N>` / `<voice_N>` to a concrete `node_id`; skip items that fail resolution and log to `pipeline_errors.jsonl`.
- [ ] For each interaction/event/hypothesis node:
  - `graph.add_text_node(text, clip_id, text_type=<kind>)`
  - `graph.add_edge(new_id, referenced_entity_node_id)` for every `<face_N>` or `<voice_N>` mention
  - Attach `evidence_refs` covering frame path(s), subtitle span id(s), voice segment id(s) — multi-modality fixes G7.
- [ ] `tests/test_stage2_shapes.py`: schema-validate every output node against the wire format; assert 0 rows have empty `supporting_evidence`.

**Exit criteria:** on 6 canonical examples (one per benchmark), summary.json shows entity count > baseline, non-empty subtitles, evidence modality diversity ∈ {frame, subtitle, voice}.

### Phase 4 — Level-3 relationship memory & Level-4 distillation (1 day)

**Fixes G6 (second half).**

**Deliverable:** `grounding/memory/consolidation.py`.

- [ ] `detect_relationship_shifts(graph, clip_id)`:
  - For each character pair `(c_i, c_j)`, build a bag-of-relations vector over the last K clips.
  - Compare with the preceding K-window vector; when cosine-distance > `relationship_diff_threshold`, emit an `event` node `relationship_shift` linked to both characters. Link any pre-existing `social_hypothesis` nodes with `metadata["revised_from"] = <prev_id>` to satisfy the "revision history" requirement in `video_benchmarks_grounding.md` §3.5.
- [ ] `distill_character(graph, character_id)` runs every `level4_every_n_clips` clips:
  - `info_nodes = graph.get_entity_info([character_id])` (vendored)
  - Run an LLM prompt "Summarize this character's role, relationships, and state changes over the evidence attached" → produce 1–3 `semantic` nodes with `level=4`, linked to the character node.
  - Apply `drop_threshold` dedup via m3-agent's `_average_similarity`.
- [ ] `tests/test_consolidation.py`: craft a 4-clip synthetic graph with a deliberate alliance→conflict switch and assert exactly one `relationship_shift` event is emitted.

### Phase 5 — Benchmark adapters (1 day, parallelizable)

**Fixes the shared interface gap and packages everything for `run_eval`.**

**Deliverable:** `grounding/adapters/*.py`, all subclasses of `BaseAdapter`.

```python
class BaseAdapter:
    def build(self, video_path: str, **kwargs) -> SocialVideoGraph | DirectContext: ...
    def evaluate(self, video_path: str, question: str, **kwargs) -> dict: ...
```

| Adapter | `build()` | `evaluate()` specifics |
|---|---|---|
| `VideoHolmesAdapter` | `mode="direct"`, all clips in-context | MC answer + reasoning trace |
| `SIVBenchAdapter` | `mode="direct"`, `subtitle_mode ∈ {origin, added, removed}` (re-runs sampler with alternate SRT) | MC with subtitle ablation |
| `VRBenchAdapter` | `mode="retrieval"` | Multi-step `graph.search` loop, emits `ReasoningTrace` consumed by controller |
| `LongVideoBenchAdapter` | `mode="retrieval"`, subtitle text included in text-node corpus | Single retrieval round + MC |
| `CGBenchAdapter` | `mode="retrieval"` | Returns `{"answer", "evidence": [clip_id, time_span, node_id, …]}` (clue grounding is scored) |
| `M3BenchAdapter` | `mode="retrieval"` with `entity_tracking=True`, `voice_tracking=True` | `question = graph.back_translate(question)` → retrieve → `answer = graph.translate(raw_answer)` |

- [ ] Each adapter has `tests/test_adapter_<name>.py` that runs on one example from `dataset_examples/` and asserts the schema of the return value.

### Phase 6 — Validation (0.5 day)

**Deliverable:** `Video_Skills/out/grounding_v1/` replacing `out/claude_grounding/` as the new baseline.

Exit criteria vs. the gaps in §0:

| Metric | Claude baseline (`out/claude_grounding/summary.json`) | Target (`out/grounding_v1/`) |
|---|---|---|
| `entities` — long_video_bench / vrbench / cg_bench | 0 / 0 / 0 | ≥ 5 each |
| `entities` — m3_bench | 3 | ≥ 10 with non-empty `character_mappings` |
| Non-empty `subtitle_spans` | 0 / 6 benchmarks | 6 / 6 |
| `semantic` nodes per long video | 1 | ≥ 5 |
| Cross-clip entity consistency (share of characters seen in ≥ 2 clips) | 0 | ≥ 0.8 |
| Evidence modality diversity | {frame} | {frame, subtitle, voice} |

Plus a 20-item end-to-end accuracy smoke per benchmark. The numbers here are a sanity floor (not the final paper target):

| Benchmark | Claude single-pass smoke | Exit floor |
|---|---|---|
| Video-Holmes | — | ≥ 65 % |
| SIV-Bench | — | ≥ 60 % |
| M3-Bench-robot | — | ≥ 55 % |

---

## 4. Extensions to the `SocialVideoGraph` node vocabulary

`m3-agent.VideoGraph` allows `Node.type ∈ {img, voice, episodic, semantic}`. We extend to the full grounding vocabulary required by `video_benchmarks_grounding.md` §2.1.

```python
VALID_NODE_TYPES = {
    "img",                 # vendor: face embedding cluster
    "voice",               # vendor: speaker embedding cluster
    "episodic",            # vendor: time-anchored clip description
    "semantic",            # vendor: long-range summary (now multi-level: L1–L4)
    "interaction",         # NEW: (src_entity, rel, dst_entity, confidence, evidence)
    "event",               # NEW: (event_type, agents, confidence, evidence)
    "social_hypothesis",   # NEW: (hyp_type, target, value, confidence, provenance, supporting_evidence, contradicting_evidence, revised_from)
    "state",               # NEW: query-time snapshot (social/spatial subfields; ephemeral)
}
```

Edge conventions:

- `entity(img|voice) ↔ episodic|interaction|event|social_hypothesis`: "entity participates in node".
- `interaction ↔ event`: "interaction is part of event".
- `social_hypothesis ↔ social_hypothesis`: revision chain via `metadata.revised_from`.

All new node types reuse m3-agent's bidirectional weighted edges (`add_edge`, `update_edge_weight`, `reinforce_node`) and DBSCAN-based dedup. The vendor-native `fix_collisions` / `refresh_equivalences` flow is unchanged; equivalence assertions remain scoped to `img/voice` pairs.

---

## 5. Risks & trade-offs

1. **Stage-1 model choice.** m3-agent ships with Qwen2.5-Omni-7B as the default caption-with-IDs model because it ingests video + audio jointly. With API models (Claude / GPT-4o) we emulate this by serializing face thumbnails + voice ASR into the prompt. Quality may drop for dense social scenes; plan to A/B test Qwen2.5-Omni-7B vs. Claude Sonnet 4.5 on SIV-Bench in Phase 6.
2. **Two-stage cost.** A 30 min video with 30 s clips runs ~60 stage-1 calls + 60 stage-2 calls. Mitigation: stage-1 on local Qwen2.5-Omni-7B, stage-2 on API. Cost envelope tracked in `out/grounding_v1/cost.csv`.
3. **Equivalence assertions.** `refresh_equivalences` requires the stage-1 caption to produce strings like `equivalence: <face_3> = <voice_5>`; if a fine-tuned model drops this convention, face/voice entities don't merge. Mitigation: post-hoc cross-modal matcher that compares mouth-movement keyframes against active voice segments when equivalence count is suspiciously low.
4. **Scene-cut false negatives on steady-cam footage.** `ContentDetector(27.0)` can miss slow crossfades in M3-Bench-robot. Mitigation: per-benchmark cut threshold override in `configs/processing_config.json`.
5. **License compatibility.** m3-agent is Apache-2.0; our repo is currently unlicensed-internal. Vendoring preserves Apache-2.0 headers on the copied files — acceptable, but any future public release must review the license mix.

---

## 6. Effort budget

| Phase | Est. person-days | Parallelizable with |
|---|---|---|
| 0 — Foundation | 0.5 | — |
| 1 — Adaptive sampler | 0.5–1 | Phase 2 |
| 2 — Entity stack | 1 | Phase 1 |
| 3 — Two-stage extraction | 1.5–2 | — |
| 4 — Consolidation | 1 | — |
| 5 — Adapters | 1 | Internally parallel (six adapters) |
| 6 — Validation | 0.5 | — |
| **Total** | **5 – 6.5** | |

---

## 7. Cross-reference checklist

This plan, when executed, closes the following items in [`plan_docs_implementation_checklist.md`](plan_docs_implementation_checklist.md):

- §3 (`video_benchmarks_grounding.md`)
  - [x] Grounded window schema (see §2.6 of the grounding plan — wire format)
  - [x] Grounding confidence and uncertainty fields (same §)
  - [x] Entity resolution / re-identification policy (see §2.7 of the grounding plan)
  - [x] Benchmark-to-capability mapping (see §6.1 of the grounding plan)
  - [x] Adapter definitions per benchmark (Phase 5 above)
  - [x] Grounding error taxonomy (see §11 of the grounding plan)
- §2 (`agentic_memory_design.md`)
  - [x] Entity profile schema (see "Entity-centric indexing" subsection)

Infrastructure primitives expected by [`atomic_skills_hop_refactor_execution_plan.md`](atomic_skills_hop_refactor_execution_plan.md) are realized here:

| Primitive | Realized by |
|---|---|
| `observe_segment` | `sampling/adaptive.py:sample_video` |
| `detect_entities` | `perception/{face,voice}.py` (vendored) |
| `build_episodic` | `pipeline/stage1_caption.py` + `process_memories` |
| `build_semantic` | `memory/consolidation.py:distill_character` |
| `update_state` | state node writer in `pipeline/stage2_extract.py` |
| `search_memory` | `SocialVideoGraph.search` (wraps `search_text_nodes`, `get_entity_info`, `get_timeline`) |

---

*This document is the stable execution reference for the grounding pipeline; the canonical design lives in [`video_benchmarks_grounding.md`](video_benchmarks_grounding.md). Update this file when phase status changes.*
