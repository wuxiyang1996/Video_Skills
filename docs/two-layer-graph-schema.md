# Two-Layer Graph Schema

Last updated: 2026-06-30

This document defines the **complete and feasible** schema contract for:

1. **Layer 1 — Clue-memory graph** (`ClueMemoryGraph`): perception, clip index, entity/event store
2. **Layer 2 — Reasoning rollout** (`SkillGraphRollout`): question-conditioned skill program

Both layers must work across **short / long / streaming** video and all four datasets:
Video-Holmes, SIV-Bench, CG-Bench, VRBench.

JSON schemas:

- [`schemas/clue_memory_graph.schema.json`](../schemas/clue_memory_graph.schema.json)
- [`schemas/skill_graph_rollout.schema.json`](../schemas/skill_graph_rollout.schema.json)
- [`schemas/canonical_video_example.schema.json`](../schemas/canonical_video_example.schema.json) (wraps layer-1 in `evidence_index`)

Code builders:

- `dataset_clip_wrapper/clue_memory.py` — `extract_clue_memory_graph()`, `make_reasoning_rollout_shell()`
- `dataset_clip_wrapper/dataset_graph_presets.py` — per-dataset regime defaults
- `dataset_clip_wrapper/pipeline.py` — attaches both layers to `metadata`

---

## Architecture

```text
CanonicalVideoExample
  evidence_index          -> Layer 1 raw index (clips + optional perception nodes)
  metadata.clue_memory_graph   -> Layer 1 normalized export (question-blind)
  metadata.reasoning_rollout_shell -> Layer 2 empty shell linked to Layer 1

Offline memorization (Layer 1):
  video -> clip_policy(regime) -> [long] retrieve top-k coarse -> fine perception
       -> optional clip-schema backend -> graph-crafting skills -> clue nodes/edges

Online QA (Layer 2):
  question + clue_memory_graph -> retrieve -> reasoning skills -> claims -> answer
```

**Hard rule:** Layer 1 must not depend on the question in `video_only` mode.
Layer 2 must cite Layer 1 `node_id` / `evidence_id` for every committed claim.

---

## Layer 1 — ClueMemoryGraph

### Purpose

- Store **what the video contains** at timestamped granularity
- Support retrieval before reasoning
- Scale from ~5 clips (SIV) to ~98 coarse clips (CG-Bench) without full-graph LLM cost

### Required fields

| Field | Meaning |
|-------|---------|
| `graph_id` | Stable id `clue_memory:{example_id}` |
| `video_regime` | `short` \| `long` \| `streaming` |
| `clip_policy` | Segmentation hyperparameters |
| `retrieval` | Coarse top-k gate for long video |
| `observation_end_s` | Streaming visibility cutoff |
| `nodes` / `edges` | Clip index + observations/events/entities |

### Node types (feasible now)

| Type | Source |
|------|--------|
| `clip` | `segment_video` |
| `observation` | subtitles, captions, clip-schema, annotations |
| `dialogue_span` | SRT / ASR |
| `event` | inference shots, reasoning_process steps (expert_demo) |
| `entity` / `entity_mention` | graph-crafting skills (when LLM pipeline runs) |

### Video regime handling

| Regime | Clip policy | Layer-1 behavior |
|--------|-------------|------------------|
| **short** | `whole_video` + 4s fine | Full index; retrieval disabled |
| **long** | `hierarchical` 30s coarse + `retrieval_gated` 8s fine | Index = coarse only; perception on top-k parents |
| **streaming** | short: `fixed_window` 4s + `online=true`; CG/VR: **30s coarse only** + `online=true` | Only nodes with `time_span.end_s <= observation_end_s` |

### Build phases (`perception.build_phase`)

| Phase | Meaning |
|-------|---------|
| `index_only` | Clips + dataset-visible text only (default smoke / CLI) |
| `perception_partial` | Qwen or `video_tools` clip-schema on capped clips |
| `perception_full` | Full perception budget (future / batch) |

---

## Layer 2 — SkillGraphRollout

### Purpose

- Executable **reasoning program** for one question
- Retrieves from Layer 1, never replaces it

### Required link to Layer 1

```json
{
  "layer": "reasoning",
  "clue_memory_ref": {
    "graph_id": "clue_memory:cg_bench:14",
    "index_id": "cg_bench:cg_bench:14:clip_index:v0",
    "observation_end_s": null
  },
  "retrieval_budget": {
    "topk_coarse": 2,
    "max_retrieval_steps": 5
  }
}
```

### Skill nodes

Use the **Reasoning Graph Assembly** atomic skills:

- 19 core skills (`retrieve_by_*`, `infer_*`, `verify_claim_support`, `commit_answer`, …).
- 6 option-level multi-hop/social extensions:
  `generate_answer_hypotheses`, `retrieve_evidence_for_hypothesis`,
  `score_hypothesis_support`, `compare_hypotheses`, `bridge_evidence_hops`,
  and `verify_temporal_social_consistency`.

Every `claim` must list `supported_by_refs` pointing to Layer-1 node ids.

### Verification boundary

The two-layer graph is sufficient. Do **not** add a third verification graph.

Verification belongs in two places:

1. **Layer-2 atomic verification skills** are planner/controller-visible
   actions because they affect reasoning state, claim status, option comparison,
   and final answer selection. Examples:
   `verify_claim_support`, `verify_temporal_social_consistency`,
   `score_hypothesis_support`, and `compare_hypotheses`.
2. **Runtime verifier invariants** are system checks, not planner actions. They
   run after rollout construction and write `verifier_summary`,
   `failure_reasons`, or reward signals. Examples: schema validity, evidence ref
   existence, hidden-supervision leakage, streaming timestamp visibility, and
   the rule that retrieval score alone is not answer support.

This keeps the architecture simple:

```text
Layer 1 = evidence state
Layer 2 = reasoning program + verification trace
Runtime verifier = hard acceptance gates over both layers
```

### Modes

| Mode | Layer-2 supervision |
|------|----------------------|
| `expert_demo` | Teacher / LLM may fit skill graph using hidden annotations for training |
| `video_only` | Only Layer-1 nodes visible at inference; verifier checks no hidden leak |

---

## Four-dataset feasibility matrix

| Dataset | Default regime | Layer-1 index | Expert_demo seeds | video_only feasible |
|---------|----------------|---------------|-------------------|---------------------|
| **Video-Holmes** | short | whole + 4s (~63 clips) | segment + inference shots | clips + subtitles only |
| **SIV-Bench** | short | whole + 4s (~5 clips) | subtitles | clips + parsed SRT |
| **CG-Bench** | long | 30s coarse (~98) + gated fine | clue_intervals, clue clips | coarse index + retrieve |
| **VRBench** | long | 30s coarse + gated fine | reasoning_process timestamps | coarse index + retrieve |

### Streaming notes

Streaming is **schema-compatible for all datasets** even when benchmarks are offline QA:

- Set `regime=streaming` and `observation_end_s` (defaults to `duration_s` in wrapper)
- Layer 1 filters future clips/nodes
- Layer 2 `clue_memory_ref.observation_end_s` must match

For CG-Bench / VRBench, `observation_end_s` can later be tied to question-specific timestamps; MVP uses full observed prefix `[0, duration_s]`.

**CG-Bench / VRBench streaming** uses a **30s coarse online index only** (`index_fine_expansion=none`), avoiding ~900+ 4s fine windows while preserving M3-style long-video indexing.

---

## Per-dataset hidden supervision (Layer 1 only in expert_demo)

| Dataset | Hidden sources |
|---------|----------------|
| Video-Holmes | `segment_annotations`, `inference_shots`, `key_relationships`, `official_answer` |
| SIV-Bench | `official_answer` |
| CG-Bench | `clue_intervals`, `clue_clips`, `official_answer` |
| VRBench | `reasoning_process`, `video_summary`, `official_answer` |

These may appear as Layer-1 nodes in `expert_demo` for trace fitting.
They are **stripped** from `extract_clue_memory_graph(..., mode=video_only)`.

---

## LLM roles (do not blur layers)

| Stage | Model | Layer | Question input |
|-------|-------|-------|----------------|
| Clip-schema | Qwen3.5-9B | 1 | **No** in video_only |
| Graph-crafting planner | gpt-oss-120B | 1 | **No** |
| Reasoning planner / labeler | gpt-oss / gpt-5-mini | 2 | **Yes** |

---

## Validation

```bash
python dataset_clip_wrapper/smoke_test_two_layer_schema.py
python dataset_clip_wrapper/smoke_test_reasoning_rollout.py
python dataset_clip_wrapper/smoke_test_multi_hop_reasoning_skills.py
```

Checks:

- JSON schema validity for both layers
- `clue_memory_ref` linkage
- `video_only` hidden-node leakage
- streaming `observation_end_s` enforcement
- Layer-2 atomic verification skills remain inside `SkillGraphRollout`

---

## Related docs

- [Clip processing policy](clip-processing-policy.md)
- [Unified video skill schema](unified-video-skill-schema.md) §15 (M3 borrowings)
- [Implementation status](implementation-status.md)
