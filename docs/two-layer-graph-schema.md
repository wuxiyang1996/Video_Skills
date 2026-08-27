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

- `dataset_clip_wrapper/l1_clue_graph/clue_memory.py` — `extract_clue_memory_graph()`, `make_reasoning_rollout_shell()`
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

Long-video repair follows the same boundary. It is a controller protocol over
the existing two layers, not a third graph:

```text
coarse visual index
  -> GPT-OSS clue_need_spec planner
  -> GPT-OSS coarse-window selector
  -> retrieved fine clips
  -> L1 repair patch
  -> L2 hypothesis / bridge
  -> verify_claim_support
```

The clue planner produces a structured `clue_need_spec`: visual target,
attributes to resolve, positive evidence criteria, negative evidence to exclude,
objective background facts, bridge evidence criteria, forbidden modalities, and
Qwen clip-inspection instructions. The coarse-window selector then reads the
full coarse visual summary index and chooses candidate windows to inspect in
`direct_visual`, `bridge_context`, or `exploratory_probe` mode. The
`exploratory_probe` retry is used when the model sees that coarse summaries are
too lossy to mention a short event or small object, but the question remains
visually answerable. Lexical retrieval is only a dry-run/no-api fallback. These
are specialized evidence-seeking actions, not independent answer agents. Their
outputs are candidate evidence packs and negative-window diagnostics.

Final answer commit has two final acceptance levels. `resolved_strong` requires
GPT-OSS-backed `verify_claim_support` to ground the claim in non-diagnostic
visual evidence, with enough refs, verifier confidence, and option margin over
the next candidate. `accepted_bridge` is weaker but useful for social, causal,
or background-heavy benchmarks: it requires real visual anchor refs plus stable
objective background facts that disambiguate one option. Those background facts
are L2 bridge context, not L1 evidence nodes, and the report must mark
`not_direct_visual_evidence=true`. `accepted_weak` is not final acceptance; it
is a repair-needed intermediate state.

Repair reports are option-wise. In API runs, GPT-OSS first selects a compact
evidence pack for each option from a budgeted L1 evidence table. Each pack has
positive visual refs, negative refs, missing requirements, selector reason,
verifier decision, confidence, and a short verifier reason. Token-overlap
selection is a no-API diagnostic fallback only. Rule-only verifier runs may
validate structure and surface evidence gaps, but they cannot produce
`resolved_strong`.

Repair reports also expose the recursive L2 process as graph data:

- `l2_trajectory.rounds[]` records compact POMDP/Semi-MDP-compatible steps:
  state snapshot, tool/action, observation summary, graph delta, verifier
  signal, reward proxy, and terminal status.
- `repair_subgraph` contains explicit Layer-2 nodes for gap diagnosis, repair
  planning, L1 patch reference, GPT-OSS evidence selection, option
  verification, optional commonsense/objective bridge verification, and final
  commit or abstain.

This is a bounded recursive trace, not an unbounded agent loop. The default
budget is two repair rounds, and commonsense/background facts remain L2 bridge
context rather than L1 visual evidence.

The option verifier is evidence-gated. If the GPT-OSS evidence selector returns
no positive refs for an option, the verifier is not called for that option and
the option is marked `no_positive_refs_selected`. This keeps model prose from
being treated as support when the graph has no evidence pack.

This matters for long-video QA because `L1 graph_quality=high` only means the
observed clips were converted into a dense graph; it does not prove that the
graph covers the question target. The repair report records `failure_type`
values such as:

- `l1_target_coverage_failure`: retrieved clips do not contain the target event
  or object.
- `l1_attribute_or_evidence_resolution_failure`: target context exists but the
  attribute or support is not resolved.
- `l1_context_partial_l2_bridge_needed`: visual context exists, but answer
  selection requires social, causal, or commonsense bridging.
- `resolved_with_objective_background_bridge`: visual anchors plus stable
  background facts support one option, but the answer is not directly visible.
- `l2_option_margin_insufficient`: multiple answer options have similar support,
  so the verifier cannot commit a final answer.
- `visual_only_benchmark_limitation`: the gold answer appears to require
  audio/subtitle/hidden context outside the video-only scope.
- `l2_verifier_rejects_unsupported`: L2 proposed an answer but verifier rejected
  the evidence pack.

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
python -m dataset_clip_wrapper.tests.smoke_test_two_layer_schema
python -m dataset_clip_wrapper.tests.smoke_test_reasoning_rollout
python -m dataset_clip_wrapper.tests.smoke_test_multi_hop_reasoning_skills
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
