# Unified Video Skill Schema

Schema version: `video-skills-relaunch/v0.1`

This document defines the common data interface for trace-to-skill fitting
across video reasoning datasets. It borrows the useful invariants from
`Multi-hop-Reasoning-VLM-Agent/data_structure` but reshapes them around video
assets, temporal evidence, typed skill graphs, and expert demonstration
rollouts.

The goal is not to force every benchmark to look identical. The goal is to make
all datasets enter the system through the same typed contract:

```text
raw dataset item
  -> dataset adapter
  -> CanonicalVideoExample
  -> EvidenceCandidate ledger
  -> expert SkillGraphRollout
  -> controller training / verifier / local repair
```

There are two intended modes:

```text
current expert-demo mode:
  dataset annotations + captions + ground truth clues
    -> expert rollout labeler
    -> trace-to-skill fitting supervision

ultimate video-only mode:
  video + question
    -> automatic segmentation / captioning / clue discovery
    -> atomic skill graph execution
    -> hidden-supervision evaluation
```

The current version may use ground-truth clues and dataset annotations to build
expert demonstrations. The final agent should not receive those clues at
inference time; it should discover evidence from the video itself.

## 1. Design Invariants

The old `data_structure` package has four ideas worth preserving:

- **Episode-local context.** Runtime reasoning should not depend on an external
  memory store. Retrieval is over the current video's captions, annotations,
  clue intervals, subtitles, or derived clips.
- **Evidence-driven steps.** Any reasoning step that grounds, checks, or commits
  a claim must cite evidence.
- **Claim status tracking.** Claims should be marked as `candidate`,
  `verified`, `contradicted`, or `insufficient`.
- **Answer support chain.** A final answer is valid only if it can be traced
  back to verified claims and evidence references.

This relaunch changes the implementation surface:

- Replace generic `Experience` steps with `SkillInvocationNode`.
- Replace generic `Episode` with `SkillGraphRollout`.
- Make video assets, timestamps, captions, clips, and annotation provenance
  first-class.
- Treat composed skills as optional motifs that expand into atomic skill nodes,
  not as new primitive actions.

## 2. File Layout

Recommended persisted files:

```text
data/
  canonical_examples/
    video_holmes.jsonl
    siv_bench.jsonl
    cg_bench.jsonl
    vrbench.jsonl
  expert_rollouts/
    video_holmes.jsonl
    siv_bench.jsonl
    cg_bench.jsonl
    vrbench.jsonl
  skill_ontology/
    atomic_skills.json
    composed_motifs.jsonl
  evidence_indexes/
    video_holmes/
    siv_bench/
    cg_bench/
    vrbench/
  schemas/
    canonical_video_example.schema.json
    skill_graph_rollout.schema.json
```

Each JSON object should carry:

```json
{
  "schema_version": "video-skills-relaunch/v0.1"
}
```

## 3. Canonical Video Example

`CanonicalVideoExample` is the dataset-independent input unit. One item usually
corresponds to one question about one video, but it may point to multiple clips
or annotation sources.

```json
{
  "schema_version": "video-skills-relaunch/v0.1",
  "example_id": "cg_bench:000123",
  "dataset": "cg_bench",
  "split": "train",
  "task_family": "long_video_clue_grounded_qa",
  "video": {
    "video_id": "BVxxxx",
    "primary_path": "datasets/CG-Bench/cg_videos/BVxxxx.mp4",
    "duration_s": 1494.2,
    "fps": 25.0,
    "resolution": {"width": 1280, "height": 720},
    "language": "zh",
    "subtitle_tracks": [
      {
        "track_id": "subtitle:main",
        "path": "datasets/CG-Bench/subtitles/BVxxxx.srt",
        "format": "srt"
      }
    ],
    "caption_tracks": [],
    "derived_clips": [
      {
        "clip_id": "clue_clip:q000123",
        "path": "datasets/CG-Bench/cg_videos_clue/q000123.mp4",
        "source_span": {"start_s": 720.0, "end_s": 742.0}
      }
    ]
  },
  "question": {
    "question_id": "q000123",
    "question_text": "Why did the character change their plan?",
    "question_type": "causal_reasoning",
    "options": [
      {"label": "A", "text": "..."},
      {"label": "B", "text": "..."}
    ],
    "answer": {"label": "B", "text": "..."},
    "answer_format": "multiple_choice"
  },
  "evidence_candidates": [],
  "available_inputs": {
    "mode": "expert_demo",
    "visible_to_agent": [
      "video",
      "question",
      "subtitles",
      "captions",
      "dataset_annotations",
      "ground_truth_clues"
    ],
    "notes": "Expert-demo generation may use supervision that will be hidden at final inference."
  },
  "hidden_supervision": {
    "available_for_training": true,
    "available_for_inference": false,
    "sources": [
      "official_answer",
      "clue_intervals",
      "reasoning_process",
      "segment_annotations"
    ]
  },
  "evidence_index": {
    "index_id": "cg_bench:BVxxxx:m3_style_index:v0",
    "index_type": "clip_memory_graph",
    "visible_in_modes": ["video_only", "expert_demo"],
    "clip_policy": {
      "strategy": "fixed_window",
      "window_s": 30,
      "overlap_s": 0
    },
    "node_types": ["clip", "episodic_text", "semantic_text", "face", "voice", "entity"],
    "edge_types": ["temporal_next", "entity_mention", "face_voice_equivalence", "retrieval_score"],
    "artifact_refs": [
      {
        "artifact_type": "memory_graph",
        "path": "data/evidence_indexes/cg_bench/BVxxxx.pkl"
      }
    ]
  },
  "raw_source_refs": [
    {
      "source_name": "cgbench.json",
      "source_item_id": "q000123",
      "fields_used": ["question", "answer", "choices", "clue_intervals"]
    }
  ],
  "trust_policy": {
    "gold_sources": ["clue_intervals", "official_answer"],
    "strong_sources": ["clue_clip"],
    "weak_sources": ["subtitle"],
    "model_labeled_sources": []
  },
  "metadata": {
    "domain": "film",
    "original_category": "causal"
  }
}
```

### 3.1 Required Fields

| Field | Purpose |
|-------|---------|
| `example_id` | Stable unique id: `{dataset}:{native_id}`. |
| `dataset` | One of `video_holmes`, `siv_bench`, `cg_bench`, `vrbench`. |
| `task_family` | Broad training/eval family, not a benchmark-specific label. |
| `video` | Organized video asset record. |
| `question` | Normalized QA payload. |
| `evidence_candidates` | Unified evidence ledger available to the labeler/controller. |
| `available_inputs` | Inputs visible under the selected mode. |
| `hidden_supervision` | Ground truth or annotations used only for labeling/evaluation. |
| `evidence_index` | Optional runtime retrieval/index structure built from the video. |
| `raw_source_refs` | Provenance back to the original dataset files. |
| `trust_policy` | Which evidence sources are gold, strong, weak, or model-labeled. |

### 3.2 Input Visibility Modes

The same canonical example can be used in two modes:

| Mode | Agent-visible inputs | Hidden from agent | Purpose |
|------|----------------------|-------------------|---------|
| `expert_demo` | Video, question, captions/subtitles, dataset annotations, GT clues if available. | Nothing required, but provenance must be marked. | Generate high-quality expert rollouts for trace-to-skill fitting. |
| `video_only` | Video, question, automatic clips, automatic captions/subtitles, tool-produced evidence. | Official answer, clue intervals, reasoning process, official segment annotations. | Final evaluation of clue discovery and reasoning. |

This distinction is critical. CG-Bench `clue_intervals` and VRBench
`reasoning_process` may be used to create expert demonstrations, but they should
be hidden when evaluating whether the learned agent can find clues by itself.

Use `available_inputs` to say what the labeler/controller can see in the current
run. Use `hidden_supervision` to store ground truth that can score retrieval,
clue recall, evidence-chain quality, and answer correctness.

### 3.3 Evidence Index vs Evidence Candidate

`evidence_index` is not final answer evidence. It is the searchable structure
used to discover candidate evidence from the video. `EvidenceCandidate` is the
unit that a skill node can cite.

This distinction lets us reuse M3-Agent-style organization without turning
memory retrieval into unverified answer support:

```text
video
  -> clips / frames / audio
  -> evidence_index
  -> retrieval skill
  -> EvidenceCandidate
  -> verifier
  -> answer_support_chain
```

In `expert_demo` mode, the index can be seeded or audited with GT clues. In
`video_only` mode, the index must be built only from visible video-derived
signals such as clips, frames, subtitles, ASR, generated captions, face tracks,
voice tracks, and entity links.

### 3.4 Unified Graph Container, Typed Layers

The graph used for video memory and clue organization is not the same object as
the graph composed by the agent for reasoning. The first graph is produced by
perception, indexing, retrieval, and dataset adapters. It stores what has been
seen or inferred from the current video. The second graph is produced by agent
control. It stores the skill chain or skill graph used to perform multi-hop
reasoning over the perceived clues.

The intended relationship is:

```text
video
  -> perception / indexing
  -> EvidenceGraph or clue-memory graph
  -> agent composes SkillGraphRollout
  -> verifier checks evidence bindings and answer support
```

We can model these layers in one shared graph container for inspection,
debugging, or export, but they must remain semantically typed. The intended
structure is a heterogeneous graph with explicit namespaces:

```text
UnifiedVideoReasoningGraph
  evidence.*
    clip
    frame_window
    caption
    subtitle
    asr
    object
    event
    entity
  reasoning.*
    skill_invocation
    intermediate_claim
    hypothesis
    verification_result
    answer
  cross_layer.*
    uses_evidence
    supported_by
    refuted_by
    verified_by
```

This gives one implementation surface for short, long, and streaming videos:
short videos may have a small evidence layer; long videos may have a rich
memory/index layer; streaming videos may have an append-only evidence layer
bounded by `observation_end_s`. The reasoning layer can stay stable across all
three.

The important semantic boundary is:

```text
memory/evidence layer = perceived clue and memory organization
reasoning/skill layer = agent-composed skill program over that graph
```

Recommended flow:

```text
evidence.caption / evidence.event / evidence.entity
  -> reasoning.skill_invocation(retrieve_by_event)
  -> reasoning.skill_invocation(compose_evidence_chain)
  -> reasoning.skill_invocation(verify_claim_support)
  -> reasoning.answer
```

Concerns:

- **Do not collapse the clue-memory graph and the skill graph.** The former
  organizes perceived video content; the latter records the agent's executable
  reasoning actions over that content.
- **Do not treat retrieval as support.** A `retrieval_score` edge only says a
  clip was found; it does not prove that the clip supports the answer.
- **Do not let semantic memory become final evidence by itself.** Semantic
  summaries are useful retrieval priors, but final answers should cite lower
  level clip/caption/frame evidence unless the task explicitly allows summaries.
- **Do not collapse namespaces.** `temporal_next` between clips, `data` between
  skill nodes, and `supported_by` between claim and evidence are different edge
  semantics and should not share one loose `related_to` type.
- **Do not make M3-style memory graph the core architecture.** It is one possible
  evidence-index backend. The core project object is a verifiable skill graph
  over typed evidence.
- **Keep first implementation layered.** Implement `EvidenceGraph` and
  `SkillGraphRollout` as separate layers that can be exported or inspected as a
  unified heterogeneous graph. This avoids early complexity while preserving the
  long-term unified representation.

Safe implementation rule:

```text
Conceptually unified, engineering-layered.
```

For the first implementation, do not replace the existing canonical structures
with one large `UnifiedVideoReasoningGraph` object. Keep the runtime objects
separate and connect them explicitly:

```json
{
  "evidence_graph": {
    "nodes": [],
    "edges": []
  },
  "skill_graph_rollout": {
    "nodes": [],
    "edges": []
  },
  "cross_layer_links": [
    {
      "source": "reasoning.step:001",
      "target": "evidence.clip:042",
      "edge_type": "uses_evidence"
    }
  ]
}
```

This keeps responsibilities clear:

- `EvidenceGraph` handles clips, captions, subtitles, clue intervals, entity
  links, and retrieval metadata.
- `SkillGraphRollout` handles skill calls, reasoning steps, intermediate claims,
  verification nodes, and final answers.
- `CrossLayerLinks` handles explicit bindings such as `uses_evidence`,
  `supports_claim`, `refutes_claim`, and `verified_by`.

Long-video datasets such as CG-Bench and VRBench motivate the unified graph view
because their clue intervals and timestamped reasoning processes naturally bind
evidence nodes to reasoning nodes. However, the first implementation should
preserve the three-part layout above, then optionally export or inspect it as one
heterogeneous graph after adapters, verifiers, and training formats are stable.

## 4. Video Asset Schema

`VideoAsset` describes the media and all aligned text/clip resources.

```json
{
  "video_id": "video_001",
  "primary_path": "/abs/or/repo/relative/video.mp4",
  "duration_s": 192.4,
  "fps": 30.0,
  "resolution": {"width": 1920, "height": 1080},
  "language": "en",
  "subtitle_tracks": [],
  "caption_tracks": [],
  "derived_clips": [],
  "segments": [
    {
      "segment_id": "seg_0001",
      "time_span": {"start_s": 12.0, "end_s": 24.0},
      "source_type": "dataset_segment_description",
      "text": "The man hides the key before leaving.",
      "provenance": {"source_file": "annotation.json", "field": "Segment Description"}
    }
  ]
}
```

Use `segments` for dataset-provided segment descriptions, inferred shots,
timestamped reasoning steps, subtitle windows, or generated captions. Keep the
original provenance so that verifier gates can distinguish official annotations
from model-created labels.

### 4.1 Clip Processing Policy

Long, short, and streaming videos should share the same evidence interface. The
difference is the clip policy used to build `derived_clips`, `segments`, and the
runtime `evidence_index`.

```text
video
  -> clip segmentation
  -> per-clip captions / ASR / objects / events
  -> evidence index
  -> skill graph retrieves relevant clips
  -> verifier checks that cited evidence supports the answer
```

Recommended policies:

| Setting | Clip policy | Purpose |
|---------|-------------|---------|
| Short video | `whole_video` plus small `fixed_window` clips, usually 2-5s with light overlap. | Preserve global context while giving skills precise local evidence. |
| Long video | `hierarchical`: coarse 30-60s windows for retrieval, then 5-10s fine windows inside top candidates. | Avoid full-video reasoning cost while keeping final evidence timestamped. |
| Streaming video | `fixed_window` or `hierarchical` with `online=true` and `observation_end_s=t`. | Enforce causal access: only clips in `[0, t]` are visible, future clips are hidden. |

Default clip sizes should be shared where possible. For both short-video and
streaming-video MVPs, start with `window_s=4` and `overlap_s=1`. The difference
is not the clip size, but the visibility rule: short-video examples may retrieve
from all clips, while streaming examples may retrieve only clips whose
`time_span.end_s <= observation_end_s`. If very low-latency streaming perception
is required, reduce the streaming window to 2s with 0.5-1s overlap; otherwise
keep the shared 4s/1s default for simpler ablations.

Example `clip_policy` values:

```json
{
  "strategy": "whole_video",
  "window_s": 4,
  "overlap_s": 1,
  "online": false,
  "observation_end_s": null
}
```

```json
{
  "strategy": "hierarchical",
  "coarse_window_s": 45,
  "fine_window_s": 8,
  "overlap_s": 2,
  "online": false,
  "observation_end_s": null
}
```

```json
{
  "strategy": "fixed_window",
  "window_s": 5,
  "overlap_s": 1,
  "online": true,
  "observation_end_s": 32.0
}
```

The core invariant is that final answers cite `EvidenceCandidate` records, not
the raw index. For streaming or partial-video QA, every cited evidence span must
satisfy `time_span.end_s <= observation_end_s`. This lets streaming datasets
share the same schema without making the main project depend on streaming as
the default setting.

For the full reference — benchmark presets, legacy `Video_Skills` mapping,
M3 borrowing, atomic-skill entry point, implementation gaps, and streaming QA
field mapping — see [clip-processing-policy.md](clip-processing-policy.md).

### 4.2 Benchmark Clip Presets

| Dataset | Regime | Recommended policy |
|---------|--------|-------------------|
| Video-Holmes | Short | `whole_video` + `fixed_window(4s, 1s)` |
| SIV-Bench | Very short | `whole_video` + subtitle-aligned spans |
| CG-Bench | Long | `hierarchical(45s coarse, 8s fine)` |
| VRBench | Long | `hierarchical(45s coarse, 8s fine)` |
| M3-Bench | Long + memory | `fixed_window(30s)` or imported M3 graph clips |

In `expert_demo` mode, dataset annotations and clue intervals may seed the
evidence index directly. In `video_only` mode, the same `clip_policy` builds
the index from automatic segmentation and perception only.

## 5. Evidence Candidate Schema

`EvidenceCandidate` is the common unit consumed and produced by skill nodes.

```json
{
  "evidence_id": "ev:q000123:clue_interval:0",
  "source_type": "clue_interval",
  "time_span": {"start_s": 720.0, "end_s": 742.0},
  "media_ref": {
    "video_id": "BVxxxx",
    "path": "datasets/CG-Bench/cg_videos/BVxxxx.mp4"
  },
  "text": "The character says they cannot leave, then later exits through the back door.",
  "entities": [
    {"entity_id": "person:main_character", "surface": "the character"}
  ],
  "claims": [
    {
      "claim_id": "claim:q000123:0",
      "text": "The character's later action contradicts their stated plan.",
      "claim_status": "candidate"
    }
  ],
  "evidence_role": "contradicting_action",
  "confidence": 1.0,
  "trust_level": "gold",
  "provenance": {
    "source_file": "cgbench.json",
    "source_field": "clue_intervals",
    "created_by": "dataset_adapter"
  }
}
```

### 5.1 Evidence Source Types

Use this controlled vocabulary first:

```text
video_segment
frame_window
subtitle_span
caption_span
segment_description
inference_shot
key_relationship
clue_interval
clue_clip
reasoning_process_step
video_summary
qa_answer
model_labeled_span
```

### 5.2 Trust Levels

| Trust level | Meaning |
|-------------|---------|
| `gold` | Official timestamp, clue, answer, or annotation from dataset. |
| `strong` | Dataset-provided but indirect evidence, such as official summaries. |
| `weak` | Useful but not guaranteed aligned, such as subtitles without evidence span. |
| `model_labeled` | Produced by GPT-5 mini / gpt-oss-120B labeler. |
| `derived` | Programmatically derived from other evidence. |

### 5.3 Runtime Discovery Status

Evidence should also record whether it was given by the dataset or discovered by
the agent/tool pipeline:

```json
{
  "evidence_id": "ev:auto:clip_001",
  "source_type": "model_labeled_span",
  "trust_level": "model_labeled",
  "discovery_status": "discovered_runtime",
  "provenance": {
    "created_by": "auto_segment_and_caption",
    "visible_in_mode": "video_only"
  }
}
```

Recommended values:

```text
provided_supervision
provided_visible_context
discovered_runtime
derived_runtime
hidden_eval_only
```

For the final video-only setting, answer evidence should come from
`discovered_runtime`, `derived_runtime`, or `provided_visible_context`, not from
`provided_supervision` or `hidden_eval_only`.

### 5.4 Mapping M3-Style Memory Nodes to Evidence

M3-Agent organizes long videos as a multimodal memory graph:

```text
clip_id
  -> episodic text nodes
  -> semantic text nodes
  -> face nodes
  -> voice nodes
  -> face/voice/entity equivalence edges
```

For our system, these should map into evidence as follows:

| M3-style object | Our schema target | Notes |
|-----------------|-------------------|-------|
| 30s clip | `VideoAsset.derived_clips` and `EvidenceCandidate.source_type=video_segment` | Keep `clip_id`, start/end time, and path. |
| Episodic memory | `EvidenceCandidate.source_type=caption_span` | Good for observable events, actions, dialogue, scene facts. |
| Semantic memory | `EvidenceCandidate.source_type=model_labeled_span` | Useful as retrieval prior, but lower trust unless verified by raw clip/caption. |
| Face node | `entities[].modality=face` | Supports entity tracking and identity resolution. |
| Voice node | `entities[].modality=voice` | Supports speaker/dialogue grounding. |
| Equivalence line | `entity_link` or `face_voice_equivalence` edge in `evidence_index` | Should not be final answer evidence by itself. |
| Retrieval score | `provenance.retrieval_score` | Useful for ranking, not sufficient for verification. |

The important improvement is to keep both levels:

```text
episodic evidence = what happened in a clip
semantic evidence = what the system inferred across clips
```

When semantic evidence supports an answer, the verifier should ask for at least
one lower-level episodic/clip citation unless the task explicitly allows
summary-level evidence.

## 6. Atomic Skill Specification

Atomic skills are typed primitive operators. They are selected before controller
training and then frozen.

```json
{
  "skill_id": "retrieve_by_time",
  "display_name": "Retrieve By Time",
  "family": "retrieval",
  "controller_visible": true,
  "input_schema": {
    "query": "string",
    "anchor_event_or_time": "string",
    "window_before": "number",
    "window_after": "number",
    "source_filter": "array<string>"
  },
  "output_schema": {
    "evidence_refs": "array<string>",
    "retrieval_rationale": "string"
  },
  "preconditions": [
    "question target or anchor evidence is available"
  ],
  "effects": [
    "adds temporally adjacent evidence candidates"
  ],
  "verifier": {
    "verifier_id": "temporal_span_overlap_or_caption_match",
    "required": true,
    "hardness": "semi_hard"
  },
  "failure_codes": [
    "NO_NEIGHBOR_FOUND",
    "BAD_TIMESTAMP",
    "WRONG_ENTITY",
    "EVIDENCE_TOO_BROAD"
  ],
  "executor_type": "retriever"
}
```

### 6.1 Atomic Skill Runtime Contract

Every invocation must produce:

```json
{
  "node_id": "n3",
  "skill_id": "retrieve_by_time",
  "args": {
    "anchor_event_or_time": "ev:q000123:clue_interval:0",
    "window_before": 30,
    "window_after": 30
  },
  "inputs_from": ["n2"],
  "outputs": {
    "evidence_refs": ["ev:q000123:subtitle:18", "ev:q000123:subtitle:19"]
  },
  "evidence_refs": ["ev:q000123:subtitle:18", "ev:q000123:subtitle:19"],
  "claim_ids": [],
  "status": "verified",
  "verifier_result": {
    "passed": true,
    "score": 0.86,
    "messages": []
  },
  "failure_code": null,
  "cost": {
    "model": null,
    "input_tokens": 0,
    "output_tokens": 0,
    "latency_ms": 42
  }
}
```

## 7. Composed Motif Specification

Composed skills should be treated as reusable motifs, not new atomic actions.
They provide graph templates, evidence-role templates, and repair priors. They
must always expand into atomic skill nodes before execution and verification.

```json
{
  "motif_id": "resolve_alibi_contradiction",
  "name": "Resolve Alibi Contradiction",
  "status": "optional_prior",
  "trigger": {
    "question_patterns": ["why was .* inconsistent", "who was lying"],
    "required_reasoning_types": ["social_contradiction", "temporal_relation"]
  },
  "evidence_role_template": [
    "stated_claim",
    "contradicting_visual_or_dialogue_evidence",
    "linking_action",
    "temporal_order"
  ],
  "expands_to": [
    {"node": "parse_question_target"},
    {"node": "propose_evidence_roles"},
    {"node": "extract_claim"},
    {"node": "retrieve_by_event"},
    {"node": "retrieve_by_time"},
    {"node": "compose_evidence_chain"},
    {"node": "verify_claim_support"},
    {"node": "commit_answer"}
  ],
  "argument_templates": {
    "person": "{{question.target_entity}}",
    "claim": "{{evidence.stated_claim}}"
  },
  "repair_templates": [
    {
      "failure_code": "MISSING_LINKING_ACTION",
      "insert_skill": "retrieve_by_time",
      "search_direction": "earlier"
    }
  ],
  "constraints": [
    "motif carries no old-video facts",
    "motif cannot bypass node-level verifier",
    "motif cannot create a new atomic skill id"
  ]
}
```

## 8. Skill Graph Rollout Schema

`SkillGraphRollout` is the expert demonstration format used for controller
training, offline fitting, and verifier evaluation.

```json
{
  "schema_version": "video-skills-relaunch/v0.1",
  "rollout_id": "rollout:cg_bench:q000123:gpt5mini:v1",
  "example_id": "cg_bench:q000123",
  "rollout_source": "gpt5mini_labeler_with_gold_clue_intervals",
  "labeler": {
    "model": "gpt-5-mini",
    "prompt_version": "trace_to_skill_labeler_v0.1",
    "temperature": 0.0
  },
  "used_motifs": [
    {
      "motif_id": "resolve_alibi_contradiction",
      "expanded": true,
      "instantiated_nodes": ["n1", "n2", "n3", "n4"]
    }
  ],
  "nodes": [],
  "edges": [
    {
      "edge_id": "e1",
      "src": "n1",
      "dst": "n2",
      "edge_type": "data",
      "payload": {"field": "question_target"}
    }
  ],
  "claims": [
    {
      "claim_id": "claim:q000123:answer",
      "text": "The answer is B because the clue interval shows the action contradicting the claim.",
      "claim_status": "verified",
      "supported_by_refs": ["ev:q000123:clue_interval:0"],
      "contradicted_by_refs": [],
      "parent_claims": ["claim:q000123:0"]
    }
  ],
  "answer_support_chain": [
    {
      "node_id": "n6",
      "claim_id": "claim:q000123:answer",
      "evidence_refs": ["ev:q000123:clue_interval:0"]
    }
  ],
  "final_answer": {
    "label": "B",
    "text": "...",
    "confidence": 0.92
  },
  "verifier_summary": {
    "schema_valid": true,
    "all_commits_have_evidence": true,
    "answer_chain_valid": true,
    "timestamp_valid": true,
    "no_old_video_fact_leakage": true
  },
  "acceptance_status": "accepted",
  "failure_reasons": []
}
```

### 8.1 Edge Types

Use this vocabulary:

```text
data
temporal
causal
entity
evidence
claim_support
claim_refute
repair
control
alternative
```

### 8.2 Node Statuses

```text
planned
executed
verified
contradicted
insufficient
repaired
failed
skipped
```

## 9. Dataset Adapter Contracts

Every adapter implements the same two functions:

```python
def to_canonical_examples(raw_root: str, split: str) -> Iterable[CanonicalVideoExample]:
    ...

def seed_evidence_candidates(example: CanonicalVideoExample) -> list[EvidenceCandidate]:
    ...

def build_evidence_index(example: CanonicalVideoExample, mode: str) -> EvidenceIndex:
    ...
```

Model labelers then consume only `CanonicalVideoExample`, not raw benchmark
formats.

For short datasets, `build_evidence_index` can be lightweight: whole-video clip,
subtitles, captions, and detected entities. For long datasets, use a richer
clip-level index similar to M3-Agent: fixed-window clips, episodic memories,
semantic memories, face/voice nodes, and retrieval scores.

### 9.1 Video-Holmes

Useful sources:

- QA question, options, answer, and explanation.
- Per-video `Segment Description` / `SegmentDescription`.
- `Inference Shots` / `InferenceScenes`.
- `Key Relationships` / `KeyRelationships`.
- Cropped video.

Adapter behavior:

- Convert each segment description into `segment_description` evidence.
- Convert inference shots into `inference_shot` evidence with `gold` or
  `strong` trust depending on timestamp precision.
- Convert key relationships into `key_relationship` evidence.
- Seed expected evidence roles such as `setup_event`, `hidden_action`,
  `contradicting_event`, `motive_cue`, and `social_relation`.

Best use:

- Train trace-to-skill fitting for short but reasoning-heavy examples.
- Fit social contradiction, intention, causal support, and evidence-role
  assignment.

### 9.2 SIV-Bench

Useful sources:

- Short videos.
- QA TSV with social reasoning categories.
- Subtitles for many videos.
- No reliable gold evidence interval in the current local copy.

Adapter behavior:

- Use the whole video as weak `video_segment` evidence.
- Convert subtitle windows into `subtitle_span` evidence.
- Treat category labels as `task_family` hints, not as skill labels.
- Ask the labeler to propose evidence spans, but mark them as
  `model_labeled`.

Best use:

- Evaluate whether skills learned from Video-Holmes transfer to short
  social-interaction reasoning.
- Use lower trust for evidence-span supervision.

### 9.3 CG-Bench

Useful sources:

- Full video.
- `clue_intervals`.
- Clue clips.
- QA, options, correct answer, domain, sub-category, duration.
- Some subtitles.

Adapter behavior:

- Convert each clue interval into `gold` `clue_interval` evidence.
- Link clue clips to their source spans as `strong` `clue_clip` evidence.
- Use subtitles around clue intervals as weak or derived context.
- Keep full-video metadata for retrieval-cost experiments.
- In `expert_demo` mode, expose clue intervals to the labeler for expert rollout
  construction.
- In `video_only` mode, move clue intervals to `hidden_supervision`; the agent
  must recover matching spans through retrieval/segmentation.

Best use:

- Train and evaluate long-video clue retrieval, temporal neighborhood
  retrieval, and grounded QA with strong evidence anchors.

### 9.4 VRBench

Useful sources:

- Long video.
- Video summary.
- Multiple QA items per video.
- Timestamped `reasoning_process`.

Adapter behavior:

- Convert video summary into `strong` `video_summary` evidence.
- Convert each timestamped reasoning step into `reasoning_process_step`
  evidence.
- Preserve the order of reasoning steps as initial temporal/claim edges.
- Use official answer as gold answer, not as free evidence for intermediate
  claims.
- In `expert_demo` mode, expose timestamped reasoning steps to the labeler.
- In `video_only` mode, keep timestamped reasoning steps hidden for evaluation
  of evidence-chain discovery.

Best use:

- Train/evaluate long-video multi-step temporal and evidence-chain reasoning.
- Stress-test whether the controller can assemble multiple retrieval and
  verification nodes over a long horizon.

## 10. Acceptance Gates

An expert rollout is accepted only if all gates pass:

| Gate | Requirement |
|------|-------------|
| `G0_schema` | JSON validates against the schema version. |
| `G1_evidence_refs` | Every referenced evidence id exists in the example ledger. |
| `G2_timestamp` | Every cited timestamp is inside video duration. |
| `G3_commit_evidence` | Every answer/claim commit has evidence refs. |
| `G4_answer_chain` | Final answer resolves to at least one verified root claim. |
| `G5_no_fact_leakage` | Motifs/cases do not inject facts from another video. |
| `G6_skill_validity` | Every skill id exists in frozen `atomic_skills.json`. |
| `G7_motif_expansion` | Every used motif is expanded into atomic nodes. |
| `G8_visibility` | In `video_only` mode, no node cites hidden supervision. |

For weakly supervised datasets such as SIV-Bench, gates should distinguish
`accepted_gold`, `accepted_weak`, and `accepted_model_labeled` instead of
pretending all evidence has the same strength.

For `expert_demo` mode, citing gold clues is allowed when provenance is explicit.
For `video_only` mode, citing gold clues is a leakage failure unless the span was
rediscovered by a runtime retrieval or segmentation skill.

## 11. Labeler Output Policy

GPT-5 mini or gpt-oss-120B should label structure, not invent unsupported facts.
The labeler receives:

- normalized question and answer;
- video-level captions/subtitles/annotations;
- candidate evidence ledger with trust levels;
- frozen atomic skill ontology;
- optional motif templates.

The labeler outputs:

- a skill graph rollout;
- evidence-role assignments;
- claim graph;
- final answer support chain;
- local repair attempts if an initial node fails.

The labeler must not:

- create new atomic skill ids;
- cite evidence outside the current example;
- use benchmark category names as final skill labels;
- use a motif as a black-box executor;
- claim gold evidence when the source is model-labeled.

## 12. Why This Interface Works for the Four Datasets

The four datasets differ mostly in evidence strength and video length, not in
the reasoning interface:

| Dataset | Length regime | Evidence strength | Main schema path |
|---------|---------------|-------------------|------------------|
| Video-Holmes | Short, reasoning-heavy | Strong annotations | Segment/inference/key relationship evidence -> skill graph. |
| SIV-Bench | Very short social clips | Weak span supervision | Subtitle/whole-video evidence -> model-labeled roles. |
| CG-Bench | Medium/long clue-grounded | Gold clue intervals | Clue interval/clip evidence -> retrieval and QA graph. |
| VRBench | Long multi-step | Timestamped reasoning process | Reasoning steps -> temporal evidence-chain graph. |

This means we can train one controller interface while preserving dataset-specific
provenance and trust levels.

The same interface also supports the long-term objective: train from
supervision-rich expert demonstrations, then evaluate in a stricter video-only
setting where clue intervals and reasoning chains are hidden labels.

## 13. Minimal Implementation Order

1. Implement JSON schema dataclasses for `CanonicalVideoExample`,
   `EvidenceCandidate`, `AtomicSkillSpec`, `CompositeMotif`, and
   `SkillGraphRollout`.
2. Implement a minimal `EvidenceIndex` interface inspired by M3-Agent:
   fixed-window clips, episodic/semantic text nodes, entity links, and retrieval
   traces.
3. Write adapters for CG-Bench and Video-Holmes first, because they have the
   clearest evidence anchors.
4. Add VRBench, preserving timestamped reasoning steps as ordered evidence.
5. Add SIV-Bench with explicit weak/model-labeled evidence status.
6. Build the expert labeler prompt over the canonical schema only.
7. Run acceptance gates before any rollout enters training.
8. Train the controller on frozen atomic skill ids, with motifs used only as
   optional priors.
9. Add a `video_only` evaluation loader that hides GT clues/reasoning processes
   and scores discovered evidence against them.

## 14. Composed Motif Extraction

Yes, we should design an extraction module for composed skills, but the object
being extracted should be a verified sub-graph motif rather than a new tool.

The extraction target is:

```text
frequent verified atomic-skill subgraph
  + reusable evidence roles
  + argument-binding template
  + typical failure/repair pattern
  -> CompositeMotif
```

The extraction target is not:

```text
new atomic skill
new black-box executor
benchmark-specific shortcut
old-video fact memory
```

### 14.1 Inputs

Motif extraction consumes accepted `SkillGraphRollout` objects only:

```text
accepted_gold
accepted_strong
accepted_weak, only for low-confidence motif proposals
```

Each rollout contributes:

- atomic skill node ids and skill ids;
- dependency edges;
- evidence roles;
- claim statuses;
- answer support chain;
- verifier results;
- local repair records;
- dataset/task metadata.

Rejected rollouts should not create motifs. They can only contribute negative
failure statistics.

### 14.2 Canonicalization Before Mining

Before mining, remove dataset-specific surface forms:

```text
person names        -> ENTITY_PERSON_A / ENTITY_PERSON_B
object names        -> ENTITY_OBJECT_A
absolute timestamps -> relative order or span role
option labels       -> ANSWER_CANDIDATE_A / ANSWER_CANDIDATE_B
dataset categories  -> broad task_family
```

Keep the evidence role and graph structure:

```text
stated_claim
contradicting_event
linking_action
temporal_order
causal_trigger
state_before
state_after
answer_support
```

This is the main protection against benchmark-specific hand-crafting.

### 14.3 Candidate Mining

A simple first version can use graph n-grams instead of complex subgraph mining:

```text
length-2 paths:  skill_a -> skill_b
length-3 paths:  skill_a -> skill_b -> skill_c
small DAGs:      one retrieval node feeding two verification/inference nodes
repair motifs:   failed node -> repair node -> verified replacement
```

For each candidate subgraph, compute:

| Metric | Meaning |
|--------|---------|
| `support_count` | Number of rollouts containing this motif. |
| `dataset_coverage` | Number of datasets where it appears. |
| `task_family_coverage` | Number of broad task families where it appears. |
| `verifier_pass_rate` | Fraction of motif instances whose final claims verify. |
| `repair_success_rate` | Fraction of failed instances repaired locally. |
| `compression_gain` | Planning steps reduced if used as a prior. |
| `confusion_risk` | How often similar triggers choose the wrong motif. |

### 14.4 Promotion Rule

A candidate becomes a `CompositeMotif` only if it passes all gates:

```text
support_count >= threshold
dataset_coverage >= 2, unless explicitly marked dataset_local
verifier_pass_rate >= threshold
confusion_risk <= threshold
all nodes are frozen atomic skills
all evidence roles are abstract roles, not video-specific facts
motif expands into a valid SkillGraphRollout fragment
```

Dataset-local motifs can be kept for analysis, but should not be used as the
main transfer story.

### 14.5 Extracted Motif Record

```json
{
  "motif_id": "motif:social_contradiction:claim_vs_action:v1",
  "source": "offline_subgraph_mining",
  "status": "candidate",
  "support": {
    "support_count": 184,
    "dataset_coverage": ["video_holmes", "siv_bench"],
    "task_family_coverage": ["social_contradiction", "intent_reasoning"],
    "verifier_pass_rate": 0.81,
    "repair_success_rate": 0.34,
    "compression_gain": 2.7,
    "confusion_risk": 0.09
  },
  "trigger_signature": {
    "question_intents": ["contradiction", "deception", "inconsistent_action"],
    "required_evidence_roles": [
      "stated_claim",
      "contradicting_event",
      "linking_action"
    ]
  },
  "graph_template": {
    "nodes": [
      {"slot": "q", "skill_id": "parse_question_target"},
      {"slot": "roles", "skill_id": "propose_evidence_roles"},
      {"slot": "claim", "skill_id": "extract_claim"},
      {"slot": "event", "skill_id": "retrieve_by_event"},
      {"slot": "chain", "skill_id": "compose_evidence_chain"},
      {"slot": "verify", "skill_id": "verify_claim_support"},
      {"slot": "answer", "skill_id": "commit_answer"}
    ],
    "edges": [
      {"src": "q", "dst": "roles", "edge_type": "data"},
      {"src": "roles", "dst": "claim", "edge_type": "evidence"},
      {"src": "roles", "dst": "event", "edge_type": "evidence"},
      {"src": "claim", "dst": "chain", "edge_type": "claim_support"},
      {"src": "event", "dst": "chain", "edge_type": "claim_refute"},
      {"src": "chain", "dst": "verify", "edge_type": "control"},
      {"src": "verify", "dst": "answer", "edge_type": "control"}
    ]
  },
  "argument_template": {
    "target_entity": "{{question.target_entity}}",
    "claim_evidence": "{{evidence_role.stated_claim}}",
    "counter_evidence": "{{evidence_role.contradicting_event}}"
  },
  "repair_template": [
    {
      "failure_code": "MISSING_LINKING_ACTION",
      "insert_skill": "retrieve_by_time",
      "position": "before:compose_evidence_chain"
    }
  ],
  "constraints": [
    "expand_before_execution",
    "cite_current_video_evidence_only",
    "do_not_create_atomic_skill",
    "run_node_level_verifiers"
  ]
}
```

### 14.6 Runtime Use

At inference time, motifs should be used in this order:

```text
parse question
  -> retrieve top-k motif templates by trigger_signature
  -> instantiate motif arguments on current video evidence
  -> expand into atomic skill graph
  -> execute atomic nodes
  -> verify evidence and claims
  -> locally repair failed nodes
```

The controller can still ignore the motif if the current evidence does not fit.
This keeps motifs as priors, not commands.

### 14.7 Ablations We Should Run

To make this defensible to reviewers:

| Setting | Purpose |
|---------|---------|
| No motif | Pure atomic skill graph assembly baseline. |
| Retrieved motif as prompt only | Tests whether high-level planning prior helps. |
| Retrieved motif expanded to graph | Tests structured prior with verifier. |
| Dataset-local motif only | Checks benchmark overfitting. |
| Cross-dataset motif transfer | Main evidence that motifs are general. |

The expected claim should be modest:

> Composed motifs are reusable verified subgraph priors mined from successful
> traces. They improve planning efficiency and local repair, while final
> execution remains grounded in atomic skills and current-video evidence.

## 15. Reference: What to Borrow From M3-Agent

M3-Agent is useful for the ultimate video-only goal because it shows a practical
way to organize long-video evidence before answering:

```text
video
  -> 30s clips
  -> face detection + speaker diarization
  -> episodic memory per clip
  -> semantic memory per clip/entity
  -> multimodal graph
  -> iterative search/answer control loop
```

We should borrow:

- fixed-window clip indexing as a simple first segmentation policy;
- separate episodic and semantic text memories;
- entity-centric links between face, voice, and text nodes;
- query-time clip retrieval with scores;
- iterative search traces as supervision for retrieval skills;
- `before_clip` style temporal constraints for streaming or partial-video QA.

We should not directly borrow:

- unrestricted long-term memory as answer evidence;
- semantic memory without lower-level clip support;
- free-form `[SEARCH]` / `[ANSWER]` control as the final policy interface;
- persistent facts across videos for benchmark QA.

The right integration is:

```text
M3-style memory graph = evidence_index
atomic skills = operations over the evidence_index
EvidenceCandidate = cited, verifiable evidence extracted from the index
SkillGraphRollout = reasoning chain over extracted evidence
```

This gives us the best of both designs: M3-style organization for clue discovery,
and our typed skill graph for verifiable reasoning.
