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

## 6. Atomic Skill Specification

Atomic skills are typed primitive operators. They are selected before controller
training and then frozen.

```json
{
  "skill_id": "retrieve_temporal_neighborhood",
  "display_name": "Retrieve Temporal Neighborhood",
  "family": "retrieval",
  "controller_visible": true,
  "input_schema": {
    "query": "string",
    "anchor_evidence_id": "string|null",
    "window_s": "number",
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
  "skill_id": "retrieve_temporal_neighborhood",
  "args": {
    "anchor_evidence_id": "ev:q000123:clue_interval:0",
    "window_s": 30
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
    {"node": "extract_dialogue_claim"},
    {"node": "retrieve_event"},
    {"node": "retrieve_temporal_neighborhood"},
    {"node": "compose_evidence_chain"},
    {"node": "verify_evidence_supports_claim"}
  ],
  "argument_templates": {
    "person": "{{question.target_entity}}",
    "claim": "{{evidence.stated_claim}}"
  },
  "repair_templates": [
    {
      "failure_code": "MISSING_LINKING_ACTION",
      "insert_skill": "retrieve_temporal_neighborhood",
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
```

Model labelers then consume only `CanonicalVideoExample`, not raw benchmark
formats.

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
2. Write adapters for CG-Bench and Video-Holmes first, because they have the
   clearest evidence anchors.
3. Add VRBench, preserving timestamped reasoning steps as ordered evidence.
4. Add SIV-Bench with explicit weak/model-labeled evidence status.
5. Build the expert labeler prompt over the canonical schema only.
6. Run acceptance gates before any rollout enters training.
7. Train the controller on frozen atomic skill ids, with motifs used only as
   optional priors.
8. Add a `video_only` evaluation loader that hides GT clues/reasoning processes
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
      {"slot": "claim", "skill_id": "extract_dialogue_claim"},
      {"slot": "event", "skill_id": "retrieve_event"},
      {"slot": "chain", "skill_id": "compose_evidence_chain"},
      {"slot": "verify", "skill_id": "verify_evidence_supports_claim"}
    ],
    "edges": [
      {"src": "q", "dst": "roles", "edge_type": "data"},
      {"src": "roles", "dst": "claim", "edge_type": "evidence"},
      {"src": "roles", "dst": "event", "edge_type": "evidence"},
      {"src": "claim", "dst": "chain", "edge_type": "claim_support"},
      {"src": "event", "dst": "chain", "edge_type": "claim_refute"},
      {"src": "chain", "dst": "verify", "edge_type": "control"}
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
      "insert_skill": "retrieve_temporal_neighborhood",
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
