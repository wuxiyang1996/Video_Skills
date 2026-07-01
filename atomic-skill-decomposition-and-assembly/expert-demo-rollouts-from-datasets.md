# Expert Demo Rollouts From Local Video Datasets

Last updated: 2026-06-30

See also:

- [Clip processing policy](../docs/clip-processing-policy.md) — short / long / streaming `clip_policy` per regime and benchmark
- [Implementation status](../docs/implementation-status.md) — runnable scripts, staging, and current gaps
- [Unified video skill schema](../docs/unified-video-skill-schema.md) — canonical data contract

## Goal

Build expert demonstration rollouts for Trace-to-Skill Fitting from the local datasets in:

```text
/fs/gamma-projects/vlm-robot/datasets
```

The target output is not free-form chain-of-thought. The target output is an
executable, typed skill graph composed by the agent over a perceived
clue-memory graph:

```text
video
  -> clips / captions / subtitles / entities / events / clue candidates
  -> clue-memory graph

question + clue-memory graph
  -> agent-composed skill chain / skill graph
  -> skill_id + typed_args + evidence_refs + dependency_edges
  -> answer
```

The clue-memory graph organizes what perception and dataset adapters have made
available. The skill graph records how the agent performs multi-hop reasoning
over that graph. Atomic skills are the executable operations in the skill graph;
composed skills are reusable templates that must expand into atomic skill
invocations before execution.

## Dataset Supervision Audit

| Dataset | Ground truth | Caption / subtitle / summary | Evidence anchors | Best use for rollouts | Confidence |
|---|---|---|---|---|---|
| `CG-Bench` | `cgbench.json`: 12,129 QA, choices, answer | 519 `.srt` files | `clue_intervals` per QA; `cg_videos_clue/{qid}.mp4` clips | high-precision evidence retrieval / clue grounding rollouts | high |
| `Video-Holmes` | train/test QA with answer + explanation | per-video annotation JSON: segment descriptions, inference scenes/shots, key relationships | inference shots/scenes have timestamps; segment descriptions have time ranges | social contradiction, intention, causal/temporal reasoning traces | very high |
| `VRBench` | `VRBench_eval.jsonl`: MCQs + answers | `video_summary` | `mcq[*].reasoning_process` contains timestamped steps | long-video multi-step temporal/evidence-chain rollouts | high |
| `SIV-Bench` | `SIV-Bench-QA.tsv`: QA, answer, category | 2,777 English `.srt` transcripts in `wo_sub`; `w_sub`/`wo_sub` videos | no explicit evidence intervals; align answer/explanation-like rationale from QA to transcript/video | social relation, intent, emotion, counterfactual weak rollouts | medium |
| `M3-Bench` | local copy has videos/subtitles/memory graphs, but no top-level QA JSON found | 100 robot `.srt`; memory graph pickles/tarballs | memory graph nodes likely include event/entity evidence, but need loader | memory-query / temporal-chain rollouts once graph reader is implemented | medium |
| `TIR-Bench` | image QA | image assets | image fields, no video | not primary for video skill rollouts | low |
| `VisualToolBench` | tool-use rubrics + golden answers | images in parquet | tool trajectories in parquet | useful for tool-use format ideas, not video memory | low |

## Recommended Rollout Sources

### Tier 1: Use first

1. **Video-Holmes**
   - Strongest for Trace-to-Skill Fitting.
   - Has QA answer/explanation plus per-video:
     - segment descriptions with time ranges
     - inference shots/scenes with timestamps
     - key relationships
     - supernatural/core-theme fields
   - Can produce rich expert traces without running a visual model for every frame.

2. **CG-Bench**
   - Strongest for evidence grounding.
   - `clue_intervals` and clue clips make verifier labels easy.
   - Good for `localize_clue`, `assign_evidence_role`, `verify_claim_support`.

3. **VRBench**
   - Strongest for long-video multi-step reasoning.
   - `reasoning_process` already gives timestamped intermediate steps.
   - Good for temporal neighborhood retrieval and evidence-chain ordering.

### Tier 2: Add after pipeline works

4. **SIV-Bench**
   - Good for social/intent/emotion/relation coverage.
   - Needs transcript/video alignment by model labeling because no explicit evidence intervals.

5. **M3-Bench**
   - Good for memory graph rollouts.
   - Needs a local graph reader for `.pkl` / tarball memory graph objects.

## Atomic Skills for Demo Rollouts

For the first `expert_demo` pass, use the Reasoning Graph Assembly skill set as
the controller-visible vocabulary. Evidence Graph Construction skills are used
offline by the graph builder and audit trace.

```text
parse_question_target
propose_evidence_roles
retrieve_by_event
retrieve_by_entity
retrieve_by_time
retrieve_by_relation
localize_clue
extract_claim
assign_evidence_role
compose_evidence_chain
infer_temporal_relation
infer_social_contradiction
verify_claim_support
commit_answer
```

For task-specific controller windows, expose additional reasoning atoms as
needed:

```text
detect_missing_role
search_counterevidence
infer_state_change
infer_causal_relation
infer_intention_or_motive
```

## Rollout Record Schema

Each generated expert demo should be stored as JSONL:

```json
{
  "demo_id": "video_holmes:fH6bbNJJfqk:q1",
  "dataset": "Video-Holmes",
  "video_id": "fH6bbNJJfqk",
  "question_id": "1",
  "question": "...",
  "answer": "F",
  "answer_text": "friend",
  "source_supervision": {
    "qa_answer": true,
    "explanation": "...",
    "segment_descriptions": [...],
    "inference_shots": [...],
    "subtitles": [...]
  },
  "skill_graph": {
    "nodes": [
      {
        "node_id": "n1",
        "skill_id": "parse_question_target",
        "args": {"question_text": "..."},
        "output": {"target_entities": ["black-bearded man", "Benjamin"]},
        "evidence_refs": [],
        "confidence": 1.0
      }
    ],
    "edges": [
      {"src": "n1", "dst": "n2", "type": "data"}
    ]
  },
  "verifier": {
    "schema_valid": true,
    "evidence_grounded": true,
    "answer_matches_gt": true,
    "accepted": true
  }
}
```

## Labeling Roles: Rules vs Model

Use rules where the dataset already provides structure. Use `gpt-oss-120` / `gpt-5-mini` only where semantic mapping is needed.

### Rule-based extraction

Do with deterministic code:

- parse MCQ answer / correct option
- parse timestamps from `reasoning_process`, `clue_intervals`, `TimeRange`, `Time`
- map CG `qid` to `cg_videos_clue/{qid}.mp4`
- parse `.srt` into timestamped subtitle chunks
- create initial `EvidenceRef` objects
- validate answer option and timestamp formats

### Model labeling

Use `gpt-5-mini` for higher precision labels and `gpt-oss-120` for cheaper bulk candidate labels, if available.

Ask the model to produce structured JSON only:

1. **Trace segmentation**
   - Input: QA, explanation, segment descriptions, inference shots, subtitles.
   - Output: list of reasoning/evidence operations.

2. **Skill fitting**
   - Input: operation step + allowed skill ontology.
   - Output: `skill_id`, typed args, evidence refs, dependency roles.

3. **Evidence-role labeling**
   - Input: evidence span + question + explanation.
   - Output: role such as `claim_scene`, `contradiction_evidence`, `earlier_linking_action`, `temporal_waypoint`.

4. **Repair target generation**
   - Input: intentionally corrupted skill graph + failure code.
   - Output: local patch, not a rewritten full chain.

Never ask the model to invent new atomic skills in this labeling pass.

## Dataset-Specific Rollout Recipes

### Video-Holmes

Inputs:

```text
Benchmark/train_Video-Holmes.json
Benchmark/test_Video-Holmes.json
Benchmark/annotations/*.json
Benchmark/annotation_training/*.json
Benchmark/videos_cropped/*.mp4
```

Recipe:

1. Load QA record.
2. Load per-video annotation JSON.
3. Normalize fields:
   - `Segment Description` / `SegmentDescription`
   - `Inference Shots` / `InferenceScenes`
   - `Key Relationships` / `KeyRelationships`
4. Create `EvidenceRef`s from segment descriptions and inference shots.
5. Model-label explanation into evidence roles.
6. Fit into skill graph.
7. Accept only if answer option matches and required evidence roles are grounded.

Best skill chains:

```text
parse_question_target
propose_evidence_roles
localize_clue
extract_claim
assign_evidence_role
compose_evidence_chain
infer_social_contradiction / infer_intention_or_motive / infer_causal_relation
verify_claim_support
commit_answer
```

### CG-Bench

Inputs:

```text
cgbench.json
cgbench_mini.json
cg_videos/{video_uid}.mp4
cg_videos_clue/{qid}.mp4
cg_subtitles/cg_subtitles/{video_uid}.srt
```

Recipe:

1. Load QA.
2. Use `clue_intervals` as gold evidence anchors.
3. Use `cg_videos_clue/{qid}.mp4` as gold clue clip if present.
4. Pull matching subtitle lines from clue interval when SRT exists.
5. Label evidence role from question/answer/clue interval.
6. Generate short expert chain.

Best skill chains:

```text
parse_question_target
propose_evidence_roles
localize_clue
assign_evidence_role
compose_evidence_chain
verify_claim_support
commit_answer
```

### VRBench

Inputs:

```text
VRBench_eval.jsonl
v001_360p/*.mp4
```

Recipe:

1. For each `mcq.qa*`, parse `reasoning_process`.
2. Extract timestamp ranges from each reasoning step.
3. Convert each step to an evidence waypoint.
4. Fit timestamped waypoints to `retrieve_by_time`, `infer_temporal_relation`, `compose_evidence_chain`.
5. Use `answer` / `original_answer` as final anchor.

Best skill chains:

```text
parse_question_target
propose_evidence_roles
retrieve_by_event
retrieve_by_time
infer_temporal_relation
compose_evidence_chain
infer_causal_relation / infer_state_change / infer_intention_or_motive
verify_claim_support
commit_answer
```

### SIV-Bench

Inputs:

```text
SIV-Bench-QA.tsv
wo_sub/**/*.srt
w_sub/**/*.mp4
wo_sub/**/*.mp4
origin/**/*.mp4
```

Recipe:

1. Load QA row and category.
2. Locate matching video and transcript.
3. Use model to align answer rationale to subtitle spans and coarse video spans.
4. Generate lower-confidence social reasoning demos.
5. Accept only if model can cite at least one transcript/video span.

Best skill chains:

```text
parse_question_target
propose_evidence_roles
extract_claim
assign_evidence_role
compose_evidence_chain
infer_intention_or_motive / infer_social_contradiction
verify_claim_support
commit_answer
```

### M3-Bench

Inputs:

```text
videos/robot/*.mp4
videos/web/*.mp4
subtitles/robot/*.srt
memory_graphs/**/*.pkl / *.tar.gz
```

Recipe:

1. First implement a graph reader.
2. Extract event/entity/memory nodes with timestamps and evidence refs.
3. Generate rollouts over memory operations, not raw video.
4. Fit to memory retrieval / temporal chain skills.

Best skill chains:

```text
retrieve_by_entity
retrieve_by_event
retrieve_by_time
query_temporal_chain
infer_temporal_relation
compose_evidence_chain
verify_claim_support
commit_answer
```

## Labeling Prompt Skeleton

Use this for `gpt-5-mini` / `gpt-oss-120` structured labeling:

```text
You are converting a video QA example into an executable skill graph.

Allowed skill_ids:
{SKILL_ONTOLOGY}

Input:
- dataset: {dataset}
- question: {question}
- options: {options}
- gold_answer: {answer}
- explanation: {explanation}
- evidence candidates: {evidence_candidates}

Rules:
1. Use only allowed skill_ids.
2. Every non-meta step must cite evidence_ref ids.
3. Do not invent video facts.
4. Do not create new atomic skills.
5. Prefer the shortest graph that supports the answer.
6. If evidence is insufficient, output needs_review=true.

Return JSON:
{
  "nodes": [
    {
      "node_id": "n1",
      "skill_id": "...",
      "args": {},
      "output": {},
      "evidence_refs": [],
      "failure_modes_prevented": []
    }
  ],
  "edges": [
    {"src": "n1", "dst": "n2", "type": "data|temporal|causal|evidence"}
  ],
  "answer_node": "nK",
  "needs_review": false,
  "notes": "short rationale without hidden chain-of-thought"
}
```

## Acceptance Gates

Do not add a rollout to SFT data unless:

```text
schema_valid == true
all_skill_ids_allowed == true
all_required_args_present == true
all_evidence_refs_resolve == true
timestamp_format_valid == true
answer_matches_ground_truth == true
no_unanchored_claims == true
```

Dataset-specific gates:

- CG-Bench: at least one node must cite `clue_intervals` or `cg_videos_clue/{qid}.mp4`.
- Video-Holmes: at least one node must cite `Inference Shots/Scenes` or segment description.
- VRBench: every reasoning-process step should become a timestamped evidence node.
- SIV-Bench: model-labeled spans are `weak_supervision`; keep lower weight.

## Mining Composed Motifs From Accepted Rollouts

Accepted expert-demo rollouts can be mined for reusable composed motifs, but the
motifs remain templates that expand into atomic skills. They must not become new
primitive actions.

Early Video-Holmes generation already shows one candidate family:

```text
video_holmes:train:oZ4pa_5R0nY:q1

Question pattern:
  a later iron fence shot is almost identical to an earlier fence shot

Ground truth:
  F / the man walked back to his original position

Candidate high-level motif:
  resolve_visual_repetition_implication

Observed expansion:
  parse_question_target
  -> propose_evidence_roles
  -> retrieve_by_event
  -> retrieve_by_entity
  -> retrieve_by_time
  -> localize_clue
  -> extract_claim
  -> assign_evidence_role
  -> compose_evidence_chain
  -> infer_temporal_relation
  -> verify_claim_support
  -> commit_answer
```

The evidence roles in this example were:

```text
spatial_anchor_before
spatial_anchor_after
temporal_order
supernatural_or_loop_explanans
```

Two reusable sub-motifs are likely:

```text
match_before_after_spatial_anchor:
  retrieve_by_event
  -> retrieve_by_entity
  -> retrieve_by_time
  -> localize_clue
  -> assign_evidence_role

infer_loop_or_return_from_repeated_clue:
  compose_evidence_chain
  -> infer_temporal_relation
  -> verify_claim_support
  -> commit_answer
```

Promotion rule: only promote a candidate motif after batch statistics show high
subgraph frequency, stable role schemas, verifier pass rate, and answer-support
validity across held-out examples.

## Training Split Recommendation

Start small:

```text
P0:
  Video-Holmes train: 500 QA
  CG-Bench mini: 500 QA
  VRBench: 100 videos x all QA

P1:
  full Video-Holmes train
  CG-Bench mini full
  SIV-Bench selected categories: Relation / Intent / Emotion

P2:
  VRBench full
  CG-Bench full
  M3-Bench after graph reader
```

## What This Gives Us

This produces the expert demos needed for:

- trace-to-skill fitting SFT
- argument/evidence binding SFT
- dependency-edge prediction
- verifier calibration
- repair training via corrupted demos
- ontology-size ablation: `K=8/12/16/24`

The important claim is:

```text
We do not ask the model to imitate teacher CoT.
We convert benchmark supervision, captions, summaries, and teacher labels
into executable verifier-filtered skill graphs.
```
