# Implementation Status

Last updated: 2026-07-05

This document tracks what is designed, what is implemented, and how to run the
current code. It consolidates status from README, atomic skills v1, dataset
rollout plans, and recent experiments.

## 0. Packaging / Bundle Cleanup Status

`dataset_clip_wrapper/` has been split into physical bundle subpackages:

- `perception/`: clip policy, Qwen/video-tools clip schemas, subtitles, video probes, OpenRouter client.
- `l1_clue_graph/`: clue-memory graph extraction, graph compose, retrieval, L1 gating.
- `l2_reasoning_graph/`: reasoning planner, reasoning rollout, recursive trace, local fault repair.
- `verification/`: repair protocol, quality reports, final acceptance, runtime verifier.
- `runners/`: staged and non-staged API pipelines.
- `tests/`: smoke tests.

The package root keeps core schema/config modules plus thin compatibility
entrypoints such as `run_repair_protocol.py` and `run_staged_llm_pipeline.py`.
Old import paths are aliased in `dataset_clip_wrapper/__init__.py`; new code
should import from the bundle paths directly.

## 0.1 Latest 5-Dataset x 3-Sample Batch Status

The current 5-dataset x 3-sample strict batch artifacts are:

- `dataset_clip_wrapper/output/batch3_latest_trace_base.jsonl`
- `dataset_clip_wrapper/output/batch3_latest_trace_quality_report.json`
- `dataset_clip_wrapper/output/batch3_latest_trace_repair_skipapi_report.json`
- `dataset_clip_wrapper/output/batch3_latest_trace_final_acceptance_report.json`
- `dataset_clip_wrapper/output/batch3_latest_trace_failure_taxonomy_report.json`

This batch uses the latest L2 trajectory schema on the base artifacts and a
strict `--skip-api` repair pass to validate repair graph structure without
calling GPT-OSS for the selector/verifier. The final structural status is:

```text
examples=15
high_l1_all=true
strict_vlm_perception_all=true
l2_trajectory_complete_all=true
repair_subgraph_complete_for_repaired=true
heuristic_final_acceptance_count=0
accepted_all=false
final_l2_status_counts={accepted_strong: 4, needs_more_evidence: 11}
```

The failure taxonomy reports:

```text
failure_stage_counts={repair_selector: 11}
missing_evidence_type_counts={
  commonsense_bridge_without_discriminative_visual_anchor: 3,
  discriminative_visual_evidence_gap: 2,
  long_video_retrieval_or_fine_evidence_gap: 6
}
dataset_failure_counts={video_holmes: 2, videomme: 1, ovo_bench: 2, cg_bench: 3, vrbench: 3}
```

Interpretation: the graph protocol is structurally clean on 15 samples, but the
repair selector/verifier has not been API-exercised for this batch. The next API
run should target only the 11 `repair_selector` failures, not rebuild all 15 L1
graphs.

## 0.2 Latest L2 Recursive Trace Status

The L2 path now records bounded recursive repair as a first-class graph/trace
artifact:

- `reasoning_planner.py` attaches `metadata.l2_trajectory` to the initial
  GPT-OSS L2 rollout. Round 0 records `question + L1 graph -> L2 reasoning
  graph -> verifier status`; weak/rejected outputs end with
  `terminal_status=repair_requested`.
- `run_repair_protocol.py` writes `l2_trajectory` and `repair_subgraph` into
  both the repair report and the L2 verifier artifact. The repair subgraph
  contains nodes for gap diagnosis, repair planning, L1 patching, GPT-OSS
  evidence selection, option verification, optional objective bridge
  verification, and final commit/abstain.
- The trace is POMDP/Semi-MDP-compatible logging, not current MDP training.
  Each round stores compact state snapshots, action/tool records, graph deltas,
  verifier signals, reward proxies, and terminal status.

## 1. Project Architecture

### 1.1 Two Graph Layers

```text
perception / indexing
  -> EvidenceGraph / clue-memory graph

question + clue-memory graph
  -> agent-composed SkillGraphRollout / skill chain

cross_layer_links
  -> uses_evidence, supports_claim, verified_by, ...
```

The clue-memory graph and the skill graph are **not** the same object:

- **Clue-memory graph**: what perception and dataset adapters made available
- **Skill graph**: the agent's executable multi-hop reasoning program

Conceptually unified, engineering-layered: keep runtime objects separate, export
as one heterogeneous graph only after adapters and verifiers are stable.

Verification does not introduce a third graph layer. Current boundary:

- Planner-visible verification is represented as Layer-2 atomic skills
  (`verify_claim_support`, `verify_temporal_social_consistency`,
  `score_hypothesis_support`, `compare_hypotheses`).
- Runtime verifier invariants remain hard acceptance gates over both layers
  (schema validity, evidence-ref existence, hidden-supervision leakage,
  streaming visibility, and retrieval-score-is-not-support).

### 1.2 Two Atomic Skill Bundles

| Bundle | Count | Role in MVP |
|--------|-------|-------------|
| Evidence Graph Construction | 9 | Offline graph builder / audit trace |
| Reasoning Graph Assembly | 25 | 19 core actions + 6 option-level multi-hop/social extensions |

Total: **34 executable atomic skills** in `atomic_skills/`.

Evidence Graph Construction skills:

```text
segment_video_or_select_clip
extract_observation
extract_dialogue_span
detect_entity_mention
resolve_entity_coreference
create_event_node
create_state_node
link_graph_relation
assign_provenance_trust
```

These are **lightweight deterministic Python functions**. They operate on text,
annotations, captions, tool-produced clip schemas, and clip time spans. Raw
video decoding is available through the optional `video_tools` clip-schema
backend, but the atomic functions themselves still receive structured evidence
rather than owning detector / tracker / OCR / ASR implementations.

### 1.3 Two Runtime Modes

| Mode | Purpose | Clip/graph construction |
|------|---------|-------------------------|
| `expert_demo` | Trace-to-skill fitting supervision | Graph often seeded from dataset annotations offline |
| `video_only` | Final evaluation | Agent must discover evidence from video/tools only |

## 2. Datasets

Local root: `/fs/gamma-projects/vlm-robot/datasets`

### 2.1 Core Targets

| Dataset | Tier | Length | Primary use |
|---------|------|--------|-------------|
| Video-Holmes | 1 | Short | Social/causal/temporal reasoning traces |
| CG-Bench | 1 | Long | Clue grounding and evidence retrieval |
| VRBench | 1 | Long | Timestamped multi-step reasoning chains |
| SIV-Bench | 2 | Very short | Social/intent/emotion with weak spans |
| M3-Bench | 2 (deferred) | Long + memory graph | Memory-query rollouts after graph reader |

### 2.2 Non-Primary / Reference

| Dataset | Status |
|---------|--------|
| TIR-Bench | Image QA; not a video rollout source |
| VisualToolBench | Tool-use format reference only |

### 2.3 Training Splits

```text
P0:
  Video-Holmes train: 500 QA
  CG-Bench mini: 500 QA
  VRBench: 100 videos x all QA

P1:
  full Video-Holmes train
  CG-Bench mini full
  SIV-Bench selected categories

P2:
  VRBench full
  CG-Bench full
  M3-Bench after graph reader
```

Split discipline:

- Keep benchmark test splits as evaluation-only. They must not be used for
  expert-demo generation, SFT, reward tuning, motif mining, verifier calibration,
  or GRPO sampling.
- Use the training split for all supervision-producing stages:
  `expert_demo` rollouts, SFT / behavioral cloning targets, corrupted-rollout
  repair data, reward-model or verifier calibration, and policy rollouts for
  verified RL / GRPO.
- Hold out a validation slice from the training split before any labeling
  iteration. Use it for early stopping, prompt/schema changes, reward-weight
  selection, and motif acceptance thresholds.
- Report final numbers on the untouched test split in `video_only` mode where
  possible. In `expert_demo` ablations, clearly mark that the run measures
  trace-fitting quality under supervision rather than final clue discovery.

Recommended P0 split:

```text
train_for_labeling:
  Video-Holmes train first 400 QA
  CG-Bench mini first 400 QA
  VRBench 80 videos x all QA

validation:
  Video-Holmes train held-out 100 QA
  CG-Bench mini held-out 100 QA
  VRBench held-out 20 videos x all QA

test:
  official benchmark test/eval split only
```

Full dataset recipes, labeling rules, and acceptance gates:
[expert-demo-rollouts-from-datasets.md](../atomic-skill-decomposition-and-assembly/expert-demo-rollouts-from-datasets.md).

## 3. Implementation Staging

### Stage A — Expert-Demo Reasoning Assembly (current)

- Build clue-memory graphs from dataset annotations with atomic graph-construction skills
- Fit reasoning skill graphs with teacher/LLM labeler over frozen ontology
- Datasets: Video-Holmes, CG-Bench mini
- **Skips raw-video perception**

Alternative graph-building path:

- Build the same clue-memory graph interface from visible video/tool outputs
  instead of dataset clues.
- Current version: `--clip-schema-backend video_tools` samples frames, computes
  frame-change signals, optionally reads OCR, emits clip schemas, and then uses
  the same graph-composition bridge.
- Intended next version: add VLM captions, ASR/subtitles, entity linking,
  temporal event extraction, and provenance/trust assignment while keeping
  `hidden_supervision` unavailable in `video_only`.
- This path is the `video_only` perception-first counterpart to Stage A's
  annotation-seeded graph construction, so both paths should export compatible
  `evidence_index` / clue-memory graph objects.

### Stage B — Broader Reasoning Coverage

- CG-Bench full, selected VRBench, M3-Bench memory tasks
- Graph construction remains mostly offline

### Stage C — Video-Only Graph Construction

- Activate perception skills as tool-mediated actions
- Automatic captions/ASR, entity linking, graph edges without hidden clues
- First local raw-video tool backend implemented for frame sampling,
  frame-change signals, and optional OCR
- Full VLM/ASR/detector/tracker integration remains future work

## 4. Runnable Code

### 4.1 Current API L1/L2 Probe: Qwen + GPT-OSS

The current `video_only` API path was probed with:

```text
video
  -> qwen/qwen3.5-9b clip schemas
  -> openai/gpt-oss-120b neighbor_vlm_l1 graph composition
  -> L1 query-memory retrieval
  -> openai/gpt-oss-120b L2 reasoning planner
```

Observed outputs from the July 4 probe:

| Dataset | Scope | L1 graph | L1 query-memory | L2 result | Status |
|---------|-------|----------|-----------------|-----------|--------|
| SIV-Bench | one short example | 14 nodes / 16 edges | chose A, gold B | chose B | L2 answer correct, but L1 did not support it strongly |
| CG-Bench | one small 3-clip slice | 101 nodes / 0 edges | chose E, gold E | chose A | L1 option signal correct; L2 degraded it |
| VRBench | one small 4-clip slice | 89 nodes / 6 edges | chose D, gold D | chose A | L1 option signal correct; L2 degraded it |
| Video-Holmes | 12 saved Qwen clip schemas | incomplete | incomplete | incomplete | stopped in serial GPT-OSS L1 compose latency |

Takeaway:

- The current path is feasible: Qwen clip schemas and GPT-OSS neighbor-local L1
  can produce graph nodes and, on some examples, useful semantic edges such as
  `same_entity`, `same_place`, `supports_observation`, and `temporal_next`.
- The latest recursive-trace five-dataset check is structurally clean: all five
  examples have high L1 quality, strict Qwen perception, complete L2
  trajectories, and complete repair subgraphs for repaired examples.
  Final L2 status is three `accepted_strong`, one `accepted_bridge`, and one
  `needs_more_evidence` on Video-Holmes q1. This is a verifier abstention, not
  an unsupported answer commit.
- Long-video L2 now distinguishes direct evidence (`resolved_strong`) from
  visual-anchor-plus-objective-background inference (`accepted_bridge`).
- Throughput is improved by process workers for `neighbor_vlm_l1` graph
  composition, per-clip staged cache/resume for neighbor graph outputs, and
  optional process workers for repair clip-schema generation.

Current remaining risks:

- Strict Qwen-only perception is now clean on the five one-video API check:
  `strict_vlm_perception_all=true`, `fallback_clip_schema_total=0`, and
  `model_error_clip_schema_total=0`. The main remaining perception risk is
  output budget, not fallback logic: failed Qwen rows can be caused by
  `finish_reason=length`, so failed-only retries may need a higher
  `--clip-schema-max-tokens`.
- Prompt/output length must stay visible in reports. The current final report
  records clip-schema and graph-compose prompt chars, approximate tokens,
  output chars, malformed JSON count, timeout count, compact retry count, and
  cache hit/miss counts.
- L2 selector JSON robustness is now handled by compact retry rather than
  heuristic fallback. The Video-Holmes q1 repair run exercised this path:
  malformed full selector output was retried with a compact JSON prompt, and
  the final status remained `needs_more_evidence`.
- Baseline long-video L1 uses full coarse coverage plus selected fine
  neighborhoods; final answer support must come from verified fine evidence or
  explicit objective bridge verification, not retrieval scores.
- L2 should continue to run only from gated evidence packs. `accepted_weak` is
  now treated as a repair-needed intermediate state, not final success.
- Final direct acceptance requires GPT-OSS-backed option-wise verification with
  enough visual refs, confidence, and margin over the next option. The visual
  refs must also come from the GPT-OSS option evidence selector in API runs.
  Rule-only verifier output is diagnostic only.

### 4.2 Smoke Tests (no API key)

Runs the original 28 core atomic skills on a synthetic social-contradiction
example:

```bash
cd /fs/gamma-projects/vlm-robot/video_skills_relaunched
python experiments/smoke_test_atomic_skills.py
```

Validates the 19 core deterministic L2 rollout skills:

```bash
python -m dataset_clip_wrapper.tests.smoke_test_reasoning_rollout
```

Validates the 6 option-level multi-hop/social L2 extensions:

```bash
python -m dataset_clip_wrapper.tests.smoke_test_multi_hop_reasoning_skills
```

### 4.3 Graph Crafting from Video-Holmes (no API key)

`experiments/expert_demo_gpt5mini.py` exposes `load_video_holmes_example()` and
`build_seed_clue_memory_graph()`, which chain graph-construction atomic skills
over dataset annotations.

### 4.4 Dataset Clip Wrapper (no API key by default)

```bash
python -m dataset_clip_wrapper.tests.smoke_test
python -m dataset_clip_wrapper.tests.smoke_test_retrieval
python -m dataset_clip_wrapper.tests.smoke_test_video_only_takein
python -m dataset_clip_wrapper.tests.smoke_test_coarse_fine_graph_crafting

python -m dataset_clip_wrapper.cli \
  --dataset video_holmes --regime short --limit 5 \
  --output dataset_clip_wrapper/output/video_holmes_short.jsonl

# Long video: coarse index only (~98 clips on CG-Bench vs legacy ~574)
python -m dataset_clip_wrapper.cli \
  --dataset cg_bench --regime long --limit 1

# LLM pipeline with M3-style retrieve-gated fine expansion
python -m dataset_clip_wrapper.run_llm_pipeline \
  --dataset cg_bench --regime long --limit 1 \
  --retrieval-topk 2 --clip-schema-max-clips 10

# Offline raw-video tool backend, no OpenRouter key required when graph is deterministic
python -m dataset_clip_wrapper.run_llm_pipeline \
  --dataset video_holmes --regime short --limit 1 \
  --clip-schema-backend video_tools --graph-deterministic

# VRBench video-only coverage probe against hidden timestamp targets
python dataset_clip_wrapper/evaluate_vrbench_video_only_graph.py \
  --limit 1 --clip-schema-max-clips 4 --retrieval-topk 4
```

`smoke_test_video_only_takein.py` is the all-dataset contract test for
`video_only`: each adapter must load a real video, produce clip schemas, craft
a clue-memory graph, and avoid hidden-supervision leakage.

`smoke_test_coarse_fine_graph_crafting.py` is the hierarchical contract test:
long-video datasets build a full-video coarse graph first, expand fine graph
nodes only inside retrieved coarse neighborhoods, and connect fine clips back to
their parent coarse clips with `refines` links. Short-video datasets validate
the fine graph directly.

For five-dataset "entire video" checks, use this coarse/fine contract rather
than full fine-grained scanning. The expected behavior is:

```text
Video-Holmes / VideoMME:
  full short-video fine graph

OVO-Bench:
  streaming/short profile with causal observation handles

CG-Bench / VRBench:
  full-video coarse graph
  -> baseline video_only: question-blind coarse neighborhoods
  -> optional query-time retrieval over the visible question / timestamp anchors
  -> fine graph inside selected neighborhoods
  -> fine --refines--> coarse links
```

The runtime pipeline exposes this as `metadata.coarse_fine_graph`, with
`coarse_graph`, `fine_graph`, and `coarse_to_fine_links` sections. The coarse
graph now includes `coarse_summary` nodes so query-memory experiments can search
full-video handles even when fine perception has only been expanded in selected
neighborhoods. Final answer support should still come from discovered fine
evidence rather than a coarse retrieval score alone.

The current final acceptance report can be regenerated with:

```bash
python -m dataset_clip_wrapper.report_final_acceptance \
  --quality-report dataset_clip_wrapper/output/rerun5_quality_report_strict_qwen.json \
  --repair-report dataset_clip_wrapper/output/repair_long_objective_bridge_api_v7_cg_rebuild_report.json \
  --repair-report dataset_clip_wrapper/output/repair_long_objective_bridge_api_v7_vr_rebuild_report.json \
  --output dataset_clip_wrapper/output/rerun5_final_acceptance_report.json
```

Current strict final acceptance summary:

```text
examples=5
high_l1_all=true
accepted_all=true
strict_vlm_perception_all=true
fallback_clip_schema_total=0
model_error_clip_schema_total=0
final_l2_status_counts={accepted_strong: 4, accepted_bridge: 1}
repair_needed_after_final=0
graph_compose_cache_hits=205
graph_compose_cache_misses=105
repair_plan_calls=5
repair_l2_verifier_calls=1
```

The latest strict repair rerun is stored in
`dataset_clip_wrapper/output/repair_long_objective_bridge_api_v9_strict_report.json`.
It resolves CG-Bench with direct repaired evidence (`resolved_strong`) and
VRBench with visual anchors plus objective background bridge (`accepted_bridge`).
The repair report now includes planner/selector/bridge telemetry. Two concrete
fixes were needed:

- clue planner compact retry, because GPT-OSS can return malformed or truncated
  JSON for long repair prompts;
- option-aware repair verifier evidence retrieval, because negative repair
  evidence can otherwise outrank the positive option-specific visual clue.
- all-regime repair routing, because short/streaming `accepted_weak` or
  `rejected` examples were previously skipped by the long-only repair runner.
- GPT-OSS option evidence-pack selection before verification, because
  token-overlap ref selection could still decide what the verifier saw.

Five-dataset x three-sample strict batch:

```text
examples=15
strict_qwen_only=15/15
high_l1=15/15
fallback_clip_schema_total=0
model_error_clip_schema_total=0
l2_status_counts={accepted_strong: 4, accepted_weak: 8, rejected: 3}
repair_needed=11/15
```

Interpretation: strict video-only perception and L1 clue graph construction now
scale beyond the one-video demo. L2 acceptance does not yet scale without
targeted repair: weak/rejected batch examples should be routed to repair rather
than counted as accepted.

Latest repair-protocol code path:

- `run_repair_protocol.py` defaults to all five datasets and all regimes
  (`short`, `streaming`, `long`).
- short/streaming repair starts with `existing_l1_option_verification`, which
  constructs option-specific evidence packs from the existing L1 graph without
  new Qwen perception calls or GPT-OSS clue-planner calls.
- long repair still supports local coarse-neighborhood expansion, full reroute,
  process-worker Qwen repair clip schemas, staged cache/resume, and objective
  background bridge verification.
- repaired reports now expose `option_evidence_packs` with positive refs,
  negative refs, verifier decision, confidence, and `reason_short`.
- in API runs, `option_evidence_packs` are selected by GPT-OSS from a compact L1
  evidence table before `verify_claim_support` runs; the legacy lexical selector
  is restricted to no-API diagnostics or explicit `--allow-lexical-fallback`.
- rule-only checks can validate report shape, but only GPT-OSS evidence-pack
  selection plus GPT-OSS `verify_claim_support`, or the objective bridge
  verifier, can produce final acceptance.

Long-video defaults (`ClipPolicyConfig.for_regime(LONG)`):

- `coarse_window_s=30`, `fine_window_s=8`, `index_fine_expansion=retrieval_gated`
- `ClipRetrievalConfig.topk=2`; default `video_only` L1 is question-blind
  sequential coverage, while `--query-time-retrieval` can use the visible
  question plus timestamp-anchor expansion
- Index layer stores coarse clips only; fine windows expand inside retrieved parents for perception / LLM pipeline

L1 query-memory diagnostics:

```bash
python dataset_clip_wrapper/evaluate_l1_query_memory.py \
  --topk 5 \
  --output dataset_clip_wrapper/output/l1_query_memory_eval.json \
  dataset_clip_wrapper/output/cg_bench_video_only_qwen_gptoss_entire.jsonl
```

This report checks question-only evidence hits, option scores, selected coarse
indices, and whether an L2 rollout copied hidden gold answer text. For valid
`video_only` L2 experiments, `question.answer` is evaluation-only and must not
be shown to the L2 planner.

See [two-layer graph schema](../docs/two-layer-graph-schema.md) and
[implementation status](../docs/implementation-status.md).

### 4.6 Expert Demo with LLM Labeling (API key required)

Reads `OPENROUTER_API_KEY` from `/fs/gamma-projects/vlm-robot/keys.py` or from
the environment. Default model: `openai/gpt-5-mini`.

```bash
# Toy example
python experiments/expert_demo_gpt5mini.py --dataset toy

# Video-Holmes
python experiments/expert_demo_gpt5mini.py --dataset video_holmes --split train --index 0
```

The LLM fits a `SkillGraphRollout` over the seed clue-memory graph. It does
not perform raw video perception.

### 4.4 Toy Two-Layer Graph Experiment (API key required)

```bash
python experiments/toy_graph_skill_reasoning.py
```

Uses synthetic perceived-video notes, asks the model to emit both graphs, and
runs a local verifier on cross-layer bindings.

## 5. What Is Not Implemented Yet

| Item | Notes |
|------|-------|
| Dataset adapters for CG-Bench / VRBench / SIV-Bench | Implemented in `dataset_clip_wrapper/` |
| Canonical JSONL export (`data/canonical_examples/`) | Use `dataset_clip_wrapper/cli.py` |
| Embedding-based coarse retrieval (M3-style) | Lexical gate in `clip_retrieval.py`; embedding API not wired |
| `shot_boundary` / `scene_boundary` / `adaptive` strategies | Schema enum only |
| Port of legacy `Video_Skills` segmenter | Exists in sibling repo, not wired to relaunch |
| Local raw-video frame tool backend | Implemented as `video_tool_backend.py`; produces same clip-schema fields as Qwen path |
| VRBench video-only graph-quality probe | Implemented as `evaluate_vrbench_video_only_graph.py`; current bottleneck is reaching the right long-video temporal neighborhood |
| Full raw VLM caption / ASR / object tracking stack | Stage C future work |
| `create_semantic_memory_node` | Discussed in early skill drafts; not in final 9-skill set |
| M3-Bench memory graph reader | Blocker for M3 rollouts |
| Controller training / verified RL | Design only |
| `--graph-only` batch exporter | Not yet added |

## 5.1 Training Feasibility Assessment

For an ICLR submission, the broad version should remain the north-star claim:
learn a compact controller that turns raw video/question inputs into
verifiable, evidence-grounded skill graphs under `video_only` constraints.
The paper should not present Stage A as the final problem. Stage A is the
supervision and ablation scaffold that makes the broad claim measurable.

The most plausible first technical success case is narrower: train the
controller to assemble typed reasoning graphs over a mostly fixed clue-memory
graph. This should be feasible if the expert rollouts are high precision,
because the action space is small, the verifier can reject malformed or
ungrounded traces, and the evaluation can measure process quality rather than
only final answer text.

The broad `video_only` version is the right ICLR-facing problem but the risky
part experimentally. There the controller must both discover the right evidence
and compose a reasoning graph, so failures in captioning, retrieval, temporal
localization, entity linking, and reasoning all compound. The submission should
therefore use a staged evidence ladder: show the full `video_only` setting, then
use Stage A/B ablations to identify whether failures come from evidence
discovery, graph assembly, verification, or repair.

Suggested paper positioning:

```text
Main claim:
  Video reasoning should be learned as verifier-grounded skill-graph control
  over discovered evidence, not as free-form chain-of-thought imitation.

Primary target:
  video_only evidence discovery + reasoning graph assembly.

Training scaffold:
  expert_demo traces from train split provide SFT/offline-RL warm-up only.

Key ablation:
  prebuilt evidence graph vs video_only evidence discovery.
```

Expected working path:

```text
1. expert_demo graph fitting works on Video-Holmes / CG-Bench train
2. SFT learns skill choice, argument binding, and evidence-role structure
3. verifier-grounded repair improves evidence F1 and trace validity
4. GRPO / verified RL improves sampled rollouts after SFT warm-up
5. video_only improves when evidence discovery is trained/evaluated separately
6. full broad claim is supported if video_only gains survive held-out test
```

Expected failure modes:

- Expert rollouts are too teacher-specific, so SFT learns formatting but not
  reusable assembly.
- The verifier mostly checks schema, not semantic support, so RL optimizes easy
  surface signals.
- Retrieval recall is too low in `video_only`, making the reasoning controller
  look worse even when its policy is reasonable.
- Motifs overfit common train patterns and quietly leak dataset-specific
  shortcuts unless accepted on held-out validation examples.

Minimum credible ICLR package:

- A broad `video_only` evaluation with no hidden clue intervals, answers, or
  official reasoning visible to the agent.
- A supervised `expert_demo` warm-up built only from train split examples.
- A controlled ablation where the same controller receives a prebuilt evidence
  graph, measuring pure reasoning-graph assembly.
- Evidence discovery metrics such as clue recall, evidence precision, timestamp
  error, and hidden-supervision leakage rate.
- Reasoning graph metrics such as schema validity, evidence-ref validity,
  claim-support precision, repair success, answer accuracy, and tool cost.

## 6. Labeling Policy Summary

| Use rules (deterministic) | Use model (gpt-5-mini / gpt-oss-120) |
|---------------------------|--------------------------------------|
| Parse MCQ answers and timestamps | Trace segmentation |
| Map CG qid to clue clips | Skill fitting to frozen ontology |
| Parse SRT into subtitle chunks | Evidence-role labeling |
| Create initial EvidenceRef objects | Repair-target generation |
| Schema and format validation | |

Never invent new atomic skills during labeling.

## 7. Paper Scope Decisions

Documented in `problem-formulation-zh.html` and preserved here for implementation alignment:

- Paper 1 focuses on decomposition/assembly + query/verify/repair over teacher-built or dataset-seeded memory
- Store/update/merge/forget/streaming writer-reasoner split deferred to paper 2
- Streaming remains schema-compatible via `clip_policy.online` and `observation_end_s`
- Core evaluation trio: Video-Holmes + SIV-Bench + VRBench; CG-Bench and M3-Bench optional

## 8. Related Documents

- [MDP formulation](mdp-formulation.md)
- [Clip processing policy](clip-processing-policy.md)
- [Unified video skill schema](unified-video-skill-schema.md)
- [Atomic skills v1](../atomic-skill-decomposition-and-assembly/atomic-skills-v1.md)
- [Expert demo rollouts from datasets](../atomic-skill-decomposition-and-assembly/expert-demo-rollouts-from-datasets.md)
