# Implementation Status

Last updated: 2026-07-06

This is the current entry point for implemented L1/L2 graph work in
`video_skills_relaunched`. Older probe notes, rerun logs, and pre-P5 batch
status have been archived in
[`docs/legacy/implementation-status-pre-p5.md`](legacy/implementation-status-pre-p5.md).

## Current Version

The latest validated run is the **P5 five-dataset x three-sample video-only
L1/L2 + repair batch**.

Primary artifacts:

- `dataset_clip_wrapper/output/batch3_p5_final_acceptance_api_report.json`
- `dataset_clip_wrapper/output/batch3_p5_failure_taxonomy_api_report.json`
- `dataset_clip_wrapper/output/batch3_p5_audit_guided_repair_api_report.json`
- `dataset_clip_wrapper/output/batch3_p5b_existing_l1_semantic_repair_api_report.json`
- `dataset_clip_wrapper/output/batch3_p5c_verifier_calibration_api_report.json`
- `dataset_clip_wrapper/output/expert_demos/batch3_p5_video_only_expert_demos.jsonl`
- `dataset_clip_wrapper/output/expert_demos/batch3_p5_video_only_expert_demo_quality.json`
- `dataset_clip_wrapper/output/expert_demos/batch3_p5_video_only_expert_demos_compact.jsonl`
- `dataset_clip_wrapper/output/expert_demos/batch3_p5_video_only_expert_demo_quality_compact.json`

Current P5 final acceptance:

```text
examples=15
datasets={video_holmes, videomme, ovo_bench, cg_bench, vrbench}
l1_quality_counts={high: 15}
strict_vlm_perception_all=true
l2_trajectory_complete_all=true
repair_subgraph_complete_for_repaired=true
heuristic_final_acceptance_count=0
fallback_clip_schema_total=0
model_error_clip_schema_total=0
final_l2_status_counts={accepted_strong: 12, needs_more_evidence: 3}
repair_applied=11
repair_needed_after_final=3
```

This supersedes the earlier `batch3_latest_trace_*` status, which improved
from 4/15 structural no-API acceptance to 8/15 after the first API repair pass.
The P5 path adds audit-guided L1 semantic repair and verifier calibration,
raising final strong acceptance to 12/15 without heuristic final acceptance.

## What Is Implemented

### Package Bundles

`dataset_clip_wrapper/` is physically split into implementation bundles:

- `perception/`: clip policy, Qwen/video-tools clip schemas, subtitles, video probes, OpenRouter client.
- `l1_clue_graph/`: clue-memory graph extraction, graph composition, retrieval, L1 gating.
- `l2_reasoning_graph/`: reasoning planner, deterministic rollout, bounded recursive trace, local fault repair.
- `verification/`: repair protocol, evidence audit, quality reports, final acceptance, runtime verifier.
- `runners/`: staged and non-staged API pipelines.
- `expert_demos/`: video-only expert-demo export.
- `manifests/`: split-aware training manifest builder.
- `training/`: compact demo to `ReasoningTrace` / SFT chat export.
- `tests/`: smoke tests.

Top-level `dataset_clip_wrapper/*.py` command modules remain as compatibility
entrypoints. New code should import from the bundle paths directly.

### L1

Layer 1 is a question-blind clue-memory graph in `video_only` mode. The current
path is:

```text
video
  -> qwen/qwen3.5-9b or local video_tools clip schemas
  -> openai/gpt-oss-120b neighbor_vlm_l1 graph composition
  -> coarse/fine reference graph for long videos
  -> answerability diagnostics and optional bounded L1 repair patches
```

Current P5 batch status:

- all 15 examples have `L1_quality.grade=high`;
- no hidden-memory leakage is counted in final acceptance;
- strict Qwen perception produced no fallback clip schemas and no model-error clip schemas;
- repaired semantic L1 nodes must cite visual `support_refs` and remain visible as `video_only` evidence, not answer shortcuts.

### L2

Layer 2 consumes `question + L1 clue_memory_graph` and builds a
question-conditioned reasoning graph. It is now a **bounded recursive repair
trace**, not an unbounded agent loop.

Current L2 records include:

- initial GPT-OSS reasoning plan and skill execution trace;
- verifier-gated final answer or abstention;
- `metadata.l2_trajectory` with compact state/action/observation rounds;
- repair reports with `repair_subgraph` nodes for gap diagnosis, repair plan,
  L1 patching, option evidence selection, verifier decision, optional bridge,
  and final commit/abstain.

`accepted_weak` is not treated as final success. Weak/rejected outputs route to
repair or remain `needs_more_evidence`.

## Current P5 Outcomes

Accepted strong examples:

```text
video_holmes:train:oZ4pa_5R0nY:q1
video_holmes:train:oZ4pa_5R0nY:q3
videomme:streambridge_demo:0
videomme:streambridge_demo:1
videomme:streambridge_demo:2
ovo_bench:streaming_tiny_000_00
ovo_bench:streaming_tiny_000_01
ovo_bench:streaming_tiny_000_02
cg_bench:14
cg_bench:17
vrbench:TZk_p-q8Fzo:qa2
vrbench:TZk_p-q8Fzo:qa3
```

Remaining `needs_more_evidence` examples:

```text
video_holmes:train:oZ4pa_5R0nY:q2
cg_bench:19
vrbench:TZk_p-q8Fzo:qa1
```

Latest failure taxonomy:

```text
failure_stage_counts={repair_verifier: 3}
missing_evidence_type_counts={
  commonsense_bridge_without_discriminative_visual_anchor: 1,
  long_video_retrieval_or_fine_evidence_gap: 2
}
dataset_failure_counts={video_holmes: 1, cg_bench: 1, vrbench: 1}
repairable_failure_count=3
needs_dataset_replacement_count=0
```

Interpretation: the bottleneck has moved from L1 construction and trace shape
to evidence sufficiency / verifier acceptance on three hard cases. The current
protocol correctly abstains instead of committing weak answers.

## Expert Demo Seed

The current video-only expert-demo exporter consumes the P5 final acceptance
report and writes sanitized training rows.

Current compact seed quality:

```text
examples=15
training_candidate_count=12
abstain_candidate_count=3
demo_type_counts={direct_strong: 4, repair_strong: 8, abstain_needs_more_evidence: 3}
visible_gold_key_leak_count=0
training_views={compact}
compact_evidence_node_count=1200
strict_vlm_perception_all=true
high_l1_all=true
heuristic_final_acceptance_count=0
```

The positive training seed is therefore 12 video-only trajectories, plus 3
abstain trajectories for repair/uncertainty behavior.

The split-aware manifest builder is implemented:

- `dataset_clip_wrapper/manifests/build_training_manifests.py`
- compatibility entrypoint: `python -m dataset_clip_wrapper.build_training_manifests`
- smoke test: `dataset_clip_wrapper/tests/smoke_test_training_manifests.py`

It groups examples by `dataset:video_id`, strips visible gold question fields,
and records hidden supervision only as non-inference bookkeeping.

## Current Commands

Smoke checks:

```bash
python -m dataset_clip_wrapper.tests.smoke_test_module_bundles
python -m dataset_clip_wrapper.tests.smoke_test_two_layer_schema
python -m dataset_clip_wrapper.tests.smoke_test_reasoning_rollout
python -m dataset_clip_wrapper.tests.smoke_test_l2_recursive_trace
python -m dataset_clip_wrapper.tests.smoke_test_export_expert_demos
python -m dataset_clip_wrapper.tests.smoke_test_training_manifests
python -m dataset_clip_wrapper.tests.smoke_test_trace_adapter
```

Local/offline clip-schema smoke:

```bash
python -m dataset_clip_wrapper.run_llm_pipeline \
  --dataset video_holmes --regime short --limit 1 \
  --clip-schema-backend video_tools --clip-schema-max-clips 1 \
  --graph-deterministic
```

Report the current P5 final acceptance:

```bash
python - <<'PY'
import json
from pathlib import Path

path = Path("dataset_clip_wrapper/output/batch3_p5_final_acceptance_api_report.json")
print(json.dumps(json.loads(path.read_text())["summary"], indent=2))
PY
```

Export compact expert demos from a final acceptance report:

```bash
python -m dataset_clip_wrapper.export_expert_demos \
  --final-acceptance-report dataset_clip_wrapper/output/batch3_p5_final_acceptance_api_report.json \
  --output dataset_clip_wrapper/output/expert_demos/batch3_p5_video_only_expert_demos_compact.jsonl \
  --quality-output dataset_clip_wrapper/output/expert_demos/batch3_p5_video_only_expert_demo_quality_compact.json \
  --training-view compact \
  --max-l1-nodes 80
```

Export SFT-ready traces after compact demos:

```bash
python -m dataset_clip_wrapper.export_reasoning_traces \
  --input dataset_clip_wrapper/output/expert_demos/batch3_p5_video_only_expert_demos_compact.jsonl \
  --output-dir dataset_clip_wrapper/output/training/batch3_p5_compact
```

## Current Gaps

- Scale P5 from 15 seed examples to split-controlled train/dev/test manifests.
- Run compact demo export and trace/SFT export on the expanded training split.
- Add a controller SFT trainer entrypoint and dry-run it on sanitized compact demos.
- Implement dataset-aware training reward: evaluation remains hard 0/1
  correctness/acceptance/leakage metrics, while training may use RLVR-style
  progressive rewards weighted by supervision density. Hidden GT stays outside
  `video_only` L1/L2 inputs in both cases.
- Improve the three remaining `needs_more_evidence` cases with another bounded
  repair round or stronger discriminative visual anchors.
- Add embedding-based coarse retrieval if lexical/sequential retrieval remains
  the long-video bottleneck.
- Keep official test splits evaluation-only; do not use them for expert-demo
  generation, SFT, reward tuning, motif mining, verifier calibration, or GRPO
  sampling.

## Related Documents

- [Repository bundle map](repo-bundle-map.md)
- [Repository cleanup audit](repo-cleanup-audit.md)
- [Two-layer graph schema](two-layer-graph-schema.md)
- [Clip processing policy](clip-processing-policy.md)
- [MDP formulation](mdp-formulation.md)
- [Unified video skill schema](unified-video-skill-schema.md)
- [Atomic skills v1](../atomic-skill-decomposition-and-assembly/atomic-skills-v1.md)
- [Expert demo rollouts from datasets](../atomic-skill-decomposition-and-assembly/expert-demo-rollouts-from-datasets.md)
- [Legacy implementation status](legacy/implementation-status-pre-p5.md)
