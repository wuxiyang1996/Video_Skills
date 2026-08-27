# Dataset Clip Wrapper File Purposes

Last updated: 2026-07-24

This document explains why each `dataset_clip_wrapper` folder/file exists.
Use it when adding new modules or deciding where a change belongs.

Functional flow:

```text
adapters + perception
  -> l1_clue_graph
  -> l2_reasoning_graph
  -> verification (repair / accept / abstain)
  -> motifs (optional post-hoc priors)
training/ exports SFT transitions across L1, L2, repair, verifier, motif
runners/ orchestrate the end-to-end path
```

## Package Root

| Path | Purpose |
|------|---------|
| `__init__.py` | Public package exports and temporary compatibility aliases for old import paths. New code should import from bundle subpackages directly. |
| `README.md` | How to run the wrapper, staged API pipeline, repair protocol, and benchmark checks. |
| `FILE_PURPOSES.md` | This file-purpose map. |
| `module_bundles.py` | Machine-readable bundle registry used by `tests/smoke_test_module_bundles.py`. |
| `schemas.py` | Dataclasses/enums for wrapper config, clip policies, model configs, regimes, and canonical wrapper settings. |
| `dataset_graph_presets.py` | Dataset/regime defaults for clip policy, retrieval, hidden-source policy, and benchmark profiles. |
| `pipeline.py` | Lightweight canonical-example builder that does not own API/VLM orchestration. |
| `cli.py` | Simple CLI wrapper around canonical-example export. |
| `build_training_manifests.py` | Compatibility entrypoint for `manifests/build_training_manifests.py`. |
| `export_reasoning_traces.py` | Compatibility entrypoint for `training/trace_adapter.py`. |
| `build_motif_bank.py` | Compatibility entrypoint for `motifs/build_motif_bank.py`. |
| `run_llm_pipeline.py` | Compatibility entrypoint for `runners/run_llm_pipeline.py`. |
| `run_staged_llm_pipeline.py` | Compatibility entrypoint for `runners/run_staged_llm_pipeline.py`. |
| `run_repair_protocol.py` | Compatibility entrypoint for `verification/run_repair_protocol.py`. |
| `report_l1_l2_quality.py` | Compatibility entrypoint for `verification/report_l1_l2_quality.py`. |
| `report_final_acceptance.py` | Compatibility entrypoint for `verification/report_final_acceptance.py`. |
| `report_failure_taxonomy.py` | Compatibility entrypoint for `verification/report_failure_taxonomy.py`. |
| `report_evidence_audit.py` | Compatibility entrypoint for `verification/report_evidence_audit.py`. |
| `export_expert_demos.py` | Compatibility entrypoint for `expert_demos/export_expert_demos.py`. |
| `export_l1_builder_sft.py` | Compatibility entrypoint for `training/l1_builder_sft_adapter.py`. |
| `export_l1_patch_sft.py` | Compatibility entrypoint for `training/l1_patch_sft_adapter.py`. |
| `export_l2_retrieval_sft.py` | Compatibility entrypoint for `training/l2_retrieval_sft_adapter.py`. |
| `export_motif_sft.py` | Compatibility entrypoint for `training/motif_sft_adapter.py`. |
| `export_stepwise_sft.py` | Compatibility entrypoint for `training/stepwise_sft_adapter.py`. |
| `export_verifier_sft.py` | Compatibility entrypoint for `training/verifier_sft_adapter.py`. |
| `collect_sft_snapshot.py` | Compatibility entrypoint for `training/collect_sft_snapshot.py`. |
| `retrofit_l2_trajectory.py` | Compatibility entrypoint for `verification/retrofit_l2_trajectory.py`. |
| `output/` | Generated API outputs, staged caches, repair artifacts, and reports. Only `.gitkeep` should be tracked. |

## `adapters/`

Dataset readers. These modules convert benchmark-specific files into
`RawDatasetItem` records. They should not build L1/L2 graphs or call models.

| Path | Purpose |
|------|---------|
| `adapters/base.py` | Adapter base class and `RawDatasetItem` contract. |
| `adapters/video_holmes.py` | Video-Holmes reader. |
| `adapters/cg_bench.py` | CG-Bench reader. |
| `adapters/vrbench.py` | VRBench reader. |
| `adapters/siv_bench.py` | SIV-Bench reader. |
| `adapters/streaming_video.py` | StreamBridge-style OVO-Bench and VideoMME readers. |
| `adapters/__init__.py` | Adapter registry and `get_adapter()`. |

## `perception/`

Video/clip input tools. These modules produce structured clip schemas or clip
references. They should never commit answers.

| Path | Purpose |
|------|---------|
| `perception/backbone.py` | Generic perception-backbone helpers and API-key loading for perception calls. |
| `perception/clip_policy.py` | Short/long/streaming clip segmentation and `ClipSpan` utilities. |
| `perception/clip_schema.py` | Qwen clip-schema producer and clip-level VLM schema contract. |
| `perception/openrouter_client.py` | OpenRouter/OpenAI-compatible JSON chat client and total timeout handling. |
| `perception/subtitles.py` | Subtitle parsing helpers. Audio/ASR is not part of the current visual-only scope unless explicitly enabled. |
| `perception/video_probe.py` | Lightweight video duration probing. |
| `perception/video_tool_backend.py` | Local `video_tools` perception backend for smoke/offline checks. |

## `l1_clue_graph/`

Layer-1 clue memory. L1 stores visible, question-agnostic video evidence and
retrieval structure. It may mark missing visual clues, but it must not turn
commonsense/background facts into visual evidence.

| Path | Purpose |
|------|---------|
| `l1_clue_graph/clue_memory.py` | Extracts `ClueMemoryGraph` from canonical examples and creates L2 rollout shells. |
| `l1_clue_graph/graph_composer.py` | Builds semantic L1 nodes/edges from clip schemas, including neighbor-local GPT-OSS composition. |
| `l1_clue_graph/graph_plan_validator.py` | Validates/coerces graph-composition skill plans. |
| `l1_clue_graph/clip_retrieval.py` | Coarse/fine clip retrieval over visual summaries and time anchors. |
| `l1_clue_graph/gate_l1_for_l2.py` | L1 quality/answerability gate before expensive L2 calls. |
| `l1_clue_graph/skill_graph_bridge.py` | Converts canonical examples into skill-graph-compatible structures. |

## `l2_reasoning_graph/`

Layer-2 reasoning. L2 consumes `question + L1 graph`, builds reasoning rollouts,
records bounded recursive trajectories, and asks verification/repair to resolve
weak evidence.

| Path | Purpose |
|------|---------|
| `l2_reasoning_graph/reasoning_planner.py` | GPT-OSS question-conditioned reasoning planner and execution over atomic skills. |
| `l2_reasoning_graph/reasoning_rollout.py` | Deterministic/base reasoning rollout builder. |
| `l2_reasoning_graph/l2_recursive_trace.py` | POMDP/Semi-MDP-compatible trajectory and repair-subgraph encoding. |
| `l2_reasoning_graph/fault_repair.py` | Local trace-level fault localization/repair for failed skill steps. |

## `verification/`

Repair, verification, and reporting. This bundle decides whether L2 can commit,
bridge, or abstain.

| Path | Purpose |
|------|---------|
| `verification/run_repair_protocol.py` | Bounded repair protocol: gap diagnosis, repair clip selection, L1 patch, audit-guided semantic patch nodes, GPT-OSS option evidence selector, verifier, bridge verifier. |
| `verification/runtime_verifier.py` | Runtime schema/evidence/leakage invariants. |
| `verification/evaluate_l1_query_memory.py` | L1 query-memory quality and answerability diagnostics. |
| `verification/evaluate_vrbench_video_only_graph.py` | VRBench-specific video-only graph evaluation helper. |
| `verification/report_l1_l2_quality.py` | Batch L1/L2 quality report with trajectory completeness fields. |
| `verification/report_final_acceptance.py` | Merges base quality and repair reports into final acceptance status. |
| `verification/report_failure_taxonomy.py` | Classifies non-accepted examples by failure stage, missing evidence type, repairability, and dataset-fit risk. |
| `verification/report_evidence_audit.py` | Uses GPT-OSS to audit non-accepted examples from packed L1/L2/repair evidence instead of heuristic labels. |
| `verification/retrofit_l2_trajectory.py` | Adds latest L2 trajectory metadata to older JSONL artifacts without re-running perception. |

## `expert_demos/`

Training-data export. This bundle turns verified video-only L1/L2/repair traces
into expert-demo candidates while keeping hidden supervision out of visible
inputs.

| Path | Purpose |
|------|---------|
| `expert_demos/export_expert_demos.py` | Exports direct, repaired, bridge, and abstain trajectories plus a demo quality report. |

## `manifests/`

Split-aware dataset manifests. This bundle decides which examples may be used
for demo gathering, prompt/dev tuning, or held-out evaluation.

| Path | Purpose |
|------|---------|
| `manifests/build_training_manifests.py` | Builds deterministic train/dev/test manifests grouped by video id to prevent cross-split video leakage. |

## `training/`

Controller-training exports. This bundle does not perform new reasoning; it
maps verified expert-demo trajectories into canonical runtime/training formats.

| Path | Purpose |
|------|---------|
| `training/trace_adapter.py` | Converts compact L1/L2 expert demos into `video_skills.contracts.ReasoningTrace` JSONL and compact chat-SFT JSONL. |
| `training/__init__.py` | Public exports for training adapters. |

## `motifs/`

Motif extraction and management. Deterministic bank/lifecycle mining lives
alongside a Qwen3.5/GPT-OSS motif agent that proposes, curates, and stores
expandable atomic graph templates rather than callable skill agents.

| Path | Purpose |
|------|---------|
| `motifs/agent.py` | High-level Motif Agent orchestration with `hybrid`, `llm`, and `deterministic` modes. |
| `motifs/llm_agent.py` | Qwen3.5/GPT-OSS propose-and-curate adapter for reusable L1/L2 motif candidates. |
| `motifs/miner.py` | Deterministic L1/L2 graph miner used by the bank/lifecycle path. |
| `motifs/instance_miner.py` | Agent-facing seed/fallback extractor for trajectory-round and repair-subgraph motif instances. |
| `motifs/registry.py` | JSONL-backed motif bank with support, pass-rate, dataset/task coverage, agent metadata, and expansion templates. |
| `motifs/promotion.py` | Support/pass-rate/dataset-coverage gates for promoting candidates. |
| `motifs/expansion.py` | Converts promoted motifs into future planning-prior objects; motifs still expand before execution. |
| `motifs/canonicalize.py` | Canonical signature helpers that remove surface labels from motif ids. |
| `motifs/build_motif_bank.py` | CLI for building a motif bank from accepted L1/L2 artifacts; defaults to `--agent-mode hybrid`. |

## `training/`

Controller SFT adapters and collection. One adapter family per specialist
(L1 builder/patch, L2 retrieval, repair stepwise, verifier, motif). Chat SFT
rows are MDP-style transitions, not one-shot graph dumps.

| Path | Purpose |
|------|---------|
| `training/l1_builder_sft_adapter.py` | Atomic L1 create-node/edge/segment/skip exports. |
| `training/l1_patch_sft_adapter.py` | Repair-triggered L1 patch exports. |
| `training/l2_retrieval_sft_adapter.py` | Coarse retrieval / recovery action exports. |
| `training/l2_specialist_sft_adapter.py` | Expanded L2 specialist rows for five-LoRA packages. |
| `training/verifier_sft_adapter.py` | Auxiliary verifier supported/insufficient exports. |
| `training/motif_sft_adapter.py` | Motif lifecycle SFT exports. |
| `training/motif_evidence_sft_adapter.py` | Motif evidence-ref audit exports. |
| `training/stepwise_sft_adapter.py` | L2/repair round stepwise exports. |
| `training/repair_report_stepwise_sft_adapter.py` | Repair-report-derived stepwise exports. |
| `training/collect_sft_snapshot.py` | Snapshot collector that gathers gated controller JSONL bundles. |
| `training/build_sft_splits.py` | Strict train/dev split builder with controller mixture gates. |
| `training/build_split_manifest.py` | Freeze source-video roles (`sft_seed` / `opd_pool` / `grpo_pool` / `dev_tune` / `heldout_test`). |
| `training/build_specialist_sft_v4.py` | Filter v3 five_lora → v4 using split_manifest (`sft_seed`/`dev_tune` only). |
| `training/evaluate_sft_package_gates.py` | Preflight gates for five-LoRA SFT packages before training. |
| `training/evaluate_lora_sft_gates.py` | Post-SFT warm-up gates vs base-9B and majority-action baselines. |
| `training/train_lora_sft.py` | Small LoRA SFT trainer entrypoint. |
| `training/sft_common.py` | Shared JSONL IO, leakage gates, and report helpers. |

## `runners/`

End-to-end orchestration. Runners may connect adapters, perception, L1, L2, and
verification, but implementation details should remain in the bundle modules.

| Path | Purpose |
|------|---------|
| `runners/llm_pipeline.py` | Non-staged API/VLM pipeline for building enriched examples. |
| `runners/run_llm_pipeline.py` | CLI for `llm_pipeline.py`. |
| `runners/run_staged_llm_pipeline.py` | Resumable staged Qwen + GPT-OSS pipeline with per-stage artifacts/cache. |

## `tests/`

Executable smoke tests, grouped by function. Implementation modules live under
bundle subpackages; root `tests/smoke_test_*.py` files are thin compatibility
shims so `python -m dataset_clip_wrapper.tests.smoke_test_*` keeps working.

| Subpackage | Covers |
|------------|--------|
| `tests/core/` | Module-bundle registry, two-layer schema, general wrapper smoke |
| `tests/perception/` | `video_tools` backend, video-only take-in contract |
| `tests/l1/` | Graph compose, retrieval, coarse/fine profiles, plan validation |
| `tests/l2/` | Reasoning rollout, multi-hop skills, recursive trace, executor, fault repair |
| `tests/verification/` | Long-video retrieval repair |
| `tests/motifs/` | Motif agent propose/curate fallback |
| `tests/training/` | Expert-demo export and training manifests |

## Placement Rules

- New video/clip input code goes in `perception/`.
- New question-blind evidence-memory code goes in `l1_clue_graph/`.
- New reasoning rollout or trajectory code goes in `l2_reasoning_graph/`.
- New answer verification, evidence repair, or acceptance reporting goes in
  `verification/`.
- New training-data/demo export code goes in `expert_demos/`.
- New controller-training trace/chat adapters and SFT snapshot collection go in `training/`.
- New split or dataset manifest code goes in `manifests/`.
- New motif mining, promotion, registry, or expansion code goes in `motifs/`.
- New orchestration commands go in `runners/` plus a thin compatibility
  entrypoint at the package root only when an existing public command needs it.
- New smoke tests go under `tests/<bundle>/` and must be registered in
  `module_bundles.py` (plus a root shim if a public `-m` path is required).
