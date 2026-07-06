# Repository Structure

Last updated: 2026-07-06

This branch starts from `backup/pre-merge-l1l2-training-20260706`, the clean
L1/L2 relaunch base before the larger `Video_Skills` runtime and legacy
game-agent directories were merged in.

The active branch for this cleanup is `clean/l1l2-from-pre-merge-20260706`.
The goal is to keep the relaunch focused on video-only L1/L2 graph construction,
verification, repair, expert-demo export, and future controller-training data.

## Active Path

| Path | Status | Purpose |
|------|--------|---------|
| `dataset_clip_wrapper/` | Active | Dataset adapters, video-only perception, L1 clue graph, L2 reasoning/repair/verification, manifests, and expert demo export. |
| `atomic_skills/` | Active | Frozen evidence-graph and reasoning-graph atomic skill basis used by L1/L2 graph protocols. |
| `video_skills/` | Active | Relaunch package namespace for video-skill schema/runtime work present in this clean base. |
| `visual_grounding/` | Active/reference | Grounding code kept in this base; current L1 path still lives under `dataset_clip_wrapper/`. |
| `docs/` | Active docs | L1/L2 graph schema, implementation status, MDP framing, bundle map, and cleanup decisions. |
| `atomic-skill-decomposition-and-assembly/` | Active docs | Problem formulation, atomic skill inventory, and expert-demo rollout plan. |
| `tests/` and `dataset_clip_wrapper/tests/` | Active tests | Smoke tests for schema boundaries, graph builders, repair, manifests, and bundle layout. |

## Active Commands

Use these entrypoints for the current L1/L2 path:

```bash
python -m dataset_clip_wrapper.run_staged_llm_pipeline
python -m dataset_clip_wrapper.report_l1_l2_quality
python -m dataset_clip_wrapper.run_repair_protocol
python -m dataset_clip_wrapper.report_final_acceptance
python -m dataset_clip_wrapper.export_expert_demos
python -m dataset_clip_wrapper.tests.smoke_test_module_bundles
python -m dataset_clip_wrapper.tests.smoke_test_training_manifests
```

## Not In This Clean Base

These directories existed in the larger integration branch, but are intentionally
absent from this clean base:

| Path | Decision |
|------|----------|
| `cold_start/` | Old game rollout generation. Do not re-add for L1/L2 video work. |
| `data_structure/` | Old `Episode` / `Experience` game schema. Do not use for new L1/L2 traces. |
| `decision_agents/` | Old VLM game decision-agent runtime. Do not make it the L1/L2 controller. |
| `dataset_examples/` | Old small fixture bundle. Add only tiny fixtures later if tests require them. |
| `skill_agents/` | Old GRPO/LoRA skill-bank system. Use only as an external reference if needed; do not make it the motif runtime. |
| `trainer/` | Old broader training stack. Add a new L1/L2 trainer only after the export format is stable. |

## Future Motif Layer

Future composed-motif work should live under `dataset_clip_wrapper/motifs/`
once implementation starts. Motifs are optional verified subgraph priors that
expand into frozen atomic skills. They are not new primitive tools, callable
skill agents, or hidden evidence.

Planned ownership:

```text
dataset_clip_wrapper/motifs/
  canonicalize.py
  miner.py
  registry.py
  promotion.py
  expansion.py
```

Do not add this package to `dataset_clip_wrapper/module_bundles.py` until real
implementation modules exist.

## Cleanup Rules

- New L1 graph code goes in `dataset_clip_wrapper/l1_clue_graph/`.
- New L2 reasoning or recursive trajectory code goes in
  `dataset_clip_wrapper/l2_reasoning_graph/`.
- New verification, repair, audit, and final-acceptance logic goes in
  `dataset_clip_wrapper/verification/`.
- New expert-demo export code goes in `dataset_clip_wrapper/expert_demos/`.
- New split/manifest code goes in `dataset_clip_wrapper/manifests/`.
- New controller-training adapters should go in `dataset_clip_wrapper/training/`
  once the trace format is stable.
- Do not add new generated JSONL/JSON artifacts to git unless they are tiny
  checked fixtures for tests.
- Do not reintroduce legacy top-level packages without updating this document
  and `scripts/check_repo_layout.py`.

## Next Cleanup Steps

1. Keep this clean branch free of old game-agent and skill-bank directories.
2. Add the first `dataset_clip_wrapper/motifs/` implementation only after
   accepted L2 rollout graphs are stable enough to mine.
3. Add an L1/L2 controller-training export adapter under
   `dataset_clip_wrapper/training/`.
4. Add a small controller SFT trainer only after prompt-visible fields are
   sanitized against hidden-supervision leakage.
