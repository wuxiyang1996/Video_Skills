# Repository Structure

Last updated: 2026-07-06

This repository currently combines two histories:

- the L1/L2 video clue-graph protocol from `video_skills_relaunched`
- the `Video_Skills` controller runtime, contracts, harness, and training
  infrastructure from `main`

The active integration branch is `integration/l1l2-controller-training`.
This document is the cleanup map: it names the current path for new work and
marks older directories that should not receive new L1/L2 controller-training
code unless they are explicitly revived.

## Active Path

These directories are the current supported path for video-only graph creation,
verification, repair, demo export, and controller-training data.

| Path | Status | Purpose |
|------|--------|---------|
| `dataset_clip_wrapper/` | Active | Dataset adapters, video-only perception, L1 clue graph, L2 reasoning/repair/verification, expert demo export, training trace export. |
| `atomic_skills/` | Active | Frozen evidence-graph and reasoning-graph atomic skill basis used by L1/L2 graph protocols. |
| `video_skills/` | Active | Canonical runtime contracts, memory, retriever, verifier, harness, rule controller, and `ReasoningTrace` schema. |
| `trainer/` | Active but mixed | Existing SFT/GRPO infrastructure. New L1/L2 controller SFT should consume `dataset_clip_wrapper.export_reasoning_traces` output before deeper trainer refactors. |
| `infra_plans/` | Active docs | Normative design docs for controller, memory, harness, skills, training, and evaluation. |
| `docs/` | Active docs | L1/L2 graph schema, implementation status, MDP framing, repo bundle map, and cleanup notes. |
| `tests/video_skills/` | Active tests | Runtime contract/harness/memory/verifier tests. |
| `dataset_clip_wrapper/tests/` | Active tests | L1/L2 wrapper, graph, repair, manifest, and training-export smoke tests. |

Future composed-motif work should live under `dataset_clip_wrapper/motifs/`
once implementation starts. Motifs are optional verified subgraph priors that
expand into frozen atomic skills; they are not new primitive tools and should
not depend on `skill_agents/`.

## Active Commands

Use these entrypoints for the current L1/L2 controller-training path:

```bash
python -m dataset_clip_wrapper.run_staged_llm_pipeline
python -m dataset_clip_wrapper.report_l1_l2_quality
python -m dataset_clip_wrapper.run_repair_protocol
python -m dataset_clip_wrapper.report_final_acceptance
python -m dataset_clip_wrapper.export_expert_demos
python -m dataset_clip_wrapper.export_reasoning_traces
python -m dataset_clip_wrapper.tests.smoke_test_module_bundles
python -m dataset_clip_wrapper.tests.smoke_test_trace_adapter
python -m pytest -q tests/video_skills
```

## Legacy Or Specialized Paths

These directories came from earlier phases or adjacent systems. They may contain
useful code, but new L1/L2 controller-training work should not be added here by
default.

| Path | Status | Cleanup decision |
|------|--------|------------------|
| `skill_agents/` | Legacy/specialized | Keep for now. Contains older GRPO, LoRA, skill extraction, bank maintenance, and evaluation utilities that may be mined later. Do not make it the primary L1/L2 controller-training path. |
| `decision_agents/` | Legacy/specialized | Keep for old decision-agent experiments. New controller work should target `video_skills/` and `trainer/`. |
| `cold_start/` | Legacy data generation | Keep until SFT data flow is fully replaced by L1/L2 `ReasoningTrace` exports. |
| `labeling/` | Legacy teacher labeling | Keep as historical teacher-data tooling. Do not mix with video-only L1/L2 exports without an adapter. |
| `visual_grounding/` | Older grounding substrate | Keep as a reference implementation and possible future grounding layer. Current video-only L1 graph work lives under `dataset_clip_wrapper/perception/` and `dataset_clip_wrapper/l1_clue_graph/`. |
| `rag/` | Specialized retrieval | Keep as experimental retrieval infrastructure. Current L1 retrieval lives under `dataset_clip_wrapper/l1_clue_graph/`. |
| `inference/` | Legacy scripts | Keep scripts that still run; new runnable commands should prefer package entrypoints. |
| `scripts/` | Mixed scripts | Keep, but avoid adding new L1/L2 pipeline commands here unless they wrap package entrypoints. |
| `reflection/` | Placeholder/legacy | Review before use. |
| `data_structure/` | Legacy structures | Review before use. |
| `schemas/` | Active-ish schema docs | Keep JSON schemas, but update only when the L1/L2 external wire format changes. |
| `atomic-skill-decomposition-and-assembly/` | Historical docs | Keep as problem formulation and earlier atomic-skill notes. |

## Generated Or Example Data

| Path | Status | Policy |
|------|--------|--------|
| `dataset_clip_wrapper/output/` | Generated | Only `.gitkeep` should be tracked. Use it for local API outputs, staged caches, reports, and demo banks. |
| `out/` | Historical generated examples | Currently tracked as small grounding snapshots. Do not add new large outputs here; prefer `dataset_clip_wrapper/output/` or external artifact storage. |
| `dataset_examples/` | Small examples | Keep tiny examples only. Do not add full benchmark videos here. |
| `.pytest_cache/`, `__pycache__/` | Generated | Must remain untracked. |

## Root Files

| File | Status | Purpose |
|------|--------|---------|
| `README.md` | Active relaunch overview | L1/L2 graph-first research framing. |
| `readme.md` | Active legacy/main overview | Controller runtime and original Video_Skills architecture. |
| `REPO_STRUCTURE.md` | Active cleanup map | Integration branch source of truth for where new code should go. |
| `pyproject.toml`, `requirements.txt`, `INSTALL.md` | Active setup | Python package/test/install metadata. |
| `.env.example` | Active template | Example environment variables only. |
| `.env` | Local secret file | Must remain untracked. |

## Cleanup Rules Going Forward

- New L1 graph code goes in `dataset_clip_wrapper/l1_clue_graph/`.
- New L2 reasoning or recursive trajectory code goes in
  `dataset_clip_wrapper/l2_reasoning_graph/`.
- New verification, repair, audit, and final-acceptance logic goes in
  `dataset_clip_wrapper/verification/`.
- New expert-demo export code goes in `dataset_clip_wrapper/expert_demos/`.
- New controller-training adapters go in `dataset_clip_wrapper/training/`.
- New motif mining, promotion, registry, and expansion code goes in
  `dataset_clip_wrapper/motifs/` after the first implementation starts. Motifs
  must expand into existing atomic skill graph fragments before execution.
- New runtime contracts, harness, memory, retriever, or verifier code goes in
  `video_skills/`.
- New trainer code should either extend `trainer/` or add a clearly named
  subpackage such as `trainer/l1l2_controller_sft/`.
- Do not add new generated JSONL/JSON artifacts to git unless they are tiny
  checked fixtures for tests.
- Do not add new root-level Python packages without updating
  `scripts/check_repo_layout.py` and this document.
- Do not wire `skill_agents/` into the L1/L2 critical path. It remains a
  legacy/reference source for GRPO, LoRA, reward, and bank-maintenance ideas
  until its useful pieces are explicitly ported.

## Next Cleanup Steps

1. Add the first `dataset_clip_wrapper/motifs/` implementation only after the
   accepted rollout graph format is stable enough to mine.
2. Add a real L1/L2 controller SFT trainer entrypoint under `trainer/`.
3. Convert current compact demo bank into `ReasoningTrace` and SFT chat JSONL
   using `dataset_clip_wrapper.export_reasoning_traces`.
4. Decide whether `out/` snapshots should stay tracked as examples or move to
   external artifacts.
5. After the first SFT run works, move obsolete legacy script entrypoints behind
   docs-only references or a `legacy/` namespace in a separate commit.
