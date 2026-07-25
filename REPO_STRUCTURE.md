# Repository Structure

Last updated: 2026-07-24

This branch starts from `backup/pre-merge-l1l2-training-20260706`, the clean
L1/L2 relaunch base before the larger `Video_Skills` runtime and legacy
game-agent directories were merged in.

The active branch for this cleanup is `clean/l1l2-from-pre-merge-20260706`.
The goal is to keep the relaunch focused on video-only L1/L2 graph construction,
verification, repair, expert-demo export, and future controller-training data.

High-level control is split into three agents:

```text
Agent 1: L1 Graph Crafter
Agent 2: L2 Recursive Reasoning / Answer Agent
Agent 3: Motif Extraction and Management Agent
```

See `docs/three-agent-architecture.md` for the ownership and training/eval
boundary.

## Active Path

| Path | Status | Purpose |
|------|--------|---------|
| `dataset_clip_wrapper/` | Active | Dataset adapters, video-only perception, L1 clue graph, L2 reasoning/repair/verification, manifests, expert demo export, and canonical Motif Agent under `motifs/`. |
| `atomic_skills/` | Active | Frozen evidence-graph and reasoning-graph atomic skill basis used by L1/L2 graph protocols. |
| `docs/` | Active docs | L1/L2 graph schema, implementation status, MDP framing, bundle map, cleanup decisions, and prior-work notes (`self-repair-prior-work.md`, moved from top-level `reflection/`). |
| `atomic-skill-decomposition-and-assembly/` | Active docs | Problem formulation, atomic skill inventory, and expert-demo rollout plan. |
| `experiments/` | Active | Small experiment / toy scripts for graph skill reasoning. |
| `schemas/` | Active | Shared schema notes and contracts. |
| `scripts/` | Active | Layout guardrails, SFT pilot helpers, and operational scripts. |
| `tests/` and `dataset_clip_wrapper/tests/` | Active tests | Top-level: `tests/motif/`, `tests/sft/`. Wrapper smokes are grouped under `dataset_clip_wrapper/tests/{core,perception,l1,l2,verification,motifs,training}/` with root `smoke_test_*.py` shims for `-m` compat. |

## Functional Map

```text
adapters + perception
  -> l1_clue_graph
  -> l2_reasoning_graph
  -> verification (repair / accept / abstain)
  -> motifs (optional post-hoc priors)

training/   SFT adapters across L1 / L2 / repair / verifier / motif
runners/    end-to-end orchestration
```

Top-level tests follow this split: motif coverage under `tests/motif/`, SFT
adapter/export coverage under `tests/sft/`. Pipeline ownership remains under
`dataset_clip_wrapper/` bundles.
## Transitional / Local Paths

| Path | Status | Purpose |
|------|--------|---------|
| `motif/` | Transitional | Legacy/parallel motif package still used by `scripts/sft_pilot` and `tests/motif/`. Canonical Motif Agent lives under `dataset_clip_wrapper/motifs/`; consolidate in a later P1 cleanup. |
| `backups/` | Local archive | Local tarball location for SFT artifacts (for example five_lora SFT). Not required to be git-tracked. |

Empty ghost packages `visual_grounding/`, `rag/`, and `video_skills/` (pycache-only shells with no tracked sources) were removed from this clean base on 2026-07-24. Do not reintroduce them without a documented ownership plan.

Local/generated paths such as `.venv-*`, `.env`, and `.pytest_cache/` are ignored by
`scripts/check_repo_layout.py` and should stay out of git.

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
| `visual_grounding/` | Ghost leftover cleared (P0). L1 grounding path lives under `dataset_clip_wrapper/`. |
| `rag/` | Ghost leftover cleared (P0). Do not re-add empty shells. |
| `video_skills/` | Ghost leftover cleared (P0). Relaunch work lives under `dataset_clip_wrapper/` and related active paths. |

## Motif Layer

Canonical composed-motif work lives under `dataset_clip_wrapper/motifs/`.
Motifs are optional verified subgraph priors that expand into frozen atomic
skills. They are not new primitive tools, callable skill agents, or hidden
evidence.

Current ownership:

```text
dataset_clip_wrapper/motifs/
  agent.py
  canonicalize.py
  llm_agent.py
  miner.py
  registry.py
  promotion.py
  expansion.py
```

The top-level `motif/` package remains transitional for a subset of SFT pilot
scripts and motif tests. New Motif Agent features should land under
`dataset_clip_wrapper/motifs/`; plan to consolidate `motif/` into that package
in a later cleanup.

The current implementation follows the old skill-bank-agent shape on the new
L1/L2 schema: Qwen3.5 proposes motifs, GPT-OSS curates approve/defer/veto bank
decisions, and the JSONL motif bank stores support/pass-rate/promotion metadata.
The deterministic miner extracts trajectory-round and repair-subgraph motifs as
seed/fallback/audit, not as the whole Motif Agent.

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
- New motif extraction and management code goes in `dataset_clip_wrapper/motifs/`.
- Keep SFT tarball archives under `backups/` (local); do not commit large
  binaries unless explicitly intended.
- Do not add new generated JSONL/JSON artifacts to git unless they are tiny
  checked fixtures for tests.
- Do not reintroduce legacy top-level packages or cleared ghost packages
  (`visual_grounding/`, `rag/`, `video_skills/`) without updating this document
  and `scripts/check_repo_layout.py`.
- Local/generated paths (`.venv*`, `.env`, `.pytest_cache/`, `__pycache__/`)
  must stay untracked.

## Next Cleanup Steps

1. Keep this clean branch free of old game-agent and skill-bank directories.
2. Consolidate transitional top-level `motif/` into
   `dataset_clip_wrapper/motifs/` (P1), updating `scripts/sft_pilot` and
   `tests/motif/` callers.
3. Expand Qwen/GPT-OSS motif proposal from trajectory/repair path templates to
   full atomic skill subgraphs once accepted L2 rollout nodes are exported
   consistently.
4. Add an L1/L2 controller-training export adapter under
   `dataset_clip_wrapper/training/`.
5. Add a small controller SFT trainer only after prompt-visible fields are
   sanitized against hidden-supervision leakage.
6. Disk hygiene for large generated `output/` trees and local `.venv-*`
   directories (do not delete casually; archive or prune intentionally).
