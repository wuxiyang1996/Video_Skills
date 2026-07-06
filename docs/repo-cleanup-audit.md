# Repository Cleanup Audit

Last updated: 2026-07-06

## Current Assessment

This branch is the clean L1/L2 relaunch base from
`backup/pre-merge-l1l2-training-20260706`. It intentionally avoids the larger
legacy game-agent, skill-bank, and trainer directories that were present in the
broader integration branch.

The codebase is now easier to reason about: the active path is the L1 clue graph,
L2 reasoning graph, verification/repair, manifests, expert-demo export, frozen
atomic skills, and schema docs.

The target control split is three high-level agents:

- L1 Graph Crafter for visible evidence graph construction.
- L2 Recursive Reasoning / Answer Agent for reasoning graph construction,
  verifier use, bounded repair, and final answer/abstain.
- Motif Extraction and Management Agent for mining and promoting reusable
  atomic-subgraph priors from accepted rollouts.

## What Is Clean Enough Now

- `dataset_clip_wrapper/` has physical bundle boundaries for perception, L1,
  L2, verification/repair, manifests, expert demos, runners, and tests.
- `atomic_skills/` contains the frozen atomic basis for evidence construction
  and reasoning assembly.
- Legacy top-level directories such as `cold_start/`, `data_structure/`,
  `decision_agents/`, `dataset_examples/`, `skill_agents/`, and `trainer/` are
  not present in this clean base.
- Generated `dataset_clip_wrapper/output/` artifacts are ignored.

## Safety Rules

- Do not re-add old game-agent or skill-bank directories for L1/L2 video work.
- Do not add a `skill_agents/` dependency for motif runtime.
- Do not introduce new top-level packages without updating
  `REPO_STRUCTURE.md` and `scripts/check_repo_layout.py`.
- Keep hidden dataset supervision outside `video_only` L1/L2 visible inputs.
- Keep evaluation binary at the task level while allowing RLVR/progressive
  training rewards on train prompts.

## Recommended Cleanup Milestones

### P0: Clean Base Guardrail

- Track this branch as the clean base.
- Add `REPO_STRUCTURE.md`.
- Add `docs/motif-layer-boundary.md`.
- Add `scripts/check_repo_layout.py`.

### P1: Training Export

- Stabilize the accepted L2 rollout graph format.
- Add a training export adapter under `dataset_clip_wrapper/training/`.
- Confirm prompt-visible records contain no hidden answers, clue intervals,
  official reasoning processes, or annotation shortcuts.

### P1.5: Motif Boundary Freeze

- Keep motif code under `dataset_clip_wrapper/motifs/` once implemented.
- Mine only accepted `SkillGraphRollout` graphs for positive candidates.
- Require promotion gates for support, verifier pass rate, cross-dataset
  coverage, confusion risk, expansion validity, and hidden-supervision leakage.

### P2: Controller Training

- Add a small L1/L2 controller SFT trainer only after the export format is
  stable.
- Use RLVR/progressive reward for training, while reporting held-out evaluation
  as hard 0/1 or True/False metrics.
