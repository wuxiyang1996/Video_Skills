# Repository Cleanup Audit

Last updated: 2026-07-24

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
  L2, verification/repair, manifests, expert demos, runners, motifs, and tests.
- `atomic_skills/` contains the frozen atomic basis for evidence construction
  and reasoning assembly.
- Legacy top-level directories such as `cold_start/`, `data_structure/`,
  `decision_agents/`, `dataset_examples/`, `skill_agents/`, and `trainer/` are
  not present in this clean base.
- Generated `dataset_clip_wrapper/output/` artifacts are ignored.
- Empty ghost packages `rag/`, `visual_grounding/`, and `video_skills/`
  (pycache-only shells with zero git-tracked sources) were removed in the P0
  layout cleanup. Layout guardrails in `scripts/check_repo_layout.py` and
  `REPO_STRUCTURE.md` were updated accordingly.
- Intentional transitional/local top-level paths are documented:
  `motif/` (still used by some SFT pilot scripts/tests), `backups/` (local SFT
  tarballs), and `docs/`.

## Safety Rules

- Do not re-add old game-agent or skill-bank directories for L1/L2 video work.
- Do not add a `skill_agents/` dependency for motif runtime.
- Do not introduce new top-level packages without updating
  `REPO_STRUCTURE.md` and `scripts/check_repo_layout.py`.
- Do not reintroduce cleared ghost packages (`rag/`, `visual_grounding/`,
  `video_skills/`) as empty shells.
- Keep hidden dataset supervision outside `video_only` L1/L2 visible inputs.
- Keep evaluation binary at the task level while allowing RLVR/progressive
  training rewards on train prompts.
- Keep `.venv-*`, `.env`, and `.pytest_cache/` local/untracked; the layout
  checker ignores them and does not require them.

## Recommended Cleanup Milestones

### P0: Clean Base Guardrail (done / refreshed 2026-07-24)

- Track this branch as the clean base.
- Add `REPO_STRUCTURE.md`.
- Add `docs/motif-layer-boundary.md`.
- Add `scripts/check_repo_layout.py`.
- Remove empty `rag/`, `visual_grounding/`, and `video_skills/` pycache shells.
- Update layout known-set to include `docs/`, `motif/`, and `backups/`, and to
  ignore local/generated `.venv-*`, `.env`, and `.pytest_cache/`.

### P1: Motif Consolidation And Disk Hygiene

- Consolidate transitional top-level `motif/` into
  `dataset_clip_wrapper/motifs/`, updating `scripts/sft_pilot` and
  `tests/motif/` callers.
- Keep motif code under `dataset_clip_wrapper/motifs/` once consolidation is
  complete.
- Mine only accepted `SkillGraphRollout` graphs for positive candidates.
- Require promotion gates for support, verifier pass rate, cross-dataset
  coverage, confusion risk, expansion validity, and hidden-supervision leakage.
- Disk hygiene for large generated `output/` trees and local `.venv-*`
  directories (archive or prune intentionally; do not delete casually).

### P1.5: Training Export

- Stabilize the accepted L2 rollout graph format.
- Add a training export adapter under `dataset_clip_wrapper/training/`.
- Confirm prompt-visible records contain no hidden answers, clue intervals,
  official reasoning processes, or annotation shortcuts.

### P2: Controller Training

- Add a small L1/L2 controller SFT trainer only after the export format is
  stable.
- Use RLVR/progressive reward for training, while reporting held-out evaluation
  as hard 0/1 or True/False metrics.
