# Repository Cleanup Audit

Last updated: 2026-07-06

## Current Assessment

The repository is usable, and the main docs are now cleaner: the current P5
L1/L2/repair status is the canonical status page, while expired implementation
history has been moved under `docs/legacy/`. The codebase still combines the
new L1/L2 clue-graph protocol with the older Video_Skills runtime and several
earlier training/skill-agent experiments, so code cleanup should still proceed
by classification first and deletion later.

## What Is Clean Enough Now

- `dataset_clip_wrapper/` has bundle boundaries for perception, L1 graph, L2
  reasoning, verification/repair, expert demos, manifests, and training export.
- `video_skills/` has a runnable Phase-1 runtime with tests.
- `dataset_clip_wrapper/training/trace_adapter.py` bridges compact expert demos
  into canonical `ReasoningTrace` and SFT chat JSONL.
- Generated `dataset_clip_wrapper/output/` artifacts are ignored except
  `.gitkeep`.
- `docs/implementation-status.md` now reflects the latest P5 batch only; older
  probe/rerun history is archived in `docs/legacy/implementation-status-pre-p5.md`.

## Main Mess Sources

| Source | Why it is messy | Current decision |
|--------|-----------------|------------------|
| Two README files | `README.md` came from relaunch, `readme.md` came from main. | Keep both for now, use `REPO_STRUCTURE.md` as integration map. |
| `skill_agents/` | Large older GRPO/skill-bank system with overlapping concepts. | Keep as legacy/specialized until L1/L2 controller SFT is stable. |
| `visual_grounding/` | Older grounding stack overlaps with `dataset_clip_wrapper/perception`. | Keep as reference; current L1 perception path remains in wrapper. |
| `out/` | Tracked generated-looking snapshots. | Do not add more; decide later whether to move to artifact storage. |
| Many script directories | Earlier experiments used script-first entrypoints. | New code should expose `python -m package.module` entrypoints. |
| Expired status prose | Old API probes and rerun summaries made the current status hard to read. | Keep historical notes in `docs/legacy/`; current docs should point at P5 artifacts. |

## Safety Rules Before Moving Files

- Do not move `video_skills/`, `dataset_clip_wrapper/`, `atomic_skills/`, or
  `trainer/` until the first SFT-ready training export and dry-run trainer pass.
- Do not delete `skill_agents/`; it contains reusable GRPO/LoRA/reward code.
- Do not wire `skill_agents/` into `dataset_clip_wrapper/` for motif runtime.
  The motif layer should be new code over accepted `SkillGraphRollout` objects,
  with any useful `skill_agents/` ideas ported explicitly.
- Do not delete `visual_grounding/`; it may still inform grounding-layer work.
- Do not merge `README.md` and `readme.md` until the integration branch is ready
  for `main`.
- Prefer adding compatibility wrappers before moving public command modules.

## Recommended Cleanup Milestones

### P0: Classification

Done in this pass:

- Add `REPO_STRUCTURE.md`.
- Add `scripts/check_repo_layout.py`.
- Update README pointers.
- Update bundle map for the new training export bundle.
- Archive pre-P5 implementation status under `docs/legacy/`.
- Replace the main implementation status with a compact P5-current entry point.

### P1: Training Path Freeze

Before any major physical refactor:

- Run real compact demo export.
- Run `dataset_clip_wrapper.export_reasoning_traces`.
- Confirm SFT chat JSONL has no hidden/gold fields in the prompt.
- Add an L1/L2 controller SFT trainer entrypoint.

### P1.5: Motif Boundary Freeze

Before implementing online motif extraction:

- Keep motif code out of `skill_agents/`; add it under
  `dataset_clip_wrapper/motifs/` when real implementation starts.
- Define the motif record as an expandable atomic subgraph template, not a
  callable agent or new atomic action.
- Mine only accepted `SkillGraphRollout` graphs for positive candidates.
- Require promotion gates for support, verifier pass rate, cross-dataset
  coverage, confusion risk, expansion validity, and hidden-supervision leakage.
- Keep evaluation binary at the task level while allowing RLVR/progressive
  training rewards over schema validity, evidence binding, verifier success,
  repair success, and final correctness.

### P2: Generated Artifact Cleanup

- Decide whether tracked `out/` files are fixtures or historical outputs.
- Move large non-fixture artifacts out of git.
- Keep only tiny examples under `dataset_examples/`.

### P3: Legacy Namespace

Only after P1:

- Move or clearly deprecate old script-first workflows.
- Consider `legacy/skill_agents_*` only if imports and docs are updated.
- Keep compatibility shims for one milestone before deletion.
