# Motif Layer Boundary

Last updated: 2026-07-06

This document fixes the cleanup decision for composed motifs and the older
`skill_agents/` stack in the L1/L2 controller-training branch.

## Decision

Keep composed motifs as an optional L1/L2 planning layer, but implement that
layer as new, small code aligned with the current graph schemas. Do not make
`skill_agents/` the motif runtime or a required dependency of the L1/L2 path.

```text
accepted SkillGraphRollout
  -> motif miner
  -> candidate motif registry
  -> promotion gates
  -> motif retrieval as planning prior
  -> atomic graph expansion
  -> verifier / repair
```

The current active path remains:

```text
dataset_clip_wrapper/perception/
dataset_clip_wrapper/l1_clue_graph/
dataset_clip_wrapper/l2_reasoning_graph/
dataset_clip_wrapper/verification/
dataset_clip_wrapper/expert_demos/
dataset_clip_wrapper/training/
atomic_skills/
video_skills/
```

## What A Motif Is

A motif is a reusable, verified subgraph prior:

- a canonical atomic-skill subgraph pattern;
- abstract evidence roles, not copied video facts;
- argument-binding templates;
- known local repair templates;
- support, failure, and verifier statistics.

A motif is not:

- a new atomic skill id;
- a black-box executor;
- a benchmark-specific answer shortcut;
- persistent evidence from older videos;
- a way to bypass node-level verification.

Every runtime use must expand into frozen atomic skill nodes before execution.

## Online Extraction Policy

Online motif extraction is allowed only as candidate mining:

```text
L2 rollout
  -> final verifier result
  -> accepted graph only
  -> canonicalize entity/time/option labels
  -> mine small connected atomic subgraphs
  -> update candidate statistics
```

Online extraction must not immediately mutate the controller action space. A
candidate can become a reusable motif only after promotion gates pass.

Suggested first gates:

```text
support_count >= k
verifier_pass_rate >= threshold
dataset_coverage >= 2, unless dataset_local
confusion_risk <= threshold
all nodes map to frozen atomic skills
expansion validates as a SkillGraphRollout fragment
no hidden supervision appears in runtime-visible fields
```

Rejected rollouts may contribute negative statistics, but should not create
positive motif templates.

## Code Location

The first implementation lives under:

```text
dataset_clip_wrapper/motifs/
  __init__.py
  bank.py
  lifecycle.py
  mine_existing_l1_l2.py
  miner.py
  retrieval.py
  schemas.py
  transfer.py
```

Expected ownership:

| Module | Responsibility |
|--------|----------------|
| `schemas.py` | Define motif records, evidence refs, lifecycle status, and transfer reports. |
| `bank.py` | Store, load, summarize, and query motif records. |
| `lifecycle.py` | Move motifs through draft, candidate, verified, active, shadow, rejected, and retired states. |
| `retrieval.py` | Select visible motif priors by trigger signature. |
| `transfer.py` | Compare baseline and motif-assisted runs on held-out examples. |
| `miner.py` | Extract first-pass candidate motifs from accepted L1/L2 rollout graphs. |
| `mine_existing_l1_l2.py` | CLI for mining motif candidates from saved JSON/JSONL outputs. |

Future extensions should add canonicalization, promotion-gate, and expansion
modules once candidate motifs need to become runtime planning priors.

## Relationship To skill_agents/

`skill_agents/` is legacy/specialized reference code for older game-oriented
skill-bank work. It can be mined for ideas or small utilities, especially:

- GRPO buffer and reward utilities;
- LoRA training wrappers;
- bank maintenance heuristics;
- candidate staging / promotion concepts.

It should not be used directly for motif runtime because its object model is a
skill-bank agent pipeline, while the L1/L2 motif layer needs verified subgraph
templates over `SkillGraphRollout`.

Allowed:

```text
read or port small utilities from skill_agents/
reuse promotion ideas after adapting schemas
compare reward designs for RLVR training
```

Not allowed in the current L1/L2 critical path:

```text
call SkillBankAgent to execute a motif
store motifs as callable skill agents
let skill_agents create new atomic ids for L1/L2
make dataset_clip_wrapper depend on skill_agents
```

## Training And Evaluation Boundary

Evaluation remains hard 0/1 or True/False at the answer/task level.

Training may use RLVR-style progressive rewards over valid schema, evidence
binding, verifier pass rate, local repair success, and final answer correctness.
Motif statistics may help shape planning or repair rewards, but final
acceptance still requires verifier-backed evidence and the task-level eval
metric remains binary.

## Cleanup Implication

Do not roll back to `backup/pre-merge-l1l2-training-20260706`. Use that branch
as a clean historical boundary for the minimal L1/L2 core, while continuing
cleanup on the current integration branch.

Do not physically move `skill_agents/` until old imports in `trainer/`,
`inference/`, `labeling/`, and `scripts/` are either removed or wrapped with
compatibility shims.
