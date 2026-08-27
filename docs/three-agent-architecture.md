# Three-Agent L1/L2 Architecture

Last updated: 2026-07-06

This document defines the high-level agent split for the clean L1/L2 relaunch
branch. The system should be organized as three cooperating agents over frozen
atomic skills and typed graph objects, not as one monolithic open-ended agent.

## Summary

```text
video + question
  -> Agent 1: L1 Graph Crafter
  -> Agent 2: L2 Recursive Reasoning / Answer Agent
  -> Agent 3: Motif Extraction and Management Agent
```

The agents have different responsibilities and different visibility boundaries:

| Agent | Main object | Writes | Must not do |
|-------|-------------|--------|-------------|
| L1 Graph Crafter | `ClueMemoryGraph` / evidence graph | visible video evidence nodes, edges, retrieval indexes, answerability gaps | commit answers or copy hidden GT into visible evidence |
| L2 Recursive Reasoning / Answer Agent | `SkillGraphRollout`, `l2_trajectory`, `repair_subgraph` | reasoning nodes, claims, verifier calls, bounded repair rounds, final answer or abstain | invent evidence, bypass verifier, run unbounded recursion |
| Motif Extraction and Management Agent | `CompositeMotif` registry | Qwen3.5/GPT-OSS motif proposals, curator decisions, support stats, promotion decisions, atomic expansion templates | create new atomic skills, execute as a black-box skill agent, mine held-out test data |

## Agent 1: L1 Graph Crafter

Purpose: craft the question-agnostic visual clue graph from visible video/tool
outputs.

Current code location:

```text
dataset_clip_wrapper/perception/
dataset_clip_wrapper/l1_clue_graph/
```

Responsibilities:

- segment/select clips according to clip policy;
- produce clip schemas from visible video and optional subtitles;
- compose semantic L1 nodes and edges from observed clips;
- build retrieval indexes and coarse/fine references;
- mark answerability gaps and missing modality requirements;
- run the L1 gate before expensive L2 reasoning.

Inputs visible to this agent:

```text
video
question metadata needed for routing/gating
visible subtitles or clip schemas, when allowed by the benchmark setting
```

Hidden from this agent in `video_only` mode:

```text
gold answers
official clue intervals
official reasoning_process
dataset explanations and rationales
```

The L1 agent may create missing-evidence or answerability-gap records, but it
does not commit answers.

## Agent 2: L2 Recursive Reasoning / Answer Agent

Purpose: consume `question + L1 graph`, craft the L2 reasoning graph, verify
claims, run bounded repair, and either commit an answer or abstain.

Current code location:

```text
dataset_clip_wrapper/l2_reasoning_graph/
dataset_clip_wrapper/verification/
```

Current trace objects:

```text
SkillGraphRollout
l2_trajectory
repair_subgraph
```

Responsibilities:

- craft a question-conditioned skill graph over L1 evidence;
- bind evidence refs to claims;
- call verifier gates before answer commitment;
- diagnose evidence gaps when verification fails;
- request bounded L1 repair patches or option-specific evidence packs;
- append bounded recursive repair rounds;
- commit only `accepted_strong` / `resolved_strong` / allowed bridge outcomes;
- abstain as `needs_more_evidence` when support is insufficient.

The current implementation is not a trained closed-loop MDP policy yet. It is:

```text
gpt-oss open-loop L2 planning
  + bounded recursive repair trace logging
  + verifier-filtered final acceptance
```

The desired training path is:

```text
expert trajectories
  -> SFT / behavioral cloning
  -> RLVR-style progressive reward
  -> closed-loop controller over frozen atomic skill ids
```

Training reward may be progressive, but evaluation remains binary/exact:

```text
training:
  schema validity
  evidence progress
  reasoning-chain progress
  verifier support
  repair success
  final-answer correctness

evaluation:
  answer_correct: true / false
  accepted_or_abstained_correctly: true / false
  evidence_refs_valid: true / false
  no_hidden_leakage: true / false
```

## Agent 3: Motif Extraction and Management Agent

Purpose: use a Qwen3.5/GPT-OSS agent pipeline to extract, curate, maintain, and
promote reusable graph motifs from accepted L2 rollouts.

Code location:

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

Responsibilities:

- consume accepted `SkillGraphRollout` / `l2_trajectory` records;
- summarize traces into compact motif-extraction inputs;
- call the Qwen3.5 motif proposal agent;
- call the GPT-OSS motif curator for approve/defer/veto bank decisions;
- canonicalize entity names, timestamps, option labels, and dataset terms;
- use deterministic trajectory-round and repair-subgraph mining as
  seed/fallback/audit, not as the whole agent;
- update candidate motif statistics;
- apply promotion gates;
- provide promoted motifs as optional planning/repair priors;
- expand every motif into frozen atomic skill nodes before runtime use.

Agent 3 intentionally follows the old skill-bank-agent pipeline shape:

```text
ingest accepted rollout
  -> propose candidates
  -> curate bank mutations
  -> update persistent bank
  -> serve retrieved priors
```

The implementation ports that design onto the current L1/L2 schema instead of
reintroducing the old `skill_agents/` runtime.

The motif agent should operate asynchronously or periodically over rollout
logs. It is not part of the answer-critical path for a single example until a
promoted motif is retrieved as a prior and expanded into atomic nodes.

Allowed online behavior:

```text
after rollout completes:
  accepted graph -> Qwen/GPT-OSS motif proposal+curation -> registry update
```

Not allowed:

```text
mid-rollout creation of new atomic skills
black-box motif execution
using held-out test examples for motif thresholds
using hidden GT as motif content
```

## Why Three Agents

The split keeps each learning problem clean:

- L1 learns or improves evidence discovery without answering.
- L2 learns reasoning, verifier use, and bounded repair over explicit evidence.
- Motif management learns reusable graph priors without changing the atomic
  action basis.

This also gives cleaner ablations:

| Setting | What it tests |
|---------|---------------|
| L1 only | Evidence recall and answerability detection |
| L1 + L2 no repair | Reasoning graph assembly over fixed evidence |
| L1 + L2 recursive repair | Whether bounded repair improves acceptance |
| L1 + L2 + motif priors | Whether mined motifs improve planning cost or transfer |
| Motif disabled | Pure atomic graph assembly baseline |

## Implementation Boundary

Current branch status:

- Agent 1 is implemented as L1 graph/perception/gating modules.
- Agent 2 is implemented as GPT-OSS L2 planning plus bounded recursive repair
  traces and final acceptance reports.
- Agent 3 has a first implementation under `dataset_clip_wrapper/motifs/` with
  `hybrid`, `llm`, and `deterministic` modes. `hybrid` defaults to Qwen3.5
  proposal plus GPT-OSS curation, with deterministic seed/fallback when API
  access is unavailable. Full atomic-subgraph mining is the next extension.

Do not re-add the old `skill_agents/` package to implement Agent 3. If GRPO,
LoRA, reward-buffer, or bank-maintenance ideas are useful, port small utilities
explicitly into the current L1/L2 schema.
