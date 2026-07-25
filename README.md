# Video Skills Relaunched

Trace-to-Skill Fitting for verifiable video reasoning.

## Core Idea

This relaunch studies whether complex video reasoning can be converted from
free-form teacher chain-of-thought into typed, executable, verifier-filtered
skill graphs.

The project does **not** start from open-ended online skill invention. Instead:

```text
teacher / tool / evidence demonstrations
  -> trace segmentation
  -> operation-intent clustering
  -> candidate operation ontology
  -> coverage / confusion / cost selection
  -> frozen typed atomic basis
  -> trace-to-skill fitting
  -> controller training over frozen basis
  -> optional verified motif retrieval
```

The main claim is:

> A compact controller should learn to assemble reusable, evidence-grounded
> video-reasoning programs rather than imitate full teacher reasoning strings.

Equivalently:

```text
perception / indexing builds a clue-memory graph from video
agent control composes a skill chain or skill graph over that graph
verification checks that the composed reasoning cites valid evidence
```

The clue-memory graph organizes what has been perceived: clips, captions,
subtitles, entities, events, clue candidates, episodic memories, and semantic
memories. The skill graph is different: it is the agent's executable
multi-hop reasoning program over the clue-memory graph.

## Design Principles

- **Atomic skills are primitives.** They are small typed operators with explicit
  inputs, outputs, evidence pointers, verifiers, and failure codes.
- **Graphs have layered roles.** The evidence/memory graph stores perceived
  video clues and retrieval structure; the skill graph stores the agent's
  composed reasoning actions over that graph.
- **The atomic basis is frozen before controller training.** Demonstrations can
  induce the candidate ontology, but the controller does not create new atomic
  skills online.
- **Skill graphs are executable.** Each node is a skill invocation; each edge is
  a typed dependency such as data, temporal, causal, or evidence.
- **Verification is central.** A rollout is useful only if its schema, evidence
  references, timestamps, and answer anchors can be checked.
- **Repair is local.** Failed nodes, arguments, evidence bindings, or edges are
  repaired without regenerating the whole answer.
- **Composed motifs are optional.** Reusable reasoning motifs are assembly or
  repair priors only; they must expand into frozen atomic skill graphs before
  execution.
- **Composed motifs are agent-managed.** Qwen3.5 proposes reusable motifs from
  accepted L1/L2 rollouts, GPT-OSS curates approve/defer/veto decisions, and
  deterministic mining remains a seed/fallback/audit path.
- **Motif code stays in the L1/L2 stack.** Motif extraction and management lives
  under `dataset_clip_wrapper/motifs/` and operates on accepted L1/L2 rollout
  graphs. Do not re-add the old `skill_agents/` package as the motif runtime;
  port only the propose -> curate -> bank-maintenance idea.
- **Evidence indexes are retrieval substrates.** M3-style clip memory graphs can
  help the agent discover clues, but final answers must cite verifiable
  `EvidenceCandidate` records extracted from the index.
- **Clip processing is policy-driven.** Short, long, and streaming videos share
  the same evidence interface; they differ by `clip_policy` (`whole_video`,
  `fixed_window`, `hierarchical`, or online causal windows). Streaming evidence
  must not cite future spans beyond the current observation time.
- **Memory and reasoning can share a graph container, not loose semantics.**
  Evidence/memory nodes and reasoning/skill nodes should live in a typed
  heterogeneous graph with separate namespaces and explicit cross-layer edges.
- **Unify conceptually, implement in layers first.** The MVP should keep
  `EvidenceGraph`, `SkillGraphRollout`, and `CrossLayerLinks` separate, then
  export them as one heterogeneous graph only after adapters and verifiers are
  stable.

## First Target

The first implementation target is expert demonstration generation from local
video reasoning datasets:

- `Video-Holmes`: complex short-video social, causal, temporal reasoning.
- `CG-Bench`: clue-grounded long-video evidence retrieval.
- `VRBench`: long-video timestamped multi-step reasoning.
- `SIV-Bench`: short social-interaction reasoning with weak evidence alignment;
  under the current no-audio video-only scope it is treated as an answerability
  gap / repair stress test rather than a primary success metric.

M3-Bench is intentionally deferred until its memory graph reader is ready.

For the current video-only benchmark track, the primary set is
`Video-Holmes`, `VideoMME`, `OVO-Bench`, `CG-Bench`, and `VRBench`. SIV-Bench
remains useful for testing whether the graph can honestly mark social/common-
sense gaps: many SIV questions depend on dialogue, confidential information,
or interpersonal intent that is not visible without audio/ASR. In this setting,
L2 may form a commonsense repair hypothesis, but it must not commit an answer
unless concrete video evidence verifies it.

## Documents

- [Problem formulation, English](atomic-skill-decomposition-and-assembly/problem-formulation-en.html)
- [Problem formulation, Chinese](atomic-skill-decomposition-and-assembly/problem-formulation-zh.html)
- [Atomic skills v1](atomic-skill-decomposition-and-assembly/atomic-skills-v1.md)
- [Expert demo rollouts from datasets](atomic-skill-decomposition-and-assembly/expert-demo-rollouts-from-datasets.md)
- [Unified video skill schema](docs/unified-video-skill-schema.md)
- [Clip processing policy](docs/clip-processing-policy.md) — short / long / streaming segmentation
- [Three-agent architecture](docs/three-agent-architecture.md) — L1 graph crafter, L2 recursive answer agent, motif manager
- [Repository bundle map](docs/repo-bundle-map.md) — L1/L2/verifier/tooling ownership boundaries
- [Repository structure](REPO_STRUCTURE.md) — clean branch active path and excluded legacy directories
- [Repository cleanup audit](docs/repo-cleanup-audit.md) — cleanup milestones and guardrails
- [Motif layer boundary](docs/motif-layer-boundary.md) — composed motifs vs legacy skill agents
- [MDP formulation](docs/mdp-formulation.md) — atomic skill invocations as graph-state actions
- [MDP-style SFT data generation](docs/sft-data-generation.md) — L1 builder/patch, L2/repair, verifier, and motif cold-start exports
- [Implementation status](docs/implementation-status.md) — runnable code, datasets, gaps
- [Dataset clip wrapper](dataset_clip_wrapper/README.md) — core + streaming benchmark canonical clip exporter
- [Canonical example JSON schema](schemas/canonical_video_example.schema.json)
- [Skill graph rollout JSON schema](schemas/skill_graph_rollout.schema.json)

## Recommended First Experiment

Start with expert-demo reasoning assembly over prebuilt evidence-memory graphs.
The first controller-visible basis should come from Reasoning Graph Assembly
Skills rather than graph-construction skills:

```text
parse_question_target
propose_evidence_roles
retrieve_by_event
retrieve_by_entity
retrieve_by_time
retrieve_by_relation
localize_clue
extract_claim
assign_evidence_role
compose_evidence_chain
infer_temporal_relation
infer_social_contradiction
verify_claim_support
commit_answer
```

The graph-construction skill set remains part of the full atomic vocabulary.
There are two graph-building paths:

- `expert_demo`: an offline graph-builder/audit interface seeds the clue-memory
  graph from dataset annotations, captions, subtitles, and ground-truth clue
  intervals.
- `video_only`: a perception-first builder seeds the same clue-memory graph
  interface from visible video/tool outputs such as automatic clips, VLM
  captions, ASR/subtitles, OCR, entity mentions, and temporal event signals.

For `video_only`, graph building is staged:

```text
question-blind L1 video memory
  -> full coarse coverage / full short-video fine coverage
  -> query-time memory retrieval from the visible question
  -> retrieved fine graph expansion
  -> no-gold-answer L2 reasoning and verifier checks
```

This split matters because a structurally valid graph can still be a weak
reasoning memory if clip captions miss objects, places, repeated props, speech,
or timestamp anchors. The current L1 prompt and deterministic composer therefore
prefer graph-ready fields such as salient objects, place cues, cross-clip cues,
searchable phrases, and `same_entity` links over generic captions alone.

Both paths should export compatible `evidence_index` / clue-memory graph
objects so the same reasoning-graph controller and verifier can run over either
supervised or discovered evidence.

## Evaluation Questions

The key experiments should answer:

- In `video_only` mode, can a controller discover evidence from visible
  video/tool outputs and assemble a verified skill graph without hidden clues?
- Does typed skill-graph fitting outperform free-form CoT distillation?
- Does graph structure outperform a linear tool chain?
- Does verifier-grounded local repair improve evidence F1 and answer accuracy?
- Is the selected atomic basis an empirical elbow point over `K=8/12/16/24`?
- Do discovered motifs improve planning cost or transfer without becoming new
  black-box skills?

## Status

Current branch state:

- **Design docs**: unified schema, clip policy, dataset rollout recipes, problem formulation
- **Atomic skills**: 28 executable Python functions (9 graph construction + 19 reasoning assembly) in `atomic_skills/`
- **Runtime path**: Qwen clip schemas → L1 clue graph → L2 retrieval/reasoning → verifier → bounded repair → optional motif curation
- **Cold-start SFT**: five-specialist chat package at
  `dataset_clip_wrapper/output/sft_cold_start/specialist_sft_v3_20260722/five_lora/`
  (`all_hard_gates_passed=true`; backup tarball under `backups/`)
- **Code layout**: functional bundles under `dataset_clip_wrapper/`
  (`perception` → `l1_clue_graph` → `l2_reasoning_graph` → `verification` → `motifs`,
  plus `training/` SFT adapters); smokes grouped the same way under `tests/<bundle>/`
- **Still open**: full VLM/ASR/tracker perception, embedding coarse retrieval, gated L2 claim/compose SFT, fine-grained repair traces, GRPO / closed-loop controller; consolidate top-level `motif/` into `dataset_clip_wrapper/motifs/`

Local datasets live under `/fs/gamma-projects/vlm-robot/datasets`. See
[implementation status](docs/implementation-status.md),
[SFT data generation](docs/sft-data-generation.md), and
[two-layer graph schema](docs/two-layer-graph-schema.md) for commands and layer contracts.

### Controller walkthrough (L1 / L2 / repair / motif / verifier)

```text
video
  -> [perception] Qwen clip schemas (not one of the five LoRAs)
  -> L1: question-blind clue-memory graph build / patch
  -> L2: question + L1 -> coarse/fine retrieval + reasoning control
  -> Verifier: is option evidence sufficient?
       |- supported -> may commit; later motif mining
       `- insufficient -> Repair (bounded L1 patch / reroute / re-verify)
  -> Motif: post-hoc reusable prior from accepted traces (must expand before use)
```

The deterministic runtime verifier remains the hard gate. Learned verifier and
motif models are auxiliaries only.

| Specialist | Role | v3 rows (train/dev) | Dominant supervised actions |
|---|---|---:|---|
| **L1** | Build/patch visible evidence graph; never answer | 15,690 (12,686 / 3,004) | `create_node`, `create_schema_anchor`, `create_edge`, `segment_*`, `apply_l1_evidence_patch` |
| **L2** | Query-time retrieval and recovery over cached L1 | 867 (684 / 183); **core=23** | `select_coarse_clips`, rank/select next coarse, recovery diagnose / reject-commit |
| **Repair** | Bounded fix after verifier failure | 127 (115 / 12) | mostly `bounded_recursive_repair` (`reroute`, `existing_l1_option_verification`) |
| **Verifier** | `supported` / `insufficient` on claim + evidence pack | 92 (79 / 13); 60/32 | `emit_verifier_decision` |
| **Motif** | Lifecycle + evidence-ref audit; non-executable prior | 320 (262 / 58) | `set_motif_evidence_ref_audit`, `set_motif_lifecycle_status` |

Each SFT row is one MDP-style chat transition: visible `state_t` in the user
turn, next tool-action JSON in the assistant turn. Split unit is source-video
group; `prompt_forbidden_key_hits=0`.

### What is missing or thin

| Gap | Notes |
|---|---|
| L2 claim / compose assembly | Designed actions (`extract_claim`, `assign_evidence_role`, `compose_evidence_chain`, …) are mostly **not** in the v3 SFT package; current L2 is retrieval/recovery-heavy |
| L2 core positives | Only 23 core rows vs 844 derived; raw `accepted_strong` remains **no-go** without correctness + option verifier gates |
| Fine-grained repair | Repair rows collapse diagnose → patch → re-verify → abstain into coarse round actions |
| Verifier coverage | Small and CG-Bench-heavy; Video-Holmes nearly absent |
| Motif positives | Rejected/audit-heavy; few candidate/shadow and almost no promoted-use traces |
| Classic 9 L1 atomics as tool names | Folded into `neighbor_vlm_l1_*` (+ segment/patch); `extract_observation` etc. not exported as distinct tools |
| Perception LoRA | Clip schemas stay on local Qwen; not part of the five specialists |
| Dataset scope | SFT is CG-Bench + Video-Holmes only; VRBench / VideoMME / OVO held out by design |
| Training loop | Export + LoRA trainer exist; GRPO / joint L1+L2 / closed-loop policy still open |
| Mixture balance | Doc target ~35/35/20/10 (L1 / L2+repair / verifier / motif); v3 is ~91% L1 by row count |

Priority fills: gated L2 claim/compose, fine-step repair, Video-Holmes verifier negatives.

Quick smoke test:

```bash
python experiments/smoke_test_atomic_skills.py
```
