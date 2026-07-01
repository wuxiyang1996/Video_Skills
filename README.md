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
- **Composed motifs can be mined.** Offline extraction may promote frequent
  verified atomic-skill subgraphs into reusable motif templates, but never into
  new primitive tools.
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
- `SIV-Bench`: short social-interaction reasoning with weak evidence alignment.

M3-Bench is intentionally deferred until its memory graph reader is ready.

## Documents

- [Problem formulation, English](atomic-skill-decomposition-and-assembly/problem-formulation-en.html)
- [Problem formulation, Chinese](atomic-skill-decomposition-and-assembly/problem-formulation-zh.html)
- [Atomic skills v1](atomic-skill-decomposition-and-assembly/atomic-skills-v1.md)
- [Expert demo rollouts from datasets](atomic-skill-decomposition-and-assembly/expert-demo-rollouts-from-datasets.md)
- [Unified video skill schema](docs/unified-video-skill-schema.md)
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

The graph-construction skill set remains part of the full atomic vocabulary, but
it should act as an offline graph-builder/audit interface for the first
`expert_demo` experiments. Later `video_only` experiments can activate selected
graph-construction skills as tool-mediated actions.

## Evaluation Questions

The key experiments should answer:

- Does typed skill-graph fitting outperform free-form CoT distillation?
- Does graph structure outperform a linear tool chain?
- Does verifier-grounded local repair improve evidence F1 and answer accuracy?
- Is the selected atomic basis an empirical elbow point over `K=8/12/16/24`?
- Do discovered motifs improve planning cost or transfer without becoming new
  black-box skills?

## Status

This branch is a clean relaunch surface. The current contents are design docs
and project scaffolding for the next implementation pass.
