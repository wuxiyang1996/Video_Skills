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

## Design Principles

- **Atomic skills are primitives.** They are small typed operators with explicit
  inputs, outputs, evidence pointers, verifiers, and failure codes.
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

## Recommended MVP

Start with a compact controller-visible basis rather than exposing the full
candidate ontology:

```text
parse_question_target
propose_evidence_roles
retrieve_event
retrieve_temporal_neighborhood
resolve_entity_reference
localize_clue
extract_dialogue_claim
mark_evidence_role
compose_evidence_chain
order_events
verify_evidence_supports_claim
repair_by_requery
```

The 24-skill vocabulary is a candidate ontology for fitting and ablation, not
the default online action space.

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
