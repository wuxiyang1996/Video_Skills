# Implementation Status

Last updated: 2026-06-30

This document tracks what is designed, what is implemented, and how to run the
current code. It consolidates status from README, atomic skills v1, dataset
rollout plans, and recent experiments.

## 1. Project Architecture

### 1.1 Two Graph Layers

```text
perception / indexing
  -> EvidenceGraph / clue-memory graph

question + clue-memory graph
  -> agent-composed SkillGraphRollout / skill chain

cross_layer_links
  -> uses_evidence, supports_claim, verified_by, ...
```

The clue-memory graph and the skill graph are **not** the same object:

- **Clue-memory graph**: what perception and dataset adapters made available
- **Skill graph**: the agent's executable multi-hop reasoning program

Conceptually unified, engineering-layered: keep runtime objects separate, export
as one heterogeneous graph only after adapters and verifiers are stable.

### 1.2 Two Atomic Skill Bundles

| Bundle | Count | Role in MVP |
|--------|-------|-------------|
| Evidence Graph Construction | 9 | Offline graph builder / audit trace |
| Reasoning Graph Assembly | 19 | Primary controller-visible action set |

Total: **28 executable atomic skills** in `atomic_skills/`.

Evidence Graph Construction skills:

```text
segment_video_or_select_clip
extract_observation
extract_dialogue_span
detect_entity_mention
resolve_entity_coreference
create_event_node
create_state_node
link_graph_relation
assign_provenance_trust
```

These are **lightweight deterministic Python functions**. They operate on text,
annotations, captions, and clip time spans. They do **not** decode raw video
frames or call a VLM for perception yet.

### 1.3 Two Runtime Modes

| Mode | Purpose | Clip/graph construction |
|------|---------|-------------------------|
| `expert_demo` | Trace-to-skill fitting supervision | Graph often seeded from dataset annotations offline |
| `video_only` | Final evaluation | Agent must discover evidence from video/tools only |

## 2. Datasets

Local root: `/fs/gamma-projects/vlm-robot/datasets`

### 2.1 Core Targets

| Dataset | Tier | Length | Primary use |
|---------|------|--------|-------------|
| Video-Holmes | 1 | Short | Social/causal/temporal reasoning traces |
| CG-Bench | 1 | Long | Clue grounding and evidence retrieval |
| VRBench | 1 | Long | Timestamped multi-step reasoning chains |
| SIV-Bench | 2 | Very short | Social/intent/emotion with weak spans |
| M3-Bench | 2 (deferred) | Long + memory graph | Memory-query rollouts after graph reader |

### 2.2 Non-Primary / Reference

| Dataset | Status |
|---------|--------|
| TIR-Bench | Image QA; not a video rollout source |
| VisualToolBench | Tool-use format reference only |

### 2.3 Training Splits

```text
P0:
  Video-Holmes train: 500 QA
  CG-Bench mini: 500 QA
  VRBench: 100 videos x all QA

P1:
  full Video-Holmes train
  CG-Bench mini full
  SIV-Bench selected categories

P2:
  VRBench full
  CG-Bench full
  M3-Bench after graph reader
```

Full dataset recipes, labeling rules, and acceptance gates:
[expert-demo-rollouts-from-datasets.md](../atomic-skill-decomposition-and-assembly/expert-demo-rollouts-from-datasets.md).

## 3. Implementation Staging

### Stage A — Expert-Demo Reasoning Assembly (current)

- Build clue-memory graphs from dataset annotations with atomic graph-construction skills
- Fit reasoning skill graphs with teacher/LLM labeler over frozen ontology
- Datasets: Video-Holmes, CG-Bench mini
- **Skips raw-video perception**

### Stage B — Broader Reasoning Coverage

- CG-Bench full, selected VRBench, M3-Bench memory tasks
- Graph construction remains mostly offline

### Stage C — Video-Only Graph Construction

- Activate perception skills as tool-mediated actions
- Automatic captions/ASR, entity linking, graph edges without hidden clues
- Requires VLM/ASR pipeline integration

## 4. Runnable Code

### 4.1 Smoke Test (no API key)

Runs all 28 atomic skills on a synthetic social-contradiction example:

```bash
cd /fs/gamma-projects/vlm-robot/video_skills_relaunched
python experiments/smoke_test_atomic_skills.py
```

### 4.2 Graph Crafting from Video-Holmes (no API key)

`experiments/expert_demo_gpt5mini.py` exposes `load_video_holmes_example()` and
`build_seed_clue_memory_graph()`, which chain graph-construction atomic skills
over dataset annotations.

### 4.5 Dataset Clip Wrapper (no API key by default)

```bash
python dataset_clip_wrapper/smoke_test.py

python -m dataset_clip_wrapper.cli \
  --dataset video_holmes --regime short --limit 5 \
  --output dataset_clip_wrapper/output/video_holmes_short.jsonl
```

See [dataset_clip_wrapper/README.md](../dataset_clip_wrapper/README.md).

### 4.6 Expert Demo with LLM Labeling (API key required)

Reads `OPENROUTER_API_KEY` from `/fs/gamma-projects/vlm-robot/keys.py` or from
the environment. Default model: `openai/gpt-5-mini`.

```bash
# Toy example
python experiments/expert_demo_gpt5mini.py --dataset toy

# Video-Holmes
python experiments/expert_demo_gpt5mini.py --dataset video_holmes --split train --index 0
```

The LLM fits a `SkillGraphRollout` over the seed clue-memory graph. It does
not perform raw video perception.

### 4.4 Toy Two-Layer Graph Experiment (API key required)

```bash
python experiments/toy_graph_skill_reasoning.py
```

Uses synthetic perceived-video notes, asks the model to emit both graphs, and
runs a local verifier on cross-layer bindings.

## 5. What Is Not Implemented Yet

| Item | Notes |
|------|-------|
| Dataset adapters for CG-Bench / VRBench / SIV-Bench | Implemented in `dataset_clip_wrapper/` |
| Canonical JSONL export (`data/canonical_examples/`) | Use `dataset_clip_wrapper/cli.py` |
| True `hierarchical` coarse→fine segmentation | Documented; code uses flat sliding windows |
| `shot_boundary` / `scene_boundary` / `adaptive` strategies | Schema enum only |
| Port of legacy `Video_Skills` segmenter | Exists in sibling repo, not wired to relaunch |
| Raw VLM caption / ASR / object perception | Stage C |
| `create_semantic_memory_node` | Discussed in early skill drafts; not in final 9-skill set |
| M3-Bench memory graph reader | Blocker for M3 rollouts |
| Controller training / verified RL | Design only |
| `--graph-only` batch exporter | Not yet added |

## 6. Labeling Policy Summary

| Use rules (deterministic) | Use model (gpt-5-mini / gpt-oss-120) |
|---------------------------|--------------------------------------|
| Parse MCQ answers and timestamps | Trace segmentation |
| Map CG qid to clue clips | Skill fitting to frozen ontology |
| Parse SRT into subtitle chunks | Evidence-role labeling |
| Create initial EvidenceRef objects | Repair-target generation |
| Schema and format validation | |

Never invent new atomic skills during labeling.

## 7. Paper Scope Decisions

Documented in `problem-formulation-zh.html` and preserved here for implementation alignment:

- Paper 1 focuses on decomposition/assembly + query/verify/repair over teacher-built or dataset-seeded memory
- Store/update/merge/forget/streaming writer-reasoner split deferred to paper 2
- Streaming remains schema-compatible via `clip_policy.online` and `observation_end_s`
- Core evaluation trio: Video-Holmes + SIV-Bench + VRBench; CG-Bench and M3-Bench optional

## 8. Related Documents

- [Clip processing policy](clip-processing-policy.md)
- [Unified video skill schema](unified-video-skill-schema.md)
- [Atomic skills v1](../atomic-skill-decomposition-and-assembly/atomic-skills-v1.md)
- [Expert demo rollouts from datasets](../atomic-skill-decomposition-and-assembly/expert-demo-rollouts-from-datasets.md)
