# Repository Bundle Map

Last updated: 2026-07-24

This map classifies the current physical package layout. The top-level
`dataset_clip_wrapper` package keeps a few compatibility entrypoints, while
implementation code lives in bundle subpackages.

Empty ghost top-level packages `visual_grounding/`, `rag/`, and `video_skills/`
are not active ownership paths and were cleared from the clean base. Canonical
Motif Agent ownership is `dataset_clip_wrapper/motifs/`; transitional top-level
`motif/` remains only for some SFT pilot scripts/tests until consolidation.

The high-level agent split maps to bundles as follows:

| Agent | Bundle ownership |
|-------|------------------|
| L1 Graph Crafter | `dataset_clip_wrapper/perception/`, `dataset_clip_wrapper/l1_clue_graph/` |
| L2 Recursive Reasoning / Answer Agent | `dataset_clip_wrapper/l2_reasoning_graph/`, `dataset_clip_wrapper/verification/` |
| Motif Extraction and Management Agent | `dataset_clip_wrapper/motifs/` |

## Functional Map

```text
perception -> L1 -> L2 -> verification/repair -> motif
training (SFT adapters) sits across L1/L2/repair/verifier/motif exports
```

Top-level tests mirror this split: `tests/motif/` for motif bank/miner
coverage, `tests/sft/` for LoRA SFT / quality-gate / split / specialist-expansion
coverage. Self-repair prior-work notes live under `docs/self-repair-prior-work.md`
(formerly top-level `reflection/`).

## High-Level Bundles

| Bundle | Scope | Main paths |
|--------|-------|------------|
| Atomic skill basis | Frozen executable skill functions and skill backend plumbing | `atomic_skills/` |
| Dataset adapters | Dataset-specific readers into canonical examples | `dataset_clip_wrapper/adapters/` |
| Perception / clip tools | Clip policies, Qwen/video-tools clip schemas, subtitle/video probes, model clients | `dataset_clip_wrapper/perception/` |
| L1 clue graph | Question-agnostic clue-memory graph construction, retrieval, graph compose, L1 gating | `dataset_clip_wrapper/l1_clue_graph/` |
| L2 reasoning graph | Question-conditioned reasoning rollout, GPT-OSS planner, recursive trace shell | `dataset_clip_wrapper/l2_reasoning_graph/` |
| L2 repair / verifier | Evidence-gap repair, option evidence selection, verifier gates, final acceptance reports | `dataset_clip_wrapper/verification/` |
| Expert demos / manifests | Verified expert-demo export plus split-aware manifests | `dataset_clip_wrapper/expert_demos/`, `dataset_clip_wrapper/manifests/` |
| Motif extraction and management | Qwen3.5/GPT-OSS motif proposal, curation, registry, promotion, deterministic seed/fallback, and atomic expansion templates | `dataset_clip_wrapper/motifs/` |
| Pipeline runners | End-to-end orchestration commands | `dataset_clip_wrapper/runners/` |
| Smoke tests | Small executable boundary checks | `dataset_clip_wrapper/tests/` |
| Generated artifacts | API outputs, staged caches, repair outputs | `dataset_clip_wrapper/output/` |

## Atomic Skills

`atomic_skills/` is the frozen action basis. Keep primitive skills here; do not
hide benchmark-specific logic inside these functions.

| Sub-bundle | Role |
|------------|------|
| `evidence_graph_construction/` | L1-oriented atomic graph construction skills: clip, observation, entity, event, state, provenance |
| `reasoning_graph_assembly/` | L2-oriented reasoning, hypothesis, bridge, verification, and answer-commit skills |
| `skill_executor.py` | Dispatch layer that chooses rule, LLM, VLM, or verifier execution |
| `skill_backends.py`, `skill_model_client.py` | Backend configuration and model API transport |
| `registry.py`, `common.py` | Ontology export, stable ids, result helpers, shared utilities |

## Dataset Clip Wrapper Bundles

The machine-readable registry lives in
`dataset_clip_wrapper/module_bundles.py`. Run
`python -m dataset_clip_wrapper.tests.smoke_test_module_bundles` after adding new
wrapper modules.

### Core Schema And Config

Purpose: canonical examples, config classes, and dataset profile defaults.

Modules:

```text
__init__.py
cli.py
dataset_graph_presets.py
pipeline.py
schemas.py
```

### Dataset Adapters

Purpose: keep benchmark quirks outside L1/L2 logic.

Modules:

```text
adapters/base.py
adapters/video_holmes.py
adapters/cg_bench.py
adapters/vrbench.py
adapters/siv_bench.py
adapters/streaming_video.py
```

### Perception And Clip Tools

Purpose: produce structured clip schemas and clip references from video-only
inputs. These modules should not commit answers.

Modules:

```text
perception/backbone.py
perception/clip_policy.py
perception/clip_schema.py
perception/openrouter_client.py
perception/subtitles.py
perception/video_probe.py
perception/video_tool_backend.py
```

### L1 Clue Graph

Purpose: build and query question-agnostic visual clue memory. L1 may record
missing visual clues or answerability gaps, but should not turn commonsense into
visual evidence.

Modules:

```text
l1_clue_graph/clip_retrieval.py
l1_clue_graph/clue_memory.py
l1_clue_graph/gate_l1_for_l2.py
l1_clue_graph/graph_composer.py
l1_clue_graph/graph_plan_validator.py
l1_clue_graph/skill_graph_bridge.py
```

### L2 Reasoning Graph

Purpose: consume `question + L1 graph`, build reasoning rollouts, and record
bounded recursive trajectories.

Modules:

```text
l2_reasoning_graph/fault_repair.py
l2_reasoning_graph/l2_recursive_trace.py
l2_reasoning_graph/reasoning_planner.py
l2_reasoning_graph/reasoning_rollout.py
```

### L2 Repair And Verification

Purpose: diagnose missing evidence, run bounded repair, verify option-specific
evidence packs, and produce final acceptance reports.

Modules:

```text
verification/evaluate_l1_query_memory.py
verification/evaluate_vrbench_video_only_graph.py
verification/report_evidence_audit.py
verification/report_final_acceptance.py
verification/report_failure_taxonomy.py
verification/report_l1_l2_quality.py
verification/retrofit_l2_trajectory.py
verification/run_repair_protocol.py
verification/runtime_verifier.py
```

### Pipeline Runners

Purpose: compose datasets, perception, L1, L2, repair, and reporting into
executable commands.

Modules:

```text
runners/llm_pipeline.py
runners/run_llm_pipeline.py
runners/run_staged_llm_pipeline.py
```

Top-level modules with the same command names remain as compatibility
entrypoints for `python -m dataset_clip_wrapper.run_repair_protocol`,
`python -m dataset_clip_wrapper.run_staged_llm_pipeline`, and related commands.

### Expert Demos And Manifests

Purpose: export accepted/abstaining L1/L2/repair trajectories and build
split-aware manifests without exposing hidden supervision to `video_only`
inputs.

Modules:

```text
expert_demos/export_expert_demos.py
manifests/build_training_manifests.py
```

Compatibility entrypoints:

```text
dataset_clip_wrapper/export_expert_demos.py
dataset_clip_wrapper/build_training_manifests.py
```

### Motif Extraction And Management

Purpose: use a Qwen3.5/GPT-OSS motif agent to propose and curate reusable
verified subgraph motifs from accepted L2 rollouts, then use promoted motifs as
optional planning/repair priors.

Modules:

```text
motifs/agent.py
motifs/build_motif_bank.py
motifs/canonicalize.py
motifs/llm_agent.py
motifs/miner.py
motifs/registry.py
motifs/promotion.py
motifs/expansion.py
```

This package consumes final acceptance reports or expert-demo JSONL files. It
defaults to a `hybrid` agent mode: Qwen3.5 proposes motif candidates, GPT-OSS
curates bank decisions, and deterministic trajectory/repair mining is used as a
seed/fallback/audit path. Future L1/L2 agents may retrieve promoted records as
planning priors, but every motif must still expand back into frozen atomic
graph nodes before execution.

## Cleanup Rules

- Keep `dataset_clip_wrapper/output/` as generated artifacts. Only
  `.gitkeep` should be tracked.
- Keep `__pycache__/`, `*.pyc`, `.pytest_cache/`, `.env`, and `.venv*` out of
  git.
- Keep this clean base free of legacy top-level directories such as
  `cold_start/`, `data_structure/`, `decision_agents/`, `dataset_examples/`,
  `skill_agents/`, and `trainer/` unless they are intentionally reintroduced
  with a documented migration plan.
- Do not reintroduce cleared ghost packages `visual_grounding/`, `rag/`, or
  `video_skills/` without documented ownership and tracked sources.
- Do not add benchmark-specific answer shortcuts to L1 graph construction or
  atomic skills.
- Heuristic retrieval/scoring may remain diagnostic, but final acceptance must
  come from GPT-OSS evidence selection/verifier or explicit bridge verification.
- Prefer adding new modules to an existing bundle and updating
  `module_bundles.py`; create a new bundle only when the ownership boundary is
  genuinely new.
- Do not use `skill_agents/` as the motif runtime. Treat it as an external
  reference only; any useful GRPO, LoRA, reward, or promotion utility must be
  ported explicitly before use.

## Compatibility Policy

The first physical refactor keeps top-level compatibility aliases in
`dataset_clip_wrapper/__init__.py` and thin `python -m` entrypoints for common
commands. New code should import from the bundle paths directly, for example:

```python
from dataset_clip_wrapper.l1_clue_graph.clue_memory import extract_clue_memory_graph
from dataset_clip_wrapper.l2_reasoning_graph.reasoning_planner import build_llm_reasoning_rollout
from dataset_clip_wrapper.verification.run_repair_protocol import main
```

Once downstream scripts have migrated, the compatibility aliases can be removed
in a separate cleanup commit.
