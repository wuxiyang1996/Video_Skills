# Motif Layer Boundary

Last updated: 2026-07-06

This document fixes the cleanup decision for composed motifs on the clean
L1/L2 relaunch base.

In the three-agent architecture, this document defines Agent 3: the Motif
Extraction and Management Agent. Agent 3 uses a Qwen3.5/GPT-OSS agent pipeline
to inspect accepted L2 traces, propose reusable motifs, curate candidates, and
manage motif promotion, but it does not answer questions directly.

## Decision

Keep composed motifs as an optional L1/L2 planning layer, but implement that
layer as new, small code aligned with the current graph schemas. Do not re-add
the old `skill_agents/` stack as the motif runtime.

Agent 3 is therefore an LLM-backed registry/mining/management agent, not a
runtime black-box skill executor. The deterministic miner is a seed/fallback
and audit path, not the full Motif Agent.

```text
accepted SkillGraphRollout
  -> Qwen3.5 motif proposal agent
  -> GPT-OSS motif curator
  -> candidate motif registry
  -> deterministic seed/audit checks
  -> promotion gates
  -> motif retrieval as planning prior
  -> atomic graph expansion
  -> verifier / repair
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

## Agent Pipeline

Agent 3 follows the old skill-bank-agent shape, but on the current L1/L2 graph
schema:

```text
accepted rollout
  -> extractor agent proposes motif candidates
  -> curator agent approves / defers / vetoes candidates
  -> motif bank stores support stats and expansion templates
  -> promotion gates decide candidate vs promoted
```

The intended model split is:

| Stage | Default model | Role |
|-------|---------------|------|
| Motif proposal | `qwen/qwen3.5` | Read compact accepted rollout traces and propose reusable graph motifs. |
| Motif curation | `openai/gpt-oss-120b` | Filter proposals for reuse, leakage risk, expansion safety, and bank fit. |
| Deterministic seed/audit | local code | Extract trajectory/repair path seeds and provide offline fallback. |

This borrows the persistent bank and curator idea from the old `skill_agents/`
pipeline, but not the old package as a runtime dependency.

## Online Extraction Policy

Online motif extraction is allowed only as candidate mining:

```text
L2 rollout
  -> final verifier result
  -> accepted graph only
  -> compact accepted trace for Qwen/GPT-OSS motif agents
  -> canonicalize entity/time/option labels
  -> propose/curate reusable graph motifs
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
  agent.py
  __init__.py
  canonicalize.py
  miner.py
  llm_agent.py
  registry.py
  promotion.py
  expansion.py
```

Expected ownership:

| Module | Responsibility |
|--------|----------------|
| `agent.py` | High-level Motif Agent orchestration with `hybrid`, `llm`, and `deterministic` modes. |
| `llm_agent.py` | Qwen3.5/GPT-OSS extractor-curator adapter for motif proposal and bank-maintenance decisions. |
| `canonicalize.py` | Replace surface entities, timestamps, option labels, and dataset-specific terms with abstract roles. |
| `miner.py` | Deterministic seed/fallback extraction of trajectory-round and repair-subgraph path candidates. |
| `registry.py` | Store support, failure, dataset, task-family, example refs, and LLM curator metadata. |
| `promotion.py` | Apply support, verifier, confusion, leakage, and expansion gates. |
| `expansion.py` | Instantiate promoted motifs on current evidence and expand them into atomic skill nodes. |

## Training And Evaluation Boundary

Evaluation remains hard 0/1 or True/False at the answer/task level.

Training may use RLVR-style progressive rewards over valid schema, evidence
binding, verifier pass rate, local repair success, and final answer correctness.
Motif statistics may help shape planning or repair rewards, but final
acceptance still requires verifier-backed evidence and the task-level eval
metric remains binary.
