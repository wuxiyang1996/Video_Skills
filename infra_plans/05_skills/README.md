# 05_skills — Reasoning Skill Bank

> **Layer purpose:** Define the **evolving layer** of the system — but with a strict scope: only **reasoning skills** belong here. Atomic reasoning operators (entity grounding, temporal linking, causal linking, perspective check, evidence sufficiency, abstention, …) and the composites that chain them. Memory-management operations live in [`../02_memory/`](../02_memory/README.md), not here.
>
> **Trainability in v1:** **Curated**, not free-form. The starter inventory is hand-authored. Synthesis from traces ships behind a feature flag with doubled phase-1 thresholds and human-in-the-loop activation. Free-form bank growth is a Phase 3 item.

---

## Files in this folder

| File | What it is | When you read it |
|------|------------|------------------|
| [`skill_extraction_bank.md`](skill_extraction_bank.md) | The full **bank specification**: design principle, **bank scope** (reasoning only), **phase-1 bank policy** (curated, no free growth), **separate registries** (Memory Procedure Registry vs Reasoning Skill Bank), formal **`SkillRecord` schema** (`SkillRecord`, `SkillUsage`, `SkillVersion`, `TriggerSpec`, `VerificationCheckSpec`), **canonical starter atomic skill inventory** (§4.9), **composite skill formation rules**, trigger and verification formats. | When you add or modify a reasoning skill; when you implement the bank loader / serializer / RAG-style selection layer; when you need to know what the bank's wire format is. |
| [`skill_decomposition_atomic_composition.md`](skill_decomposition_atomic_composition.md) | The **conceptual hierarchy**: atomic operators as the only executable units (perception / memory / reasoning / control atoms), composite skills as DAGs over atoms, synthesis from repeated atomic traces, failure-driven atomic repair. Provides the rationale and worked examples behind the formal schema in `skill_extraction_bank.md`. | When you need to understand *why* the bank is split into atomic vs composite, or when you are designing a new family of atoms. |
| [`skill_synthetics_agents.md`](skill_synthetics_agents.md) | The **synthesis pipeline** spec: trace-first synthesis (not segment-tag-first), verifiability and non-leakiness gates, **promotion thresholds**, **bank versioning and rollback**, failure taxonomy, update rules (patch / split / merge / retire / promote). Includes the **phase-1 conservative synthesis policy** and the **preconditions for later self-evolution**. | When you implement the synthesizer (gated to later phases); when you add a new failure category; when you change a promotion threshold or rollback rule. |

---

## What "skill" means here, precisely

A **reasoning skill** is a typed reasoning operator with:

- **input schema** (what slots it expects),
- **output schema** (what it produces),
- **trigger conditions** (when the controller should pick it),
- **verification rule** (what must hold for its output to count),
- **failure modes** (what going wrong looks like),
- **required memory fields** (what it reads from memory),
- **usage stats** and **version history**.

Atomic skills are minimal operators. Composite skills are short, stable chains of atomic skills. **Neither** is a free-form prompt or an opaque LLM call.

## What is *not* a skill in this layer

- **Memory operations** (`open_episode_thread`, `append_grounded_event`, …) — these are **Memory Procedures** in [`../02_memory/`](../02_memory/README.md), governed by a separate, manually versioned registry. They do not evolve.
- **Pipeline primitives** (`detect_faces`, `embed_voice`, `chunk_video`, …) — these are infrastructure, defined in [`../01_grounding/grounding_pipeline_execution_plan.md`](../01_grounding/grounding_pipeline_execution_plan.md). They do not evolve.
- **Tool calls** to frozen 72B/32B models — wrapped as primitives, not as evolving skills.
- **Scene / action / intention tags** (NAVIGATE, MANIPULATE, …) — kept only as auxiliary metadata or trigger features, not as skill ontology.

## Why synthesis is gated in v1

If the bank were allowed to grow freely while the controller was being trained, the controller's action space would shift mid-training and the GRPO reward signal would degrade. v1 ships with a curated atomic inventory + a small set of composites, demonstrates the MVP success criterion, and only then turns on synthesis. The full preconditions are listed in [`skill_synthetics_agents.md`](skill_synthetics_agents.md#03-preconditions-for-later-self-evolution).

## Open work

See [`../99_meta/plan_docs_implementation_checklist.md`](../99_meta/plan_docs_implementation_checklist.md) §4 (bank schema, starter set, formation rules) and §5 (trace-first rewrite, verifiability gates, promotion thresholds).
