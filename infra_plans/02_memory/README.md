# 02_memory — Agentic Memory Layer

> **Layer purpose:** Persist grounded perception into a structured, queryable substrate that the controller can plan over and the harness can attach evidence to. The design is the **`SocialVideoGraph` + three-store** architecture: episodic / semantic / state, with an evidence layer.
>
> **Trainability in v1:** **None.** v1 implements the **"Stable Memory, Evolving Reasoning"** principle: memory is built and maintained by **fixed procedures** drawn from a manually versioned registry. Memory itself is not a learned policy. Adaptive memory policies are a Phase 2 / 3 item.

---

## Files in this folder

| File | What it is | When you read it |
|------|------------|------------------|
| [`agentic_memory_design.md`](agentic_memory_design.md) | The full memory design: three stores (**episodic**, **semantic**, **state**), the **evidence layer**, **entity-centric indexing**, the **lifecycle implementation table** (write / update / decay / refresh triggers), and the **Memory Procedure Registry** (`open_episode_thread`, `append_grounded_event`, `update_entity_profile`, `refresh_state_memory`, `compress_episode_cluster`, `attach_evidence_ref`, `resolve_entity_alias`, `revise_belief_state`, `mark_memory_conflict`). Includes the explicit distinction between **memory-management skills** (fixed) and **reasoning skills** (evolving). | When you implement any memory write / update / read; when you need to know *which* memory operation is allowed in v1; when you need to add a new memory procedure (must be added to the Registry, not improvised). |

---

## Why memory is fixed in v1

If both the substrate (memory) and the layer that reasons over it (controller + skills) evolved at the same time, training signals would chase a moving target. Locking memory in v1 means:

- the controller learns over a stationary state space,
- evaluation ablations are reproducible,
- failure modes can be attributed to the controller / verifier / retriever, not to memory drift.

See [`../00_overview/mvp_build_order.md`](../00_overview/mvp_build_order.md) for the full justification, and [`../05_skills/skill_extraction_bank.md`](../05_skills/skill_extraction_bank.md#03-separate-registries) for the **separate registries** policy that keeps memory procedures out of the evolving reasoning bank.

## What this layer is responsible for

- **Episodic** writes from grounded windows (one entry per grounded event, with evidence refs)
- **Semantic** consolidation of repeated episodic patterns (slow, periodic)
- **State** updates for the current social and spatial world (who is where, who knows what, who trusts whom)
- **Entity profiles** with face / voice IDs, aliases, cross-episode identity persistence
- **Conflict marking** and **confidence decay** under fixed rules
- **Compression / eviction** of episodic detail under fixed rules

All of these are implemented as **deterministic, testable functions** in the Memory Procedure Registry.

## What this layer does *not* do

- It does **not** decide *what to retrieve* in response to a question — that's the **retriever** subsystem in [`../03_controller/actors_reasoning_model.md`](../03_controller/actors_reasoning_model.md#2b-retriever-as-a-first-class-subsystem).
- It does **not** verify claims against evidence — that's the **verifier** subsystem in [`../03_controller/actors_reasoning_model.md`](../03_controller/actors_reasoning_model.md#2c-verifier-as-a-first-class-subsystem).
- It does **not** evolve, prune, or rewrite its own procedures — Phase 2 / 3 only.

## Open work

See [`../99_meta/plan_docs_implementation_checklist.md`](../99_meta/plan_docs_implementation_checklist.md) §2 — entity-profile schema is closed; remaining open items: write/update triggers, contradiction rules, confidence decay, compression / eviction, semantic refresh.
