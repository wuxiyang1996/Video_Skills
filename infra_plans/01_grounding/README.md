# 01_grounding — Visual Grounding Layer

> **Layer purpose:** Convert raw video into the canonical [`GroundedWindow`](../00_overview/runtime_contracts.md) objects that every downstream module consumes. This is where the **frozen 72B / 32B large VLMs** are wired in as visual specialists, alongside the perception stack (face / voice / scene / subtitle / entity resolution).
>
> **Trainability in v1:** None. Large VLMs are frozen; perception modules are configured, not learned.

---

## Files in this folder

| File | What it is | When you read it |
|------|------------|------------------|
| [`video_benchmarks_grounding.md`](video_benchmarks_grounding.md) | The **design plan**: benchmark landscape, the shared `SocialVideoGraph` schema, normative `GroundedWindow` wire format (§2.6), entity-resolution policy (§2.7), benchmark-to-capability matrix (§6.1), error taxonomy (§11). | When you need to understand *what* the grounding layer must produce and *why* it has the shape it does. |
| [`grounding_pipeline_execution_plan.md`](grounding_pipeline_execution_plan.md) | The **execution plan**: a Cursor-ready, file-by-file plan to build the pipeline by vendoring [m3-agent](https://github.com/bytedance/m3-agent)'s perception / entity / memory stack and layering the social-semantic schema on top. Phase 0 → Phase 6. Replaces the schema-only smoke test in `out/claude_grounding/` with `out/grounding_v1/`. | When you are about to implement any part of the grounding layer. |

---

## What this layer produces

Every module downstream consumes only one shape: `GroundedWindow` (defined in [`../00_overview/runtime_contracts.md`](../00_overview/runtime_contracts.md#21-groundedwindow)). It carries:

- adaptive-sampled clip windows aligned to scene cuts and subtitles
- entities with cross-clip identity (face + voice clusters resolved into a single `entity_id`)
- interactions, events, social hypotheses, dialogue spans
- evidence pointers (frame / clip / subtitle / voice modalities)
- per-field `confidence`, `provenance`, `inferred` flags

## What this layer does *not* do

- It does **not** decide which clips to ground in response to a question — that's the controller's retrieval-planning job (see [`../03_controller/`](../03_controller/README.md)).
- It does **not** reason multi-hop — that's the harness + skills + controller (see [`../04_harness/`](../04_harness/README.md), [`../05_skills/`](../05_skills/README.md)).
- It does **not** maintain long-term state — that's memory's job (see [`../02_memory/`](../02_memory/README.md)).

## Open work

Grounding-related items are mostly **closed** (see status update in [`../99_meta/plan_docs_implementation_checklist.md`](../99_meta/plan_docs_implementation_checklist.md)). The open items are: **executing** Phases 0 → 6 of the pipeline plan and integrating its outputs with the memory write triggers in [`../02_memory/agentic_memory_design.md`](../02_memory/agentic_memory_design.md).
