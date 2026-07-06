# Video_Skills — Infra Plans

> Design plans for the **Video_Skills** project: an evidence-grounded multi-hop video-reasoning system in which a small **trainable 8B controller** orchestrates **frozen large VLMs (72B / 32B)** as visual specialists, over a **stable structured memory** of grounded perception, while drawing reasoning programs from an **evolving reasoning skill bank**.
>
> This folder is the **normative source of truth** for the design. The repo's main [`readme.md`](../readme.md) is the user-facing entry point; this folder is what informs it.

---

## 0. Reading order

If you are new to the project, read in this order:

1. [`00_overview/system_overview.md`](00_overview/system_overview.md) — one-page integrative picture
2. [`00_overview/mvp_build_order.md`](00_overview/mvp_build_order.md) — what to build, in what order, with what success criterion
3. [`00_overview/runtime_contracts.md`](00_overview/runtime_contracts.md) — the typed objects every module passes around
4. Then dive into the layer folders (`01_grounding/` → `07_evaluation/`) in numerical order

---

## 1. Design principle (one paragraph)

**Stable memory, evolving reasoning.**
72B/32B models are **frozen specialists** for visual grounding, social/spatial extraction, and evidence-to-answer generation. The 8B is the **only trainable orchestrator**: question decomposition, hop planning, skill selection, retrieval, evidence sufficiency, verification, abstention, and (later) reflection. **Memory** is treated as a **fixed procedural substrate** with manually versioned operations, not a learned policy. The **evolving bank** is restricted to **reasoning skills** (atomic and composite), and even there, v1 is **curated**, not free-form. The first milestone is to prove that this stack beats direct large-VLM QA and naive retrieval baselines on long, multi-hop, perspective-bearing video-reasoning benchmarks — *before* any self-evolution is enabled.

Full statement: [`00_overview/system_overview.md`](00_overview/system_overview.md), [`00_overview/mvp_build_order.md`](00_overview/mvp_build_order.md).

---

## 2. Folder map (by system layer)

| Folder | Layer | What lives here |
|--------|-------|-----------------|
| [`00_overview/`](00_overview/README.md) | **Cross-cutting orientation** | system overview, MVP build order, canonical runtime contracts |
| [`01_grounding/`](01_grounding/README.md) | **Visual grounding layer** (frozen 72B/32B specialists + perception stack) | benchmark landscape, grounded-window schema, m3-agent–based pipeline plan |
| [`02_memory/`](02_memory/README.md) | **Agentic memory layer** (fixed substrate in v1) | three-store design (episodic / semantic / state), evidence layer, fixed memory procedures |
| [`03_controller/`](03_controller/README.md) | **8B controller / orchestrator** (the only trainable component) | role split, reasoning loop, retriever / verifier specs, training-signal table, anti-hacking constraints |
| [`04_harness/`](04_harness/README.md) | **Execution runtime** (non-trainable) | hop / atomic-step contract, harness runtime spec, MVP failure handling, MCP terminology map |
| [`05_skills/`](05_skills/README.md) | **Reasoning skill bank** (the evolving layer — restricted to reasoning skills) | bank schema, atomic / composite hierarchy, synthesis pipeline (later phases) |
| [`06_training/`](06_training/README.md) | **Training plan** for the 8B controller | staged SFT → GRPO recipe, LoRA layout, reward shaping, anti-hacking caps |
| [`07_evaluation/`](07_evaluation/README.md) | **Evaluation & ablation matrix** | external baselines, subsystem ablations, error buckets, MVP eval priority |
| [`99_meta/`](99_meta/README.md) | **Tracking** | per-file implementation checklist, what is open vs closed |

---

## 3. Layer dependency graph

```
                         ┌─────────────────────────────┐
                         │ 00_overview                 │
                         │ system_overview             │
                         │ mvp_build_order             │
                         │ runtime_contracts (schemas) │
                         └──────────────┬──────────────┘
                                        │ defines typed objects all layers consume
            ┌───────────────────────────┼───────────────────────────┐
            ▼                           ▼                           ▼
  ┌────────────────┐         ┌────────────────────┐       ┌──────────────────┐
  │ 01_grounding   │  ───►   │ 02_memory          │  ───► │ 03_controller    │
  │ (frozen 72B)   │ writes  │ (fixed procedures) │ reads │ (trainable 8B)   │
  └────────────────┘         └────────────────────┘       └────────┬─────────┘
                                                                   │ plans hops, picks skills
                                                                   ▼
                                                         ┌──────────────────┐
                                                         │ 04_harness       │ ◄──── 05_skills
                                                         │ (executes hops,  │      (atomic + composite
                                                         │  logs traces)    │       reasoning skills)
                                                         └────────┬─────────┘
                                                                  │ ReasoningTrace
                                                                  ▼
                                                ┌──────────────────────────────┐
                                                │ 06_training (SFT → GRPO)     │
                                                │ 07_evaluation (ablations)    │
                                                └──────────────────────────────┘
```

The **direction of trainability** is monotonic: layers `01` / `02` / `04` are stable substrate, layer `03` is the trainable controller, layer `05` is the evolving bank (reasoning only, conservative in v1).

---

## 4. MVP success criterion

> Outperform **direct large-VLM QA** and **naive retrieval** baselines on **evidence-grounded multi-hop video reasoning** using the 8B controller over structured memory and grounded evidence.

Defined in [`00_overview/mvp_build_order.md`](00_overview/mvp_build_order.md#mvp-success-criterion). Measured by the matrix in [`07_evaluation/evaluation_ablation_plan.md`](07_evaluation/evaluation_ablation_plan.md). Not measured by skill-bank size, not by self-evolution metrics, not by per-component cleverness.

---

## 5. What is *not* in scope for v1

- Adaptive memory policies (memory writes / decay / refresh remain fixed procedures)
- Free-form skill-bank growth (bank is curated; promotion is human-reviewed and shadow-deployed)
- Trace-driven synthesis at production scale (synthesizer ships behind a feature flag with doubled thresholds)
- Training the 72B/32B specialists (frozen, used only as visual / evidence tools)

These are **Phase 2** and **Phase 3** items in [`00_overview/mvp_build_order.md`](00_overview/mvp_build_order.md), gated by preconditions in [`05_skills/skill_synthetics_agents.md`](05_skills/skill_synthetics_agents.md#03-preconditions-for-later-self-evolution).

---

## 6. Conventions inside this folder

- Each layer folder has its own `README.md` describing what's inside, who reads it, and what's still open.
- Cross-references between plan files use **relative paths** (e.g. `../03_controller/actors_reasoning_model.md`). The link rewriter in `tools/` (if added later) keeps these consistent on rename.
- Open work in each layer is tracked in [`99_meta/plan_docs_implementation_checklist.md`](99_meta/plan_docs_implementation_checklist.md) and summarized in [`00_overview/system_overview.md`](00_overview/system_overview.md#after-this-what-still-needs-to-be-worked-out).
- "Phase 1 / Phase 2 / Phase 3" terminology in any plan refers to [`00_overview/mvp_build_order.md`](00_overview/mvp_build_order.md). It does **not** refer to grounding-pipeline phases (which are numbered 0–6 in [`01_grounding/grounding_pipeline_execution_plan.md`](01_grounding/grounding_pipeline_execution_plan.md)).
