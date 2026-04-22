# MCP vs. Agent Harness — Terminology Map (infra_plans)

> **Purpose:** Relate external vocabulary (**MCP**, **agent harness**) to what this repo’s `infra_plans` already describe under other names. This file does **not** introduce new runtime requirements; it is a naming and layering map for readers and tooling (e.g. Cursor integrations).
>
> **Related plans:**
>
> - [Actors / Reasoning Model](../03_controller/actors_reasoning_model.md) — 8B controller, orchestration, frozen tools, when-to-call rules
> - [Video Benchmarks & Grounding](../01_grounding/video_benchmarks_grounding.md) — shared reasoning protocol, benchmarks, adapters
> - [Agentic Memory](../02_memory/agentic_memory_design.md) — memory stores and evidence layer the controller reads/writes

---

## 1. What the plan docs do *not* say literally

- **`MCP` (Model Context Protocol):** The current `infra_plans` markdown does not standardize or name MCP. Tool-like behavior appears as **frozen VLMs**, **observers**, **embedders**, and **graph/memory primitives**, not as a protocol-level abstraction.
- **`Harness`:** The word does not appear as a named component. The same *role* is spread across **controller / orchestrator**, **reasoning loop**, and **benchmark execution**.

So: search for exact strings `MCP` or `harness` will miss the ideas; use the equivalents below.

---

## 2. Equivalent terms already in these plans

| External / informal term | What to read for the same idea in `infra_plans` |
|--------------------------|--------------------------------------------------|
| **Tool interface** | Frozen tools the 8B controller invokes: Observer-A, Observer-B, Reasoner, embedding models; plus explicit **when-to-call** rules per stage ([Actors / Reasoning Model](../03_controller/actors_reasoning_model.md)). |
| **Controller / orchestrator loop** | The **trainable 8B controller**: memory updates, retrieval plans, skill selection, prompt composition, evidence sufficiency, reflection — the central runtime that decides *what* runs *when* ([Actors / Reasoning Model](../03_controller/actors_reasoning_model.md)). |
| **Evaluation runner** | Benchmark tiers, **direct vs retrieval** regimes, and adapters in [Video Benchmarks & Grounding](../01_grounding/video_benchmarks_grounding.md); episodic scoring and trace expectations vary by benchmark but share a common reasoning style. |
| **Shared execution API** | The documented multi-step protocol (**`[Think]` → `[Search]` → `[Think]` → `[Answer]`**), **`reason(...)`**, and direct vs retrieval modes — a reusable “outer loop” shape across benchmarks ([Video Benchmarks & Grounding](../01_grounding/video_benchmarks_grounding.md)). |

---

## 3. A practical two-layer framing (for integrations, not a spec)

These labels help separate **what gets called** from **who runs the loop**:

### 3.1 MCP-like layer (standardized tool surface)

**Meaning here:** A stable, swappable interface for **observer / grounding / retrieval / reasoner** calls (inputs, outputs, errors), whether or not it ever matches a real MCP server.

**In current plans:** Described as **frozen tool contracts** and controller **routing rules**, not as a named wire protocol.

**Gap vs literal MCP:** The repo design reads as **harness-first** (orchestration + loop + logging); an explicit **MCP-shaped** layer would be an optional standardization on top of those tool contracts.

### 3.2 Harness layer (runtime)

**Meaning here:** The code path that **executes** `reason(...)`, **logs traces**, **manages retries**, **routes** to tools, and **scores** benchmark episodes.

**In current plans:** Closest single anchor is the **8B controller/orchestrator** plus the **shared reasoning protocol** and benchmark **evaluation** story — together, the natural place a “harness” would sit.

---

## 4. Concise read for repo maintainers

| Question | Short answer |
|----------|----------------|
| Is **MCP** defined in `infra_plans`? | **No** — not by name or as a protocol. |
| Is there a **harness-like** design? | **Yes** — implicitly: **controller + frozen tools + shared `reason` / `[Think]`… protocol + benchmark runners**. |
| What would “add MCP” mean here? | **Standardize** the tool surface (schemas, discovery, errors) for observers/reasoner/retrieval — orthogonal to defining the controller loop. |

---

## 5. Optional next step (out of scope for this note)

Mapping current files into a clean **MCP layer + harness layer** diagram for Cursor (or other agents) is a documentation/implementation exercise: this file only fixes **terminology** and **where to look** in existing plans.
