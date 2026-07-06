# 03_controller — 8B Controller / Orchestrator

> **Layer purpose:** Specify the **only trainable component** in the system: a small 8B model that orchestrates everything else. The controller does not look at raw pixels. It plans hops, picks reasoning skills, calls the retriever, judges evidence sufficiency, calls the verifier, and decides answer vs abstain. The retriever and verifier live here too because they are the controller's first-class subsystems.
>
> **Trainability in v1:** **Yes — this is the trainable layer.** Trained via staged SFT → GRPO with LoRA adapters; see [`../06_training/`](../06_training/README.md).

---

## Files in this folder

| File | What it is | When you read it |
|------|------------|------------------|
| [`actors_reasoning_model.md`](actors_reasoning_model.md) | The full controller spec: **role split** (72B vs 8B vs harness), **MVP controller objective**, **first-phase training focus**, **canonical runtime data contracts** (§2A), **retriever** as a first-class subsystem (§2B), **verifier** as a first-class subsystem (§2C), **abstention policy** (§2D), and **training signals** (§2E) including the reward / supervision table and **anti-hacking constraints** (over-retrieving, over-abstaining, no-progress hops, shortcut exploitation, verifier collusion, trace padding). Also defines the **`[Think] → [Search] → [Think] → [Answer]` reasoning loop** and direct vs retrieval modes. | When you implement any controller behavior, define any reward, or wire the retriever / verifier; when you need to know what the controller *does* at each stage and what it *outputs* for logging / training. |

---

## What the controller does (in one paragraph)

For every question, the controller (i) decomposes it into hops, (ii) for each hop selects a reasoning skill from [`../05_skills/`](../05_skills/README.md) (atomic or composite), (iii) calls the retriever to fetch evidence from [`../02_memory/`](../02_memory/README.md), (iv) calls the verifier to check the hop's claim against that evidence, (v) decides whether to continue, abstain, or answer, and (vi) emits a `ReasoningTrace` covering every step. The harness in [`../04_harness/`](../04_harness/README.md) does the actual execution; the controller does the *deciding*.

## Subsystems specified in this folder

| Subsystem | Section | Role |
|-----------|---------|------|
| **Controller core** | §0 – §2 | Question decomposition, hop planning, skill routing, output schema |
| **Retriever** | §2B | Query rewriting, entity / time / perspective filters, top-k fusion, broaden ladder |
| **Verifier** | §2C | Local checks (`claim_evidence_alignment`, `evidence_sufficiency`, `temporal_consistency`, `perspective_consistency`, `entity_consistency`), `support_threshold`, `abstain_threshold` |
| **Abstention policy** | §2D | When to refuse vs continue vs answer |
| **Training signals** | §2E | Reward / supervision table; anti-hacking constraints |

## What this layer does *not* do

- It does **not** ground pixels — that's [`../01_grounding/`](../01_grounding/README.md).
- It does **not** mutate memory state — only fixed Memory Procedures in [`../02_memory/`](../02_memory/README.md) do that, called via the harness.
- It does **not** execute hops — that's [`../04_harness/`](../04_harness/README.md). The controller decides; the harness runs.
- It does **not** mint new reasoning skills — synthesis is in [`../05_skills/`](../05_skills/README.md), gated to later phases.

## Why the controller, the retriever, and the verifier sit in one document

They are tightly coupled: the controller decides what to retrieve, what to verify, and how to react to verification results. Splitting them into three plans would bury the closed-loop relationship that defines the MVP. The current single document keeps the loop legible. If they grow further, they can be split into three side-by-side files within this folder.
