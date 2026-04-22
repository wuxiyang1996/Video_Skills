# 04_harness — Execution Runtime

> **Layer purpose:** Define the **non-trainable execution runtime** that takes a hop plan from the controller, expands composite reasoning skills into their atomic chains, binds entity slots, calls memory procedures and frozen 72B/32B tools, runs the verifier locally, and writes a `ReasoningTrace`. The harness is the boundary between *deciding* (controller) and *doing* (everything else).
>
> **Trainability in v1:** **None.** The harness is a deterministic interpreter, not a learned policy.

---

## Files in this folder

| File | What it is | When you read it |
|------|------------|------------------|
| [`atomic_skills_hop_refactor_execution_plan.md`](atomic_skills_hop_refactor_execution_plan.md) | The full **harness runtime spec**: hop length and termination rules, atomic-step input/output contract, composite expansion rules, **trace logging format**, **MVP harness mission**, **MVP failure handling** (retrieval failure, grounding failure, unsupported claims, perspective mismatch, premature answers — all without autonomous repair in v1), and the failure-localization protocol. | When you implement the harness loop, add a new atomic primitive, define a new failure handler, or wire a composite skill's expansion. |
| [`mcp_harness_terminology_map.md`](mcp_harness_terminology_map.md) | A **terminology map** (not a runtime spec) for readers coming from external vocabularies (MCP, "agent harness"). Explains what term in this repo corresponds to what external term, so external integrations and tooling don't get lost in naming. | When integrating with external MCP-style tooling or onboarding readers familiar with `agent harness` / `MCP server` lingo. |

---

## Why "harness" exists as its own layer

Without a clean execution layer, the controller and the skill bank would each grow their own ad-hoc interpreters, and trace logging would diverge across runs. Putting execution in one place gives:

- a **single point** that produces `AtomicStepResult` and `HopRecord` (defined in [`../00_overview/runtime_contracts.md`](../00_overview/runtime_contracts.md)),
- a **single trace format** that SFT, GRPO, ablation, and synthesis all consume,
- a **single failure-handling policy** in v1 (no autonomous repair; conservative fallback + log + surface to controller).

## The harness contract (one paragraph)

Given a `HopGoal` and a chosen skill (atomic or composite), the harness expands the skill into a sequence of atomic steps, binds each step's inputs from the current `EvidenceBundle` and the controller-provided entity slots, calls the appropriate primitive (a Memory Procedure from [`../02_memory/`](../02_memory/README.md), a frozen 72B/32B tool from [`../01_grounding/`](../01_grounding/README.md), or a pure reasoning function from [`../05_skills/`](../05_skills/README.md)), runs the step's `verification_rule` locally, and emits one `AtomicStepResult` per step plus one `HopRecord` per hop. It does **not** decide what to do next on failure — it surfaces structured failures to the controller, which decides.

## What this layer does *not* do

- It does **not** plan hops or pick skills — that's the controller in [`../03_controller/`](../03_controller/README.md).
- It does **not** mutate memory directly — only via the Memory Procedure Registry.
- It does **not** synthesize new skills — synthesis lives in [`../05_skills/`](../05_skills/README.md) and is gated to later phases.
- It does **not** make abstain / answer decisions — those come from the controller after reading the trace.

## On "MCP" specifically

This repo does **not** standardize on Model Context Protocol as a wire format. Frozen tools are called through internal Python APIs. The terminology map in `mcp_harness_terminology_map.md` exists so contributors who think in MCP terms can locate the corresponding repo concepts without inferring a protocol that isn't there.
