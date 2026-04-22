# 06_training — Training Plan for the 8B Controller

> **Layer purpose:** Specify what is **trained**, in what order, with what data, against what reward — for the **only trainable component** in the system, the 8B controller. This is the training-side counterpart to [`../00_overview/mvp_build_order.md`](../00_overview/mvp_build_order.md): the build order tells you what to *build*; this folder tells you what to *train*.
>
> **Scope:** **8B controller only.** 72B / 32B specialists are frozen. Memory procedures, the harness, and the bank are not parameter-learned in v1 (the bank is curated; promotion is a non-RL human-reviewed step).

---

## Files in this folder

| File | What it is | When you read it |
|------|------------|------------------|
| [`training_plan_sft_grpo.md`](training_plan_sft_grpo.md) | The full **staged SFT → GRPO recipe**: headline recommendation (one 8B base + staged LoRAs), what to train vs not to train, **Stages 0 – 5** (substrate → SFT cold-start → GRPO planning/routing → GRPO verification/abstention → composite promotion → reflection-LoRA, last stage gated by synthesis preconditions), recommended training schedule, LoRA layout (LoRA-A planning/routing, LoRA-B verification/abstention, LoRA-C reflection/synthesis), reward components, and **anti-hacking constraints** carried over from the controller spec. | When you implement any training stage, build a reward function, define a LoRA adapter, or decide whether a given stage is ready to start. |

---

## What gets trained (one paragraph)

A **single 8B base model** with **staged LoRA adapters**. **LoRA-A** is trained first (planning + routing — hop generation, skill selection, retrieval triggers). **LoRA-B** comes second (verification + abstention — local check classification, support-threshold calibration, abstain decision). **LoRA-C** is the late-phase reflection / synthesis adapter and is added only after Stages 1–3 are stable and the [synthesis preconditions](../05_skills/skill_synthetics_agents.md#03-preconditions-for-later-self-evolution) hold. SFT cold-starts each adapter from teacher-labelled `ReasoningTrace` data; GRPO then optimizes against the reward shape in [`../03_controller/actors_reasoning_model.md`](../03_controller/actors_reasoning_model.md#2e-training-signals-for-the-controller).

## What does *not* get trained in v1

| Component | Why not |
|-----------|---------|
| **72B / 32B observers and reasoner** | Frozen large-VLM tools. Treated as visual / evidence specialists, not orchestrators. |
| **Harness** | Deterministic interpreter ([`../04_harness/`](../04_harness/README.md)). Execution-interface problem, not parameter-learning. |
| **Memory procedures** | "Stable memory, evolving reasoning" ([`../02_memory/`](../02_memory/README.md)). Memory is the substrate; not a learned policy in v1. |
| **Skill bank** | Structured registry, curated in v1 ([`../05_skills/skill_extraction_bank.md`](../05_skills/skill_extraction_bank.md#02-phase-1-bank-policy)). Promotion is a non-RL human-reviewed step. |

## Why staged SFT → GRPO and not pure RL

The controller's outputs are structured (`HopGoal`, `AtomicStepResult`, `VerificationResult`, `ReasoningTrace`). Pure RL from random init would burn most compute discovering basic format compliance. SFT cold-starts each adapter on teacher traces so GRPO starts from a competent prior; GRPO then refines the planning, routing, verification, and abstention decisions where reward shape actually changes the policy.

## Anti-hacking, briefly

The reward is **weighted, not summed naively**, with hard caps and shaping penalties for: over-retrieving, over-abstaining, no-progress hops, shortcut exploitation, verifier collusion, and trace padding. Full table in [`../03_controller/actors_reasoning_model.md`](../03_controller/actors_reasoning_model.md#2e2-anti-hacking-constraints).

## Open work

Implement the reward computation against `ReasoningTrace`, build the GRPO training loop, and validate the anti-hacking caps on a small probe set before scaling. Tracked in [`../99_meta/plan_docs_implementation_checklist.md`](../99_meta/plan_docs_implementation_checklist.md).
