# Controller Training Plan — Staged SFT → GRPO

> Goal: Define the **staged SFT → GRPO training plan** for the system. Concretely: **train the 8B controller as the main adaptive module**, keep the 72B/32B observers and reasoner frozen, keep the harness and memory procedures fixed, and stage reflection/synthesis training behind a stable runtime substrate.
>
> This file is the training-side counterpart to [MVP Build Order](mvp_build_order.md): the build order tells you what to **build**; this file tells you what to **train**, in what order, with what data, against what reward.
>
> **Related plans:**
> - [MVP Build Order](mvp_build_order.md) — phase 1 / 2 / 3 sequencing the substrate this plan trains over
> - [Actors / Reasoning Model](actors_reasoning_model.md) — 8B controller, role split, runtime contracts, training signal table
> - [Agentic Memory](agentic_memory_design.md) — fixed memory procedures (not trained)
> - [Atomic Skills & Hop Refactor](atomic_skills_hop_refactor_execution_plan.md) — harness runtime (not trained)
> - [Skill Extraction / Bank](skill_extraction_bank.md) — reasoning skill bank (curated v1; promoted later)
> - [Skill Synthetics Agents](skill_synthetics_agents.md) — synthesis pipeline (later phases)
> - [Evaluation & Ablation Plan](evaluation_ablation_plan.md) — how each training stage is measured
> - [System Overview and Open Work](system_overview_and_open_work.md) — integrative view this plan slots into

---

## 0. Headline recommendation

Use **one 8B base model with staged LoRAs**. Do not train many separately tuned full models, and do not GRPO-train the 72B/32B models in this project version. The current actor plan ([Actors §0.1 — Role Split](actors_reasoning_model.md#01-role-split-72b-vs-8b-vs-harness)) already puts almost all adaptive behavior on the 8B side: question decomposition, hop planning, skill selection, memory retrieval, evidence sufficiency, verification, abstention, and (later) reflection / skill-bank updates.

In one sentence:

> **Train the 8B controller as the main adaptive module via staged LoRAs (SFT first, then GRPO), keep 72B/32B as frozen inference tools, keep harness and memory procedures fixed, and delay reflection / synthesis training until the runtime substrate is stable.**

---

## 1. What to train vs what not to train

### 1.1 Train (Phase 1)

**8B Controller / Orchestrator** — the only Phase-1 RL target. It must learn to:

- decompose a question into hop goals
- choose atomic / composite reasoning skills from the curated bank
- decide when to retrieve memory
- decide when to call 72B grounding tools
- decide when accumulated evidence is sufficient
- decide answer vs abstain
- emit structured hop traces and intermediate claims

These are exactly the controller responsibilities and outputs already specified in [Actors §2.3 / §2.6 / §2.7](actors_reasoning_model.md#23-what-the-8b-controller-does-at-each-stage) and the four phase-1 training behaviors in [Actors §0.3](actors_reasoning_model.md#03-first-phase-training-focus).

### 1.2 Optional later trainable LoRAs

Use separate LoRAs only if a single controller LoRA becomes unstable or under-fits. The drafted candidates:

| LoRA | Focus | When to add |
|---|---|---|
| **LoRA-A: Planning / Skill Routing** | Hop goal generation, skill selection, retrieval planning, evidence collection | Default Phase-1 LoRA; may be the only LoRA trained |
| **LoRA-B: Verification / Abstention** | Evidence sufficiency, contradiction checking, perspective correctness, answer-vs-abstain | Add only if planning + verification entangle in one LoRA and hurt either behavior |
| **LoRA-C: Reflection / Skill Evolution** | Failure classification, atomic-step localization, patch/split/retire/promote suggestions, synthesis-worthiness scoring | Add **only after** Stages 1–3 are stable and the [preconditions for self-evolution](skill_synthetics_agents.md#03-preconditions-for-later-self-evolution) are met |

### 1.3 Do NOT train

| Component | Why not |
|---|---|
| **72B / 32B observers and reasoner** | The plan treats them as **frozen** large-VLM tools for social extraction, spatial extraction, and evidence-to-answer generation ([Actors §0.1](actors_reasoning_model.md#01-role-split-72b-vs-8b-vs-harness)). |
| **Harness** | Runtime / execution layer that expands composites into atomics, calls memory + 72B primitives, logs traces. This is an **execution interface** problem, not a parameter-learning problem ([Atomic Skills & Hop Refactor — Harness](atomic_skills_hop_refactor_execution_plan.md#harness-runtime-specification)). |
| **Memory procedures** | Phase-1 direction is **stable memory, evolving reasoning** ([Agentic Memory §0.1](agentic_memory_design.md#01-fixed-memory-procedures-in-phase-1)). Memory is the substrate; it is not a learned policy in v1. |
| **Skill bank as a free-form model** | The bank is a **structured registry** of reasoning skills with conservative promotion later from repeated successful traces ([Skill Extraction / Bank §0.2](skill_extraction_bank.md#02-phase-1-bank-policy)). Not a separately trained free-form model in Phase 1. |

---

## 2. Recommended SFT → GRPO plan

The training stages mirror the build order in [MVP Build Order](mvp_build_order.md). RL never starts before the substrate it depends on is in place.

### 2.1 Stage 0 — Build the fixed substrate first (no training)

Before any RL, finish the minimum runtime substrate:

- canonical runtime schemas ([Actors §2A](actors_reasoning_model.md#2a-canonical-runtime-data-contracts))
- grounding outputs ([Grounding Pipeline](grounding_pipeline_execution_plan.md))
- fixed memory lifecycle and procedures ([Agentic Memory §0.2 / Lifecycle table](agentic_memory_design.md#lifecycle-implementation-table))
- retriever / verifier baseline ([Actors §2B / §2C](actors_reasoning_model.md#2b-retriever-as-a-first-class-subsystem))
- harness logging ([Harness runtime spec](atomic_skills_hop_refactor_execution_plan.md#harness-runtime-specification))

The middle-layer glue called out in [System Overview — Group A](system_overview_and_open_work.md#group-a--middle-layer-glue-that-must-be-nailed-down-before-implementation) and the [Plan Docs Implementation Checklist](plan_docs_implementation_checklist.md) is **higher priority** than aggressive evolution. RL training without these pieces produces non-reproducible traces and unlearnable rewards.

### 2.2 Stage 1 — SFT the 8B controller

Start with **SFT only on the 8B controller**. The objective is to make it imitate the target runtime behavior before any RL.

#### SFT targets

Train the controller to produce, in canonical format:

- `HopGoal` decomposition for a question
- selected atomic / composite `skill_id`s
- retrieval decisions (when to call retriever, what `RetrievalQuery`)
- evidence chains (`EvidenceBundle` references it cites)
- intermediate claims per hop
- `AbstainDecision` / final answer
- the full `ReasoningTrace` shape ([Actors §2A.7](actors_reasoning_model.md#2a7-reasoningtrace))

These are the controller outputs already enumerated in [Actors §2.7](actors_reasoning_model.md#27-controller-outputs-during-reasoning).

#### SFT data sources

Mix three sources:

1. **Teacher traces from 72B / 32B inference**
   - difficult clip grounding
   - curated evidence chains
   - high-quality answer justifications
2. **Rule-generated traces**
   - fixed memory procedure outputs
   - fixed retriever / verifier outputs
   - deterministic hop traces from starter reasoning skills (e.g. the §4.9 inventory)
3. **Synthetic positive / negative cases**
   - sufficient evidence → answer (positive)
   - insufficient evidence → abstain (positive)
   - wrong-perspective evidence → reject (negative example to learn rejection)
   - missing counterevidence check → failure label (negative)

All examples are stored as canonical `ReasoningTrace` objects so SFT, GRPO, and reflection consume the same format.

#### SFT scope

| Approach | When to use |
|---|---|
| **Default:** one main LoRA `8B_Controller_Main` | Always start here |
| **Fallback:** split into `8B_Controller_PlanningRouting` + `8B_Controller_VerificationAbstain` | Only if the single LoRA shows entangled regressions (planning improves, verification regresses, or vice versa) |
| **Do NOT add:** `8B_Controller_ReflectionSynthesis` at this stage | Reflection requires stable traces it does not yet have |

### 2.3 Stage 2 — GRPO on the 8B planning / routing policy

After SFT stabilizes the output **format** and basic reasoning style, use GRPO on the planning / routing **behavior**.

#### GRPO target

Optimize:

- better hop decomposition
- better skill selection from the curated bank
- better retrieval timing
- less wasted search
- stronger evidence chains
- fewer shallow or redundant hops

#### Reward design (planning / routing)

Use a weighted reward consistent with [Actors §2E.1](actors_reasoning_model.md#2e1-reward--supervision-table):

| Term | Source | Sign | Purpose |
|---|---|---|---|
| `r_answer` | final answer vs gold | + | Final correctness |
| `r_hop` | per-hop `hop_verification.passed` against the hop's `success_predicate` | + | Each hop achieves its local subgoal |
| `r_retrieval` | retrieval recall + precision against gold evidence | + | Reward useful retrieval; penalize useless / repeated retrieval |
| `r_evidence` | `EvidenceBundle.refs` precision and support quality | + | Cited evidence actually supports the claim |
| `r_efficiency` | `cost.hops`, `cost.retrieval_calls`, `cost.atomic_steps` against budget | − | Penalize too many hops, too many search calls, bloated traces |
| `r_perspective` | `perspective_consistency` check on perspective-bound claims | + | Character-perspective correctness where applicable |

This matches the controller's actual responsibilities and the checklist's call for a concrete reward / supervision table over decomposition quality, retrieval recall, evidence precision, perspective correctness, abstention correctness, and final-answer correctness.

#### Anti-hacking constraints (planning / routing)

Carry over the hard caps and shaping penalties in [Actors §2E.2](actors_reasoning_model.md#2e2-anti-hacking-constraints):

- **over-retrieving** — `p_extra_retrieval`, hard cap at 2× budget
- **repeated empty searches** — count against `cost.retrieval_calls` even when empty
- **answering without evidence** — `final_evidence.refs == []` ⇒ `r_answer = 0` regardless of literal match
- **many low-value hops** — no-progress hops penalized at 0.5× cost; 3 consecutive no-progress hops trigger automatic abstention
- **abstain as a safe default** — symmetric `r_abstain_correct`; rolling abstention rate cap
- **trace padding** — `cost.atomic_steps` and `cost.tokens` enter efficiency penalty linearly

### 2.4 Stage 3 — GRPO on verification / abstain policy

Once planning is decent, GRPO-train the verification / abstain behavior — either on the same LoRA or on a dedicated `8B_Controller_VerificationAbstain` LoRA.

#### Why separate the stages

Planning and verification are related but not identical:

- Planning learns **what to look for**.
- Verification learns **when the evidence is enough, contradictory, or insufficient.**

Keeping them as separate GRPO stages (and optionally separate LoRAs) usually makes RL more stable because the gradients optimize different output positions in the trace.

#### Reward terms (verification / abstain)

| Term | Source | Sign | Purpose |
|---|---|---|---|
| `r_verif_support` | `claim_evidence_alignment` check on each step | + | Reward correct support assessment |
| `r_verif_counter` | `counterevidence` check; `EvidenceBundle.contradictions` consulted | + | Reward finding counterevidence when present |
| `r_abstain` | `AbstainDecision` vs gold answerability | + | Reward abstaining only when evidence is truly insufficient |
| `r_false_answer_penalty` | confident answer with `claim_evidence_alignment` failed | − (heavy) | Heavily penalize unsupported confident answers |
| `r_false_abstain_penalty` | abstention on questions with sufficient evidence | − | Penalize abstaining when evidence supports an answer |

This directly supports the controller outputs in [Actors §2.7](actors_reasoning_model.md#27-controller-outputs-during-reasoning): per-hop verification result, confidence, and abstain decision.

### 2.5 Stage 4 — Conservative composite promotion (not full RL bank evolution)

Only after Stages 1–3 are stable should composite promotion begin. This is a **non-RL** step: the synthesizer surfaces candidates, the bank versions them, and a human reviewer signs off in v1 ([Skill Synthetics §0.2](skill_synthetics_agents.md#02-phase-1-conservative-synthesis-policy)).

#### Do NOT do

- fully automatic free-growing bank
- full patch / split / retire RL loop
- evolving memory policy

#### Do instead

Use simple, auditable promotion rules consistent with [Skill Extraction / Bank — Composite Skill Formation Rules](skill_extraction_bank.md#composite-skill-formation-rules):

- chain appears frequently (`N_repeat ≥ 10` in v1)
- chain verifies reliably (`τ_success ≥ 0.8`, `mean ≥ τ_stable`, `variance < σ_stable`)
- chain transfers across multiple examples (`≥ 3` distinct task families in v1)
- chain reduces hop count or improves evidence quality

### 2.6 Stage 5 — Optional GRPO for reflection / synthesis

Only do this after the [preconditions for self-evolution](skill_synthetics_agents.md#03-preconditions-for-later-self-evolution) are met:

- runtime schemas are stable
- hop traces are clean
- retrieval / verifier behavior is reliable
- failure localization is trustworthy

If you reach this stage, train an `8B_Controller_ReflectionSynthesis` LoRA to:

- classify failure type ([Skill Synthetics §5](skill_synthetics_agents.md#5-failure-taxonomy))
- localize the failing atomic step ([Harness — Failure Localization Protocol](atomic_skills_hop_refactor_execution_plan.md#failure-localization-protocol))
- suggest patch / split / retire / promote actions ([Skill Synthetics §4.3](skill_synthetics_agents.md#43-update-rules))
- score whether a trace is synthesis-worthy

This is **Phase 3 work**, not MVP.

---

## 3. Recommended training schedule (MVP)

| Stage | Training step | Trainable target | Frozen | Gating prerequisite |
|---|---|---|---|---|
| 0 | (no training) Build substrate | — | 72B/32B, harness, memory, retriever, verifier, bank | [MVP Build Order — Phase 1 items 1–6](mvp_build_order.md#phase-1-stable-substrate) |
| 1 | **SFT-1** | `8B_Controller_Main` LoRA on hop planning, skill routing, retrieval decisions, evidence-chain generation, answer/abstain format | All else | Stage 0 complete; canonical traces flow end-to-end |
| 2 | **GRPO-1** | Same LoRA (or `8B_Controller_PlanningRouting` if split) — planning/routing rewards | All else | SFT-1 outputs are well-formed and pass verifier on a held-out fraction |
| 3 | **SFT-2 / GRPO-2** | `8B_Controller_VerificationAbstain` LoRA *if* needed | All else | GRPO-1 plateaus on planning but verification regresses or under-fits |
| 4 | **Composite promotion** (non-RL) | Bank entries via synthesizer + human sign-off | All else | GRPO-1 (and GRPO-2 if used) stable; trace quality high |
| 5 | **GRPO-3 (later only)** | `8B_Controller_ReflectionSynthesis` LoRA | All else | [Synthesis preconditions](skill_synthetics_agents.md#03-preconditions-for-later-self-evolution) met |

This is the highest-probability route given that 72B/32B remain frozen inference tools and memory stays fixed.

---

## 4. Practical recommendation: how many trainable parts?

| Tier | LoRAs trained | When this is the right choice |
|---|---|---|
| **Best first version** | **1 LoRA** — `8B_Controller_Main` | Always start here. Simplest, easiest to debug, smallest data demand. |
| **Better but still manageable** | **2 LoRAs** — `8B_Controller_PlanningRouting` + `8B_Controller_VerificationAbstain` | Use only if the single LoRA shows entangled regressions or under-fits one of the two behaviors. |
| **Later only** | **3 LoRAs** — add `8B_Controller_ReflectionSynthesis` | Only after Stages 1–3 are stable and synthesis preconditions are met. |

I would strongly avoid training more than this in the first serious pass. Each additional LoRA roughly multiplies the data, ablation, and instability cost.

---

## 5. Final summary

If SFT comes first, then GRPO, and 72B/32B are inference-only, then the right plan is:

- **Train the 8B controller as the main adaptive module.**
- Optionally split it into **planning / routing** and **verification / abstain** LoRAs.
- **Keep harness fixed.**
- **Keep memory procedures fixed.**
- **Keep 72B / 32B frozen.**
- **Delay reflection / synthesis GRPO** until the runtime substrate is stable.

This is the training strategy most consistent with the current repo architecture: the 8B is explicitly the trainable controller, the 72B models are frozen tools, memory is the stable substrate, and the bank evolves later under conservative gates.

---

## 6. Next deliverable (suggested)

Turn this into a **copy-paste Cursor plan** for the SFT → GRPO training design:

- per-stage data manifests (which traces go into SFT-1, GRPO-1, …)
- per-stage reward configuration files (mapping `r_*` terms to weights and gates)
- per-stage eval harness slices ([Evaluation & Ablation Plan](evaluation_ablation_plan.md)) so each stage's success criterion is automatically checkable
- a `training_schedule.yaml` listing LoRAs, gating prerequisites, and abort criteria

This gives the implementation team a runnable training calendar, not just a strategy.
