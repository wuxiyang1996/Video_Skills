# Atomic skills, hop composition & layer split — execution plan

> **Purpose:** Cursor-ready checklist and file-by-file edit plan (English) for refactoring `infra_plans` so reasoning skills stay **reusable inference operators**, coarse skills split into **atomic** + **composite** forms, **one hop** is a short verifiable chain of atomics, and **infrastructure** (observation, tracking, memory build, retrieval) is not stored as bank skills.
>
> **Related plans (apply together):**
> - [Skill Extraction & Skill Bank](skill_extraction_bank.md) — canonical bank design
> - [Skill Synthetics Agents](skill_synthetics_agents.md) — evolution & reflection
> - [Actors / Reasoning Model](actors_reasoning_model.md) — controller & hop execution
> - [Agentic Memory](agentic_memory_design.md) — three stores + evidence

---

## Overall objective

Refactor the plan so the system has three clean layers:

1. **Infrastructure / observer pipeline** — segmentation, entity tracking, episodic / semantic / state memory construction, grounding outputs.
2. **Reasoning skill bank** — atomic skills, composite skills, hop composition, verification skills.
3. **Reflection / evolution loop** — failure diagnosis, atomic-step localization, patch / split / promote / retire.

This aligns with the compact three-store memory design and the 8B controller as orchestrator over memory and perspective threads.

---

## Conceptual center (use in bank + actors)

One reasoning hop may involve a few atomic skills rather than a single monolithic reasoning skill. Atomic skills are the minimal reusable reasoning operators, while composite skills are stable, reusable short chains of atomic skills that can still be expanded for diagnosis and revision.

We treat one reasoning hop as a short, verifiable composition of several atomic skills rather than a monolithic reasoning step. Infrastructure operations such as observation, tracking, storage, and retrieval are not first-class reasoning skills; they are pipeline primitives that reasoning skills may call.

---

## Infrastructure primitives (not bank skills)

These remain callable system primitives but are **not** stored as reasoning skills in the bank:

- `observe_segment`
- `detect_entities`
- `build_episodic`
- `build_semantic`
- `update_state`
- `search_memory`

They belong to observation, memory, and retrieval infrastructure.

---

## File-by-file goals

### 1) `skill_extraction_bank.md` (main rewrite)

- Separate infrastructure primitives from reasoning skills.
- Introduce atomic skills, composite skills, and task-level policies.
- Define one reasoning hop as a short composition of atomic skills.
- Add atomic skill families (parsing, retrieval/grounding, temporal, causal, social, verification, reflection).
- Replace coarse composition examples with atomic-chain / hop examples.
- Use layered skill schema: `level`, `output_type`, `child_skills`, `verification_rule`, `failure_modes`, `repair_strategies`, etc.

### 2) `skill_synthetics_agents.md`

- Evolution unit: **atomic chain / hop / composite skill** (not only coarse skills).
- **Reflection unit:** atomic-step failure localization (observation vs reasoning vs verification; localize failed atomic skill(s) within a hop).
- Failure categories (observation-side, reasoning-side, verification-side).
- Update rules: patch, split, promote, merge, retire.
- Promotion criteria for atomic chain → composite (tie back to bank design).

### 3) `actors_reasoning_model.md`

- Controller operates over: atomic skills, composite skills, hop traces, reflection outputs.
- Skill execution granularity: hops = short atomic compositions; composites expandable to atomic traces.
- Reasoning loop with verification, abstain, and reflection trace emission.
- Explicit controller outputs during reasoning.

### 4) `agentic_memory_design.md`

- Short clarification: reasoning skills operate **over** memory outputs; they do not replace episodic / semantic / state functions.

---

## Constraints

- Keep the system **compact**; do not add extra top-level memory types.
- Do not treat generic tool calls as first-class reasoning skills.
- Do not make dataset-specific templates the main bank content.
- Emphasize reusable, verifiable, evolvable reasoning operators.

---

## Minimal atomic skill set (implementation v1)

**First pass (12):**

- `identify_question_target`
- `decompose_into_subgoals`
- `retrieve_relevant_episode`
- `ground_entity_reference`
- `ground_event_span`
- `infer_observation_access`
- `order_two_events`
- `check_state_change`
- `check_causal_support`
- `update_belief_state`
- `check_evidence_sufficiency`
- `decide_answer_or_abstain`

**Second pass add:** `check_alternative_hypothesis`, `locate_counterevidence`, `classify_failure_type`, `localize_failed_step`.

---

## Suggested Cursor task list

1. **skill_extraction_bank.md:** infrastructure vs skills; atomic/composite/hops; families; hop examples; layered schema.
2. **skill_synthetics_agents.md:** localization; failure categories; patch/split/promote/merge/retire; promotion criteria.
3. **actors_reasoning_model.md:** hop planning; composite expansion; reasoning loop; traces and outputs.
4. **agentic_memory_design.md:** skills vs memory substrate paragraph.

---

*This document is the stable reference for the refactor; the normative design lives in the linked `infra_plans` files above.*
