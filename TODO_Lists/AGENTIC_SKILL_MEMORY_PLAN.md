# Agentic Skill-Memory Layer — Plan (English)

**Created:** 2026-03-13  
**Status:** Planning (no code changes yet)

---

## Overall Goal

Move the current skill bank agent from:

> **"A system that discovers, maintains, and retrieves symbolic skill contracts"**

to:

> **"An agentic skill-memory layer that can select skills under the current state and provide stronger execution guidance to the decision agent."**

So the goal is **not** to add another new stage. The goal is to straighten this chain:

**tag-aware Stage 1 → stronger Stage 2 labels → faster new-skill growth → state-aware query and execution guidance**

---

## Phase 1: Fix the Query Path First

The first change is the query layer — smallest code change with the largest immediate benefit.

### Current State

- **`select()`** in `query.py` is the richer, state-aware selection interface.
- **`query_skill()`** still behaves like a backward-compatible retrieval wrapper.

### What to Do

Change **`query_skill()`** so that:

- **If** `current_state` or `current_predicates` is available → **default to `select_skill()` / `select()`**.
- **Only if** no state is available → fall back to the old `query()` path.

Then the skill bank will actually use:

- applicability  
- pass rate  
- matched/missing effects  
- state-conditioned ranking  

instead of only semantic relevance.

### Why This Matters

Right now the bank helps more with *"which skill looks relevant?"* and less with *"which skill is appropriate right now?"*. This change makes the skill bank useful at **decision time**, not just retrieval time.

### Additional Change in This Phase

Upgrade the returned result from a thin retrieval output into a **structured guidance package**.

Instead of mostly returning a skill name and a shallow `micro_plan`, each selected skill should return:

| Field | Purpose |
|-------|--------|
| `skill_name` | Identifier |
| `why_selected` | Short rationale |
| `applicability_score` | How well it fits current state |
| `expected_effects` | What the skill typically achieves |
| `preconditions` | When it is valid |
| `termination_hint` | How to know it is done |
| `failure_modes` | What to watch for |
| `execution_hint` | How to carry it out |

This does not need to be a full option policy yet — it just needs to be much more actionable for the decision agent.

---

## Phase 2: Use Intention Tags in Stage 1 Boundary Proposal

Intention tags are a core part of the plan. The current architecture fits well:

- Stage 1 is a **high-recall boundary proposal** module.
- It should **propose** likely boundaries, not make final segmentation decisions.

So intention tags are a natural fit.

### What to Do

Explicitly add **two tag-based signals** into Stage 1:

1. **Tag change → boundary candidate**  
   If the decision-making intention tag changes from step `t-1` to step `t`, add a boundary candidate at `t`.  
   Example: `[MOVE_TO_ONION]` → `[PICK_ONION]` is often a good skill boundary proposal.

2. **Tag completion / done → stronger boundary event**  
   If the current intention is marked completed, or the step has a completion signal (e.g. `done=True`), treat that point as a **stronger** event-like boundary.

### Design Choice

**Do not** make tag the only boundary source. Instead:

- **Tag provides a strong proposal prior.**
- **Stage 2 still makes the final decision.**

Principle: **tag proposes, Stage 2 decides.**

### Stability Protections to Add

- `min_segment_len`
- Tag canonicalization / normalization
- Repeated-tag filtering
- Keep original event / state-change / changepoint signals as backup

This keeps the tag signal useful without making the pipeline brittle.

---

## Phase 3: Strengthen Stage 2 (Boundary Quality and Uncertain Labels)

This is the most important phase for training stability. Almost everything downstream depends on Stage 2 labels being reasonably correct:

- contract learning  
- split/merge  
- query-time retrieval  
- new skill growth  

If Stage 2 mixes two different skills into one label, Stage 3 and Stage 4 are mostly doing damage control.

### What to Do

#### 1. Add boundary preference learning

Right now Stage 2 is mostly about scoring candidate segments and labels. Add an explicit signal:

- **"cut here"** vs **"do not cut here"**

So the model learns **boundary preference directly**, not only indirectly through segment-label scoring. In long trajectories, boundary errors are often more damaging than label-name errors.

#### 2. Add an uncertain-label path

Do **not** force every segment into a known skill too early. Instead, three outcomes:

| Outcome | Handling |
|--------|----------|
| **Confident known skill** | Normal known-skill path |
| **Low-confidence known skill** | Uncertain pool; reconsider after later bank updates or proto-skill formation |
| **NEW / UNKNOWN** | Treated as such |

This is more stable than over-committing too early.

#### 3. Add stronger diagnostics

Stage 2 should record more useful confidence statistics:

- top1–top2 margin  
- label entropy  
- compatibility margin  
- boundary confidence  

Useful for: debugging, filtering noisy data, ablations, curriculum design.

---

## Phase 4: Proto-Skill Layer (Do Not Let Stage 3 Learn Raw NEW)

Do **not** change Stage 3 so it learns contracts directly from raw NEW segments. The current design — where `run_contract_learning()` skips NEW/`__NEW__` — is reasonable; raw new segments are often too noisy.

### What to Do Instead

Insert a **proto-skill layer** between NEW and fully materialized skills.

**Current:**  
`NEW → cluster → materialize → real skill`

**Target:**  
`NEW → cluster → proto-skill → light verification → real skill`

### What a Proto-Skill Contains

Lightweight structure such as:

- `proto_id`
- member segments
- candidate effects
- support
- consistency
- separability
- tag distribution
- typical segment length
- context before/after the segment

### Why This Helps

- New skills get a meaningful intermediate representation before full promotion.
- Stage 2 can already consider proto-skills as candidate labels.
- Growth is faster and smoother and depends less on one-shot clustering quality.

This is better than either ignoring NEW for too long or contract-learning NEW too early.

---

## Phase 5: Add Distilled Execution Hints to Each Skill

Address the concern that the system is still more of a symbolic contract bank than a skill policy bank. Do **not** jump to full per-skill policies (expensive and unstable).

### What to Do Instead

For each skill, in addition to symbolic contract fields, store a **lightweight distilled execution hint** derived from successful segments:

- common preconditions  
- common target objects  
- state-transition pattern  
- termination cues  
- common failure modes  
- short natural-language execution description  

### Why This Matters

Then the skill bank is no longer only saying *"this skill usually causes these effects"*. It can also say:

- *"this is how this skill is typically carried out"*  
- *"this is what to watch for"*  
- *"this is how to tell if it succeeded or failed"*  

That is the bridge from retrieval to practical decision guidance.

---

## Execution Order

Implement in this order:

| Order | Phase | Rationale |
|-------|--------|-----------|
| **First** | Phase 1: Make `query_skill()` default to state-aware selection | Immediate online benefit with relatively small code changes |
| **Second** | Phase 2: Add tag-based boundary proposal into Stage 1 | Cheap and effective way to improve boundary recall |
| **Third** | Phase 3: Strengthen Stage 2 with boundary preference and uncertain-label handling | Core stability improvement |
| **Fourth** | Phase 4: Add proto-skills for NEW | Improves open-world growth; new skills join the loop earlier |
| **Fifth** | Phase 5: Add distilled execution hints | Upgrades the bank from retrievable symbolic memory to state-aware skill guidance |

---

## What This Plan Achieves

After these changes, the skill bank agent would no longer be just:

> a symbolic contract discovery and retrieval module

It would become:

> a system that discovers skills from long trajectories, verifies and maintains them, allows new skills to grow gradually, and returns **state-aware execution guidance** at query time.

That gives a cleaner story:

- **Upstream:** segmentation and skill discovery  
- **Middle:** contract learning, proto-skill growth, bank maintenance  
- **Downstream:** state-aware skill selection and execution guidance  

---

## One-Line Summary

**First** make query truly state-aware, **then** use intention tags to strengthen Stage 1, **then** stabilize Stage 2, **then** speed up new-skill growth with proto-skills, and **finally** add execution hints so retrieved skills become more actionable.

---

## Next Step (Optional)

This plan can be turned into a concrete coding roadmap with:

- files to modify  
- data structures to add  
- expected output of each change  
