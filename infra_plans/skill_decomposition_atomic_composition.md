# Skill Decomposition: Atomic Operators and Composed Skills

Yes — and your current repo structure is already close to the right abstraction. The key is to stop treating a skill as one monolithic reasoning routine and instead define it as a typed composition of atomic operators over your three memory roles: episodic, semantic, and state. Your current plans already define skills as reusable inference operators, with universal skills like `temporal_reason`, `causal_reason`, and `social_reason`, plus memory-building and retrieval operators like `observe_segment`, `build_episodic`, `build_semantic`, `update_state`, and `search_memory`. They also define a synthesis loop that segments trajectories, extracts contracts from state changes, and generates reusable protocols.

A clean way to decompose and synthesize skills is this:

## 1. Make atomic skills the only executable units

Instead of storing a skill like “infer belief update” as one opaque tool, break it into four small operator families:

### Perception / grounding atoms

- locate entity
- ground event span
- resolve speaker / actor
- check co-visibility / witness
- extract region / frame evidence

### Memory atoms

- fetch episodic evidence
- fetch semantic summary
- read current social state
- read current spatial state
- write/update state hypothesis

### Reasoning atoms

- compare before vs after
- infer temporal order
- infer causal link
- test consistency
- project perspective
- aggregate weak cues
- abstain under uncertainty

### Control atoms

- plan next hop
- decide retrieval target
- verify support
- stop / answer / defer

This fits your repo well because your memory design is already function-based: episodic answers “what happened,” semantic answers “what stays true,” and state answers “what is true now,” with visual data attached as evidence rather than a separate top-level memory.

## 2. Define a skill as a graph over atomic skills

A reusable skill should be represented as:

- **intent:** what reasoning job it solves
- **input schema:** question type, available entities, memory mode
- **atomic DAG / program:** ordered or branched atomic steps
- **contract:** what new intermediate variables or state it should produce
- **evidence requirements:** minimum support needed
- **failure modes:** how it commonly breaks

So instead of:

`infer_belief_update`

you store:

`infer_belief_update` =

1. identify target character  
2. ground pivot event  
3. retrieve pre-event perspective state  
4. test whether pivot event was observed  
5. infer post-event belief  
6. verify against later actions/dialogue  
7. emit belief delta + evidence  

That is much more transferable than a single prompt-written skill. It also matches your existing social skills, which already look like structured execution patterns over memory and perspective threads.

## 3. Separate atomic reuse from macro-skill reuse

Use two levels:

### Level A: atomic bank

A small stable library of 20–40 atomic operators shared across all tasks.

### Level B: composed skill bank

Task-facing skills built from those atoms, such as:

- `who_saw_what`
- `infer_belief_update`
- `track_commitment`
- `detect_intention_shift`
- `resolve_conflicting_testimony`

This is better than directly learning hundreds of full skills, because cross-task transfer mostly happens through the atoms, not the surface name of the macro-skill.

**Example:** Three different tasks may reuse the same atoms:

| Task | Example |
|------|---------|
| browser | “find who clicked submit after reading the dialog” |
| long video social | “who learned the secret after the hallway exchange” |
| image multi-hop | “which person could have seen the object move” |

All three may reuse:

- `ground_event`
- `resolve_entity`
- `check_visibility`
- `project_perspective`
- `compare_before_after`

## 4. Synthesize macro-skills from repeated atomic traces

Your current synthesis plan already has:

- temporal segmentation
- contract extraction from predicate deltas
- protocol generation from similar segments
- cross-video merge/refine/retire

**Change the synthesis target slightly:**

Instead of generating only a natural-language protocol, generate:

```text
Skill {
  skill_id
  name
  intent_tag
  input_schema
  atomic_trace_template
  branching_conditions
  contract
  evidence_requirements
  failure_signatures
  support_segments
  n_instances
}
```

Where `atomic_trace_template` looks like:

```text
[
  resolve_target_entity,
  retrieve_relevant_episode,
  ground_pivot_event,
  read_pre_state,
  check_observation_link,
  infer_state_transition,
  verify_with_future_evidence
]
```

Then protocol text is derived from the trace, not the reverse. That makes synthesis much more robust.

## 5. Use failure-driven decomposition

Your repo already has a good failure taxonomy:

- missed evidence
- wrong temporal linkage
- wrong entity grounding
- perspective confusion
- false-belief reasoning error
- overconfident inference

This can directly tell you which atomic skill is missing or weak.

| Failure | Likely atomic failure |
|---------|------------------------|
| missed evidence | retrieval atom failed |
| wrong temporal linkage | temporal-order atom failed |
| wrong entity grounding | entity-resolution atom failed |
| perspective confusion | perspective-projection atom failed |
| overconfident inference | verification / abstention atom failed |

So a failed macro-skill should be decomposed into atomic blame, then repaired at the atomic level first.

**Main trick for self-evolution:**

macro failure → atomic diagnosis → atomic repair → macro resynthesis

## 6. Use a typed interface for every atomic skill

Each atomic skill should expose:

- name
- input_slots
- output_slots
- preconditions
- postconditions
- evidence_access
- cost
- confidence_rule

**Example:**

**AtomicSkill:** `check_visibility`

- **inputs:** `[observer_id, event_id, spatial_state, episodic_refs]`
- **outputs:** `[visible:boolean, confidence:float, evidence_refs:list]`
- **preconditions:** `observer_resolved && event_grounded`
- **postconditions:** `observation_status_available`

This is what lets you synthesize new skills compositionally rather than by prompt-only writing.

## 7. Keep atoms reasoning-centric, not benchmark-centric

Do **not** define atoms like:

- `answer_video_holmes_question`
- `solve_m3bench_social_query`

Too high-level and non-transferable.

**Prefer:**

- `retrieve_supporting_episode`
- `align_entities_across_time`
- `identify_belief_holder`
- `check_observer_access`
- `compare_claims`
- `infer_hidden_action`
- `validate_with_evidence`

That will transfer across short videos, long videos, social reasoning, image reasoning, even browser/game settings if their state is structured similarly.

## 8. A practical decomposition template for your repo

### Atomic skill categories

**A. Grounding atoms**

- `resolve_entity`
- `ground_event`
- `align_coreference`
- `extract_observer_set`
- `ground_scene_change`

**B. Retrieval atoms**

- `search_episode`
- `search_semantic_summary`
- `read_social_state`
- `read_spatial_state`
- `collect_support_evidence`

**C. Relational reasoning atoms**

- `compare_timestamps`
- `compare_states`
- `infer_cause`
- `infer_goal`
- `infer_belief`
- `infer_intention_shift`
- `check_claim_consistency`

**D. Verification atoms**

- `cross_check_modalities`
- `measure_evidence_strength`
- `detect_ambiguity`
- `abstain_or_continue`

**E. Composition/control atoms**

- `plan_reasoning_hop`
- `select_next_atom`
- `terminate_with_answer`

Then macro-skills become compositions.

### Examples

**`who_saw_what`**

1. `resolve_entity`
2. `ground_event`
3. `extract_observer_set`
4. `read_spatial_state`
5. `verify_visibility`
6. return witness partition

**`resolve_conflicting_testimony`**

1. `align_coreference`
2. `retrieve_claim_events`
3. `project_perspective` for each speaker
4. `check_claim_consistency`
5. score evidence strength
6. produce reconciliation

## 9. Synthesis rule: build new skills only when composition recurs

Do not create a new macro-skill from one successful trace.

Create it only if:

- same atomic trace pattern appears across multiple videos/tasks
- contract is stable
- evidence dependencies are similar
- success is repeatable

This aligns with your existing cross-video accumulation and merge/refine/retire loop.

**Rules of thumb:**

- **new atomic skill:** only when repeated failures cannot be explained by existing atoms
- **new macro-skill:** when the same atom composition recurs at least N times with stable outputs

## 10. Best design principle for you

Given your cross-task goal, the strongest formulation is:

- **Atomic skills** = benchmark-agnostic reasoning operators  
- **Composed skills** = reusable reasoning programs over those operators  

That is much better than saying all skills are MCP-style tools or all skills are plain prompts.

---

## Recommended repo updates (follow-up)

In `skill_extraction_bank.md`, add one new section:

**“Skill Decomposition Hierarchy”**

- atomic skills
- composed skills
- synthesis from traces
- failure-driven atomic refinement
- merge/split criteria at both atomic and macro levels

In `skill_synthetics_agents.md`, revise stage 3 so it outputs both:

- natural-language protocol
- structured atomic trace template

That makes the whole system far more learnable, debuggable, and transferable.

---

## Compact formulation (for cross-referencing in other plan files)

We represent each reusable skill as a composition of atomic reasoning operators over episodic, semantic, and state memory. Atomic skills are the smallest executable evidence-grounded units, such as entity resolution, event grounding, temporal comparison, perspective projection, causal inference, and evidence verification. Higher-level skills are synthesized as reusable programs over these atoms, with explicit contracts, evidence requirements, and failure signatures. This enables cross-task transfer, because reuse happens primarily at the atomic level, while macro-skills remain interpretable compositions specialized to recurring reasoning patterns.
