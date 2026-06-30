# Atomic Skills v1: 24-Skill Target With MVP-12 Subset

Last updated: 2026-06-30

## Honest Recommendation

The version most likely to work first is **MVP-12**, not the full 24.

The 24-skill version is the right **target vocabulary** for the paper: it is expressive enough to cover Video-Holmes-style social contradiction, CG-Bench-style clue grounding, and M3-Bench-style memory operations. But training or evaluating all 24 at once will likely make the first experiment noisy.

Use this split:

- **MVP-12**: the smallest runnable set for trace-to-skill fitting, evidence chains, verification, and one local repair path.
- **Target-24**: the complete v1 skill vocabulary to describe the method and expand experiments after the MVP works.

The core principle is: every skill must be typed, executable, verifiable, reusable, and locally repairable.

## Refinement Boundary From Prior Repo History

The refinement loop must **not** depend on a 35B model to invent, patch, merge, or retire skills.

The commit history in `Multi-hop-Reasoning-VLM-Agent` is a warning: the old 35B Crafter path caused skill-bank pollution. The documented failure was not just "the prompt was bad"; the failure mode was structural:

- 35B-generated/patched skills collapsed into degenerate one-step patterns.
- Proposed predicates were often unverifiable at runtime.
- Skill mutation happened from abstract failure descriptions rather than grounded execution traces.
- The bank update path trusted LLM-generated structure too much.

So for this project:

- 35B/frontier teachers may generate **training traces**.
- 35B/frontier teachers may help **fit a trace to the fixed ontology**.
- 35B may write **offline natural-language analysis**: clearer descriptions, failure lessons, exemplar ranking.
- 35B must **not** modify structural fields: skill ids, input/output schemas, verifiers, preconditions, effects, failure codes, or dependency types.
- Merge/split/retire decisions must be driven by deterministic statistics and verifier outcomes, not model-authored refinement.

The novelty should therefore be framed as:

```text
fixed candidate ontology
  -> trace-to-skill fitting
  -> execution/verifier statistics
  -> deterministic confusion analysis
  -> gated automatic ontology revision
  -> smaller controller-visible basis
```

not:

```text
35B observes failures
  -> 35B invents better skills
  -> agent trusts new skills
```

## Automated Refinement Protocol

The system should refine automatically, but the automation is **statistical and gated**, not free-form LLM synthesis.

### State tracked per skill

For each skill `s`, maintain:

```text
selection_count(s)
execution_success_rate(s)
verifier_pass_rate(s)
evidence_support_rate(s)
repair_rate_after_failure(s)
mean_tool_cost(s)
mean_downstream_answer_gain(s)
confusion_matrix(s_i -> s_j)
argument_error_rate(s)
missing_evidence_role_rate(s)
```

### Allowed automatic operations

Only these structural updates are allowed:

| Operation | Automatic trigger | Structural source | Safety gate |
|---|---|---|---|
| `activate_candidate_skill` | MVP skill repeatedly fails on a typed bottleneck that a dormant Target-24 skill covers | existing Target-24 ontology only | candidate passes replay on held-out traces |
| `deactivate_skill` | low use, low verifier pass, or negative downstream gain across enough rollouts | existing active set | no required role uniquely depends on it |
| `merge_alias_skills` | two skills have high mutual confusion and indistinguishable verifier outcomes | predeclared alias group only | merged schema is already defined |
| `split_by_router` | one skill has two separable failure clusters with different argument schemas | predeclared child skills only | both children improve replay pass rate |
| `tighten_trigger` | skill is often selected but rejected by verifier for the same reason | trigger threshold/config only | no schema change |
| `update_description` | failures show recurring natural-language misunderstanding | prompt text only | meaning-preservation check |

No operation may create a new skill id, new schema field, new verifier, or new failure code from scratch.

### Refinement loop

```text
for iteration k:
  1. run controller on train rollouts
  2. log every skill action, arguments, evidence refs, verifier result, answer result
  3. compute per-skill and pairwise confusion statistics
  4. propose deterministic bank updates from allowed operations
  5. replay proposed active set on frozen validation traces
  6. accept update only if:
       verifier pass does not drop
       evidence support improves or stays flat
       answer accuracy improves or stays flat
       tool cost does not exceed budget
       active skill count remains within cap
  7. version the active set and keep rollback pointer
```

### Controller-visible cap

Even if Target-24 exists, the controller should see at most `K=8..12` actions at a time. The router can expose a different subset by task family:

```text
Video-Holmes: social contradiction / intention subset
CG-Bench: clue localization / evidence support subset
M3-Bench: memory retrieval / temporal-chain subset
VRBench: long-video temporal-neighborhood subset
```

This gives automation without exploding the action space.

### Role of 35B in automated refinement

35B can annotate the logs, but it cannot decide or enact structural updates.

Allowed:

```text
summarize recurring failures
rank exemplars
rewrite descriptions
suggest which predeclared failure bucket a trace belongs to
```

Not allowed:

```text
create new skill
patch schema
invent verifier
merge/split skills directly
override replay gate
```

## Families

| Family | Purpose |
|---|---|
| `question_programming` | Turn a question into evidence roles and a skill plan. |
| `memory_retrieval` | Query evidence-bearing memory nodes over entities, events, and time. |
| `video_grounding` | Convert raw video/audio/clips into typed evidence. |
| `evidence_organization` | Turn retrieved evidence into role-labeled reasoning chains. |
| `reasoning` | Perform temporal, causal, social, and state reasoning over evidence. |
| `verification_repair` | Check claims and locally repair failed evidence acquisition. |

## MVP-12

These are the 12 I would implement first for a working prototype:

1. `parse_question_target`
2. `propose_evidence_roles`
3. `retrieve_event`
4. `retrieve_temporal_neighborhood`
5. `resolve_entity_reference`
6. `localize_clue`
7. `extract_dialogue_claim`
8. `mark_evidence_role`
9. `compose_evidence_chain`
10. `order_events`
11. `verify_evidence_supports_claim`
12. `repair_by_requery`

This subset is enough to support the central loop:

```text
question
  -> evidence roles
  -> memory/video retrieval
  -> clue/dialogue grounding
  -> role labeling
  -> evidence-chain composition
  -> temporal reasoning
  -> verify
  -> local repair
```

## Target-24 Skill Table

| # | Skill | Family | MVP-12 | Inputs | Outputs | Verifier | Failure codes |
|---:|---|---|---|---|---|---|---|
| 1 | `parse_question_target` | `question_programming` | yes | `question_text` | `target_entities`, `target_events`, `question_focus`, `constraints` | entity and constraint parse is non-empty when required | `missing_target`, `ambiguous_target` |
| 2 | `propose_evidence_roles` | `question_programming` | yes | `question_text`, `parsed_target` | `evidence_roles`, `role_constraints` | roles are typed and relevant to the parsed target | `missing_role`, `overbroad_roles` |
| 3 | `retrieve_event` | `memory_retrieval` | yes | `event_description`, `time_range`, `entity_filter` | `event_nodes`, `evidence_refs` | returned nodes have timestamps and evidence pointers | `empty_evidence`, `timestamp_missing` |
| 4 | `retrieve_entity_history` | `memory_retrieval` | no | `entity_id`, `time_range`, `predicate_filter` | `entity_timeline`, `evidence_refs` | timeline entries refer to same resolved entity | `entity_unresolved`, `empty_history` |
| 5 | `retrieve_temporal_neighborhood` | `memory_retrieval` | yes | `anchor_event`, `window_before`, `window_after` | `neighbor_events`, `evidence_refs` | neighbors overlap requested temporal window | `anchor_missing`, `empty_neighborhood` |
| 6 | `query_temporal_chain` | `memory_retrieval` | no | `start_event`, `end_event`, `constraints` | `ordered_event_chain`, `gaps` | chain order is timestamp-consistent | `start_missing`, `end_missing`, `chain_gap` |
| 7 | `resolve_entity_reference` | `memory_retrieval` | yes | `mention`, `context`, `candidate_entities` | `entity_id`, `alias_edges`, `confidence` | resolved id exists and evidence supports alias | `unresolved_entity`, `ambiguous_entity` |
| 8 | `localize_event` | `video_grounding` | no | `video_id`, `event_description`, `search_range` | `time_spans`, `clip_refs` | clip evidence visually/textually supports event | `event_not_found`, `low_confidence` |
| 9 | `localize_clue` | `video_grounding` | yes | `video_id`, `clue_description`, `search_range` | `clue_spans`, `evidence_refs` | clue span supports one requested evidence role | `clue_not_found`, `role_mismatch` |
| 10 | `extract_visual_evidence` | `video_grounding` | no | `clip_ref`, `visual_query` | `visual_observations`, `evidence_refs` | observation is grounded to frames or clip span | `no_visual_support`, `tool_failure` |
| 11 | `extract_dialogue_claim` | `video_grounding` | yes | `subtitle_or_audio_ref`, `speaker_hint` | `speaker`, `claim_text`, `time_span`, `evidence_ref` | claim is anchored to speech/subtitle evidence | `speaker_unknown`, `claim_not_found` |
| 12 | `track_entity` | `video_grounding` | no | `entity_id_or_mention`, `time_range` | `track_spans`, `locations`, `evidence_refs` | track is temporally continuous enough for task | `track_lost`, `entity_not_visible` |
| 13 | `detect_contact_or_interaction` | `video_grounding` | no | `entity_a`, `entity_b_or_object`, `time_range` | `interaction_event`, `time_span`, `evidence_refs` | entities co-occur and interaction cue exists | `not_visible`, `no_interaction` |
| 14 | `mark_evidence_role` | `evidence_organization` | yes | `evidence_ref`, `role_schema`, `question_context` | `role_labeled_evidence` | role assignment matches evidence content | `role_unsupported`, `duplicate_role` |
| 15 | `compose_evidence_chain` | `evidence_organization` | yes | `role_labeled_evidence`, `dependency_template` | `evidence_chain`, `missing_roles` | chain satisfies required role/dependency coverage | `missing_role`, `invalid_dependency` |
| 16 | `find_missing_evidence_role` | `evidence_organization` | no | `evidence_chain`, `required_roles` | `missing_roles`, `suggested_queries` | missing roles are not already filled | `no_missing_role`, `bad_query_hint` |
| 17 | `locate_counterevidence` | `evidence_organization` | no | `claim`, `supporting_evidence`, `search_scope` | `counterevidence_refs`, `counter_claims` | counterevidence contradicts or weakens claim | `no_counterevidence`, `false_counterevidence` |
| 18 | `order_events` | `reasoning` | yes | `event_a`, `event_b` | `temporal_relation`, `supporting_evidence` | timestamps support relation | `missing_timestamp`, `overlap_uncertain` |
| 19 | `check_state_change` | `reasoning` | no | `entity_or_object`, `state_predicate`, `before_after_refs` | `changed`, `before_state`, `after_state` | before/after states are grounded and ordered | `missing_before`, `missing_after`, `no_change` |
| 20 | `infer_causal_support` | `reasoning` | no | `candidate_cause`, `candidate_effect`, `evidence_chain` | `causal_support`, `causal_rationale` | cause precedes effect and evidence links them | `temporal_violation`, `weak_causal_link` |
| 21 | `infer_intention_or_motive` | `reasoning` | no | `agent`, `actions`, `context_evidence` | `intention_claim`, `supporting_roles` | intention is supported by action/context evidence | `insufficient_social_cue`, `alternative_motive` |
| 22 | `infer_social_contradiction` | `reasoning` | yes | `claim_or_alibi`, `evidence_chain`, `counterevidence` | `contradiction_claim`, `supporting_evidence` | claim and evidence cannot both hold | `missing_claim`, `missing_contradiction_link` |
| 23 | `verify_evidence_supports_claim` | `verification_repair` | yes | `claim`, `evidence_chain` | `verification_score`, `passed`, `failure_code` | evidence entails or strongly supports claim | `unsupported_claim`, `insufficient_evidence`, `contradicted` |
| 24 | `repair_by_requery` | `verification_repair` | yes | `failed_role_or_step`, `failure_code`, `query_hints` | `new_query`, `replacement_evidence` | replacement evidence fixes the failed role/step | `repair_failed`, `no_new_evidence` |

## Active Target-24

```text
parse_question_target
propose_evidence_roles
retrieve_event
retrieve_entity_history
retrieve_temporal_neighborhood
query_temporal_chain
resolve_entity_reference
localize_event
localize_clue
extract_visual_evidence
extract_dialogue_claim
track_entity
detect_contact_or_interaction
mark_evidence_role
compose_evidence_chain
find_missing_evidence_role
locate_counterevidence
order_events
check_state_change
infer_causal_support
infer_intention_or_motive
infer_social_contradiction
verify_evidence_supports_claim
repair_by_requery
```

## Implementation Staging

### Stage A: MVP-12

Goal: produce valid skill traces for a small Video-Holmes subset and prove the method is not just free-text CoT.

Required datasets:

- Video-Holmes for social contradiction / intention / causal clues.
- CG-Bench mini for clue localization and evidence support.

Required metrics:

- final answer accuracy
- evidence role coverage
- evidence-chain support
- verifier pass rate
- repair success after missing evidence

### Stage B: Active Target-24

Goal: expand the same action space to memory-heavy and long-video tasks.

Add:

- CG-Bench full clue retrieval
- M3-Bench memory graph tasks
- selected VRBench long-video cases

Required new metrics:

- temporal-neighborhood recall
- entity-history retrieval quality
- cross-hop evidence reuse
- tool/retrieval cost
- local repair improvement over no-repair baseline

## Baselines To Keep Honest

Compare against:

1. direct VLM answer
2. caption RAG + CoT
3. graph retrieval + CoT
4. teacher full planning
5. 8B SFT-only skill actions
6. 8B SFT + verified RL
7. 8B SFT + verified RL + repair

The method only becomes convincing if the 8B controller improves process metrics, repair, cost, or transfer, not merely if it imitates teacher traces.
