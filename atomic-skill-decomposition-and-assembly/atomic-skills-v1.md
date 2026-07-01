# Atomic Skills v1: Two Atomic Skill Sets

Last updated: 2026-06-30

## Honest Recommendation

The version most likely to work first is **expert-demo reasoning assembly over a
prebuilt evidence graph**, not end-to-end video perception plus reasoning.

The atomic skill vocabulary should be organized as two sets:

```text
Evidence Graph Construction Skills
  video / captions / annotations / tool outputs
    -> EvidenceMemoryGraph / clue-memory graph

Reasoning Graph Assembly Skills
  question + EvidenceMemoryGraph
    -> SkillGraphRollout / skill reasoning graph
    -> verified answer
```

The core principle is: every skill must be typed, executable, verifiable,
reusable, locally repairable, and small enough to compose into larger motifs.
Coverage matters more than forcing the total count to stay at 24.

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
| `activate_candidate_skill` | Active skill repeatedly fails on a typed bottleneck that a dormant candidate skill covers | existing atomic skill sets only | candidate passes replay on held-out traces |
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

Even if the full atomic skill sets are larger, the controller should see at most `K=8..12` actions at a time. The router can expose a different subset by task family:

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

## Atomicity and Coverage Criteria

Atomic skills should be the smallest reusable units that can compose into larger
skills, motifs, and expert rollouts. A candidate belongs in the atomic set only
if it satisfies all of these conditions:

- **Minimal.** It should not naturally split into two common operations with
  separate inputs, outputs, and failure modes.
- **Composable.** It should be useful inside multiple composed skills, not tied
  to one benchmark template such as "solve alibi contradiction".
- **Graph-grounded.** It either writes typed evidence/memory nodes and edges, or
  reads them through explicit `evidence_refs`.
- **Typed.** Its inputs, outputs, graph read/write scope, and failure codes are
  stable enough to validate.
- **Verifier-visible.** A local, tool-based, or model-assisted verifier can check
  whether the operation succeeded.
- **High coverage.** The full set should cover most composed reasoning motifs we
  expect in Video-Holmes, CG-Bench, VRBench, SIV-Bench, and later M3-Bench.

The count is therefore secondary. It is better to keep a slightly larger set of
clean atomic units than to merge unrelated operations only to hit a round number.
Later confusion statistics can merge skills that are empirically indistinguishable.

## Atomic Skill Bundles

The two sets share a typed operator contract, but they have different assembly
targets, verifier families, and controller visibility.

```text
Evidence Graph Construction Skills
  assembly target: EvidenceMemoryGraph
  expert_demo role: fixed/offline graph builder and audit trace
  video_only role: tool-mediated or controller-visible perception actions

Reasoning Graph Assembly Skills
  assembly target: SkillGraphRollout
  expert_demo role: primary controller-visible action set
  video_only role: same reasoning layer over discovered evidence
```

### Evidence Graph Construction Skills

These skills decompose perception, captions, annotations, and tool outputs into
minimal graph-construction operations.

| # | Skill | Purpose | Inputs | Outputs | Verifier focus |
|---:|---|---|---|---|---|
| 1 | `segment_video_or_select_clip` | Create clip/window nodes under a whole-video, fixed-window, hierarchical, or streaming policy. | `video_id`, `clip_policy`, `observation_end_s?` | `clip_nodes`, `time_spans` | windows are valid and respect visibility constraints |
| 2 | `extract_observation` | Extract observable facts from a clip, caption, ASR span, or annotation. | `clip_or_text_ref`, `modality`, `observation_query?` | `observation_nodes`, `evidence_refs` | observation is grounded to source span |
| 3 | `extract_dialogue_span` | Extract speaker, utterance, and timestamp from subtitle/ASR/dialogue annotation. | `subtitle_or_asr_ref`, `speaker_hint?` | `dialogue_span_node`, `speaker_mention`, `evidence_ref` | dialogue span has source and timestamp |
| 4 | `detect_entity_mention` | Detect person, object, place, and speaker mentions. | `observation_ref`, `entity_type?` | `mention_nodes`, `surface_forms`, `time_spans` | mention is supported by text/visual/audio evidence |
| 5 | `resolve_entity_coreference` | Link mentions across clips or modalities to the same entity. | `mention_nodes`, `context_edges?` | `entity_node`, `same_entity_edges`, `confidence` | linked mentions are compatible and not contradictory |
| 6 | `create_event_node` | Convert observations or dialogue spans into timestamped event nodes. | `observation_refs`, `event_description`, `time_span` | `event_node`, `event_type`, `evidence_refs` | event is grounded and not a duplicate |
| 7 | `create_state_node` | Represent an entity/object state such as location, possession, emotion, visibility, or relation. | `entity_ref`, `state_predicate`, `evidence_refs`, `time_span?` | `state_node`, `state_value`, `confidence` | state is grounded and temporally scoped |
| 8 | `link_graph_relation` | Add graph edges such as `temporal_next`, `entity_mention`, `derived_from`, `causal_hint`, or `same_entity`. | `source_node`, `target_node`, `edge_type`, `evidence_refs?` | `memory_edge`, `confidence` | edge type is allowed and endpoints exist |
| 9 | `assign_provenance_trust` | Attach source, trust level, visibility mode, and hidden-supervision status. | `node_or_edge_ref`, `source_ref`, `mode`, `trust_policy` | `provenance`, `trust_level`, `discovery_status` | provenance and visibility are consistent |

### Reasoning Graph Assembly Skills

These skills decompose expert reasoning traces into minimal operations that can
be assembled into composed reasoning motifs.

| # | Skill | Purpose | Inputs | Outputs | Verifier focus |
|---:|---|---|---|---|---|
| 1 | `parse_question_target` | Extract target entities, events, time constraints, answer format, and question focus. | `question_text`, `options?` | `target_entities`, `target_events`, `constraints`, `answer_format` | required targets are present |
| 2 | `propose_evidence_roles` | Propose reusable evidence roles needed to answer the question. | `question_text`, `parsed_target`, `task_family?` | `evidence_roles`, `role_constraints`, `expected_chain_shape` | roles are typed and relevant |
| 3 | `retrieve_by_event` | Retrieve event/evidence nodes matching an event description. | `event_description`, `time_range?`, `entity_filter?` | `event_nodes`, `evidence_refs`, `retrieval_scores` | retrieved nodes match event intent |
| 4 | `retrieve_by_entity` | Retrieve an entity timeline, history, or related evidence. | `entity_id`, `time_range?`, `predicate_filter?` | `entity_timeline`, `evidence_refs` | evidence refers to the same entity |
| 5 | `retrieve_by_time` | Retrieve evidence around an anchor event or time window. | `anchor_event_or_time`, `window_before`, `window_after` | `neighbor_events`, `evidence_refs` | timestamps overlap requested window |
| 6 | `retrieve_by_relation` | Query graph paths or relation edges such as `same_entity`, `temporal_next`, or `causal_hint`. | `source_node`, `relation_type`, `hop_limit?` | `related_nodes`, `path_edges`, `evidence_refs` | relation path is valid |
| 7 | `localize_clue` | Select the most relevant clue span/node for a requested role. | `candidate_evidence`, `role_constraint`, `question_context` | `clue_refs`, `clue_spans`, `confidence` | clue supports the requested role |
| 8 | `extract_claim` | Extract a claim from dialogue, annotation, or evidence text. | `evidence_ref`, `speaker_hint?`, `claim_query?` | `claim_id`, `claim_text`, `speaker?`, `evidence_ref` | claim is anchored to evidence |
| 9 | `assign_evidence_role` | Bind evidence to a role such as `stated_claim`, `contradicting_event`, or `motive_cue`. | `evidence_ref`, `role_schema`, `question_context` | `role_labeled_evidence`, `role_confidence` | role assignment matches content |
| 10 | `compose_evidence_chain` | Assemble role-labeled evidence into an answer-support chain. | `role_labeled_evidence`, `dependency_template` | `evidence_chain`, `chain_edges`, `missing_roles` | chain covers required roles |
| 11 | `detect_missing_role` | Identify missing evidence roles and generate query hints. | `evidence_chain`, `required_roles` | `missing_roles`, `suggested_queries` | missing roles are truly absent |
| 12 | `search_counterevidence` | Find evidence that contradicts or weakens a claim. | `claim`, `supporting_evidence`, `search_scope` | `counterevidence_refs`, `counter_claims` | counterevidence is relevant |
| 13 | `infer_temporal_relation` | Infer before/after/overlap/order among events. | `event_refs`, `evidence_refs` | `temporal_relation`, `supporting_evidence` | timestamps support relation |
| 14 | `infer_state_change` | Infer before/after state change for an entity or object. | `entity_or_object`, `state_predicate`, `before_after_refs` | `state_change_claim`, `before_state`, `after_state` | states are grounded and ordered |
| 15 | `infer_causal_relation` | Infer cause-effect support between events or states. | `candidate_cause`, `candidate_effect`, `evidence_chain` | `causal_claim`, `supporting_roles` | cause precedes effect and evidence links them |
| 16 | `infer_intention_or_motive` | Infer agent intention, goal, or motive from actions and context. | `agent`, `actions`, `context_evidence` | `intention_claim`, `alternatives`, `supporting_roles` | intention is evidence-supported |
| 17 | `infer_social_contradiction` | Infer conflict between statement/alibi/promise and later action/evidence. | `claim_or_alibi`, `evidence_chain`, `counterevidence?` | `contradiction_claim`, `supporting_evidence` | claim and evidence cannot both hold |
| 18 | `verify_claim_support` | Verify that an evidence chain supports a claim. | `claim`, `evidence_chain`, `support_policy` | `verification_score`, `passed`, `failure_code`, `messages` | evidence entails or supports claim |
| 19 | `commit_answer` | Map a verified claim to final answer text or MCQ option and record support. | `verified_claim`, `options?`, `answer_format`, `support_chain` | `final_answer`, `answer_support_chain`, `confidence` | final answer follows from verified claim |

## Composed Skill Coverage

Composed skills are reusable motifs assembled from these atomic units. They are
not new primitive actions.

| Composed motif | Example expansion |
|---|---|
| `resolve_alibi_contradiction` | `parse_question_target` -> `propose_evidence_roles` -> `retrieve_by_entity` -> `extract_claim` -> `retrieve_by_time` -> `assign_evidence_role` -> `compose_evidence_chain` -> `infer_temporal_relation` -> `infer_social_contradiction` -> `verify_claim_support` -> `commit_answer` |
| `explain_state_change` | `parse_question_target` -> `propose_evidence_roles` -> `retrieve_by_entity` -> `retrieve_by_time` -> `assign_evidence_role` -> `compose_evidence_chain` -> `infer_state_change` -> `infer_causal_relation` -> `verify_claim_support` -> `commit_answer` |
| `find_long_video_clue` | `parse_question_target` -> `propose_evidence_roles` -> `retrieve_by_event` -> `retrieve_by_time` -> `retrieve_by_relation` -> `localize_clue` -> `assign_evidence_role` -> `compose_evidence_chain` -> `verify_claim_support` -> `commit_answer` |
| `infer_motive_from_context` | `parse_question_target` -> `retrieve_by_entity` -> `retrieve_by_event` -> `compose_evidence_chain` -> `infer_intention_or_motive` -> `verify_claim_support` -> `commit_answer` |

## Expert-Demo Activation

For the first expert-demo experiments, graph-construction skills define the
offline graph builder and audit trace. The controller-visible action set should
come from the Reasoning Graph Assembly Skills. Later `video_only` experiments
can activate some graph-construction skills as tool-mediated actions.

`segment_video_or_select_clip` is the graph-construction entry point for clip
segmentation. Short, long, and streaming videos share the same skill interface
but use different `clip_policy` values. See
[clip-processing-policy.md](../docs/clip-processing-policy.md) for regime
defaults, benchmark presets, streaming visibility rules, and implementation
status.

## Implementation Staging

### Stage A: Expert-Demo Reasoning Assembly

Goal: produce valid reasoning skill traces over prebuilt evidence-memory graphs
for a small Video-Holmes and CG-Bench subset. This proves the method is not just
free-text CoT before we tackle raw-video graph construction.

Required datasets:

- Video-Holmes for social contradiction / intention / causal clues.
- CG-Bench mini for clue localization and evidence support.

Required metrics:

- final answer accuracy
- evidence role coverage
- evidence-chain support
- verifier pass rate
- answer commit validity
- repair success after missing evidence

### Stage B: Broader Reasoning Coverage

Goal: expand Reasoning Graph Assembly Skills to memory-heavy and long-video
tasks while keeping graph construction mostly offline.

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

### Stage C: Video-Only Graph Construction

Goal: activate selected Evidence Graph Construction Skills as tool-mediated
actions once expert-demo reasoning assembly is stable.

Add:

- raw or semi-raw clip observations
- automatic captions/ASR
- entity linking across clips
- graph edge construction without hidden clue intervals

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
