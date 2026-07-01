# MDP Formulation for Atomic Skill Control

Last updated: 2026-07-01

This note defines how the two graph layers in `video_skills_relaunched` can be
viewed as a Markov decision process where atomic skill invocations are actions.

## Core View

The project has two graph objects with different roles:

```text
ClueMemoryGraph / EvidenceGraph
  = environment-side evidence state: clips, observations, dialogue, events,
    entities, retrieval metadata, provenance, and visibility constraints

SkillGraphRollout
  = controller-side action trace: skill invocations, dependencies, claims,
    verifier results, repairs, and final answer support
```

These should remain separate runtime objects. The MDP state can contain both,
but the evidence graph should not be collapsed into the reasoning graph.

## State

At step `t`, define the state as:

```text
s_t = (
  q,
  G_evidence_t,
  G_reasoning_t,
  mode,
  observation_end_s,
  verifier_state_t,
  budget_state_t
)
```

where:

- `q` is the normalized question, options, answer format, and task family.
- `G_evidence_t` is the current clue-memory / evidence graph.
- `G_reasoning_t` is the partial skill graph rollout built so far.
- `mode` is `expert_demo`, `video_only`, or a streaming-compatible setting.
- `observation_end_s` gates future evidence for streaming or partial-video runs.
- `verifier_state_t` records local pass/fail status, failure codes, and claim
  support status.
- `budget_state_t` tracks remaining retrieval calls, perception calls, tokens,
  latency, or maximum rollout length.

For Stage A, `G_evidence_t` is usually fixed and only `G_reasoning_t` grows. For
Stage C, graph-construction skills can also update `G_evidence_t`.

## Actions

An action is a typed atomic skill invocation:

```text
a_t = (skill_id, typed_args)
```

The action is not just the skill id. For example, the action should be:

```text
retrieve_by_time(anchor_event_or_time, window_before=30, window_after=30)
```

not merely:

```text
retrieve_by_time
```

This distinction matters because the skill id defines the operator, while the
arguments define the actual graph read/write operation.

The first controller-visible action set should be the 19 Reasoning Graph
Assembly skills:

```text
parse_question_target
propose_evidence_roles
retrieve_by_event
retrieve_by_entity
retrieve_by_time
retrieve_by_relation
localize_clue
extract_claim
assign_evidence_role
compose_evidence_chain
detect_missing_role
search_counterevidence
infer_temporal_relation
infer_state_change
infer_causal_relation
infer_intention_or_motive
infer_social_contradiction
verify_claim_support
commit_answer
```

In later `video_only` experiments, selected Evidence Graph Construction skills
can also become tool-mediated actions:

```text
segment_video_or_select_clip
extract_observation
extract_dialogue_span
detect_entity_mention
resolve_entity_coreference
create_event_node
create_state_node
link_graph_relation
assign_provenance_trust
```

## Transition

The transition applies the selected skill, validates its local result, and
updates the graph state:

```text
s_{t+1} = T(s_t, a_t)
```

Typical transition effects include:

- append a `SkillInvocationNode` to `G_reasoning_t`;
- add data/control/evidence/claim-support edges between skill nodes;
- retrieve or bind evidence refs from `G_evidence_t`;
- add or update claims and claim statuses;
- write verifier outputs, failure codes, and confidence;
- in Stage C, append new clip, observation, dialogue, event, entity, state, or
  provenance nodes to `G_evidence_t`.

Action masks should be derived from skill preconditions. For example,
`commit_answer` should be unavailable until there is a verified claim or a
support chain, and `retrieve_by_time` should require either a time span or an
anchor evidence/event node.

## Reward

The reward should combine final task success with process-level verification:

```text
R = answer_reward
  + evidence_support_reward
  + verifier_pass_reward
  + role_coverage_reward
  + repair_success_reward
  - cost_penalty
  - leakage_penalty
```

Useful components:

- final answer correctness for MCQ or free-form scoring;
- every committed claim has valid `supported_by_refs`;
- cited evidence exists and respects timestamp / visibility constraints;
- evidence roles required by the question are covered;
- local repair turns failed nodes into verified nodes;
- retrieval / model / tool cost stays within budget;
- `video_only` rollouts do not cite hidden supervision.

For expert-demo training, hidden supervision may be used to label the target
rollout, but the visibility field must make this explicit. For `video_only`
evaluation, any citation of hidden clue intervals, official reasoning processes,
or official answers should receive a hard leakage penalty or rejection.

## Stage-Specific MDPs

### Stage A: Reasoning Assembly MDP

```text
state:
  question + fixed clue-memory graph + partial SkillGraphRollout

actions:
  19 Reasoning Graph Assembly skills

goal:
  produce a verified answer support chain
```

This is the strongest first formulation because the evidence graph is already
available from dataset annotations, captions, subtitles, or deterministic graph
builders. The controller learns how to assemble reasoning programs over a
prebuilt graph.

### Stage C: Video-Only Graph Construction MDP

```text
state:
  question + partial clue-memory graph + partial SkillGraphRollout

actions:
  selected Evidence Graph Construction skills
  + 19 Reasoning Graph Assembly skills

goal:
  discover evidence from visible video/tool outputs and answer without hidden
  supervision
```

This version is closer to the final video-only objective, but it has a larger
action space and more expensive transitions.

## MDP, Semi-MDP, or POMDP

The simplest framing is a finite-horizon graph-structured MDP.

For skills with variable duration or cost, such as retrieval, captioning,
verification, or raw-video perception, the formulation is more naturally a
semi-MDP: each action is a temporally extended operator with a cost and latency.

For raw-video settings where the true video state is not fully observed, the
more precise formulation is a POMDP. In that case, the clue-memory graph acts as
the agent's current belief state over video evidence.

Recommended paper wording:

```text
We model controller learning as a finite-horizon graph-structured MDP over a
two-layer video reasoning state. Atomic skills are typed actions that update a
partial reasoning graph and, in video-only settings, may also update the
evidence graph. Skill preconditions induce action masks, and verifier outputs
provide dense process rewards in addition to final answer reward.
```

