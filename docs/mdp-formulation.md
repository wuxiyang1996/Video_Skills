# MDP Formulation for Atomic Skill Control

Last updated: 2026-07-06

This note defines how the two graph layers in `video_skills_relaunched` can be
viewed as a Markov decision process where atomic skill invocations are actions.

The high-level controller split has three agents:

```text
Agent 1: L1 Graph Crafter
  crafts visible clue/evidence graphs without committing answers

Agent 2: L2 Recursive Reasoning / Answer Agent
  crafts the L2 reasoning graph, runs bounded recursive repair, and commits or abstains

Agent 3: Motif Extraction and Management Agent
  mines accepted L2 graphs for reusable atomic-subgraph priors
```

The MDP below primarily describes Agent 2 and its future closed-loop training.
Agent 1 defines the evidence-state transition substrate, while Agent 3 updates
optional motif priors outside the single-example answer-critical path.

## Implementation Staging

The MDP formulation describes the **target agent system**. Implementation is
staged so that data collection and model training can proceed incrementally:

### Stage 0: Expert Demo Craft + Bounded Repair Traces (current)

The current implementation uses gpt-oss as an **open-loop expert planner** for
the initial L2 reasoning rollout, then records bounded repair as an
MDP-compatible trajectory when verification is weak. This is intentionally not
yet RL training:

- gpt-oss plans a full skill sequence given (question, L1 evidence graph).
- The initial plan is executed sequentially.
- If verification is weak, the repair protocol records a bounded recursive
  round: gap diagnosis, evidence selection, optional L1 patch, option
  verification, objective bridge verification, and commit/abstain.
- The logged state is a compact graph snapshot, not a duplicated full graph.
- No learned action mask or learned policy is applied yet.

This produces **expert demonstration trajectories** that serve as:

1. Supervised training data for imitation learning (behavioral cloning).
2. Offline RL dataset (state, action, reward tuples extracted post-hoc).
3. Validation of the skill ontology and graph schema under real LLM outputs.

The initial L2 planner lives in `dataset_clip_wrapper/l2_reasoning_graph/reasoning_planner.py`.
The repair protocol lives in `dataset_clip_wrapper/run_repair_protocol.py`.
Both write `l2_trajectory` records using
`dataset_clip_wrapper/l2_reasoning_graph/l2_recursive_trace.py`.

The current process should be described as:

```text
POMDP/Semi-MDP-compatible bounded recursive graph agent trace
```

not as a trained MDP policy. The hidden video semantics are only partially
observed through selected clips and VLM outputs, while repair stages are
macro-actions with variable cost and duration.

### Stage 1: Closed-Loop MDP Controller (future)

The learned controller will follow the full MDP specification below:

- Observe current state `s_t` (including partial graph and verifier feedback).
- Select one action `a_t = (skill_id, typed_args)` via policy `π(a|s)`.
- Execute the skill and observe transition `s_{t+1} = T(s_t, a_t)`.
- Apply action masks from skill preconditions.
- Support repair: if a skill fails, the controller can retry with different
  arguments or choose an alternative skill.
- Budget and termination conditions gate the episode.

The transition from Stage 0 to Stage 1 requires:

- Collecting sufficient expert trajectories across datasets and regimes.
- Defining the reward function over collected rollouts.
- Training or fine-tuning a policy model on the offline data.
- Implementing the closed-loop execution harness with action masks and budget.

```text
Stage 0 (now):   gpt-oss open-loop planner → expert trajectories
                   + bounded repair trajectory logs
                                                    ↓
Stage 1 (next):  trajectories → train policy → closed-loop MDP controller
```

### Train / Test Boundary

Use one benchmark training split as the source for all learning signals, but
keep the roles separate:

```text
train prompts
  -> expert_demo rollouts
  -> SFT / behavioral cloning targets
  -> corrupted traces for repair training
  -> current-policy sampled rollouts for verifier-grounded RL / GRPO

validation prompts
  -> prompt/schema iteration
  -> reward-weight selection
  -> motif acceptance thresholds
  -> early stopping

test prompts
  -> final held-out evaluation only
```

The same train examples may seed expert rollouts, SFT targets, and later
policy-sampled RL rollouts. The important constraint is that official test
examples are never used for expert planning, SFT, reward tuning, motif mining,
or GRPO sampling.

SFT should use accepted expert trajectories. GRPO-style training should sample
multiple candidate skill graphs from the current policy on train prompts and
score them with answer, evidence, verifier, leakage, and cost rewards. It should
not simply replay the same expert trajectories.

### Offline RL Positioning

Stage 0 can be converted into an offline RL dataset:

```text
(s_t, a_t, r_t, s_{t+1}, done, metadata)
```

where `s_t` is the partial two-layer graph state, `a_t` is a typed skill
invocation, `r_t` comes from answer/evidence/verifier/cost checks, and
`metadata` records mode, provenance, dataset, and hidden-supervision visibility.

This supports an honest offline-RL framing for the logged expert-demo corpus.
However, GRPO is not purely fixed-dataset offline RL if the current policy
generates new rollouts during training. A better description is:

```text
static-benchmark, verifier-grounded policy optimization over typed skill actions
```

The dataset and videos are fixed, but rollouts can be newly sampled from the
current controller and scored by deterministic or model-assisted verifiers.

### Skill and Knowledge Accumulation

The controller does not accumulate new primitive skills during training. The
atomic skill basis is frozen before controller training.

What may accumulate:

- model parameters that improve skill selection, argument binding, and repair;
- accepted rollout logs and failure traces;
- verifier-calibrated rewards and diagnostics;
- mined composed motifs from frequent verified atomic subgraphs;
- episode-local evidence graphs built from visible video/tool outputs.

Mined motifs are planning or repair priors, not new black-box actions. Before
execution, every motif must expand into atomic skill invocations from the frozen
basis.

---

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
current bounded repair, `G_evidence_t` may receive a non-destructive L1 patch
from additional VLM clip schemas, while commonsense/background bridges stay in
`G_reasoning_t` and are marked as not-direct visual evidence. For Stage C,
graph-construction skills can also update `G_evidence_t`.

The implementation records compact state snapshots in `l2_trajectory.rounds[]`:

```text
state_snapshot:
  l1: graph id, node/edge counts, regime, observation boundary
  l2: rollout id, node/edge/claim counts, acceptance status
  repair_plan: gap types, selected spans, retrieval-round count
  l1_patch: patch node/edge counts
```

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

The first controller-visible action set should include the 19 core Reasoning
Graph Assembly skills:

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

For multi-hop and complex social MCQ, the current expert-demo action set also
includes 6 option-level extensions:

```text
generate_answer_hypotheses
retrieve_evidence_for_hypothesis
score_hypothesis_support
compare_hypotheses
bridge_evidence_hops
verify_temporal_social_consistency
```

These are deliberately generic atomic skills. They support hypothesis
competition, option-specific evidence retrieval, multi-hop evidence bridging,
and consistency checking without introducing a hand-coded social motif library.

Some verification operations are intentionally atomic actions. They update the
reasoning state and can influence later action selection:

```text
verify_claim_support
verify_temporal_social_consistency
score_hypothesis_support
compare_hypotheses
```

These should stay inside `G_reasoning_t` as `SkillInvocationNode`s or associated
verification results. They do not require a third graph layer.

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

There are two kinds of verification:

1. **Atomic verification skills** are actions selected by the controller, such
   as checking claim support or comparing scored hypotheses. Their outputs are
   part of `G_reasoning_t`.
2. **Runtime verifier invariants** are hard acceptance gates applied by the
   environment after or between actions. They are not actions. Examples include
   JSON schema validity, evidence-ref existence, hidden-supervision leakage,
   streaming visibility (`time_span.end_s <= observation_end_s`), and the rule
   that retrieval scores alone do not prove answer support.

The state may store runtime verifier results in `verifier_state_t`, but this is
state metadata, not a separate verifier graph.

Current repair traces also include tool-level macro-actions:

```text
call_gptoss_reasoning_planner
bounded_recursive_repair
option_evidence_selector
verify_claim_support
objective_background_bridge
commit_or_abstain
```

These are MDP-compatible action records. They are not yet actions sampled by a
learned policy.

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

### Progressive Reward From Hidden Ground Truth

Most target benchmarks provide a final answer, but they do not provide equally
dense evidence or reasoning supervision. The training/evaluation boundary is:

```text
policy-visible inputs:
  video_only video/question/options + visible L1 evidence graph

reward/evaluator-only inputs:
  hidden answer, hidden clue intervals, hidden annotations,
  hidden reasoning_process, and other dataset supervision
```

Ground truth may score a completed rollout, but it must not be injected into
L1, L2 planning, repair selection, or verifier prompts as visible evidence.
Any path that copies gold answers, clue intervals, official reasoning steps, or
dataset annotations into `video_only` visible inputs is leakage and should get
a hard rejection.

Keep evaluation and training reward separate:

```text
evaluation metrics:
  binary / exact / held-out checks such as answer_correct,
  accepted_strong, evidence_valid, no_hidden_leakage

training reward:
  RLVR-style progressive reward from verifier/rule/GT checks
```

Evaluation should report hard 0/1 or True/False outcomes. Evidence recall,
timestamp IoU, support-ref count, and repair progress may appear as diagnostics,
but they should not replace final held-out correctness and acceptance metrics.

Training can use progressive reward rather than only final-answer reward:

```text
R0 schema_reward:
  valid L1/L2 schema, valid skill ids, valid JSON, resolved evidence refs

R1 visibility_reward:
  no hidden-supervision leakage, no answer-copy shortcut, legal timestamps

R2 evidence_reward:
  retrieved refs overlap clue intervals / reasoning timestamps when available,
  selected coarse/fine neighborhoods contain target evidence,
  evidence precision is not just broad lexical overlap

R3 reasoning_chain_reward:
  role coverage, temporal order, evidence-chain structure, bridge validity,
  repair improves the evidence pack

R4 verifier_reward:
  verify_claim_support passes with non-diagnostic visual refs,
  option evidence selector finds positive refs,
  confidence/margin/support-count thresholds are met

R5 answer_reward:
  final answer matches hidden gold when the rollout commits,
  wrong strong commits receive a large penalty,
  abstention can be neutral or positive when evidence is genuinely insufficient
```

A reasonable first RLVR training reward shape is:

```text
R = 0.10 * R0_schema
  + 0.15 * R1_visibility
  + 0.25 * R2_evidence
  + 0.20 * R3_reasoning_chain
  + 0.20 * R4_verifier
  + 0.10 * R5_answer
  - cost_penalty
```

The weights should be dataset-aware:

| Dataset | Final answer GT | Evidence/process GT | Reward emphasis |
|---------|-----------------|---------------------|-----------------|
| Video-Holmes | strong | segment descriptions, inference shots, relationships, explanations | evidence roles, social/causal support, verifier, answer |
| CG-Bench | strong | clue intervals and clue clips | clue localization, retrieval neighborhood, evidence precision, answer |
| VRBench | strong | timestamped reasoning_process and summaries | temporal chain, multi-step evidence order, answer |
| SIV-Bench | strong | weak; mostly subtitles/video, no explicit clue intervals | final answer, verifier support, weak transcript/video alignment |

For CG-Bench and VRBench, evidence/timestamp rewards can be weighted strongly.
For Video-Holmes, evidence-role and verifier rewards should carry more weight.
For SIV-Bench, dense evidence terms should be lower-confidence because explicit
clue intervals are not provided.

The logged `reward_proxy` is intentionally simple and diagnostic:

- `1.0` for `accepted_strong` / `resolved_strong`;
- `0.65` for `accepted_bridge`;
- negative values for unsupported weak commits, rejected answers, or
  `needs_more_evidence`.

This is not a final training reward. It makes trajectories auditable and
convertible to offline-RL style tuples later.

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
  19 core Reasoning Graph Assembly skills
  + 6 option-level multi-hop/social extensions

goal:
  produce a verified answer support chain
```

This is the strongest first formulation because the evidence graph is already
available from dataset annotations, captions, subtitles, or deterministic graph
builders. The controller learns how to assemble reasoning programs over a
prebuilt graph. For the paper, this is the controlled reasoning-assembly
ablation, not the final task.

### Stage C: Video-Only Graph Construction MDP

```text
state:
  question + partial clue-memory graph + partial SkillGraphRollout

actions:
  selected Evidence Graph Construction skills
  + 19 core Reasoning Graph Assembly skills
  + 6 option-level multi-hop/social extensions

goal:
  discover evidence from visible video/tool outputs and answer without hidden
  supervision
```

This is the broad ICLR-facing objective: the agent must build enough evidence
state from visible video/tool outputs and then assemble a verified reasoning
graph without access to hidden clues, official reasoning, or answers. It has a
larger action space and more expensive transitions than Stage A, so the paper
should report it together with diagnostic ablations rather than as a single
opaque number.

Recommended broad-version evaluation ladder:

```text
E0: expert_demo / prebuilt evidence graph
    Measures trace-to-skill fitting and reasoning graph assembly.

E1: video_only / frozen evidence discovery
    Uses automatic clips/captions/retrieval, but freezes the controller.
    Measures whether the evidence substrate has enough recall.

E2: video_only / trained controller
    Uses the same automatic evidence substrate, but trains the skill controller.
    Measures policy learning over imperfect evidence.

E3: video_only / repair or GRPO
    Adds verifier-grounded repair or policy optimization.
    Measures whether process feedback improves evidence grounding and answer
    quality without hidden-supervision leakage.
```

A convincing broad result does not require every perception component to be
novel. It requires showing that, under the same video-only evidence interface,
typed skill-graph control and verifier feedback improve over free-form CoT,
flat retrieval-augmented QA, and linear tool-chain baselines.

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
