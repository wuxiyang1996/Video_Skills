# Skill Architecture Redesign — TODO Plan

**Created:** 2026-03-13  
**Status:** Planning (no code changes yet)

---

## Current State Audit

### What exists

| Component | Current State |
|-----------|--------------|
| **Skill representation** | `SkillEffectsContract` — flat effect signature: `eff_add`, `eff_del`, `eff_event`, name, description |
| **Sub-episodes** | `SubTask_Experience` class exists; produced by segmentation but **not stored in or linked to skill bank** |
| **Skill bank** | `skill_bank.jsonl` — one contract per line, no sub-episode data attached |
| **Decision agent** | Queries contracts via `query_skill_bank()` / `skill_bank_to_text()` — sees effect summaries only |
| **Skill agent** | `SkillBankAgent` does split/merge/refine/materialize at contract level — no per-sub-episode quality check |
| **Boundary detection** | `IntentionSignalExtractor` uses tag changes as boundary candidates — no penalty for frequent switching |
| **Rollout integration** | `RolloutStep` / `TrajectoryFrame` lack `intentions` field — Strategy C not implemented |

### What's missing (gap summary)

1. **No strategic definition layer** — skills are flat effects, not grouped under a strategic concept with a protocol
2. **No protocol** — decision agent gets effects but no actionable step-by-step guidance
3. **Sub-episodes not stored with skills** — segmentation produces them but they are discarded
4. **No sub-episode quality management** — no aggregate/update/drop decisions
5. **No tag-change penalty** — every tag transition is an equally-weighted boundary candidate

---

## Requirement 1: Skills as Sub-Episode Collections Under Strategic Definitions with Protocols

### Goal
A skill should be a **strategic concept** (e.g., "Corner Anchoring in 2048") that contains:
- A **protocol**: actionable decision guidance the decision agent follows (preconditions, step sequence, success criteria)
- A **sub-episode collection**: the concrete trajectory segments that evidence this skill

### Design

```
Skill {
    skill_id: str
    version: int

    # --- Strategic Definition ---
    name: str                          # Human-readable (e.g. "Corner Anchoring")
    strategic_description: str         # What this skill achieves and when to use it
    tags: List[str]                    # Canonical intention tags involved (e.g. [POSITION, MERGE])

    # --- Protocol (for Decision Agent) ---
    protocol: {
        preconditions: List[str]       # When to invoke (state predicates)
        steps: List[str]              # Ordered high-level action guidance
        success_criteria: List[str]    # How to know it worked
        abort_criteria: List[str]      # When to abandon
        expected_duration: int         # Typical number of steps
    }

    # --- Effects Contract (existing, kept) ---
    contract: SkillEffectsContract     # eff_add, eff_del, eff_event, support

    # --- Sub-Episode Collection (new, internal) ---
    sub_episodes: List[SubEpisodeRef] {
        episode_id: str
        seg_start: int
        seg_end: int
        outcome: "success" | "partial" | "failure"
        cumulative_reward: float
        quality_score: float           # Assigned by skill agent
        added_at: float
    }

    # --- Metadata ---
    n_instances: int
    created_at: float
    updated_at: float
}
```

### Changes needed

| File | Change |
|------|--------|
| `skill_agents/stage3_mvp/schemas.py` | Extend `SkillEffectsContract` or create new `Skill` class wrapping contract + protocol + sub_episode_refs |
| `skill_agents/skill_bank/bank.py` | `SkillBankMVP` stores and manages the new `Skill` objects; `skill_bank.jsonl` gains `protocol` and `sub_episodes` fields |
| `data_structure/experience.py` | `SubTask_Experience` gains `quality_score` and `outcome` classification |
| `labeling/label_and_extract_skills_gpt54.py` | Phase 2 writes sub-episode refs into skill objects instead of discarding them |
| `scripts/extract_skills_qwen.py` | Same — link sub-episodes to skills during extraction |

### Migration strategy
- Keep the existing `contract` field inside the new `Skill` wrapper for backward compat
- Old `skill_bank.jsonl` files are loadable as `Skill` objects with empty `protocol` and `sub_episodes`

---

## Requirement 2: Updatable Skill Bank and Protocols from New Rollouts

### Goal
When new rollouts come in:
- **Skill bank** is updated: new sub-episodes are added, existing skill contracts are refined
- **Protocols** are updated: outcomes of new rollouts feed back into the protocol's success/abort criteria and step guidance

### Design

#### Sub-episode ingestion from rollouts
```
New rollout → segment using IntentionSignalExtractor
    → match segments to existing skills (contract compat score)
    → if matched: append sub-episode ref to skill, recompute contract support
    → if unmatched: create candidate new skill (pending quality gate)
```

#### Protocol update loop
```
For each skill with new sub-episodes:
    1. Classify sub-episode outcomes: success / partial / failure
    2. If success rate changed significantly:
       - LLM call: "Given these new successful/failed examples, update the protocol"
       - Update preconditions, steps, success_criteria, abort_criteria
    3. Bump version
```

### Changes needed

| File | Change |
|------|--------|
| `trainer/common/metrics.py` | Add `intentions: Optional[str]` to `RolloutStep` (Strategy C prerequisite) |
| `trainer/skillbank/ingest_rollouts.py` | Map `intentions` into `TrajectoryFrame`; produce sub-episode refs |
| `trainer/skillbank/stages/stage0_predicates.py` | Parse `[TAG]` → `tag_*` predicates when intentions are present |
| `skill_agents/pipeline.py` | New `update_protocols()` stage after contract learning |
| `skill_agents/skill_bank/bank.py` | `ingest_sub_episode(skill_id, sub_episode_ref)` — append + recompute stats |
| `trainer/launch_coevolution.py` | After `em_trainer.run()`, call protocol update |

### Protocol versioning
- Each protocol update bumps `skill.version`
- Keep last N protocol versions for rollback (stored in `protocol_history` list)

---

## Requirement 3: Decision Agent Queries Protocols Only, Not Sub-Episodes

### Goal
The decision agent should see **protocols** (preconditions, steps, success/abort criteria) but never raw sub-episode trajectories.

### Current state
Already partially satisfied — `query_skill_bank()` returns contracts, not sub-episodes. But contracts are just effects, not actionable protocols.

### Design

```
query_skill_bank(key) →
    1. Retrieve top-k skills by semantic match on key vs skill.strategic_description
    2. Return for each:
       {
           skill_id,
           name,
           strategic_description,
           protocol: { preconditions, steps, success_criteria, abort_criteria, expected_duration },
           confidence: float  (based on n_instances and success rate)
       }
    3. Do NOT include: sub_episodes, contract internals (eff_add/eff_del)
```

```
skill_bank_to_text(bank) →
    For each skill:
        "[skill_id] name — strategic_description (confidence: X%, ~N steps)"
    (No sub-episodes, no effect literals)
```

### Changes needed

| File | Change |
|------|--------|
| `decision_agents/agent_helper.py` | Rewrite `query_skill_bank()` to return protocols; rewrite `skill_bank_to_text()` to show strategic descriptions |
| `skill_agents/query.py` | `SkillQueryEngine` indexes on `strategic_description` + `protocol.preconditions` instead of (or in addition to) effect literals |
| `decision_agents/agent.py` | Update QUERY_SKILL tool response formatting to display protocol steps |

### Enforcement
- `Skill.to_decision_agent_view()` method that strips sub-episodes and contract internals
- All decision-agent-facing functions use this view exclusively

---

## Requirement 4: Skill Agent Checks Sub-Episode Quality, Aggregates/Updates/Drops

### Goal
A dedicated skill agent (or extended `SkillBankAgent`) inspects sub-episodes and makes quality decisions:
- **Aggregate**: group similar sub-episodes to strengthen a skill
- **Update**: revise the skill when new sub-episodes show better strategies
- **Drop**: remove low-quality sub-episodes that hurt the skill

### Design

#### Quality scoring per sub-episode
```
quality_score(sub_ep) = weighted combination of:
    - outcome_reward:   normalized cumulative reward of the segment
    - follow_through:   did the skill's success_criteria get met?
    - consistency:      does the tag sequence match the skill's expected tag pattern?
    - compactness:      segment length vs. expected_duration (penalty for too long/short)
```

#### Skill agent decisions (new Stage 4.5 in pipeline)
```
For each skill:
    sub_eps = skill.sub_episodes (sorted by quality_score)

    # DROP: remove bottom-quality sub-episodes
    drop_threshold = adaptive (e.g. bottom 20% or absolute threshold)
    for se in sub_eps:
        if se.quality_score < drop_threshold:
            skill.sub_episodes.remove(se)

    # AGGREGATE: if enough high-quality sub-episodes exist, summarize them
    if len(high_quality_sub_eps) >= min_aggregate_count:
        protocol = synthesize_protocol(high_quality_sub_eps)
        skill.protocol = protocol
        skill.contract = recompute_contract(high_quality_sub_eps)

    # UPDATE: if new sub-episodes significantly differ from existing protocol
    if protocol_drift_detected(new_sub_eps, skill.protocol):
        skill.protocol = update_protocol(skill.protocol, new_sub_eps)
        skill.version += 1

    # PROMOTE / RETIRE: if too few sub-episodes remain, retire skill
    if len(skill.sub_episodes) < min_viable_count:
        mark_skill_retired(skill)
```

### Changes needed

| File | Change |
|------|--------|
| `skill_agents/pipeline.py` | Add `run_sub_episode_quality_check()` stage between Stage 3 and Stage 4 |
| `skill_agents/stage3_mvp/schemas.py` | Add `SubEpisodeRef` dataclass with `quality_score`, `outcome` |
| `skill_agents/skill_bank/bank.py` | `drop_sub_episode()`, `get_sub_episodes()`, `recompute_stats()` methods |
| New: `skill_agents/quality/sub_episode_evaluator.py` | Quality scoring function, drift detection, protocol synthesis |
| `labeling/label_and_extract_skills_gpt54.py` | After segmentation, run quality scoring on produced sub-episodes |

### Quality scoring implementation
- `outcome_reward`: min-max normalized across all sub-episodes of the same skill
- `follow_through`: check if `success_criteria` predicates appear in the sub-episode's final state
- `consistency`: count tag changes within the sub-episode vs. expected (penalize excessive switching)
- `compactness`: `1.0 - abs(len(sub_ep) - expected_duration) / expected_duration`, clipped to [0, 1]

---

## Requirement 5: Multi-Step Skills with Tag-Based Boundaries and Tag-Change Penalty

### Goal
- Skills are **multi-step** (spanning several actions under a consistent strategic intention)
- Intention tags mark the boundaries, but **frequent tag switching is penalized**
- Tag change = boundary **candidate**, not automatic boundary

### Current state
- `IntentionSignalExtractor.extract_event_times()` treats every tag change as a boundary candidate (equal weight)
- No penalty for rapid back-and-forth switching (e.g., `[MERGE] → [POSITION] → [MERGE]`)

### Design

#### Tag-change penalty model
```
For each tag change at time t:
    boundary_score(t) = base_score
                        - consistency_penalty * rapid_switch_indicator(t)
                        + reward_signal_bonus(t)
                        + state_change_bonus(t)

where:
    rapid_switch_indicator(t) =
        1.0 if tag[t-1] == tag[t+1] (ping-pong: A→B→A)
        0.5 if time_since_last_change(t) < min_segment_length
        0.0 otherwise

    consistency_penalty = configurable weight (default 0.3)
    min_segment_length = configurable (default 3 steps)
```

This means:
- A tag change after a long consistent run → high boundary score (good candidate)
- A tag change in a ping-pong pattern → penalized (probably noise, not a real boundary)
- A tag change supported by reward spike or state change → boosted

#### Multi-step skill enforcement
```
After boundary detection:
    for each proposed segment:
        if len(segment) < min_skill_length (e.g., 2 steps):
            merge with adjacent segment (prefer merging into the longer neighbor)
```

#### Tag sequence pattern in skills
```
Skill.expected_tag_pattern: List[str]
    e.g. ["SETUP", "MERGE", "MERGE", "POSITION"]  (most common sequence)

Used by:
    - Quality scoring (consistency dimension)
    - Skill matching (does a new sub-episode's tag sequence fit this skill?)
```

### Changes needed

| File | Change |
|------|--------|
| `skill_agents/boundary_proposal/signal_extractors.py` | `IntentionSignalExtractor.extract_event_times()` → return scored boundaries instead of flat list; add penalty logic |
| `skill_agents/boundary_proposal/signal_extractors.py` | New method `score_boundary_candidates()` implementing the penalty model |
| `skill_agents/pipeline.py` | `PipelineConfig` gains `min_segment_length`, `consistency_penalty`, `min_skill_length` |
| `skill_agents/infer_segmentation/` | Stage 2 uses boundary scores (not just positions) when decoding segments |
| `skill_agents/stage3_mvp/schemas.py` | `Skill` gains `expected_tag_pattern: List[str]` |
| `labeling/label_and_extract_skills_gpt54.py` | `_intention_based_segmentation` uses penalty model instead of raw tag-change boundaries |
| `scripts/extract_skills_qwen.py` | Same — use scored boundaries |

### Boundary scoring interface
```python
@dataclass
class ScoredBoundary:
    time: int
    score: float          # Higher = more likely a real boundary
    tag_before: str
    tag_after: str
    is_ping_pong: bool
    time_since_last: int
```

Stage 1 returns `List[ScoredBoundary]` instead of `List[int]`.  
Stage 2 uses scores as priors in the segmentation decode.

---

## Implementation Order

### Phase A: Data structure foundations (Requirements 1, 5)
1. **A1.** Define `Skill` wrapper class (contract + protocol + sub_episode_refs)
2. **A2.** Define `SubEpisodeRef` dataclass with quality fields
3. **A3.** Define `Protocol` dataclass (preconditions, steps, success/abort criteria)
4. **A4.** Define `ScoredBoundary` dataclass
5. **A5.** Update `SkillBankMVP` to store/load the new `Skill` format (backward-compat with old `skill_bank.jsonl`)

### Phase B: Boundary scoring with tag-change penalty (Requirement 5)
1. **B1.** Implement `score_boundary_candidates()` in `IntentionSignalExtractor`
2. **B2.** Add `min_segment_length`, `consistency_penalty` to `PipelineConfig`
3. **B3.** Update Stage 2 decode to consume scored boundaries
4. **B4.** Add post-processing: merge segments shorter than `min_skill_length`
5. **B5.** Update `_intention_based_segmentation` in labeling script

### Phase C: Sub-episode linking and quality (Requirements 1, 4)
1. **C1.** After segmentation, produce `SubEpisodeRef` objects linked to skills
2. **C2.** Implement quality scoring function (`sub_episode_evaluator.py`)
3. **C3.** Implement aggregate/update/drop logic as Stage 4.5 in pipeline
4. **C4.** Wire into `label_and_extract_skills_gpt54.py` and `extract_skills_qwen.py`

### Phase D: Protocol generation and decision-agent interface (Requirements 2, 3)
1. **D1.** Implement protocol synthesis from high-quality sub-episodes (LLM call)
2. **D2.** Rewrite `query_skill_bank()` to return protocols
3. **D3.** Rewrite `skill_bank_to_text()` to show strategic descriptions + protocol previews
4. **D4.** Add `Skill.to_decision_agent_view()` that strips internals
5. **D5.** Update `SkillQueryEngine` to index on `strategic_description` + `preconditions`

### Phase E: Rollout-based updates (Requirement 2, Strategy C)
1. **E1.** Add `intentions` to `RolloutStep` and `TrajectoryFrame`
2. **E2.** Populate `intentions` during rollout collection from decision agent
3. **E3.** Map intentions → predicates in `stage0_predicates.py`
4. **E4.** Implement protocol update loop after EM training
5. **E5.** Protocol versioning and rollback

---

## Key Design Principles

1. **Separation of concerns**: Decision agent sees protocols; skill agent manages sub-episodes and contracts. These are separate views of the same `Skill` object.

2. **Backward compatibility**: Old `skill_bank.jsonl` files load as `Skill` objects with empty `protocol` and `sub_episodes`. No migration script needed.

3. **Tag changes are candidates, not boundaries**: The penalty model ensures that only meaningful tag transitions become segment boundaries. Ping-pong patterns and rapid switching are suppressed.

4. **Quality-gated protocol updates**: Protocols are only updated when enough high-quality sub-episodes exist. Low-quality evidence is dropped, not incorporated.

5. **Sub-episodes are internal**: They live inside `Skill` objects for the skill agent to inspect and manage. They are never exposed to the decision agent.

---

## Files Affected (Summary)

| File | Phases |
|------|--------|
| `skill_agents/stage3_mvp/schemas.py` | A1, A2, A3, A4 |
| `skill_agents/skill_bank/bank.py` | A5, C1, C3 |
| `skill_agents/boundary_proposal/signal_extractors.py` | B1, B4 |
| `skill_agents/pipeline.py` | B2, C3, E4 |
| `skill_agents/infer_segmentation/` (Stage 2) | B3 |
| New: `skill_agents/quality/sub_episode_evaluator.py` | C2 |
| `labeling/label_and_extract_skills_gpt54.py` | B5, C4, D1 |
| `scripts/extract_skills_qwen.py` | B5, C4 |
| `decision_agents/agent_helper.py` | D2, D3 |
| `decision_agents/agent.py` | D3 |
| `skill_agents/query.py` | D5 |
| `trainer/common/metrics.py` | E1 |
| `trainer/skillbank/ingest_rollouts.py` | E1, E2 |
| `trainer/skillbank/stages/stage0_predicates.py` | E3 |
| `trainer/launch_coevolution.py` | E4 |
| `data_structure/experience.py` | A2 |
