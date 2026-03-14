# Skill Bank GRPO + Tool-Calling Agent Plan

**Created:** 2026-03-14  
**Updated:** 2026-03-14 — Rewritten to target production `skill_agents/` modules per `SKILLBANK_GRPO_PLAN_REWRITE.md` gap analysis  
**Status:** Draft (v2)  
**Depends on:** `SKILLBANK_AUDIT_GAPS.md`, existing Hard-EM pipeline, Decision Agent GRPO trainer

---

## Overview

Three workstreams:

1. **Stages 1–3 GRPO**: Train GRPO LoRA adapters on Qwen3-14B for the LLM generation tasks in each stage — boundary proposal, segment preference generation, and contract generation — rewarded by downstream metrics. The adapters augment (not replace) the existing algorithmic/heuristic infrastructure in `skill_agents/`.
2. **Stage 4 LLM-Advised Bank Maintenance**: Add a single-turn LLM filter (CURATOR LoRA) to the algorithmic bank maintenance pipeline in `skill_agents/bank_maintenance/`. The algorithm proposes candidates (split/merge/refine/materialize/promote); the LLM approves, vetoes, or defers.
3. **`select_skill` GRPO**: Train the decision agent's skill-selection policy with GRPO so it learns _when_ to query, _which_ skill to pick, and _when_ to switch — rewarded by downstream episode return.

**Target codebase:** This plan targets the production `skill_agents/` implementations, NOT the simplified EM trainer stages in `trainer/skillbank/stages/`. The trainer stages are kept as fallback baselines.

| Stage | Production module (GRPO target) | Simplified fallback |
|-------|-------------------------------|---------------------|
| 1 — Boundary | `skill_agents/boundary_proposal/` | `trainer/skillbank/stages/stage1_propose_cuts.py` |
| 2 — Segmentation | `skill_agents/infer_segmentation/` | `trainer/skillbank/stages/stage2_decode.py` |
| 3 — Contracts | `skill_agents/stage3_mvp/` | `trainer/skillbank/stages/stage3_contracts.py` |
| 4 — Maintenance | `skill_agents/bank_maintenance/` | `trainer/skillbank/stages/stage4_update.py` |

**Model convention:** This project uses a single Qwen model size throughout — **Qwen3-14B** for all components (vLLM serving, LoRA adapters, decision agent, tool-calling). No mixed model sizes. All existing references to Qwen3-8B in the codebase (`skill_agents/lora/config.py`, `trainer/common/configs/skillbank_em.yaml`, etc.) must be updated to Qwen3-14B.

### GRPO-Trained LoRA Adapters

5 LoRA adapters on the shared Qwen3-14B base, each trained independently via GRPO:

| Adapter | Stage | LLM Generation Task | Augments / Replaces |
|---------|-------|---------------------|---------------------|
| BOUNDARY (LoRA #1) | 1 | Generate boundary cut proposals from multi-signal trajectory input | Augments `propose_boundary_candidates()` in `boundary_proposal/proposal.py` |
| SEGMENT (LoRA #2) | 2 | Generate segment-skill pairwise preferences that feed into `PreferenceStore` + Bradley-Terry scorer | Replaces LLM teacher calls (`collect_segment_preferences`, `collect_transition_preferences`, `collect_uncertain_prefs`) in `infer_segmentation/llm_teacher.py` |
| CONTRACT (LoRA #3) | 3 | Generate effects contracts from segment evidence (union enrichment) | Augments frequency-based `learn_effects_contract()` in `stage3_mvp/contract_learn.py` |
| CURATOR (LoRA #4) | 4 | Single-turn filter — approve/veto/defer candidate bank mutations | Augments algorithmic `run_bank_maintenance()` in `bank_maintenance/run_bank_maintenance.py` |
| RETRIEVAL (LoRA #5) | select_skill | Re-rank skill candidates for decision agent | Augments `SkillQueryEngine.select()` fixed scoring |

### Non-GRPO Functions (Infrastructure)

These `SkillBankAgent` functions are NOT LLM generation tasks. They stay as-is and run as infrastructure around the GRPO-trained stages:

| Function | Role | Why not GRPO |
|----------|------|-------------|
| `run_sub_episode_quality_check()` | Data quality gate between Stage 3 and Stage 4 — scores sub-episodes, retires depleted skills | Heuristic scoring (outcome_reward, follow_through, consistency, compactness). No LLM. |
| `distill_execution_hints()` | Post-Stage 4 — derives termination cues, failure modes, micro-plans from successful sub-episodes | Deterministic extraction from sub-episode patterns. No LLM. |
| `update_protocols()` | Post-Stage 4 — LLM synthesizes Protocol objects from high-quality sub-episodes. Stays as plain LLM inference (not GRPO) because: reward signal too indirect, data volume too small per skill, self-correcting via quality loop. | Summarization task, not judgment. |
| `_apply_alias_map()` | Post-merge bookkeeping — relabels segments referencing merged-away skills | Mechanical string replacement. |
| `_get_or_create_preference_store()` | Stage 2 integration — `PreferenceStore` accumulates pairwise preferences. GRPO-generated preferences feed INTO this store. | Infrastructure, not a generation task. |
| `run_until_stable()` / `_is_converged()` | Outer loop convergence detection | Orchestration logic. |
| `form_proto_skills()` / `verify_proto_skills()` / `promote_proto_skills()` | Proto-skill pipeline — execution logic is algorithmic; CURATOR LoRA decides whether to approve | LLM role is in CURATOR filter, not formation/verification. |

### Co-Evolution Training Loop

```
for co_evolution_step in range(total_steps):
    # ── Decision agent GRPO (existing) ──
    rollouts = collect_rollouts(decision_agent, env, skill_bank)
    decision_grpo_update(rollouts)                              # trains RETRIEVAL LoRA

    if co_evolution_step % bank_update_cadence == 0:
        trajectories = ingest_rollouts(rollouts)

        # ── Stage 1 GRPO ──
        stage1_grpo_step(trajectories, bank)                    # trains BOUNDARY LoRA

        # ── Stage 2 GRPO ──
        stage2_grpo_step(trajectories, bank)                    # trains SEGMENT LoRA
                                                                # (generates preferences → PreferenceStore)

        # ── Stage 3 GRPO ──
        stage3_grpo_step(trajectories, bank)                    # trains CONTRACT LoRA

        # ── Infrastructure: data quality gate (Stage 4.5) ──
        run_sub_episode_quality_check()                         # heuristic, no LLM

        # ── Stage 4: algorithm proposes, LLM filters ──
        candidates = propose_candidates(bank, bank_maintenance) # algorithmic (SkillProfile, indices)
        approved = filter_candidates(candidates, bank, vllm)    # CURATOR LoRA
        execute_approved(approved, bank, new_pool, proto_mgr)   # algorithmic
        _apply_alias_map(alias_map)                             # bookkeeping

        # ── Infrastructure: post-Stage 4 ──
        distill_execution_hints()                               # heuristic, no LLM
        update_protocols()                                      # LLM inference (not GRPO)

        # ── SkillEval gating (6 dimensions + holistic) ──
        if not skilleval_passes(bank):
            rollback_bank()
```

---

## 1. Stages 1–3: GRPO-Trained Skill Bank Agents

### 1.1 Architecture

Each stage uses the shared Qwen3-14B base model with a dedicated LoRA adapter (BOUNDARY, SEGMENT, CONTRACT). During GRPO training, each adapter is updated independently. The shared base stays frozen. This is the same Qwen3-14B used for vLLM serving and the decision agent — one model size for the entire project.

Each adapter augments (not replaces) the existing algorithmic infrastructure in `skill_agents/`:

| Stage | GRPO adapter role | Existing algorithm (stays as infrastructure) |
|-------|------------------|---------------------------------------------|
| 1 | Generate boundary proposals as additional signal channel | `propose_boundary_candidates()` — signal extraction, merge, density control |
| 2 | Generate pairwise preferences (replaces LLM teacher) | `PreferenceStore` → `PreferenceScorer` (Bradley-Terry) → `SegmentScorer` (6-term) → DP/beam decoder |
| 3 | Generate contract suggestions (union enrichment) | `learn_effects_contract()` (frequency counting) → `verify_effects_contract()` → `refine_effects_contract()` |

**GPU memory note:** Qwen3-14B in bf16 ≈ 28GB base. With LoRA rank 16 and 5 adapters loaded, add ~200MB per adapter (LoRA params are tiny relative to the base). GRPO training adds optimizer states + gradients for adapter params only (~1–2GB total). Fits comfortably on a single 80GB A100. For inference via vLLM, the existing `--gpu-memory-utilization 0.75` (≈60GB on 80GB) is sufficient for 14B with generous KV cache.

```
For each EM batch of trajectories:
  Stage 1: BOUNDARY adapter generates cut proposals      → rewarded by Stage 2 decode quality
           (augments boundary_proposal/proposal.py signal pipeline)
  Stage 2: SEGMENT adapter generates pairwise preferences → rewarded by Stage 3 pass rate + follow score
           (replaces infer_segmentation/llm_teacher.py LLM teacher calls)
  Stage 3: CONTRACT adapter generates contract suggestions → rewarded by holdout verification + follow score
           (union-enriches stage3_mvp/contract_learn.py frequency consensus)
```

### 1.2 Stage 1 — Boundary Proposal (GRPO)

**Current state:** `skill_agents/boundary_proposal/` implements a multi-signal boundary proposal pipeline. `propose_boundary_candidates()` fuses 6+ signal types (predicate flips, surprisal spikes, changepoint scores via CUSUM/sliding-window cosine, intention-tag transitions with 20+ aliases, done flags, hard events) through trigger extraction → merge → density control → optional preference filtering. Outputs `List[BoundaryCandidate]` with `center`, `half_window`, and `source` attribution. The existing `LLMSignalExtractor` already uses LLM for predicate extraction per chunk.

**What changes:**

The BOUNDARY LoRA adapter generates boundary cut proposals given the full multi-signal trajectory summary. GRPO trains the adapter to produce cuts that maximize downstream Stage 2 decode quality. The adapter augments — not replaces — the existing signal pipeline: it can be used as an additional signal channel feeding into `propose_boundary_candidates()`, or as a re-ranker over the algorithm's proposals.

**Design decision:** GRPO generates preference pairs from rollouts that feed into `BoundaryPreferenceScorer` (Bradley-Terry). This integrates naturally: the scorer already accepts pairwise preferences via `add_preference(t_win, t_lose)` and feeds a `decoding_bonus()` into Stage 2.

| Component | Detail |
|-----------|--------|
| **Input (prompt)** | Multi-signal trajectory summary: per-timestep predicates (booleanized), action labels, surprisal values, changepoint scores (CUSUM + sliding-window cosine), intention tags (canonical via `_TAG_ALIASES`), done flags, hard event times (reward spikes, phase transitions). Truncated to last 200 steps if needed. |
| **Output (generation)** | JSON list of `BoundaryCandidate` objects: `[{"center": 12, "half_window": 2, "source": "grpo"}, ...]` |
| **Group generation** | `G=4` independent proposals per trajectory (low G because boundary proposals are cheap to evaluate) |
| **Reward** | Computed after running Stage 2 decode on each proposal. See reward table below. |

**Concrete I/O example:**

*Input prompt:*

```
Propose boundary cut positions for this trajectory using all available signals.

Trajectory (45 timesteps):
t=0:  preds=[at_counter, not_holding, onion_on_counter]  action=noop        surp=0.1  cp=0.02  tag=PICKUP
t=1:  preds=[at_counter, not_holding, onion_on_counter]  action=interact     surp=0.2  cp=0.05  tag=PICKUP
t=2:  preds=[at_counter, holding_onion]                  action=move_south   surp=0.8  cp=0.41  tag=CARRY     event=picked_up_onion
t=3:  preds=[near_counter, holding_onion]                action=move_south   surp=0.3  cp=0.12  tag=CARRY
...
t=11: preds=[near_pot, holding_onion, facing_pot]        action=move_south   surp=0.2  cp=0.08  tag=CARRY
t=12: preds=[at_pot, holding_onion, facing_pot]          action=interact     surp=0.9  cp=0.55  tag=SETUP     event=placed_in_pot  done=true
t=13: preds=[at_pot, not_holding, onion_in_pot]          action=noop         surp=0.7  cp=0.38  tag=NAVIGATE
t=14: preds=[at_pot, not_holding, onion_in_pot]          action=move_north   surp=0.5  cp=0.15  tag=NAVIGATE
...
t=26: preds=[at_counter, not_holding, dish_on_counter]   action=interact     surp=0.8  cp=0.48  tag=PICKUP    event=picked_up_dish
t=27: preds=[at_counter, holding_dish]                   action=move_south   surp=0.6  cp=0.32  tag=CARRY
...
t=38: preds=[at_pot, holding_dish, pot_cooked]           action=interact     surp=0.9  cp=0.61  tag=SETUP     event=plated_soup
t=39: preds=[at_pot, holding_soup]                       action=move_east    surp=0.7  cp=0.29  tag=DELIVER
...
t=44: preds=[at_serve, holding_soup]                     action=interact     surp=0.8  cp=0.52  tag=DELIVER   event=served
t=45: preds=[at_serve, not_holding, soup_delivered]      action=noop         surp=0.1  cp=0.05  tag=IDLE      done=true

Signal legend: surp=action surprisal, cp=changepoint score (CUSUM), tag=intention tag, event=hard event
ProposalConfig: merge_radius=5, window_half_width=2, soft_max_per_minute=20, tag_min_segment_len=3

Respond with a JSON list of boundary candidates.
```

*Output (generation):*

```json
[
  {"center": 12, "half_window": 2, "source": "grpo"},
  {"center": 26, "half_window": 2, "source": "grpo"},
  {"center": 38, "half_window": 2, "source": "grpo"},
  {"center": 45, "half_window": 0, "source": "grpo"}
]
```

*Interpretation:* The model proposes 4 segments: t=0–12 (pick up onion, carry to pot), t=13–26 (go get a dish), t=27–38 (carry dish to pot, plate soup), t=39–45 (serve soup). Each cut aligns with high changepoint scores, intention-tag transitions, and hard events. The `half_window=2` allows Stage 2 to fine-tune exact positions.

**Reward function for Stage 1:**

| Component | Formula | Weight | Rationale |
|-----------|---------|--------|-----------|
| `r_decode_margin` | `mean(seg.margin for seg in decode_result.segments)` normalized to [0,1] | 0.30 | Segments decoded with high margin → cuts landed at natural boundaries |
| `r_known_rate` | `1.0 - (n_new / n_total_segments)` | 0.20 | Low NEW rate → cuts align with known skills |
| `r_pass_rate` | `mean(contract.pass_rate for contract in stage3_result)` | 0.20 | Downstream contract quality validates boundary placement |
| `r_frag_penalty` | `-max(0, n_segments - 2 * n_expected) / n_expected` | 0.15 | Prevents over-fragmentation |
| `r_boundary_plausibility` | `mean(BoundaryPreferenceScorer.boundary_score_value(t) for t in cuts)` | 0.15 | Proposed cuts should align with multi-signal plausibility |

**BoundaryPreferenceScorer integration:**

GRPO rollouts generate training data for the existing `BoundaryPreferenceScorer`:

```python
# After GRPO evaluation, convert reward-ranked proposals to pairwise preferences
for (traj, group, rewards) in evaluated_proposals:
    ranked = sorted(zip(group, rewards), key=lambda x: x[1], reverse=True)
    best_cuts = ranked[0][0]
    worst_cuts = ranked[-1][0]
    for t_win in best_cuts:
        for t_lose in worst_cuts:
            if t_lose not in best_cuts:
                boundary_scorer.add_preference(t_win, t_lose)
```

This feeds back into Stage 2 via `BoundaryPreferenceScorer.decoding_bonus(seg_start, seg_end)`.

**Training loop:**

```python
for batch in trajectory_batches:
    proposals = []
    for traj in batch:
        signals = extract_signals(traj.experiences, env_name, embedder)
        prompt = format_boundary_prompt(traj, signals)
        group = [llm.generate(SkillFunction.BOUNDARY, prompt, temperature=0.7)
                 for _ in range(G)]
        proposals.append((traj, signals, group))

    rewards = []
    for traj, signals, group in proposals:
        group_rewards = []
        for candidate_json in group:
            candidates = parse_boundary_candidates(candidate_json)
            centers = candidate_centers_only(candidates)
            result = infer_segmentation(centers, len(traj), skill_names, obs, actions,
                                        predicates, config, scorer)
            r = compute_boundary_reward(result, traj, boundary_scorer)
            group_rewards.append(r)
        rewards.append(group_rewards)

    advantages = compute_grpo_advantages(rewards)
    update_lora(SkillFunction.BOUNDARY, prompts, generations, advantages)
```

**Files to create/modify:**

- [ ] `trainer/skillbank/grpo/stage1_grpo.py` — training loop, multi-signal prompt formatting, reward computation, `BoundaryPreferenceScorer` preference generation
- [ ] `trainer/skillbank/grpo/prompts.py` — shared prompt templates for all stages
- [ ] Modify `skill_agents/boundary_proposal/proposal.py` — add GRPO adapter as optional signal channel in `propose_boundary_candidates()`
- [ ] Modify `skill_agents/boundary_proposal/boundary_preference.py` — add batch preference ingestion from GRPO rollouts
- [ ] Modify `skill_agents/boundary_proposal/episode_adapter.py` — wire GRPO boundary proposer into `propose_from_episode()`
- [ ] Keep `trainer/skillbank/stages/stage1_propose_cuts.py` as simplified fallback

---

### 1.3 Stage 2 — Decode / Segmentation (GRPO)

**Current state:** `skill_agents/infer_segmentation/` implements a preference-learning pipeline:
1. **LLM teacher** (`llm_teacher.py`) ranks skills per segment → pairwise preferences
2. **PreferenceStore** accumulates `PreferenceExample` objects (segment + transition preferences)
3. **PreferenceScorer** trains Bradley-Terry model on preferences → learned behavior-fit and transition scores
4. **SegmentScorer** (`scorer.py`) computes a 6-term composite score per (segment, skill) pair
5. **Decoder** (Viterbi DP or beam search with `beam_width=16`) finds the optimal segmentation
6. **Active learning** loop: 3 iterations of uncertain-segment queries (`margin < 1.0`, max 5 per iteration), retrain, re-decode

**The 6-term scoring function (`SegmentScorer.score()`):**

```
total = w.behavior_fit       * behavior_fit(obs, actions, skill, i, j)   [default w=1.0]
      + w.duration_prior     * duration_prior(length, skill)              [default w=0.3]
      + w.transition_prior   * transition_prior(skill, prev_skill)        [default w=1.0]
      + w.contract_compat    * contract_compat(skill, preds_start, end)   [default w=0.0]
      + boundary_quality(i, j)                                            [no weight]
      + w.boundary_preference * boundary_preference(i, j)                 [default w=0.5]
```

`behavior_fit` and `transition_prior` come from the Bradley-Terry `PreferenceScorer`. `contract_compat` is a feedback loop from Stage 3 effects (configurable via `ContractFeedbackConfig`). `boundary_preference` comes from Stage 1's `BoundaryPreferenceScorer.decoding_bonus()`.

**What changes:**

GRPO replaces the LLM teacher calls (`collect_segment_preferences`, `collect_transition_preferences`, `collect_uncertain_prefs`) with the SEGMENT LoRA adapter. The adapter generates pairwise preferences that feed into the existing `PreferenceStore` → Bradley-Terry → `SegmentScorer` → decoder pipeline. This is **Option A** from the design space — GRPO replaces the LLM teacher, not the scorer or decoder.

| Component | Detail |
|-----------|--------|
| **Input (prompt)** | Per-segment: start/end predicates, action sequence, top-K candidates with `SegmentScorer.score_breakdown()` (behavior_fit, duration_prior, transition_prior, contract_compat, boundary_preference). |
| **Output (generation)** | JSON ranking: `{"ranking": ["S04_pick_and_carry_onion", "S11_navigate_to_pot", "__NEW__"], "evidence": "..."}` — converted to pairwise preferences via `ranking_to_pairwise()` |
| **Group generation** | `G=8` independent rankings per episode (higher G because preferences are more variable) |
| **Reward** | Contract pass rate + decision agent follow score + diagnostics quality. See reward table. |

**Concrete I/O example:**

*Input prompt:*

```
Rank the candidate skills for each segment. For each segment, you see predicates,
actions, and the scorer breakdown for each candidate.

Segment 1 (t=0 to t=12):
  start_preds: [at_counter, not_holding, onion_on_counter]
  end_preds:   [at_pot, holding_onion, facing_pot]
  actions:     [noop, interact, move_south x9, interact]
  candidates:
    1. S04_pick_and_carry_onion
       breakdown: behavior_fit=0.45, duration_prior=-0.12, transition_prior=0.30,
                  contract_compat=0.19, boundary_pref=0.08  total=0.90
       contract: eff_add={holding_onion, at_pot} eff_del={onion_on_counter, not_holding}
    2. S11_navigate_to_pot
       breakdown: behavior_fit=0.20, duration_prior=-0.15, transition_prior=0.25,
                  contract_compat=0.10, boundary_pref=0.08  total=0.48
       contract: eff_add={at_pot, facing_pot} eff_del={far_from_pot}
    3. __NEW__
       breakdown: behavior_fit=0.00, duration_prior=-3.00, transition_prior=0.00,
                  contract_compat=0.00, boundary_pref=0.08  total=-2.92
       (penalty=5.0)

Segment 2 (t=13 to t=26):
  start_preds: [at_pot, not_holding, onion_in_pot]
  end_preds:   [at_counter, not_holding, dish_on_counter]
  actions:     [noop, move_north x11, interact]
  candidates:
    1. S08_fetch_dish
       breakdown: behavior_fit=0.38, duration_prior=-0.10, transition_prior=0.22,
                  contract_compat=0.18, boundary_pref=0.06  total=0.74
    2. S11_navigate_to_counter
       breakdown: behavior_fit=0.15, duration_prior=-0.18, transition_prior=0.20,
                  contract_compat=0.08, boundary_pref=0.06  total=0.31
    3. __NEW__
       breakdown: behavior_fit=0.00, duration_prior=-3.00, ...  total=-2.94

For each segment, return a ranking from best to worst skill.
```

*Output (generation):*

```json
{
  "segments": [
    {
      "start": 0, "end": 12,
      "ranking": ["S04_pick_and_carry_onion", "S11_navigate_to_pot", "__NEW__"],
      "evidence": "S04 matches both holding_onion and at_pot effects; S11 only covers navigation"
    },
    {
      "start": 13, "end": 26,
      "ranking": ["S08_fetch_dish", "S11_navigate_to_counter", "__NEW__"],
      "evidence": "S08 captures dish pickup + counter arrival; S11 is too generic"
    }
  ]
}
```

*Preference conversion:* Each ranking `[A, B, C]` produces pairwise preferences `A≻B`, `A≻C`, `B≻C` via `ranking_to_pairwise()`. These are added to `PreferenceStore` with `source="grpo"`.

**Reward function for Stage 2:**

| Component | Formula | Weight | Rationale |
|-----------|---------|--------|-----------|
| `r_pass_rate` | `mean(contract.pass_rate for skill in assignment)` | 0.25 | Assigned skills should have valid contracts |
| `r_margin` | `mean(seg.margin for seg in result.segments)` | 0.15 | Confident assignments (margin = top1 - top2 score) |
| `r_follow` | `mean(r_follow_t)` from decision agent follow-shaping over holdout episodes | 0.20 | Downstream: does the decision agent benefit from this segmentation? |
| `r_new_penalty` | `-0.5 * new_rate` where `new_rate = len(result.new_segments()) / len(result.segments)` | 0.10 | Discourage excessive NEW labels |
| `r_label_entropy` | `-mean(seg.label_entropy for seg in result.segments)` | 0.10 | Low entropy = unambiguous assignments |
| `r_compat_margin` | `mean(seg.compat_margin for seg in result.segments)` | 0.10 | Contract compatibility gap between top-1 and top-2 |
| `r_confusion_penalty` | `-mean(confusion_overlap for confuser_pairs)` | 0.10 | Different skills should have different effects |

**Key design choice — GRPO replaces LLM teacher, not the decoder:**

The existing pipeline has clear separation: LLM teacher → preferences → scorer → decoder. GRPO plugs in at the LLM teacher level:

1. **SEGMENT adapter generates rankings** (replacing `collect_segment_preferences` + `collect_transition_preferences`)
2. Rankings are converted to `PreferenceExample` objects via `ranking_to_pairwise()`
3. `PreferenceStore` accumulates preferences (GRPO + any retained human/rule-based prefs)
4. `PreferenceScorer.train()` fits Bradley-Terry model (unchanged)
5. `SegmentScorer` uses trained scores for behavior_fit + transition_prior (unchanged)
6. Viterbi DP or beam decoder finds optimal path (unchanged)
7. Active learning loop: `collect_uncertain_preferences` also uses SEGMENT adapter (replaces LLM teacher queries)

This preserves the preference-learning structure while letting GRPO optimize the preference generation. The decoder, scorer weights, and active learning loop are infrastructure, not GRPO targets.

**Transition preferences (also GRPO-trained):**

The SEGMENT adapter also generates transition preferences (which `prev_skill → skill` sequences are natural). These are stored as `PreferenceExample(segment_start=-1, segment_end=-1, skill_win="prev->A", skill_lose="prev->B")` and train the `_transition_scores` in `PreferenceScorer`.

**Files to create/modify:**

- [ ] `trainer/skillbank/grpo/stage2_grpo.py` — training loop: generate rankings, convert to preferences, build scorer, decode, compute reward
- [ ] Modify `skill_agents/infer_segmentation/llm_teacher.py` — add GRPO adapter variants of `collect_segment_preferences()`, `collect_transition_preferences()`, `collect_uncertain_preferences()`
- [ ] Modify `skill_agents/infer_segmentation/preference.py` — accept `source="grpo"` preferences; add batch ingestion from GRPO rollouts
- [ ] Modify `skill_agents/infer_segmentation/episode_adapter.py` — wire GRPO adapter into `infer_and_segment()` active learning loop
- [ ] Keep `skill_agents/infer_segmentation/dp_decoder.py` and `beam_decoder.py` unchanged (infrastructure)
- [ ] Keep `trainer/skillbank/stages/stage2_decode.py` as simplified fallback

---

### 1.4 Stage 3 — Contract Learning (GRPO)

**Current state:** `skill_agents/stage3_mvp/` implements a 7-step contract pipeline:

1. **Summarize** (`segment_summarize.py`): Smooth predicate windows with `start_end_window=5`, OR-aggregate for UI predicates, mean for vision/HUD. Produces `SegmentRecord` with `P_start`, `P_end` (probabilities) and `B_start`, `B_end` (booleanized sets).
2. **Effects compute** (`effects_compute.py`): `eff_add = B_end - B_start`, `eff_del = B_start - B_end`, `eff_event` from normalized UI events. Filtered by `reliability_min_for_effects=0.7`.
3. **Learn** (`contract_learn.py`): Frequency counting — `learn_effects_contract()` counts literal occurrences, thresholds at `eff_freq=0.8`.
4. **Verify** (`contract_verify.py`): `verify_effects_contract()` → `VerificationReport` with per-literal success rates, failure signatures, worst segments. Pass rule: instance passes if `(total_literals - n_fails) / total_literals >= instance_pass_literal_frac` (default 0.7).
5. **Refine** (`contract_refine.py`): Drop-only — removes literals with success rate below `eff_freq`. No strengthening.
6. **Re-verify**: Run `verify_effects_contract()` on refined contract.
7. **Persist**: `bank.add_or_update(refined_contract, report)`.

**Existing LLM usage:** `llm_contract.py` provides `llm_summarize_contract()` using the CONTRACT LoRA adapter, but this is **union-only enrichment** (adds to frequency-based consensus, doesn't replace it) and is **only used in the EM trainer path** (`trainer/skillbank/stages/stage3_contracts.py`), NOT in `run_stage3_mvp.py`.

**What changes:**

The CONTRACT LoRA adapter augments the frequency-based pipeline with contextual judgment. GRPO trains the adapter to generate better union-enrichment suggestions. The frequency-counting core (`learn_effects_contract()`) stays as the consensus baseline; the CONTRACT adapter adds/refines literals that frequency counting misses or wrongly includes.

| Component | Detail |
|-----------|--------|
| **Input (prompt)** | Skill ID, N representative `SegmentRecord` instances (booleanized `B_start`/`B_end`, computed `eff_add`/`eff_del`/`eff_event`, actions, events), frequency statistics across all instances, current bank contract (if exists), `VerificationReport` from initial verification (per-literal success rates, failure signatures). |
| **Output (generation)** | JSON: `{"eff_add": [...], "eff_del": [...], "eff_event": [...], "description": "..."}` — merged with frequency consensus via union |
| **Group generation** | `G=4` per skill (contracts are less variable than segmentations) |
| **Reward** | Holdout `VerificationReport` quality + decision agent follow score. See reward table. |

**Concrete I/O example:**

*Input prompt:*

```
Generate an effects contract for this skill based on its segment instances.
The frequency-based consensus and initial verification report are provided.

Skill: S04_pick_and_carry_onion
Current contract (if any): eff_add={holding_onion, at_pot}, eff_del={onion_on_counter, not_holding}

Frequency consensus (eff_freq=0.80, 23 instances):
  eff_add:  holding_onion (100%), near_pot (87%), facing_pot (83%), at_pot (70%)
  eff_del:  not_holding (100%), onion_on_counter (96%)
  events:   picked_up_onion (100%)
  → Consensus contract: eff_add={holding_onion, near_pot, facing_pot}, eff_del={not_holding, onion_on_counter}, eff_event={picked_up_onion}

Initial VerificationReport:
  overall_pass_rate: 0.78
  eff_add_success_rate: {holding_onion: 1.0, near_pot: 0.87, facing_pot: 0.83}
  eff_del_success_rate: {not_holding: 1.0, onion_on_counter: 0.96}
  failure_signatures: {"miss_add:near_pot": 3, "miss_add:facing_pot": 4}
  worst_segments: [seg_17, seg_22, seg_09]

Representative instances (5 of 23 total):

Instance 1 (seg_01):
  B_start: {at_counter, not_holding, onion_on_counter, far_from_pot}
  B_end:   {at_pot, holding_onion, facing_pot, near_pot}
  eff_add: {at_pot, holding_onion, facing_pot, near_pot}
  eff_del: {not_holding, onion_on_counter, far_from_pot}
  events:  [picked_up_onion]

Instance 2 (seg_05):
  B_start: {at_counter, not_holding, onion_on_counter}
  B_end:   {near_pot, holding_onion}
  eff_add: {near_pot, holding_onion}
  eff_del: {not_holding, onion_on_counter}
  events:  [picked_up_onion]

... (3 more instances)

Config: reliability_min_for_effects=0.7, instance_pass_literal_frac=0.7, max_effects_per_skill=50

Generate a JSON effects contract. Include predicates that are reliably caused
by this skill, not coincidental. The contract will be union-merged with the
frequency consensus.
```

*Output (generation):*

```json
{
  "eff_add": ["holding_onion", "near_pot", "facing_pot"],
  "eff_del": ["not_holding", "onion_on_counter"],
  "eff_event": ["picked_up_onion"],
  "description": "Pick up an onion from the counter and carry it toward the pot. Reliably results in holding the onion and being near/facing the pot. at_pot excluded (only 70% — some instances stop short)."
}
```

*Interpretation:* The model confirms `near_pot` (87%) and `facing_pot` (83%) as core despite being near the threshold. It excludes `at_pot` (70%) because the failure signatures show `miss_add:near_pot` and `miss_add:facing_pot` are the top failures, not `miss_add:at_pot` — the skill is about carrying toward the pot, not necessarily arriving. The union merge with frequency consensus produces the same contract since the LLM agrees with the frequency filter here. In cases of disagreement, the LLM can add literals frequency missed (below threshold but semantically important) or omit literals frequency included (above threshold but noisy).

**Reward function for Stage 3:**

| Component | Formula | Weight | Rationale |
|-----------|---------|--------|-----------|
| `r_holdout_pass` | `verify_effects_contract(contract, holdout_instances).overall_pass_rate` | 0.30 | Contract must generalize to unseen instances |
| `r_decision_follow` | `mean(r_follow)` from decision agent using the new contract on holdout episodes | 0.20 | Downstream: does the decision agent follow this contract better? |
| `r_per_literal_quality` | `mean(min(success_rate, 1.0) for lit in eff_add_success_rate ∪ eff_del_success_rate)` | 0.15 | Per-literal success rates from `VerificationReport` — every literal should be reliable |
| `r_sparsity` | `-max(0, n_literals - budget) / budget` | 0.10 | Prevent bloated contracts |
| `r_coverage` | `n_instances_covered / n_total_instances` | 0.10 | Contract should explain most instances |
| `r_failure_sig_diversity` | `-len(failure_signatures) / n_instances` | 0.10 | Fewer distinct failure modes = more consistent contract |
| `r_overfit_penalty` | `-(train_pass_rate - holdout_pass_rate)` if gap > 0.1 | 0.05 | Generalization check |

**Critical difference from current approach:**

Current: frequency counting is deterministic — same input always produces same contract. Refine is drop-only (removes weak literals, never adds).

GRPO: the LLM generates diverse contract suggestions, and the best ones (by holdout `VerificationReport` + decision-agent utility) are reinforced. Union merge means the LLM can only add value, never make things worse than frequency consensus alone. This allows the model to learn:

- When to include a predicate at 70% frequency (below threshold but semantically important)
- When to exclude a predicate at 90% frequency (noisy/irrelevant)
- Contextual effects (predicate X matters only when predicate Y is present)
- Using failure signatures to guide contract refinement

**Full pipeline with GRPO integration:**

```
summarize_segment()  →  compute_effects()  →  learn_effects_contract() (frequency)
                                                       │
                                              GRPO CONTRACT adapter (union enrichment)
                                                       │
                                              verify_effects_contract()
                                                       │
                                              refine_effects_contract() (drop weak)
                                                       │
                                              verify_effects_contract() (re-verify)
                                                       │
                                              bank.add_or_update()
```

**Files to create/modify:**

- [ ] `trainer/skillbank/grpo/stage3_grpo.py` — training loop: format evidence from `SegmentRecord`, generate contracts, compute holdout `VerificationReport` reward
- [ ] Modify `skill_agents/stage3_mvp/llm_contract.py` — use GRPO-trained CONTRACT adapter; add structured input with `VerificationReport` fields
- [ ] Modify `skill_agents/stage3_mvp/run_stage3_mvp.py` — integrate LLM enrichment into the 7-step pipeline (between learn and verify)
- [ ] Keep `skill_agents/stage3_mvp/contract_learn.py` frequency counting as consensus baseline (never replaced)
- [ ] Keep `skill_agents/stage3_mvp/contract_refine.py` drop-only refine as post-verification cleanup
- [ ] Keep `trainer/skillbank/stages/stage3_contracts.py` as simplified fallback

---

### 1.5 Shared GRPO Infrastructure

All three stages share:

- [ ] `trainer/skillbank/grpo/grpo_lora_updater.py` — GRPO advantage computation + LoRA parameter update on Qwen3-14B. Reuses advantage normalization logic from `trainer/decision/grpo_trainer.py` but operates on LoRA params only.
- [ ] `trainer/skillbank/grpo/config.py` — per-stage GRPO hyperparameters (G, clip_ratio, kl_coeff, lr). Lower G and higher kl_coeff than decision agent since skill bank outputs are more structured.
- [ ] `trainer/common/configs/skillbank_grpo.yaml` — unified config file.
- [ ] Update `skill_agents/lora/config.py` — change `base_model_name_or_path` default from `"Qwen/Qwen3-8B"` to `"Qwen/Qwen3-14B"`
- [ ] Update `trainer/common/configs/skillbank_em.yaml` — change `lora.base_model_name_or_path` from `"Qwen/Qwen3-8B"` to `"Qwen/Qwen3-14B"`

**Hyperparameter defaults:**

```yaml
stage1_boundary:
  group_size: 4
  clip_ratio: 0.2
  kl_coeff: 0.05      # higher KL — boundary outputs are structured, don't drift too far
  lr: 5.0e-5
  epochs_per_batch: 2

stage2_decode:
  group_size: 8
  clip_ratio: 0.2
  kl_coeff: 0.02
  lr: 3.0e-5
  epochs_per_batch: 3

stage3_contracts:
  group_size: 4
  clip_ratio: 0.2
  kl_coeff: 0.05
  lr: 5.0e-5
  epochs_per_batch: 2
```

### 1.6 Training Schedule (Co-evolution)

The EM loop in `em_trainer.py` currently runs: propose → decode → contract → update → gate.

With GRPO, the outer loop becomes:

```
for co_evolution_step in range(total_steps):
    # Phase 1: Collect decision agent rollouts (GRPO, existing)
    rollouts = collect_rollouts(decision_agent, env, skill_bank)
    decision_grpo_update(rollouts)

    # Phase 2: Skill bank GRPO update (every bank_update_cadence steps)
    if co_evolution_step % bank_update_cadence == 0:
        trajectories = ingest_rollouts(rollouts)

        # Stage 1 GRPO: train boundary proposer
        stage1_grpo_step(trajectories, bank)

        # Stage 2 GRPO: train segment decoder
        stage2_grpo_step(trajectories, bank)

        # Stage 3 GRPO: train contract learner
        stage3_grpo_step(trajectories, bank)

        # Stage 4: tool-calling agent curates the bank (see §2)
        stage4_agent_step(bank, trajectories)

        # SkillEval gating
        if not skilleval_passes(bank):
            rollback_bank()
```

---

## 2. Stage 4: LLM-Advised Bank Maintenance

### 2.1 Design Principle

**Production module:** `skill_agents/bank_maintenance/` — implements `run_bank_maintenance()` with a pipeline of: profile building → index construction → split → merge → refine → local re-decode.

Stage 4 mutates the skill bank through five actions — **refine, merge, split, materialize, promote**. Skills stored in the bank are living objects: their contracts (eff_add, eff_del, eff_event) evolve as new evidence arrives. Refine is the most frequent action and includes both **weakening** (dropping unreliable literals) and **strengthening** (adding discriminative literals vs confusion partners).

The architecture is **propose-filter-execute**:

- **Propose:** The algorithmic `run_bank_maintenance()` builds `SkillProfile` per skill (with effect signatures, embedding centroids/variance, transition top-k, duration stats, pass rates, failure signatures), constructs indices (`EffectInvertedIndex`, `MinHashLSH`, `EmbeddingANN`), and generates a ranked candidate action list.
- **Filter:** A single LLM call (CURATOR LoRA) reviews candidates against the full `SkillProfile` diagnostics and returns approved/vetoed/deferred decisions.
- **Execute:** The algorithm executes approved actions using the existing deterministic functions (`refine_skill`, `execute_merge`, `execute_split`, `form_from_cluster`, `promote_ready`).

The LLM's role is narrow: contextual judgment on "should we do this?" The algorithm handles "what could we do?" and "how do we do it?"

### 2.2 Why Not a Full Multi-Turn Agent

The original design considered a 20-turn tool-calling agent loop with sandboxed trials, but this has stability risks disproportionate to Stage 4's role:

- Stage 4 runs **once per EM batch** (~every 500 episodes). Multi-turn compounding (turn N's accept mutates the bank state turn N+1 reasons about) is fragile for sparse execution.
- Sandboxed deep-copies per trial are 10-20x the compute of the algorithmic version.
- Multi-turn = 20 failure points for malformed JSON, hallucinated skill_ids, etc.

The full agentic design is preserved as Phase 3 (§2.10) for when the simpler approach hits a ceiling.

### 2.3 The Five Actions

Five actions modify the skill bank. The first three operate on existing skills; the last two handle the lifecycle of new skills entering the bank.

| Action | What it does | Triggers (from `bank_maintenance/`) | What changes in the bank |
|--------|-------------|--------------------------------------|-------------------------|
| **Refine** | Weakens (drops unreliable literals) AND strengthens (adds discriminative literals vs confusion partners) a skill's contract | 3 triggers: `too_strong` (top_violating_literals non-empty), `too_strong_low_pass_rate` (pass_rate < 0.60), `too_weak_confusers` (confusion partners detected from Stage 2 diagnostics) | Contract updated: weak literals dropped (success_rate < `refine_drop_success_rate=0.60`), discriminative literals added (freq_self ≥ 0.90, max_confuser_freq ≤ 0.30). Version bumped, duration model updated. |
| **Merge** | Combines two skills with near-identical profiles into one | **3 metrics must ALL pass**: effect Jaccard ≥ 0.85, embedding cosine ≥ 0.90, transition overlap ≥ 0.50 (avg of prev/next top-5). Candidates retrieved via `EffectInvertedIndex` (min_shared=3) + `MinHashLSH` + `EmbeddingANN`. | One skill removed, survivor inherits combined instances, contract re-learned, alias map created for relabeling |
| **Split** | Breaks a weak skill into sub-skills by re-clustering its instances | **4 triggers**: (1) low pass_rate < 0.70, (2) failure concentration ≥ 0.60 in top-2 signatures, (3) high embedding variance > 2.0, (4) SSE drop from quick 2-means. Clustering: effect-signature first, then sparse-effects Jaccard seeding. Children must pass `child_pass_rate_thresh=0.80`. | Original replaced by ≤2 children, each with own contract. Local re-decode triggered on affected trajectories (`redecode_window_pad=300`). |
| **Materialize** | Graduates a `__NEW__` cluster into a proto-skill in the staging area | Cluster meets `min_cluster_size=5`, `min_consistency=0.5`, `min_distinctiveness=0.25` | Proto-skill created in `ProtoSkillManager`; participates in Stage 2 decoding as candidate label |
| **Promote** | Promotes a verified proto-skill to a full bank skill | Proto-skill passes verification with `promotion_min_pass_rate=0.6`, `promotion_min_support=5`, `promotion_min_consistency=0.5` | Proto-skill removed from staging; real skill added to bank with verified contract and LLM-generated name |

**Refine is critical for skill evolution.** A skill like `navigate_to_pot` might start with a noisy contract `eff_add={near_pot, near_counter}` from early instances. After 50 more instances, the evidence clearly shows `eff_add={near_pot, facing_pot}` is more accurate. Refine catches this drift with two mechanisms:

- **Weaken** (`weaken_contract`): Drop literals with success rate below `refine_drop_success_rate=0.60` using per-literal rates from `VerificationReport`.
- **Strengthen** (`strengthen_contract`): Add literals that discriminate this skill from its confusion partners. Discriminative score: `score(p) = freq_self(p) - max_confuser freq_confuser(p)`. Add if `freq_self ≥ 0.90` and `max_confuser ≤ 0.30`. Take top `refine_top_n_add=5` by score.
- **Confusion partners** (`extract_confusion_partners`): Extracted from Stage 2 diagnostics — for segments assigned to this skill, count which skills appear at ranks 2+ in `SegmentDiagnostic.candidates`. Top-3 confusers by count.

### 2.4 Proto-Skill Staging Pipeline

New skills do NOT enter the bank directly. The codebase (`skill_agents/skill_bank/new_pool.py`) implements a multi-gate staging area that the GRPO plan must respect.

**Lifecycle:**

```
__NEW__ segments (from Stage 2 decode)
        │
        ▼
  NewPoolManager.add()              ← accumulate with context (predecessor/successor skill)
        │
        ▼
  NewPoolManager.cluster()          ← group by effect similarity (Jaccard / agglomerative)
        │                              produces ClusterSummary per cluster
        ▼
  ┌─────────────────────────────────────┐
  │  Gate 1: min_cluster_size ≥ 5       │
  │  Gate 2: min_consistency ≥ 0.5      │  ← fraction sharing majority effect signature
  │  Gate 3: min_distinctiveness ≥ 0.25 │  ← Jaccard distance from ALL existing skills
  └─────────────┬───────────────────────┘
                │ passes
                ▼
  ProtoSkillManager.form_from_cluster() ← MATERIALIZE action lands here
        │
        ▼
  Proto-skill in staging area
  - Has candidate_effects_add/del/event (centroid of cluster)
  - Has support count, consistency score
  - Participates in Stage 2 decoding as candidate label
  - NOT yet in the real bank
        │
        ▼
  ProtoSkillManager.verify()        ← light Stage 3 verification
  - Runs run_stage3_mvp() on proto-skill's member segments
  - Records pass_rate, sets verified=True
  - Cleans up trial contract from bank
        │
        ▼
  ┌─────────────────────────────────────┐
  │  Gate 4: is_promotable              │
  │  - support ≥ promotion_min_support  │  ← default 5
  │  - consistency ≥ 0.5                │
  │  - verification_pass_rate ≥ 0.6     │
  └─────────────┬───────────────────────┘
                │ passes
                ▼
  ProtoSkillManager.promote_ready() ← PROMOTE action lands here
  - Calls proto.to_skill() → real Skill object
  - bank.add_or_update(skill.contract)
  - bank.add_or_update_skill(skill)
  - LLM suggests human-readable name (best-effort)
        │
        ▼
  Real skill in bank ✓
```

**Why this matters for Stage 4:**

1. **Materialize ≠ "add to bank."** The old GRPO plan treated materialize as a single step. In reality, it creates a proto-skill that must accumulate evidence and pass verification before promotion. The LLM filter should see both the NEW pool status AND the proto-skill status.

2. **Proto-skills participate in Stage 2.** `ProtoSkillManager.candidate_labels()` returns labels that Stage 2 decoding can assign to segments. This means materializing a proto-skill has an immediate effect on the next decode pass — segments that were `__NEW__` can now match the proto-skill, building its support count.

3. **Promotion is a separate decision from materialization.** A proto-skill might be materialized (created from a cluster) but not yet ready for promotion (insufficient support or low pass_rate). The LLM filter should be able to approve materialization while deferring promotion until the proto-skill matures.

4. **Rollback on promotion failure.** `NewPoolManager.promote()` (line 444 of `new_pool.py`) removes the skill from the bank and reverts segments to `__NEW__` if Stage 3 verification fails. The execution path must handle this gracefully.

**Config thresholds (from `NewPoolConfig` and `ProtoSkillConfig`):**

| Threshold | Default | Gate |
|-----------|---------|------|
| `min_cluster_size` | 5 | Materialization: cluster must have ≥ N segments |
| `min_consistency` | 0.5 | Materialization: majority effect pattern fraction |
| `min_distinctiveness` | 0.25 | Materialization: Jaccard distance from existing skills |
| `cluster_similarity_thresh` | 0.4 | Clustering: Jaccard threshold for merging NEW candidates |
| `promotion_min_support` | 5 | Promotion: minimum segment instances |
| `promotion_min_consistency` | 0.5 | Promotion: effect pattern consistency |
| `promotion_min_pass_rate` | 0.6 | Promotion: Stage 3 verification pass rate |
| `max_promotions_per_call` | 10 | Promotion: cap per EM iteration |

### 2.5 Model & Serving

**Qwen3-14B via vLLM** — same model as everything else. Uses the `/v1/chat/completions` endpoint, single turn, no tool-calling. Structured output via vLLM's JSON constrained decoding (`response_format={"type": "json_object"}`).

### 2.6 Candidate Generation (Algorithm)

The production `run_bank_maintenance()` pipeline is refactored to produce candidates without executing them. Key infrastructure from `bank_maintenance/`:

- **`SkillProfile`** per skill: `eff_add/del/event` (FrozenSet), `effect_sparse_vec` (predicate → normalized support), `embedding_centroid/var_diag`, `transition_topk_prev/next` (top-5), `duration_mean/var`, `overall_pass_rate`, `top_violating_literals`, `failure_signature_counts`, `n_instances`
- **`EffectInvertedIndex`**: predicate → skill_ids, `candidates_for(effects, min_shared=2)` for merge retrieval
- **`MinHashLSH`**: 128 permutations, banding threshold 0.50, `candidate_pairs()` for merge candidates
- **`EmbeddingANN`**: brute-force cosine search, `query(centroid, k=5)` for merge and split candidates

Candidate generation:

```python
@dataclass
class CandidateAction:
    action_type: str          # "refine" | "merge" | "split" | "materialize" | "promote"
    skill_ids: List[str]
    rationale: str            # human-readable reason this was proposed
    priority: float           # algorithm's confidence (used for default ordering)
    details: Dict[str, Any]   # action-specific data (SkillProfile fields, metrics, etc.)
    conflicts_with: List[int] = field(default_factory=list)
    llm_verdict: Optional[str] = None  # "approve" | "veto" | "defer" (filled by LLM)
    llm_reason: Optional[str] = None


def propose_candidates(
    profiles: Dict[str, SkillProfile],
    instances: Dict[str, List[SegmentRecord]],
    bank: SkillBankMVP,
    inv_index: EffectInvertedIndex,
    lsh: MinHashLSH,
    ann: Optional[EmbeddingANN],
    new_pool_mgr: NewPoolManager,
    proto_mgr: ProtoSkillManager,
    stage2_diagnostics: Optional[List[SegmentDiagnostic]],
    config: BankMaintenanceConfig,
) -> List[CandidateAction]:
    """Scan the bank using SkillProfiles + indices, return ranked candidates."""
    candidates = []

    # 1. Refine: 3 triggers from check_refine_triggers()
    for skill_id, profile in profiles.items():
        confusion_partners = extract_confusion_partners(
            skill_id, stage2_diagnostics, top_n=config.refine_top_confusers
        ) if stage2_diagnostics else []
        triggered, reason = check_refine_triggers(profile, confusion_partners, config)
        if triggered:
            candidates.append(CandidateAction(
                action_type="refine",
                skill_ids=[skill_id],
                rationale=f"Trigger: {reason}, pass_rate={profile.overall_pass_rate:.2f}, "
                          f"{profile.n_instances} instances, "
                          f"violating={profile.top_violating_literals[:3]}, "
                          f"confusers={confusion_partners[:2]}",
                priority=1.0 - profile.overall_pass_rate,
                details={
                    "trigger": reason,
                    "pass_rate": profile.overall_pass_rate,
                    "top_violating": profile.top_violating_literals,
                    "confusion_partners": confusion_partners,
                    "failure_signatures": dict(profile.failure_signature_counts),
                    "eff_add": sorted(profile.eff_add),
                    "eff_del": sorted(profile.eff_del),
                },
            ))

    # 2. Merge: 3 metrics from verify_merge_pair() — all must pass
    #    Candidates from EffectInvertedIndex (min_shared=3) + MinHashLSH + EmbeddingANN
    merge_pairs = retrieve_merge_candidates(profiles, inv_index, lsh, ann, config)
    for pair in merge_pairs:
        id_a, id_b = sorted(pair)
        passed, metrics = verify_merge_pair(profiles[id_a], profiles[id_b], config)
        if passed:
            candidates.append(CandidateAction(
                action_type="merge",
                skill_ids=[id_a, id_b],
                rationale=f"eff_jaccard={metrics['eff_jaccard']:.2f}, "
                          f"emb_cosine={metrics.get('emb_cosine', 'N/A')}, "
                          f"transition_overlap={metrics.get('transition_overlap', 'N/A')}, "
                          f"combined {profiles[id_a].n_instances + profiles[id_b].n_instances} instances",
                priority=metrics['eff_jaccard'],
                details=metrics,
            ))

    # 3. Split: 4 triggers from check_split_triggers()
    for skill_id, profile in profiles.items():
        skill_instances = instances.get(skill_id, [])
        embeddings = None  # from profile.embedding_centroid if available
        triggered, reason = check_split_triggers(profile, skill_instances, config, embeddings)
        if triggered:
            candidates.append(CandidateAction(
                action_type="split",
                skill_ids=[skill_id],
                rationale=f"Trigger: {reason}, pass_rate={profile.overall_pass_rate:.2f}, "
                          f"{profile.n_instances} instances, "
                          f"top_failures={list(profile.failure_signature_counts.keys())[:2]}",
                priority=1.0 - profile.overall_pass_rate,
                details={
                    "trigger": reason,
                    "pass_rate": profile.overall_pass_rate,
                    "n_instances": profile.n_instances,
                    "failure_signatures": dict(profile.failure_signature_counts),
                    "embedding_var": sum(profile.embedding_var_diag) if profile.embedding_var_diag else None,
                },
            ))

    # 4. Materialize: NEW clusters from NewPoolManager
    for summary in new_pool_mgr.get_candidates():
        if not _is_distinctive(summary, bank, config):
            continue
        candidates.append(CandidateAction(
            action_type="materialize",
            skill_ids=[f"cluster_{summary.cluster_id}"],
            rationale=f"{summary.size} segments, consistency {summary.consistency:.2f}, "
                      f"sig={summary.representative_sig}",
            priority=(summary.consistency * summary.size) / 20.0,
            details={
                "n_segments": summary.size,
                "consistency": summary.consistency,
                "mean_duration": summary.mean_duration,
                "effect_centroid_add": sorted(summary.effect_centroid_add),
                "effect_centroid_del": sorted(summary.effect_centroid_del),
            },
        ))

    # 5. Promote: proto-skills from ProtoSkillManager
    for pid in proto_mgr.proto_ids:
        proto = proto_mgr.get(pid)
        if proto is None or not proto.verified:
            continue
        candidates.append(CandidateAction(
            action_type="promote",
            skill_ids=[pid],
            rationale=f"Proto-skill: support={proto.support}, "
                      f"consistency={proto.consistency:.2f}, "
                      f"pass_rate={proto.verification_pass_rate:.2f}, "
                      f"promotable={proto.is_promotable}",
            priority=proto.verification_pass_rate * proto.consistency,
            details={
                "support": proto.support,
                "consistency": proto.consistency,
                "pass_rate": proto.verification_pass_rate,
                "is_promotable": proto.is_promotable,
                "effects_add": sorted(proto.candidate_effects_add),
                "effects_del": sorted(proto.candidate_effects_del),
            },
        ))

    _annotate_conflicts(candidates)

    return sorted(candidates, key=lambda c: c.priority, reverse=True)
```

### 2.7 Conflict Detection

The algorithm flags conflicts before sending to the LLM:

| Conflict | Why it's a problem | Resolution hint |
|----------|-------------------|-----------------|
| refine(X) + merge(X, Y) | Refining X's contract then merging overwrites the refinement | Pick one: refine if X is worth keeping solo, merge if X and Y are truly duplicates |
| split(X) + merge(X, Y) | Contradictory: splitting X while also merging it | Usually split takes priority (low pass_rate triggered both) |
| materialize(cluster) + promote(proto) with overlapping effects | New proto-skill and existing proto-skill cover the same behavior | Promote the more mature proto-skill; defer materialize |
| promote(proto) + merge(proto_sig, existing) | Promoting a proto-skill whose effects overlap an existing skill | Merge the proto-skill's instances into the existing skill instead of promoting |

Conflicts are annotated as `conflicts_with: [idx]` on each candidate. The prompt explicitly asks the LLM to resolve them.

### 2.8 Deferral Lifecycle

When the LLM defers an action, here's what happens:

**1. Deferred actions are logged, not queued.** There is no separate deferral queue. The algorithm re-scans the bank from scratch every EM iteration based on current state. If the same conditions still hold (same skill still has low pass_rate, same evidence delta), the candidate naturally reappears in the next `propose_candidates()` call.

**2. Deferral history is annotated on re-proposed candidates.** The logged deferral is matched to the re-proposed candidate by `(action_type, skill_ids)`. The prompt includes context so the LLM can see what changed:

```
[4] SPLIT S23_place_onion (priority=0.38)
    pass_rate 0.62, 18 instances, failure sigs suggest 2 sub-clusters
    ⏳ DEFERRED 1x (last iteration: 12 instances, reason: "borderline, want more evidence")
    Δ since deferral: +6 instances, pass_rate 0.62→0.62 (unchanged)
```

The LLM now sees: 6 more instances arrived but pass_rate didn't improve — stronger signal to approve the split this time.

**3. Deferral expiry: auto-approve after N consecutive deferrals.** If the same candidate is deferred `max_deferrals` times (default: 3) and still meets the algorithm's thresholds, it auto-approves on the next iteration without LLM review. This prevents the LLM from indefinitely blocking actions the algorithm keeps proposing.

```python
@dataclass
class DeferralRecord:
    action_type: str
    skill_ids: List[str]
    reason: str
    em_iteration: int
    n_instances_at_deferral: int
    pass_rate_at_deferral: float
    consecutive_deferrals: int = 1

def annotate_deferrals(
    candidates: List[CandidateAction],
    deferral_log: List[DeferralRecord],
    max_deferrals: int = 3,
) -> List[CandidateAction]:
    approved_by_expiry = []
    for candidate in candidates:
        key = (candidate.action_type, tuple(sorted(candidate.skill_ids)))
        prev = _find_deferral(deferral_log, key)
        if prev is None:
            continue
        if prev.consecutive_deferrals >= max_deferrals:
            candidate.llm_verdict = "approve"
            candidate.llm_reason = f"Auto-approved: deferred {prev.consecutive_deferrals}x, algorithm threshold still met"
            approved_by_expiry.append(candidate)
        else:
            candidate.deferral_history = prev  # attached for prompt formatting
    return approved_by_expiry
```

**4. Possible outcomes for a deferred action next iteration:**

| What changed since deferral | Likely next verdict | Why |
|----------------------------|--------------------|----|
| More instances, pass_rate still low | **approve** (split/refine) | More evidence confirms the problem |
| More instances, pass_rate improved | **action disappears** | Algorithm no longer proposes it (above threshold) |
| No new instances | **defer again** | Still not enough evidence — counter increments |
| 3rd consecutive deferral, thresholds still met | **auto-approve** | Expiry kicks in — trust the algorithm |
| Conflicting action was approved last time | **action disappears** | Skill was merged/refined, split no longer applies |

### 2.9 LLM Filter (Single Call)

One LLM call. No tool-calling. Constrained JSON output.

**Concrete I/O example:**

*System prompt:*

```
You are the skill bank curator. The algorithm has proposed bank mutations.
Your job: approve good actions, veto harmful ones, defer uncertain ones.

Skills in the bank have contracts (eff_add, eff_del) that describe what
changes when the skill executes. These contracts evolve — refine updates
them with better evidence. Merge/split change the bank structure.

New skills enter through a staging pipeline:
- MATERIALIZE: creates a proto-skill from a __NEW__ cluster (staging area)
- PROMOTE: graduates a verified proto-skill to a real bank skill
Proto-skills participate in decoding but are NOT full bank skills yet.

Rules:
- approve: execute this action
- veto: skip (you MUST explain why — e.g. "semantically different despite high Jaccard")
- defer: skip for now, revisit next iteration (e.g. "not enough instances yet")
- If two actions conflict, approve at most one and explain
- When in doubt, approve — the algorithm's thresholds are conservative
- For PROMOTE: only approve if pass_rate and consistency are strong

Respond with JSON matching this schema exactly.
```

*Input (user message):*

```
## Bank Summary
42 skills, 456 segments, mean_pass_rate=0.74, new_rate=0.08

## NEW Pool & Proto-Skills
NEW pool: 34 segments across 6 clusters
Proto-skills in staging: 2
  proto_1710000_3: support=7, consistency=0.71, pass_rate=0.68, promotable=YES
    effects_add={holding_dish, near_serve}, effects_del={at_counter}
  proto_1710000_5: support=4, consistency=0.50, pass_rate=0.42, promotable=NO
    effects_add={at_pot}, effects_del={far_from_pot}

## Proposed Actions

[1] REFINE S12_navigate_pot (priority=0.33)
    Trigger: too_strong (top_violating_literals non-empty)
    SkillProfile: pass_rate=0.55, n_instances=8, duration_mean=14.2
      top_violating_literals: [near_counter (rate=0.38), facing_pot (rate=0.62)]
      failure_signatures: {"miss_add:near_counter": 5, "miss_add:facing_pot": 3}
      confusion_partners: [S05_go_to_pot, S11_navigate_to_counter]
    Current contract: eff_add={near_pot, near_counter}, eff_del={}
    Proposed weaken: drop near_counter (rate=0.38 < 0.60 threshold)
    Proposed strengthen: add far_from_pot (freq_self=0.92, max_confuser=0.15) — discriminates from S05

[2] MERGE S05_go_to_pot + S12_navigate_pot (priority=0.87)
    Metrics: eff_jaccard=0.87, emb_cosine=0.92, transition_overlap=0.60
    Combined instances: 15
    S05 profile: pass_rate=0.78, duration_mean=11.5, top_transitions_next=[S06_plate, S04_carry]
    S12 profile: pass_rate=0.55, duration_mean=14.2, top_transitions_next=[S06_plate, S08_fetch]
    ⚠ CONFLICTS WITH [1] — both touch S12_navigate_pot

[3] MATERIALIZE cluster_7 → proto-skill (priority=0.34)
    8 segments, consistency 0.85, sig=A:near_pot,facing_pot|D:
    Distinctiveness from existing skills: 0.42 (passes min 0.25)

[4] SPLIT S23_place_onion (priority=0.38)
    Trigger: low_pass_rate + failure_concentration
    SkillProfile: pass_rate=0.62, n_instances=12, duration_mean=18.7
      failure_signatures: {"miss_add:onion_in_pot": 3, "miss_del:holding_onion|miss_add:at_pot": 2}
      failure_concentration=0.65 (top-2 sigs / total, threshold=0.60)
      embedding_var=1.8 (threshold=2.0, not triggered)

[5] REFINE S07_pick_onion (priority=0.50)
    Trigger: too_weak_confusers (confusion_partners detected)
    SkillProfile: pass_rate=0.71, n_instances=14
      confusion_partners: [S04_carry_onion]
      Current eff_del={}, but onion_on_counter appears in 96% of instances
    Proposed strengthen: add eff_del={onion_on_counter} (freq_self=0.96, max_confuser=0.10)

[6] PROMOTE proto_1710000_3 → real skill (priority=0.48)
    support=7, consistency=0.71, pass_rate=0.68, promotable=YES
    effects_add={holding_dish, near_serve}, effects_del={at_counter}
```

*Output (generation):*

```json
{
  "decisions": [
    {"idx": 1, "verdict": "approve", "reason": "High evidence delta, contract clearly stale"},
    {"idx": 2, "verdict": "veto", "reason": "S05 is go-to-pot-from-counter, S12 is navigate-to-pot-from-anywhere — semantically different despite Jaccard. Conflicts with approved refine on S12."},
    {"idx": 3, "verdict": "approve", "reason": "Strong cluster with 0.85 consistency and good distinctiveness — ready for proto-skill staging"},
    {"idx": 4, "verdict": "defer", "reason": "12 instances borderline, want more evidence before splitting"},
    {"idx": 5, "verdict": "approve", "reason": "Clear missing eff_del, 50% delta with 14 instances is strong signal"},
    {"idx": 6, "verdict": "approve", "reason": "Proto-skill has 7 instances, 0.71 consistency, 0.68 pass_rate — meets all promotion gates. Effects look like a carry-dish-to-serve skill."}
  ]
}
```

*Interpretation:* The LLM approves two refines, one materialize (cluster → proto-skill), and one promote (proto-skill → real skill). It vetoes a merge that the algorithm flagged on Jaccard alone (semantic difference), and defers a split for insufficient evidence. Actions [1] and [2] conflict — resolved by approving [1] and vetoing [2]. After execution:
- S12's contract gets tightened (refine)
- S07 gains a missing eff_del (refine)
- cluster_7 becomes a proto-skill in staging (materialize) — will participate in Stage 2 decoding next iteration
- proto_1710000_3 graduates to a real bank skill with LLM-generated name (promote)
- S23 is untouched — re-evaluated next iteration with more instances
- proto_1710000_5 stays in staging (not proposed — pass_rate too low, `promotable=NO`)

**Output schema (for constrained decoding):**

```json
{
  "type": "object",
  "required": ["decisions"],
  "properties": {
    "decisions": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["idx", "verdict"],
        "properties": {
          "idx":     {"type": "integer"},
          "verdict": {"enum": ["approve", "veto", "defer"]},
          "reason":  {"type": "string"}
        }
      }
    }
  }
}
```

With vLLM's JSON constrained decoding + this schema, the LLM **cannot** produce invalid JSON. The only structural failure mode is referencing an out-of-range `idx`, which is caught by validation.

### 2.10 Parsing, Validation & Fallback

```python
def filter_candidates(
    candidates: List[CandidateAction],
    bank: SkillBankMVP,
    vllm_client,
    config: Stage4FilterConfig,
) -> List[CandidateAction]:
    prompt = format_filter_prompt(candidates, bank)

    try:
        response = vllm_client.chat.completions.create(
            model="Qwen/Qwen3-14B",
            messages=[
                {"role": "system", "content": CURATOR_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            response_format={"type": "json_object"},
            extra_body={"guided_json": FILTER_SCHEMA},
            timeout=30,
        )
        decisions = json.loads(response.choices[0].message.content)

        # Validate: idx in range, verdict in {approve, veto, defer}
        approved = _apply_decisions(candidates, decisions["decisions"])

    except (json.JSONDecodeError, KeyError, ValidationError) as e:
        logger.warning("LLM filter parse error (%s), retrying once", e)
        # Single retry with corrective prompt
        try:
            response = vllm_client.chat.completions.create(
                model="Qwen/Qwen3-14B",
                messages=[
                    {"role": "system", "content": CURATOR_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": response.choices[0].message.content},
                    {"role": "user", "content": f"Invalid response. Valid idx range: 1-{len(candidates)}. "
                                                 "Verdict must be approve/veto/defer. Try again."},
                ],
                temperature=0.0,
                response_format={"type": "json_object"},
                extra_body={"guided_json": FILTER_SCHEMA},
                timeout=30,
            )
            decisions = json.loads(response.choices[0].message.content)
            approved = _apply_decisions(candidates, decisions["decisions"])
        except Exception:
            logger.warning("LLM filter retry failed, executing full candidate list")
            approved = candidates

    except Exception:
        logger.warning("LLM filter unavailable, executing full candidate list")
        approved = candidates

    return approved
```

**Fallback hierarchy:**
1. Constrained decoding → structurally valid JSON guaranteed
2. Semantic validation → catch out-of-range idx, retry once
3. Retry fails → execute full candidate list (algorithmic default)

With constrained decoding, step 3 should only trigger when vLLM is down. The system is **never worse** than the current algorithmic-only version.

### 2.11 Execution

After filtering, approved actions execute using the production `bank_maintenance/` functions. Key additions vs old plan: **refine includes strengthening**, **split triggers local re-decode**, **merge creates alias maps**, and **materialize/promote** go through proper staging.

```python
def execute_approved(
    approved: List[CandidateAction],
    instances: Dict[str, List[SegmentRecord]],
    bank: SkillBankMVP,
    new_pool_mgr: NewPoolManager,
    proto_mgr: ProtoSkillManager,
    duration_store: DurationModelStore,
    observations_by_traj: Dict[str, list],
    stage2_diagnostics: Optional[List[SegmentDiagnostic]],
    decode_fn: Optional[Callable],
    config: BankMaintenanceConfig,
) -> BankMaintenanceResult:
    result = BankMaintenanceResult()
    redecode_requests = []
    alias_map = {}

    for action in approved:
        if action.action_type == "refine":
            skill_id = action.skill_ids[0]
            contract = bank.get_contract(skill_id)
            report = bank.get_report(skill_id)
            self_instances = instances.get(skill_id, [])
            confuser_ids = action.details.get("confusion_partners", [])
            confuser_instances = [inst for cid in confuser_ids
                                 for inst in instances.get(cid, [])]
            # Weaken (drop unreliable) + strengthen (add discriminative)
            refine_result = refine_skill(contract, report, self_instances,
                                         confuser_instances, config)
            bank.add_or_update(refine_result.new_contract, report)
            update_duration_model(duration_store, skill_id, self_instances)
            result.diff.add(BankDiffEntry("REFINE", skill_id,
                            {"dropped": refine_result.dropped_literals,
                             "added": refine_result.added_literals}))

        elif action.action_type == "merge":
            k1, k2 = action.skill_ids
            merge_result = execute_merge(k1, k2, instances.get(k1, []),
                                         instances.get(k2, []), config)
            if merge_result.accepted:
                bank.add_or_update(merge_result.contract, merge_result.report)
                bank.remove(merge_result.merged_ids[0])
                alias_map.update(merge_result.alias_map)
                # Queue re-decode for affected trajectories
                reqs = redecode_requests_for_merge(k1, k2, instances.get(k1, []),
                                                    instances.get(k2, []), config,
                                                    merge_result.canonical_id)
                redecode_requests.extend(reqs)
                result.diff.add(BankDiffEntry("MERGE", merge_result.canonical_id,
                                {"merged": merge_result.merged_ids}))

        elif action.action_type == "split":
            skill_id = action.skill_ids[0]
            split_result = execute_split(skill_id, instances.get(skill_id, []),
                                          config, bank.get_version(skill_id))
            if split_result.accepted:
                bank.remove(skill_id)
                for child in split_result.children:
                    bank.add_or_update(child.contract, child.report)
                # Queue local re-decode on affected trajectory windows
                reqs = redecode_requests_for_split(skill_id,
                        instances.get(skill_id, []), config, split_result.children)
                redecode_requests.extend(reqs)
                result.diff.add(BankDiffEntry("SPLIT", skill_id,
                                {"children": [c.skill_id for c in split_result.children]}))

        elif action.action_type == "materialize":
            cluster_id = int(action.skill_ids[0].replace("cluster_", ""))
            summary = _find_cluster_summary(new_pool_mgr, cluster_id)
            if summary is not None:
                records = new_pool_mgr.get_cluster_records(cluster_id)
                proto = proto_mgr.form_from_cluster(summary, records,
                            existing_bank_skills=set(bank.skill_ids))
                if proto is not None:
                    result.diff.add(BankDiffEntry("ADD", proto.proto_id,
                                    {"cluster_id": cluster_id, "support": summary.size}))

        elif action.action_type == "promote":
            proto_id = action.skill_ids[0]
            pass_rate = proto_mgr.verify(proto_id, bank, observations_by_traj)
            if pass_rate is not None and pass_rate >= config.promotion_min_pass_rate:
                promoted = proto_mgr.promote_ready(bank)
                if proto_id in promoted:
                    result.diff.add(BankDiffEntry("ADD", proto_id,
                                    {"pass_rate": pass_rate, "promoted": True}))

    # Apply alias map for merged skills (relabel segments)
    if alias_map:
        relabel_via_alias(all_segments, alias_map)

    # Execute local re-decode for splits and merges
    if redecode_requests and decode_fn:
        redecode_windows(redecode_requests, decode_fn)

    return result
```

**Key differences from simplified `stage4_update.py`:**

| Action | Simplified fallback | Production (`bank_maintenance/`) |
|--------|-------------------|--------------------------------|
| Refine | Drop-only (weaken) | Weaken + strengthen (discriminative literals vs confusion partners) |
| Merge | Single metric (Jaccard only) | 3 metrics must all pass (Jaccard + cosine + transition overlap) |
| Split | Single trigger (pass_rate) | 4 triggers + 2 clustering strategies + child pass_rate gate |
| Split post | No re-decode | Local re-decode on affected trajectory windows (`redecode_window_pad=300`) |
| Merge post | No relabeling | Alias map + relabeling of all segments referencing merged skill |
| Duration | Not updated | `DurationModelStore.update()` on refine |
| Materialize | Direct bank add | `ProtoSkillManager.form_from_cluster()` → staging area |
| Promote | Did not exist | `ProtoSkillManager.verify()` + `promote_ready()` → full verification |

SkillEval gate checks after all actions. If bank quality drops, rollback the whole batch. Proto-skills that were materialized but not yet promoted survive rollback (they're in the staging area, not the bank).

### 2.12 Cold-Start Data Collection (Phase 0)

Before the LLM filter can be trained, we need labeled data on which actions help and which hurt. Phase 0 runs alongside the existing EM loop with **no LLM involvement**:

```python
def collect_counterfactuals(
    candidates: List[CandidateAction],
    decode_results: List[DecodeResult],
    contracts: Dict[str, LearnedContract],
    bank: SkillBankMVP,
    config: UpdateConfig,
) -> List[CounterfactualRecord]:
    """For each candidate, measure its marginal contribution."""
    # Baseline: execute ALL candidates
    bank_all = deepcopy(bank)
    execute_approved(candidates, decode_results, contracts, bank_all, config)
    q_all = compute_bank_quality(bank_all)

    records = []
    for i, candidate in enumerate(candidates):
        # Execute all EXCEPT candidate i
        bank_without = deepcopy(bank)
        subset = [c for j, c in enumerate(candidates) if j != i]
        execute_approved(subset, decode_results, contracts, bank_without, config)
        q_without = compute_bank_quality(bank_without)

        records.append(CounterfactualRecord(
            candidate=candidate,
            quality_with=q_all,
            quality_without=q_without,
            marginal_delta=q_all - q_without,  # positive = this action helped
        ))

    return records
```

For a typical batch of 5-10 candidates, this is 5-10 extra runs of fast deterministic functions. Cheap.

After ~100 EM iterations, this produces labeled data:

| Label | Meaning | Training signal |
|-------|---------|-----------------|
| `marginal_delta > 0` | Action helped — should be approved | Positive example |
| `marginal_delta < 0` | Action hurt — should be vetoed | Negative example |
| `marginal_delta ≈ 0` | Action was neutral — defer is fine | Weak/ambiguous example |

This data also reveals **which action types** benefit most from LLM filtering (hypothesis: merge and split have the most context-dependent outcomes, while refine is usually safe to auto-approve).

### 2.13 GRPO Training (Phase 2)

After Phase 0 data collection and Phase 1 SFT warm-start, the filter is trained with GRPO. This is a clean single-turn GRPO setup:

**Why GRPO fits here:**
- Single turn — no compounding errors across turns
- Structured output — constrained JSON, trivially parseable
- Verifiable reward — bank quality delta is deterministic
- Counterfactual baseline — we know what "execute all" yields

**GRPO setup:**

```python
def stage4_grpo_step(
    candidates: List[CandidateAction],
    decode_results, contracts, bank, config,
    vllm_client, curator_adapter,
):
    prompt = format_filter_prompt(candidates, bank)

    # Baseline: quality if we execute all (no filtering)
    bank_all = deepcopy(bank)
    execute_approved(candidates, decode_results, contracts, bank_all, config)
    q_baseline = compute_bank_quality(bank_all)

    # Generate G=8 diverse filter decisions
    group_rewards = []
    group_outputs = []
    for g in range(G):
        response = vllm_client.chat.completions.create(
            model="Qwen/Qwen3-14B",
            messages=[{"role": "system", "content": CURATOR_SYSTEM_PROMPT},
                      {"role": "user", "content": prompt}],
            temperature=0.7,  # higher temp for diversity during GRPO
            response_format={"type": "json_object"},
            extra_body={"guided_json": FILTER_SCHEMA},
        )
        decisions = json.loads(response.choices[0].message.content)
        approved = _apply_decisions(candidates, decisions["decisions"])

        # Execute this filtering and measure quality
        bank_filtered = deepcopy(bank)
        execute_approved(approved, decode_results, contracts, bank_filtered, config)
        q_filtered = compute_bank_quality(bank_filtered)

        # Reward: did filtering beat executing everything?
        r = q_filtered - q_baseline
        group_rewards.append(r)
        group_outputs.append(response.choices[0].message.content)

    # GRPO advantage: group-normalized
    advantages = compute_grpo_advantages(group_rewards)
    update_lora(curator_adapter, [prompt] * G, group_outputs, advantages)
```

**Reward signal:** `r = quality_after_filtered - quality_after_all`

| r value | Meaning | What GRPO does |
|---------|---------|----------------|
| r > 0 | Filtering improved over executing everything | Reinforce these veto decisions |
| r = 0 | Filtering matched executing everything | Neutral — no gradient |
| r < 0 | Filtering was worse (vetoed good actions) | Discourage these veto decisions |

This directly trains the CURATOR adapter to only veto when it genuinely helps. The baseline "execute all" is always available, so the LLM is incentivized to add value over the algorithm, not just not break things.

**Hyperparameters:**

```yaml
stage4_curator:
  group_size: 8
  clip_ratio: 0.2
  kl_coeff: 0.03
  lr: 3.0e-5
  epochs_per_batch: 2
  temperature_train: 0.7    # diversity during GRPO sampling
  temperature_inference: 0.1 # conservative during real execution
```

### 2.14 Training Schedule

| Phase | EM Iterations | What happens | LLM role |
|-------|--------------|--------------|----------|
| **Phase 0** | 0–100 | Algorithmic only. Log all candidates + counterfactual marginal deltas. | None |
| **Phase 1** | 100–200 | SFT warm-start: train CURATOR LoRA on Phase 0 data (positive marginal → approve, negative → veto). LLM filter enabled with fallback. | Advisor (SFT) |
| **Phase 2** | 200+ | GRPO online: G=8 filter decisions per iteration, reward = `q_filtered - q_baseline`. | Advisor (GRPO) |

Phase 0 costs nothing beyond logging. Phase 1 is standard SFT. Phase 2 adds G=8 executions of `execute_approved()` per EM step — but these are fast deterministic functions, not LLM calls.

### 2.15 Future: Full Agentic Mode (Phase 3)

Once the single-turn advisor is stable and we have data on where the algorithm's proposals miss, consider upgrading to a multi-turn tool-calling agent for cases where:

- The algorithm's fixed thresholds miss semantic relationships (Jaccard < 0.85 but skills are truly identical)
- Iterative exploration adds value (trial a merge, observe delta, then decide to also refine the survivor)
- The LLM proposes actions the algorithm can't (e.g., "merge aspects of S03 and S17 into a new composite skill")

The tool definitions (inspect_skill, trial_merge, trial_split, trial_refine, trial_materialize, accept/reject, finish) remain the target API for Phase 3. Phase 1-2 ships without them.

### 2.16 Files to Create/Modify

- [ ] `trainer/skillbank/stage4_candidates.py` — `propose_candidates()` using `SkillProfile`, `EffectInvertedIndex`, `MinHashLSH`, `EmbeddingANN` from `bank_maintenance/`. `CandidateAction` dataclass, conflict detection, `_annotate_conflicts()`, deferral annotation. Calls production trigger functions: `check_refine_triggers()`, `check_split_triggers()`, `retrieve_merge_candidates()`, `verify_merge_pair()`, `extract_confusion_partners()`.
- [ ] `trainer/skillbank/stage4_deferrals.py` — `DeferralRecord`, `annotate_deferrals()`, deferral log persistence, expiry logic (`max_deferrals=3`)
- [ ] `trainer/skillbank/stage4_filter.py` — LLM filter call, constrained decoding, retry logic, `_apply_decisions()`, fallback
- [ ] `trainer/skillbank/stage4_prompts.py` — curator system prompt, diagnostics formatter with `SkillProfile`-level data (violation literals, failure signatures, confusion partners, embedding variance, transition top-k), NEW pool status, proto-skill staging summary, output schema
- [ ] `trainer/skillbank/stage4_counterfactual.py` — `collect_counterfactuals()`, `CounterfactualRecord`, Phase 0 logging
- [ ] `trainer/skillbank/stage4_grpo.py` — GRPO training loop for CURATOR adapter (§2.13)
- [ ] Modify `skill_agents/bank_maintenance/run_bank_maintenance.py` — refactor into `build_profiles()` + `propose_candidates()` (without executing), expose for use by `stage4_candidates.py`. Add `execute_approved()` entry point that calls `refine_skill()`, `execute_merge()`, `execute_split()` from the production modules + `redecode_windows()` for splits/merges.
- [ ] Modify `skill_agents/bank_maintenance/refine.py` — expose `extract_confusion_partners()` for use in candidate formatting
- [ ] Modify `skill_agents/bank_maintenance/local_redecode.py` — expose `redecode_requests_for_split()` and `redecode_requests_for_merge()` for use after execution
- [ ] Modify `trainer/skillbank/em_trainer.py` — call `propose_candidates()` → `filter_candidates()` → `execute_approved()` instead of `run_update()`. Pass `NewPoolManager`, `ProtoSkillManager`, `DurationModelStore`, and Stage 2 `SegmentDiagnostic` instances through.
- [ ] Modify `skill_agents/skill_bank/new_pool.py` — expose `get_candidates()` summary data in a format `stage4_candidates.py` can consume for materialize proposals
- [ ] Modify `trainer/common/configs/skillbank_em.yaml` — add `stage4_filter` config section (enabled, vllm_url, temperature, fallback_on_failure), `stage4_curator` GRPO config, and `proto_skill` promotion thresholds
- [ ] Modify `trainer/skillbank/grpo/config.py` — add CURATOR adapter config alongside BOUNDARY/SEGMENT/CONTRACT
- [ ] Keep `trainer/skillbank/stages/stage4_update.py: run_update()` as simplified algorithmic-only fallback path

### 2.17 Guardrails

| Guardrail | Implementation |
|-----------|---------------|
| **Constrained decoding** | vLLM `guided_json` with strict schema — LLM cannot produce invalid JSON structure |
| **Semantic validation** | Check idx range, verdict values; invalid entries default to "approve" (algorithm's recommendation stands) |
| **Single retry** | One corrective retry on validation failure; then fallback |
| **Fallback to algorithm** | If vLLM is down or both attempts fail, execute full candidate list — never worse than algorithmic-only |
| **Quality gate** | SkillEval runs after execution (6 dimensions: coherence, discriminability, composability, generalization, utility, granularity + holistic pass with recommendations KEEP/REFINE/SPLIT/MERGE/DISCARD) — rollback if bank quality drops |
| **Conflict enforcement** | If LLM approves both sides of a conflict, only the higher-priority one executes |
| **Deferral expiry** | After `max_deferrals=3` consecutive deferrals with algorithm threshold still met, auto-approve without LLM review |
| **Split child quality** | Split children must pass `child_pass_rate_thresh=0.80`; if either child fails, split is rejected |
| **Local re-decode after split/merge** | `redecode_windows()` re-runs Stage 2 on affected trajectory windows (`redecode_window_pad=300`, `redecode_min_window=50`) to fix stale labels |
| **Alias map relabeling** | After merge, `relabel_via_alias()` updates all segments referencing the retired skill to point to the canonical survivor |
| **Duration model update** | After refine, `DurationModelStore.update()` refreshes per-skill duration histograms |
| **Promote rollback** | If `ProtoSkillManager.verify()` returns pass_rate below threshold during promote execution, proto-skill stays in staging — no bank mutation |
| **Materialize idempotency** | Materializing the same cluster twice is a no-op — `ProtoSkillManager.form_from_cluster()` deduplicates by effect signature |
| **Stage 4.5 quality check** | `run_sub_episode_quality_check()` runs between Stage 3 and Stage 4: scores sub-episodes on outcome_reward, follow_through, consistency, compactness; retires depleted skills. Ensures Stage 4 operates on clean data. |
| **Logging** | Every (prompt, response, executed_actions, deferrals, proto_skill_changes, q_before, q_after, counterfactuals, `BankDiffReport`) logged for training |

---

## 3. `select_skill` in the Decision Agent (GRPO)

### 3.1 Current State

The decision agent's `select_skill` is a deterministic pipeline:

1. `VLMDecisionAgent.step()` generates `TOOL: select_skill, ARGS: {"key": "..."}` via LLM
2. `run_tool(TOOL_SELECT_SKILL, ...)` calls `select_skill_from_bank()`
3. `select_skill_from_bank()` tries multiple fallback paths:
   - `SkillQueryEngine.select()` → relevance (embedding + keyword Jaccard) + applicability (contract match) + confidence blend → `SkillSelectionResult`
   - `query_for_decision_agent()` → same but returns single best
   - `SkillBankAgent.select_skill()` → delegated
   - Fallback: TF-IDF keyword scoring
4. Result includes: `skill_id`, `protocol` (steps/preconditions/success_criteria/abort_criteria), `execution_hint`, `termination_hint`, `failure_modes`, `micro_plan`
5. Agent follows the protocol steps via `take_action`

**What's NOT learned:**
- The query key generation (`ARGS: {"key": "..."}`) is already part of the LLM's policy — GRPO trains this.
- But `SkillQueryEngine.select()` itself is a fixed scoring function: `confidence = 0.4 * relevance + 0.35 * applicability + 0.25 * pass_rate`. No learning.
- The decision of _when_ to call `select_skill` vs. continuing with current skill is heuristic: `steps_since_retrieval >= budget_n or stuck_counter >= 3`.

### 3.2 What GRPO Trains

GRPO already trains the decision agent's full policy (via `GRPOTrainer` / VERL `RayPPOTrainer`). The `select_skill` action is part of the action space. What needs to improve:

#### A. Query Key Generation (already trained by GRPO)

The LLM generates `{"key": "navigate to pot with onion"}`. GRPO reward flows back through the full episode return. The agent learns to generate better query keys because better keys → better skill matches → better follow-shaping reward → higher episode return.

No additional work needed here. The existing `r_total = r_env + w_follow * r_follow + r_cost` already captures this.

#### B. When to Select (timing policy)

**Current:** Heuristic — `can_select = steps_since_retrieval >= N or stuck_counter >= 3 or active_skill_id is None`

**Proposed:** Remove the hard gating. Let GRPO learn when to select.

```python
# REMOVE this:
can_select = s.steps_since_retrieval >= self.retrieval_budget_n or ...

# KEEP the cost signal:
# c_skill = -0.05 per select_skill call (already in reward_func.py)
# c_switch = -0.10 per skill switch (already in reward_func.py)
```

The cost penalties (`c_skill`, `c_switch`) already exist in `decision_grpo.yaml`. By removing the hard gate and relying on GRPO to learn the timing through cost-vs-benefit tradeoff, the agent learns:

- Don't re-select every step (costs accumulate)
- Do re-select when stuck (future reward justifies the cost)
- Do re-select when the current skill's termination hint is met (need a new skill)

#### C. Skill Ranking (replacing the fixed scoring function)

**Current:** `SkillQueryEngine.select()` uses a fixed weighted sum: `0.4 * relevance + 0.35 * applicability + 0.25 * pass_rate`

**Proposed:** Train a RETRIEVAL LoRA adapter (on the shared Qwen3-14B base) to re-rank skills.

| Component | Detail |
|-----------|--------|
| **Input** | Query key + current state predicates + top-K candidates from `SkillQueryEngine.select()` (with their scores, contracts, match details) |
| **Output** | Re-ranked list with selected skill_id and reasoning |
| **Reward** | Episode-level: did selecting this skill lead to better return? |

This is a **contextual bandit within GRPO**: the RETRIEVAL adapter sees the candidate list and picks one. The episode return provides the reward signal. Over many episodes, GRPO learns which ranking decisions lead to better outcomes.

Implementation approach — add an LLM re-ranking step inside `select_skill_from_bank()`:

```python
def select_skill_from_bank(skill_bank, key, current_state, memory, top_k=1):
    # Step 1: existing SkillQueryEngine.select() produces candidates
    engine = SkillQueryEngine(skill_bank)
    candidates = engine.select(key, current_state=current_state, top_k=5)

    # Step 2: LLM re-rank (RETRIEVAL adapter)
    reranked = llm_rerank_skills(candidates, key, current_state)
    if reranked:
        return reranked[0]

    # Step 3: fallback to SkillQueryEngine's ranking
    return candidates[0].to_dict()
```

The `llm_rerank_skills()` function formats the candidates into a prompt, calls the RETRIEVAL LoRA adapter, and parses the selection. During GRPO training, the gradient flows through the re-ranking decision.

### 3.3 Reward Signal for select_skill

The reward for `select_skill` is already decomposed in the existing codebase:

| Component | Source | What it captures |
|-----------|--------|-----------------|
| `r_env` | Environment reward | Did the selected skill lead to game progress? |
| `r_follow` | `reward_func.py: _compute_follow()` | Did the agent follow the skill's contract (eff_add satisfied)? |
| `r_cost` | `reward_func.py: _compute_cost()` | Cost of querying/switching skills |
| `r_tool` | `tool_call_reward.py` | Retrieval relevance + utility of the selected skill |

Total: `r_total = r_env + w_follow * r_follow + r_cost + r_tool`

This already exists and flows into GRPO. The changes are:

1. **Remove hard gating** on when `select_skill` can be called
2. **Add LLM re-ranking** step using RETRIEVAL adapter
3. **Include re-ranking in the GRPO computation graph** so gradients reach the RETRIEVAL adapter

### 3.4 Files to Create/Modify

- [ ] Modify `decision_agents/agent.py` — remove `can_select` hard gate, always allow `select_skill` in the action space
- [ ] Modify `decision_agents/agent_helper.py: select_skill_from_bank()` — add LLM re-ranking step using RETRIEVAL adapter
- [ ] Create `decision_agents/skill_reranker.py` — RETRIEVAL adapter re-ranking logic, prompt formatting
- [ ] Modify `trainer/decision/reward_shaping.py` — include `r_tool` in the per-step reward passed to GRPO
- [ ] Modify `trainer/common/configs/decision_grpo.yaml` — add `reranker` section (enabled, top_k_candidates, temperature)
- [ ] The existing `skill_agents/tool_call_reward.py` already computes `r_relevance` and `r_utility` — no changes needed

### 3.5 Training Flow

```
GRPO episode rollout:
  step 0: get_state_summary → "near_counter=true, holding_onion=true, ..."
  step 1: LLM generates TOOL: select_skill, ARGS: {"key": "place onion in pot"}
          → SkillQueryEngine.select() returns top-5 candidates
          → RETRIEVAL adapter re-ranks → picks "navigate_to_pot"
          → r_tool = relevance_score * 0.5 + utility_score * 0.5
  step 2: LLM generates TOOL: take_action, ARGS: {"action": "move_south"}
          → r_env = 0.0, r_follow = +0.05 (near_pot predicate getting closer)
  step 3: take_action → r_env = 0.0, r_follow = +0.05
  step 4: take_action → r_env = +1.0 (onion placed!), r_follow = +0.20 (completion)
  ...

Total episode return → GRPO advantage → updates:
  - Decision agent policy (when to select, what key to generate, which action to take)
  - RETRIEVAL adapter (which candidate to pick from the re-ranked list)
```

---

## Dependency Graph

```
                    ┌─────────────────────┐
                    │  Cold-start bank     │
                    │  (GPT-5.4 rollouts)  │
                    └──────────┬──────────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                │
    ┌─────────▼──────┐  ┌─────▼──────┐  ┌──────▼─────────┐
    │ §1: Stages 1-3 │  │ §2: Stage 4│  │ §3: select_skill│
    │ GRPO adapters  │  │ LLM-advised│  │ GRPO + reranker │
    │ (Qwen3-14B     │  │ propose →  │  │ (Qwen3-14B     │
    │  + LoRA:       │  │ filter →   │  │  + LoRA:       │
    │  BOUNDARY,     │  │ execute    │  │  RETRIEVAL)    │
    │  SEGMENT,      │  │ (Qwen3-14B │  │                │
    │  CONTRACT)     │  │  + LoRA:   │  │                │
    │                │  │  CURATOR)  │  │                │
    └────────┬───────┘  └─────┬──────┘  └──────┬─────────┘
             │                │                │
             └────────────────┼────────────────┘
                              │
                    ┌─────────▼─────────┐
                    │  Co-evolution loop │
                    │  (launch_coevo.py) │
                    └───────────────────┘

    All components use Qwen3-14B. No mixed model sizes.
    5 LoRA adapters total: BOUNDARY, SEGMENT, CONTRACT, CURATOR, RETRIEVAL.
```

**§1 and §3 can be developed in parallel** — they share the Qwen3-14B LoRA infrastructure but operate on different adapters (BOUNDARY/SEGMENT/CONTRACT vs RETRIEVAL).

**§2 Phase 0 starts immediately** — counterfactual logging requires zero new infrastructure (just wrap `run_bank_maintenance()` with leave-one-out evaluation). Phase 1 (SFT) and Phase 2 (GRPO) depend on §1.5 shared GRPO infrastructure.

## Implementation Priority

| Priority | Item | Effort | Rationale |
|----------|------|--------|-----------|
| **P0** | §1.5 — Shared GRPO infrastructure (`grpo_lora_updater.py`) | 2 days | Everything else depends on this |
| **P0** | §3.4 — Remove hard gate + add RETRIEVAL re-ranker | 2 days | Simplest win, immediately improves decision agent |
| **P0** | §2 Phase 0 — Counterfactual logging in `em_trainer.py` | 1 day | Zero-cost data collection, starts immediately |
| **P1** | §1.4 — Stage 3 GRPO (CONTRACT adapter) — integrate with `stage3_mvp/` | 3 days | Highest impact: contracts feed into both decode scoring and follow-shaping reward |
| **P1** | §2 Phases 1-2 — Stage 4 LLM filter + CURATOR GRPO — integrate with `bank_maintenance/` | 3 days | Simpler than old design: single-turn filter + standard GRPO |
| **P2** | §1.3 — Stage 2 GRPO (SEGMENT adapter) — replaces LLM teacher in `infer_segmentation/llm_teacher.py` | 4 days | Largest gap: preference-learning pipeline needs careful GRPO integration |
| **P2** | §1.2 — Stage 1 GRPO (BOUNDARY adapter) — integrates with `boundary_proposal/` | 2 days | Depends on §1.3 for reward signal (decode quality) |

---

## Open Questions

1. **GRPO batch size vs EM cadence**: Currently EM runs every 500 decision-agent episodes. With GRPO on Stages 1–3, each EM step is more expensive (G parallel generations + evaluation). Should we increase cadence to 1000?

2. **Shared vs separate GRPO optimizers**: Each LoRA adapter has its own optimizer state. Should we use a single learning rate schedule across all 5 adapters, or tune independently?

3. **RETRIEVAL adapter training data**: The RETRIEVAL adapter needs (query, candidates, selected, outcome_reward) tuples. During GRPO, these come from rollouts. But initially the adapter has no training — should we warm-start from the `SkillQueryEngine.select()` rankings (distillation)?

4. **GPU memory for LoRA training on 14B**: Qwen3-14B in bf16 ≈ 28GB. With LoRA (rank 16) and 5 adapters loaded, add ~1GB total (LoRA params are tiny). GRPO training adds optimizer states + gradients for adapter params only (~1–2GB). Fits on a single 80GB A100 but requires careful `gpu-memory-utilization` tuning when vLLM is also serving on the same GPU. Consider: (a) dedicated GPU for LoRA training separate from vLLM serving, or (b) time-slicing — pause vLLM during GRPO LoRA updates, resume for inference.

5. **Stage 4 Phase 0 duration**: How many EM iterations before transitioning from Phase 0 (logging only) to Phase 1 (SFT)? 100 is the current estimate — is this enough counterfactual data for reliable SFT?

6. **Refine auto-approve threshold**: Phase 0 data may show that refine with `too_strong` trigger (top_violating_literals non-empty) and clear evidence is always beneficial. If so, auto-approve high-confidence refines without LLM review (reduce prompt size, faster execution).

7. **Stage 2 GRPO integration point**: The plan uses Option A (GRPO replaces LLM teacher, preferences feed into existing PreferenceStore → Bradley-Terry → SegmentScorer → decoder). Should we start with Option C (re-ranker on top) for lower risk, and upgrade to Option A after validation?

8. **Model mismatch between cold-start and co-evolution**: Cold-start extraction uses GPT-5.4 (`labeling/extract_skillbank_gpt54.py`); co-evolution uses Qwen3-14B. The initial bank quality may degrade when switching models. Should we run a calibration pass where Qwen3-14B re-processes the cold-start bank?

9. **Stage 4 local re-decode cost**: After splits and merges, `redecode_windows()` re-runs Stage 2 on affected trajectory windows. With `redecode_window_pad=300`, this can be expensive for large banks with many splits per iteration. Should we batch re-decode requests and run them asynchronously?

---

## Codebase Cleanup — Qwen3-8B → Qwen3-14B

The following files currently reference `Qwen3-8B` and must be updated to `Qwen3-14B`:

- [ ] `skill_agents/lora/config.py` — `MultiLoraConfig.base_model_name_or_path` default
- [ ] `skill_agents/lora/config.py` — `LoraTrainingConfig.base_model_name_or_path` default
- [ ] `skill_agents/lora/model.py` — docstring references
- [ ] `skill_agents/lora/README.md`
- [ ] `trainer/common/configs/skillbank_em.yaml` — `lora.base_model_name_or_path`
- [ ] `trainer/skillbank/lora/train_lora.py`
- [ ] `configs/skillbank_lora.yaml`
- [ ] `scripts/skillbank_agent_train.sh`
- [ ] `scripts/coevolution_train.sh`
- [ ] `scripts/decision_agent_train.sh`
- [ ] `tests/test_lora_dispatch.py`
- [ ] `readme.md`
- [ ] `labeling/SKILL_INTEGRATION_STRATEGIES.md`
