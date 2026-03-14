# Skill Bank GRPO + Tool-Calling Agent Plan

**Created:** 2026-03-14  
**Status:** Draft  
**Depends on:** `SKILLBANK_AUDIT_GAPS.md`, existing Hard-EM pipeline, Decision Agent GRPO trainer

---

## Overview

Three workstreams:

1. **Stages 1–3 GRPO**: Replace Hard-EM's SFT/frequency-counting with GRPO-trained LoRA adapters on Qwen3-14B, so each stage _generates_ its output and is rewarded by downstream metrics.
2. **Stage 4 Tool-Calling Agent**: Replace the algorithmic `run_update()` with a Qwen3-14B tool-calling agent that inspects the bank, trials mutations (merge/split/materialize), and accepts/rejects based on verifiable quality deltas.
3. **`select_skill` GRPO**: Train the decision agent's skill-selection policy with GRPO so it learns _when_ to query, _which_ skill to pick, and _when_ to switch — rewarded by downstream episode return.

**Model convention:** This project uses a single Qwen model size throughout — **Qwen3-14B** for all components (vLLM serving, LoRA adapters, decision agent, tool-calling). No mixed model sizes. All existing references to Qwen3-8B in the codebase (`skill_agents/lora/config.py`, `trainer/common/configs/skillbank_em.yaml`, etc.) must be updated to Qwen3-14B.

---

## 1. Stages 1–3: GRPO-Trained Skill Bank Agents

### 1.1 Architecture

Each stage uses the shared Qwen3-14B base model with a dedicated LoRA adapter (BOUNDARY, SEGMENT, CONTRACT). During GRPO training, each adapter is updated independently. The shared base stays frozen. This is the same Qwen3-14B used for vLLM serving and the decision agent — one model size for the entire project.

**GPU memory note:** Qwen3-14B in bf16 ≈ 28GB base. With LoRA rank 16 and 4 adapters loaded, add ~200MB per adapter (LoRA params are tiny relative to the base). GRPO training adds optimizer states + gradients for adapter params only (~1–2GB total). Fits comfortably on a single 80GB A100. For inference via vLLM, the existing `--gpu-memory-utilization 0.75` (≈60GB on 80GB) is sufficient for 14B with generous KV cache.

```
For each EM batch of trajectories:
  Stage 1: boundary adapter generates cut proposals     → rewarded by Stage 2 decode quality
  Stage 2: segment adapter generates skill assignments  → rewarded by Stage 3 contract pass rate
  Stage 3: contract adapter generates effect contracts   → rewarded by holdout verification + decision follow score
```

### 1.2 Stage 1 — Boundary Proposal (GRPO)

**Current state:** Rule-based (`stage1_propose_cuts.py`). Computes predicate-change signals, surprisal, action transitions, merges nearby cuts, enforces minimum width. No learning.

**What changes:**

The BOUNDARY LoRA adapter generates a set of cut positions given a trajectory's predicate sequence.

| Component | Detail |
|-----------|--------|
| **Input (prompt)** | Trajectory summary: per-timestep predicates (booleanized), action labels, surprisal values. Truncated to last 200 steps if needed. |
| **Output (generation)** | JSON list of cut positions: `[12, 27, 45, ...]` |
| **Group generation** | `G=4` independent proposals per trajectory (low G because boundary proposals are cheap to evaluate) |
| **Reward** | Computed after running Stage 2 decode on each proposal. See reward table below. |

**Concrete I/O example:**

*Input prompt:*

```
Propose boundary cut positions for this trajectory.

Trajectory (45 timesteps):
t=0:  preds=[at_counter, not_holding, onion_on_counter]  action=noop        surprisal=0.1
t=1:  preds=[at_counter, not_holding, onion_on_counter]  action=interact     surprisal=0.2
t=2:  preds=[at_counter, holding_onion, not_onion_on_counter] action=move_south surprisal=0.8
t=3:  preds=[near_counter, holding_onion]                action=move_south   surprisal=0.3
t=4:  preds=[mid_map, holding_onion]                     action=move_south   surprisal=0.2
...
t=11: preds=[near_pot, holding_onion, facing_pot]        action=move_south   surprisal=0.2
t=12: preds=[at_pot, holding_onion, facing_pot]          action=interact     surprisal=0.9
t=13: preds=[at_pot, not_holding, onion_in_pot]          action=noop         surprisal=0.7
t=14: preds=[at_pot, not_holding, onion_in_pot]          action=move_north   surprisal=0.5
t=15: preds=[near_pot, not_holding, onion_in_pot]        action=move_north   surprisal=0.3
...
t=26: preds=[at_counter, not_holding, dish_on_counter]   action=interact     surprisal=0.8
t=27: preds=[at_counter, holding_dish]                   action=move_south   surprisal=0.6
...
t=38: preds=[at_pot, holding_dish, pot_cooked]           action=interact     surprisal=0.9
t=39: preds=[at_pot, holding_soup]                       action=move_east    surprisal=0.7
...
t=44: preds=[at_serve, holding_soup]                     action=interact     surprisal=0.8
t=45: preds=[at_serve, not_holding, soup_delivered]      action=noop         surprisal=0.1

Respond with a JSON list of cut positions (timestep indices where segments end).
```

*Output (generation):*

```json
[12, 26, 38, 45]
```

*Interpretation:* The model proposes 4 segments: t=0–12 (pick up onion, carry to pot), t=13–26 (go get a dish), t=27–38 (carry dish to pot, plate soup), t=39–45 (serve soup). Each cut lands at a high-surprisal action transition.

**Reward function for Stage 1:**

| Component | Formula | Weight | Rationale |
|-----------|---------|--------|-----------|
| `r_decode_margin` | `mean(seg.margin for seg in decode_result.segments)` normalized to [0,1] | 0.35 | Segments decoded with high margin → cuts landed at natural boundaries |
| `r_known_rate` | `1.0 - (n_new / n_total_segments)` | 0.25 | Low NEW rate → cuts align with known skills |
| `r_pass_rate` | `mean(contract.pass_rate for contract in stage3_result)` | 0.25 | Downstream contract quality validates boundary placement |
| `r_frag_penalty` | `-max(0, n_segments - 2 * n_expected) / n_expected` | 0.15 | Prevents over-fragmentation |

**Training loop:**

```python
for batch in trajectory_batches:
    proposals = []
    for traj in batch:
        prompt = format_boundary_prompt(traj)
        # Generate G proposals from BOUNDARY adapter
        group = [llm.generate(SkillFunction.BOUNDARY, prompt, temperature=0.7)
                 for _ in range(G)]
        proposals.append((traj, group))

    # Evaluate: run Stage 2 decode for each proposal
    rewards = []
    for traj, group in proposals:
        group_rewards = []
        for cuts_json in group:
            cuts = parse_cuts(cuts_json)
            decode_result = decode_trajectory(traj, cuts, bank)
            r = compute_boundary_reward(decode_result, traj)
            group_rewards.append(r)
        rewards.append(group_rewards)

    # GRPO update on BOUNDARY adapter
    advantages = compute_grpo_advantages(rewards)  # group-normalized
    update_lora(SkillFunction.BOUNDARY, prompts, generations, advantages)
```

**Files to create/modify:**

- [ ] `trainer/skillbank/grpo/stage1_grpo.py` — training loop, prompt formatting, reward computation
- [ ] `trainer/skillbank/grpo/prompts.py` — shared prompt templates for all 3 stages
- [ ] Modify `trainer/skillbank/em_trainer.py` — call GRPO boundary proposer instead of rule-based `propose_cuts()`
- [ ] Keep `stage1_propose_cuts.py` as fallback / baseline comparison

---

### 1.3 Stage 2 — Decode / Segmentation (GRPO)

**Current state:** Dynamic programming (`stage2_decode.py`). Scores each (segment, skill) pair via `_score_skill()` (effect matching + surprisal), runs Viterbi-style DP, optional LLM re-ranking.

**What changes:**

The SEGMENT LoRA adapter generates skill-label assignments given segments (from Stage 1 cuts) and the current skill bank.

| Component | Detail |
|-----------|--------|
| **Input (prompt)** | Per-segment: predicates at start/end, action sequence summary, candidate skill list with contract summaries (top-M from DP pre-filter). |
| **Output (generation)** | JSON: `{"segments": [{"start": 0, "end": 12, "skill": "navigate_pot"}, ...]}` |
| **Group generation** | `G=8` independent decodings per episode (higher G because assignments are more variable) |
| **Reward** | Contract pass rate + follow-shaping score from decision agent. See reward table. |

**Concrete I/O example:**

*Input prompt:*

```
Assign skills to each segment. For each segment, you see the predicates at
start/end, the action sequence, and the top-3 candidate skills from the bank.

Segment 1 (t=0 to t=12):
  start_preds: [at_counter, not_holding, onion_on_counter]
  end_preds:   [at_pot, holding_onion, facing_pot]
  actions:     [noop, interact, move_south x9, interact]
  candidates:
    1. S04_pick_and_carry_onion (score=0.82)
       contract: eff_add={holding_onion, at_pot} eff_del={onion_on_counter, not_holding}
    2. S11_navigate_to_pot      (score=0.65)
       contract: eff_add={at_pot, facing_pot} eff_del={far_from_pot}
    3. NEW                      (score=0.30)

Segment 2 (t=13 to t=26):
  start_preds: [at_pot, not_holding, onion_in_pot]
  end_preds:   [at_counter, not_holding, dish_on_counter]
  actions:     [noop, move_north x11, interact]
  candidates:
    1. S08_fetch_dish           (score=0.78)
       contract: eff_add={at_counter, dish_on_counter} eff_del={at_pot}
    2. S11_navigate_to_counter  (score=0.55)
       contract: eff_add={at_counter} eff_del={at_pot}
    3. NEW                      (score=0.35)

Segment 3 (t=27 to t=38):
  start_preds: [at_counter, holding_dish]
  end_preds:   [at_pot, holding_soup, pot_cooked]
  actions:     [move_south x11, interact]
  candidates:
    1. S06_plate_soup           (score=0.88)
       contract: eff_add={holding_soup} eff_del={holding_dish, pot_cooked}
    2. S04_pick_and_carry_onion (score=0.20)
       contract: eff_add={holding_onion, at_pot} eff_del={onion_on_counter}
    3. NEW                      (score=0.25)

Segment 4 (t=39 to t=45):
  start_preds: [at_pot, holding_soup]
  end_preds:   [at_serve, not_holding, soup_delivered]
  actions:     [move_east x5, interact]
  candidates:
    1. S09_serve_soup           (score=0.91)
       contract: eff_add={soup_delivered, not_holding} eff_del={holding_soup}
    2. S11_navigate_to_serve    (score=0.60)
       contract: eff_add={at_serve} eff_del={at_pot}
    3. NEW                      (score=0.20)

Respond with JSON assigning one skill to each segment.
```

*Output (generation):*

```json
{
  "segments": [
    {"start": 0,  "end": 12, "skill": "S04_pick_and_carry_onion"},
    {"start": 13, "end": 26, "skill": "S08_fetch_dish"},
    {"start": 27, "end": 38, "skill": "S06_plate_soup"},
    {"start": 39, "end": 45, "skill": "S09_serve_soup"}
  ]
}
```

*Interpretation:* The model picks the top-ranked candidate for each segment. All 4 are existing skills (no NEW labels), and each assignment's contract effects match the observed start→end predicate changes. A weaker generation might pick `S11_navigate_to_pot` for segment 1 (missing the `holding_onion` effect) or label segment 2 as NEW.

**Reward function for Stage 2:**

| Component | Formula | Weight | Rationale |
|-----------|---------|--------|-----------|
| `r_pass_rate` | `mean(contract.pass_rate for skill in assignment)` | 0.30 | Assigned skills should have valid contracts |
| `r_margin` | `mean(seg.margin)` — how much best skill beat second-best | 0.20 | Confident assignments |
| `r_follow` | `mean(r_follow_t)` from decision agent reward shaping over holdout episodes | 0.25 | Downstream: does the decision agent benefit from this segmentation? |
| `r_new_penalty` | `-0.5 * new_rate` | 0.10 | Discourage excessive NEW labels |
| `r_confusion_penalty` | `-mean(confusion_overlap for confuser_pairs)` | 0.15 | Different skills should have different effects |

**Key design choice — DP pre-filter + LLM re-rank:**

We do NOT replace DP entirely. Instead:

1. DP generates top-M candidate assignments per segment (fast, CPU)
2. SEGMENT adapter re-ranks/selects among candidates (LLM, GPU)
3. GRPO trains the re-ranker

This keeps the combinatorial search tractable while letting the LLM learn subtle preferences that DP's hand-crafted scoring misses.

**Files to create/modify:**

- [ ] `trainer/skillbank/grpo/stage2_grpo.py` — training loop, candidate generation via DP, LLM re-ranking, reward
- [ ] Modify `trainer/skillbank/stages/stage2_decode.py` — add `decode_with_llm_rerank()` that uses the SEGMENT adapter
- [ ] Keep existing `decode_batch()` as the DP baseline

---

### 1.4 Stage 3 — Contract Learning (GRPO)

**Current state:** Frequency counting (`stage3_contracts.py` + `contract_learn.py`). Counts predicate occurrences across instances, thresholds at `eff_freq=0.8`, optional LLM enrichment via CONTRACT adapter (union only).

**What changes:**

The CONTRACT LoRA adapter generates full effect contracts given segment evidence, replacing frequency counting entirely.

| Component | Detail |
|-----------|--------|
| **Input (prompt)** | Skill ID, N representative segment instances (predicates at start/end, actions, events), current bank contract (if exists). |
| **Output (generation)** | JSON: `{"eff_add": [...], "eff_del": [...], "eff_event": [...], "description": "..."}` |
| **Group generation** | `G=4` per skill (contracts are less variable than segmentations) |
| **Reward** | Holdout verification pass rate + decision agent follow score. See reward table. |

**Concrete I/O example:**

*Input prompt:*

```
Generate an effects contract for this skill based on its segment instances.

Skill: S04_pick_and_carry_onion
Current contract (if any): eff_add={holding_onion, at_pot}, eff_del={onion_on_counter, not_holding}

Representative instances (8 of 23 total):

Instance 1:
  start_preds: [at_counter, not_holding, onion_on_counter, far_from_pot]
  end_preds:   [at_pot, holding_onion, facing_pot, near_pot]
  actions:     [interact, move_south x10, interact]
  events:      [picked_up_onion]

Instance 2:
  start_preds: [at_counter, not_holding, onion_on_counter, teammate_at_pot]
  end_preds:   [near_pot, holding_onion, facing_pot]
  actions:     [interact, move_south x8]
  events:      [picked_up_onion]

Instance 3:
  start_preds: [near_counter, not_holding, onion_on_counter]
  end_preds:   [at_pot, holding_onion, facing_pot, near_pot]
  actions:     [move_east, interact, move_south x9, interact]
  events:      [picked_up_onion]

Instance 4:
  start_preds: [at_counter, not_holding, onion_on_counter, pot_empty]
  end_preds:   [at_pot, holding_onion, facing_pot, near_pot, pot_empty]
  actions:     [interact, move_south x10, interact]
  events:      [picked_up_onion]

Instance 5:
  start_preds: [at_counter, not_holding, onion_on_counter]
  end_preds:   [near_pot, holding_onion]
  actions:     [interact, move_south x7]
  events:      [picked_up_onion]

... (3 more instances omitted for brevity)

Predicate frequencies across all 23 instances:
  eff_add:  holding_onion (100%), near_pot (87%), facing_pot (83%), at_pot (70%)
  eff_del:  not_holding (100%), onion_on_counter (96%)
  events:   picked_up_onion (100%)

Generate a JSON effects contract. Include predicates that are reliably caused
by this skill, not coincidental.
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

*Interpretation:* The model includes `near_pot` (87%) and `facing_pot` (83%) despite being below the old 80% frequency threshold, because they're semantically core to the skill. It excludes `at_pot` (70%) because many instances stop a tile short. The old frequency-counting approach would include `at_pot` at threshold=0.7 but exclude `facing_pot` at threshold=0.85 — the LLM makes a better contextual call. A bad generation might bloat the contract with `pot_empty` (appears in some instances but is a background predicate, not caused by this skill).

**Reward function for Stage 3:**

| Component | Formula | Weight | Rationale |
|-----------|---------|--------|-----------|
| `r_holdout_pass` | `verify_effects_contract(contract, holdout_instances).overall_pass_rate` | 0.35 | Contract must generalize to unseen instances |
| `r_decision_follow` | `mean(r_follow)` from decision agent using the new contract on holdout episodes | 0.25 | Downstream: does the decision agent follow this contract better? |
| `r_sparsity` | `-max(0, n_literals - budget) / budget` | 0.15 | Prevent bloated contracts |
| `r_coverage` | `n_instances_covered / n_total_instances` | 0.15 | Contract should explain most instances |
| `r_overfit_penalty` | `-(train_pass_rate - holdout_pass_rate)` if gap > 0.1 | 0.10 | Generalization check |

**Critical difference from current approach:**

Current: frequency counting is deterministic — same input always produces same contract.  
GRPO: the LLM generates diverse contracts, and the best ones (by holdout verification + decision-agent utility) are reinforced. This allows the model to learn:

- When to include a predicate at 70% frequency (below threshold but semantically important)
- When to exclude a predicate at 90% frequency (noisy/irrelevant)
- Contextual effects (predicate X matters only when predicate Y is present)

**Files to create/modify:**

- [ ] `trainer/skillbank/grpo/stage3_grpo.py` — training loop, evidence formatting, holdout verification, reward
- [ ] Modify `trainer/skillbank/stages/stage3_contracts.py` — add `learn_contracts_grpo()` that calls CONTRACT adapter
- [ ] Keep existing `learn_contracts()` as the frequency-counting baseline

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

Stage 4 mutates the skill bank through four actions — **refine, merge, split, materialize**. Skills stored in the bank are living objects: their contracts (eff_add, eff_del, eff_event) evolve as new evidence arrives. Refine is typically the most frequent action, tightening or correcting a skill's contract based on fresh segment instances.

The architecture is **propose-filter-execute**:

- **Propose:** The existing algorithmic `run_update()` scans the bank and generates a ranked candidate action list with computed thresholds and quality estimates for all four action types.
- **Filter:** A single LLM call reviews the candidates against the full diagnostics context and returns an approved/vetoed/deferred subset.
- **Execute:** The algorithm executes approved actions using the existing deterministic functions (`_refine`, `_merge_similar`, `_split_weak`, `_materialize_new`).

The LLM's role is narrow: contextual judgment on "should we do this?" The algorithm handles "what could we do?" and "how do we do it?"

### 2.2 Why Not a Full Multi-Turn Agent

The original design considered a 20-turn tool-calling agent loop with sandboxed trials, but this has stability risks disproportionate to Stage 4's role:

- Stage 4 runs **once per EM batch** (~every 500 episodes). Multi-turn compounding (turn N's accept mutates the bank state turn N+1 reasons about) is fragile for sparse execution.
- Sandboxed deep-copies per trial are 10-20x the compute of the algorithmic version.
- Multi-turn = 20 failure points for malformed JSON, hallucinated skill_ids, etc.

The full agentic design is preserved as Phase 3 (§2.10) for when the simpler approach hits a ceiling.

### 2.3 The Five Actions

Five actions modify the skill bank. The first three operate on existing skills; the last two handle the lifecycle of new skills entering the bank.

| Action | What it does | When it fires | What changes in the bank |
|--------|-------------|---------------|-------------------------|
| **Refine** | Re-learns a skill's contract (eff_add/eff_del) from latest segment evidence | Evidence delta ≥ 5% between current contract and new observations | Skill's contract updated in-place, version bumped, pass_rate recalculated |
| **Merge** | Combines two skills with near-identical contracts into one | Jaccard similarity ≥ 0.85 between contracts | One skill removed, survivor inherits combined instances, contract re-learned |
| **Split** | Breaks a weak skill into sub-skills by re-clustering its instances | pass_rate < 0.70 and ≥ 6 instances | Original skill replaced by 2+ children, each with its own contract |
| **Materialize** | Graduates a `__NEW__` cluster into a proto-skill in the staging area | Cluster meets `min_cluster_size`, `min_consistency`, `min_distinctiveness` | Proto-skill created in `ProtoSkillManager`; participates in Stage 2 decoding as candidate label |
| **Promote** | Promotes a verified proto-skill to a full bank skill | Proto-skill passes Stage 3 verification with `pass_rate ≥ 0.7` | Proto-skill removed from staging; real skill added to bank with verified contract and LLM-generated name |

**Refine is critical for skill evolution.** A skill like `navigate_to_pot` might start with a noisy contract `eff_add={near_pot, near_counter}` from early instances. After 50 more instances, the evidence clearly shows `eff_add={near_pot, facing_pot}` is more accurate. Refine catches this drift. Without it, stale contracts degrade pass_rate and confuse the decision agent's follow-shaping reward.

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

The existing `run_update()` logic is refactored to produce candidates without executing them:

```python
@dataclass
class CandidateAction:
    action_type: str          # "refine" | "merge" | "split" | "materialize" | "promote"
    skill_ids: List[str]
    rationale: str            # human-readable reason this was proposed
    priority: float           # algorithm's confidence (used for default ordering)
    details: Dict[str, Any]   # action-specific data (old/new contract for refine, jaccard for merge, etc.)
    conflicts_with: List[int] = field(default_factory=list)  # indices of conflicting candidates
    llm_verdict: Optional[str] = None  # "approve" | "veto" | "defer" (filled by LLM)
    llm_reason: Optional[str] = None


def propose_candidates(
    decode_results: List[DecodeResult],
    contracts: Dict[str, LearnedContract],
    bank: SkillBankMVP,
    new_pool_mgr: NewPoolManager,
    proto_mgr: ProtoSkillManager,
    config: UpdateConfig,
) -> List[CandidateAction]:
    """Scan the bank and return ranked candidate actions without executing any."""
    candidates = []

    # 1. Refine: skills where new evidence diverges from current contract
    for skill_id, lc in contracts.items():
        existing = bank.get_contract(skill_id)
        if existing is None or not lc.verified:
            continue
        delta = _refine_delta(existing, lc)
        if delta >= config.refine_delta_threshold:
            candidates.append(CandidateAction(
                action_type="refine",
                skill_ids=[skill_id],
                rationale=f"Evidence delta {delta:.0%}, pass_rate {lc.pass_rate:.2f}, "
                          f"{lc.n_instances} instances",
                priority=delta,
                details={
                    "old_eff_add": sorted(existing.eff_add),
                    "old_eff_del": sorted(existing.eff_del),
                    "new_eff_add": sorted(lc.eff_add),
                    "new_eff_del": sorted(lc.eff_del),
                    "pass_rate": lc.pass_rate,
                },
            ))

    # 2. Merge: skill pairs with Jaccard above threshold
    for (id_a, id_b), jaccard in _pairwise_jaccard(bank, config):
        if jaccard >= config.merge_jaccard_threshold:
            candidates.append(CandidateAction(
                action_type="merge",
                skill_ids=[id_a, id_b],
                rationale=f"Jaccard {jaccard:.2f}, combined "
                          f"{_combined_instances(bank, id_a, id_b)} instances",
                priority=jaccard,
                details={"jaccard": jaccard},
            ))

    # 3. Split: weak skills with enough instances for sub-clustering
    for skill_id, lc in contracts.items():
        if (lc.pass_rate < config.split_pass_rate_threshold
                and lc.n_instances >= config.min_child_size * 2):
            n_sub = _n_subclusters(lc)
            candidates.append(CandidateAction(
                action_type="split",
                skill_ids=[skill_id],
                rationale=f"pass_rate {lc.pass_rate:.2f}, {lc.n_instances} instances, "
                          f"failure sigs suggest {n_sub} sub-clusters",
                priority=1.0 - lc.pass_rate,
                details={"pass_rate": lc.pass_rate, "n_subclusters": n_sub},
            ))

    # 4. Materialize: NEW clusters ready for proto-skill formation
    #    Sourced from NewPoolManager.get_candidates() — clusters that pass
    #    min_cluster_size, min_consistency, min_distinctiveness gates
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

    # 5. Promote: proto-skills ready for full bank promotion
    #    Sourced from ProtoSkillManager — proto-skills that passed
    #    light verification and meet promotion thresholds
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

    # Detect conflicts (e.g. refine(S12) + merge(S05,S12) on the same skill)
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
    Old contract: eff_add={near_pot, near_counter}, eff_del={}
    New evidence:  eff_add={near_pot, facing_pot}, eff_del={far_from_pot}
    Rationale: Evidence delta 33%, pass_rate 0.55, 8 instances

[2] MERGE S05_go_to_pot + S12_navigate_pot (priority=0.87)
    Jaccard: 0.87, combined 15 instances
    ⚠ CONFLICTS WITH [1] — both touch S12_navigate_pot

[3] MATERIALIZE cluster_7 → proto-skill (priority=0.34)
    8 segments, consistency 0.85, sig=A:near_pot,facing_pot|D:
    Distinctiveness from existing skills: 0.42 (passes min 0.25)

[4] SPLIT S23_place_onion (priority=0.38)
    pass_rate 0.62, 12 instances, failure sigs suggest 2 sub-clusters

[5] REFINE S07_pick_onion (priority=0.50)
    Old contract: eff_del={}
    New evidence:  eff_del={onion_on_counter}
    Rationale: Evidence delta 50%, pass_rate 0.71, 14 instances

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

After filtering, approved actions execute using the existing deterministic functions. The key change from the old plan: **materialize** and **promote** go through the proper `NewPoolManager` / `ProtoSkillManager` pipeline, not a simplified shortcut.

```python
def execute_approved(
    approved: List[CandidateAction],
    decode_results: List[DecodeResult],
    contracts: Dict[str, LearnedContract],
    bank: SkillBankMVP,
    new_pool_mgr: NewPoolManager,
    proto_mgr: ProtoSkillManager,
    observations_by_traj: Dict[str, list],
    config: UpdateConfig,
) -> UpdateResult:
    result = UpdateResult()
    for action in approved:
        if action.action_type == "refine":
            _refine_single(action.skill_ids[0], contracts, bank, config, result)

        elif action.action_type == "merge":
            _merge_pair(action.skill_ids[0], action.skill_ids[1], bank, config, result)

        elif action.action_type == "split":
            _split_single(action.skill_ids[0], contracts, bank, config, result)

        elif action.action_type == "materialize":
            # Cluster → proto-skill via ProtoSkillManager
            cluster_id = int(action.skill_ids[0].replace("cluster_", ""))
            summary = _find_cluster_summary(new_pool_mgr, cluster_id)
            if summary is not None:
                records = new_pool_mgr.get_cluster_records(cluster_id)
                proto = proto_mgr.form_from_cluster(
                    summary, records,
                    existing_bank_skills=set(bank.skill_ids),
                )
                if proto is not None:
                    result.n_materialize += 1
                    result.events.append(UpdateEvent(
                        event_type="materialize",
                        skill_ids=[proto.proto_id],
                        details={"cluster_id": cluster_id, "support": summary.size},
                    ))

        elif action.action_type == "promote":
            # Proto-skill → real skill via ProtoSkillManager
            proto_id = action.skill_ids[0]
            pass_rate = proto_mgr.verify(proto_id, bank, observations_by_traj)
            if pass_rate is not None and pass_rate >= config.promotion_min_pass_rate:
                promoted = proto_mgr.promote_ready(bank)
                if proto_id in promoted:
                    result.n_materialize += 1
                    result.events.append(UpdateEvent(
                        event_type="promote",
                        skill_ids=[proto_id],
                        details={"pass_rate": pass_rate},
                    ))

    return result
```

**Key differences from old execution:**

| Action | Old plan | New plan |
|--------|----------|----------|
| Refine | `_refine_single()` | Same — no change |
| Merge | `_merge_pair()` | Same — no change |
| Split | `_split_single()` | Same — no change |
| Materialize | `_materialize_single()` — directly adds to bank | `ProtoSkillManager.form_from_cluster()` — creates proto-skill in staging area |
| Promote | Did not exist | `ProtoSkillManager.verify()` + `promote_ready()` — full Stage 3 verification, then bank add |

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

- [ ] `trainer/skillbank/stage4_candidates.py` — `propose_candidates()`, `CandidateAction`, conflict detection, `_annotate_conflicts()`, deferral annotation. Sources materialize candidates from `NewPoolManager.get_candidates()` and promote candidates from `ProtoSkillManager`
- [ ] `trainer/skillbank/stage4_deferrals.py` — `DeferralRecord`, `annotate_deferrals()`, deferral log persistence, expiry logic (`max_deferrals=3`)
- [ ] `trainer/skillbank/stage4_filter.py` — LLM filter call, constrained decoding, retry logic, `_apply_decisions()`, fallback
- [ ] `trainer/skillbank/stage4_prompts.py` — curator system prompt, diagnostics formatter (includes NEW pool status, proto-skill staging summary), output schema
- [ ] `trainer/skillbank/stage4_counterfactual.py` — `collect_counterfactuals()`, `CounterfactualRecord`, Phase 0 logging
- [ ] `trainer/skillbank/stage4_grpo.py` — GRPO training loop for CURATOR adapter (§2.13)
- [ ] Refactor `trainer/skillbank/stages/stage4_update.py` — extract `_refine`, `_merge_similar`, `_split_weak` into individually-callable `_refine_single()`, `_merge_pair()`, `_split_single()`. Remove `_materialize_new()` shortcut — materialize now goes through `NewPoolManager`/`ProtoSkillManager`
- [ ] Modify `trainer/skillbank/em_trainer.py` — call `propose_candidates()` → `filter_candidates()` → `execute_approved()` instead of `run_update()`. Pass `NewPoolManager` and `ProtoSkillManager` instances through
- [ ] Modify `skill_agents/skill_bank/new_pool.py` — expose `get_candidates()` summary data in a format `stage4_candidates.py` can consume for materialize proposals
- [ ] Modify `trainer/common/configs/skillbank_em.yaml` — add `stage4_filter` config section (enabled, vllm_url, temperature, fallback_on_failure), `stage4_curator` GRPO config, and `proto_skill` promotion thresholds
- [ ] Modify `trainer/skillbank/grpo/config.py` — add CURATOR adapter config alongside BOUNDARY/SEGMENT/CONTRACT
- [ ] Keep `trainer/skillbank/stages/stage4_update.py: run_update()` as algorithmic-only fallback path (still uses the old direct `_materialize_new()` for simplicity)

### 2.17 Guardrails

| Guardrail | Implementation |
|-----------|---------------|
| **Constrained decoding** | vLLM `guided_json` with strict schema — LLM cannot produce invalid JSON structure |
| **Semantic validation** | Check idx range, verdict values; invalid entries default to "approve" (algorithm's recommendation stands) |
| **Single retry** | One corrective retry on validation failure; then fallback |
| **Fallback to algorithm** | If vLLM is down or both attempts fail, execute full candidate list — never worse than algorithmic-only |
| **Quality gate** | SkillEval runs after execution — rollback if bank quality drops |
| **Conflict enforcement** | If LLM approves both sides of a conflict, only the higher-priority one executes |
| **Deferral expiry** | After `max_deferrals=3` consecutive deferrals with algorithm threshold still met, auto-approve without LLM review |
| **Promote rollback** | If `ProtoSkillManager.verify()` returns pass_rate below threshold during promote execution, proto-skill stays in staging — no bank mutation |
| **Materialize idempotency** | Materializing the same cluster twice is a no-op — `ProtoSkillManager.form_from_cluster()` deduplicates by effect signature |
| **Logging** | Every (prompt, response, executed_actions, deferrals, proto_skill_changes, q_before, q_after, counterfactuals) logged for training |

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

**§2 Phase 0 starts immediately** — counterfactual logging requires zero new infrastructure (just wrap `run_update()` with leave-one-out evaluation). Phase 1 (SFT) and Phase 2 (GRPO) depend on §1.5 shared GRPO infrastructure.

## Implementation Priority

| Priority | Item | Effort | Rationale |
|----------|------|--------|-----------|
| **P0** | §1.5 — Shared GRPO infrastructure (`grpo_lora_updater.py`) | 2 days | Everything else depends on this |
| **P0** | §3.4 — Remove hard gate + add RETRIEVAL re-ranker | 2 days | Simplest win, immediately improves decision agent |
| **P0** | §2 Phase 0 — Counterfactual logging in `em_trainer.py` | 1 day | Zero-cost data collection, starts immediately |
| **P1** | §1.4 — Stage 3 GRPO (CONTRACT adapter) | 3 days | Highest impact: contracts feed into both decode scoring and follow-shaping reward |
| **P1** | §2 Phases 1-2 — Stage 4 LLM filter + CURATOR GRPO | 3 days | Simpler than old design: single-turn filter + standard GRPO |
| **P2** | §1.3 — Stage 2 GRPO (SEGMENT adapter) | 3 days | Depends on §1.4 for reward signal |
| **P2** | §1.2 — Stage 1 GRPO (BOUNDARY adapter) | 2 days | Depends on §1.3 for reward signal (decode quality) |

---

## Open Questions

1. **GRPO batch size vs EM cadence**: Currently EM runs every 500 decision-agent episodes. With GRPO on Stages 1–3, each EM step is more expensive (G parallel generations + evaluation). Should we increase cadence to 1000?

2. **Shared vs separate GRPO optimizers**: Each LoRA adapter has its own optimizer state. Should we use a single learning rate schedule across all 5 adapters, or tune independently?

3. **RETRIEVAL adapter training data**: The RETRIEVAL adapter needs (query, candidates, selected, outcome_reward) tuples. During GRPO, these come from rollouts. But initially the adapter has no training — should we warm-start from the `SkillQueryEngine.select()` rankings (distillation)?

4. **GPU memory for LoRA training on 14B**: Qwen3-14B in bf16 ≈ 28GB. With LoRA (rank 16) and 5 adapters loaded, add ~1GB total (LoRA params are tiny). GRPO training adds optimizer states + gradients for adapter params only (~1–2GB). Fits on a single 80GB A100 but requires careful `gpu-memory-utilization` tuning when vLLM is also serving on the same GPU. Consider: (a) dedicated GPU for LoRA training separate from vLLM serving, or (b) time-slicing — pause vLLM during GRPO LoRA updates, resume for inference.

5. **Stage 4 Phase 0 duration**: How many EM iterations before transitioning from Phase 0 (logging only) to Phase 1 (SFT)? 100 is the current estimate — is this enough counterfactual data for reliable SFT?

6. **Refine auto-approve threshold**: Phase 0 data may show that refine actions with evidence delta > 20% are always beneficial. If so, auto-approve high-confidence refines without LLM review (reduce prompt size, faster execution).

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
