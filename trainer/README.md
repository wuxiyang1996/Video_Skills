# Training Infrastructure

**Preferred: VERL (verl-agent).** For distributed GiGPO/PPO training with vLLM/sglang and FSDP, use [VERL](https://github.com/verl-project/verl) via [verl-agent](https://github.com/verl-project/verl-agent):

```bash
# From Game-AI-Agent repo root; requires verl-agent at ../verl-agent
python -m scripts.run_trainer --verl
python -m scripts.run_trainer --verl algorithm.adv_estimator=gigpo trainer.nnodes=2
```

Or run the trainer module (no `--config` → delegates to VERL):

```bash
python -m trainer.decision.launch_train env.env_name=gameai ...
```

Standalone (in-repo) training remains available for debugging or when VERL is not installed: `python -m scripts.run_trainer --config trainer/common/configs/decision_grpo.yaml`.

---

Co-evolution training for two agents:

- **Agent A (VLM Decision Agent):** Trained with **GRPO** (Group Relative Policy Optimization), where `QUERY_MEM` / `QUERY_SKILL` / `CALL_SKILL` are treated as actions with query/call costs + optional skill-follow shaping.
- **Agent B (SkillBank Agent):** Trained/updated via **Hard-EM** (decode → update → gate) as an algorithmic pipeline, with optional small supervised learners (boundary classifier, top-2 tie-breaker). No global LLM scoring.

---

## 0) Repo Layout

```
trainer/
  README.md
  common/
    configs/
      decision_grpo.yaml       # GRPO hyperparams, costs, shaping weights
      skillbank_em.yaml        # EM hyperparams, gating thresholds
    logging.py                 # Structured logging for both trainers
    eval_harness.py            # Fixed-seed evaluation harness (shared)
    seeds.py                   # Deterministic seed management
    metrics.py                 # RolloutRecord schema, metric aggregation
  decision/
    env_wrapper.py             # EnvWrapper: retrieval-as-action, reward computation
    policy_interface.py        # PolicyInterface: logprob extraction for GRPO
    reward_shaping.py          # compute_reward(prev, action, next, bank_state)
    rollout_collector.py       # Parallel rollout collection
    grpo_trainer.py            # GRPO training loop
    replay_buffer.py           # Episode replay buffer
    launch_train.py            # CLI entry point for decision agent training
  skillbank/
    ingest_rollouts.py         # Convert decision agent rollouts → trajectory objects
    em_trainer.py              # Hard-EM loop driver (propose → decode → contract → update → gate)
    stages/
      stage0_predicates.py     # Extract/booleanize predicates from observations
      stage1_propose_cuts.py   # Propose boundary candidates
      stage2_decode.py         # Viterbi/DP decode: assign skill labels to segments
      stage3_contracts.py      # Learn/verify effects-only contracts
      stage4_update.py         # Refine, materialize NEW, merge/split
      skilleval.py             # SkillEval gating: pass rate, support, discriminability
    learners/
      boundary_train.py        # Optional: boundary classifier (supervised)
      tiebreaker_train.py      # Optional: top-2 tie-breaker (supervised)
    bank_io/
      bank_store.py            # Versioned bank storage with commit/rollback
      indices.py               # Skill retrieval indices (keyword, embedding)
      diff_logger.py           # Bank diff reports between versions
  launch_coevolution.py        # Top-level co-evolution orchestrator
```

---

## 1) Shared Interfaces

### 1.1 Rollout Record Format (single source of truth)

Defined in `trainer/common/metrics.py` — `RolloutRecord` dataclass used by both trainers:

| Field | Type | Description |
|---|---|---|
| `step` | int | Timestep index |
| `obs_id` / `frame_ptr` | str | Observation identifier |
| `action` | str | Action taken (primitive or retrieval) |
| `action_type` | str | `"primitive"`, `"QUERY_MEM"`, `"QUERY_SKILL"`, `"CALL_SKILL"` |
| `ui_events` | list[str] | UI events observed |
| `predicates` | dict[str, float] | Predicate probabilities |
| `embedding` | list[float] (opt.) | Observation embedding |
| `r_env` | float | Environment reward |
| `r_follow` | float | Skill-following shaping reward |
| `r_cost` | float | Action cost |
| `r_total` | float | Combined reward |
| `done` | bool | Episode termination |
| `episode_id` | str | Episode identifier |
| `traj_id` | str | Trajectory identifier |
| `seed` | int | Environment seed |
| `active_skill_id` | str \| None | Active skill before/after |
| `query_key` | str \| None | Key used in QUERY_MEM/QUERY_SKILL |

### 1.2 Reward Tool Contract

Defined in `trainer/decision/reward_shaping.py` — `compute_reward(prev, action, next, bank_state)` returns:

- `r_env` — raw environment reward
- `r_follow` — termination-free skill-following shaping
- `r_cost` — query/call/switch costs
- `r_total` — combined reward

Wraps `decision_agents.reward_func.RewardComputer` with additional bank-state-aware shaping.

### 1.3 SkillBank Query Contract

Defined in `trainer/skillbank/bank_io/bank_store.py`:

- `query_skill(key) -> topK SkillCards`
- `query_memory(key) -> topK MemoryCards`
- `SkillCard` fields: `skill_id`, `effects`, `typical_len`, `confusers`, `profile`

---

## 2) VLM Decision Agent Trainer (GRPO)

### Milestone D1 — Environment Wrapper & Actionization

`EnvWrapper.step(action)`:
- Primitive actions → forward to game env
- `QUERY_MEM`/`QUERY_SKILL` → call tool, store cards in context, apply query cost
- `CALL_SKILL` → set `active_skill` in wrapper state, apply call cost
- Returns `(obs_{t+1}, r_env, info, done, metadata)`

**Files:** `trainer/decision/env_wrapper.py`, `trainer/decision/reward_shaping.py`

### Milestone D2 — Reward Shaping

Without skill termination:
- Progress-to-late-states when `active_skill != None`
- Query/call/switch costs
- Outputs full reward breakdown

**Files:** `trainer/decision/reward_shaping.py`

### Milestone D3 — Rollout Collector

Collects rollouts with:
- Actions, `r_env`, `r_total`, reward breakdown
- Saved `obs_id`, `ui_events`, predicates, embeddings

**Files:** `trainer/decision/rollout_collector.py`, `trainer/decision/replay_buffer.py`

### Milestone D4 — GRPO Trainer

GRPO loop (group sampling + ranking objective):
1. Sample a batch of episodes with current policy
2. Compute returns/advantages from `r_total`
3. GRPO update on action logprobs
4. Log: win/score, query/call rates, average costs, skill switching frequency

**Files:** `trainer/decision/grpo_trainer.py`, `trainer/decision/launch_train.py`

---

## 3) SkillBank Agent Trainer (Hard-EM + Gating)

### Milestone S1 — Ingest Recent Rollouts

`ingest_rollouts(D_k)`:
- Loads last N trajectories from Decision Agent buffer
- Extracts `ui_events`, predicates, embeddings
- Produces per-trajectory objects for Stage 1/2

**Files:** `trainer/skillbank/ingest_rollouts.py`

### Milestone S2 — EM Loop Driver

`em_trainer.py`:
- Input: `Bank_k`, `TrajBatch D_k`
- Output: `Bank_{k+1}`, segmentation results, diff report
- Per iteration (1–3 per batch):
  1. Stage 1: ProposeCuts
  2. Stage 2: Decode → segments + diagnostics
  3. Stage 3: Build/Verify Contracts
  4. Stage 4: Update (refine + materialize NEW in MVP)
  5. SkillEval gating on: pass rate, support, discriminability, complexity
  6. Commit or reject; write bank snapshot + diff logs

**Files:** `trainer/skillbank/em_trainer.py`, `trainer/skillbank/stages/*`, `trainer/skillbank/bank_io/*`

### Milestone S3 — Bank Versioning & Rollback

Transactional updates:
- Build `Bank'` in-memory
- Run SkillEval + quick eval
- If accepted → write snapshot `Bank_{k+1}`, rebuild indices
- Else → keep `Bank_k`

**Files:** `trainer/skillbank/bank_io/bank_store.py`, `indices.py`, `diff_logger.py`

### Milestone S4 — Quick Evaluation for Gating

`SkillBankQuickEval` on fixed seeds:
- Re-decode a small holdout batch using `Bank'`
- Metrics: NEW rate, avg margin, contract pass rate distribution, confusion matrix
- If metrics regress beyond thresholds → reject

**File:** `trainer/common/eval_harness.py`

---

## 4) Co-Evolution Training Schedule

Top-level orchestrator (`trainer/launch_coevolution.py`):
1. Run Decision GRPO continuously
2. Every E episodes:
   - Freeze a rollout batch `D_k`
   - Run SkillBank EM trainer → propose `Bank_{k+1}`
   - Gate using fixed-seed eval (decision agent frozen during eval)
   - If accepted, deploy `Bank_{k+1}` to env wrapper/retriever
   - Continue GRPO with new bank
3. Stability rules:
   - Don't update bank every few episodes; use slower cadence (200–1000 episodes)
   - Always keep last-good bank for rollback

---

## 5) Metrics & Dashboards

### Decision Agent Metrics
- Win rate / score / objective completion
- `query_skill` rate, `query_mem` rate, `call_skill` rate
- Average query key length
- `r_env`, `r_follow`, `r_cost`, `r_total` (means and histograms)
- Skill switching rate

### SkillBank Agent Metrics
- Number of skills, NEW pool size
- Contract pass rate per skill
- Avg margin and confusion pairs
- Refine/materialize/merge/split event counts
- Bank size growth rate and churn rate
- Bank diff report per version

---

## 6) Implementation Order

1. Shared schemas + reward shaping contract (unblocks everything)
2. Decision env wrapper "retrieval-as-action" + reward breakdown
3. Decision rollout collector + GRPO loop (even if rough)
4. SkillBank `ingest_rollouts` + `bank_store` versioning
5. Stage 1 propose cuts + Stage 2 decode (MVP)
6. Stage 3 contracts + SkillEval + Stage 4 refine/materialize NEW
7. Co-evolution orchestrator + gating/rollback
8. Optional learners: boundary classifier, tie-breaker

---

## 7) Config Knobs

See `trainer/common/configs/decision_grpo.yaml` and `trainer/common/configs/skillbank_em.yaml` for all hyperparameters.
