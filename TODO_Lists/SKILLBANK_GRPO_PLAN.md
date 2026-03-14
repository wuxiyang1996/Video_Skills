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

## 2. Stage 4: Tool-Calling Bank Maintenance Agent

### 2.1 Why Tool-Calling

Stage 4 (bank update: refine, materialize, merge, split) is fundamentally different from Stages 1–3:

- **Combinatorial action space**: merge(A,B), split(C), materialize(cluster_7) — not a single structured output
- **Multi-step reasoning**: inspect → trial → evaluate → accept/reject → next action
- **Sparse execution**: runs once per EM batch, not per trajectory

A tool-calling agent fits naturally: the LLM proposes actions, external tools execute them on a sandboxed bank copy, and the LLM observes the quality delta before committing.

### 2.2 Model & Serving

**Qwen3-14B via vLLM** — same model as everything else in the project.

Stage 4 uses the vLLM OpenAI-compatible `/v1/chat/completions` endpoint (not the single-turn `MultiLoraSkillBankLLM.generate()` interface) because tool-calling requires multi-turn chat. The vLLM server is already in the stack (`run_qwen3_skillbank_agent.sh` launches it). During training, keep it alive.

### 2.3 Tool Definitions

```python
STAGE4_TOOLS = [
    {
        "name": "inspect_skill",
        "description": "Get detailed diagnostics for a skill: pass_rate, n_instances, contract effects, confuser pairs, failure signatures.",
        "parameters": {"skill_id": {"type": "string"}}
    },
    {
        "name": "trial_merge",
        "description": "Trial-merge two skills on a sandboxed bank copy. Returns quality delta, new pass_rate, and merged contract preview.",
        "parameters": {
            "skill_a": {"type": "string"},
            "skill_b": {"type": "string"}
        }
    },
    {
        "name": "trial_split",
        "description": "Trial-split a weak skill into sub-skills by re-clustering its instances. Returns child pass_rates and quality delta.",
        "parameters": {"skill_id": {"type": "string"}}
    },
    {
        "name": "trial_materialize",
        "description": "Trial-materialize a NEW cluster as a real skill. Returns proposed contract and quality delta.",
        "parameters": {"cluster_id": {"type": "string"}}
    },
    {
        "name": "accept_action",
        "description": "Commit a previously trialed action to the real bank.",
        "parameters": {"action_id": {"type": "string"}}
    },
    {
        "name": "reject_action",
        "description": "Discard a previously trialed action.",
        "parameters": {"action_id": {"type": "string"}}
    },
    {
        "name": "finish",
        "description": "End the bank maintenance session. No more actions.",
        "parameters": {}
    }
]
```

### 2.4 How Tools Work Internally

Each `trial_*` tool:

1. Deep-copies the current bank
2. Applies the mutation (merge/split/materialize) on the copy
3. Re-runs `decode_batch()` on a sample of recent trajectories (CPU, fast)
4. Re-runs `learn_contracts()` on the re-decoded segments
5. Computes `bank_quality = mean_pass_rate * (1 - new_rate) * discriminability`
6. Returns structured observation:

```json
{
    "action_id": "act_3",
    "action_type": "merge",
    "skills_involved": ["navigate_pot", "go_to_pot"],
    "quality_before": 0.72,
    "quality_after": 0.76,
    "quality_delta": "+0.04 (IMPROVED)",
    "new_pass_rate": 0.81,
    "new_bank_size": 41,
    "details": "Merged contract: eff_add={near_pot, facing_pot}, eff_del={}"
}
```

The LLM sees "+0.04 (IMPROVED)" and decides whether to `accept_action("act_3")` or `reject_action("act_3")`. This keeps the LLM's role as **proposer/navigator** while all verification is external and deterministic.

### 2.5 Agent Loop

```python
class Stage4Agent:
    def __init__(self, vllm_client, bank, trajectories, decode_fn, contract_fn):
        self.client = vllm_client  # OpenAI-compatible client → vLLM
        self.bank = bank
        self.trajectories = trajectories
        self._pending_actions = {}
        self._action_counter = 0

    def run(self, max_turns: int = 20) -> UpdateResult:
        diagnostics = self._compute_bank_diagnostics()
        messages = [
            {"role": "system", "content": STAGE4_SYSTEM_PROMPT},
            {"role": "user", "content": f"Bank diagnostics:\n{json.dumps(diagnostics, indent=2)}"}
        ]

        for turn in range(max_turns):
            response = self.client.chat.completions.create(
                model="Qwen/Qwen3-14B",
                messages=messages,
                tools=STAGE4_TOOLS,
                temperature=0.2,
            )
            msg = response.choices[0].message

            if msg.tool_calls is None or len(msg.tool_calls) == 0:
                break  # LLM chose to finish (or no tool call)

            for tool_call in msg.tool_calls:
                fn_name = tool_call.function.name
                fn_args = json.loads(tool_call.function.arguments)

                if fn_name == "finish":
                    return self._compile_result()

                observation = self._execute_tool(fn_name, fn_args)
                messages.append(msg)  # assistant message with tool_call
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(observation)
                })

        return self._compile_result()
```

### 2.6 Diagnostics Prompt

The initial prompt gives the LLM a dashboard:

```
Bank: 42 skills, 456 segments, mean_pass_rate=0.74, new_rate=0.08

Weak skills (pass_rate < 0.70):
  S12_navigate_pot: pass_rate=0.55, 8 instances, failures: miss_add:near_pot (5x)
  S23_place_onion:  pass_rate=0.62, 12 instances, failures: miss_del:holding_onion (4x)

Merge candidates (jaccard > 0.80):
  (S05_go_to_pot, S12_navigate_pot): jaccard=0.87, combined_instances=15

Split candidates (pass_rate < 0.70, instances >= 6):
  S23_place_onion: pass_rate=0.62, 12 instances — failure signatures suggest 2 sub-clusters

NEW clusters ready for materialization:
  cluster_7: 8 segments, signature=A:near_pot,facing_pot|D:, consistency=0.85

Your job: inspect, trial-merge/split/materialize, accept good changes, reject bad ones, then finish.
```

### 2.7 Files to Create/Modify

- [ ] `trainer/skillbank/stage4_agent.py` — `Stage4Agent` class, tool execution, diagnostics, quality computation
- [ ] `trainer/skillbank/stage4_tools.py` — tool definitions, `_execute_tool()` dispatch, sandboxed bank operations
- [ ] `trainer/skillbank/stage4_prompts.py` — system prompt, diagnostics formatter
- [ ] Modify `trainer/skillbank/em_trainer.py` — replace `run_update()` call with `Stage4Agent.run()`
- [ ] Modify `trainer/common/configs/skillbank_em.yaml` — add `stage4_agent` config section (max_turns, vllm_url, temperature)
- [ ] Modify `scripts/run_qwen3_skillbank_agent.sh` — ensure vLLM stays alive for Stage 4 during training
- [ ] Keep `trainer/skillbank/stages/stage4_update.py` as algorithmic fallback

### 2.8 Guardrails

| Guardrail | Implementation |
|-----------|---------------|
| **Max actions per session** | `max_turns=20`, hard cap in agent loop |
| **Quality gate** | SkillEval runs after Stage 4 completes — rollback if bank quality drops |
| **Idempotent trials** | Each `trial_*` operates on a deep copy, never mutates the real bank until `accept_action` |
| **Timeout** | If LLM doesn't call `finish` within max_turns, auto-finish with whatever was accepted |
| **Fallback** | If vLLM is unavailable, fall back to algorithmic `run_update()` |

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
    │ GRPO adapters  │  │ Tool-call  │  │ GRPO + reranker │
    │ (Qwen3-14B     │  │ (Qwen3-14B │  │ (Qwen3-14B     │
    │  + LoRA)       │  │  via vLLM) │  │  + LoRA)       │
    └────────┬───────┘  └─────┬──────┘  └──────┬─────────┘
             │                │                │
             └────────────────┼────────────────┘
                              │
                    ┌─────────▼─────────┐
                    │  Co-evolution loop │
                    │  (launch_coevo.py) │
                    └───────────────────┘

    All components use Qwen3-14B. No mixed model sizes.
```

**§1 and §3 can be developed in parallel** — they share the Qwen3-14B LoRA infrastructure but operate on different adapters (BOUNDARY/SEGMENT/CONTRACT vs RETRIEVAL).

**§2 depends on §1** — the tool-calling agent needs decode/contract functions to work for trial evaluations, but those already exist. What §2 needs from §1 is the GRPO-trained variants for better quality deltas, but it can start with the existing frequency-based Stage 2/3 for trials.

## Implementation Priority

| Priority | Item | Effort | Rationale |
|----------|------|--------|-----------|
| **P0** | §1.5 — Shared GRPO infrastructure (`grpo_lora_updater.py`) | 2 days | Everything else depends on this |
| **P0** | §3.4 — Remove hard gate + add RETRIEVAL re-ranker | 2 days | Simplest win, immediately improves decision agent |
| **P1** | §1.4 — Stage 3 GRPO (CONTRACT adapter) | 3 days | Highest impact: contracts feed into both decode scoring and follow-shaping reward |
| **P1** | §2 — Stage 4 tool-calling agent | 4 days | Most complex new component |
| **P2** | §1.3 — Stage 2 GRPO (SEGMENT adapter) | 3 days | Depends on §1.4 for reward signal |
| **P2** | §1.2 — Stage 1 GRPO (BOUNDARY adapter) | 2 days | Depends on §1.3 for reward signal (decode quality) |

---

## Open Questions

1. **GRPO batch size vs EM cadence**: Currently EM runs every 500 decision-agent episodes. With GRPO on Stages 1–3, each EM step is more expensive (G parallel generations + evaluation). Should we increase cadence to 1000?

2. **Shared vs separate GRPO optimizers**: Each LoRA adapter has its own optimizer state. Should we use a single learning rate schedule across all 3 stages, or tune independently?

3. **Stage 4 budget**: How many tool calls should the Stage 4 agent be allowed per session? Too few = can't explore enough mutations. Too many = expensive vLLM calls.

4. **RETRIEVAL adapter training data**: The RETRIEVAL adapter needs (query, candidates, selected, outcome_reward) tuples. During GRPO, these come from rollouts. But initially the adapter has no training — should we warm-start from the `SkillQueryEngine.select()` rankings (distillation)?

5. **GPU memory for LoRA training on 14B**: Qwen3-14B in bf16 ≈ 28GB. With LoRA (rank 16), training adds optimizer states + gradients for adapter params only (~1–2GB). This fits in a single 80GB A100 but requires careful `gpu-memory-utilization` tuning when vLLM is also serving on the same GPU. Consider: (a) dedicated GPU for LoRA training separate from vLLM serving, or (b) time-slicing — pause vLLM during GRPO LoRA updates, resume for Stage 4 tool-calling.

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
