# Skill Bank GRPO + Tool-Calling Agent Plan

**Created:** 2026-03-14  
**Updated:** 2026-03-14 — v3: GRPO Wrapper Architecture — reuse existing LLM functions, ~680 lines vs ~3000+  
**Status:** Draft (v3)  
**Depends on:** `SKILLBANK_AUDIT_GAPS.md`, existing Hard-EM pipeline, Decision Agent GRPO trainer

---

## Overview

### Key Design Principle: Wrap, Don't Rebuild

Instead of building 5 custom GRPO training loops with new prompts and I/O formats, we wrap the **existing LLM call points** in the EM pipeline with a generic GRPO sampler. Each LLM call produces G samples instead of 1, evaluates them with a downstream metric, stores (prompt, completion, reward) for deferred training, and returns the best sample to the pipeline. The EM pipeline runs unchanged — it just gets better LLM outputs as GRPO training progresses.

### Scope: 3 GRPO Targets (not 5)

| Target | Existing LLM function | Reward signal | Status |
|--------|----------------------|---------------|--------|
| **Stage 3 CONTRACT** (P0) | `llm_summarize_contract()` in `stage3_mvp/llm_contract.py` | `verify_effects_contract().overall_pass_rate` (CPU-only) | Wrap existing |
| **Stage 4 CURATOR** (P1) | New `filter_candidates()` | `bank_quality_delta` (CPU-only) | New LLM call |
| **Stage 2 SEGMENT** (P1) | `collect_segment_preferences()` in `infer_segmentation/llm_teacher.py` | `SegmentationDiagnostics` (CPU-only) | Wrap existing |
| ~~Stage 1 BOUNDARY~~ | `LLMSignalExtractor._extract_predicates_chunk()` | Indirect (extracts predicates, not boundaries) | **SKIP** — reward too indirect |
| ~~RETRIEVAL~~ | `llm_retrieve_skills()` | Requires env rollout per sample | **SKIP** — use existing decision agent GRPO trainer |

**Why skip BOUNDARY:** The LLM extracts predicates, not boundaries. Boundaries are computed algorithmically from predicates + 5 other signals. GRPO reward (decode quality) would be extremely indirect.

**Why skip RETRIEVAL:** Reward requires full environment rollout per sample (G=4 rollouts per select_skill call). The existing decision agent GRPO trainer already trains the full policy including skill selection — the episode return signal flows through naturally.

**Target codebase:** This plan targets the production `skill_agents/` implementations, NOT the simplified EM trainer stages in `trainer/skillbank/stages/`. The trainer stages are kept as fallback baselines.

| Stage | Production module | Simplified fallback |
|-------|------------------|---------------------|
| 2 — Segmentation | `skill_agents/infer_segmentation/` | `trainer/skillbank/stages/stage2_decode.py` |
| 3 — Contracts | `skill_agents/stage3_mvp/` | `trainer/skillbank/stages/stage3_contracts.py` |
| 4 — Maintenance | `skill_agents/bank_maintenance/` | `trainer/skillbank/stages/stage4_update.py` |

**Model convention:** This project uses a single Qwen model size throughout — **Qwen3-14B** for all components (vLLM serving, LoRA adapters, decision agent, tool-calling). No mixed model sizes. All existing references to Qwen3-8B in the codebase (`skill_agents/lora/config.py`, `trainer/common/configs/skillbank_em.yaml`, etc.) must be updated to Qwen3-14B.

### GRPO-Trained LoRA Adapters

3 LoRA adapters on the shared Qwen3-14B base, trained via the generic GRPO wrapper:

| Adapter | Stage | Wraps existing function | Reward |
|---------|-------|------------------------|--------|
| SEGMENT (LoRA #1) | 2 | `collect_segment_preferences()` in `llm_teacher.py` — generates pairwise skill rankings per segment | `SegmentationDiagnostics` (mean_margin, n_confident, n_new) after scorer rebuild + decode |
| CONTRACT (LoRA #2) | 3 | `llm_summarize_contract()` in `llm_contract.py` — generates effect contract suggestions | `verify_effects_contract().overall_pass_rate` on holdout instances |
| CURATOR (LoRA #3) | 4 | New `filter_candidates()` — approves/vetoes/defers bank mutations | `bank_quality_delta` = q_filtered - q_all |

### Two-Phase GRPO Architecture

```
Phase 1 — Rollout (during EM):
  EM pipeline calls LLM function as normal
  → GRPOCallWrapper intercepts
  → Generates G samples (inference_mode, temperature=0.7)
  → Computes reward per sample (CPU-only downstream metrics)
  → Stores (prompt, completions, rewards) in GRPOBuffer
  → Returns best sample to pipeline (EM continues unchanged)

Phase 2 — Training (after EM step):
  GRPOLoRATrainer reads buffer
  → Recomputes log_probs with gradients enabled
  → Group-normalizes rewards → advantages
  → GRPO policy gradient loss
  → Updates LoRA adapter weights
  → Clears buffer
```

New code: `MultiLoraSkillBankLLM.log_probs()` (~50 lines), `GRPOCallWrapper` (~100 lines), `GRPOBuffer` (~80 lines), `GRPOLoRATrainer` (~150 lines), 3 reward functions (~100 lines), `filter_candidates()` (~200 lines). **Total: ~680 lines** vs ~3000+ in the v2 plan.

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

### Co-Evolution Training Loop (Wrapper Architecture)

```python
grpo_buffer = GRPOBuffer()

for co_evolution_step in range(total_steps):
    # ── Decision agent GRPO (existing — trains full policy incl. skill selection) ──
    rollouts = collect_rollouts(decision_agent, env, skill_bank)
    decision_grpo_update(rollouts)

    if co_evolution_step % bank_update_cadence == 0:
        trajectories = ingest_rollouts(rollouts)

        # ── Phase 1: EM pipeline with GRPO wrappers active ──
        # Each wrapped LLM call:
        #   1. Generates G samples (inference, temp=0.7)
        #   2. Evaluates with CPU-only reward
        #   3. Stores (prompt, completions, rewards) in buffer
        #   4. Returns best sample to pipeline (EM continues unchanged)

        # Stage 2: collect_segment_preferences() → SEGMENT wrapper
        run_segmentation(trajectories, bank)                    # wrapper stores preferences + rewards

        # Stage 3: llm_summarize_contract() → CONTRACT wrapper
        run_contracts(trajectories, bank)                       # wrapper stores contracts + rewards

        # Infrastructure: data quality gate (Stage 4.5)
        run_sub_episode_quality_check()                         # heuristic, no LLM

        # Stage 4: propose → filter (CURATOR wrapper) → execute
        candidates = propose_candidates(bank, bank_maintenance)
        approved = filter_candidates(candidates, bank, vllm)    # CURATOR wrapper stores decisions + rewards
        execute_approved(approved, bank, new_pool, proto_mgr)
        _apply_alias_map(alias_map)

        # Infrastructure: post-Stage 4
        distill_execution_hints()
        update_protocols()                                      # LLM inference (not GRPO)

        # ── Phase 2: GRPO training from buffer ──
        grpo_trainer.train_step(grpo_buffer)                    # one gradient step per adapter
        grpo_buffer.clear()

        # ── SkillEval gating (6 dimensions + holistic) ──
        if not skilleval_passes(bank):
            rollback_bank()
```

---

## 1. GRPO Wrapper Infrastructure

### 1.1 Architecture

Each GRPO-targeted stage uses the shared Qwen3-14B base model with a dedicated LoRA adapter (SEGMENT, CONTRACT, CURATOR). During GRPO training, each adapter is updated independently. The shared base stays frozen. This is the same Qwen3-14B used for vLLM serving and the decision agent — one model size for the entire project.

**The wrapper approach:** Instead of custom training loops per stage, a single `GRPOCallWrapper` class wraps any existing LLM function. During EM rollouts, the wrapper generates G samples, scores them, stores data, and returns the best. After each EM step, `GRPOLoRATrainer` performs one gradient step per adapter.

| Stage | Wrapped function | Adapter role | Existing algorithm (unchanged) |
|-------|-----------------|-------------|-------------------------------|
| 2 | `collect_segment_preferences()` | Generate pairwise preferences | `PreferenceStore` → `PreferenceScorer` (Bradley-Terry) → `SegmentScorer` (6-term) → DP/beam decoder |
| 3 | `llm_summarize_contract()` | Generate contract suggestions (union enrichment) | `learn_effects_contract()` (frequency counting) → `verify_effects_contract()` → `refine_effects_contract()` |
| 4 | `filter_candidates()` (new) | Approve/veto/defer bank mutations | `run_bank_maintenance()` propose → execute pipeline |

**GPU memory note:** Qwen3-14B in bf16 ≈ 28GB base. With LoRA rank 16 and 3 adapters loaded, add ~200MB per adapter. GRPO training adds optimizer states + gradients for adapter params only (~1–2GB total). Fits comfortably on a single 80GB A100.

### 1.1.1 Critical Prerequisite: `log_probs()` Method

`MultiLoraSkillBankLLM` currently only exposes inference (generate text). GRPO training requires per-token log-probabilities with gradients. New method:

```python
def log_probs(self, function: SkillFunction, prompt: str, completion: str) -> torch.Tensor:
    """Compute per-token log-probs of completion given prompt, with gradients."""
    self._activate_adapter(function)
    full_text = prompt + completion
    inputs = self._tokenizer(full_text, return_tensors="pt").to(self._model.device)
    prompt_len = self._tokenizer(prompt, return_tensors="pt")["input_ids"].shape[1]
    with torch.enable_grad():
        outputs = self._model(**inputs)
        logits = outputs.logits[:, prompt_len-1:-1, :]
        target_ids = inputs["input_ids"][:, prompt_len:]
        log_probs = torch.log_softmax(logits, dim=-1)
        return log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
```

### 1.1.2 Generic GRPO Wrapper

```python
class GRPOCallWrapper:
    def __init__(self, adapter: SkillFunction, reward_fn, G: int, buffer: GRPOBuffer):
        self.adapter = adapter
        self.reward_fn = reward_fn
        self.G = G
        self.buffer = buffer

    def wrap(self, original_fn):
        @wraps(original_fn)
        def wrapped(*args, **kwargs):
            kwargs_high_temp = {**kwargs, "temperature": 0.7}
            samples = [original_fn(*args, **kwargs_high_temp) for _ in range(self.G)]
            rewards = [self.reward_fn(s, *args, **kwargs) for s in samples]
            self.buffer.add(self.adapter, args, kwargs, samples, rewards)
            return samples[max(range(len(rewards)), key=lambda i: rewards[i])]
        return wrapped
```

### 1.1.3 Per-Stage Compute Cost

| Stage | Wrapped call | Reward computation | Cost per G=4 sample | Stability |
|-------|-------------|-------------------|---------------------|-----------|
| 3 CONTRACT | `llm_summarize_contract()` | `verify_effects_contract().overall_pass_rate` | ~820ms/skill (CPU-only, no LLM) | Very high |
| 4 CURATOR | `filter_candidates()` | `bank_quality_delta` | ~3s/EM iter (CPU-only) | High |
| 2 SEGMENT | `collect_segment_preferences()` | `SegmentationDiagnostics` | ~12.5s/episode (requires scorer rebuild + decode) | High |

### 1.2 Stage 1 — Boundary Proposal (SKIPPED)

> **Decision:** Do not GRPO-wrap boundary proposal. The LLM function in `skill_agents/boundary_proposal/` extracts predicates (via `LLMSignalExtractor._extract_predicates_chunk()`), NOT boundaries. Boundaries are computed algorithmically from predicates + 5 other signals (surprisal, changepoints, intention tags, done flags, hard events). GRPO reward (decode quality) would be extremely indirect — any signal has to propagate through 6-signal fusion → merge → density control → decode before being measurable. Keep the existing algorithmic pipeline unchanged.

---

### 1.3 Stage 2 — Segmentation (GRPO Wrapper)

**Wrapped function:** `collect_segment_preferences()` in `skill_agents/infer_segmentation/llm_teacher.py`

**Current state:** `skill_agents/infer_segmentation/` implements a preference-learning pipeline:
1. **LLM teacher** (`llm_teacher.py`) ranks skills per segment → pairwise preferences
2. **PreferenceStore** accumulates `PreferenceExample` objects (segment + transition preferences)
3. **PreferenceScorer** trains Bradley-Terry model on preferences → learned behavior-fit and transition scores
4. **SegmentScorer** (`scorer.py`) computes a 6-term composite score per (segment, skill) pair
5. **Decoder** (Viterbi DP or beam search with `beam_width=16`) finds the optimal segmentation
6. **Active learning** loop: 3 iterations of uncertain-segment queries (`margin < 1.0`, max 5 per iteration), retrain, re-decode

**What the wrapper does:** Wraps `collect_segment_preferences()` at the **batch level** (per episode). For each episode, generates G=4 complete sets of preferences, feeds each through scorer rebuild + decode, measures `SegmentationDiagnostics`, stores data, returns the best. The rest of the pipeline (PreferenceStore, PreferenceScorer, decoder, active learning) runs unchanged.

**Reward function:**

| Component | Formula | Weight | Rationale |
|-----------|---------|--------|-----------|
| `r_margin` | `mean(seg.margin for seg in result.segments)` | 0.30 | Confident assignments |
| `r_new_penalty` | `-0.5 * new_rate` | 0.20 | Discourage excessive NEW labels |
| `r_label_entropy` | `-mean(seg.label_entropy for seg in result.segments)` | 0.20 | Low entropy = unambiguous |
| `r_compat_margin` | `mean(seg.compat_margin for seg in result.segments)` | 0.15 | Contract compatibility gap between top-1 and top-2 |
| `r_confusion_penalty` | `-mean(confusion_overlap for confuser_pairs)` | 0.15 | Different skills should have different effects |

**Compute cost per G=4:** ~12.5s/episode (4× scorer rebuild + decode). All CPU-only — no additional LLM inference for reward.

**Complexity note:** This is the most complex wrapper because reward requires rebuilding the scorer and re-running the decoder per sample. Do this stage last.

**Files to modify:**

- [ ] `trainer/skillbank/grpo/rewards.py` — `segmentation_reward()` function
- [ ] `trainer/skillbank/grpo/wrappers.py` — register `collect_segment_preferences` wrapper
- [ ] No modifications needed to `llm_teacher.py`, `preference.py`, `scorer.py`, or decoders — the wrapper sits outside these

---

### 1.4 Stage 3 — Contract Learning (GRPO Wrapper — P0, Implement First)

**Wrapped function:** `llm_summarize_contract()` in `skill_agents/stage3_mvp/llm_contract.py`

**Why P0:** Easiest stage to wrap. The existing `llm_summarize_contract()` already produces contract JSON. Reward is `verify_effects_contract().overall_pass_rate` — CPU-only, ~200ms/call, deterministic. Very high signal-to-noise ratio.

**Current state:** `skill_agents/stage3_mvp/` implements a 7-step contract pipeline. `llm_summarize_contract()` takes segment records + frequency stats and returns `{"eff_add": [...], "eff_del": [...], "eff_event": [...]}`. Currently used as union-only enrichment in the EM trainer path.

**What the wrapper does:** Intercepts `llm_summarize_contract()` calls. Generates G=4 contract suggestions, evaluates each against holdout instances via `verify_effects_contract()`, stores data, returns the best.

**Reward function (simplified from v2):**

| Component | Formula | Weight | Rationale |
|-----------|---------|--------|-----------|
| `r_holdout_pass` | `verify_effects_contract(contract, holdout_instances).overall_pass_rate` | 0.50 | Contract must generalize |
| `r_per_literal_quality` | `mean(success_rate for lit in all_literals)` | 0.25 | Every literal should be reliable |
| `r_sparsity` | `-max(0, n_literals - budget) / budget` | 0.15 | Prevent bloated contracts |
| `r_coverage` | `n_instances_covered / n_total_instances` | 0.10 | Contract should explain most instances |

**Compute cost per G=4:** ~820ms/skill (4× `verify_effects_contract()` at ~200ms each). All CPU-only.

**Files to modify:**

- [ ] `trainer/skillbank/grpo/rewards.py` — `contract_reward()` function
- [ ] `trainer/skillbank/grpo/wrappers.py` — register `llm_summarize_contract` wrapper
- [ ] No modifications to the 7-step pipeline — wrapper sits outside `run_stage3_mvp.py`

---

### 1.5 Shared GRPO Infrastructure (New Code)

New files to create:

- [ ] `trainer/skillbank/grpo/__init__.py` — package init
- [ ] `trainer/skillbank/grpo/buffer.py` — `GRPOBuffer` dataclass: stores `(adapter, prompt, completions, rewards)` tuples per stage (~80 lines)
- [ ] `trainer/skillbank/grpo/wrapper.py` — `GRPOCallWrapper` class: generic function wrapper that samples G times, evaluates rewards, stores in buffer (~100 lines)
- [ ] `trainer/skillbank/grpo/trainer.py` — `GRPOLoRATrainer`: reads buffer, calls `log_probs()`, computes GRPO loss, updates LoRA weights (~150 lines)
- [ ] `trainer/skillbank/grpo/rewards.py` — `contract_reward()`, `curator_reward()`, `segmentation_reward()` (~100 lines)
- [ ] `trainer/skillbank/grpo/config.py` — per-stage GRPO hyperparameters
- [ ] `trainer/common/configs/skillbank_grpo.yaml` — unified config file

Existing files to modify:

- [ ] `skill_agents/lora/model.py` — add `log_probs()` method to `MultiLoraSkillBankLLM` (~50 lines)
- [ ] `skill_agents/lora/config.py` — change `base_model_name_or_path` default from `"Qwen/Qwen3-8B"` to `"Qwen/Qwen3-14B"`
- [ ] `trainer/common/configs/skillbank_em.yaml` — change `lora.base_model_name_or_path` from `"Qwen/Qwen3-8B"` to `"Qwen/Qwen3-14B"`

**Hyperparameter defaults:**

```yaml
grpo:
  stage3_contract:
    group_size: 4
    clip_ratio: 0.2
    kl_coeff: 0.05
    lr: 5.0e-5
    epochs_per_batch: 2

  stage4_curator:
    group_size: 4
    clip_ratio: 0.2
    kl_coeff: 0.05
    lr: 5.0e-5
    epochs_per_batch: 2

  stage2_segment:
    group_size: 4
    clip_ratio: 0.2
    kl_coeff: 0.02
    lr: 3.0e-5
    epochs_per_batch: 3
```

---

## 2. Stage 4: LLM-Advised Bank Maintenance (GRPO Wrapper — P1)

### 2.1 GRPO Wrapper Summary

**New LLM function:** `filter_candidates()` (does not exist yet — must be created)

Unlike Stages 2 and 3 which wrap existing LLM functions, Stage 4 requires a **new** `filter_candidates()` call. The algorithmic pipeline proposes candidate mutations; the CURATOR LoRA reviews them and returns approve/veto/defer decisions.

**Reward function:**

| Component | Formula | Weight | Rationale |
|-----------|---------|--------|-----------|
| `r_quality_delta` | `q_filtered - q_all` (where q = mean pass_rate of affected skills) | 0.50 | Filtering should improve bank quality |
| `r_conservative_bonus` | `+0.1 * n_deferred / n_total` | 0.20 | Encourage caution (prefer defer over bad approve) |
| `r_action_diversity` | `-entropy(action_type_distribution)` if all same type | 0.15 | Don't always approve/veto everything |
| `r_veto_precision` | `ratio of vetoed items that would have lowered quality` | 0.15 | Vetoes should be justified |

**Compute cost per G=4:** ~3s/EM iteration (4× bank quality computation). All CPU-only.

### 2.2 Design Principle

**Production module:** `skill_agents/bank_maintenance/` — implements `run_bank_maintenance()` with a pipeline of: profile building → index construction → split → merge → refine → local re-decode.

Stage 4 mutates the skill bank through five actions — **refine, merge, split, materialize, promote**. The architecture is **propose-filter-execute**:

- **Propose:** The algorithmic `run_bank_maintenance()` builds `SkillProfile` per skill and generates a ranked candidate action list.
- **Filter:** A single LLM call (CURATOR LoRA) reviews candidates against `SkillProfile` diagnostics and returns approved/vetoed/deferred decisions. **This is the GRPO wrapper target.**
- **Execute:** The algorithm executes approved actions using existing deterministic functions.

The LLM's role is narrow: contextual judgment on "should we do this?" The algorithm handles "what could we do?" and "how do we do it?"

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

## 3. `select_skill` in the Decision Agent (SKIPPED)

> **Decision:** Do not add a RETRIEVAL LoRA adapter. The existing decision agent GRPO trainer already trains the full policy including skill selection. The episode return signal (`r_env + r_follow + r_cost + r_tool`) flows through `select_skill` naturally. Adding a separate RETRIEVAL adapter would require full environment rollouts per GRPO sample (~30s/episode), making it prohibitively expensive. The existing `SkillQueryEngine.select()` fixed scoring + GRPO-trained query generation is sufficient for v1.

---

## Dependency Graph (Wrapper Architecture)

```
                    ┌─────────────────────┐
                    │  Cold-start bank     │
                    │  (GPT-5.4 rollouts)  │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │  §1.1: Prerequisites │
                    │  log_probs() method  │
                    │  GRPOBuffer          │
                    │  GRPOCallWrapper     │
                    │  GRPOLoRATrainer     │
                    └──────────┬──────────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                │
    ┌─────────▼──────┐  ┌─────▼──────┐  ┌──────▼──────────┐
    │ §1.4: Stage 3  │  │ §2: Stage 4│  │ §1.3: Stage 2   │
    │ CONTRACT       │  │ CURATOR    │  │ SEGMENT          │
    │ (wrap existing │  │ (new LLM   │  │ (wrap existing   │
    │  llm_summarize │  │  filter_   │  │  collect_segment │
    │  _contract)    │  │  candidates│  │  _preferences)   │
    │                │  │  + wrap)   │  │                  │
    │ P0: Do first   │  │ P1        │  │ P1: Do last      │
    └────────┬───────┘  └─────┬──────┘  └──────┬──────────┘
             │                │                │
             └────────────────┼────────────────┘
                              │
                    ┌─────────▼─────────┐
                    │  Co-evolution loop │
                    │  Phase 1: wrappers │
                    │  Phase 2: training │
                    └───────────────────┘

    All components use Qwen3-14B. 3 LoRA adapters: SEGMENT, CONTRACT, CURATOR.
    ~680 lines of new code total.
```

## Implementation Priority

| Priority | Item | Effort | Rationale |
|----------|------|--------|-----------|
| **P0** | `log_probs()` on `MultiLoraSkillBankLLM` | 0.5 day | Everything depends on this |
| **P0** | `GRPOCallWrapper` + `GRPOBuffer` + `GRPOLoRATrainer` | 1.5 days | Generic infrastructure for all stages |
| **P0** | Stage 3 CONTRACT wrapper + `contract_reward()` | 1 day | Easiest stage — validate the wrapper approach |
| **P1** | Stage 4 CURATOR `filter_candidates()` + wrapper | 1.5 days | New LLM call, but simple reward |
| **P1** | Stage 2 SEGMENT wrapper + `segmentation_reward()` | 2 days | Most complex reward (scorer rebuild + decode) — do last |

**Total: ~6.5 days** (vs ~17 days in v2 plan)

---

## Open Questions

1. **GRPO batch size vs EM cadence**: Currently EM runs every 500 decision-agent episodes. With G=4 sampling per LLM call, each EM step is ~4× more LLM-inference-heavy. Should we increase cadence to 1000?

2. **HuggingFace native vs VERL/TRL for `GRPOLoRATrainer`**: Option A (HF native) is simpler (~150 lines, direct PyTorch optimizer on LoRA params). Option B (VERL/TRL) offers distributed training and memory optimization. **Recommend Option A for v1.**

3. **GPU memory for dual-mode `MultiLoraSkillBankLLM`**: `log_probs()` requires `torch.enable_grad()` which increases memory. Need to verify Qwen3-14B + LoRA gradient computation fits in remaining GPU memory alongside vLLM KV cache. May need to pause vLLM during training phase.

4. **Model mismatch between cold-start and co-evolution**: Cold-start extraction uses GPT-5.4; co-evolution uses Qwen3-14B. Should we run a calibration pass where Qwen3-14B re-processes the cold-start bank?

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
