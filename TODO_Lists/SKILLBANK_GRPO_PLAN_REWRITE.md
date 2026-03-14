# GRPO Plan Rewrite — Gap Analysis & Action Items

**Created:** 2026-03-14  
**Status:** Planning  
**Depends on:** `SKILLBANK_GRPO_PLAN.md` (current draft), `SKILLBANK_AUDIT_GAPS.md`  
**Goal:** Rewrite `SKILLBANK_GRPO_PLAN.md` to target the production `skill_agents/` implementations instead of the simplified EM trainer stages in `trainer/skillbank/stages/`.

---

## What the New GRPO Plan Does

GRPO trains the backend Qwen3-14B LLM (via LoRA adapters) for each stage where the pipeline calls an LLM to generate structured output. It does NOT replace the algorithmic/heuristic functions — those stay as infrastructure around the GRPO-trained stages.

### GRPO-trained LoRA adapters (LLM generation tasks)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     GRPO Training Targets (LLM tasks)                      │
├──────────────┬──────────────────────────────────────────────────────────────┤
│ BOUNDARY     │ Stage 1: Generate boundary cut proposals from multi-signal  │
│ (LoRA #1)    │ trajectory input (predicates, surprisal, changepoints,      │
│              │ intention tags, events)                                      │
│              │ Replaces/augments: propose_boundary_candidates()             │
│              │ Reward: Stage 2 decode quality (margin, known_rate)          │
├──────────────┼──────────────────────────────────────────────────────────────┤
│ SEGMENT      │ Stage 2: Generate segment-skill rankings/preferences that   │
│ (LoRA #2)    │ feed into PreferenceStore + Bradley-Terry scorer             │
│              │ Replaces: LLM teacher (collect_segment_preferences,          │
│              │   collect_transition_preferences, collect_uncertain_prefs)   │
│              │ Reward: downstream contract pass rate + decision agent       │
│              │   follow score                                               │
├──────────────┼──────────────────────────────────────────────────────────────┤
│ CONTRACT     │ Stage 3: Generate effects contracts from segment evidence   │
│ (LoRA #3)    │ Augments: frequency-based learn_effects_contract()           │
│              │   (union enrichment, not full replacement)                   │
│              │ Reward: holdout verification pass rate + decision agent      │
│              │   follow score                                               │
├──────────────┼──────────────────────────────────────────────────────────────┤
│ CURATOR      │ Stage 4: Single-turn filter — approve/veto/defer candidate  │
│ (LoRA #4)    │ bank mutations (refine, merge, split, materialize, promote) │
│              │ Augments: algorithmic propose_candidates() from              │
│              │   bank_maintenance/ (SkillProfile, indices, triggers)        │
│              │ Reward: bank_quality_delta (filtered vs unfiltered)          │
├──────────────┼──────────────────────────────────────────────────────────────┤
│ RETRIEVAL    │ select_skill: Re-rank skill candidates for decision agent   │
│ (LoRA #5)    │ Augments: SkillQueryEngine.select() fixed scoring           │
│              │ Reward: episode return (r_env + r_follow + r_cost + r_tool)  │
└──────────────┴──────────────────────────────────────────────────────────────┘
```

### Non-GRPO functions (infrastructure in the training loop)

These `SkillBankAgent` functions are NOT LLM generation tasks. They stay as-is and run as infrastructure around the GRPO-trained stages:

| Function | Role in training loop | Why not GRPO |
|----------|----------------------|--------------|
| `run_sub_episode_quality_check()` | **Data quality gate** — runs between Stage 3 and Stage 4; scores sub-episodes, retires depleted skills. Ensures GRPO trains on clean data. | Heuristic scoring (outcome_reward, follow_through, consistency, compactness). No LLM. |
| `distill_execution_hints()` | **Post-Stage 4** — derives termination cues, failure modes, micro-plans from successful sub-episodes. Keeps decision agent hints fresh. | Deterministic extraction from sub-episode patterns. No LLM. |
| `_apply_alias_map()` | **Post-merge bookkeeping** — relabels all segments referencing merged-away skills. | Mechanical string replacement. No LLM. |
| `_get_or_create_preference_store()` | **Stage 2 integration point** — PreferenceStore accumulates pairwise preferences. GRPO-generated preferences feed INTO this store. | Infrastructure for Stage 2, not a generation task. |
| `run_until_stable()` / `_is_converged()` | **Outer loop** — convergence detection (pass_rate plateau, new_rate stable). | Orchestration logic. GRPO has its own stopping criteria. |
| `_take_snapshot()` | **Logging** — records iteration metrics for monitoring. | Pure instrumentation. |
| `load()` / `save()` | **Persistence** — bank + preference store + iteration history. | I/O. |
| `form_proto_skills()` / `verify_proto_skills()` / `promote_proto_skills()` | **Proto-skill pipeline** — already in the plan as the materialize/promote actions. Execution logic is algorithmic; CURATOR LoRA decides whether to approve. | The LLM role is in the CURATOR filter (approve/veto), not in the formation/verification logic itself. |
| `query_by_effects()` | **Retrieval** — effect-based skill lookup. | Covered conceptually by RETRIEVAL LoRA (§3 select_skill). |

### Not a GRPO target: `update_protocols()`

`update_protocols()` uses LLM to synthesize Protocol objects (preconditions, steps, success/abort criteria) from high-quality sub-episodes. It stays as plain LLM inference (not GRPO-trained) because:

- **Summarization, not judgment** — extracting steps from evidence is a straightforward task; basic prompting handles it fine.
- **Reward signal too indirect** — you'd need to measure decision agent follow-through across many episodes after a protocol update; too slow, too noisy, too many confounders.
- **Data volume too small** — protocols only update when ≥3 high-quality sub-episodes accumulate per skill; not enough rollouts for GRPO.
- **Self-correcting** — bad protocols → low follow-through → low quality scores → next `update_protocols()` recomputes from better evidence. The feedback loop already exists without GRPO.
- **Deterministic fallback works** — the no-LLM path in `pipeline.py` turns contract effects into functional steps. Good enough for the decision agent.

### Co-evolution training loop with non-GRPO functions included

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

        # ── Infrastructure: data quality gate ──
        run_sub_episode_quality_check()                         # heuristic, no LLM

        # ── Stage 4: algorithm proposes, LLM filters ──
        candidates = propose_candidates(bank, bank_maintenance) # algorithmic (SkillProfile, indices)
        approved = filter_candidates(candidates, bank, vllm)    # CURATOR LoRA
        execute_approved(approved, bank, new_pool, proto_mgr)   # algorithmic
        _apply_alias_map(alias_map)                             # bookkeeping

        # ── Infrastructure: post-Stage 4 ──
        distill_execution_hints()                               # heuristic, no LLM
        update_protocols()                                      # LLM inference (not GRPO-trained, see note below)

        # ── SkillEval gating ──
        if not skilleval_passes(bank):
            rollback_bank()
```

---

## Key Decision

The current GRPO plan targets the EM trainer's simplified stage implementations:

| Stage | Plan targets (trainer/) | Should target (skill_agents/) |
|-------|------------------------|-------------------------------|
| 1 | `trainer/skillbank/stages/stage1_propose_cuts.py` | `skill_agents/boundary_proposal/` |
| 2 | `trainer/skillbank/stages/stage2_decode.py` | `skill_agents/infer_segmentation/` |
| 3 | `trainer/skillbank/stages/stage3_contracts.py` | `skill_agents/stage3_mvp/` |
| 4 | `trainer/skillbank/stages/stage4_update.py` | `skill_agents/bank_maintenance/` |

The `skill_agents/` versions have significantly richer data structures, scoring functions, and pipelines. The rewrite must align GRPO I/O descriptions, reward functions, and training loops with these production implementations.

---

## Stage 1 — Boundary Proposal

### Gap Table

| What the codebase has | What the plan says | What needs to change |
|----------------------|-------------------|---------------------|
| **6+ signal types**: predicate changes, surprisal, changepoint scores (CUSUM / sliding-window cosine on embeddings), intention-tag transitions (20+ aliases), done flags, hard events (reward spikes, phase transitions) | Only mentions predicates, surprisal, action transitions | Add all signal types to the I/O description; GRPO input prompt must include or summarize all available signals |
| **LLM predicate extraction** (`llm_extractor.py`): LLM extracts predicates from NL state descriptions per chunk, optional significance filter | Says "no learning" in Stage 1 | Acknowledge existing LLM usage; clarify that GRPO replaces/augments the signal-combination logic, not the individual extractors |
| **BoundaryPreferenceScorer** (`boundary_preference.py`): Bradley-Terry preferences on boundaries; `filter_candidates()` + Stage 2 `decoding_bonus()` | Not mentioned | Add to plan: GRPO can generate preference pairs from rollouts that feed into BoundaryPreferenceScorer; describe the feedback loop |
| **Richer output**: `BoundaryCandidate` with `center`, `half_window`, `source` attribution | Assumes bare `[12, 27, 45]` | Update I/O example to show `BoundaryCandidate` list; GRPO output should include window and source |
| **ProposalConfig**: `merge_radius=5`, `window_half_width=2`, `surprisal_std_factor=2.0`, `soft_max_per_minute=20`, `tag_min_segment_len=3`, etc. | No thresholds documented | Document key thresholds that constrain GRPO output validation |
| **Two implementations**: `stage1_propose_cuts.py` (simple) vs `boundary_proposal/` (rich) | Doesn't clarify which one | Explicitly target `boundary_proposal/`; note the simple version as fallback |

### Key files

- `skill_agents/boundary_proposal/proposal.py` — `propose_boundary_candidates()`, `propose_from_episode()`
- `skill_agents/boundary_proposal/boundary_preference.py` — `BoundaryPreferenceScorer`, `BoundaryPreferenceConfig`
- `skill_agents/boundary_proposal/changepoint.py` — `compute_changepoint_scores()`, CUSUM
- `skill_agents/boundary_proposal/llm_extractor.py` — `LLMSignalExtractor`
- `skill_agents/boundary_proposal/signal_extractors.py` — `SignalExtractor`, `extract_signals()`
- `skill_agents/boundary_proposal/episode_adapter.py` — `propose_from_episode()`

### Action items

- [ ] Rewrite §1.2 I/O table: input includes all 6+ signal types (not just predicates + surprisal)
- [ ] Rewrite §1.2 I/O example: show realistic multi-signal input with changepoints, intention tags, events
- [ ] Update output format: `List[BoundaryCandidate]` with `center`, `half_window`, `source`
- [ ] Add BoundaryPreferenceScorer integration: GRPO rollouts → preference pairs → scorer → Stage 2 bonus
- [ ] Document ProposalConfig thresholds relevant to GRPO output validation
- [ ] Clarify: GRPO targets `boundary_proposal/` (rich pipeline), not `stage1_propose_cuts.py`
- [ ] Design decision: does GRPO replace `propose_boundary_candidates()` entirely, or generate one of the signal channels that feeds into it?

---

## Stage 2 — Decode / Segmentation

### Gap Table

| What the codebase has | What the plan says | What needs to change |
|----------------------|-------------------|---------------------|
| **Preference-learning pipeline**: LLM teacher ranks segments → pairwise preferences → `PreferenceStore` → Bradley-Terry `PreferenceScorer.train()` → trained scorer feeds decoder | Assumes SEGMENT adapter directly generates skill assignments | Describe the full preference pipeline; decide where GRPO plugs in |
| **Five-term scoring** in `SegmentScorer.score()`: `behavior_fit + duration_prior + transition_prior + contract_compat + boundary_preference` with configurable weights | Mentions "effect matching + surprisal" (surprisal is Stage 1, not Stage 2) | Document the actual scoring function with all 5 terms and their weights |
| **Active learning loop**: 3 iterations of uncertain-segment queries (`margin < 1.0`, max 5 queries per iteration), retrain scorer, re-decode | Not mentioned | Add active learning to the training loop description |
| **Transition preferences**: LLM ranks `prev_skill → next_skill` transitions; PreferenceScorer learns transition scores | Not mentioned | Add transition preferences to Stage 2 I/O and reward |
| **Beam decoder** (`beam_decoder.py`): alternative to Viterbi DP, `beam_width=16`, early stopping | Only DP mentioned | Add beam decoder as option; note tradeoffs (exact vs approximate) |
| **`suggest_skill_name()`**: LLM suggests human-readable names for `__NEW__` segments | Not mentioned | Add to Stage 2 post-processing |
| **Contract feedback loop**: `compat_fn` from bank feeds Stage 3 effects back into Stage 2 scoring | Not mentioned | Document the `compat_fn` feedback loop |
| **Rich diagnostics**: `SegmentDiagnostic` with `margin`, `label_entropy`, `compat_margin`, `label_category`, `boundary_confidence`; `SegmentationResult` with `uncertain_segments()`, `confident_segments()`, `new_segments()` | Plan uses only `margin` | Use richer diagnostics in GRPO reward computation |
| **Data structures**: `SegmentationResult` / `SegmentDiagnostic` / `SkillCandidate` with `breakdown` dict | Plan assumes `DecodeResult` / `DecodedSegment` from EM trainer | Align all data structure references |
| **PreferenceStore persistence**: store saves/loads preferences across iterations | Not mentioned | GRPO should accumulate preferences across training |
| **Two implementations**: `stage2_decode.py` (EM trainer) vs `infer_segmentation/` (rich) | Doesn't clarify | Target `infer_segmentation/`; note EM trainer version as simplified fallback |

### Key design decision

Where does GRPO plug into the preference-learning pipeline?

**Option A — GRPO replaces the LLM teacher:**
GRPO generates pairwise preferences (which skill is better for this segment?) instead of the LLM teacher. PreferenceStore and Bradley-Terry scorer stay. GRPO reward = downstream episode return. This is the most natural fit because the LLM teacher already produces rankings that get converted to preferences.

**Option B — GRPO replaces the PreferenceScorer:**
GRPO directly trains the SEGMENT adapter to output scores, bypassing Bradley-Terry. Loses the preference-learning structure but is simpler.

**Option C — GRPO sits on top as re-ranker (current plan):**
PreferenceScorer + DP produces candidates, SEGMENT adapter re-ranks. Keeps existing pipeline intact, adds GRPO layer.

Recommendation: **Option A** — most aligned with existing architecture, lowest integration risk.

### Key files

- `skill_agents/infer_segmentation/episode_adapter.py` — `infer_and_segment()` (main entry point)
- `skill_agents/infer_segmentation/scorer.py` — `SegmentScorer.score()` (5-term scoring)
- `skill_agents/infer_segmentation/preference.py` — `PreferenceStore`, `PreferenceScorer` (Bradley-Terry)
- `skill_agents/infer_segmentation/dp_decoder.py` — Viterbi DP decoder
- `skill_agents/infer_segmentation/beam_decoder.py` — beam search decoder
- `skill_agents/infer_segmentation/llm_teacher.py` — `collect_segment_preferences()`, `collect_transition_preferences()`, `collect_uncertain_preferences()`, `suggest_skill_name()`
- `skill_agents/infer_segmentation/config.py` — `SegmentationConfig`, `ScorerWeights`, `PreferenceConfig`
- `skill_agents/infer_segmentation/diagnostics.py` — `SegmentDiagnostic`, `SegmentationResult`, `SegmentationDiagnostics`

### Action items

- [ ] Rewrite §1.3 to describe the full preference-learning pipeline (LLM teacher → PreferenceStore → Bradley-Terry → SegmentScorer → decoder)
- [ ] Document the five-term scoring function with weights
- [ ] Decide GRPO integration point (Option A/B/C) and document rationale
- [ ] Rewrite I/O example with actual `SegmentationResult` / `SegmentDiagnostic` data structures
- [ ] Add active learning loop to training description
- [ ] Add transition preferences to reward computation
- [ ] Add beam decoder as alternative to DP
- [ ] Use richer diagnostics (`label_entropy`, `compat_margin`, `boundary_confidence`) in GRPO reward
- [ ] Document `compat_fn` feedback loop from Stage 3
- [ ] Add `suggest_skill_name()` for `__NEW__` segments

---

## Stage 3 — Contract Learning

### Gap Table

| What the codebase has | What the plan says | What needs to change |
|----------------------|-------------------|---------------------|
| **Summarize step** (`segment_summarize.py`): smoothed predicate windows with `start_end_window=5`, OR for UI predicates, mean for vision/HUD; produces `P_start`, `P_end`, `B_start`, `B_end`, `events` | Not mentioned | Add summarize step to Stage 3 pipeline description |
| **Effects compute** (`effects_compute.py`): `eff_add = B_end - B_start`, `eff_del = B_start - B_end`, `eff_event` from normalized UI events; filtered by `reliability_min_for_effects` | Not described | Document the actual effects computation |
| **LLM enrichment is union-only** (`llm_contract.py`): CONTRACT adapter augments frequency-based consensus, doesn't replace it; only used in EM trainer, not in `run_stage3_mvp.py` | Says "CONTRACT adapter generates full contracts, replacing frequency counting entirely" | Correct the description: LLM enrichment is additive; plan must decide whether GRPO truly replaces frequency counting or augments it |
| **VerificationReport richness**: per-literal `eff_add_success_rate`, `eff_del_success_rate`, `eff_event_rate`; `failure_signatures` (e.g. `"miss_add:p1\|miss_del:p2" → count`); `worst_segments`; `instance_pass_literal_frac` (default 0.7) | Plan only mentions `overall_pass_rate` | Use richer verification metrics in GRPO reward |
| **Refine is minimal** (`contract_refine.py`): only drops low-success literals (below `eff_freq`); no strengthening, no contextual refinement | Plan implies more sophisticated refinement | Accurately describe current refine; if GRPO should do better, say so explicitly as a new capability |
| **Full pipeline**: summarize → effects → learn → verify → refine → re-verify → persist | Plan shows learn → verify only | Document the full 7-step pipeline |
| **`run_stage3_mvp.py` does not use `llm_contract.py`** | Plan assumes LLM is integrated | Clarify: LLM enrichment is only in EM trainer path (`stage3_contracts.py`), not in standalone `run_stage3_mvp.py` |

### Key files

- `skill_agents/stage3_mvp/run_stage3_mvp.py` — full pipeline orchestrator
- `skill_agents/stage3_mvp/contract_learn.py` — `learn_effects_contract()` (frequency counting)
- `skill_agents/stage3_mvp/contract_verify.py` — `verify_effects_contract()` → `VerificationReport`
- `skill_agents/stage3_mvp/contract_refine.py` — `refine_effects_contract()` (drop low-success literals)
- `skill_agents/stage3_mvp/effects_compute.py` — `compute_effects()` (per-instance eff_add/eff_del)
- `skill_agents/stage3_mvp/segment_summarize.py` — `summarize_segment()` (predicate windowing)
- `skill_agents/stage3_mvp/llm_contract.py` — `llm_summarize_contract()` (CONTRACT LoRA, union-only)
- `skill_agents/stage3_mvp/schemas.py` — `SegmentRecord`, `SkillEffectsContract`, `VerificationReport`, `Skill`, `Protocol`, `ProtoSkill`
- `skill_agents/stage3_mvp/config.py` — `Stage3MVPConfig`

### Action items

- [ ] Add summarize + effects compute steps to §1.4 pipeline description
- [ ] Document the full 7-step pipeline (summarize → effects → learn → verify → refine → re-verify → persist)
- [ ] Correct "replaces frequency counting" → clarify whether GRPO replaces or augments
- [ ] Use richer `VerificationReport` fields in GRPO reward: per-literal success rates, failure_signatures, worst_segments
- [ ] Document `instance_pass_literal_frac` threshold (default 0.7)
- [ ] Accurately describe current refine capability (drop-only, no strengthening)
- [ ] Rewrite I/O example with actual `SegmentRecord` → `SkillEffectsContract` data flow
- [ ] Note that `run_stage3_mvp.py` does not use `llm_contract.py`; LLM enrichment is EM-trainer-only

---

## Stage 4 — Bank Maintenance

### Gap Table

| What the codebase has | What the plan says | What needs to change |
|----------------------|-------------------|---------------------|
| **SkillProfile** (`run_bank_maintenance.py`): per-skill profile with `effect_sparse_vec`, `embedding_centroid`, `embedding_var_diag`, `transition_topk_prev/next`, `duration_mean/var`, `overall_pass_rate`, `top_violating_literals`, `failure_signature_counts` | Not mentioned | Add SkillProfile to diagnostics that feed the LLM filter prompt |
| **Indices** (`indices.py`): `EffectInvertedIndex` (predicate → skill_ids), `MinHashLSH` (128 permutations, banding), `EmbeddingANN` (brute-force cosine) | Not mentioned | Document indices as the algorithmic backbone for merge/split candidate retrieval |
| **Split triggers** (`split.py`): 4 triggers — low pass rate, failure concentration, high embedding variance, SSE drop from 2-means | Plan only mentions pass_rate < 0.70 | Add all 4 split triggers to candidate generation |
| **Split clustering**: priority — effect signature clustering → sparse effects Jaccard seeding; children must pass `child_pass_rate_thresh` (0.80) | Not described | Document the actual split algorithm |
| **Merge: 3 metrics** (`merge.py`): Jaccard + cosine similarity of embedding centroids + transition overlap; all 3 must pass | Plan only mentions Jaccard | Add all 3 merge metrics to candidate generation |
| **Refine: weaken + strengthen** (`refine.py`): weaken = drop low-success literals; strengthen = add discriminative literals vs confusers (`score = freq_self - max_confuser_freq`) | Plan only describes weakening | Add strengthening to refine action; include confusion partner extraction |
| **Duration model** (`duration_model.py`): per-skill duration histograms | Not mentioned | Note duration model update during refine |
| **Local re-decode** (`local_redecode.py`): `redecode_windows()` re-runs Stage 2 on affected trajectory windows after splits | Not mentioned | Add re-decode step after split execution |
| **Confusion partners**: extracted from Stage 2 diagnostics; used in refine strengthening | Not mentioned | Document confusion partner extraction and usage |
| **Plan references wrong module**: `trainer/skillbank/stages/stage4_update.py` | Should reference `skill_agents/bank_maintenance/` | Update all file references |

### Key files

- `skill_agents/bank_maintenance/run_bank_maintenance.py` — full pipeline: profile → indices → split → merge → refine → redecode
- `skill_agents/bank_maintenance/split.py` — `check_split_triggers()`, `execute_split()`, clustering strategies
- `skill_agents/bank_maintenance/merge.py` — `retrieve_merge_candidates()`, `verify_merge_pair()`, `execute_merge()`
- `skill_agents/bank_maintenance/refine.py` — `check_refine_triggers()`, `refine_skill()` (weaken + strengthen)
- `skill_agents/bank_maintenance/local_redecode.py` — `redecode_windows()`, `collect_affected_trajectories()`
- `skill_agents/bank_maintenance/indices.py` — `EffectInvertedIndex`, `MinHashLSH`, `EmbeddingANN`
- `skill_agents/bank_maintenance/duration_model.py` — `DurationModelStore`
- `skill_agents/bank_maintenance/config.py` — `BankMaintenanceConfig`
- `skill_agents/bank_maintenance/schemas.py` — `SkillProfile`, `SplitResult`, `MergeResult`, `RefineResult`, `BankDiffReport`

### Action items

- [ ] Update all §2 file references from `trainer/skillbank/stages/stage4_update.py` to `skill_agents/bank_maintenance/`
- [ ] Add SkillProfile to the diagnostics dashboard that feeds the LLM filter
- [ ] Document indices (EffectInvertedIndex, MinHashLSH, EmbeddingANN) as candidate retrieval backbone
- [ ] Expand split candidate generation: 4 triggers, 2 clustering strategies, child pass_rate threshold
- [ ] Expand merge candidate generation: 3 metrics (Jaccard + cosine + transition overlap)
- [ ] Add refine strengthening (discriminative literals vs confusers) alongside weakening
- [ ] Add local re-decode step after split execution
- [ ] Add confusion partner extraction and usage in refine
- [ ] Add duration model update to refine action
- [ ] Update LLM filter prompt example to include SkillProfile-level diagnostics

---

## Pipeline / Orchestration

### Gap Table

| What the codebase has | What the plan says | What needs to change |
|----------------------|-------------------|---------------------|
| **`extract_skillbank_gpt54.py` uses its own sequence**: segment → contract → maintenance → materialize → evaluation; skips `run_sub_episode_quality_check()`, proto-skill layer, `distill_execution_hints()` | Not mentioned | Document both orchestration paths (extraction vs co-evolution) |
| **`run_full_iteration()` doesn't call `run_evaluation()`** | Plan shows evaluation in co-evolution loop | Clarify: evaluation is separate from the iteration loop |
| **Sub-episode quality check** (Stage 4.5): `sub_episode_evaluator.py` scores outcome_reward, follow_through, consistency, compactness; retires depleted skills | Not mentioned | Add Stage 4.5 to the plan |
| **Post-processing not in plan**: protocol generation (`generate_skill_protocol()` via GPT-5.4), skill catalogs (`skill_catalog.json`), cross-game archetypes (`skill_archetypes.json`), intention-based fallback, sub-episode linking, episode annotation | Not mentioned | Note as out-of-scope for GRPO but document dependencies |
| **Model mismatch**: extraction uses GPT-5.4; GRPO plan uses Qwen3-14B | Not reconciled | Clarify: cold-start extraction uses GPT-5.4, co-evolution uses Qwen3-14B |
| **SkillEval** (`skill_evaluation/`): LLM-as-judge over 6 dimensions (coherence, discriminability, composability, generalization, utility, granularity) + holistic pass with recommendations (KEEP/REFINE/SPLIT/MERGE/DISCARD) | Plan mentions SkillEval as binary gate | Document the full SkillEval dimensions and how recommendations feed back into Stage 4 |

### Key files

- `skill_agents/pipeline.py` — `SkillBankAgent`, `run_full_iteration()`, `segment_episode()`, query APIs
- `labeling/extract_skillbank_gpt54.py` — cold-start extraction pipeline
- `skill_agents/quality/sub_episode_evaluator.py` — Stage 4.5 quality scoring
- `skill_agents/skill_evaluation/run_evaluation.py` — SkillEval orchestrator
- `skill_agents/skill_evaluation/evaluators.py` — LLM evaluator dimensions

### Action items

- [ ] Document both orchestration paths: extraction (cold-start) vs co-evolution (GRPO training)
- [ ] Add Stage 4.5 (sub-episode quality check) to the plan
- [ ] Clarify SkillEval: 6 dimensions + holistic recommendations, not just a binary gate
- [ ] Note model mismatch: cold-start = GPT-5.4, co-evolution = Qwen3-14B
- [ ] Document post-processing (protocols, catalogs, archetypes) as out-of-scope but note dependencies
- [ ] Fix `run_full_iteration()` vs `run_evaluation()` placement

---

## Rewrite Execution Order

| Priority | Section | Effort | Rationale |
|----------|---------|--------|-----------|
| **1** | Stage 2 (decode/segmentation) | High | Largest gap; preference-learning pipeline is fundamentally different from plan |
| **2** | Stage 4 (bank maintenance) | Medium | Real implementation (`bank_maintenance/`) is much richer than `stage4_update.py` |
| **3** | Stage 1 (boundary proposal) | Medium | Multiple signals and BoundaryPreferenceScorer need documenting |
| **4** | Stage 3 (contract learning) | Low | Gaps are smaller; mostly adding pipeline steps and richer verification |
| **5** | Pipeline/orchestration | Low | Mostly documentation; clarify two paths and add Stage 4.5 |
