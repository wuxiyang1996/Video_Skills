# Skill Bank Co-Evolution — TODO Plan

## Status Legend

- [ ] Not started
- [~] In progress
- [x] Completed

---

## Phase 1: GRPO Reward Differentiation (COMPLETED)

All items in this phase have been implemented.

- [x] Fix `segmentation_reward` yielding identical values despite unique completions
  - Set `repr=False` on `PreferenceExample.timestamp`
  - Enriched `_preference_list_fingerprint` to include raw LLM response hash
  - Created `PreferenceListWithRollouts` to carry raw rollouts
- [x] Fix `contract_reward` / `curator_reward` all-zero (connection errors)
  - Created `_llm_retry.py` with `sync_ask_with_retry` for robust retrying
  - Integrated retry into `llm_summarize_contract` and `filter_candidates`
  - Enhanced `ask_vllm` with `max_retries` for the OpenAI client
  - Added LLM fallback chain: LoRA -> local vLLM -> OpenRouter
- [x] Add raw-text blend to contract/curator rewards
  - `_raw_completion_fingerprint` blends 8% raw text hash into rewards
  - Ensures reward diversity even for structurally identical parsed outputs
- [x] Add tie-breaking to GRPO advantage calculation
  - Created `compute_grpo_group_advantages` in `advantage_utils.py`
  - Completion-based tie-break when reward variance is near zero
- [x] Create `SkillBankLLMOutput` to carry raw completions for log-prob training
- [x] Make training threshold configurable via `SKILLBANK_TRAIN_THRESHOLD` env var

## Phase 2: Diverse Skill Initialization (COMPLETED)

- [x] Add game-stage-aware default skill seeds (`phase:TAG` compound labels)
  - `_GAME_DEFAULT_SEEDS` in `pipeline.py` for tetris, 2048, sokoban, etc.
  - `_GENERIC_DEFAULT_SEEDS` fallback for unknown games
  - Seeds merged into `_seed_skills_from_intentions` output
- [x] Update `episode_adapter.py` guard: `< 2 skill_names` now a warning (safety net)

## Phase 3: Skill Enrichment Pipeline (COMPLETED)

Ported techniques from `labeling/extract_skillbank_gpt54.py` into online co-evolution.

- [x] Tag-aware protocol generation (`enrich_skill_protocols`)
  - Deterministic templates per `SUBGOAL_TAG` (SETUP, CLEAR, MERGE, etc.)
  - Augmented with contract effects and phase-based preconditions
- [x] Execution hint generation (`enrich_execution_hints`)
  - Termination cues, common failure modes, state-transition patterns
- [x] Compute `expected_duration` from segment data
- [x] Link sub-episode outcomes for skill quality scoring (`link_sub_episode_outcomes`)
- [x] Orchestration: `enrich_bank_after_update` wired as Stage 4 in `skillbank_pipeline.py`

## Phase 4: LLM Protocol Refinement Loop (COMPLETED)

Iteratively upgrade tag-based template protocols to LLM-synthesized protocols
using actual rollout evidence from Qwen (local vLLM).

- [x] Add `source` field to `Protocol` dataclass (`"template"` / `"llm"` / `"deterministic"`)
  - Both `skill_agents_grpo` and `skill_agents` schemas updated
  - Serialized in `to_dict`, deserialized in `from_dict`
- [x] Guard tag enrichment: skip overwrite when `source == "llm"`
- [x] Route `_llm_synthesize_protocol` to local vLLM with retry
  - Primary: `sync_ask_with_retry(ask_vllm, ...)` — zero API cost
  - Fallback: `ask_model` (OpenRouter/GPT)
- [x] Source-aware gating in `update_protocols`
  - Template protocols upgraded eagerly (1+ sub-episode evidence)
  - LLM protocols require 3+ high-quality sub-episodes to re-synthesize
- [x] Periodic `refine_low_pass_protocols` for underperforming skills
  - Configurable via `PROTOCOL_REFINE_EVERY` env var (default: every 3 iterations)
- [x] Wire as Stage 5 in `skillbank_pipeline.py` (after enrichment, before save)
- [x] Diagnostics: per-skill source transitions logged, aggregate upgrade counts

---

## Phase 5: Protocol Quality Improvement (NEXT)

Further improvements to make LLM-generated protocols more actionable.

- [ ] Include game state diffs (predicates_start vs predicates_end) in synthesis prompt
  - Sub-episode evidence currently uses only summaries
  - State diffs give the LLM concrete before/after context
- [ ] Failure-aware re-synthesis
  - Track which protocol step the skill tracker aborted at
  - Include failure patterns in the re-synthesis prompt so Qwen can adjust
  - E.g. "Step 3 'Clear bottom rows' failed in 4/5 executions because stack_h > 18"
- [ ] Richer sub-episode summaries
  - Include action sequences (top-5 most common) in evidence
  - Include reward trajectory shape (rising, flat, declining)

## Phase 6: Decision Agent Integration

- [ ] Validate `_format_skill_guidance_for_prompt` handles LLM protocols correctly
  - Verify step_checks, predicate_success, predicate_abort are passed through
- [ ] Test skill tracker protocol advancement with LLM-generated step_checks
  - Tag templates have no step_checks; LLM protocols do
  - Verify `compute_step_advancement` works with real predicates
- [ ] Monitor protocol effectiveness
  - Compare success rates for skills with template vs LLM protocols
  - Log protocol source in episode results for offline analysis

## Environment Variables Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `SKILLBANK_LLM_RETRIES` | `5` | Max LLM call attempts with exponential backoff |
| `SKILLBANK_LLM_RETRY_DELAY_S` | `1.0` | Base backoff delay in seconds |
| `VLLM_OPENAI_MAX_RETRIES` | `3` | OpenAI client retry count for vLLM |
| `GRPO_TIEBREAK_SCALE` | `0.02` | Scale for tie-breaking noise in advantage calc |
| `SKILLBANK_TRAIN_THRESHOLD` | `32` | Min GRPO samples before training |
| `PROTOCOL_REFINE_EVERY` | `3` | Re-synthesize low-pass protocols every N iterations |

## Key Files

| File | Role |
|------|------|
| `skill_agents_grpo/pipeline.py` | Skill bank agent: protocol synthesis, update, refinement |
| `skill_agents_grpo/grpo/rewards.py` | GRPO reward functions (segment, contract, curator) |
| `skill_agents_grpo/grpo/advantage_utils.py` | Advantage calculation with tie-breaking |
| `skill_agents_grpo/grpo/grpo_outputs.py` | `SkillBankLLMOutput` for raw completion tracking |
| `skill_agents_grpo/_llm_retry.py` | Retry wrapper for LLM calls |
| `trainer/coevolution/skillbank_pipeline.py` | Async pipeline: Stages 1-5 orchestration |
| `trainer/coevolution/skill_enrichment.py` | Tag-based enrichment (protocols, hints, durations) |
| `trainer/coevolution/episode_runner.py` | Decision agent rollout + skill tracker |
| `skill_agents_grpo/stage3_mvp/schemas.py` | Protocol, Skill, Contract dataclasses |
