# skill_agents (GRPO Skill Bank Pipeline)

Build and maintain a **Skill Bank** from long-horizon game trajectories: segment trajectories into skills, learn symbolic contracts (effects), and serve queries for the [decision_agents](../decision_agents/README.md) VLM agent. The pipeline supports **GRPO-trained LoRA adapters** that wrap existing LLM call points: each call produces G samples, is scored with CPU-only rewards, and the best sample is returned so the EM pipeline runs unchanged while adapters improve over time.

**Model convention:** This project uses **Qwen3-14B** for all skill-bank components (vLLM serving, LoRA adapters, boundary/protocol/contract/curator calls). All configs and code references use Qwen3-14B; legacy Qwen3-8B references are being updated. See [TODO_Lists/SKILLBANK_GRPO_PLAN.md](../TODO_Lists/SKILLBANK_GRPO_PLAN.md) for the full GRPO architecture.

**Model-agnostic design:** The pipeline and decision agent use the same skill-bank functions regardless of LLM backend. GPT and Qwen differ only in which API `ask_model` calls; set `PipelineConfig.llm_model` and/or `extractor_model` (e.g. `Qwen/Qwen3-14B` or `gpt-4o-mini`) for protocol synthesis and boundary proposal.

**Reasoning-model compatibility:** When using Qwen3 or other reasoning models that emit `<think>` blocks, all LLM call sites are wrapped via [`_llm_compat.py`](_llm_compat.py): prompts get `/no_think` appended and responses are stripped of think tags. See [Reasoning-model compatibility](#reasoning-model-compatibility) below.

---

## Overview

| Component | Purpose |
|-----------|--------|
| **SkillBankAgent** | Full pipeline: ingest episodes → segment → learn contracts → maintain bank → query. |
| **SkillQueryEngine** | Rich retrieval + skill selection: keyword, effect-based, and state-aware search. |
| **SkillBankMVP** | Persistent storage (JSONL) for skill contracts and verification reports. Provides `compat_fn` for Stage 3 → Stage 2 feedback. |
| **NewPoolManager** | Rich tracking of `__NEW__` segments; clusters graduate via **proto-skill staging** (materialize → verify → promote). |
| **ProtoSkillManager** | Staging area for new skills: materialize creates proto-skills; verify + promote graduate them to the real bank. |
| **GRPO wrappers** | Wrap existing LLM calls (Stage 2 preferences, Stage 3 contract, Stage 4 filter): G samples, CPU-only reward, best returned to pipeline. |
| **tool_call_reward** | Reward for tool calls (query_skill / query_memory / call_skill) for agentic RL. |
| **PLAN.md** | Operating plan (stages, constraints, data model). |

Subpackages implement each stage:

- **boundary_proposal** — Stage 1: high-recall candidate cut points + boundary plausibility scoring. *Not GRPO-wrapped* (boundaries are algorithmic from predicates + signals).
- **infer_segmentation** — Stage 2: skill labelling with preference learning + contract feedback. **GRPO:** SEGMENT LoRA wraps `collect_segment_preferences()`; reward = `SegmentationDiagnostics`.
- **stage3_mvp** — Stage 3: effects-only contract learn/verify/refine. **GRPO:** CONTRACT LoRA wraps `llm_summarize_contract()`; reward = `verify_effects_contract().overall_pass_rate`.
- **bank_maintenance** — Stage 4: propose candidates → **filter (CURATOR LoRA)** → execute. Refine/merge/split/materialize/promote; proto-skill lifecycle respected.
- **skill_evaluation** — Quality assessment (optional LLM judge).

---

## GRPO Skill Bank Pipeline

**Design: wrap, don’t rebuild.** Existing LLM functions are wrapped with a generic GRPO sampler. Each call produces G samples, evaluates them with a downstream metric (CPU-only), stores (prompt, completions, rewards) in a buffer, and returns the best sample to the pipeline. The EM pipeline runs unchanged.

### Two-phase architecture

| Phase | When | What happens |
|-------|------|---------------|
| **Phase 1 — Rollout** | During EM | Pipeline calls LLM as normal → `GRPOCallWrapper` generates G samples (temperature=0.7) → reward computed per sample → data stored in `GRPOBuffer` → best sample returned to pipeline. |
| **Phase 2 — Training** | After EM step | `GRPOLoRATrainer` reads buffer → recomputes log_probs (gradients on) → group-normalized rewards → GRPO policy gradient → updates LoRA adapter → clears buffer. |

### Three GRPO targets (3 LoRA adapters on Qwen3-14B)

| Adapter | Stage | Wrapped function | Reward |
|---------|-------|------------------|--------|
| **CONTRACT** (P0) | 3 | `llm_summarize_contract()` in `stage3_mvp/llm_contract.py` | `verify_effects_contract().overall_pass_rate` on holdout |
| **CURATOR** (P1) | 4 | `filter_candidates()` — approve/veto/defer bank mutations | `bank_quality_delta` = q_filtered − q_all |
| **SEGMENT** (P1) | 2 | `collect_segment_preferences()` in `infer_segmentation/llm_teacher.py` | `SegmentationDiagnostics` (margin, confidence, compat) |

Stage 1 (boundary) and RETRIEVAL are **not** GRPO-wrapped: boundaries are algorithmic; skill selection is trained by the decision-agent GRPO trainer via episode return.

### Stage 4: propose → filter → execute

Stage 4 uses a **propose-filter-execute** flow:

1. **Propose:** Algorithm builds `SkillProfile` per skill and proposes candidate actions (refine, merge, split, materialize, promote) with triggers from `bank_maintenance/`.
2. **Filter:** One LLM call (`filter_candidates()`) with the CURATOR LoRA — returns approve/veto/defer per candidate. Conflict resolution and deferral expiry (e.g. auto-approve after 3 defers) apply.
3. **Execute:** Approved actions run via production `bank_maintenance/` (refine = weaken + strengthen; merge/split with alias map and local re-decode; materialize → proto-skill staging; promote after verification).

New skills enter through **proto-skill staging:** `__NEW__` clusters → materialize → proto-skill in staging → verify → promote to real bank. See [TODO_Lists/SKILLBANK_GRPO_PLAN.md](../TODO_Lists/SKILLBANK_GRPO_PLAN.md) for full lifecycle and guardrails.

### Infrastructure (not GRPO)

- `run_sub_episode_quality_check()` — data quality gate (heuristic scoring).
- `distill_execution_hints()` — deterministic extraction from sub-episodes.
- `update_protocols()` — LLM synthesis (summarization, not GRPO).
- `run_until_stable()` / convergence, alias map application, proto-skill formation/verification logic.

---

## Quick start

### 1. Build a skill bank from raw episodes

```python
from data_structure.experience import Episode
from skill_agents import SkillBankAgent, PipelineConfig

# Config (optional; defaults work)
config = PipelineConfig(
    bank_path="data/skill_bank.jsonl",
    env_name="llm+overcooked",      # or "llm", "overcooked", etc.
    preference_store_path="data/preferences.json",
    report_dir="data/reports",
    min_instances_per_skill=5,
    # Contract feedback: Stage 3 → Stage 2 closed loop
    contract_feedback_mode="weak",   # "off" | "weak" | "strong"
    contract_feedback_strength=0.3,
    # NEW pool management
    new_pool_min_cluster_size=5,
    new_pool_min_consistency=0.5,
)

agent = SkillBankAgent(config=config)
agent.load()   # load existing bank if path exists

# Episodes: list of Episode (each has .experiences, .task)
episodes = [ep1, ep2, ep3]

# Ingest: Stage 1+2 (segment) + Stage 3 (learn contracts)
agent.ingest_episodes(episodes, env_name="llm+overcooked")

# Optional: iterate until stable (Stage 3 → Stage 4 → materialize NEW)
agent.run_until_stable(max_iterations=3)

agent.save()
```

### 2. Query skills (for decision_agents or scripts)

```python
# Natural-language key (scene / objective / entities)
results = agent.query_skill("navigate to pot and place onion", top_k=3)
for r in results:
    print(r["skill_id"], r["score"], r.get("micro_plan"))

# Rich skill selection (preferred for decision agents):
# separates retrieval relevance from execution applicability
results = agent.select_skill(
    query="navigate to pot",
    current_state={"near_pot": 0.1, "holding_onion": 0.9},
    top_k=3,
)
for r in results:
    print(r["skill_id"], r["relevance"], r["applicability"], r["confidence"])

# Effect-based: find skills that achieve desired state changes
results = agent.query_by_effects(
    desired_add={"at_pot", "holding_onion"},
    desired_del={"at_spawn"},
    top_k=3,
)

# List all skills
summary = agent.list_skills()

# Full detail for one skill
detail = agent.get_skill_detail("nav_to_pot")
```

### 3. Use with the decision agent

Pass the **SkillBankAgent** (or a plain **SkillBankMVP**) as the decision agent’s skill bank. The decision agent calls `select_skill_from_bank` (via `run_tool(TOOL_SELECT_SKILL, ...)`); the helper uses the richest available API (SkillQueryEngine when present, else name match). The same path is used for both GPT and Qwen — only the `model` string changes, and `API_func.ask_model` routes to the correct backend.

```python
from decision_agents import VLMDecisionAgent, run_episode_vlm_agent, RewardConfig
from skill_agents import SkillBankAgent

# Build or load skill bank
skill_agent = SkillBankAgent(bank_path="data/skill_bank.jsonl")
skill_agent.load()

# Decision agent: pass any model name (GPT, Qwen, etc.); same code path
vlm_agent = VLMDecisionAgent(
    model="gpt-4o-mini",      # or "Qwen/Qwen3-14B", etc.
    skill_bank=skill_agent,   # or skill_agent.bank for plain bank
    reward_config=RewardConfig(w_follow=0.1),
)

result = run_episode_vlm_agent(env, agent=vlm_agent, max_steps=500, verbose=True)
```

### 4. Tool-call reward (agentic RL)

Use `compute_tool_call_reward` to get a reward signal for each tool call (query_skill, query_memory, call_skill). Combine with env reward and decision_agents’ `reward` tool for RL training.

```python
from skill_agents import compute_tool_call_reward, ToolCallRewardConfig, SkillBankMVP

bank = SkillBankMVP("data/skill_bank.jsonl")
bank.load()

# After the decision agent calls query_skill and you have the outcome:
reward_result = compute_tool_call_reward(
    tool_name="query_skill",
    tool_args={"key": "navigate to pot and place onion"},
    context_observation="chef near pot, holding onion",
    outcome_observation="onion in pot, soup cooking",
    skill_bank=bank,
    retrieved_skill_id="nav_to_pot",
    retrieved_result={"skill_id": "nav_to_pot", "score": 0.85},
    config=ToolCallRewardConfig(w_relevance=1.0, w_utility=1.0),
)
print(reward_result.r_total)   # for RL loss / value target
print(reward_result.to_dict()) # r_relevance, r_utility, details
```

For a full episode of tool calls, use `compute_episode_tool_call_returns(tool_call_trajectory, skill_bank=bank)` to get a list of per-step rewards.

### How tool-call reward is computed

**r_total = w_relevance × r_relevance + w_utility × r_utility** (for agentic RL: reward for “was this tool call good?”).

- **r_relevance**
  - **query_skill**: If you have a retrieval **score** (e.g. from `SkillQueryEngine.query(key)` or `retrieved_result["score"]`), **r_relevance = score × relevance_scale** (default scale 0.5). If no score is passed but `skill_bank` and query `key` are provided, the code builds a `SkillQueryEngine`, runs `query(key, top_k=1)`, and uses the top result’s score. Otherwise **r_relevance = default_query_reward** (0).
  - **query_memory**: If you pass **retrieval_quality** in [0,1], **r_relevance = retrieval_quality × relevance_scale**; else **r_relevance = default_memory_reward** (0).

- **r_utility** (for **query_skill** and **call_skill**)
  - The skill’s **eff_add** set is taken from the contract (or from the bank using `retrieved_skill_id`). Each predicate is **satisfied** if all of its tokens (length ≥ 2) appear in the **outcome_observation** text (case-insensitive). Then **r_utility = (# satisfied) × utility_per_predicate** (default 0.1), plus **utility_full_completion** (default 0.3) when all **eff_add** predicates are satisfied. No state across steps—purely outcome-based.

Predicate satisfaction is implemented by tokenizing the predicate and the outcome string and checking containment (see `_predicates_satisfied_in_text` in `tool_call_reward.py`). This reward is separate from the decision agent’s **r_follow** (which is stateful and uses the current observation); use both for RL (e.g. env + decision reward + tool-call reward).

---

## Main APIs

### SkillBankAgent

| Method | Description |
|--------|-------------|
| `load()` | Load bank and optional preference store from disk. |
| `save()` | Persist bank, preferences, iteration history. |
| `segment_episode(episode, env_name=..., skill_names=...)` | Run Stage 1+2 on one episode; returns `(SegmentationResult, list[SubTask_Experience])`. Accumulates segment records. |
| `ingest_episodes(episodes, ...)` | Segment each episode then run Stage 3 (contract learning). Returns list of `(result, sub_episodes)` per episode. |
| `run_contract_learning()` | Stage 3: learn/verify/refine effects contracts from accumulated segments. |
| `run_bank_maintenance(...)` | Stage 4: split / merge / refine skills. |
| `materialize_new_skills()` | Promote `__NEW__` segments via `NewPoolManager` (rich clustering + consistency checks). |
| `run_evaluation(episode_outcomes=...)` | Run skill quality evaluation (optional). |
| `run_full_iteration(episodes=...)` | One pass: (optional ingest) → Stage 3 → Stage 4 → materialize → snapshot. |
| `run_until_stable(max_iterations=...)` | Iterate until convergence; then `save()`. |
| `query_skill(key, top_k=3)` | Keyword-style retrieval; returns list of `{skill_id, score, contract, micro_plan}`. |
| `select_skill(query, current_state=..., top_k=3)` | **Rich skill selection**: relevance + applicability + confidence. Preferred for decision agents. |
| `query_by_effects(desired_add=..., desired_del=..., top_k=3)` | Effect-based retrieval. |
| `list_skills()` | Compact list of all skills. |
| `get_skill_detail(skill_id)` | Full contract + report for one skill. |
| `add_skill(skill_id, eff_add, eff_del, eff_event)` | Manually add a skill. |
| `remove_skill(skill_id)` | Remove a skill. |
| `update_skill(skill_id, ...)` | Update contract fields and bump version. |

### SkillQueryEngine

Use when you already have a **SkillBankMVP** and want retrieval without the full pipeline.
The engine auto-loads a RAG `TextEmbedder` (Qwen3-Embedding-0.6B) for
cosine-similarity scoring, blended with keyword Jaccard:

```python
from skill_agents import SkillQueryEngine
from skill_agents.skill_bank.bank import SkillBankMVP

bank = SkillBankMVP("bank.jsonl")
bank.load()

engine = SkillQueryEngine(bank)            # auto-loads RAG embedder

# Simple retrieval (backward compatible)
results = engine.query("place onion in pot", top_k=3)
detail = engine.get_detail("place_onion")
list_all = engine.list_all()

# Rich skill selection (preferred for decision agents):
# separates retrieval relevance from execution applicability
selections = engine.select(
    query="place onion in pot",
    current_state={"near_pot": 0.8, "holding_onion": 0.9},
    top_k=3,
)
for s in selections:
    print(s.skill_id, s.relevance, s.applicability, s.confidence)
    print("  matched:", s.matched_effects, "missing:", s.missing_effects)

# Format expected by decision_agents run_tool(QUERY_SKILL)
decision_result = engine.query_for_decision_agent(
    "place onion",
    current_state={"near_pot": 0.8},  # optional: enables applicability scoring
    top_k=1,
)
# → {"skill_id": "...", "relevance": 0.8, "applicability": 0.6, "confidence": 0.7, ...}
```

### PipelineConfig

Key options (see `pipeline.PipelineConfig` for all). **Model convention:** use `Qwen/Qwen3-14B` for GRPO and co-evolution.

| Field | Default | Meaning |
|-------|--------|--------|
| `bank_path` | `None` | JSONL path for the skill bank. |
| `env_name` | `"llm"` | Signal extraction: `"llm"`, `"llm+overcooked"`, `"overcooked"`, etc. |
| `extractor_model` | `None` | LLM for Stage 1 boundary proposal (e.g. `Qwen/Qwen3-14B`, `gpt-4o-mini`). |
| `llm_model` | `None` | LLM for protocol synthesis and pipeline LLM calls. |
| `merge_radius` | `5` | Merge boundary candidates within this many steps (Stage 1). |
| `preference_iterations` | `3` | Active-learning rounds in Stage 2. |
| `margin_threshold` | `1.0` | Segments with margin below this get preference queries. |
| `contract_feedback_mode` | `"off"` | Stage 3 → Stage 2 contract feedback: `"off"`, `"weak"`, `"strong"`. |
| `contract_feedback_strength` | `0.3` | Weight of contract compatibility term in Stage 2 scoring. |
| `eff_freq` | `0.8` | Min frequency for a literal to be in a contract (Stage 3). |
| `min_instances_per_skill` | `5` | Skip skills with fewer instances in Stage 3. |
| `new_pool_min_cluster_size` | `5` | Min cluster size for NEW → materialize (proto-skill staging). |
| `new_pool_min_consistency` | `0.5` | Min effect pattern consistency for materialize. |
| `new_pool_min_distinctiveness` | `0.25` | Min Jaccard distance from existing skills. |
| `max_iterations` | `5` | Cap for `run_until_stable()`. |

**GRPO:** Per-stage hyperparameters (group_size, clip_ratio, kl_coeff, lr) live in `trainer/common/configs/skillbank_grpo.yaml`; see [SKILLBANK_GRPO_PLAN.md](../TODO_Lists/SKILLBANK_GRPO_PLAN.md).

---

## Data flow

1. **Episode** (from env rollouts or demos) has `experiences` and `task`. Use `Episode` from `data_structure.experience`.
2. **Stage 1** (boundary_proposal): extract signals → propose candidate cut points **C**. Optionally filter with `BoundaryPreferenceScorer`. *Not GRPO-wrapped.*
3. **Stage 2** (infer_segmentation): decode over **C** with preference-learned scorer → segments + skill labels (including `__NEW__`). Contract feedback uses `bank.compat_fn`. **GRPO:** SEGMENT LoRA wraps `collect_segment_preferences()`; reward from `SegmentationDiagnostics` after scorer rebuild + decode.
4. **Stage 3** (stage3_mvp): for each non-NEW skill, learn effects contract, verify, refine; persist to bank. **GRPO:** CONTRACT LoRA wraps `llm_summarize_contract()`; reward = `verify_effects_contract().overall_pass_rate`. Contracts feed back into Stage 2 via `compat_fn`.
5. **Stage 4** (bank_maintenance): **propose** candidates (refine/merge/split/materialize/promote) from `SkillProfile` + triggers → **filter** via `filter_candidates()` (CURATOR LoRA: approve/veto/defer) → **execute** approved actions. Refine = weaken + strengthen; merge/split use alias map and local re-decode. **GRPO:** CURATOR LoRA reward = `bank_quality_delta`.
6. **Proto-skill staging:** `__NEW__` → `NewPoolManager` clusters → materialize → proto-skill in staging → verify → promote to real bank. Proto-skills participate in Stage 2 decoding before promotion.
7. **Query / Select:** decision agent uses `select_skill(query, current_state)` or `query_skill(key)` / `query_by_effects(...)`.

**GRPO co-evolution:** Each EM step runs Phase 1 (wrappers generate G samples, store in buffer, return best); then Phase 2 (`GRPOLoRATrainer` one step per adapter, clear buffer). Protocol synthesis (`update_protocols()`) remains plain LLM inference when `llm_model` / `extractor_model` is set.

---

## Reasoning-model compatibility

Reasoning models (e.g. **Qwen3-14B**, QwQ) default to an internal “thinking” mode that emits `<think>…</think>` blocks before the actual answer. Those blocks consume the `max_tokens` budget and often leave little or no room for the structured output (JSON, rankings, protocols) the pipeline needs.

The module [`skill_agents/_llm_compat.py`](_llm_compat.py) provides:

| Function | Purpose |
|----------|--------|
| `strip_think_tags(text)` | Remove `<think>` blocks from LLM output. |
| `is_reasoning_model(model_name)` | Detect Qwen3 / QwQ-style model names. |
| `wrap_ask_for_reasoning_models(ask_fn, model_hint=...)` | Wrap any `ask_model`-style callable: appends `/no_think` to prompts for reasoning models and strips think tags from responses. |

All LLM entry points in skill_agents use this wrapper:

- **infer_segmentation/llm_teacher.py** — segment/transition rankings, skill naming
- **pipeline.py** — protocol synthesis (`_llm_synthesize_protocol`)
- **boundary_proposal/llm_extractor.py** — predicate extraction, boundary significance
- **skill_bank/llm_retrieval.py** — RETRIEVAL adapter
- **stage3_mvp/llm_contract.py** — CONTRACT adapter
- **skill_evaluation/evaluators.py** — LLM judge

Protocol and naming prompts have also been tightened (game-AI expert roles, concrete steps, tag-specific execution-hint failure modes) so that with the full token budget, outputs are concrete and game-specific rather than generic.

---

## Training integration (GRPO co-evolution)

During co-evolution training, the skill bank pipeline runs with **GRPO wrappers** active. The **SkillBankCoEvolutionCallback** (in `trainer/decision/coevolution_callback.py`) drives the loop; the GRPO pipeline lives under `trainer/skillbank/grpo/`.

### Co-evolution loop (wrapper architecture)

1. **Decision agent GRPO** (existing): collect rollouts, update full policy (including skill selection).
2. At bank-update cadence: trajectories are ingested and the **EM pipeline runs with GRPO wrappers**:
   - **Stage 2:** `collect_segment_preferences()` → SEGMENT wrapper (G samples, `SegmentationDiagnostics` reward, buffer, best returned).
   - **Stage 3:** `llm_summarize_contract()` → CONTRACT wrapper (G samples, `verify_effects_contract` reward, buffer, best returned).
   - **Stage 4:** `propose_candidates()` → `filter_candidates()` (CURATOR wrapper) → `execute_approved()`. Filter: G samples, `bank_quality_delta` reward, buffer, best returned.
3. Infrastructure steps (no GRPO): `run_sub_episode_quality_check()`, `distill_execution_hints()`, `update_protocols()`.
4. **Phase 2:** `GRPOLoRATrainer.train_step(grpo_buffer)` — one gradient step per adapter; buffer cleared.
5. SkillEval gating: if bank quality drops, rollback; otherwise bank is hot-swapped into workers.

Shared infrastructure: `GRPOCallWrapper`, `GRPOBuffer`, `GRPOLoRATrainer`; `MultiLoraSkillBankLLM.log_probs()` for training. Config: `trainer/common/configs/skillbank_grpo.yaml` and `skillbank_em.yaml`.

### Segmentation persistence

The `SegmentationStore` keeps per-trajectory segmentations updated during training. When EM re-segments with a newer bank, the store replaces the old entry so replay/evaluation/visualization see the latest segmentation.

### Tool-call reward in training

`tool_call_reward` is integrated via `TrainRewardShaper` (in `trainer/decision/reward_shaping.py`). For QUERY_SKILL, QUERY_MEM, CALL_SKILL the shaper adds `r_tool`:

`r_total = r_env + w_follow × r_follow + r_cost + tool_call_reward_weight × r_tool`

See [trainer/README.md](../trainer/README.md) for co-evolution setup and [TODO_Lists/SKILLBANK_GRPO_PLAN.md](../TODO_Lists/SKILLBANK_GRPO_PLAN.md) for the full GRPO plan (rewards, Stage 4 propose-filter-execute, proto-skill lifecycle, guardrails).

---

## Subpackage docs

- [boundary_proposal/README.md](boundary_proposal/README.md) — Stage 1 signals and `segment_episode` / `propose_from_episode`.
- [infer_segmentation/README.md](infer_segmentation/README.md) — Stage 2 preference learning and decoders.
- [PLAN.md](PLAN.md) — Full operating plan (constraints, thresholds, module map).
- [TODO_Lists/SKILLBANK_GRPO_PLAN.md](../TODO_Lists/SKILLBANK_GRPO_PLAN.md) — GRPO wrapper architecture, 3 LoRA adapters, Stage 4 propose-filter-execute, proto-skill staging, guardrails.

---

## File layout

```
skill_agents/
├── README.md              # This file
├── PLAN.md                # SkillBank Agent operating plan
├── __init__.py            # SkillBankAgent, SkillQueryEngine, NewPoolManager, etc.
├── _llm_compat.py         # Reasoning-model compatibility (strip_think_tags, /no_think wrapper)
├── pipeline.py            # SkillBankAgent orchestrator (contract feedback, NEW pool, proto-skill flow)
├── query.py               # SkillQueryEngine + SkillSelectionResult (retrieval + selection policy)
├── tool_call_reward.py    # Reward for tool calls (agentic RL)
├── skill_bank/
│   ├── bank.py            # SkillBankMVP persistence + compat_fn (Stage 3→2 feedback)
│   └── new_pool.py        # NewPoolManager: NEW tracking; get_candidates() for Stage 4 materialize
├── boundary_proposal/     # Stage 1 (not GRPO-wrapped)
├── infer_segmentation/    # Stage 2 — SEGMENT LoRA wraps collect_segment_preferences()
├── stage3_mvp/            # Stage 3 — CONTRACT LoRA wraps llm_summarize_contract()
├── bank_maintenance/      # Stage 4 — propose (profiles, triggers) + execute (refine/merge/split/materialize/promote)
├── contract_verification/
├── skill_evaluation/      # Quality evaluation
└── lora/                  # MultiLoraSkillBankLLM (log_probs for GRPO), Qwen3-14B config

trainer/skillbank/
├── grpo/                  # GRPO wrapper pipeline
│   ├── buffer.py          # GRPOBuffer (adapter, prompt, completions, rewards)
│   ├── wrapper.py         # GRPOCallWrapper (G samples, reward, store, return best)
│   ├── trainer.py         # GRPOLoRATrainer (log_probs, advantages, LoRA update)
│   ├── rewards.py         # contract_reward(), curator_reward(), segmentation_reward()
│   └── config.py          # Per-stage GRPO hyperparameters
├── stage4_candidates.py   # propose_candidates(), CandidateAction, conflict/deferral annotation
├── stage4_filter.py       # filter_candidates() — CURATOR LoRA, approve/veto/defer
├── stage4_prompts.py      # Curator system prompt, output schema
└── em_trainer.py          # EM loop: propose → filter → execute; Phase 2 GRPO train_step
```
