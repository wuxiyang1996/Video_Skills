# skill_agents

Build and maintain a **Skill Bank** from long-horizon game trajectories: segment trajectories into skills, learn symbolic contracts (effects), and serve queries for the [decision_agents](../decision_agents/README.md) VLM agent.

---

## Overview

| Component | Purpose |
|-----------|--------|
| **SkillBankAgent** | Full pipeline: ingest episodes → segment → learn contracts → maintain bank → query. |
| **SkillQueryEngine** | Rich retrieval + skill selection policy: keyword, effect-based, and state-aware search. |
| **SkillBankMVP** | Persistent storage (JSONL) for skill contracts and verification reports. Provides `compat_fn` for Stage 3 → Stage 2 feedback. |
| **NewPoolManager** | Rich tracking and promotion of `__NEW__` segments into real skills. |
| **tool_call_reward** | Reward for tool calls (query_skill / query_memory / call_skill) for agentic RL training. |
| **PLAN.md** | Operating plan (stages, constraints, data model). |
| **PIPELINE_CALL_FLOW.md** | How each function is called within the agent framework (illustration). |

Subpackages implement each stage:

- **boundary_proposal** — Stage 1: high-recall candidate cut points + boundary plausibility scoring.
- **infer_segmentation** — Stage 2: skill labelling with preference learning + contract feedback loop.
- **stage3_mvp** — Stage 3: effects-only contract learn/verify/refine.
- **bank_maintenance** — Stage 4: split / merge / refine skills.
- **skill_evaluation** — Quality assessment (optional LLM judge).

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

Pass the **SkillBankAgent** (or a plain **SkillBankMVP**) as the decision agent’s skill bank. The decision agent will call `query_skill` with a key; the helper uses the richest available API (SkillBankAgent → SkillQueryEngine → name match).

```python
from decision_agents import VLMDecisionAgent, run_episode_vlm_agent, RewardConfig
from skill_agents import SkillBankAgent

# Build or load skill bank
skill_agent = SkillBankAgent(bank_path="data/skill_bank.jsonl")
skill_agent.load()

# Decision agent uses it for QUERY_SKILL and prompt context
vlm_agent = VLMDecisionAgent(
    model="gpt-4o-mini",
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

Key options (see `pipeline.PipelineConfig` for all):

| Field | Default | Meaning |
|-------|--------|--------|
| `bank_path` | `None` | JSONL path for the skill bank. |
| `env_name` | `"llm"` | Signal extraction: `"llm"`, `"llm+overcooked"`, `"overcooked"`, etc. |
| `merge_radius` | `5` | Merge boundary candidates within this many steps (Stage 1). |
| `preference_iterations` | `3` | Active-learning rounds in Stage 2. |
| `margin_threshold` | `1.0` | Segments with margin below this get preference queries. |
| `contract_feedback_mode` | `"off"` | Stage 3 → Stage 2 contract feedback: `"off"`, `"weak"`, `"strong"`. |
| `contract_feedback_strength` | `0.3` | Weight of contract compatibility term in Stage 2 scoring. |
| `eff_freq` | `0.8` | Min frequency for a literal to be in a contract (Stage 3). |
| `min_instances_per_skill` | `5` | Skip skills with fewer instances in Stage 3. |
| `min_new_cluster_size` | `5` | Min segment count to materialize a NEW skill (legacy path). |
| `new_pool_min_cluster_size` | `5` | Min cluster size for `NewPoolManager` promotion. |
| `new_pool_min_consistency` | `0.5` | Min effect pattern consistency for NEW promotion. |
| `new_pool_min_distinctiveness` | `0.25` | Min Jaccard distance from existing skills. |
| `max_iterations` | `5` | Cap for `run_until_stable()`. |

---

## Data flow

1. **Episode** (from env rollouts or demos) has `experiences` (list of state/action/reward/done, etc.) and `task`. Use `Episode` from `data_structure.experience` in this repo.
2. **Stage 1** (boundary_proposal): extract signals → propose candidate cut points **C**. Optionally filter with `BoundaryPreferenceScorer` for plausibility.
3. **Stage 2** (infer_segmentation): decode over **C** with preference-learned scorer → segments + skill labels (including `__NEW__`). When contract feedback is enabled (`contract_feedback_mode="weak"|"strong"`), the scorer uses `bank.compat_fn` to bias labelling toward skills whose contracts match observed predicate changes.
4. **Stage 3** (stage3_mvp): for each non-NEW skill, learn effects contract, verify, refine; persist to bank. **Contracts feed back into Stage 2** via `compat_fn` (closed loop).
5. **Stage 4** (bank_maintenance): split low-quality skills, merge similar ones, refine contracts; optional local re-decode.
6. **Materialize NEW**: `NewPoolManager` clusters NEW pool by effect similarity (Jaccard, not exact match), tracks context metadata (predecessor/successor, duration), and promotes clusters meeting support + consistency + separability criteria.
7. **Query / Select**: decision agent calls `select_skill(query, current_state)` for rich results (relevance + applicability + confidence), or `query_skill(key)` / `query_by_effects(...)` for backward-compatible retrieval.

---

## Training integration (VERL co-evolution)

During VERL-based training, all four skill agent stages are integrated into the training loop via the **SkillBankCoEvolutionCallback** (in `trainer/decision/coevolution_callback.py`). Each stage is wrapped as a named tool in the `SkillAgentToolPipeline`:

| Tool name | Stage | Function |
|-----------|-------|----------|
| `boundary_proposal` | Stage 1 | Propose candidate segment boundaries from predicate/surprisal signals |
| `segmentation_decode` | Stage 2 | Assign skill labels via Viterbi/DP over proposed cuts |
| `contract_learning` | Stage 3 | Learn and verify effects contracts from decoded segments |
| `bank_maintenance` | Stage 4 | Refine, materialize NEW, merge, split skills |

### How it works

1. After each VERL training step (at the configured cadence), the callback extracts rollout trajectories from the VERL batch.
2. Trajectories are converted to `TrajectoryForEM` and fed through the full pipeline.
3. The EMTrainer runs Stages 1-4 iteratively until convergence.
4. If the update is accepted (passes SkillEval gating), the bank is hot-swapped into environment workers.
5. Trajectory segmentations are persisted in a `SegmentationStore` (JSONL), keyed by trajectory ID. Each entry records the decoded segments, bank version, global step, and associated tool calls.

### Segmentation persistence

The `SegmentationStore` (in the callback module) keeps per-trajectory segmentations that are **updated during training**. When the EM pipeline re-segments a trajectory with a newer bank, the store replaces the old entry. This lets downstream consumers (e.g. replay, evaluation, visualization) access the latest segmentation for any trajectory.

### Tool-call reward in training

The `tool_call_reward` module is integrated into the training reward pipeline via `TrainRewardShaper` (in `trainer/decision/reward_shaping.py`). When the decision agent makes a tool call (QUERY_SKILL, QUERY_MEM, CALL_SKILL), the shaper computes `r_tool` using `compute_tool_call_reward` and adds it to the step reward:

```
r_total = r_env + w_follow × r_follow + r_cost + tool_call_reward_weight × r_tool
```

This provides a dense learning signal for when and how to use tools effectively.

See [trainer/README.md](../trainer/README.md) for full co-evolution setup and configuration.

---

## Subpackage docs

- [boundary_proposal/README.md](boundary_proposal/README.md) — Stage 1 signals and `segment_episode` / `propose_from_episode`.
- [infer_segmentation/README.md](infer_segmentation/README.md) — Stage 2 preference learning and decoders.
- [PLAN.md](PLAN.md) — Full operating plan (constraints, thresholds, module map).

---

## File layout

```
skill_agents/
├── README.md              # This file
├── PLAN.md                # SkillBank Agent operating plan
├── CHANGELOG_REFACTOR.md  # Detailed changelog for the refactoring
├── __init__.py            # SkillBankAgent, SkillQueryEngine, SkillSelectionResult, NewPoolManager, etc.
├── pipeline.py            # SkillBankAgent orchestrator (contract feedback, NEW pool integration)
├── query.py               # SkillQueryEngine + SkillSelectionResult (retrieval + selection policy)
├── tool_call_reward.py    # Reward for tool calls (agentic RL)
├── skill_bank/
│   ├── bank.py            # SkillBankMVP persistence + compat_fn (Stage 3→2 feedback)
│   └── new_pool.py        # NewPoolManager: rich NEW tracking + promotion
├── boundary_proposal/     # Stage 1 + boundary preference learning
│   └── boundary_preference.py  # BoundaryPreferenceScorer
├── infer_segmentation/    # Stage 2 (ContractFeedbackConfig, boundary quality term)
├── stage3_mvp/            # Stage 3 contract learn/verify/refine
├── bank_maintenance/      # Stage 4 split/merge/refine
├── contract_verification/
└── skill_evaluation/      # Quality evaluation
```
