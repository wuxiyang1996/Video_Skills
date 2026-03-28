# Enhance Agentic Decision-making in Multiple-player long-horizon games with unsupervised experiences

[![GitHub Repository](https://img.shields.io/badge/GitHub-Repository-blue)](https://github.com/wuxiyang1996/Game-AI-Agent)

## Overview

This repository provides a framework for enhancing agentic decision-making in multi-player, long-horizon games through unsupervised experience. The framework integrates with multiple game environments and supports both training-free (RAG-based) and trainable (RL-based) **text LLM** agent architectures (observations are **text** summaries; the default decision and training stack is **LLM-only**). This readme outlines each module and aims to ease integration and debugging.

**No external repos are bundled.** This repository contains only Game-AI-Agent code. For Avalon/Diplomacy you need [AgentEvolver](https://github.com/modelscope/AgentEvolver) (clone as sibling or on `PYTHONPATH`). For GamingAgent evaluation, clone that repo as a sibling when needed; see [evaluate_gamingagent/setup_gamingagent_eval_env.md](evaluate_gamingagent/setup_gamingagent_eval_env.md).

**GRPO in this repo:** The default co-evolution loop trains all five LoRA adapters with **GRPO via PyTorch FSDP** ([`trainer/coevolution/grpo_training.py`](trainer/coevolution/grpo_training.py) → [`skill_agents_grpo/grpo/fsdp_trainer.py`](skill_agents_grpo/grpo/fsdp_trainer.py)), not VERL. For an **optional** Ray/VERL stack (vLLM/sglang, GiGPO/PPO, `RayPPOTrainer`), use [verl-agent](https://github.com/verl-project/verl-agent) as a sibling; see [INSTALL.md](INSTALL.md) and [trainer/decision/grpo_trainer.py](trainer/decision/grpo_trainer.py) (`GameAITrainer`).

**Install:** See **[INSTALL.md](INSTALL.md)** for setup (optional verl-agent). Quick: add this repo to `PYTHONPATH`, or `conda env create -f environment.yml` then `conda activate game-ai-agent`.

**Contents:** 1. [Environments](#1-environments) · 2. [Data structure (skills and experiences)](#2-data-structure-skills-and-experiences) · 3. [Skill agent](#3-skill-agent) · 4. [Decision-making agent](#4-decision-making-agent) · 5. [Trainer code](#5-trainer-code) · [Implemented (done)](#implemented-done)

## Quick Links

- **🔗 Repository**: [GitHub - Game-AI-Agent](https://github.com/wuxiyang1996/Game-AI-Agent)

- **📦 Environment Wrappers** — [env_wrappers/](env_wrappers/): NL wrappers and evaluation for game environments
  - [Avalon](env_wrappers/avalon_nl_wrapper.py) · [Diplomacy](env_wrappers/diplomacy_nl_wrapper.py) — require **AgentEvolver** (external; [AgentEvolver Games](https://github.com/modelscope/AgentEvolver/blob/main/games/README.md)); eval: [evaluation_evolver/](evaluation_evolver/)
  - [GamingAgent](env_wrappers/gamingagent_nl_wrapper.py) — LMGame-Bench (2048, Sokoban, Tetris); requires **GamingAgent** (external); eval: [evaluate_gamingagent/](evaluate_gamingagent/)

- **🔍 RAG & Embeddings** — [rag/](rag/): **Text** embeddings for retrieval (default [Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)); this matches the **LLM-only** decision pipeline. Optional `MULTIMODAL_EMBEDDING_MODEL` is only for extended RAG corpora beyond plain text; the shipped decision and co-evolution paths do not require it. [rag/README.md](rag/README.md)

- **🎮 Decision Agent** — [decision_agents/](decision_agents/): **LLM** step-by-step play on **text** state summaries with a **two-turn micro-loop** per timestep: (1) **take_action** — primitives or `QUERY_MEM` / `QUERY_SKILL` / `CALL_SKILL`; (2) **reward** — composite reward. **Per-step protocol:** `get_state_summary` (required) → optional `query_skill` or `query_memory` (budget-limited) → `take_action` (required) → `get_intention` (required) → `reward` (required). Skill bank supplies **protocol store** (name, steps, preconditions) for planning and **contract** (eff_add) for r_follow; `select_skill_from_bank` / `query_skill_bank` return protocol steps and use `bank.get_contract(skill_id)` for reward. **Model-agnostic:** same code path for GPT, Qwen, etc.; pass `model="gpt-4o-mini"` or `model="Qwen/Qwen3-8B"`; callers should pass an explicit `model`. See [decision_agents/README.md](decision_agents/README.md).
  - **Core**: [agent.py](decision_agents/agent.py) — `LLMDecisionAgent`, `run_tool()`, `run_episode_llm_agent()`; [dummy_agent.py](decision_agents/dummy_agent.py) — game detection, action extraction.
  - **Helpers**: [agent_helper.py](decision_agents/agent_helper.py) — `get_state_summary()`, `build_rag_summary()`, `extract_game_facts()`, `infer_intention()` ([TAG] phrase), `EpisodicMemoryStore`, `skill_bank_to_text()`, `query_skill_bank()` / `select_skill_from_bank()`, `_get_protocol_for_skill()`, `SUBGOAL_TAGS`.
  - **Reward**: [reward_func.py](decision_agents/reward_func.py) — `RewardConfig`, `RewardComputer`; **r_total** = r_env + w_follow×r_follow + r_cost (query_mem_cost, query_skill_cost, call_skill_cost, skill_switch_cost).
  - **Tools**: `take_action`, `reward`, `get_state_summary`, `get_intention`, `query_skill`, `query_memory`. Only `query_skill` and `query_memory` are optional (budget-limited); never call both in the same timestep.

- **📚 Skill Agents** — [skill_agents/](skill_agents/): Build and maintain a Skill Bank from trajectories; consumed by decision_agents for skill retrieval. **Preferred API for decision agents:** `select_skill(query, current_state, top_k)` (relevance + applicability + confidence); backward-compatible `query_skill(key)`, `query_by_effects()`. **NewPoolManager** ([skill_bank/new_pool.py](skill_agents/skill_bank/new_pool.py)): rich tracking and promotion of `__NEW__` segments (Jaccard clustering, support + consistency + separability). **Contract feedback:** Stage 3 → Stage 2 via `contract_feedback_mode` ("off" | "weak" | "strong") and `bank.compat_fn`. **Model-agnostic:** Set `PipelineConfig.llm_model` / `extractor_model`; reasoning-model compat via [_llm_compat.py](skill_agents/_llm_compat.py) (/no_think, strip think tags). [skill_agents/README.md](skill_agents/README.md) · [PLAN.md](skill_agents/PLAN.md) · [PIPELINE_CALL_FLOW.md](skill_agents/PIPELINE_CALL_FLOW.md).
  - **Orchestrator** [SkillBankAgent](skill_agents/pipeline.py): `ingest_episodes` → segment (Stage 1+2) → learn contracts (Stage 3) → maintain bank (Stage 4) → `query_skill` / `select_skill` / `materialize_new_skills()`. Methods: `segment_episode()`, `run_contract_learning()`, `run_bank_maintenance()`, `update_protocols()`, `run_until_stable()`.
  - **Stage 1** [boundary_proposal/](skill_agents/boundary_proposal/): Candidate cut points C (signals: rule-based or LLM `env_name="llm"`; optional RAG change-point; `merge_radius`). [README](skill_agents/boundary_proposal/README.md)
  - **Stage 2** [infer_segmentation/](skill_agents/infer_segmentation/): Decode over C with preference scorer (LLM → Bradley–Terry); contract feedback uses `compat_fn` when enabled. Segments + labels (bank + `__NEW__`). [README](skill_agents/infer_segmentation/README.md)
  - **Stage 3** [stage3_mvp/](skill_agents/stage3_mvp/): Effects contract learn/verify/refine; NEW → pool; contracts feed back to Stage 2 via `compat_fn`.
  - **Stage 4** [bank_maintenance/](skill_agents/bank_maintenance/): Split/merge/refine; `materialize_new_skills()` (NewPoolManager).
  - **Query & storage**: [SkillQueryEngine](skill_agents/query.py) (RAG + keyword/effect; `select()` for rich selection), [SkillBankMVP](skill_agents/skill_bank/bank.py) (+ `compat_fn`), [tool_call_reward](skill_agents/tool_call_reward.py), [skill_evaluation/](skill_agents/skill_evaluation/).

- **🏋️ Training** — [trainer/](trainer/): Co-evolution of both agents (async loop in [`trainer/coevolution/`](trainer/coevolution/)). **Decision GRPO** uses **FSDP**; **Skill Bank** uses Hard-EM. Optional **VERL** path via verl-agent for Ray-scale training.
  - **SFT Cold-Start** ([trainer/SFT/](trainer/SFT/)): Train all 5 LoRA adapters (skill_selection, action_taking, segment, contract, curator) from teacher-labelled cold-start data before GRPO. Launch: `bash scripts/run_sft_coldstart.sh`.
  - **Agent A (Decision)**: GRPO — primitives + `QUERY_MEM` / `QUERY_SKILL` / `CALL_SKILL`; reward = r_env + shaping + costs + tool-call reward.
  - **Agent B (SkillBank)**: Hard-EM (decode → update → gate); all four stages packed as a tool pipeline in the [co-evolution callback](trainer/decision/coevolution_callback.py). Trajectory segmentations stored and updated via `SegmentationStore`.
  - **Co-evolution callback**: [coevolution_callback.py](trainer/decision/coevolution_callback.py) — `SkillBankCoEvolutionCallback` + `SkillAgentToolPipeline` + `SegmentationStore`; integrates `skill_agents.tool_call_reward` into reward shaping. On accepted EM update, [launch_coevolution.py](trainer/launch_coevolution.py) passes the training model into `SkillBankAgent` for protocol synthesis (same `ask_model` routing as inference). [launch_train](trainer/decision/launch_train.py) initializes `SkillQueryEngine` when loading the bank so training rollouts use the same retrieval path as inference.
  - Shared: [metrics](trainer/common/metrics.py), [reward_shaping](trainer/decision/reward_shaping.py), [eval_harness](trainer/common/eval_harness.py). Entry: [launch_train](trainer/decision/launch_train.py), [launch_coevolution](trainer/launch_coevolution.py). [trainer/README.md](trainer/README.md)

- **▶️ Inference** — [inference/](inference/): Run the decision agent and store rollouts in [data_structure](data_structure/experience.py) format (`Episode` + `Experience`). **Unified skill bank path:** Both `scripts/run_inference.py` (any model via `--model`) and `scripts/run_qwen3_8b_eval.py` (Qwen, `--bank` optional) use the same `load_skill_bank()`, `select_skill_from_bank()`, and `skill_bank_to_text()`; only the LLM backend differs. `run_episode_llm_agent()` returns `Episode` directly; `run_inference()` wraps it with buffer/save support. [inference/README.md](inference/README.md)

---

# 1. Environments

This framework integrates with the following game environments — **8 games** across **four** stacks, all covered by cold-start generators and NL wrappers. See [cold_start/readme.md](cold_start/readme.md) for scope, registry keys, and rollout commands.

| Game | Source | Wrapper | Evaluation |
|------|--------|---------|------------|
| **2048** | **External:** [GamingAgent](https://github.com/lmgame-org/GamingAgent) (LMGame-Bench) | [gamingagent_nl_wrapper.py](env_wrappers/gamingagent_nl_wrapper.py) | [evaluate_gamingagent/](evaluate_gamingagent/) |
| **Sokoban** | **External:** [GamingAgent](https://github.com/lmgame-org/GamingAgent) (LMGame-Bench) | [gamingagent_nl_wrapper.py](env_wrappers/gamingagent_nl_wrapper.py) | [evaluate_gamingagent/](evaluate_gamingagent/) |
| **Candy Crush** | **External:** [GamingAgent](https://github.com/lmgame-org/GamingAgent) (LMGame-Bench) | [gamingagent_nl_wrapper.py](env_wrappers/gamingagent_nl_wrapper.py) | [evaluate_gamingagent/](evaluate_gamingagent/) |
| **Tetris** | **External:** [GamingAgent](https://github.com/lmgame-org/GamingAgent) (LMGame-Bench) | [gamingagent_nl_wrapper.py](env_wrappers/gamingagent_nl_wrapper.py) | [evaluate_gamingagent/](evaluate_gamingagent/) |
| **Avalon** | **External:** [AgentEvolver Games](https://github.com/modelscope/AgentEvolver/blob/main/games/README.md) — hidden-role deduction | [avalon_nl_wrapper.py](env_wrappers/avalon_nl_wrapper.py) | [evaluation_evolver/](evaluation_evolver/) |
| **Diplomacy** | **External:** [AgentEvolver Games](https://github.com/modelscope/AgentEvolver/blob/main/games/README.md) — strategic negotiation | [diplomacy_nl_wrapper.py](env_wrappers/diplomacy_nl_wrapper.py) | [evaluation_evolver/](evaluation_evolver/) |
| **Super Mario** | **External:** Orak (gym_super_mario_bros) | [orak_nl_wrapper.py](env_wrappers/orak_nl_wrapper.py) | [evaluate_orak/](evaluate_orak/) |
| **Pokemon Red** | **External:** Orak (PyBoy + pokered) | [orak_nl_wrapper.py](env_wrappers/orak_nl_wrapper.py) | [evaluate_orak/](evaluate_orak/) |

### Wrapper identification contract

All wrappers set `info["env_name"]` and `info["game_name"]` on every `reset()` and `step()` call. These two fields, together with `episode_id` (auto-generated UUID on each `Episode`), form a **three-part identifier** that uniquely locates any trajectory across platforms:

| Wrapper | `info["env_name"]` | `info["game_name"]` | Notes |
|---------|---------------------|---------------------|-------|
| GamingAgentNLWrapper | `"gamingagent"` | Auto-detected from obs (`"sokoban"`, `"tetris"`, `"2048"`, …) or explicit `game_name` constructor arg | Constructor: `GamingAgentNLWrapper(env, game_name="sokoban")` |
| AvalonNLWrapper | `"avalon"` | `"avalon"` | Single-game platform |
| DiplomacyNLWrapper | `"diplomacy"` | `"diplomacy"` | Single-game platform |
| OrakNLWrapper | `"orak"` | Registry key (e.g. `"super_mario"`, `"pokemon_red"`) | [evaluate_orak/](evaluate_orak/); cold-start: [cold_start/](cold_start/) |
| ColdStartEnvWrapper | `"gamingagent"` | Registry key (e.g. `"sokoban"`, `"twenty_forty_eight"`) | Used in [cold_start/](cold_start/) |

`run_episode_llm_agent()` reads `info["env_name"]` and `info["game_name"]` to populate the `Episode`. Fallback chain: `info["game"]` → `detect_game(obs)` → `info["structured_state"]["game"]`.

---

## Decision & Skill Agents: Co-evolution

Two agents co-evolve: the **Decision Agent** (Qwen3-8B + LoRA, GRPO) plays games using skills from the bank, and the **Skill Bank Agent** (Qwen3-8B + 3 LoRA, Hard-EM) discovers and refines skills from the Decision Agent's trajectories. Training runs on **8 × A100-80GB GPUs** (GPUs 0-3 for Decision, GPUs 4-7 for Skill Bank). See [Section 5: Trainer code](#5-trainer-code) for full training settings and scripts.

### Big picture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                          CO-EVOLUTION LOOP                                    │
│                                                                              │
│  Decision Agent (Qwen3-8B + LoRA)       Skill Bank Agent (Qwen3-8B + 3 LoRA)
│  GPUs 0-3, GRPO (FSDP)                   GPUs 4-7, Hard-EM                   │
│  ┌──────────────────────┐                ┌──────────────────────┐            │
│  │ • take_action        │  query_skill   │ • boundary proposal  │            │
│  │ • query_skill(key)   │ ─────────────► │ • segment decode     │            │
│  │ • query_memory       │ ◄───────────── │ • contract learning  │            │
│  │ • call_skill(id)     │  bank + skills │ • bank maintenance   │            │
│  │ • reward (r_follow)  │                │ • SkillEval gating   │            │
│  └──────────┬───────────┘                └──────────┬───────────┘            │
│             │                                       │                        │
│             ▼                                       ▼                        │
│  ┌──────────────────────┐                ┌──────────────────────┐            │
│  │  Episode (trajectory) │ ────feed────► │  Skill Bank (JSONL)   │            │
│  │  summary_state (k=v)  │    back       │  versioned, queryable │            │
│  │  intentions ([TAG])   │               │  contracts + protocols│            │
│  │  reward_details       │               │  ~42 skills per game  │            │
│  └──────────────────────┘                └──────────────────────┘            │
│                                                                              │
│  Each iteration: rollouts → Skill Bank EM → Decision Agent GRPO → repeat     │
└──────────────────────────────────────────────────────────────────────────────┘
```

### How they connect

| Role | Decision Agent | Skill Bank Agent |
|------|----------------|------------------|
| **Model** | Qwen3-8B + LoRA (rank 16) | Qwen3-8B + 3 LoRA (SEGMENT, CONTRACT, CURATOR) |
| **Training** | GRPO via **FSDP** (group size 8, LR 1e-5); optional VERL | Hard-EM with LoRA fine-tuning (LR 2e-4) |
| **Provides** | Game play: primitive actions + tool calls (`QUERY_SKILL`, `CALL_SKILL`, `QUERY_MEM`) | Skill Bank: segmented trajectories, effect contracts, split/merge/refine |
| **Consumes** | Skill bank (protocols for planning, contracts for reward shaping) | Raw episodes from Decision Agent rollouts |
| **Interface** | `LLMDecisionAgent(skill_bank=..., model=...)` → `select_skill_from_bank()` / `query_skill_bank()` | `SkillBankAgent.select_skill(query, current_state)` / `SkillQueryEngine` |

The decision agent’s `skill_bank` can be a **SkillBankMVP** (plain storage) or a **SkillBankAgent** (full pipeline). Helpers (`skill_bank_to_text`, `query_skill_bank`, `select_skill_from_bank`) accept both and use the richest API available.

**Skill bank: protocol store vs contract.** Each skill has two parts: (1) **Protocol store** — `name`, `strategic_description`, `protocol` (steps, preconditions, success_criteria, expected_duration), `confidence`; used for `skill_bank_to_text()` and `active_skill_plan`. (2) **Contract** — effects (`eff_add`, `eff_del`, `eff_event`) used for segmentation, verification, and reward shaping (r_follow). The agent **plans** from protocols and is **rewarded** for progress on the contract's eff_add predicates.

**Per-game skill banks (current design).** The skill bank is **maintained separately per game**: co-evolution uses [`PerGameSkillBankManager`](trainer/coevolution/skillbank_pipeline.py) so each title gets its own `skill_bank.jsonl` under `<bank_dir>/<game>/` (skills from one game do not mix with another). **Skill-valued metrics**—pass rates, confidence, applicability/retrieval scores, GRPO rewards on bank stages, etc.—are **computed over that game’s bank and trajectories** for now, not pooled across games unless you explicitly merge or share a single bank path.

**Skill protocols for select.** When the agent selects a skill, the result includes a **protocol**: **steps** (up to 7), **preconditions**, **success_criteria**, **expected_duration**. If no protocol exists, a **micro_plan** is built from `eff_add` literals. See [agent_helper.py](decision_agents/agent_helper.py).

### RAG retrieval ([rag/](rag/))

Both agents use RAG embeddings (Qwen3-Embedding-0.6B) for retrieval, with graceful fallback to keyword-only:

| Component | Memory query | Skill query |
|-----------|-------------|-------------|
| **Decision Agent** | [EpisodicMemoryStore](decision_agents/agent_helper.py): cosine similarity + keyword overlap (70/30 blend) | Via [SkillQueryEngine](skill_agents/query.py): cosine + keyword Jaccard (60/40 blend) |
| **Skill Bank Agent** | N/A | [SkillQueryEngine](skill_agents/query.py): cosine + keyword Jaccard; optional embedding change-point scores in boundary proposal |

### Typical workflow

1. **Bootstrap** — load or create an initial skill bank from seed episodes:
   `skill_agent = SkillBankAgent(bank_path="..."); skill_agent.ingest_episodes(seed_episodes)`

2. **Play** — run the Decision Agent with the bank:
   `episode = run_episode_llm_agent(env, agent=LLMDecisionAgent(skill_bank=skill_agent), task="...")`

3. **Feed back** — ingest trajectories into the skill pipeline:
   `skill_agent.ingest_episodes([episode])` then `skill_agent.run_until_stable()`

4. **Repeat** — the bank improves (new skills, splits/merges/refinements), so the next run has better retrieval and reward shaping.

**For training**: `bash scripts/coevolution_train.sh` runs this loop with GRPO + Hard-EM on 8 GPUs. See [Section 5](#5-trainer-code) for details.

See [decision_agents/README.md](decision_agents/README.md) for the LLM decision agent API and [skill_agents/README.md](skill_agents/README.md) for the pipeline and query usage.

### Reward (decision + skill agents)

During training, reward is computed from two sources, unified by `TrainRewardShaper`:

**r_total = r_env + 0.1 × r_follow + r_cost + 0.1 × r_tool**

| Component | Signal | Source |
|-----------|--------|--------|
| **r_env** | Raw environment reward from `env.step` | [reward_func.py](decision_agents/reward_func.py) |
| **r_follow** | Skill-following shaping: per-predicate bonus for each newly satisfied `eff_add` literal in the observation, completion bonus when all satisfied, small penalty per step with no progress. Stateful over the episode. | [reward_func.py](decision_agents/reward_func.py) |
| **r_cost** | Negative costs: `query_mem_cost` (-0.05), `query_skill_cost` (-0.05), `call_skill_cost` (-0.02), `skill_switch_cost` (-0.10) | [reward_func.py](decision_agents/reward_func.py) |
| **r_tool** | Tool-call quality: `r_relevance` (retrieval score) + `r_utility` (outcome `eff_add` satisfaction). Outcome-based, no cross-step state. | [tool_call_reward.py](skill_agents/tool_call_reward.py) |

Predicate satisfaction is **text-based**: tokenize the predicate (e.g. `onion_in_pot` → `["onion", "in", "pot"]`), then check all tokens (length ≥ 2) appear in the observation string (case-insensitive).

---

# 2. Data structure (skills and experiences)

This section defines how **experiences**, **episodes**, and **sub-episodes** are represented in code. Sub-episodes are the unit used to **generate skills**: the skill pipeline segments episodes into **SubTask_Experience** segments, then converts them to lightweight **SubEpisodeRef** pointers stored in each **Skill** (see [Skill agent](#3-skill-agent) and [skill_agents/](skill_agents/)). The skill bank format follows this readme: each skill has a **protocol store** (name, steps, preconditions) for the decision agent and a **contract** (eff_add, etc.) for reward and verification; evidence is a list of `SubEpisodeRef` from sub-episodes.

- **Definitions** (classes, serialization): [data_structure/experience.py](data_structure/experience.py), [data_structure/helper.py](data_structure/helper.py).
- **Producers**: [decision_agents/](decision_agents/) (`run_episode_llm_agent()` → `Episode` with full `Experience` fields), [labeling/](labeling/) ([label_episodes_gpt54.py](labeling/label_episodes_gpt54.py) for cold-start; [labeling/readme.md](labeling/readme.md)).
- **Consumers**: [skill_agents/](skill_agents/) ingests `Episode`, segments into `SubTask_Experience`, and builds the skill bank (protocol + contract + `Skill.sub_episodes` as `SubEpisodeRef`).

## Experience

Single step: **state, action, reward, next_state, done**. Implemented in [data_structure/experience.py](data_structure/experience.py) as `Experience` with required fields and optional: **intentions**, **tasks**, **sub_tasks**, **summary**, **summary_state**, **idx**, **sub_task_done**, **reward_details**, **action_type**, **raw_state**, **raw_next_state**, **available_actions**, **interface**.

Structured formats (used by RAG and skill pipeline):

| Field | Format | Purpose |
|-------|--------|---------|
| **summary_state** | `key=value \| key=value` (e.g. `game=tetris \| stack_h=14 \| holes=32`) | Deterministic; from `build_rag_summary()` (0 LLM). Retrieval and predicates. |
| **summary** | `summary_state \| note=<strategic note>` | Facts + short LLM note (≤10 words); delta-aware. |
| **intentions** | `[TAG] subgoal phrase` (e.g. `[CLEAR] Reduce holes before stack overflows`) | 13 tags: SETUP, CLEAR, MERGE, ATTACK, DEFEND, NAVIGATE, POSITION, COLLECT, BUILD, SURVIVE, OPTIMIZE, EXPLORE, EXECUTE. Used for skill boundaries and RAG. |
| **reward_details** | `{r_env, r_follow, r_cost, r_total}` | Per-step breakdown from the LLM agent’s reward tool. |
| **action_type** | `"primitive"`, `"QUERY_MEM"`, `"QUERY_SKILL"`, `"CALL_SKILL"` | Used by trainer and metrics. |

`run_episode_llm_agent()` fills these from agent state: `summary_state` ← `get_state_summary`, `intentions` ← `get_intention`, `sub_tasks` ← `active_skill_id`, `reward_details` / `action_type` from reward tool. Skill agents read `summary_state` and `intentions` for segmentation without extra conversion.

Serialization: `Experience.from_dict()` / `to_dict()`.

## Episode

A time-ordered sequence of **Experience** with **task**, **outcome**, **summary**, **metadata**. Implemented as `Episode` in [data_structure/experience.py](data_structure/experience.py): **experiences**, **task**, **outcome**, **summary**, **metadata**; **episode_id**, **env_name**, **game_name** (three-part identifier). Methods: **get_reward()**, **get_total_reward()**, **get_length()**, **set_outcome()**, **separate_into_sub_episodes(outcome_length)**.

**Sub-episodes for skill generation:** `Episode.separate_into_sub_episodes(outcome_length)` splits the episode by **sub_tasks** boundaries (experience indices where `sub_tasks` changes) and returns a list of **SubTask_Experience**. Each segment has the experiences for one sub-task and optional outcome experiences (next `outcome_length` steps). The [skill_agents](skill_agents/) pipeline also produces `SubTask_Experience` via Stage 1+2 (boundary proposal + segmentation); those segments are then turned into **SubEpisodeRef** via `SubTask_Experience.to_sub_episode_ref()` and stored in **Skill.sub_episodes**.

### Three-part identifier

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| **episode_id** | `str` | UUID (auto or caller) | `"a3f1b2c4-..."` |
| **env_name** | `str` | Platform / wrapper | `"gamingagent"`, `"avalon"`, `"diplomacy"`, `"orak"` |
| **game_name** | `str` | Game within platform | `"sokoban"`, `"tetris"`, `"avalon"`, `"super_mario"` |

Filled by `run_episode_llm_agent()` from wrapper `info`; persisted in `to_dict()` / `from_dict()`. **metadata** holds rollout info (e.g. `cumulative_reward`, `agent_state`, `done`, `steps`).

## Sub-episode (SubTask_Experience) → skills

**SubTask_Experience** is the contiguous segment used to **generate skills**. Defined in [data_structure/experience.py](data_structure/experience.py):

| Field | Description |
|-------|-------------|
| **sub_task** | Sub-task / strategy label (skill name or ID). |
| **final_goal** | Episode-level task this segment serves. |
| **sub_task_experience** | List of `Experience` in the segment (used during processing; not stored in the skill bank). |
| **outcome_experiences** | Optional follow-up experiences to evaluate outcome. |
| **summary**, **outcome_summary** | For retrieval and quality. |
| **length**, **cumulative_reward** | Cached for quick use. |
| **seg_id**, **episode_id**, **rollout_source** | Link to pipeline segment and rollout. |
| **quality_score**, **outcome_classification** | Set by skill agent quality pipeline (`"success"` \| `"partial"` \| `"failure"`). |

The skill bank does **not** store full experience lists. **SubTask_Experience.to_sub_episode_ref()** produces a **SubEpisodeRef** (episode_id, seg range, rollout_source, summary, intention_tags, outcome, cumulative_reward, quality_score); that pointer is stored in **Skill.sub_episodes** (see [skill_agents/stage3_mvp/schemas.py](skill_agents/stage3_mvp/schemas.py)). Skill format (protocol + contract) is described in [Skill agent](#3-skill-agent) and in the Quick Links (protocol store for planning, contract for r_follow).

Serialization: `SubTask_Experience.from_dict()` / `to_dict()`; `outcome_experiences` may be `None`.

## Buffers

Defined in [data_structure/experience.py](data_structure/experience.py):

- **Experience_Replay_Buffer**: FIFO of experiences; **add_experience()** (single, list, or `Episode`), **sample_experience()**, **get_experience_summary()**.
- **Episode_Buffer**: FIFO of episodes; **add_episode()**, **sample_episode()**, **get_episode_summary()**, **save_to_json()** / **load_from_json()**.
- **Tool_Buffer**: stores **SubTask_Experience** (tools/strategies); **add_tool()**, **sample_tool()**, **get_tool_summary()**.

---

# 3. Skill agent

## Sub-task decomposition

Segment long-horizon trajectories into **sub-tasks** (skills): unlabeled episodes → candidate boundaries → skill-labeled segments → learned contracts. Input: [Episode](data_structure/experience) with `experiences` (state/action/reward/done, `summary_state` or `state`, `intentions` as `[TAG] phrase`) and `task`. Output: [SubTask_Experience](data_structure/experience) segments with skill labels, persisted as skills with protocol + contract in the bank. See [skill_agents/README.md](skill_agents/README.md) and [skill_agents_grpo/README.md](skill_agents_grpo/README.md) for the base pipeline and GRPO-wrapped pipeline.

## Pipeline implementation ([skill_agents/](skill_agents/), [skill_agents_grpo/](skill_agents_grpo/))

| Step | Implementation | Notes |
|------|----------------|--------|
| **Stage 1: Boundary proposal** | [boundary_proposal](skill_agents/boundary_proposal/): signals (rule-based per env or **LLM** `env_name="llm"` / `"llm+overcooked"`) → **candidate cut points** C. [merge_radius](skill_agents/pipeline.py) (default 5). *Not GRPO-wrapped.* | High-recall boundaries only. |
| **Stage 2: Sub-task labeling** | [infer_segmentation](skill_agents/infer_segmentation/): decode over C with **preference-learned scorer** (LLM rankings → Bradley–Terry). Options = bank IDs + `__NEW__`. Contract feedback via `compat_fn`. **GRPO:** [skill_agents_grpo](skill_agents_grpo/): SEGMENT LoRA wraps `collect_segment_preferences()`; reward = [`segmentation_reward()`](skill_agents_grpo/grpo/rewards.py) (decode + episode reward + optional `bank_skill_scores`). | Output: [SegmentationResult](skill_agents/infer_segmentation/diagnostics.py) + list of [SubTask_Experience](data_structure/experience). |
| **Stage 3: Contract learning** | [stage3_mvp](skill_agents/stage3_mvp/): effects-only contracts (eff_add, eff_del, eff_event), verify, refine → [SkillBankMVP](skill_agents/skill_bank/bank.py). **GRPO:** CONTRACT LoRA wraps `llm_summarize_contract()`; reward = [`contract_reward()`](skill_agents_grpo/grpo/rewards.py) (standalone verify + start/end coverage + consensus + reward-weighted instances). | Contracts feed back to Stage 2 via `compat_fn`. |
| **Stage 4: Bank maintenance** | [bank_maintenance](skill_agents/bank_maintenance/): **propose** (refine/merge/split/materialize/promote) → **filter** via `filter_candidates()` (CURATOR LoRA: approve/veto/defer) → **execute** approved actions. **GRPO:** CURATOR LoRA reward = [`curator_reward()`](skill_agents_grpo/grpo/rewards.py) (skill quality + exploration + outcomes when available). [materialize_new_skills](skill_agents/pipeline.py), [NewPoolManager](skill_agents/skill_bank/new_pool.py), **proto-skill staging** (`__NEW__` → materialize → verify → promote). | See [skill_agents_grpo/README.md](skill_agents_grpo/README.md) and [TODO_Lists/SKILLBANK_GRPO_PLAN.md](TODO_Lists/SKILLBANK_GRPO_PLAN.md). |

**Training for the labeling path** is implemented via the GRPO pipeline (SEGMENT LoRA for Stage 2, CONTRACT for Stage 3, CURATOR for Stage 4) and co-evolution with the decision agent; see [skill_agents_grpo/README.md](skill_agents_grpo/README.md).

**Quick usage:**

```python
from skill_agents import SkillBankAgent, PipelineConfig
from data_structure.experience import Episode

config = PipelineConfig(
    bank_path="data/skill_bank.jsonl",
    env_name="llm+overcooked",   # or "llm", "llm+gamingagent"
    merge_radius=5,
    contract_feedback_mode="weak",
    new_pool_min_cluster_size=5,
    new_pool_min_consistency=0.5,
)
agent = SkillBankAgent(config=config)
agent.load()

episodes = [ep1, ep2, ...]   # Episode with .experiences, .task
agent.ingest_episodes(episodes, env_name="llm+overcooked")  # Stage 1+2 per episode, then Stage 3
agent.run_until_stable(max_iterations=3)   # Stage 3 → Stage 4 → materialize NEW
agent.save()
```

Entry points: [SkillBankAgent](skill_agents/pipeline.py) (`segment_episode`, `ingest_episodes`, `run_until_stable`), [skill_agents/PLAN.md](skill_agents/PLAN.md).

---

# 4. Decision-making agent

The [decision_agents](decision_agents/) module provides an LLM decision agent for step-by-step game play with **skill-bank retrieval** (RAG-based), episodic memory, intention inference, and composite reward. Two backends share the same code path: **GPT-5.4** (training-free, cold-start/labeling) and **Qwen3-8B** (GRPO-trained, vLLM). See [decision_agents/README.md](decision_agents/README.md).

## RAG-mode decision agent with skill selection

The main inference pipeline that **selects skills** from the bank is **Pipeline A**: [scripts/qwen3_decision_agent.py](scripts/qwen3_decision_agent.py). It uses a **SkillQueryEngine** (RAG) and a **_SkillTracker** for protocol-aware lifecycle (re-selection on stall, duration, success/abort criteria).

**Per-step loop:** (1) **get_state_summary()** — deterministic + LLM into `key=value`; (2) **infer_intention()** — `[TAG] subgoal phrase`; (3) **Skill re-selection** when no active skill, duration exceeded, or zero-reward stall; (4) **get_skill_guidance()** — queries `SkillQueryEngine` with `game_name + intention + state` and structured state for applicability; returns skill_id, protocol (steps, preconditions, success/abort); (5) **qwen3_action()** — prompt with skill plan + current step marker → vLLM; (6) **parse_qwen_response()** — action extraction (exact → fuzzy → **RAG embedding** fallback via `ActionEmbeddingMatcher`); (7) env.step; (8) **_SkillTracker.update()** and build **Experience**.

**Skill selection (RAG):** [SkillQueryEngine](skill_agents/query.py) pre-embeds skill descriptions (Qwen3-Embedding-0.6B). **select_skill_from_bank()** routes to: **SkillQueryEngine.select()** (relevance + applicability + confidence) → **query_for_decision_agent()** → **SkillBankAgent.select_skill()** → TF-IDF fallback. **Scoring:** retrieval relevance (60% embedding + 40% keyword Jaccard), execution applicability (eff_add/eff_del vs current state), combined confidence (relevance + applicability + pass_rate). Returns skill_id, protocol, execution_hint, micro_plan, etc.

**Reward:** **r_total** = r_env + w_follow×r_follow + r_cost (and optional r_tool). r_follow uses the active skill's **eff_add** (per-predicate bonus, completion bonus, no-progress penalty). Config: [RewardConfig](decision_agents/reward_func.py); same contract used in [trainer/decision/reward_shaping.py](trainer/decision/reward_shaping.py).

**Run (with skill bank):**
```bash
python -m scripts.qwen3_decision_agent --games twenty_forty_eight --episodes 3
python -m scripts.qwen3_decision_agent --no-bank --episodes 3   # baseline without bank
```

Full API, pipelines (A vs B), and scoring details: [decision_agents/README.md](decision_agents/README.md).


## Training-free agent

This mode uses RAG to query experiences most relevant to the current situation and intentions from the experience buffer, using them as in-context learning to assist decision-making. The same **LLMDecisionAgent** is used with a **skill_bank** and optional **EpisodicMemoryStore**; no parameter updates — the backbone (e.g. GPT, Gemini, Claude) stays frozen.

## Trainable agent

This mode gathers experience via interaction and updates parameters with reinforcement learning. The **[trainer/](trainer/)** module implements it: the **LLM Decision Agent** is trained with **GRPO** (retrieval as first-class actions; reward = r_env + shaping + query/call costs), and the Skill Bank is updated via **Hard-EM**. See [trainer/README.md](trainer/README.md) and the “Trainer Code” section below.

## Inference: run and store rollouts

The **[inference/](inference/)** module runs the decision agent and stores rollouts in the [data_structure](data_structure/experience.py) format (`Episode` with list of `Experience`). Use this for collecting trajectories without training (e.g. for skill pipeline ingestion or replay buffers).

**Run one episode and get an `Episode`:**
```python
from inference import run_inference, rollout_to_episode
from data_structure.experience import Episode_Buffer, Experience_Replay_Buffer

episode = run_inference(
    env,
    task="Complete level 1",
    max_steps=500,
    verbose=True,
)
```

**Optional: add to buffers and/or append to a JSONL file:**
```python
ep_buffer = Episode_Buffer(buffer_size=100)
exp_buffer = Experience_Replay_Buffer(buffer_size=10_000)
episode = run_inference(
    env,
    task="Complete level 1",
    episode_buffer=ep_buffer,
    experience_buffer=exp_buffer,
    save_path="rollouts/episodes.jsonl",
    verbose=True,
)
```

**`run_episode_llm_agent` already returns an `Episode`:**
```python
from decision_agents import run_episode_llm_agent

episode = run_episode_llm_agent(env, task="My task", max_steps=500)
# episode.episode_id: auto-generated UUID
# episode.env_name:   from wrapper info (e.g. "gamingagent", "avalon")
# episode.game_name:  specific game (e.g. "sokoban", "2048", "avalon")
# episode.experiences: list of Experience with summary_state, intentions, sub_tasks populated
# episode.metadata: rollout-level data (cumulative_reward, agent_state, done, steps)
# episode.to_dict(): for JSON save/load (includes all identifier fields)

# Feed directly into skill pipeline:
from skill_agents.pipeline import SkillBankAgent
skill_agent = SkillBankAgent(bank_path="skills/bank.jsonl")
skill_agent.ingest_episodes([episode])
```

`rollout_to_episode()` is still available for backward compatibility with legacy flat dicts.

Storage: each episode is one JSON object per line (JSONL) when `save_path` is set. See [inference/README.md](inference/README.md) for full details.

## Design goals (decision agent)

Implemented in the RAG-mode pipeline ([scripts/qwen3_decision_agent.py](scripts/qwen3_decision_agent.py) and [decision_agents/](decision_agents/)):

1. **Summarize the state** — `get_state_summary()` / `build_rag_summary()` (deterministic `key=value`, game-aware); optional LLM compression.
2. **Intention update / skill re-selection** — `infer_intention()`; `_SkillTracker.should_reselect()` (stall, duration, success/abort criteria).
3. **RAG for skill selection** — `SkillQueryEngine` (embedding + applicability); `get_skill_guidance()`; EpisodicMemoryStore for memory query.
4. **Action generation** — `qwen3_action()` from state + skill protocol + history; `parse_qwen_response()` with RAG embedding fallback.

## Reward design for skill agent

This part is for the **skill agent** ([skill_agents/](skill_agents/)): reward signals and training objectives used when building or evaluating the skill bank. RAG is not fine-tuned (frozen).

We adopt the idea of GDPO: normalize each kind of reward under each category. Reward categories to involve in this system include:

1. **Task completion reward** — From the environment: winning condition or survival (e.g. survive as long as possible).
2. **Sub-task completion reward** — From env or skill-level signals: e.g. occupation, kill, or skill **eff_add** satisfaction (see [decision_agents/reward_func](decision_agents/reward_func.py) r_follow for the decision agent; skill agent may use similar notions for segment/skill quality).
3. **Format reward** — For valid outputs or adherence to action/state format.

The first two are env-related. A time-sensitive discount can be applied to penalize long episodes. For reward tied to the *decision* agent’s use of skills (query_skill, call_skill), see [skill_agents/tool_call_reward](skill_agents/tool_call_reward.py) (r_relevance, r_utility).

## GRPO reward functions (per adapter)

Each of the 5 LoRA adapters has its own reward signal used during GRPO training.
Decision-agent adapters receive rewards from game rollouts; skill-bank adapters
receive shaped rewards from dedicated reward functions in
[`skill_agents_grpo/grpo/rewards.py`](skill_agents_grpo/grpo/rewards.py).

### skill_selection

**Source**: raw environment reward from `env.step(action)` — assigned per step in
[`episode_runner.py`](trainer/coevolution/episode_runner.py).

```
reward = float(env_reward)   # sparse, per-step
```

An unused richer formula exists in `rewards.py` (`skill_selection_reward()`) but
is not wired into the episode runner:

```
r = 0.40 * r_env + 0.20 * r_efficiency + 0.20 * r_success + 0.10 * r_no_abort + 0.10 * r_confidence
```

### action_taking

**Source**: same raw environment reward from `env.step(action)` — identical to
`skill_selection`.

```
reward = float(env_reward)   # sparse, per-step
```

Both decision-agent adapters share the reward signal per step, stored as
`GRPORecord` objects in [`episode_runner.py`](trainer/coevolution/episode_runner.py).

### segment — `segmentation_reward()`

**Source**: shaped reward on each GRPO sample = one full **preference list** for
the episode; the wrapper calls `reward_fn(sample, *args, **kwargs)` so **each
rollout can get a different score** when preferences differ.

**Decode path** (normal: `scorer_factory` + `decode_fn` from
[`enable_segment_grpo`](skill_agents_grpo/infer_segmentation/llm_teacher.py)):

- Rebuilds the preference scorer, runs Viterbi/beam decode, then blends:
  - **Decode quality** — normalized `total_score`, mean **margins**
  - **Reuse** — fraction of segments not labeled `__NEW__`
  - **Reward alignment** — share of **positive** per-step env reward on segments
    assigned to existing skills (partial credit for `__NEW__`); uses
    `per_step_rewards` / `episode_total_reward` from the episode
  - **Value match** (when `bank_skill_scores` is passed from the pipeline) —
    high-`compute_skill_score()` skills aligned with high-reward segments

Weights differ slightly with vs without `bank_skill_scores` (see source).

**Fallback** (no scorer/decode): reuse strength + winner **diversity**,
per-segment **dominance**, segment **coverage**, optional bank-quality hint, and
a small **content fingerprint** of (segment → winning skill) so different LLM
rankings do not collapse to identical rewards.

Pipeline plumbing: [`pipeline.py`](skill_agents_grpo/pipeline.py) passes
`bank_skill_scores` into [`infer_and_segment`](skill_agents_grpo/infer_segmentation/episode_adapter.py) → `collect_segment_preferences(...)`.

### contract — `contract_reward()`

**Source**: shaped reward on parsed `{"eff_add", "eff_del"}` per GRPO sample;
**each completion is scored on its own** (standalone contract, no union-merge
with consensus for verification).

**Verification path** (holdout instances + `verify_config`; context from
[`set_contract_reward_context`](skill_agents_grpo/stage3_mvp/llm_contract.py)):

- **Standalone pass rate** — weighted by per-instance **cumulative game reward**
- **Start→end coverage** — F1 of `eff_add` vs `predicates_end`, `eff_del` vs
  `predicates_start`, sparsity, **specificity** (effects that match true
  gains/losses), plus a tiny **effect-identity fingerprint** so near-duplicate
  effect sets still differ (avoids flat rewards across the group)
- **Precision / recall** vs frequency consensus
- **Reward align** — pass rate on high- vs low-reward holdouts

**Fallback** (no holdout): same start/end F1 + sparsity + specificity +
fingerprint; empty effects → `0.05`, `None` output → `0.0`.

### curator — `curator_reward()`

**Source**: shaped reward on parsed `{"decisions": [{"idx", "verdict", "reason"}, ...]}`.

**Evidence path** (default when `action_outcomes` not set):

- **Quality alignment** — continuous: approve scores with `skill_score`, veto
  with `(1 - skill_score)`, defer mid-range
- **Exploration** — bonus for approving **materialize** / **promote** with some
  evidence (`pass_rate` / `skill_score` / `n_instances`)
- **Reason quality** — cites metrics (`pass_rate`, `skill_score`, `n_instances`,
  `skill_id`) in free text

**Outcomes path** (maintenance sets
[`set_curator_reward_context`](skill_agents_grpo/bank_maintenance/llm_curator.py)
with `action_outcomes`): weighted **approve/veto** correctness vs
`succeeded` + `|quality_delta|`, blended with the evidence path and an
**exploration** term for new-skill actions.

Curator prompt includes **mean skill score** and guidance to ground decisions in
[`Skill.compute_skill_score()`](skill_agents_grpo/stage3_mvp/schemas.py) while
encouraging new-skill exploration.

### Per-rollout variance (GRPO advantage signal)

[`GRPOCallWrapper`](skill_agents_grpo/grpo/wrapper.py) samples **G** completions
at `temperature > 0` and scores each with `reward_fn(sample_i, *args, **kwargs)`.
Rewards must **differ across samples** when outputs differ, or group advantages
collapse to zero. The skill-bank reward functions above are designed so that
different effect sets, preference rankings, or curator JSON produce different
scores; regression tests live in
[`tests/test_reward_variance.py`](tests/test_reward_variance.py) (with
[`tests/test_quality_scoring.py`](tests/test_quality_scoring.py) for segment /
skill quality scoring).

### Training flow

Rewards are stored in two forms depending on the adapter:

- **Decision agent**: `GRPORecord` per step (prompt, completion, reward) collected
  in [`episode_runner.py`](trainer/coevolution/episode_runner.py), converted to
  training data via `_records_to_training_data()`.
- **Skill bank**: `GRPOSample` per call (prompt, completions, rewards) stored in
  [`GRPOBuffer`](skill_agents_grpo/grpo/buffer.py) via
  [`GRPOCallWrapper`](skill_agents_grpo/grpo/wrapper.py), converted via
  `_samples_to_training_data()`.

Advantages are computed with `_compute_advantages()`: zero-mean, unit-variance
normalization within each group.

## Skill evaluation and exploration

**Sub-episode (segment) quality** — [`score_sub_episode()`](skill_agents_grpo/quality/sub_episode_evaluator.py)
writes `SubEpisodeRef.quality_score` using configurable
`SegmentQualityWeights`: **episode credit** (normalized cumulative game reward),
**local progress** (outcome), **segmentation validity** (compactness + tag
consistency), **contract validity** (`contract_pass_rate` / threshold), and a
**novelty bonus** only when the segment is valid and contract-verifiable.

**Skill-level score** — `Skill.compute_skill_score()` in
[`schemas.py`](skill_agents_grpo/stage3_mvp/schemas.py) (no longer dominated by raw
usage frequency):

```
skill_score ≈ w_seg * mean(quality_score)
           + w_reuse * reuse_success
           + w_contract * contract_pass
           + w_consistency * cross_episode_consistency
           + w_explore * exploration_value
```

- **reuse_success** — successful outcomes on sub-episodes (with evidence scaling),
  not raw selection counts
- **contract_pass** — verification pass rate when provided
- **cross_episode_consistency** — stability of rewards across episodes
- **exploration_value** — small bonus for young skills with a real contract

Weights are overridable via an optional `weights` dict; empty sub-episodes
return `0.5` for backward compatibility.

Used in bank maintenance and GRPO context:
- **Retirement / refinement thresholds** — still use `skill_score` heuristics as before
- **Curator candidates** — [`_collect_curator_candidates`](skill_agents_grpo/bank_maintenance/run_bank_maintenance.py) fills `skill_score` via `compute_skill_score()`
- **Segmentation GRPO** — pipeline passes per-skill scores as `bank_skill_scores`
- **Tests**: [`tests/test_quality_scoring.py`](tests/test_quality_scoring.py)

Skill selection uses a **UCB exploration bonus** to encourage trying
under-selected skills:

```
confidence = exploit + explore
exploit = 0.40 * relevance + 0.30 * applicability + 0.30 * pass_rate
explore = 0.15 * sqrt(ln(N_total + 1) / (n_skill + 1))
```

A `SelectionTracker` (per-iteration counter) tracks selection counts. Reset at
the start of each co-evolution iteration.

The **skill_selection GRPO reward** uses a delayed, per-skill-period signal
(assigned at skill-switch time) instead of raw per-step env reward:

```
r = min(1.0, max(0.0, cumulative_env_reward / steps_on_skill))
```

---

# 5. Trainer code

The training code lives in **[trainer/](trainer/)** and implements co-evolution between two agents on **8 × A100-80GB GPUs**: a **Decision Agent** (Qwen3-8B, GRPO, GPUs 0-3) that plays games and a **Skill Bank Agent** (Qwen3-8B, 3 LoRA adapters + Hard-EM, GPUs 4-7) that discovers and maintains reusable skills from the Decision Agent's trajectories. Both agents improve each other over multiple co-evolution iterations.

## Decision Agent (Agent A) — Qwen3-8B + LoRA GRPO

The Decision Agent is **LLM-only**: it consumes **text** state summaries from wrappers (NL observations, structured `key=value` fields), not raw images or a vision encoder. It selects primitive game actions and tool calls (`QUERY_SKILL`, `CALL_SKILL`, `QUERY_MEM`) against the current skill bank. **Default training:** **GRPO** with group-normalized advantages over frozen Qwen3-8B + LoRA, implemented as **multi-GPU FSDP** in [`skill_agents_grpo/grpo/fsdp_trainer.py`](skill_agents_grpo/grpo/fsdp_trainer.py) (invoked from [`trainer/coevolution/grpo_training.py`](trainer/coevolution/grpo_training.py)). **Optional:** VERL / `GameAITrainer` (`RayPPOTrainer`, `adv_estimator=grpo`) when using [verl-agent](https://github.com/verl-project/verl-agent).

**Note:** The skill-bank FSDP trainer does **not** wrap the full model in `torch.compile(dynamic=True)` (PyTorch Inductor can fail on these graphs); `torch.set_float32_matmul_precision("high")` is still applied.

**Training settings** (targets; see co-evolution config and [`scripts/configs/decision_agent_grpo_80gb.yaml`](scripts/configs/decision_agent_grpo_80gb.yaml) for YAML-aligned hyperparameters):

| Parameter | Value |
|-----------|-------|
| Base model | Qwen/Qwen3-8B |
| LoRA rank / alpha | 16 / 32 |
| GRPO group size | 8 |
| Clip ratio | 0.2 |
| KL coeff | 1e-2 (low-var KL penalty) |
| Learning rate | 1e-5 (AdamW, warmup 3%) |
| Max prompt / response length | 8192 / 4096 |
| Rollout batch size | 64 |
| Training steps per iteration | 20 |
| GPUs | GPUs 0-3 (FSDP + vLLM rollout, TP=2) |

**Reward**: `r_total = r_env + 0.1 × r_follow + r_cost + 0.1 × r_tool`
- `r_env` — raw environment reward
- `r_follow` — skill-following shaping (predicate satisfaction progress toward active skill's `eff_add`)
- `r_cost` — negative costs for retrieval queries, skill calls, and skill switching
- `r_tool` — tool-call quality reward from `skill_agents.tool_call_reward`

**Components**: [trainer/coevolution/grpo_training.py](trainer/coevolution/grpo_training.py) (orchestration → FSDP GRPO), [skill_agents_grpo/grpo/fsdp_trainer.py](skill_agents_grpo/grpo/fsdp_trainer.py), [trainer/decision/env_wrapper.py](trainer/decision/env_wrapper.py), [trainer/decision/reward_shaping.py](trainer/decision/reward_shaping.py), [trainer/decision/grpo_trainer.py](trainer/decision/grpo_trainer.py) (standalone `GRPOTrainer` + VERL notes), [trainer/decision/replay_buffer.py](trainer/decision/replay_buffer.py), [trainer/coevolution/rollout_collector.py](trainer/coevolution/rollout_collector.py).

## Skill Bank Agent (Agent B) — Qwen3-8B + 3 LoRA (Hard-EM)

The Skill Bank Agent processes trajectory rollouts from the Decision Agent through a 4-stage pipeline, with 3 GRPO-trained LoRA adapters on a shared Qwen3-8B backbone. Stage 1 (boundary) and retrieval are algorithmic / RAG — not GRPO-wrapped.

```
┌─────────────────────────────────────────┐
│       Qwen3-8B  (shared backbone)        │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐  │
│  │ SEGMENT  │ │ CONTRACT │ │ CURATOR  │  │  ← Stage 2 label preferences
│  │   LoRA   │ │   LoRA   │ │   LoRA   │  │  ← Stage 3 effect contracts
│  └──────────┘ └──────────┘ └──────────┘  │  ← Stage 4 approve/veto/defer
└─────────────────────────────────────────┘
```

**LoRA training settings** ([`scripts/skillbank_agent_train.sh`](scripts/skillbank_agent_train.sh)):

| Parameter | Value |
|-----------|-------|
| Base model | Qwen/Qwen3-8B |
| LoRA rank / alpha | 16 / 32 |
| Learning rate | 2e-4 |
| Epochs | 3 |
| Batch size / grad accum | 4 / 4 (effective 16) |
| Max sequence length | 2048 |
| GPUs | GPUs 4-7 |

**EM pipeline** (3 iterations per co-evolution step):
1. **Stage 0** — predicate extraction (booleanize observations)
2. **Stage 1** — boundary proposal (predicate changes + surprisal signals, algorithmic)
3. **Stage 2** — segmentation decode (DP/Viterbi with **SEGMENT** LoRA ranking, top-10 candidates, segments 3–100 steps)
4. **Stage 3** — contract learning (**CONTRACT** LoRA generates effect summaries, verified against holdout, pass rate ≥ 0.6)
5. **Stage 4** — bank maintenance (**CURATOR** LoRA approves/vetoes/defers **SPLIT, MERGE, REFINE, MATERIALIZE, PROMOTE** proposals)
6. **SkillEval gating** — accept/reject: avg pass rate ≥ 0.6, NEW rate ≤ 0.3, margin regression tolerance 0.1

**Components**: [trainer/skillbank/em_trainer.py](trainer/skillbank/em_trainer.py) (with segmentation store), [trainer/skillbank/stages/](trainer/skillbank/stages/), [trainer/skillbank/bank_io/](trainer/skillbank/bank_io/) (versioned store, indices, diff logger), [trainer/skillbank/learners/](trainer/skillbank/learners/) (boundary classifier, tie-breaker). LoRA routing via [skill_agents_grpo/lora/](skill_agents_grpo/lora/) — `SkillFunction` enum, `MultiLoraSkillBankLLM`, adapter routing. LLM calls use [skill_agents/_llm_compat.py](skill_agents/_llm_compat.py) for reasoning-model compatibility (`/no_think`, think-tag stripping).

See [skill_agents_grpo/README.md](skill_agents_grpo/README.md) for the full GRPO skill bank pipeline and [TODO_Lists/SKILLBANK_GRPO_PLAN.md](TODO_Lists/SKILLBANK_GRPO_PLAN.md) for the training plan. **Reward design** for SEGMENT / CONTRACT / CURATOR adapters is documented under [§ GRPO reward functions (per adapter)](#grpo-reward-functions-per-adapter) (decision-agent section).

### LoRA input / output examples (illustrative)

Formats match runtime prompts in [`episode_runner.py`](trainer/coevolution/episode_runner.py) (decision) and [`skill_agents_grpo/`](skill_agents_grpo/) (skill bank). Training concatenates prompt + completion as a single SFT text or GRPO (prompt, completion) pair.

---

#### 1. `skill_selection` (Decision — pick one strategy)

**Input (user message = system instructions + state + candidates):** the model first sees the strict format rule, then game state, intention, and numbered strategies (name, strategy, plan, confidence).

```
You are an expert game strategist. Given the current game state and a set of candidate strategies, choose the ONE strategy most likely to make progress.

Output format (strict):
REASONING: <1-2 sentences why this strategy fits the current state>
SKILL: <number>

Game state:
board=4x4 | max_tile=128 | empty=6 | note=corner strategy

Current intention: [OPTIMIZE] build toward 2048

Available strategies (pick ONE by number):
  1. merge_high_tiles
     Strategy: Combine large tiles on one edge
     Plan: sweep -> compress -> merge -> ...
     Confidence: 0.72
  2. recover_from_stall
     Strategy: Unblock when no moves along primary edge
     Plan: probe -> rotate focus -> ...
     Confidence: 0.55

Choose the best strategy. Output REASONING then SKILL number.
```

**Output (assistant):**

```
REASONING: State still has room to stack high tiles on the bottom row; primary strategy matches current layout.
SKILL: 1
```

---

#### 2. `action_taking` (Decision — subgoal + primitive action)

**Input:** expert rules, optional **Active skill** block (protocol steps, progress, success/abort), then game state, previous subgoal, recent actions/rewards, numbered env actions.

```
You are an expert game-playing agent. You receive a game state and must choose exactly one action by its NUMBER.
...
Output format (strict):
SUBGOAL: [TAG] <your immediate objective in ≤15 words>
REASONING: <1-2 sentences>
ACTION: <number>

--- Active Skill ---
Skill: merge_high_tiles
Preconditions: empty_cells >= 1
Progress: 2/4 plan steps done
Plan:
  1. [done] Press primary direction to align
  2. [current] Merge same-rank adjacent tiles
  ...
Success: max_tile increased OR board more ordered
Abort: no legal moves for 3 steps

Game state:
...

Previous subgoal: [OPTIMIZE] keep high tile in corner
Active skill: merge_high_tiles — stack and merge on bottom edge

Recent: left(+4), left(0), down(+8)

Available actions (pick ONE by number):
  1. up
  2. down
  3. left
  4. right

Subgoal tags: SETUP|EXECUTE|OPTIMIZE|...
First state your SUBGOAL, then choose the best action.
Output SUBGOAL, REASONING, then ACTION number.
```

**Output (assistant):**

```
SUBGOAL: [EXECUTE] merge the two 64s on the bottom row
REASONING: Down aligns the pair so the next left merge clears the row.
ACTION: 2
```

---

#### 3. `segment` (Skill bank — Stage 2 segmentation preferences)

**Input:** segment span, observations/actions, predicates, candidate skill IDs (with optional descriptions). Same family as [`_build_segment_ranking_prompt`](skill_agents_grpo/infer_segmentation/llm_teacher.py) / pairwise variants.

```
You are an expert at recognizing skills in agent trajectories.

A trajectory segment spans timesteps 12 to 28 (length 17).

Observations:
['grid 4x4 ...', 'score=120', ...]

Actions:
['left', 'left', 'down', ...]

State at segment start: {"empty": "6", "max": "32"}
State at segment end:   {"empty": "4", "max": "64"}

Candidate skills:
  - "SKILL_A": clear_bottom_row
  - "SKILL_B": build_corner_stack
  - "__NEW__"

Rank ALL candidate skills from best fit to worst fit for this segment.
...
Return ONLY a JSON object (no extra text):
{"ranking": ["best_skill", "second_best", ...], "reasoning": "brief explanation"}
```

**Output (assistant):**

```json
{"ranking": ["SKILL_B", "SKILL_A", "__NEW__"], "reasoning": "Actions consistently push tiles toward one corner before merging."}
```

(Pairwise mode returns `{"choice": "A" or "B", "evidence": "..."}`.)

---

#### 4. `contract` (Skill bank — Stage 3 effect summary)

**Input:** from [`_CONTRACT_PROMPT_TEMPLATE`](skill_agents_grpo/stage3_mvp/llm_contract.py).

```
You are analyzing skill effects from game trajectory segments.

Skill: SKILL_B
Number of instances: 8

Representative segment observations:
"max=32 empty=6"; "max=64 empty=4"; ...

State predicates at segment start: ["empty=6", "max=32"]
State predicates at segment end: ["empty=4", "max=64"]

Summarize the effects of this skill as a JSON object:
{"eff_add": ["predicates that become true"], "eff_del": ["predicates that become false"], "description": "one-line description"}
```

**Output (assistant):**

```json
{"eff_add": ["max_tile_increased", "tiles_merged"], "eff_del": [], "description": "Increases highest tile by merging aligned same-value cells."}
```

---

#### 5. `curator` (Skill bank — Stage 4 maintenance filter)

**Input:** from [`_CURATOR_PROMPT_TEMPLATE`](skill_agents_grpo/bank_maintenance/llm_curator.py). The pipeline proposes **five** action kinds: **SPLIT**, **MERGE**, **REFINE**, **MATERIALIZE**, **PROMOTE** — the curator sees a batch of proposed rows (often one candidate per type in a full maintenance pass, or any subset).

```
You are a skill bank maintenance curator. Review the proposed actions and decide whether to approve, veto, or defer each one.

## Bank Summary
Total skills: 42
Mean pass rate: 0.71
Skills with low pass rate (<0.60): 5

## Proposed Actions

  Action 0: SPLIT on SKILL_A
    Trigger: bi-modal behavior in instances
    Instances: 18
    ...

  Action 1: MERGE on SKILL_B
    Skill score: 0.88
    Pass rate: 0.75
    Instances: 12
    ...

  Action 2: REFINE on SKILL_C
    Pass rate: 0.52
    Instances: 20
    ...

  Action 3: MATERIALIZE on __NEW__
    Instances: 8
    ...

  Action 4: PROMOTE on __NEW__
    Pass rate: 0.68
    Instances: 11
    ...

Action types: SPLIT, MERGE, REFINE, MATERIALIZE, PROMOTE.

For each action, respond with a JSON object:
{"decisions": [{"idx": 0, "verdict": "approve|veto|defer", "reason": "brief reason"}, ...]}
...
```

**Output (assistant):** one verdict per proposed action (`idx` 0 … 4 when five are listed).

```json
{"decisions": [
  {"idx": 0, "verdict": "defer", "reason": "Need more segments before splitting."},
  {"idx": 1, "verdict": "approve", "reason": "Clear duplicate with SKILL_D; metrics support merge."},
  {"idx": 2, "verdict": "approve", "reason": "Low pass rate; refine contract is justified."},
  {"idx": 3, "verdict": "defer", "reason": "Only 8 instances for MATERIALIZE."},
  {"idx": 4, "verdict": "approve", "reason": "Proto-skill stable enough to promote."}
]}
```

---

## Co-Evolution Pipeline

The co-evolution loop alternates between training both agents. Each iteration: collect rollouts → update Skill Bank → train Decision Agent → repeat.

```
Iteration 1:
  cold-start rollouts (base Qwen3-8B)
    → Skill Bank v1 (LoRA training + Hard-EM on GPUs 4-7)
    → Decision Agent v1 (GRPO on GPUs 0-3, with Bank v1)

Iteration 2+:
  rollouts with Decision Agent v_{i-1}
    → Skill Bank v_i (LoRA + EM on new rollouts)
    → Decision Agent v_i (GRPO with updated bank)
```

**Main co-evolution loop**: [`trainer/coevolution/orchestrator.py`](trainer/coevolution/orchestrator.py) — rollouts (vLLM) → Skill Bank EM → **FSDP GRPO** on decision + skill-bank adapters → repeat. Phase C GRPO: [`trainer/coevolution/grpo_training.py`](trainer/coevolution/grpo_training.py).

**VERL callback (optional)**: [trainer/decision/coevolution_callback.py](trainer/decision/coevolution_callback.py) — `SkillBankCoEvolutionCallback` hooks **Skill Bank EM** into `GameAITrainer.fit()` when using verl-agent; `SkillAgentToolPipeline`, `SegmentationStore`, bank hot-swap in workers.

**Legacy / alternate entry**: [trainer/launch_coevolution.py](trainer/launch_coevolution.py) — standalone Decision GRPO + periodic Skill Bank EM with eval gating.

### SFT Cold-Start Training

Before running co-evolution GRPO, all 5 LoRA adapters can be warm-started via **supervised fine-tuning** on teacher-labelled cold-start data. This gives the GRPO loop a non-random starting point, significantly improving early-step performance.

**Module:** [`trainer/SFT/`](trainer/SFT/) — `SFTConfig`, `load_all_adapter_datasets`, `train_all_adapters`

**LoRA config** is identical to the co-evolution GRPO pipeline so adapters load directly without conversion:

| Parameter | Value |
|-----------|-------|
| Base model | Qwen/Qwen3-8B |
| LoRA rank / alpha | 16 / 32 |
| Dropout | 0.05 |
| Target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj` |
| Batch size / grad accum | 4 / 4 (effective 16) |
| Max sequence length | 2048 |
| Warmup | 5% |
| Precision | bf16 |

**Per-adapter training settings** — epochs and LR scale inversely with dataset size so small-data adapters get enough gradient steps while large decision adapters don't over-train:

| Adapter | Category | Examples | Epochs | LR | Data source |
|---------|----------|----------|--------|-----|-------------|
| `skill_selection` | Decision | ~34k | 3 | 2e-4 | [`grpo_coldstart/*/skill_selection.jsonl`](labeling/output/gpt54_skill_labeled/grpo_coldstart/) |
| `action_taking` | Decision | ~34k | 3 | 2e-4 | [`grpo_coldstart/*/action_taking.jsonl`](labeling/output/gpt54_skill_labeled/grpo_coldstart/) |
| `segment` | Skill Bank | ~2.7k | 8 | 2e-4 | [`gpt54_skillbank_grpo/*/teacher_io_coldstart.jsonl`](skill_agents_grpo/extract_skillbank/output/gpt54_skillbank_grpo/) |
| `contract` | Skill Bank | ~1.6k | 10 | 2e-4 | [`gpt54_skillbank_grpo/*/coldstart_io_all.jsonl`](skill_agents_grpo/extract_skillbank/output/gpt54_skillbank_grpo/) (modules: `boundary_proposal`, `pipeline`) |
| `curator` | Skill Bank | ~216 | 15 | 1e-4 | [`gpt54_skillbank_grpo/*/coldstart_io_all.jsonl`](skill_agents_grpo/extract_skillbank/output/gpt54_skillbank_grpo/) (module: `skill_naming`) |

**Data format alignment with co-evolution** — the SFT data loader transforms cold-start data so each adapter's prompt/completion format exactly matches what it will see during co-evolution GRPO:

| Adapter | Co-evolution I/O format | Cold-start alignment |
|---------|------------------------|----------------------|
| `skill_selection` | `REASONING: <text>\nSKILL: <N>` | Exact match — same prompt builder |
| `action_taking` | `SUBGOAL: [TAG] <objective>\nREASONING: <text>\nACTION: <N>` | Transformed at load time: SUBGOAL instructions injected into prompt, `intention` metadata prepended to completion (cold-start labels lack SUBGOAL) |
| `segment` | `segment_ranking` / `transition_ranking` / `pairwise_choice` → JSON ranking | Exact match — same `_build_segment_ranking_prompt` generates both |
| `contract` | `_CONTRACT_PROMPT_TEMPLATE` → `{"eff_add": [...], "eff_del": [...], "description": "..."}` | Proxy data: cold-start extraction did not run Stage 3 contract learning, so predicate analysis (`boundary_proposal`) and protocol synthesis (`pipeline`) provide domain-proximate training; GRPO refines to the actual task |
| `curator` | `_CURATOR_PROMPT_TEMPLATE` → `{"decisions": [{"idx": N, "verdict": "approve\|veto\|defer", "reason": "..."}]}` | Proxy data: cold-start extraction did not run bank maintenance, so skill naming data provides approximate domain overlap; GRPO refines to the actual curation task |

**SFT data examples** — one sample per adapter showing the prompt → completion format the model learns:

<details>
<summary><b>skill_selection</b> — pick the best strategy from candidates</summary>

**Prompt:**
```
You are an expert game strategist. Given the current game state and
candidate strategies, choose the ONE strategy most likely to make progress.

Output format (strict):
REASONING: <1-2 sentences why this strategy fits>
SKILL: <number>

Game state:
game=candy crush | phase=opening | step=0/50 | score=0 | moves=50 | board=8x8 | pairs=18 | reward=+3

Current intention: [CLEAR] Make the immediate match-producing swap for quick points and flexible board structure

Available strategies (pick ONE by number):
  1. Clear Immediate Match
     Strategy: Swap two adjacent candies to make an immediate 3+ match already
       available on the board. Use this in the endgame when a clear, guaranteed
       match helps clear remaining objectives without wasting moves.
     Plan: Scan the board row by row and column by column for near-matches where
       moving one adjacent candy completes a 3+ line → Execute the swap →
       Confirm the match and check for cascades.
  2. Build Cascade Setup
     ...
  3. Quick Opening Match
     ...

Choose the best strategy.
```

**Completion:**
```
REASONING: The game is at the opening state, and strategy 3 is the only one
explicitly grounded in the same opening condition, making it the best fit
for immediate progress.
SKILL: 3
```
</details>

<details>
<summary><b>action_taking</b> — select a game action given a skill plan</summary>

**Prompt:**
```
You are a game-playing agent. Pick the best action for the current state.

Output SUBGOAL, REASONING, then ACTION number.

--- Active Skill: Make Immediate Match ---
  Strategy: game=candy crush | phase=opening | step=0/50 | score=0 |
    moves=50 | board=8x8 ...
--- end skill ---

Game state:

game=candy crush | phase=opening | step=0/50 | score=0 | moves=50
Board (8×8): ...

Available actions:
  0. Swap (0,0)↔(0,1)
  1. Swap (0,1)↔(0,2)
  ...
  5. Swap (1,2)↔(2,2)
  ...
```

**Completion:**
```
SUBGOAL: [CLEAR] Make the immediate match-producing swap for quick points
  and flexible board structure
REASONING: Expert play.
ACTION: 5
```
</details>

<details>
<summary><b>segment</b> — rank skills that best explain a trajectory segment</summary>

**Prompt:**
```
You are an expert at recognizing skills in agent trajectories.

A trajectory segment spans timesteps 7 to 18 (length 12).

Observations:
['game=tetris | phase=opening | step=7/86 | stack_h=6 | holes=8 | piece=J
  | next=Z,T,S,L | level=1 | reward=+1', ...]

Actions:
['move_right', 'rotate_clockwise', 'hard_drop', ...]

State at segment start: {'intention': '[SETUP] Place J to support upcoming
  pieces', ...}
State at segment end:   {'intention': '[POSITION] Use T to patch holes and
  stabilize', ...}

Candidate skills:
  - "endgame:CLEAR": Tuck T Smoothly — Use a T piece to slide into a tight
      notch under an overhang, flattening the surface while preserving access
  - "endgame:SURVIVE": Keep Stack Safe — Use this when the stack is tall or
      messy: place pieces to flatten the surface, seal exposed holes
  - "midgame:CLEAR": Lay Z Flat — Place the Z piece horizontally on a flat,
      even surface so it doesn't create overhangs or new holes
  - "midgame:POSITION": Keep Surfaces Flat — Place pieces to keep the stack
      low and the top surface as even as possible
  - "opening:SETUP": Keep Stack Low — In the opening, place pieces flat and
      clear singles or doubles early to keep the stack only 2-3 rows high
  ...

Rank ALL candidate skills from best fit to worst fit for this segment. Consider:
  - Do the actions match what this skill would produce?
  - Is the segment length reasonable for this skill?
  - Is the state change consistent with this skill's purpose?

Return ONLY a JSON object (no extra text):
{"ranking": ["best_skill", "second_best", ...], "reasoning": "brief explanation"}
```

**Completion:**
```json
{"ranking": ["opening:SETUP", "midgame:POSITION", "midgame:SETUP",
  "midgame:NAVIGATE", "midgame:CLEAR", "midgame:SURVIVE", "midgame:DEFEND",
  "midgame:OPTIMIZE", "endgame:CLEAR", "endgame:SURVIVE"],
 "reasoning": "This is in the opening phase with board-shaping goals: placing
  J to support upcoming pieces, then using T to patch holes and stabilize."}
```
</details>

<details>
<summary><b>contract</b> — generate execution protocols (proxy task for effect summarization)</summary>

**Prompt:**
```
You are a game-AI protocol designer. Generate a concrete execution protocol
for the skill below.

Skill: mid_quests:SETUP
Description: (none)
Effects: achieves: phase=endgame, quest=4, quest=5;
         removes: phase=midgame, quest=3, team_size=2
Evidence from successful executions:
  - game=avalon | phase=opening | step=6/26 | quest=2 | round=2 | role=Merlin

Generate a structured protocol as JSON with these exact keys:
{"preconditions": ["..."], "steps": ["..."],
 "success_criteria": ["..."], "abort_criteria": ["..."]}
```

**Completion:**
```json
{"preconditions": ["The game is Avalon and the table is transitioning out
  of the opening into the midgame, with quest 2 resolved.",
  "Current state still reflects midgame setup conditions: quest 3 is not
  yet locked in.", "At least one of quests 4 or 5 has not yet been staged."],
 "steps": ["Evaluate previous quest outcomes to identify suspect players.",
  "Propose a 3-player team excluding suspects.", ...],
 "success_criteria": ["Phase transitions to endgame.",
  "Quest 4 or 5 is resolved."],
 "abort_criteria": ["Two consecutive failed quests."]}
```

*Note:* Cold-start extraction did not run Stage 3 contract learning.
This proxy data (boundary analysis + protocol synthesis) teaches structured
JSON generation in the game domain; GRPO refines to the actual effect
summarization task.
</details>

<details>
<summary><b>curator</b> — name and summarize skills (proxy task for approve/veto/defer)</summary>

**Prompt:**
```
Game: avalon
Skill ID: early_quests:POSITION
Effects added: quest=2
Effects removed: team_size=2
Events: none
Sample intentions from segments:
  [SETUP] Interpret Player 2's opening team proposal [2,0] to gauge alignment
  [CLEAR] Assess Player 0's motives and alignment from opening-team context
  [SETUP] Lead quest 1 with yourself and one other to gather early information

Generate:
1. A short skill name (2-5 words, imperative verb phrase)
2. A compact RAG summary in key=value format for embedding retrieval

Reply in this exact format:
NAME: <skill name>
SUMMARY: game=<game> | skill=<name> | effects=... | context=...
```

**Completion:**
```
NAME: Read Opening Team
SUMMARY: game=avalon | skill=Read Opening Team | effects=quest=2 |
  context=opening 2-player quest; assess proposer motives/alignment from
  first team proposal and approve/select a team to gather early information
```

*Note:* Cold-start extraction did not run bank maintenance (curation).
Skill naming shares domain overlap (evaluating skills) but differs from the
co-evolution approve/veto/defer format; GRPO bridges this gap.
</details>

**Output layout** (matches co-evolution `adapter_dir` structure):

```
runs/sft_coldstart/
├── decision/
│   ├── skill_selection/   # adapter_config.json + adapter_model.safetensors
│   └── action_taking/
└── skillbank/
    ├── segment/
    ├── contract/
    └── curator/
```

**Run SFT cold-start training:**

```bash
# Sequential: train all 5 adapters one after another (1 GPU)
bash scripts/run_sft_coldstart.sh

# PARALLEL: train all 5 adapters simultaneously (1 GPU per adapter, ~5x faster)
SFT_PARALLEL=1 bash scripts/run_sft_coldstart.sh

# Parallel on specific GPUs
SFT_PARALLEL=1 SFT_GPUS="0 1 2 3 4" bash scripts/run_sft_coldstart.sh

# Custom settings
SFT_EPOCHS=5 SFT_LR=1e-4 SFT_PARALLEL=1 bash scripts/run_sft_coldstart.sh

# Train a subset in parallel
SFT_PARALLEL=1 SFT_ADAPTERS="segment contract curator" bash scripts/run_sft_coldstart.sh

# Or use the Python module directly
python -m trainer.SFT.train --parallel                         # all 5, auto-detect GPUs
python -m trainer.SFT.train --parallel --gpus 0 1 2 3 4        # explicit GPU assignment
python -m trainer.SFT.train --adapters segment curator --parallel
```

Parallel mode spawns one subprocess per adapter, each pinned to a separate GPU via `CUDA_VISIBLE_DEVICES`. Qwen3-8B in bf16 uses ~16GB, so each adapter fits on a single GPU. With 5+ GPUs, wall-clock time equals the slowest adapter (~34k-example decision adapters) rather than the sum of all five. Per-adapter logs are saved to `<output_dir>/sft_gpu<N>.log`.

If fewer GPUs than adapters are available, the launcher automatically distributes multiple adapters per GPU (trained sequentially on that GPU, in parallel with other GPUs).

**Feed SFT adapters into co-evolution GRPO:**

```bash
python scripts/run_coevolution.py \
    --load-decision-adapters  runs/sft_coldstart/decision \
    --load-skillbank-adapters runs/sft_coldstart/skillbank
```

**Programmatic usage:**

```python
from trainer.SFT import SFTConfig, train_all_adapters

config = SFTConfig(output_dir="runs/sft_coldstart", epochs=3, lr=2e-4)
results = train_all_adapters(config)            # sequential
results = train_all_adapters(config, gpu=2)     # sequential on specific GPU
```

### Training Scripts

| Script | Purpose |
|--------|---------|
| [`scripts/run_sft_coldstart.sh`](scripts/run_sft_coldstart.sh) | SFT cold-start: train all 5 LoRA adapters from teacher-labelled data (run before co-evolution) |
| [`scripts/coevolution_train.sh`](scripts/coevolution_train.sh) | Main loop: cold-start rollouts → Skill Bank v1 → Decision Agent v1 → iterate (default 6 iterations) |
| [`scripts/decision_agent_train.sh`](scripts/decision_agent_train.sh) | Decision Agent GRPO on GPUs 0-3 (**FSDP** + vLLM rollout per script; not VERL) |
| [`scripts/skillbank_agent_train.sh`](scripts/skillbank_agent_train.sh) | LoRA training + Hard-EM for Skill Bank Agent (Qwen3-8B) on GPUs 4-7 |
| [`cold_start/run_100_rollouts.py`](cold_start/run_100_rollouts.py) | Batch rollout generation (100 episodes per game) |

```bash
# Full pipeline: SFT cold-start → co-evolution GRPO
bash scripts/run_sft_coldstart.sh
python scripts/run_coevolution.py \
    --load-decision-adapters  runs/sft_coldstart/decision \
    --load-skillbank-adapters runs/sft_coldstart/skillbank

# Or skip SFT and train from scratch (gaussian LoRA init)
bash scripts/coevolution_train.sh

# Custom co-evolution:
Decision_base_model=Qwen/Qwen3-8B \
SkillBank_base_model=Qwen/Qwen3-8B \
NUM_ITERATIONS=10 TRAIN_STEPS=30 \
  bash scripts/coevolution_train.sh
```

### Shared Infrastructure

- **Rollout schema**: [trainer/common/metrics.py](trainer/common/metrics.py) — `RolloutRecord` / `RolloutStep` (single source of truth). Carries `episode_id`, `env_name`, `game_name`.
- **Reward contract**: [trainer/decision/reward_shaping.py](trainer/decision/reward_shaping.py) — `compute_reward(prev, action, next, bank_state)` → r_env, r_follow, r_cost, r_tool, r_total.
- **Eval & logging**: [trainer/common/eval_harness.py](trainer/common/eval_harness.py), [trainer/common/logging.py](trainer/common/logging.py), [trainer/common/seeds.py](trainer/common/seeds.py).
- **Configs**: [scripts/configs/decision_agent_grpo_80gb.yaml](scripts/configs/decision_agent_grpo_80gb.yaml), [trainer/common/configs/skillbank_em.yaml](trainer/common/configs/skillbank_em.yaml).

See [trainer/README.md](trainer/README.md) for full layout, milestones, metrics, and implementation order.

### Inference Speed (not yet implemented)

The skill bank pipeline currently runs all ~572 LLM calls per EM step serially (batch size 1 via HuggingFace `.generate()`), taking ~47 min per iteration. The planned fix is **vLLM serving with multi-LoRA + async batching**, which would bring this down to ~4 min per iteration (~12× speedup). With cross-stage pipeline parallelism (micro-batch overlap across stages), the target is ~3 min per EM iteration. See [TODO_Lists/SKILLBANK_INFERENCE_SPEED.md](TODO_Lists/SKILLBANK_INFERENCE_SPEED.md) for the full analysis, implementation checklist, and GPU memory budget.

### Recent Bug Fixes (Co-Evolution Pipeline)

The following bugs prevented the co-evolution training pipeline from completing successfully. All have been fixed:

| # | Bug | Location | Root Cause | Fix |
|---|-----|----------|------------|-----|
| 1 | **6/9 game envs fail on construction** | `cold_start/generate_cold_start.py` | `ColdStartEnvWrapper` passed `render_mode=None` + adapter kwargs to all envs, but CandyCrush, Doom, SuperMarioBros, AceAttorney, NineteenFortyTwo, and PokemonRed have different constructor signatures | Added per-env `init_kwargs` factories in `GAME_REGISTRY` that return the correct kwargs for each env type |
| 2 | **JSONL consolidation produces 0-byte file** | `scripts/coevolution_train.sh` | `glob.glob('episode_*.json')` searched the top-level rollout dir, but episodes are in per-game subdirs (e.g. `rollouts_v0/tetris/episode_000.json`) | Changed to recursive glob with `**` pattern |
| 3 | **Skillbank ingest fails on `import pandas`** | `scripts/skillbank_agent_train.sh` | The JSONL codepath doesn't need pandas, but the `except ImportError` fallback re-imported it; also no validation on empty files | Restructured: pandas only for `.parquet`; added minimal JSONL fallback; added upfront file-exists and non-empty checks |
| 4 | **`train_lora_adapter()` doesn't exist** | `trainer/skillbank/lora/train_lora.py` | Shell script imports `train_lora_adapter` but module only had CLI-based `train(args)` | Added `train_lora_adapter()` wrapper function that constructs an `argparse.Namespace` and delegates to `train()` |
| 5 | **Exit codes masked by `\| tee`** | `scripts/skillbank_agent_train.sh`, `scripts/decision_agent_train.sh` | `python3 -c "..." \| tee log` returns tee's exit code (0), hiding Python failures | Added `set -o pipefail`; added `PIPESTATUS[0]` checks after ingest, LoRA, and EM steps |
| 6 | **Empty file passes skip-check** | `scripts/coevolution_train.sh` | `[ -f file ]` checks existence, not content; a 0-byte JSONL from failed consolidation passed the guard | Changed all rollout checks from `-f` to `-s` (exists AND non-empty); added cleanup of stale empty files |
| 7 | **Multi-LoRA model never initialized for EM** | `scripts/skillbank_agent_train.sh`, `trainer/launch_coevolution.py` | LoRA adapter paths were wired into config but `MultiLoraSkillBankLLM` was never instantiated or registered as singleton | Added `MultiLoraSkillBankLLM` initialization + `set_shared_instance()` before EM runs |
| 8 | **CONTRACT and RETRIEVAL adapters not wired** | EM stages 2 and 3 | Only BOUNDARY and SEGMENT had consumer call sites; CONTRACT and RETRIEVAL adapters were trained but unused | Added `llm_contract.py` (Stage 3 enrichment) and `llm_retrieval.py` (Stage 2 re-ranking) with graceful fallback |
| 9 | **Phase detectors broken for 5/8 games → skill banks collapse to 1 skill** | `skill_agents_grpo/infer_segmentation/phase_detector.py` | All game-specific phase extractors used `_get_state_str()` which returns the raw `state` field (verbose game text), but the extractors were designed for `summary_state` (parsed `key=value` format). Root causes per game: **Diplomacy** — regex expected `phase=S1901M` but raw state has `Phase: S1901M`; **Avalon** — raw state always contains "team" (from "Team Selection"/"Team Voting") → 100% `team_building`; **Pokemon Red** — raw state always contains "battle" (even "Not in battle") → 100% `battle`; **Tetris** — raw state is a text board, `_get_state_dict()` fails → 100% `mid`; **Super Mario** — Mario x-position constant within short episodes → 100% `early_level`; **Sokoban** — raw state table lacks "dark_storage"/"box_on_goal" labels. Only 2048 and Candy Crush worked (2048 has structured JSON state; Candy Crush uses temporal thirds). | Added `_get_summary_str()` helper that prefers `summary_state` over raw `state`. Updated all 6 broken extractors to use parsed `summary_state` fields: **Diplomacy** → regex on `phase=S1901M` codes (5 phases); **Avalon** → `quest=N` field (3 phases); **Pokemon Red** → `State: Battle/Field/Dialog` with raw-state fallback (2+ phases); **Tetris** → `stack_h=N` field (3 phases); **Super Mario** → `mario=(x,y)` with fallback to temporal thirds when span < 20px (3 phases); **Sokoban** → `solved=M/N` field (falls back to generic temporal thirds when no boxes solved). |

**After fixing, clean stale outputs before re-running:**
```bash
cd runs/coevolution
rm -f rollouts/*_rollouts_v0.jsonl
rm -rf temp_results/* lora_adapters/* logs/skillbank/*
```

---

## Reward shaping patches (Tetris + Sokoban)

Training analysis on `Qwen3-8B_20260320_085011` (30 steps) revealed that
Tetris (~11 mean reward) and Sokoban (~-1.2 mean reward) were stuck due to
structural reward problems. See [patches/README.md](patches/README.md)
(patches 004–005) for full details.

| Game | Problem | Fix | Expected impact |
|------|---------|-----|-----------------|
| **Sokoban** | Stuck detection killed episodes at step ~20; -0.1/step penalty drowned positive signals; GRPO had zero positive examples | Exempt from stuck detection; reduce penalty -0.1→-0.02; add Manhattan distance shaping (+0.1 per cell closer) | Episodes run full 200 steps; reward variance for GRPO |
| **Tetris** | Agent spam-dropped pieces (hard_drop=+1.0, positioning=0.0); no line clears ever; ~10 steps to game over | Penalize new holes (-0.3/hole); reward hole reduction (+0.2/hole); height penalty above 75% | Breaks reward trap; GRPO can distinguish good from bad placements |

Files changed: `episode_runner.py`, `sokobanEnv.py`, `tetrisEnv.py`.

---

## Baseline reward comparison caveat (GPT-5.4 vs training)

The GPT-5.4 baseline episodes in `labeling/output/gpt54_skill_labeled/` use a
**different reward accounting** than the co-evolution training runs.  Directly
comparing the numbers without normalization is misleading for Avalon and
Diplomacy.

### Root cause

| Aspect | GPT-5.4 baseline (`gpt54_skill_labeled`) | Co-evolution training (`step_log.jsonl`) |
|--------|------------------------------------------|------------------------------------------|
| **Agent view** | Multi-agent: rewards summed across **all** players/powers | Single-agent: reward for the **one** controlled player/power only |
| **Shaping** | Raw environment reward only (no `r_follow`, `r_cost`) | Shaped: `r_total = r_env + 0.1×r_follow + r_cost` |

### Per-game breakdown

**Avalon** (5 players: 3 Good, 2 Evil):
- Baseline records the terminal reward for **all 5 players** summed: Evil win = 2.0, Good win = 3.0 → baseline mean = **2.42**
- Training records reward for **1 controlled player** + shaping: win ≈ 1.0–1.3, lose ≈ -0.3–0.0 → training mean = **0.76–1.25**
- Fair single-agent baseline: 2.42 / 5 ≈ **0.48** — training exceeds this

**Diplomacy** (7 powers, each gets `supply_centers / 18` per phase):
- Baseline sums rewards across **all 7 powers** every phase: first step = 22 total SCs / 18 = 1.222, grows as powers gain territory → baseline total ≈ **35.3** over 20 phases
- Training records reward for **1 controlled power** + center-gain shaping (+0.5/center): a power with ~3 starting SCs accumulates **4–5.5** over 20 phases
- Fair single-agent baseline: 35.3 / 7 ≈ **5.04** — training matches this

### Comparable games (single-agent envs)

For Candy Crush, Tetris, 2048, and Sokoban the baseline and training use the
same single-agent reward (no multi-agent sum), so numbers are directly
comparable:

| Game | GPT-5.4 baseline | Training best | Delta |
|------|-------------------|---------------|-------|
| **Candy Crush** | 541.5 | 591.0 (step 11) | **+9.1%** |
| **2048** | 1145.4 | 1361.0 (step 26) | **+18.8%** |
| **Tetris** (macro) | 513.0 ± 333.9 | 510.9 ± 199.5 (step 4) | ~0% |
| **Sokoban** | -5.4 | -0.8 (step 24) | scale changed by patch |

**Tetris macro-action note:** Tetris uses the `TetrisMacroActionWrapper`
(placement-level actions: each decision places one entire piece at a chosen
rotation + column) instead of primitive moves. This matches the training env
chain (`make_gaming_env → GamingAgentNLWrapper → TetrisMacroActionWrapper`).
The old primitive-action numbers (GPT-5.4: 13.4, training: 13.3) are no longer
comparable. 95% CIs computed with t-distribution (N=8 episodes each).

---

## Protocol feasibility improvements

The protocol system was extended to produce more actionable, verifiable
guidance for the decision agent.  All changes are **backward-compatible** —
existing skill banks with old-format protocols continue to work without
modification.

### Changes overview

| # | Problem | Solution | Key file(s) |
|---|---------|----------|-------------|
| 1 | Steps didn't reference real game actions | LLM synthesis prompt now includes `action_vocab` when available | `pipeline.py` (`_llm_synthesize_protocol`) |
| 2 | Success/abort used fragile keyword matching | New `predicate_success` / `predicate_abort` fields (key=value, key<N, key>N) checked against `summary_state`; keyword matching is kept as fallback | `protocol_utils.py`, `_SkillTracker` |
| 3 | Steps advanced every timestep regardless of progress | New `step_checks` field — per-step conditions that must be met before advancing | `protocol_utils.py`, `_SkillTracker.update()` |
| 4 | `expected_duration` was arithmetic mean (outlier-sensitive) | Now uses **median** of sub-episode lengths, capped between 3 and 30 | `protocol_utils.compute_expected_duration` |
| 5 | Protocols were never revised after initial creation | `refine_low_pass_protocols()` re-synthesizes for skills with success rate < 40% | `pipeline.py` |
| 6 | Action prompt had no progress context | `build_progress_summary()` injects "Steps 1-2 done. Current: step 3 — ..." into the prompt | `protocol_utils.py`, `_format_skill_guidance_for_prompt` |

### New Protocol fields

All new fields are optional with empty-list defaults — old code that doesn't
know about them is unaffected.

```python
@dataclass
class Protocol:
    # ... existing fields ...
    step_checks: List[str]        # per-step completion predicates
    predicate_success: List[str]  # machine-checkable success conditions
    predicate_abort: List[str]    # machine-checkable abort conditions
    action_vocab: List[str]       # game actions available during synthesis
```

### Predicate format

Predicates are `key=value`, `key!=value`, `key>N`, `key<N`, `key>=N`, or
`key<=N`.  They are checked against the parsed `summary_state` string
(format: `key=value | key=value | ...`).

### Protocol refinement

Call `pipeline.refine_low_pass_protocols()` periodically (e.g. every 5
co-evolution iterations) to automatically re-synthesize protocols for
under-performing skills:

```python
refined = pipeline.refine_low_pass_protocols(
    pass_rate_threshold=0.4,
    min_episodes=3,
)
```
