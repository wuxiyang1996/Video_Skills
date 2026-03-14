# Enhance Agentic Decision-making in Multiple-player long-horizon games with unsupervised experiences

[![GitHub Repository](https://img.shields.io/badge/GitHub-Repository-blue)](https://github.com/wuxiyang1996/Game-AI-Agent)

## Overview

This repository provides a framework for enhancing agentic decision-making in multi-player, long-horizon games through unsupervised experience. The framework integrates with multiple game environments and supports both training-free (RAG-based) and trainable (RL-based) agent architectures. This readme outlines each module and aims to ease integration and debugging.

**No external repos are bundled.** This repository contains only Game-AI-Agent code. For Avalon/Diplomacy you need [AgentEvolver](https://github.com/modelscope/AgentEvolver) (clone as sibling or on `PYTHONPATH`). For GamingAgent evaluation, clone that repo as a sibling when needed; see [evaluate_gamingagent/setup_gamingagent_eval_env.md](evaluate_gamingagent/setup_gamingagent_eval_env.md). For **VERL training and inference** (vLLM/sglang, GiGPO/PPO), use [verl-agent](https://github.com/verl-project/verl-agent) as a sibling and see [INSTALL.md](INSTALL.md).

**Install:** See **[INSTALL.md](INSTALL.md)** for setup and VERL/verl-agent. Quick: add this repo to `PYTHONPATH`, or `conda env create -f environment.yml` then `conda activate game-ai-agent`.

**Contents:** 1. [Environments](#1-environments) · 2. [Data structure (skills and experiences)](#2-data-structure-skills-and-experiences) · 3. [Skill agent](#3-skill-agent) · 4. [Decision-making agent](#4-decision-making-agent) · 5. [Trainer code](#5-trainer-code) · [Implemented (done)](#implemented-done) · [ToDo (unfinished)](#todo-unfinished--future--consolidated)

## Quick Links

- **🔗 Repository**: [GitHub - Game-AI-Agent](https://github.com/wuxiyang1996/Game-AI-Agent)

- **📦 Environment Wrappers** — [env_wrappers/](env_wrappers/): NL wrappers and evaluation for game environments
  - [Avalon](env_wrappers/avalon_nl_wrapper.py) · [Diplomacy](env_wrappers/diplomacy_nl_wrapper.py) — require **AgentEvolver** (external; [AgentEvolver Games](https://github.com/modelscope/AgentEvolver/blob/main/games/README.md)); eval: [evaluation_evolver/](evaluation_evolver/)
  - [GamingAgent](env_wrappers/gamingagent_nl_wrapper.py) — LMGame-Bench (2048, Sokoban, Tetris); requires **GamingAgent** (external); eval: [evaluate_gamingagent/](evaluate_gamingagent/)

- **🔍 RAG & Embeddings** — [rag/](rag/): Text (default [Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)), multimodal (default [Qwen3-VL-Embedding-2B](https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B)); config: `RAG_EMBEDDING_MODEL`, `MULTIMODAL_EMBEDDING_MODEL`. [rag/README.md](rag/README.md)

- **🎮 Decision Agent** — [decision_agents/](decision_agents/): VLM step-by-step play with a **two-turn micro-loop** per timestep: (1) **take_action** — primitives or `QUERY_MEM` / `QUERY_SKILL` / `CALL_SKILL`; (2) **reward** — composite reward. **Per-step protocol:** `get_state_summary` (required) → optional `query_skill` or `query_memory` (budget-limited) → `take_action` (required) → `get_intention` (required) → `reward` (required). Skill bank supplies **protocol store** (name, steps, preconditions) for planning and **contract** (eff_add) for r_follow; `select_skill_from_bank` / `query_skill_bank` return protocol steps and use `bank.get_contract(skill_id)` for reward. **Model-agnostic:** same code path for GPT, Qwen, etc.; pass `model="gpt-4o-mini"` or `model="Qwen/Qwen3-14B"`; callers should pass an explicit `model`. See [decision_agents/README.md](decision_agents/README.md).
  - **Core**: [agent.py](decision_agents/agent.py) — `VLMDecisionAgent`, `run_tool()`, `run_episode_vlm_agent()`; [dummy_agent.py](decision_agents/dummy_agent.py) — game detection, action extraction.
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

- **🏋️ Training** — [trainer/](trainer/): Co-evolution of both agents via VERL.
  - **Agent A (Decision)**: GRPO — primitives + `QUERY_MEM` / `QUERY_SKILL` / `CALL_SKILL`; reward = r_env + shaping + costs + tool-call reward.
  - **Agent B (SkillBank)**: Hard-EM (decode → update → gate); all four stages packed as a tool pipeline in the [co-evolution callback](trainer/decision/coevolution_callback.py). Trajectory segmentations stored and updated via `SegmentationStore`.
  - **Co-evolution callback**: [coevolution_callback.py](trainer/decision/coevolution_callback.py) — `SkillBankCoEvolutionCallback` + `SkillAgentToolPipeline` + `SegmentationStore`; integrates `skill_agents.tool_call_reward` into reward shaping. On accepted EM update, [launch_coevolution.py](trainer/launch_coevolution.py) passes the training model into `SkillBankAgent` for protocol synthesis (same `ask_model` routing as inference). [launch_train](trainer/decision/launch_train.py) initializes `SkillQueryEngine` when loading the bank so training rollouts use the same retrieval path as inference.
  - Shared: [metrics](trainer/common/metrics.py), [reward_shaping](trainer/decision/reward_shaping.py), [eval_harness](trainer/common/eval_harness.py). Entry: [launch_train](trainer/decision/launch_train.py), [launch_coevolution](trainer/launch_coevolution.py). [trainer/README.md](trainer/README.md)

- **▶️ Inference** — [inference/](inference/): Run the decision agent and store rollouts in [data_structure](data_structure/experience.py) format (`Episode` + `Experience`). **Unified skill bank path:** Both `scripts/run_inference.py` (any model via `--model`) and `scripts/run_qwen3_14b_eval.py` (Qwen, `--bank` optional) use the same `load_skill_bank()`, `select_skill_from_bank()`, and `skill_bank_to_text()`; only the LLM backend differs. `run_episode_vlm_agent()` returns `Episode` directly; `run_inference()` wraps it with buffer/save support. [inference/README.md](inference/README.md)

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

`run_episode_vlm_agent()` reads `info["env_name"]` and `info["game_name"]` to populate the `Episode`. Fallback chain: `info["game"]` → `detect_game(obs)` → `info["structured_state"]["game"]`.

---

## Decision & Skill Agents: Co-evolution

Two agents co-evolve: the **Decision Agent** (Qwen3-14B + LoRA, GRPO) plays games using skills from the bank, and the **Skill Bank Agent** (Qwen3-8B + 3 LoRA, Hard-EM) discovers and refines skills from the Decision Agent's trajectories. Training runs on **8 × A100-80GB GPUs** (GPUs 0-3 for Decision, GPUs 4-7 for Skill Bank). See [Section 5: Trainer code](#5-trainer-code) for full training settings and scripts.

### Big picture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                          CO-EVOLUTION LOOP                                    │
│                                                                              │
│  Decision Agent (Qwen3-14B + LoRA)       Skill Bank Agent (Qwen3-8B + 3 LoRA)
│  GPUs 0-3, GRPO via VERL                 GPUs 4-7, Hard-EM                   │
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
| **Model** | Qwen3-14B + LoRA (rank 16) | Qwen3-8B + 3 LoRA (SEGMENT, CONTRACT, CURATOR) |
| **Training** | GRPO via VERL (group size 8, LR 1e-5) | Hard-EM with LoRA fine-tuning (LR 2e-4) |
| **Provides** | Game play: primitive actions + tool calls (`QUERY_SKILL`, `CALL_SKILL`, `QUERY_MEM`) | Skill Bank: segmented trajectories, effect contracts, split/merge/refine |
| **Consumes** | Skill bank (protocols for planning, contracts for reward shaping) | Raw episodes from Decision Agent rollouts |
| **Interface** | `VLMDecisionAgent(skill_bank=..., model=...)` → `select_skill_from_bank()` / `query_skill_bank()` | `SkillBankAgent.select_skill(query, current_state)` / `SkillQueryEngine` |

The decision agent’s `skill_bank` can be a **SkillBankMVP** (plain storage) or a **SkillBankAgent** (full pipeline). Helpers (`skill_bank_to_text`, `query_skill_bank`, `select_skill_from_bank`) accept both and use the richest API available.

**Skill bank: protocol store vs contract.** Each skill has two parts: (1) **Protocol store** — `name`, `strategic_description`, `protocol` (steps, preconditions, success_criteria, expected_duration), `confidence`; used for `skill_bank_to_text()` and `active_skill_plan`. (2) **Contract** — effects (`eff_add`, `eff_del`, `eff_event`) used for segmentation, verification, and reward shaping (r_follow). The agent **plans** from protocols and is **rewarded** for progress on the contract's eff_add predicates.

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
   `episode = run_episode_vlm_agent(env, agent=VLMDecisionAgent(skill_bank=skill_agent), task="...")`

3. **Feed back** — ingest trajectories into the skill pipeline:
   `skill_agent.ingest_episodes([episode])` then `skill_agent.run_until_stable()`

4. **Repeat** — the bank improves (new skills, splits/merges/refinements), so the next run has better retrieval and reward shaping.

**For training**: `bash scripts/coevolution_train.sh` runs this loop with GRPO + Hard-EM on 8 GPUs. See [Section 5](#5-trainer-code) for details.

See [decision_agents/README.md](decision_agents/README.md) for the VLM agent API and [skill_agents/README.md](skill_agents/README.md) for the pipeline and query usage.

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
- **Producers**: [decision_agents/](decision_agents/) (`run_episode_vlm_agent()` → `Episode` with full `Experience` fields), [labeling/](labeling/) ([label_episodes_gpt54.py](labeling/label_episodes_gpt54.py) for cold-start; [labeling/readme.md](labeling/readme.md)).
- **Consumers**: [skill_agents/](skill_agents/) ingests `Episode`, segments into `SubTask_Experience`, and builds the skill bank (protocol + contract + `Skill.sub_episodes` as `SubEpisodeRef`).

## Experience

Single step: **state, action, reward, next_state, done**. Implemented in [data_structure/experience.py](data_structure/experience.py) as `Experience` with required fields and optional: **intentions**, **tasks**, **sub_tasks**, **summary**, **summary_state**, **idx**, **sub_task_done**, **reward_details**, **action_type**, **raw_state**, **raw_next_state**, **available_actions**, **interface**.

Structured formats (used by RAG and skill pipeline):

| Field | Format | Purpose |
|-------|--------|---------|
| **summary_state** | `key=value \| key=value` (e.g. `game=tetris \| stack_h=14 \| holes=32`) | Deterministic; from `build_rag_summary()` (0 LLM). Retrieval and predicates. |
| **summary** | `summary_state \| note=<strategic note>` | Facts + short LLM note (≤10 words); delta-aware. |
| **intentions** | `[TAG] subgoal phrase` (e.g. `[CLEAR] Reduce holes before stack overflows`) | 13 tags: SETUP, CLEAR, MERGE, ATTACK, DEFEND, NAVIGATE, POSITION, COLLECT, BUILD, SURVIVE, OPTIMIZE, EXPLORE, EXECUTE. Used for skill boundaries and RAG. |
| **reward_details** | `{r_env, r_follow, r_cost, r_total}` | Per-step breakdown from VLM reward tool. |
| **action_type** | `"primitive"`, `"QUERY_MEM"`, `"QUERY_SKILL"`, `"CALL_SKILL"` | Used by trainer and metrics. |

`run_episode_vlm_agent()` fills these from agent state: `summary_state` ← `get_state_summary`, `intentions` ← `get_intention`, `sub_tasks` ← `active_skill_id`, `reward_details` / `action_type` from reward tool. Skill agents read `summary_state` and `intentions` for segmentation without extra conversion.

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

Filled by `run_episode_vlm_agent()` from wrapper `info`; persisted in `to_dict()` / `from_dict()`. **metadata** holds rollout info (e.g. `cumulative_reward`, `agent_state`, `done`, `steps`).

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
| **Stage 2: Sub-task labeling** | [infer_segmentation](skill_agents/infer_segmentation/): decode over C with **preference-learned scorer** (LLM rankings → Bradley–Terry). Options = bank IDs + `__NEW__`. Contract feedback via `compat_fn`. **GRPO:** [skill_agents_grpo](skill_agents_grpo/): SEGMENT LoRA wraps `collect_segment_preferences()`; reward = SegmentationDiagnostics. | Output: [SegmentationResult](skill_agents/infer_segmentation/diagnostics.py) + list of [SubTask_Experience](data_structure/experience). |
| **Stage 3: Contract learning** | [stage3_mvp](skill_agents/stage3_mvp/): effects-only contracts (eff_add, eff_del, eff_event), verify, refine → [SkillBankMVP](skill_agents/skill_bank/bank.py). **GRPO:** CONTRACT LoRA wraps `llm_summarize_contract()`; reward = `verify_effects_contract().overall_pass_rate`. | Contracts feed back to Stage 2 via `compat_fn`. |
| **Stage 4: Bank maintenance** | [bank_maintenance](skill_agents/bank_maintenance/): **propose** (refine/merge/split/materialize/promote) → **filter** via `filter_candidates()` (CURATOR LoRA: approve/veto/defer) → **execute** approved actions. **GRPO:** CURATOR LoRA reward = bank_quality_delta. [materialize_new_skills](skill_agents/pipeline.py), [NewPoolManager](skill_agents/skill_bank/new_pool.py), **proto-skill staging** (`__NEW__` → materialize → verify → promote). | See [skill_agents_grpo/README.md](skill_agents_grpo/README.md) and [TODO_Lists/SKILLBANK_GRPO_PLAN.md](TODO_Lists/SKILLBANK_GRPO_PLAN.md). |

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

The [decision_agents](decision_agents/) module provides an LLM decision agent for step-by-step game play with **skill-bank retrieval** (RAG-based), episodic memory, intention inference, and composite reward. Two backends share the same code path: **GPT-5.4** (training-free, cold-start/labeling) and **Qwen3-14B** (GRPO-trained, vLLM). See [decision_agents/README.md](decision_agents/README.md).

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

This mode uses RAG to query experiences most relevant to the current situation and intentions from the experience buffer, using them as in-context learning to assist decision-making. The same **VLMDecisionAgent** is used with a **skill_bank** and optional **EpisodicMemoryStore**; no parameter updates — the backbone (e.g. GPT, Gemini, Claude) stays frozen.

## Trainable agent

This mode gathers experience via interaction and updates parameters with reinforcement learning. The **[trainer/](trainer/)** module implements it: the VLM Decision Agent is trained with **GRPO** (retrieval as first-class actions; reward = r_env + shaping + query/call costs), and the Skill Bank is updated via **Hard-EM**. See [trainer/README.md](trainer/README.md) and the “Trainer Code” section below.

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

**`run_episode_vlm_agent` already returns an `Episode`:**
```python
from decision_agents import run_episode_vlm_agent

episode = run_episode_vlm_agent(env, task="My task", max_steps=500)
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

---

# 5. Trainer code

The training code lives in **[trainer/](trainer/)** and implements co-evolution between two agents on **8 × A100-80GB GPUs**: a **Decision Agent** (Qwen3-14B, GRPO, GPUs 0-3) that plays games and a **Skill Bank Agent** (Qwen3-8B, 3 LoRA adapters + Hard-EM, GPUs 4-7) that discovers and maintains reusable skills from the Decision Agent's trajectories. Both agents improve each other over multiple co-evolution iterations.

## Decision Agent (Agent A) — Qwen3-14B + LoRA GRPO

The Decision Agent selects primitive game actions and tool calls (`QUERY_SKILL`, `CALL_SKILL`, `QUERY_MEM`) against the current skill bank. Trained with **GRPO via VERL** (`GameAITrainer` subclassing `RayPPOTrainer`).

**Training settings** ([`scripts/configs/decision_agent_grpo_80gb.yaml`](scripts/configs/decision_agent_grpo_80gb.yaml)):

| Parameter | Value |
|-----------|-------|
| Base model | Qwen/Qwen3-14B |
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

**Components**: [trainer/decision/env_wrapper.py](trainer/decision/env_wrapper.py) (retrieval-as-action, tool call trace recording), [trainer/decision/reward_shaping.py](trainer/decision/reward_shaping.py), [trainer/decision/grpo_trainer.py](trainer/decision/grpo_trainer.py), [trainer/decision/replay_buffer.py](trainer/decision/replay_buffer.py), [trainer/decision/rollout_collector.py](trainer/decision/rollout_collector.py).

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
5. **Stage 4** — bank maintenance (**CURATOR** LoRA approves/vetoes/defers merge, split, materialize proposals)
6. **SkillEval gating** — accept/reject: avg pass rate ≥ 0.6, NEW rate ≤ 0.3, margin regression tolerance 0.1

**Components**: [trainer/skillbank/em_trainer.py](trainer/skillbank/em_trainer.py) (with segmentation store), [trainer/skillbank/stages/](trainer/skillbank/stages/), [trainer/skillbank/bank_io/](trainer/skillbank/bank_io/) (versioned store, indices, diff logger), [trainer/skillbank/learners/](trainer/skillbank/learners/) (boundary classifier, tie-breaker). LoRA routing via [skill_agents_grpo/lora/](skill_agents_grpo/lora/) — `SkillFunction` enum, `MultiLoraSkillBankLLM`, adapter routing. LLM calls use [skill_agents/_llm_compat.py](skill_agents/_llm_compat.py) for reasoning-model compatibility (`/no_think`, think-tag stripping).

See [skill_agents_grpo/README.md](skill_agents_grpo/README.md) for the full GRPO skill bank pipeline and [TODO_Lists/SKILLBANK_GRPO_PLAN.md](TODO_Lists/SKILLBANK_GRPO_PLAN.md) for the training plan.

## Co-Evolution Pipeline

The co-evolution loop alternates between training both agents. Each iteration: collect rollouts → update Skill Bank → train Decision Agent → repeat.

```
Iteration 1:
  cold-start rollouts (base Qwen3-14B)
    → Skill Bank v1 (LoRA training + Hard-EM on GPUs 4-7)
    → Decision Agent v1 (GRPO on GPUs 0-3, with Bank v1)

Iteration 2+:
  rollouts with Decision Agent v_{i-1}
    → Skill Bank v_i (LoRA + EM on new rollouts)
    → Decision Agent v_i (GRPO with updated bank)
```

**Co-evolution callback** (VERL): [trainer/decision/coevolution_callback.py](trainer/decision/coevolution_callback.py)
- `SkillBankCoEvolutionCallback` — injected into `GameAITrainer.fit()`; runs EM every 10 training steps.
- `SkillAgentToolPipeline` — wraps all four skill agent stages as callable tools.
- `SegmentationStore` — persistent JSONL store for per-trajectory segmentations, updated each EM cycle.
- On accepted update: hot-swaps bank into environment workers.

**Standalone orchestrator**: [trainer/launch_coevolution.py](trainer/launch_coevolution.py) — runs Decision GRPO continuously; every N episodes freezes a rollout batch, runs SkillBank EM, gates with fixed-seed eval, then commits or rolls back the bank.

### Training Scripts

| Script | Purpose |
|--------|---------|
| [`scripts/coevolution_train.sh`](scripts/coevolution_train.sh) | Main loop: cold-start rollouts → Skill Bank v1 → Decision Agent v1 → iterate (default 6 iterations) |
| [`scripts/decision_agent_train.sh`](scripts/decision_agent_train.sh) | GRPO training for Decision Agent (Qwen3-14B) on GPUs 0-3 via VERL |
| [`scripts/skillbank_agent_train.sh`](scripts/skillbank_agent_train.sh) | LoRA training + Hard-EM for Skill Bank Agent (Qwen3-8B) on GPUs 4-7 |
| [`cold_start/run_100_rollouts.py`](cold_start/run_100_rollouts.py) | Batch rollout generation (100 episodes per game) |

```bash
# Default: Qwen3-14B decision + Qwen3-8B skill bank, 6 iterations
bash scripts/coevolution_train.sh

# Custom:
Decision_base_model=Qwen/Qwen3-14B \
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

**After fixing, clean stale outputs before re-running:**
```bash
cd runs/coevolution
rm -f rollouts/*_rollouts_v0.jsonl
rm -rf temp_results/* lora_adapters/* logs/skillbank/*
```

---

# ToDo (unfinished / future) — consolidated

Unfinished or future work across the repo. See in-doc sections for details.

| Area | Item | Status |
|------|------|--------|
| **Trainer** | Decision agent LoRA for parameter-efficient fine-tuning | **Open** |

