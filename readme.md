# Enhance Agentic Decision-making in Multiple-player long-horizon games with unsupervised experiences

[![GitHub Repository](https://img.shields.io/badge/GitHub-Repository-blue)](https://github.com/wuxiyang1996/Game-AI-Agent)

## Overview

This repository provides a framework for enhancing agentic decision-making in multi-player, long-horizon games through unsupervised experience. The framework integrates with multiple game environments and supports both training-free (RAG-based) and trainable (RL-based) agent architectures. This readme outlines each module, gives initial instructions for vibe coding, and aims to ease integration and debugging (including function definitions and ToDo lists for each class).

**No external repos are bundled.** This repository contains only Game-AI-Agent code. For Avalon/Diplomacy you need [AgentEvolver](https://github.com/modelscope/AgentEvolver) (clone as sibling or on `PYTHONPATH`). For GamingAgent or VideoGameBench evaluation, clone those repos as siblings when needed; see the respective `evaluate_*/setup_*_eval_env.md` files. For **VERL training and inference** (vLLM/sglang, GiGPO/PPO), use [verl-agent](https://github.com/verl-project/verl-agent) as a sibling and see [INSTALL.md](INSTALL.md).

**Install:** See **[INSTALL.md](INSTALL.md)** for setup and VERL/verl-agent. Quick: add this repo to `PYTHONPATH`, or `conda env create -f environment.yml` then `conda activate game-ai-agent`.

**Contents:** 1. [Environments](#1-environments) · 2. [Data structure](#2-data-structure) · 3. [Skill agent](#3-skill-agent) · 4. [Decision-making agent](#4-decision-making-agent) · 5. [Trainer code](#5-trainer-code) · [ToDo (unfinished)](#todo-unfinished--future--consolidated)

## Quick Links

- **🔗 Repository**: [GitHub - Game-AI-Agent](https://github.com/wuxiyang1996/Game-AI-Agent)

- **📦 Environment Wrappers** — [env_wrappers/](env_wrappers/): NL wrappers and evaluation for game environments
  - [Overcooked AI](env_wrappers/overcooked_nl_wrapper.py) — [overcooked_ai](https://github.com/HumanCompatibleAI/overcooked_ai); eval: [evaluate_overcooked/](evaluate_overcooked/)
  - [Avalon](env_wrappers/avalon_nl_wrapper.py) · [Diplomacy](env_wrappers/diplomacy_nl_wrapper.py) — require **AgentEvolver** (external; [AgentEvolver Games](https://github.com/modelscope/AgentEvolver/blob/main/games/README.md)); eval: [evaluation_evolver/](evaluation_evolver/)
  - [GamingAgent](env_wrappers/gamingagent_nl_wrapper.py) — LMGame-Bench (2048, Sokoban, Tetris); requires **GamingAgent** (external); eval: [evaluate_gamingagent/](evaluate_gamingagent/)
  - [VideoGameBench GB](env_wrappers/videogamebench_nl_wrapper.py) — Game Boy games (Kirby, etc.); [VideoGameBench DOS](env_wrappers/videogamebench_dos_nl_wrapper.py) — DOS games (Doom, Civ, etc.); eval: [evaluate_videogamebench/](evaluate_videogamebench/)

- **🔍 RAG & Embeddings** — [rag/](rag/): Text (default [Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)), multimodal (default [Qwen3-VL-Embedding-2B](https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B)); config: `RAG_EMBEDDING_MODEL`, `MULTIMODAL_EMBEDDING_MODEL`. [rag/README.md](rag/README.md)

- **🎮 Decision Agent** — [decision_agents/](decision_agents/): VLM step-by-step play with a **two-turn micro-loop** per timestep: `take_action` (primitives or `QUERY_MEM` / `QUERY_SKILL` / `CALL_SKILL`) → `reward`. Uses a skill bank (from skill_agents) for retrieval; `QUERY_SKILL(key)` returns a micro_plan and contract.
  - **Core**: [agent.py](decision_agents/agent.py) — `VLMDecisionAgent`, `run_tool()`, `run_episode_vlm_agent()`; [dummy_agent.py](decision_agents/dummy_agent.py) — game detection, action extraction.
  - **Helpers**: [agent_helper.py](decision_agents/agent_helper.py) — `get_state_summary()`, `compact_structured_state()`, `compact_text_observation()`, `infer_intention()`, `EpisodicMemoryStore` (RAG-backed memory), `skill_bank_to_text()`, `query_skill_bank()`.
  - **Reward**: [reward_func.py](decision_agents/reward_func.py) — `RewardConfig`, `RewardComputer`; **r_total** = r_env + w_follow×r_follow + r_cost (query_mem_cost, query_skill_cost, call_skill_cost, skill_switch_cost).
  - **Tools**: `take_action`, `reward`, `get_state_summary`, `get_intention`, `query_skill`, `query_memory`. See [decision_agents/README.md](decision_agents/README.md).

- **📚 Skill Agents** — [skill_agents/](skill_agents/): Build and maintain a Skill Bank from trajectories; consumed by decision_agents for `query_skill`.
  - **Orchestrator** [SkillBankAgent](skill_agents/pipeline.py): `ingest_episodes` → segment (Stage 1+2) → learn contracts (Stage 3) → maintain bank (Stage 4) → `query_skill`. Methods: `segment_episode()`, `run_contract_learning()`, `run_bank_maintenance()`, `run_until_stable()`.
  - **Stage 1** [boundary_proposal/](skill_agents/boundary_proposal/): Candidate cut points C (signals: rule-based or LLM `env_name="llm"`; optional RAG change-point; `merge_radius`). [README](skill_agents/boundary_proposal/README.md)
  - **Stage 2** [infer_segmentation/](skill_agents/infer_segmentation/): Decode over C with preference scorer (LLM → Bradley–Terry) → segments + labels (bank + `__NEW__`). [README](skill_agents/infer_segmentation/README.md)
  - **Stage 3** [stage3_mvp/](skill_agents/stage3_mvp/): Effects contract learn/verify/refine; NEW → pool.
  - **Stage 4** [bank_maintenance/](skill_agents/bank_maintenance/): Split/merge/refine; `materialize_new_skills`.
  - **Query & storage**: [SkillQueryEngine](skill_agents/query.py) (RAG + keyword/effect), [SkillBankMVP](skill_agents/skill_bank/bank.py), [tool_call_reward](skill_agents/tool_call_reward.py), [skill_evaluation/](skill_agents/skill_evaluation/). [skill_agents/README.md](skill_agents/README.md) · [PLAN.md](skill_agents/PLAN.md)

- **🏋️ Training** — [trainer/](trainer/): Co-evolution of both agents via VERL.
  - **Agent A (Decision)**: GRPO — primitives + `QUERY_MEM` / `QUERY_SKILL` / `CALL_SKILL`; reward = r_env + shaping + costs + tool-call reward.
  - **Agent B (SkillBank)**: Hard-EM (decode → update → gate); all four stages packed as a tool pipeline in the [co-evolution callback](trainer/decision/coevolution_callback.py). Trajectory segmentations stored and updated via `SegmentationStore`.
  - **Co-evolution callback**: [coevolution_callback.py](trainer/decision/coevolution_callback.py) — `SkillBankCoEvolutionCallback` + `SkillAgentToolPipeline` + `SegmentationStore`; integrates `skill_agents.tool_call_reward` into reward shaping.
  - Shared: [metrics](trainer/common/metrics.py), [reward_shaping](trainer/decision/reward_shaping.py), [eval_harness](trainer/common/eval_harness.py). Entry: [launch_train](trainer/decision/launch_train.py), [launch_coevolution](trainer/launch_coevolution.py). [trainer/README.md](trainer/README.md)

- **▶️ Inference** — [inference/](inference/): Run the decision agent and store rollouts in [data_structure](data_structure/experience.py) format (`Episode` + `Experience`). `run_episode_vlm_agent()` returns `Episode` directly; `run_inference()` wraps it with buffer/save support. `rollout_to_episode()` remains for legacy flat-dict conversion. [inference/README.md](inference/README.md)

---

# 1. Environments

This framework integrates with the following game environments. Each has an NL wrapper in [env_wrappers/](env_wrappers/) and a dedicated evaluation script folder.

| Environment | Source | Wrapper | Evaluation |
|-------------|--------|---------|------------|
| **Overcooked AI** | [overcooked_ai](https://github.com/HumanCompatibleAI/overcooked_ai) — cooperative human-AI benchmark (Overcooked game) | [overcooked_nl_wrapper.py](env_wrappers/overcooked_nl_wrapper.py) | [evaluate_overcooked/](evaluate_overcooked/) |
| **Avalon** | **External:** [AgentEvolver Games](https://github.com/modelscope/AgentEvolver/blob/main/games/README.md) — hidden-role deduction | [avalon_nl_wrapper.py](env_wrappers/avalon_nl_wrapper.py) | [evaluation_evolver/](evaluation_evolver/) |
| **Diplomacy** | **External:** [AgentEvolver Games](https://github.com/modelscope/AgentEvolver/blob/main/games/README.md) — strategic negotiation | [diplomacy_nl_wrapper.py](env_wrappers/diplomacy_nl_wrapper.py) | [evaluation_evolver/](evaluation_evolver/) |
| **GamingAgent (LMGame-Bench)** | **External:** [GamingAgent](https://github.com/lmgame-org/GamingAgent) — 2048, Sokoban, Tetris, etc. | [gamingagent_nl_wrapper.py](env_wrappers/gamingagent_nl_wrapper.py) | [evaluate_gamingagent/](evaluate_gamingagent/) |
| **VideoGameBench (Game Boy)** | **External:** [VideoGameBench](https://github.com/) — Game Boy ROMs via PyBoy | [videogamebench_nl_wrapper.py](env_wrappers/videogamebench_nl_wrapper.py) | [evaluate_videogamebench/](evaluate_videogamebench/) |
| **VideoGameBench (DOS)** | **External:** DOS games (JS-DOS in browser) | [videogamebench_dos_nl_wrapper.py](env_wrappers/videogamebench_dos_nl_wrapper.py) | [evaluate_videogamebench/](evaluate_videogamebench/) |

### Wrapper identification contract

All wrappers set `info["env_name"]` and `info["game_name"]` on every `reset()` and `step()` call. These two fields, together with `episode_id` (auto-generated UUID on each `Episode`), form a **three-part identifier** that uniquely locates any trajectory across platforms:

| Wrapper | `info["env_name"]` | `info["game_name"]` | Notes |
|---------|---------------------|---------------------|-------|
| GamingAgentNLWrapper | `"gamingagent"` | Auto-detected from obs (`"sokoban"`, `"tetris"`, `"2048"`, …) or explicit `game_name` constructor arg | Constructor: `GamingAgentNLWrapper(env, game_name="sokoban")` |
| VideoGameBenchNLWrapper | `"videogamebench"` | From constructor `game_name` (e.g. `"kirby"`) | Constructor: `VideoGameBenchNLWrapper(env, game_name="kirby")` |
| VideoGameBenchDOSNLWrapper | `"videogamebench_dos"` | From constructor `game_name` (e.g. `"doom2"`) | Stateless helper; call `wrapper.inject_info(info)` to set fields |
| OvercookedNLWrapper | `"overcooked"` | `"overcooked"` | Single-game platform |
| AvalonNLWrapper | `"avalon"` | `"avalon"` | Single-game platform |
| DiplomacyNLWrapper | `"diplomacy"` | `"diplomacy"` | Single-game platform |
| ColdStartEnvWrapper | `"gamingagent"` | Registry key (e.g. `"sokoban"`, `"twenty_forty_eight"`) | Used in [cold_start/](cold_start/) |

`run_episode_vlm_agent()` reads `info["env_name"]` and `info["game_name"]` to populate the `Episode`. Fallback chain: `info["game"]` → `detect_game(obs)` → `info["structured_state"]["game"]`.

---

## Decision & Skill Agents: Co-evolution

The **decision_agents** (VLM decision-making agent) and **skill_agents** (Skill Bank pipeline) are designed to **co-evolve**: the decision agent plays the game using skills retrieved from the bank, and new trajectories are fed back into the skill pipeline to refine and extend the bank.

### Big picture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         CO-EVOLUTION LOOP                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌──────────────────────┐         ┌──────────────────────┐                 │
│   │   decision_agents    │         │   skill_agents       │                 │
│   │   VLMDecisionAgent   │         │   SkillBankAgent     │                 │
│   │                      │         │                      │                 │
│   │ • step(obs) → tool   │  query  │ • ingest_episodes()  │                 │
│   │ • take_action        │ ──────► │ • segment → Stage 3  │                 │
│   │ • query_skill(key)   │ ◄────── │ • maintain bank      │                 │
│   │ • query_memory       │  bank   │ • query_skill(key)   │                 │
│   │ • reward (r_follow)  │         │ • add/remove/update  │                 │
│   └──────────┬───────────┘         └──────────┬───────────┘                 │
│              │                                 │                             │
│              │ run_episode_vlm_agent(env)      │ run_until_stable()          │
│              ▼                                 ▼                             │
│   ┌──────────────────────┐         ┌──────────────────────┐                 │
│   │  Episode (trajectory) │ ──────► │  Skill Bank (JSONL)   │                 │
│   │  ID: episode_id (UUID)│  feed  │  contracts, reports  │                 │
│   │  env_name + game_name │  back  │  versioned, queryable │                 │
│   │  Experience per step: │        │                      │                 │
│   │  state, action, reward│        │                      │                 │
│   │  summary_state,       │        │                      │                 │
│   │  intentions, sub_tasks│        │                      │                 │
│   └──────────────────────┘         └──────────────────────┘                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**With training** ([trainer/](trainer/)): the same loop is driven by **GRPO** (decision agent) and **Hard-EM** (skill bank). In VERL mode, the [SkillBankCoEvolutionCallback](trainer/decision/coevolution_callback.py) is injected into `GameAITrainer.fit()` and runs all four skill agent stages (boundary proposal, segmentation decode, contract learning, bank maintenance) as a unified tool-calling pipeline after each training step. Trajectory segmentations are persisted in a `SegmentationStore` and updated as the bank evolves. Tool-call rewards from `skill_agents.tool_call_reward` are integrated into the reward shaping for dense RL signal on tool use. Standalone mode: [trainer/launch_coevolution.py](trainer/launch_coevolution.py) runs GRPO continuously, periodically freezes a rollout batch, runs the SkillBank EM trainer, and deploys the new bank only if SkillEval gating passes. See [trainer/README.md](trainer/README.md).

### How they connect

| Role | decision_agents | skill_agents |
|------|-----------------|--------------|
| **Provides** | Game play: one action per step, optional retrieval (`query_skill`, `query_memory`) | Skill Bank: segment trajectories, learn effect contracts, split/merge/refine skills |
| **Consumes** | Skill bank (list of skills + contracts for prompt and `QUERY_SKILL`) | Raw episodes (from rollouts or demos) to segment and extract skills |
| **Interface** | `VLMDecisionAgent(skill_bank=...)`; `run_tool(TOOL_QUERY_SKILL, {"key": "..."})` | `SkillBankAgent.query_skill(key)` or `SkillQueryEngine`; `skill_bank_to_text(bank)` for prompt |

The decision agent’s `skill_bank` can be a **SkillBankMVP** (plain storage) or a **SkillBankAgent** (full pipeline). Helpers (`skill_bank_to_text`, `query_skill_bank`) accept both and use the richest API available.

### Do they use the RAG models ([rag/](rag/)) for memory/skill query?

**Yes.** Both agents now use RAG embeddings (default: Qwen3-Embedding-0.6B) for retrieval:

| Component | Memory query (`query_memory`) | Skill query (`query_skill`) | RAG (rag/) usage |
|-----------|-------------------------------|-----------------------------|-------------------|
| **decision_agents** | **Yes.** [EpisodicMemoryStore](decision_agents/agent_helper.py) embeds every memory via [rag/TextEmbedder](rag/embedding/text_embedder.py) and retrieves by **cosine similarity + keyword overlap** (blended, default 70% embedding / 30% keyword). The embedder is auto-loaded from `rag.get_text_embedder()`. Falls back to keyword-only if RAG is unavailable. | **Yes** (via SkillQueryEngine). Uses [query_skill_bank](decision_agents/agent_helper.py) which delegates to **SkillQueryEngine**. | Uses [rag/](rag/) `TextEmbedder` for both memory and skill retrieval. |
| **skill_agents** | N/A (skill_agents do not implement memory query). | **Yes.** [SkillQueryEngine](skill_agents/query.py) embeds all skill descriptions (ID + effect literals) via RAG `TextEmbedder` and scores queries by **cosine similarity + keyword Jaccard** (blended, default 60% embedding / 40% keyword). Auto-loads `rag.get_text_embedder()`. `query_by_effects` also uses embedding scores. Falls back to keyword-only if RAG is unavailable. | Uses [rag/](rag/) `TextEmbedder` for skill retrieval. Also **optional** in **boundary proposal**: [skill_agents/boundary_proposal](skill_agents/boundary_proposal/) can take a RAG embedder for embedding change-point scores. |

Both agents gracefully degrade to keyword-only retrieval if the RAG module or model is unavailable (import fails or embedder init fails).

### Typical workflow

1. **Bootstrap (optional)**  
   Load or create an initial skill bank (e.g. from a few hand-labelled or pre-segmented episodes).  
   `skill_agent = SkillBankAgent(bank_path="..."); skill_agent.load()` or `skill_agent.ingest_episodes(seed_episodes)`.

2. **Play**  
   Run the VLM agent with that bank — returns an `Episode` directly:  
   `episode = run_episode_vlm_agent(env, agent=VLMDecisionAgent(skill_bank=skill_agent), task="...", ...)`.  
   Each `Experience` in the episode has `summary_state`, `intentions`, `sub_tasks`, and `reward_details` populated from agent internal state — no post-hoc conversion needed.

3. **Feed back**  
   Ingest the episode directly into the skill pipeline:  
   `skill_agent.ingest_episodes([episode])` then optionally `skill_agent.run_until_stable()`.  
   The skill agents read `summary_state`/`intentions`/`sub_tasks` from each Experience for boundary proposal and segmentation.

4. **Repeat**  
   The bank improves (new skills from `__NEW__`, splits/merges/refinements), so the next run has better `query_skill` results and reward shaping.

**Trainable co-evolution**: Use [trainer/](trainer/) to run both agents in a training loop. The **Decision Agent** is trained with **GRPO** (group sampling + ranking on r_total); retrieval actions (`QUERY_MEM`, `QUERY_SKILL`, `CALL_SKILL`) are first-class actions with costs. The **SkillBank Agent** is updated via **Hard-EM** (decode → learn contracts → update → gate); no global LLM scoring. Shared rollout schema and reward contract live in `trainer/common/`; configs in `trainer/common/configs/decision_grpo.yaml` and `skillbank_em.yaml`. Run co-evolution: `python -m trainer.launch_coevolution --decision-config ... --skillbank-config ...`. See [trainer/README.md](trainer/README.md).

See [decision_agents/README.md](decision_agents/README.md) for the VLM agent API and [skill_agents/README.md](skill_agents/README.md) for the pipeline and query usage.

### How reward is computed (decision + skill agents)

Reward for the decision/skill stack is computed in two places, now unified during training:

- **decision_agents** (`reward_func.py`): **r_total = r_env + w_follow × r_follow + r_cost**
  - **r_env**: Raw environment reward from `env.step`.
  - **r_follow**: Skill-following shaping. The active skill’s contract has `eff_add` predicates (e.g. `at_pot`, `onion_in_pot`). Each predicate is checked against the **current observation text** (tokenize predicate and observation; predicate counts as satisfied if all its tokens appear in the observation). The agent gets a **per-predicate bonus** for each **newly** satisfied predicate, a **completion bonus** when all are satisfied, and a **small penalty** when no new predicate is satisfied this step. This is stateful over the episode (tracks which predicates were already satisfied).
  - **r_cost**: Negative costs for tool use: **query_mem_cost**, **query_skill_cost**, **call_skill_cost**, and **skill_switch_cost** when the active skill changes.

- **skill_agents** (`tool_call_reward.py`): **r_tool = w_relevance × r_relevance + w_utility × r_utility** (for agentic RL: “was this tool call good?”)
  - **r_relevance**: For **query_skill**, the retrieval score (from `SkillQueryEngine.query(key)` or from `retrieved_result["score"]`) scaled by `relevance_scale`. For **query_memory**, an optional `retrieval_quality` in [0,1] scaled the same way.
  - **r_utility**: For **query_skill** and **call_skill**, the skill’s **eff_add** predicates are checked against the **outcome observation** text (after the tool call). Same token-in-text rule: each predicate satisfied in the outcome gets a per-predicate bonus; if all are satisfied, an extra completion bonus. No state across steps—purely outcome-based.

- **Training integration** (`trainer/decision/reward_shaping.py`): during VERL training, `TrainRewardShaper` combines both sources into a single per-step reward: **r_total = r_env + w_follow × r_follow + r_cost + tool_call_reward_weight × r_tool**. The tool-call reward (`r_tool`) is computed by `compute_tool_call_reward` for every QUERY_SKILL, QUERY_MEM, and CALL_SKILL action, providing a dense learning signal for effective tool use.

Predicate satisfaction in both modules is **text-based**: tokenize the predicate (e.g. `onion_in_pot` → tokens), then require all tokens (length ≥ 2) to appear in the observation/outcome string (case-insensitive).

---

# 2. Data structure

This part defines how **experiences**, **episodes**, and **sub-task (skill)** segments are represented and cleaned from external sources into a uniform format, with optional fields for intention, summary, and quality for RAG and replay.

- **Data structure details** (class definitions, buffers, serialization) are in **[data_structure/](data_structure/)**, in particular [data_structure/experience.py](data_structure/experience.py) and [data_structure/helper.py](data_structure/helper.py).
- **Usage** — how episodes and experiences are produced and consumed:
  - **[decision_agents/](decision_agents/)**: `run_episode_vlm_agent()` directly returns an `Episode` with fully-populated `Experience` objects (`summary_state`, `intentions`, `sub_tasks`, `reward_details` filled from agent internal state during rollout). No separate conversion step needed.
  - **[skill_agents/](skill_agents/)**: consumes `Episode` (list of `Experience`) for boundary proposal and segmentation; reads `summary_state`/`summary`/`state` for observations, `sub_tasks`/`intentions`/`done` for predicates. Produces `SubTask_Experience` segments and skill labels. See [skill_agents/infer_segmentation/episode_adapter.py](skill_agents/infer_segmentation/episode_adapter.py), [skill_agents/boundary_proposal/](skill_agents/boundary_proposal/), and [skill_agents/README.md](skill_agents/README.md).

## Experience

Same as in standard RL: **state, action, intention, reward, next state, done**. Implemented in [data_structure/experience.py](data_structure/experience.py) as `Experience` with required fields plus optional: **intentions**, **tasks**, **sub_tasks**, **summary**, **summary_state**, **idx**, **sub_task_done**, **reward_details**, **action_type**. Long-term goal and short-term goal can be filled as needed. A field for experience value is left for prioritized replay. The **summary** (and **summary_state**) support RAG and embedding-based retrieval. The **reward_details** dict stores per-step reward breakdown (`r_env`, `r_follow`, `r_cost`, `r_total`) from the VLM decision agent. The **action_type** classifies each step as `"primitive"`, `"QUERY_MEM"`, `"QUERY_SKILL"`, or `"CALL_SKILL"`. Depending on the environment, not all fields are provided; the actual state used by LLM agents is often a **summary of the state**, which also serves as context for RAG retrieval.

`run_episode_vlm_agent()` now populates Experience fields directly from agent internal state during the rollout:

| Experience field | Source |
|---|---|
| `summary_state` | `agent.state.last_state_summary` (from `get_state_summary` tool) |
| `intentions` | `agent.state.current_intention` (from `get_intention` tool) |
| `sub_tasks` | `agent.state.active_skill_id` (current skill being followed) |
| `reward_details` | Full reward breakdown dict from `RewardComputer` |
| `action_type` | Classified from action string: `"primitive"`, `"QUERY_MEM"`, `"QUERY_SKILL"`, `"CALL_SKILL"` |

This means skill agents can directly consume VLM agent rollouts — no conversion or post-hoc labeling needed for the fields that `_extract_obs_actions()` and `_extract_predicates()` read.

Serialization: `Experience.from_dict()` is a `@classmethod` that constructs a new Experience from a dict (including all optional fields).

## Episode

An **episode** is a sequence of experiences (time-length limited) with final reward, length, optional outcome/summary, and **metadata**. Implemented as `Episode` in [data_structure/experience.py](data_structure/experience.py): **experiences**, **task**, **outcome**, **summary**, **metadata**; methods include **get_reward()**, **get_total_reward()**, **get_length()**, **set_outcome()**, **separate_into_sub_episodes()** (splits by sub-task indices into `SubTask_Experience`).

### Three-part identifier

Every Episode carries a triple that uniquely identifies it across both GamingAgent and VideoGameBench:

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `episode_id` | `str` | Auto-generated UUID (or caller-provided) | `"a3f1b2c4-..."` |
| `env_name` | `str` | Platform / wrapper name | `"gamingagent"`, `"videogamebench"`, `"videogamebench_dos"`, `"overcooked"` |
| `game_name` | `str` | Specific game within the platform | `"sokoban"`, `"2048"`, `"kirby"`, `"doom2"`, `"overcooked"` |

For single-game platforms (Overcooked, Avalon, Diplomacy), `env_name == game_name`. For multi-game platforms (GamingAgent, VideoGameBench), `game_name` disambiguates which game was played. All three fields are populated automatically by `run_episode_vlm_agent()` from wrapper `info` and are persisted through `to_dict()` / `from_dict()`.

The **metadata** dict stores rollout-level information (cumulative reward, final agent state, etc.). When returned by `run_episode_vlm_agent()`, metadata includes: `cumulative_reward`, `agent_state`, `done`, `steps`.

Serialization: `Episode.from_dict()` is a `@classmethod` that reconstructs an Episode (including `episode_id`, `env_name`, `game_name`, metadata) from a dict.

## Sub-task (skill) experience

**SubTask_Experience** represents a contiguous segment of experiences that accomplish a sub-task (strategy/skill): **sub_task**, **final_goal**, **sub_task_experience** (list of `Experience`), **outcome_experiences** (nullable), **summary**, **outcome_summary**, **length**, **cumulative_reward**. Used by [skill_agents](skill_agents/) for skill labeling and the skill bank pipeline. Serialization: `SubTask_Experience.from_dict()` is a `@classmethod`; `to_dict()` handles `None` outcome_experiences.

## Buffers

- **Experience_Replay_Buffer**: FIFO buffer of experiences; **add_experience()** (single, list, or full `Episode`), **sample_experience()**, **get_experience_summary()**.
- **Episode_Buffer**: FIFO buffer of full episodes; **add_episode()**, **sample_episode()**, **get_episode_summary()**, **save_to_json()** / **load_from_json()**.
- **Tool_Buffer**: stores `SubTask_Experience` (tools/strategies) for retrieval; **add_tool()**, **sample_tool()**, **get_tool_summary()**.

All live in [data_structure/experience.py](data_structure/experience.py).

### ToDo (data structure)

**Status (repo-wide):** Not all ToDos are finalized. Implemented: experience/episode buffers (FIFO + add/sample), summary generation and RAG-backed memory/skill query, trainer prioritized replay, and sub-task pipeline (skill_agents). Open or partial: in/out policy scoring for buffer, per-env format helpers, completion validator, and some trainer items (LoRA, stronger replay). RAG is not fine-tuned (frozen by design). See notes below.

1. **[Partial]** Experience buffer with in/out policy models that evaluate relative quality and relevance to the proposed latent state (intentions / sub-goals) and current state; push/pop (or add/sample) for adding/removing experiences.  
   *Done:* [Experience_Replay_Buffer](data_structure/experience.py) and [Episode_Buffer](data_structure/experience.py) (add/sample, save/load); [trainer ReplayBuffer](trainer/decision/replay_buffer.py) with prioritized sampling. *Open:* policy-based in/out scoring for quality/relevance to intentions.
2. **[Partial]** Per-environment helpers to convert raw experience sequences into this format and fill blank fields; sample experiences may be required.  
   *Done:* [Experience](data_structure/experience.py) optional fields; [skill_agents](skill_agents/infer_segmentation/episode_adapter.py) uses `summary_state`/`summary`/`state`; [decision_agents](decision_agents/agent_helper.py) `get_state_summary` / `infer_intention`. *Open:* central per-env “raw → Episode” conversion helpers.
3. **[Done]** (N/A) Hooks for experience quality evaluation and prioritized replay; optional auto-ranking inside the experience buffer.  
   Not required. Existing support is sufficient: [ReplayBuffer](trainer/decision/replay_buffer.py) has priority-weighted sampling and a value field for replay. No plans for explicit quality hooks or auto-ranking in [Episode_Buffer](data_structure/experience.py) / [Experience_Replay_Buffer](data_structure/experience.py).
4. **[Done]** Experience summary generation when missing; experience embedding code for RAG and experience query.  
   *Done:* [Experience.generate_summary](data_structure/experience.py) / `generate_summary_state`; [get_state_summary](decision_agents/agent_helper.py); [EpisodicMemoryStore](decision_agents/agent_helper.py) (RAG-backed memory query); [SkillQueryEngine](skill_agents/query.py) (RAG-backed skill query). Episode_Buffer.get_episode_summary is keyword-only (no embedding).

---

# 3. Skill agent

## Sub-task Decomposition

This is for experience pre-processing coming from external source if unlabeled, or rollouts from sub-optimal policies.

Suppose that we have the experiences processed into the intended format, the intention of this step is to decompose the entire experience trajectories into sub-tasks and sub-trajectories, i.e., the input trajectory is unlabeled without any annotation over the entire trajectory, expected output should be sub-task labeling over the entire trajectory.

## Pipeline implementation ([skill_agents/](skill_agents/))

The [skill_agents](skill_agents/) module implements this via the **SkillBankAgent** pipeline: unlabeled episodes → candidate boundaries → skill-labeled segments → learned contracts (finite skill categories). Entry points: [SkillBankAgent](skill_agents/pipeline.py), [ingest_episodes](skill_agents/pipeline.py), [segment_episode](skill_agents/pipeline.py); see [skill_agents/README.md](skill_agents/README.md) and [skill_agents/PLAN.md](skill_agents/PLAN.md).

| Step | Implementation | Notes |
|------|----------------|--------|
| **Input format** | [Episode](data_structure/experience) with `experiences` (state/action/reward/done, etc.) and `task`. Per-step state can be `state`, `summary_state`, or `summary`; optional `sub_tasks` / `intentions` for pseudo-labels. | Experience clean-up / formulation can feed into this; intention inference (e.g. from [decision_agents](decision_agents/)) can add pseudo intention over the trajectory. |
| **Stage 1: Boundary proposal** | [boundary_proposal](skill_agents/boundary_proposal/): extract signals (rule-based per env or **LLM-based** `env_name="llm"`) → propose **candidate cut points** C. Constraints: [merge_radius](skill_agents/pipeline.py) merges nearby candidates (default 5 steps); cuts only at C keep search O(\|C\|²). Optional RAG embedder for embedding change-point scores. | High-recall boundaries; no final segmentation yet. |
| **Stage 2: Sub-task labeling** | [infer_segmentation](skill_agents/infer_segmentation/): decode over C with a **preference-learned scorer** (LLM provides rankings, not numeric scores). **Candidate options** = `skill_names` (existing bank IDs + `__NEW__`). Constraints: [new_skill_penalty](skill_agents/pipeline.py), [preference_iterations](skill_agents/pipeline.py), [margin_threshold](skill_agents/pipeline.py). Output: [SegmentationResult](skill_agents/infer_segmentation/diagnostics.py) + list of [SubTask_Experience](data_structure/experience) (each segment has a **skill label**). | Task types are constrained to a finite set: existing bank skills or `__NEW__`. LLM is used as a teacher for pairwise preferences; a small scorer is trained (Bradley–Terry) and DP/beam decode produces the labeling. |
| **Stage 3: Contract learning** | [stage3_mvp](skill_agents/stage3_mvp/): for each non-NEW segment, learn **effects-only contracts** (eff_add, eff_del, eff_event), verify, refine; persist to [SkillBankMVP](skill_agents/skill_bank/bank.py). [eff_freq](skill_agents/pipeline.py), [min_instances_per_skill](skill_agents/pipeline.py). | Converts segments into a finite, queryable skill taxonomy. |
| **Stage 4: Bank maintenance** | [bank_maintenance](skill_agents/bank_maintenance/): split low-quality skills, merge similar ones, refine contracts. [materialize_new_skills](skill_agents/pipeline.py) promotes `__NEW__` segments meeting support/pass-rate thresholds to new named skills. | Keeps the skill set finite and consistent. |

**Usage (unlabeled trajectories → sub-task labeled):**

```python
from skill_agents import SkillBankAgent, PipelineConfig
from data_structure.experience import Episode  # experiences + task

config = PipelineConfig(
    bank_path="data/skill_bank.jsonl",
    env_name="llm+overcooked",   # or "llm" for LLM-based predicate extraction
    merge_radius=5,               # constrain boundary proximity
    new_skill_penalty=5.0,        # penalty for assigning __NEW__
    preference_iterations=3,      # active-learning rounds for preference scorer
)
agent = SkillBankAgent(config=config)
agent.load()

# Episodes from external source or sub-optimal rollouts (unlabeled)
episodes = [ep1, ep2, ...]
results = agent.ingest_episodes(episodes)   # Stage 1+2 per episode, then Stage 3
agent.run_until_stable(max_iterations=3)    # Stage 3 → Stage 4 → materialize NEW
agent.save()
# Each result: (SegmentationResult, list[SubTask_Experience]) with skill labels per segment
```

**ToDo (beyond current pipeline):**

1. **Experience clean-up / formulation**: Helpers to convert unformatted experiences into the Episode format (state/action/summary fields). Use intention inference (e.g. from [decision_agents](decision_agents/) or a dedicated module) to add pseudo intention labels over the trajectory where missing.
2. **Tighter constraints**: Explicit max sub-task length and richer candidate options (e.g. intention-derived skill candidates per state) can be wired into Stage 1/2 config or segment filters.
3. **Training code for labeling model**: No direct supervised label signal in the current pipeline; labeling is driven by LLM preference queries and a learned scorer. For end-to-end training, one could use rollout trajectories with reward signals (sub-task assignment, task-related rewards) as in the trainer’s [GRPO](trainer/decision/grpo_trainer.py) + [Hard-EM](trainer/skillbank/em_trainer.py) co-evolution.
4. **In-context learning**: The labeling path could be extended to generate segment labels from retrieved demonstrations (e.g. RAG over past labeled segments) instead of or in addition to the current preference-based scorer.

---

# 4. Decision-making agent

## Reward design for decision agent (implemented)

The [decision_agents](decision_agents/) reward is implemented in [reward_func.py](decision_agents/reward_func.py). After every `take_action`, the **reward** tool computes a composite reward used for training-free logging and for GRPO training.

**Formula:** **r_total** = r_env + w_follow × **r_follow** + **r_cost**

| Component | Description | Config / defaults |
|-----------|-------------|-------------------|
| **r_env** | Raw environment reward from `env.step`. | From game. |
| **r_follow** | Skill-following shaping (termination-free): progress toward the active skill’s **eff_add** predicates. | [RewardConfig](decision_agents/reward_func.py): `w_follow=0.1`, `follow_predicate_bonus=0.05`, `follow_completion_bonus=0.20`, `follow_no_progress_penalty=-0.01`. |
| **r_cost** | Costs for retrieval and skill use. | `query_mem_cost=-0.05`, `query_skill_cost=-0.05`, `call_skill_cost=-0.02`, `skill_switch_cost=-0.10`. |

**r_follow** logic (stateful per episode): For the active skill’s contract **eff_add**, the [RewardComputer](decision_agents/reward_func.py) checks the current observation (keyword presence). It awards a per-predicate bonus for each newly satisfied predicate, a completion bonus when *all* eff_add are satisfied, and a small penalty per step with no new progress. No terminal “skill done” signal—shaping is dense.

**Usage:** The runner calls `RewardComputer.compute_reward(r_env, action_type, observation, active_skill_id, skill_contract)` after each step; see [agent.py](decision_agents/agent.py) and [run_episode_vlm_agent](decision_agents/agent.py). Config is [RewardConfig](decision_agents/reward_func.py); trainer uses the same contract in [trainer/decision/reward_shaping.py](trainer/decision/reward_shaping.py).

## Completion validator

Verify whether the task / sub-task is completed, using the reward/completion logic above where applicable.

| Point | Status | Where / note |
|-------|--------|----------------|
| Skill-level completion signal | **Done** | [reward_func.py](decision_agents/reward_func.py): when all **eff_add** predicates are satisfied in the observation, the agent receives `follow_completion_bonus`. No separate LLM “completion level” or “steps left” function. |
| LLM completion-level function | **Open** | Not implemented. Would load state and output completion level of current task (distinct from RAG and r_follow). e.g. high reward early → low completion score; low reward late → high completion score. |
| Steps-left estimator (initial + final state) | **Open** | Not implemented. Would estimate steps to reach final state; could use stored experience from RAG. |

CoT and causality could be used for plausibility in future implementations.


## Experience retrieval

Use RAG to retrieve experience from the buffer when generating new experience or in **training-free** mode for decision-making (in-context learning). The RAG stack is in [rag/](rag/): [TextEmbedder](rag/embedding/text_embedder.py) (default Qwen3-Embedding-0.6B), [MemoryStore](rag/retrieval.py) (add_texts / retrieve / rank by cosine similarity), [rag/README.md](rag/README.md).

| ToDo | Status | Implementation / notes |
|------|--------|-------------------------|
| 1. Deploy RAG model to generate embeddings of experiences | **Done** (for episodic memory and skills) | [EpisodicMemoryStore](decision_agents/agent_helper.py) uses [rag.retrieval.MemoryStore](rag/retrieval.py) + `get_text_embedder()`; memories are embedded on `add()`. [SkillQueryEngine](skill_agents/query.py) embeds skill descriptions. RAG embedder is [frozen](rag/embedding/text_embedder.py) during training. *Not* wired to [Episode_Buffer](data_structure/experience.py) / [Experience_Replay_Buffer](data_structure/experience.py) (they use keyword-only `get_episode_summary` / `get_experience_summary`). |
| 2. Query experience most relevant to current state and intentions (similarity from RAG) | **Done** (for decision agent memory) | `query_memory(key)` in [decision_agents](decision_agents/agent.py) calls [EpisodicMemoryStore.query(key, k)](decision_agents/agent_helper.py); key typically includes scene/objective/entities. Retrieval uses RAG cosine similarity blended with keyword overlap. See the [«Do they use the RAG models?»](#do-they-use-the-rag-models-rag-for-memoryskill-query) table above. |
| 3. Update RAG model (e.g. contrastive learning or other RAG training) | **Done** (N/A) | RAG embedder is kept frozen (no fine-tuning). No contrastive or other RAG training is planned; retrieval uses the fixed pretrained model. |

## Decision-making agent design

This part designs an agent for step-by-step decision-making in games. The framework supports both **training-free** (in-context learning with RAG) and **trainable** (reinforcement learning) modes. The implemented agent lives in **[decision_agents/](decision_agents/)**.

## Implemented agent: VLMDecisionAgent (decision_agents/)

The **VLM Decision Agent** plays games using a **two-turn micro-loop** per timestep: (1) **take_action** — execute exactly one environment or retrieval action; (2) **reward** — compute and log composite reward. It can optionally call **get_state_summary** or **get_intention** before take_action.

### Tools

| Tool | Purpose |
|------|--------|
| **take_action** | Execute one action: either a **primitive** (move, interact, etc.) or a **retrieval action** — `QUERY_MEM(key)`, `QUERY_SKILL(key)`, `CALL_SKILL(skill_id, params)`. |
| **reward** | Compute reward for the last transition (r_env, r_follow, r_cost, r_total). Call once right after take_action. |
| **get_state_summary** | Optional: compact `key=value` state summary (≤ 400 chars). Prefers wrapper-produced structured dict; deterministic compression by default, optional LLM fallback. |
| **get_intention** | Optional: short phrase for current subgoal; uses last_actions, progress_notes, task. |
| **query_skill** | Used when take_action is QUERY_SKILL: retrieve procedures from the skill bank by key. |
| **query_memory** | Used when take_action is QUERY_MEM: retrieve similar past experiences (EpisodicMemoryStore). |

Retrieval is **budget-limited** (e.g. every N steps or when stuck); never call query_skill and query_memory in the same timestep.

### Internal state (AgentState)

The agent keeps: **current_intention**, **progress_notes**, **last_actions** (last 3–5), **stuck_counter**, **active_skill_id** / **active_skill_plan** / **skill_step_index**, **steps_without_progress**, **last_state_summary**, **last_reward**. The runner calls **update_from_tool_result()** after each tool so the next **step()** sees the updated state.

### Reward (decision_agents/reward_func.py)

**r_total = r_env + w_follow × r_follow + r_cost + tool_call_reward_weight × r_tool**

- **r_env**: Raw environment reward from env.step.
- **r_follow**: Skill-following shaping (termination-free). The active skill's contract has **eff_add** predicates; each is satisfied if its tokens appear in the current observation. The agent gets a **per-predicate bonus** for newly satisfied predicates, a **completion bonus** when all are satisfied, and a **no-progress penalty** when none are newly satisfied.
- **r_cost**: **query_mem_cost**, **query_skill_cost**, **call_skill_cost**, **skill_switch_cost** (when active_skill_id changes).

Defaults: `RewardConfig(w_follow=0.1, query_mem_cost=-0.05, query_skill_cost=-0.05, call_skill_cost=-0.02, skill_switch_cost=-0.10, follow_predicate_bonus=0.05, follow_completion_bonus=0.20, follow_no_progress_penalty=-0.01)`.

### Helpers and integration

- **get_state_summary(observation, structured_state, *, max_chars, ...)** — compact `key=value` state summary (≤ 400 chars). Prefers structured dict from env wrapper; falls back to deterministic text compression. See [decision_agents/README.md](decision_agents/README.md).
- **compact_structured_state(dict, max_chars)**, **compact_text_observation(obs, max_chars)** — low-level compressors.
- **infer_intention(summary, game, model, context)** — subgoal phrase.
- **EpisodicMemoryStore** — RAG-backed memory; **add_experience()**, **query(key, k)**.
- **skill_bank_to_text(bank)**, **query_skill_bank(bank, key, top_k)** — used in prompts and by run_tool(TOOL_QUERY_SKILL).
- The agent's **skill_bank** can be a **SkillBankMVP** or a **SkillBankAgent**; **run_episode_vlm_agent(env, agent=..., task=..., max_steps=...)** runs the full two-turn loop and returns an **`Episode`** with fully-populated `Experience` objects (including `summary_state`, `intentions`, `sub_tasks`, `reward_details`, `action_type`). The Episode is identified by a triple: `episode_id` (UUID), `env_name` (platform), `game_name` (specific game) — populated automatically from wrapper `info`. Episode metadata contains cumulative_reward, agent_state, done, and step count.

See [decision_agents/README.md](decision_agents/README.md) for API, quick start, and reward details.

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
# episode.env_name:   from wrapper info (e.g. "gamingagent", "videogamebench")
# episode.game_name:  specific game (e.g. "sokoban", "kirby", "doom2")
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

## ToDo (design modules)

The decision-making agent is expected to support:

1. **Summarize the state** — GPT for training-free, Qwen (or other) for trainable.
2. **Reachable tasks / intention update** — From current state, decide whether to update sub-task or intention.
3. **(Training-free)** Integrate a RAG module to query relevant experience from the buffer.
4. **Action generation** — From current state (summary), in-episode history, and (for training-free) retrieved context, output the next action.

## Reward design for skill agent

This part is for the **skill agent** ([skill_agents/](skill_agents/)): reward signals and training objectives used when building or evaluating the skill bank. RAG is not fine-tuned (frozen).

We adopt the idea of GDPO: normalize each kind of reward under each category. Reward categories to involve in this system include:

1. **Task completion reward** — From the environment: winning condition or survival (e.g. survive as long as possible).
2. **Sub-task completion reward** — From env or skill-level signals: e.g. occupation, kill, or skill **eff_add** satisfaction (see [decision_agents/reward_func](decision_agents/reward_func.py) r_follow for the decision agent; skill agent may use similar notions for segment/skill quality).
3. **Format reward** — For valid outputs or adherence to action/state format.

The first two are env-related. A time-sensitive discount can be applied to penalize long episodes. For reward tied to the *decision* agent’s use of skills (query_skill, call_skill), see [skill_agents/tool_call_reward](skill_agents/tool_call_reward.py) (r_relevance, r_utility).

---

# 5. Trainer code

The training code uses experience to train agents so they (i) take better actions, (ii) get better summaries of the environment, and (iii) improve retrieval and skill-following. It lives in **[trainer/](trainer/)** and implements co-evolution for both agents.

## Implemented (trainer/)

- **Decision Agent (Agent A)** — **GRPO** (Group Relative Policy Optimization)
  - **Actions**: primitives + `QUERY_MEM(key)`, `QUERY_SKILL(key)`, `CALL_SKILL(id, params)`; retrieval is first-class.
  - **Reward**: r_env + w_follow × r_follow + r_cost + tool_call_reward_weight × r_tool; tool-call reward integrated from `skill_agents.tool_call_reward`.
  - **Components**: [trainer/decision/env_wrapper.py](trainer/decision/env_wrapper.py) (retrieval-as-action, tool call trace recording), [trainer/decision/reward_shaping.py](trainer/decision/reward_shaping.py), [trainer/decision/rollout_collector.py](trainer/decision/rollout_collector.py), [trainer/decision/grpo_trainer.py](trainer/decision/grpo_trainer.py), [trainer/decision/replay_buffer.py](trainer/decision/replay_buffer.py).
  - **Launch**: `python -m trainer.decision.launch_train --config trainer/common/configs/decision_grpo.yaml`

- **SkillBank Agent (Agent B)** — **Hard-EM** (decode → update → gate)
  - **Pipeline**: ingest rollouts → Stage 0 (predicates) → Stage 1 (propose cuts) → Stage 2 (decode) → Stage 3 (contracts) → Stage 4 (refine/materialize/merge/split) → SkillEval gating; no global LLM scoring.
  - **Components**: [trainer/skillbank/ingest_rollouts.py](trainer/skillbank/ingest_rollouts.py), [trainer/skillbank/em_trainer.py](trainer/skillbank/em_trainer.py) (with segmentation store support), [trainer/skillbank/stages/](trainer/skillbank/stages/), [trainer/skillbank/bank_io/](trainer/skillbank/bank_io/) (versioned store, indices, diff logger).
  - **Optional learners**: boundary classifier ([trainer/skillbank/learners/boundary_trainer.py](trainer/skillbank/learners/boundary_trainer.py)), top-2 tie-breaker ([trainer/skillbank/learners/tiebreaker_trainer.py](trainer/skillbank/learners/tiebreaker_trainer.py)).

- **Co-evolution callback** (VERL): [trainer/decision/coevolution_callback.py](trainer/decision/coevolution_callback.py)
  - `SkillBankCoEvolutionCallback` — injected into `GameAITrainer.fit()`; runs after each training step at cadence.
  - `SkillAgentToolPipeline` — wraps all four skill agent stages as callable tools: `boundary_proposal`, `segmentation_decode`, `contract_learning`, `bank_maintenance`.
  - `SegmentationStore` — persistent JSONL store for per-trajectory segmentations, keyed by trajectory ID, updated each EM cycle.
  - Integrates `skill_agents.tool_call_reward` for tool-call reward metrics.
  - On accepted update: hot-swaps bank into environment workers, persists segmentations.

- **Shared**
  - **Rollout schema**: [trainer/common/metrics.py](trainer/common/metrics.py) — `RolloutRecord` / `RolloutStep` (single source of truth for both trainers). `RolloutRecord` carries `episode_id`, `env_name`, and `game_name` from the source `Episode`.
  - **Reward contract**: [trainer/decision/reward_shaping.py](trainer/decision/reward_shaping.py) — `compute_reward(prev, action, next, bank_state)` → r_env, r_follow, r_cost, r_tool, r_total.
  - **Eval & logging**: [trainer/common/eval_harness.py](trainer/common/eval_harness.py) (fixed-seed eval, SkillBank quick eval for gating), [trainer/common/logging.py](trainer/common/logging.py), [trainer/common/seeds.py](trainer/common/seeds.py).
  - **Configs**: [trainer/common/configs/decision_grpo.yaml](trainer/common/configs/decision_grpo.yaml), [trainer/common/configs/skillbank_em.yaml](trainer/common/configs/skillbank_em.yaml).

- **Standalone co-evolution orchestrator**: [trainer/launch_coevolution.py](trainer/launch_coevolution.py)
  - Runs Decision GRPO continuously; every N episodes freezes a rollout batch, runs SkillBank EM, gates with fixed-seed eval, then commits or rolls back the bank.
  - Run: `python -m trainer.launch_coevolution --decision-config trainer/common/configs/decision_grpo.yaml --skillbank-config trainer/common/configs/skillbank_em.yaml`

See [trainer/README.md](trainer/README.md) for full layout, milestones, metrics, and implementation order.

ToDo (future):
1. Integrate PPO / GDPO code bases and LoRA for fast fine-tuning (e.g. R1-style) where applicable.
2. Strengthen prioritized experience replay in [trainer/decision/replay_buffer.py](trainer/decision/replay_buffer.py) for training performance.
3. Add LoRA options to policy interface when training decision agent with parameter updates.

---

# ToDo (unfinished / future) — consolidated

Unfinished or future work across the repo. See in-doc sections for details.

| Area | Item | Status |
|------|------|--------|
| **Data structure** | Experience buffer with in/out policy models for quality/relevance to intentions | **[Partial]** — buffers exist; policy-based scoring open |
| **Data structure** | Per-environment helpers to convert raw experience → Episode format | **[Partial]** — optional fields and adapters exist; central per-env helpers open |
| **Sub-task (skill agent)** | Experience clean-up / formulation; intention inference for pseudo-labels | **Open** |
| **Sub-task (skill agent)** | Tighter constraints (max sub-task length, intention-derived skill candidates) | **Open** |
| **Sub-task (skill agent)** | Training code for labeling model (supervised signal + GRPO/Hard-EM) | **Open** |
| **Sub-task (skill agent)** | In-context learning for segment labels from retrieved demonstrations | **Open** |
| **Completion** | LLM completion-level function (distinct from r_follow) | **Open** |
| **Completion** | Steps-left estimator (initial + final state; use RAG experience) | **Open** |
| **Decision agent** | Reachable tasks / intention update; optional RAG for training-free | **Open** (partially done: get_state_summary, infer_intention, EpisodicMemoryStore) |
| **Trainer** | PPO / GDPO / LoRA integration; stronger prioritized replay; LoRA in policy interface | **Open** |

