# Cold-Start Data Generation

Cold-start data generation for the **COS-PLAY** co-evolution framework (COLM 2026, Section 5). As described in the paper, GPT-5.4 is used as a teacher model to generate seed trajectories per game, which bootstrap the co-evolution training loop between the Decision Agent and Skill Bank Agent.

## Scope: Environments and Games We Use

We have **three** cold-start generators covering **6 games** across three environment stacks:

### 1. LMGame-Bench (`generate_cold_start_gpt54.py`)

| # | Game | Registry Key | Actions |
|---|------|--------------|---------|
| 1 | **2048** | `twenty_forty_eight` | `up`, `down`, `left`, `right` |
| 2 | **Candy Crush** | `candy_crush` | coordinate swaps, e.g. `((0,5),(1,5))` |
| 3 | **Tetris** | `tetris` | `move_left`, `move_right`, `rotate_cw`, `rotate_ccw`, `hard_drop`, `soft_drop` |

### 2. AgentEvolver (`generate_cold_start_evolver.py`)

| # | Game | Registry Key | Actions |
|---|------|--------------|---------|
| 4 | **Avalon** | `avalon` | team proposals, approve/reject votes, pass/fail, assassination target |
| 5 | **Diplomacy** | `diplomacy` | unit orders (move, hold, support, convoy, retreat, build, disband) |

### 3. Orak (`generate_cold_start_orak.py`)

| # | Game | Registry Key | Actions |
|---|------|--------------|---------|
| 6 | **Super Mario** | `super_mario` | `Jump Level : 0` … `6` |

### End conditions (how episodes terminate)

All cold-start generators use the **natural end condition** of each game engine. Episodes are never cut short by an artificial step cap.

| Game | Natural end condition | Source |
|------|----------------------|--------|
| **2048** | No valid moves (game over), or reach 2048 tile; also terminates after 10 steps with no board change. | GamingAgent env |
| **Candy Crush** | Run out of moves (`num_moves` = 50 in env config). | GamingAgent env |
| **Tetris** | Stack reaches top (game over); or 30 steps with no change. | GamingAgent env |
| **Avalon** | 3 quest failures (Evil wins) or assassination resolves after 3 quest successes. Always finite. | `AvalonGameEnvironment.done` |
| **Diplomacy** | Solo victory (`game.is_game_done`) or 20 phases elapsed (`DiplomacyConfig.max_phases = 20`). | `DiplomacyNLWrapper.done` |
| **Super Mario** | Level complete or game over; capped at 100 steps. | Orak env |

## Goal

1. **Prompt decision agents** powered by GPT-5.4 (or GPT-5-mini) to generate trajectories from game environments.
2. **Labeling is off by default**; use the separate `labeling/` folder for that. You can opt in with `--label` to label trajectories with GPT-5-mini (summaries, intentions, sub-task labels).

## Setup

```bash
# 1. Activate the cold-start conda environment
conda activate cold-start-agent

# 2. Set your API key (see .env.example for all required keys)
export OPENROUTER_API_KEY="sk-or-..."   # preferred
# or: export OPENAI_API_KEY="sk-..."

# 3. Set PYTHONPATH (from Game-AI-Agent root)
cd /path/to/Game-AI-Agent
export PYTHONPATH="$(pwd):$(pwd)/../GamingAgent:$PYTHONPATH"
```

## LMGame-Bench Rollouts (games 1-3)

### Batch rollouts (100 per game)

```bash
# All 3 games, 100 episodes each (default)
python cold_start/run_100_rollouts.py

# Specific games only
python cold_start/run_100_rollouts.py --games twenty_forty_eight tetris candy_crush

# Fewer episodes for testing
python cold_start/run_100_rollouts.py --episodes 5 --max_steps 30

# Use VLM decision agent (richer Experience fields)
python cold_start/run_100_rollouts.py --agent_type vlm

# Resume interrupted run (skips completed episodes)
python cold_start/run_100_rollouts.py --resume

# Labeling is off by default; use labeling/ for that. To enable here:
python cold_start/run_100_rollouts.py --label
```

### Per-game GPT-5.4 rollouts

```bash
# All 3 games, 50 episodes each (run until natural end per game)
bash cold_start/run_coldstart_gpt54_per_game.sh --episodes 50

# Only 2048 and Tetris, 20 episodes each
bash cold_start/run_coldstart_gpt54_per_game.sh --episodes 20 twenty_forty_eight tetris

# One game only
bash cold_start/run_coldstart_gpt54_per_game.sh --episodes 30 candy_crush

# With resume (labeling off by default)
bash cold_start/run_coldstart_gpt54_per_game.sh --episodes 100 --resume
```

Output: `cold_start/output/gpt54/<game_name>/`

### Single-game generation

```bash
python cold_start/generate_cold_start.py \
    --game twenty_forty_eight --episodes 3 --max_steps 50 --model gpt-5-mini

# VLM decision agent
python cold_start/generate_cold_start.py \
    --game twenty_forty_eight --agent_type vlm --episodes 3 --max_steps 50

# All games
python cold_start/generate_cold_start.py --all_games --episodes 3 --max_steps 50
```

Output: `cold_start/data/<game_name>/`

## AgentEvolver Rollouts — Avalon & Diplomacy (games 4-5)

Both games always run to their **natural end condition**. There is no `--max_steps` flag; the engines themselves decide when a game is over:

- **Avalon**: the `AvalonGameEnvironment` sets `done=True` when 3 quests fail (Evil wins) or the assassination phase resolves (after 3 quest successes). This matches `AvalonGame.run()` which loops on `while not self.env.done`.
- **Diplomacy**: the `DiplomacyNLWrapper` sets `done=True` when `game.is_game_done` (solo victory / draw) or `phases_processed >= 20` (matching `DiplomacyConfig.max_phases`). This matches `DiplomacyGame.run()` which loops on `while not self.game.is_game_done and phases_processed < self.config.max_phases`.

Per-power/per-player API calls within each phase are parallelized for speed.

```bash
# Both games, 20 episodes each (default)
bash cold_start/run_coldstart_evolver.sh

# Quick test
bash cold_start/run_coldstart_evolver.sh --episodes 5

# Avalon only
bash cold_start/run_coldstart_evolver.sh --games avalon

# Diplomacy only, 60 episodes
bash cold_start/run_coldstart_evolver.sh --episodes 60 --games diplomacy

# Resume interrupted run
bash cold_start/run_coldstart_evolver.sh --resume

# With labeling (opt-in)
bash cold_start/run_coldstart_evolver.sh --episodes 20 --label
```

Or call the Python script directly:

```bash
python cold_start/generate_cold_start_evolver.py --games avalon --episodes 10
python cold_start/generate_cold_start_evolver.py --games diplomacy --episodes 5 -v
python cold_start/generate_cold_start_evolver.py --resume
```

Output: `cold_start/output/gpt54_evolver/<game_name>/`

## Orak Rollouts — Super Mario (game 6)

Each Orak game needs its own conda environment.

```bash
# --- Super Mario ---
source evaluate_orak/setup_orak_mario.sh
python cold_start/generate_cold_start_orak.py --games super_mario --episodes 10
# or:
bash cold_start/run_coldstart_orak_mario.sh --episodes 10
```

Output: `cold_start/output/gpt54_orak/<game_name>/`

## Output Structure

All generators produce the same directory layout per game:

```
cold_start/output/<suite>/<game_name>/
├── episode_000.json ... episode_NNN.json   # Individual episodes
├── episode_buffer.json                      # Episode_Buffer (loadable)
├── rollouts.jsonl                           # JSONL: one Episode per line
└── rollout_summary.json                     # Per-game stats
```

Suites: `gpt54/` (LMGame-Bench), `gpt54_evolver/` (Avalon/Diplomacy), `gpt54_orak/` (Mario).

A `batch_rollout_summary.json` sits at the suite root with cross-game stats.

### Loading Rollouts into the Co-Evolution Framework

```python
from cold_start.load_rollouts import (
    load_episodes_from_jsonl,
    load_episode_buffer,
    episodes_to_rollout_records,
    load_all_game_rollouts,
)

# --- Skill pipeline ingestion ---
from skill_agents.pipeline import SkillBankAgent

episodes = load_episodes_from_jsonl("cold_start/output/gpt54/tetris/rollouts.jsonl")
agent = SkillBankAgent(bank_path="skills/bank.jsonl")
agent.ingest_episodes(episodes)
agent.run_until_stable(max_iterations=3)

# --- Trainer ingestion (Episode → RolloutRecord) ---
records = episodes_to_rollout_records(episodes)

# --- Load all games at once ---
all_rollouts = load_all_game_rollouts("cold_start/output/gpt54")
for game_name, episodes in all_rollouts.items():
    print(f"{game_name}: {len(episodes)} episodes")
```

## Agent Types

- **`dummy`** (default for LMGame-Bench): Uses `language_agent_action` with GPT function calling. Single-turn action selection per step.
- **`vlm`** (LMGame-Bench only): Uses `run_episode_vlm_agent()` which returns `Episode` objects with fully-populated Experience fields (`summary_state`, `intentions`, `sub_tasks`, `reward_details`, `action_type`).
- **GPT-5.4 function-calling** (Evolver & Orak): Each active player/power is queried with a system prompt and structured tool call (`choose_action` / `submit_orders`). Chain-of-thought reasoning is stored in the `intentions` field.

## Design Notes

### Natural end conditions (Evolver games)

Both Avalon and Diplomacy always run to their engine's natural end condition. No
artificial `--max_steps` cap is applied in the episode loop (`while not env.done`).
This ensures cold-start data matches how the games actually play in AgentEvolver's
`AvalonGame.run()` (`while not self.env.done`) and `DiplomacyGame.run()`
(`while not self.game.is_game_done and phases_processed < self.config.max_phases`).

### Diplomacy 20-phase limit

The base `diplomacy` library (from
[AI_Diplomacy](https://github.com/GoodStartLabs/AI_Diplomacy)) has no phase
limit -- games run until solo victory (18 supply centers) or draw, which rarely
happens with LLM agents. AI_Diplomacy itself uses `--max_year` (typically 1910,
~50 phases). AgentEvolver adds `max_phases=20` in `DiplomacyConfig` for
tractable rollouts. We use 20 for cold-start, training, and evaluation:

- Phases 1-20 (years 1901-1904) contain the richest strategic diversity:
  openings, alliance formation, first betrayals, early expansion.
- More episodes at 20 phases produces better seed skills than fewer episodes at
  50 phases for the same API budget -- breadth of experience beats depth.
- Training and evaluation both use 20 phases; matching this avoids distribution
  shift where the agent learns late-game stalemate tactics it never encounters
  during eval.

### Parallelized API calls

Per-power (Diplomacy, 7 concurrent) and per-player (Avalon, up to 5 concurrent)
LLM calls are parallelized within each phase via `ThreadPoolExecutor`. A shared
OpenAI client singleton avoids redundant TCP/TLS handshakes. This yields ~7x
speedup for Diplomacy episodes.

## Output Format

Each episode JSON contains:
- `episode_id` — Unique UUID
- `env_name` — Platform name (`"gamingagent"`, `"avalon"`, `"diplomacy"`, etc.)
- `game_name` — Specific game (e.g. `"tetris"`, `"avalon"`, `"super_mario"`)
- `experiences` — List of Experience objects:
  - `state`, `action`, `reward`, `next_state`, `done`
  - `intentions`, `tasks`, `sub_tasks`
  - `summary`, `summary_state`
  - `reward_details` (r_env, r_follow, r_cost, r_total)
  - `action_type` (primitive, QUERY_MEM, QUERY_SKILL, CALL_SKILL)
  - `idx` (step index)
- `task` — Task description
- `outcome` — Episode outcome
- `summary` — Episode summary
- `metadata` — Rollout stats (steps, total_reward, model, agent_type, etc.)

This format is directly compatible with:
- `Episode.from_dict()` / `Episode_Buffer.load_from_json()` for data loading
- `SkillBankAgent.ingest_episodes()` for skill pipeline ingestion
- `episodes_to_rollout_records()` → `ingest_rollouts()` for trainer ingestion

