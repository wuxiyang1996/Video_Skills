# Cold-Start Data Generation

Generate initial trajectory data and skill seeds for the Game-AI-Agent system.

## Scope: Environments and Games We Use

We only run the **4 games** supported by our evaluation stack (see [evaluate_gamingagent/README.md](../evaluate_gamingagent/README.md)):

| # | Game | Actions |
|---|------|---------|
| 1 | **2048** | `up`, `down`, `left`, `right` |
| 2 | **Sokoban** | `up`, `down`, `left`, `right` |
| 3 | **Candy Crush** | coordinate swaps, e.g. `((0,5),(1,5))` |
| 4 | **Tetris** | `move_left`, `move_right`, `rotate_cw`, `rotate_ccw`, `hard_drop`, `soft_drop` |

### End conditions and max turns (how many steps per episode)

| Game | Evaluation max steps | Cold-start default | Natural end |
|------|---------------------|--------------------|-------------|
| **2048** | 200 | 200 (natural end) | No valid moves (game over), or reach 2048 tile; also terminates after 10 steps with no board change. |
| **Sokoban** | 100 | 200 (natural end) | All boxes on targets (win); or env `max_steps_episode` (200 in config); or 5 steps with no change. |
| **Candy Crush** | 50 | 50 (natural end) | Run out of moves (`num_moves` = 50 in env config). |
| **Tetris** | 200 | 200 (natural end) | Stack reaches top (game over); or 30 steps with no change. |

- **Evaluation** limits come from [evaluate_gamingagent/game_configs.py](../evaluate_gamingagent/game_configs.py) (`max_steps` per game). The wrapper truncates the episode when `step_count >= max_steps` even if the game has not ended.
- **Cold-start** runs until **natural end** by default: per-game limits are defined in `generate_cold_start.py` (`COLD_START_MAX_STEPS_NATURAL_END`). Use `--max_steps N` to cap episodes at N steps instead.

The environments available to us are:

- **[evaluate_gamingagent](../evaluate_gamingagent/)** — LMGame-Bench (the 4 games above)

- **[evaluation_evolver](../evaluation_evolver/)** — AgentEvolver; we run **two** envs:

| # | Env | Actions |
|---|-----|---------|
| 1 | **Avalon** | social deduction (phases/turns) |
| 2 | **Diplomacy** | negotiation (phases/turns) |

- **[evaluate_orak](../evaluate_orak/)** — Orak; we run **two** envs:

| # | Env | Actions |
|---|-----|---------|
| 1 | **Super Mario** | `Jump Level : 0` … `6` |
| 2 | **StarCraft II** | 5 macro actions per step (e.g. `TRAIN ZEALOT`) |

Other envs and games (e.g. Doom, Pokemon Red, Super Mario Bros, Ace Attorney, 1942, Tic-Tac-Toe, Texas Hold'em from the full LMGame-Bench set) are **not available** in our setup due to complexity and environment availability.

## Goal

1. **Prompt decision agents** (VLMDecisionAgent or dummy language agent) powered by GPT-5-mini to generate unlabeled trajectories from game environments.
2. **Label trajectories** with GPT-5-mini to produce initial seeds for the skill database (summaries, intentions, sub-task labels).

## Setup

```bash
# 1. Activate the cold-start conda environment
conda activate cold-start-agent

# 2. Set your API key (prefer OpenRouter; used for cold-start and ask_model)
# Option A: open_router_api_key in api_keys.py (recommended)
# Option B: export OPENROUTER_API_KEY="sk-or-..."
# Option C: export OPENAI_API_KEY="sk-..."

# 3. Set PYTHONPATH (from Game-AI-Agent root)
cd /path/to/Game-AI-Agent
export PYTHONPATH="$(pwd):$(pwd)/../GamingAgent:$PYTHONPATH"
```

## Batch Rollouts (100 per game)

The primary workflow generates 100 rollout episodes per game, with output formatted
for direct ingestion by the co-evolution framework (skill pipeline + trainer).

```bash
# All available games, 100 episodes each (default)
python cold_start/run_100_rollouts.py

# Specific games only
python cold_start/run_100_rollouts.py --games twenty_forty_eight sokoban tetris candy_crush

# Fewer episodes for testing
python cold_start/run_100_rollouts.py --episodes 5 --max_steps 30

# Use VLM decision agent (richer Experience fields)
python cold_start/run_100_rollouts.py --agent_type vlm

# Resume interrupted run (skips completed episodes)
python cold_start/run_100_rollouts.py --resume

# Skip labeling for faster generation
python cold_start/run_100_rollouts.py --no_label
```

### Per-game GPT-5.4 rollouts (configurable episodes)

To run each game separately and set how many episodes to gather per game, use the shell script (from **Game-AI-Agent** root):

```bash
# All 4 games, 50 episodes each (run until natural end per game)
bash cold_start/run_coldstart_gpt54_per_game.sh --episodes 50

# All 4 games, 100 episodes each (default)
bash cold_start/run_coldstart_gpt54_per_game.sh --episodes 100

# Only 2048 and Tetris, 20 episodes each
bash cold_start/run_coldstart_gpt54_per_game.sh --episodes 20 twenty_forty_eight tetris

# One game only, custom episodes
bash cold_start/run_coldstart_gpt54_per_game.sh --episodes 30 candy_crush

# With resume and no labeling
bash cold_start/run_coldstart_gpt54_per_game.sh --episodes 100 --resume --no_label
```

Output: `cold_start/output/gpt54/<game_name>/` (episode_*.json, rollouts.jsonl, episode_buffer.json).

### Output (cold_start/output/)

```
cold_start/output/
├── batch_rollout_summary.json          # Master run summary
├── twenty_forty_eight/
│   ├── episode_000.json ... episode_099.json   # Individual episodes
│   ├── episode_buffer.json                      # Episode_Buffer (loadable)
│   ├── rollouts.jsonl                           # JSONL: one Episode per line
│   └── rollout_summary.json                     # Per-game stats
├── sokoban/
│   └── ...
├── candy_crush/
│   └── ...
├── tetris/
│   └── ...
└── ... (all available games)
```

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

episodes = load_episodes_from_jsonl("cold_start/output/tetris/rollouts.jsonl")
agent = SkillBankAgent(bank_path="skills/bank.jsonl")
agent.ingest_episodes(episodes)
agent.run_until_stable(max_iterations=3)

# --- Trainer ingestion (Episode → RolloutRecord) ---
from trainer.skillbank.ingest_rollouts import ingest_rollouts

records = episodes_to_rollout_records(episodes)
trajectories = ingest_rollouts(records)

# --- Load all games at once ---
all_rollouts = load_all_game_rollouts("cold_start/output")
for game_name, episodes in all_rollouts.items():
    print(f"{game_name}: {len(episodes)} episodes")
```

## Single-Game Generation (generate_cold_start.py)

For smaller-scale generation or single-game runs:

```bash
# Generate cold-start data for 2048 (3 episodes, 50 steps, GPT-5-mini)
python cold_start/generate_cold_start.py \
    --game twenty_forty_eight \
    --episodes 3 --max_steps 50 --model gpt-5-mini

# Use VLM decision agent
python cold_start/generate_cold_start.py \
    --game twenty_forty_eight \
    --agent_type vlm --episodes 3 --max_steps 50

# Generate for all available games
python cold_start/generate_cold_start.py --all_games --episodes 3 --max_steps 50

# Skip trajectory labeling
python cold_start/generate_cold_start.py \
    --game twenty_forty_eight --episodes 5 --max_steps 50 --no_label
```

Output goes to `cold_start/data/<game_name>/` by default.

## Available Games (In Scope)

Cold-start generation targets only the 4 games we run (see Scope above):

| Game | Registry Key | Description |
|------|-------------|-------------|
| 2048 | `twenty_forty_eight` | Tile merging puzzle |
| Sokoban | `sokoban` | Box-pushing puzzle |
| Candy Crush | `candy_crush` | Match-3 tile puzzle |
| Tetris | `tetris` | Falling block puzzle |

Other games (Doom, Pokemon Red, Super Mario Bros, Ace Attorney, 1942, Tic-Tac-Toe, Texas Hold'em) are not available in our environment and are out of scope for this cold-start pipeline.

## Agent Types

- **`dummy`** (default): Uses `language_agent_action` with GPT function calling. Simpler, single-turn action selection per step.
- **`vlm`**: Uses `run_episode_vlm_agent()` which returns `Episode` objects with fully-populated Experience fields (`summary_state`, `intentions`, `sub_tasks`, `reward_details`, `action_type`). These can be fed directly into the skill pipeline.

## Output Format

Each episode JSON contains:
- `episode_id` — Unique UUID
- `env_name` — Platform name (`"gamingagent"`)
- `game_name` — Specific game (e.g. `"tetris"`, `"sokoban"`)
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
