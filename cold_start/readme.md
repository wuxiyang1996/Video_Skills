# Cold-Start Data Generation

Cold-start data generation for the **COS-PLAY** co-evolution framework
(COLM 2026, Section 5).  GPT-5.4 generates seed trajectories per game,
bootstrapping the co-evolution training loop between the Decision Agent and
Skill Bank Agent.

## Directory Contents

| File | Purpose |
|------|---------|
| `generate_cold_start.py` | Core module: game registry, env wrapper, episode runners, labeling |
| `generate_cold_start_gpt54.py` | GPT-5.4 agent for LMGame-Bench (2048, Candy Crush, Tetris) |
| `generate_cold_start_evolver.py` | GPT-5.4 agent for Avalon & Diplomacy |
| `generate_cold_start_orak.py` | GPT-5.4 agent for Super Mario (Orak env) |
| `load_rollouts.py` | Utility: load rollout outputs into Episode / RolloutRecord |
| `run_coldstart_gpt54.sh` | Shell launcher for LMGame-Bench rollouts |
| `run_coldstart_evolver.sh` | Shell launcher for Avalon & Diplomacy rollouts |
| `run_coldstart_orak_mario.sh` | Shell launcher for Super Mario (conda + Xvfb) |

## Games Covered (6 total)

### LMGame-Bench (`generate_cold_start_gpt54.py`)

| Game | Registry Key | Actions |
|------|--------------|---------|
| **2048** | `twenty_forty_eight` | `up`, `down`, `left`, `right` |
| **Candy Crush** | `candy_crush` | coordinate swaps, e.g. `((0,5),(1,5))` |
| **Tetris** | `tetris` | `move_left`, `move_right`, `rotate_cw`, `rotate_ccw`, `hard_drop`, `soft_drop` |

### AgentEvolver (`generate_cold_start_evolver.py`)

| Game | Registry Key | Actions |
|------|--------------|---------|
| **Avalon** | `avalon` | team proposals, votes, pass/fail, assassination |
| **Diplomacy** | `diplomacy` | unit orders (move, hold, support, convoy, retreat, build, disband) |

### Orak (`generate_cold_start_orak.py`)

| Game | Registry Key | Actions |
|------|--------------|---------|
| **Super Mario** | `super_mario` | `Jump Level : 0` ... `6` |

## Setup

```bash
conda activate game-ai-agent
export OPENROUTER_API_KEY="sk-or-..."   # or OPENAI_API_KEY
cd /path/to/Game-AI-Agent
export PYTHONPATH="$(pwd):$(pwd)/../GamingAgent:$PYTHONPATH"
```

## Quick Start

### LMGame-Bench (2048, Candy Crush, Tetris)

```bash
# All 3 games, 60 episodes each
bash cold_start/run_coldstart_gpt54.sh --episodes 60

# Specific games
bash cold_start/run_coldstart_gpt54.sh --games tetris candy_crush --episodes 60

# Resume interrupted run
bash cold_start/run_coldstart_gpt54.sh --episodes 60 --resume

# Python directly
python cold_start/generate_cold_start_gpt54.py --games tetris --episodes 5 --resume
```

Output: `cold_start/output/gpt54/<game_name>/`

### Avalon & Diplomacy

```bash
# Both games, 20 episodes each (default)
bash cold_start/run_coldstart_evolver.sh

# Avalon only, 60 episodes
bash cold_start/run_coldstart_evolver.sh --games avalon --episodes 60

# Resume + verbose
bash cold_start/run_coldstart_evolver.sh --resume -v

# Python directly
python cold_start/generate_cold_start_evolver.py --games diplomacy --episodes 10 -v
```

Output: `cold_start/output/gpt54_evolver/<game_name>/`

### Super Mario

Requires the `orak-mario` conda environment and Xvfb for headless rendering.

```bash
bash cold_start/run_coldstart_orak_mario.sh --episodes 10

# Or manually
source env_wrappers/setup_orak_mario.sh
python cold_start/generate_cold_start_orak.py --games super_mario --episodes 10
```

Output: `cold_start/output/gpt54_orak/<game_name>/`

## Output Structure

All generators produce the same layout per game:

```
cold_start/output/<suite>/<game_name>/
├── episode_000.json ... episode_NNN.json   # Individual episodes
├── episode_buffer.json                      # Episode_Buffer (loadable)
├── rollouts.jsonl                           # JSONL: one Episode per line
└── rollout_summary.json                     # Per-game stats
```

Suites: `gpt54/` (LMGame-Bench), `gpt54_evolver/` (Avalon/Diplomacy),
`gpt54_orak/` (Mario).

## Loading Rollouts into the Training Pipeline

```python
from cold_start.load_rollouts import (
    load_episodes_from_jsonl,
    load_episode_buffer,
    episodes_to_rollout_records,
    load_all_game_rollouts,
)

# Load episodes from a single game
episodes = load_episodes_from_jsonl("cold_start/output/gpt54/tetris/rollouts.jsonl")

# Convert to RolloutRecord for the trainer
records = episodes_to_rollout_records(episodes)

# Load all games at once
all_rollouts = load_all_game_rollouts("cold_start/output/gpt54")
for game_name, eps in all_rollouts.items():
    print(f"{game_name}: {len(eps)} episodes")
```

## End Conditions

Episodes terminate at each game's **natural end condition** (no artificial
step cap):

| Game | Natural end condition |
|------|---------------------|
| **2048** | No valid moves, or reach 2048 tile; also 10 steps with no board change |
| **Candy Crush** | Run out of moves (50 in env config) |
| **Tetris** | Stack reaches top; or 30 steps with no change |
| **Avalon** | 3 quest failures or assassination resolves after 3 quest successes |
| **Diplomacy** | Solo victory or 20 phases elapsed |
| **Super Mario** | Level complete or game over; capped at 100 steps |

## Design Notes

- **Diplomacy 20-phase limit**: Matches AgentEvolver's `DiplomacyConfig.max_phases`.
  Phases 1-20 contain the richest strategic diversity; more episodes at 20
  phases produces better seed skills than fewer at 50 phases.
- **Parallelized API calls**: Per-power (Diplomacy, 7 concurrent) and
  per-player (Avalon, up to 5 concurrent) calls are parallelized via
  `ThreadPoolExecutor` for ~7x speedup.
- **Labeling**: Off by default. Use the separate `labeling/` folder, or
  opt in with `--label` on the shell scripts.
