# Cold-Start Data Generation

Generate initial trajectory data and skill seeds for the Game-AI-Agent system.

## Goal

1. **Prompt decision agents** (VLMDecisionAgent or dummy language agent) powered by GPT-5-mini to generate unlabeled trajectories from game environments.
2. **Label trajectories** with GPT-5-mini to produce initial seeds for the skill database (summaries, intentions, sub-task labels).

## Setup

```bash
# 1. Activate the cold-start conda environment
conda activate cold-start-agent

# 2. Set your OpenAI API key
export OPENAI_API_KEY="sk-..."

# 3. Set PYTHONPATH (from Game-AI-Agent root)
cd /path/to/Game-AI-Agent
export PYTHONPATH="$(pwd):$(pwd)/../GamingAgent:$PYTHONPATH"
```

## Usage

```bash
# Generate cold-start data for 2048 (3 episodes, 50 steps, GPT-5-mini)
python cold_start/generate_cold_start.py \
    --game twenty_forty_eight \
    --episodes 3 --max_steps 50 --model gpt-5-mini

# Use VLM decision agent instead of dummy agent
python cold_start/generate_cold_start.py \
    --game twenty_forty_eight \
    --agent_type vlm --episodes 3 --max_steps 50

# Generate for all available games
python cold_start/generate_cold_start.py --all_games --episodes 3 --max_steps 50

# Skip trajectory labeling (faster, unlabeled data only)
python cold_start/generate_cold_start.py \
    --game twenty_forty_eight --episodes 5 --max_steps 50 --no_label
```

## Output

Data is stored in `cold_start/data/<game_name>/`:

- `episode_NNN.json` — Individual episode with experiences and metadata
- `episode_buffer.json` — All episodes in Episode_Buffer format (loadable)
- `cold_start_summary.json` — Run summary with statistics

Each episode JSON contains:
- `experiences` — List of Experience objects (state, action, reward, next_state, done, intentions, summary, summary_state, sub_tasks, reward_details, etc.)
- `task` — Task description
- `outcome` — Episode outcome
- `summary` — Episode summary
- `metadata` — Rollout-level data: observations, actions, rewards, reward_details, cumulative_reward, agent_state, done, steps (plus model, agent type)

## Available Games

| Game | Description |
|------|-------------|
| `twenty_forty_eight` | 2048 tile merging puzzle |
| `sokoban` | Box-pushing puzzle |

## Agent Types

- **`dummy`** (default): Uses `language_agent_action` with GPT function calling — simpler, single-turn action selection per step. Produces `Episode` objects but with fewer populated Experience fields.
- **`vlm`**: Uses `run_episode_vlm_agent()` which returns an `Episode` directly with fully-populated `Experience` objects. Each experience includes `summary_state` (from agent's `get_state_summary`), `intentions` (from `get_intention`), `sub_tasks` (active skill ID), and `reward_details` (full reward breakdown). These episodes can be fed directly into the skill pipeline without any post-hoc labeling step.
