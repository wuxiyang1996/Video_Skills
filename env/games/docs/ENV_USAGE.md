# How to Use the Avalon & Diplomacy Game Environments

This guide explains how to use the **Avalon** and **Diplomacy** game environments — including programmatic usage, CLI evaluation, web interface, and agent configuration.

---

## 1. Setup

### Install Dependencies

From the **AgentEvolver** project root:

```powershell
pip install -r games/requirements_game.txt
```

### Set Environment Variables

Both games use LLM-powered agents and require API access:

```powershell
$env:OPENAI_BASE_URL = "your_api_url"
$env:OPENAI_API_KEY = "your_api_key"
```

Or on Linux/macOS:

```bash
export OPENAI_BASE_URL=your_api_url
export OPENAI_API_KEY=your_api_key
```

---

## 2. Environment Overview

### Two Games

| Game | Type | Players | Key Mechanics |
|------|------|---------|---------------|
| **Avalon** | Hidden-role deduction | 5–10 | Team selection, voting, quests, assassination |
| **Diplomacy** | Strategic negotiation | 7 (powers) | Negotiation rounds, simultaneous orders, territory control |

Both games are **multi-agent, long-horizon** environments with **hidden information** and **social reasoning** (deception, persuasion, negotiation).

---

### Avalon

Avalon is a hidden-role social deduction game. Players are secretly assigned to the **Good** or **Evil** team. Good players try to succeed on quests; Evil players try to sabotage them.

**Roles:**

| Role ID | Role | Side | Special Ability |
|---------|------|------|-----------------|
| 0 | Merlin | Good | Knows who is Evil |
| 1 | Percival | Good | Knows who Merlin and Morgana are |
| 2 | Morgana | Evil | Appears as Merlin to Percival |
| 3 | Mordred | Evil | Hidden from Merlin |
| 4 | Oberon | Evil | Unknown to other Evil players |
| 5 | Servant | Good | No special ability |
| 6 | Minion | Evil | No special ability |
| 7 | Assassin | Evil | Can assassinate Merlin at end |

**Game Phases:**

| Phase ID | Phase | Description |
|----------|-------|-------------|
| 0 | Team Selection | Quest leader proposes a team |
| 1 | Team Voting | All players vote to approve/reject the team |
| 2 | Quest Voting | Team members secretly vote to pass/fail the quest |
| 3 | Assassination | If Good wins 3 quests, Assassin tries to identify Merlin |

**Termination Conditions** (`env.done = True`):

The game ends in exactly two situations:

1. **3 quests fail** -- Evil wins immediately. The game sets `done = True`, `good_victory = False`.
2. **3 quests succeed** -- The Assassination phase triggers. The Assassin chooses a target:
   - If the target is Merlin: Evil wins (`good_victory = False`).
   - If the target is NOT Merlin: Good wins (`good_victory = True`).

   In both cases `done = True` is set when the Assassin acts.

**Win Conditions:**
- **Good wins**: 3 successful quests AND Assassin fails to identify Merlin
- **Evil wins**: 3 failed quests OR Assassin correctly identifies Merlin

**Reward** (per agent, at game end):

| Condition | Reward |
|-----------|--------|
| Agent's side won (Good player & `good_victory=True`, or Evil player & `good_victory=False`) | **1.0** |
| Agent's side lost | **0.0** |

Reward is binary and team-based: all Good players share the same reward, and all Evil players share the same reward. Intermediate steps yield 0.

**Outcome fields:**
- `env.done` (bool): whether the game has ended
- `env.good_victory` (bool): `True` if Good won, `False` if Evil won
- `env.quest_results` (list of bool): per-quest outcomes (`True` = success, `False` = fail)

**Player Counts & Quest Sizes:**

| Players | Good/Evil | Quest 1 | Quest 2 | Quest 3 | Quest 4 | Quest 5 |
|---------|-----------|---------|---------|---------|---------|---------|
| 5 | 3/2 | 2 | 3 | 2 | 3 | 3 |
| 6 | 4/2 | 2 | 3 | 4 | 3 | 4 |
| 7 | 4/3 | 2 | 3 | 3 | 4* | 4 |
| 8 | 5/3 | 3 | 4 | 4 | 5* | 5 |
| 9 | 6/3 | 3 | 4 | 4 | 5* | 5 |
| 10 | 6/4 | 3 | 4 | 4 | 5* | 5 |

\* Quest 4 requires 2 fail votes to fail (for 7+ players).

---

### Diplomacy

Diplomacy is a classic strategy board game of negotiation and territorial control. Seven European powers compete for dominance on a map of pre-WWI Europe.

**Powers:**

| Power | Starting Supply Centers |
|-------|------------------------|
| AUSTRIA | 3 |
| ENGLAND | 3 |
| FRANCE | 3 |
| GERMANY | 3 |
| ITALY | 3 |
| RUSSIA | 4 |
| TURKEY | 3 |

**Game Phases (per turn):**

| Phase Suffix | Phase | Description |
|--------------|-------|-------------|
| `M` | Movement | Powers negotiate and then issue orders to move/hold/support units |
| `R` | Retreat | Dislodged units must retreat or disband |
| `A` | Adjustment | Build or disband units based on supply center count |

**Key Mechanics:**
- **Negotiation**: Before each movement phase, powers can send private or global messages to negotiate alliances, coordinate attacks, or deceive opponents
- **Simultaneous Orders**: All orders are revealed and resolved at the same time
- **Support**: Units can support other units' moves or holds
- **Supply Centers**: Control 18 of 34 supply centers to win (solo victory)

**Termination Conditions** (`game.is_game_done`):

The game enters phase `COMPLETED` in three scenarios:

1. **Solo victory** -- A power controls >= 18 supply centers (50%+1 of 34), has grown from the previous year, and is alone in the lead. The winner is recorded in `game.outcome`.
2. **100-year draw** -- If no power achieves solo victory by year 100, the game automatically draws. All surviving powers are listed as winners.
3. **`max_phases` truncation** -- In the AgentEvolver game wrapper, the loop also stops when `phases_processed >= config.max_phases` (default 20–30). The `diplomacy.Game` itself is **not** set to `COMPLETED` in this case; the wrapper simply stops processing. This counts as **truncation**, not natural termination.

**Reward** (per power, at game end):

| Condition | Reward |
|-----------|--------|
| Power is eliminated (no units, no centers, no retreats) | **0.0** |
| Power survives with *N* supply centers | **min(N / 18, 1.0)** |
| Power achieves solo victory (N >= 18) | **1.0** |

Reward is continuous and individual: each power's reward depends on its own supply center count at the moment the game ends (or is truncated). A power with 9 centers scores 0.5; one with 4 centers scores ~0.22.

**Outcome fields:**
- `game.is_game_done` (bool): `True` when `game.phase == 'COMPLETED'`
- `game.outcome` (list): `[last_phase, *victor_names]`, e.g. `['F1903M', 'FRANCE']`
- `game.powers[name].centers` (list): supply centers owned by each power
- `game.powers[name].is_eliminated()` (bool): `True` if power has no units, centers, or retreats

**Scoring**: Each power's score = number of supply centers / 18

---

## 3. Configuration

Games are controlled via **YAML configuration files** with a unified role-based structure.

### Configuration Structure

```yaml
# Default settings for all roles
default_role:
  trainable: false
  act_by_user: false
  model:
    url:              # Or set OPENAI_BASE_URL env var
    api_key:          # Or set OPENAI_API_KEY env var
    model_name: qwen-plus
    temperature: 0.7
    max_tokens: 2048
    stream: false
  agent:
    type: ThinkingReActAgent
    kwargs: null

# Game-specific settings
game:
  name: avalon          # or "diplomacy"
  num_players: 5        # Avalon: 5-10
  language: en          # "en" or "zh"/"cn"
  log_dir: games/logs

# Role-specific overrides (optional)
roles:
  assassin:             # Avalon role name, or ENGLAND, FRANCE, etc. for Diplomacy
    model:
      model_name: custom-model
```

**Configuration Priority:** Role-specific `roles` entries override `default_role`. Nested dicts are merged recursively — you only need to specify fields you want to change.

### Config Files

| File | Purpose |
|------|---------|
| `games/games/avalon/configs/default_config.yaml` | Avalon default settings |
| `games/games/avalon/configs/task_config.yaml` | Avalon evaluation config (inherits default) |
| `games/games/avalon/configs/train_config.yaml` | Avalon training config |
| `games/games/avalon/configs/arena_config.yaml` | Avalon arena/leaderboard config |
| `games/games/diplomacy/configs/default_config.yaml` | Diplomacy default settings |
| `games/games/diplomacy/configs/task_config.yaml` | Diplomacy evaluation config (inherits default) |
| `games/games/diplomacy/configs/train_config.yaml` | Diplomacy training config |
| `games/games/diplomacy/configs/arena_config.yaml` | Diplomacy arena/leaderboard config |

### Avalon-Specific Config

```yaml
game:
  name: avalon
  num_players: 5
  language: en
  log_dir: games/logs
  roles_name:
    - Merlin
    - Servant
    - Servant
    - Minion
    - Assassin
```

### Diplomacy-Specific Config

```yaml
game:
  name: diplomacy
  power_names:
    - AUSTRIA
    - ENGLAND
    - FRANCE
    - GERMANY
    - ITALY
    - RUSSIA
    - TURKEY
  map_name: standard
  max_phases: 30
  negotiation_rounds: 3
  seed: 42
  language: en
  log_dir: logs
```

---

## 4. Programmatic Usage

### Avalon: Running a Game in Python

```python
import asyncio
from games.games.avalon.engine import AvalonBasicConfig, AvalonGameEnvironment
from games.games.avalon.game import AvalonGame, avalon_game
from games.agent_factory import create_model_from_config, create_agent_from_config

# 1. Create config
config = AvalonBasicConfig.from_num_players(num_players=5)
# config fields: num_players, num_good, num_evil, merlin, percival, morgana, mordred, oberon

# 2. Create model
model_config = {
    'model_name': 'qwen-plus',
    'url': 'your_api_url',        # or set OPENAI_BASE_URL
    'api_key': 'your_api_key',    # or set OPENAI_API_KEY
    'temperature': 0.7,
    'max_tokens': 2048,
}
model = create_model_from_config(model_config)

# 3. Create agents
agent_config = {'type': 'ThinkingReActAgent', 'kwargs': None}
agents = [
    create_agent_from_config(agent_config, model=model, name=f"Player{i}")
    for i in range(5)
]

# 4. Run game (convenience function)
async def main():
    good_wins = await avalon_game(
        agents=agents,
        config=config,
        log_dir="games/logs",
        language="en",
    )
    print(f"Good wins: {good_wins}")

asyncio.run(main())
```

### Avalon: Using the Engine Directly

```python
from games.games.avalon.engine import AvalonBasicConfig, AvalonGameEnvironment

# Create environment
config = AvalonBasicConfig.from_num_players(5)
env = AvalonGameEnvironment(config)

# Get assigned roles
roles = env.get_roles()
# Returns: [(role_id, role_name, is_good), ...] e.g. [(5, 'Servant', True), (7, 'Assassin', False), ...]

# Query environment state
phase_id, phase_name = env.get_phase()       # (0, "Team Selection")
leader = env.get_quest_leader()               # int: player index
team_size = env.get_team_size()               # int: required team size for current quest

# Team selection (phase 0)
team = frozenset([0, 2])  # player indices
env.choose_quest_team(team=team, leader=leader)

# Team voting (phase 1)
votes = [1, 0, 1, 1, 0]  # 1=approve, 0=reject
next_phase, done, approved = env.gather_team_votes(votes)

# Quest voting (phase 2) — only team members vote
quest_votes = [1, 1]  # 1=pass, 0=fail
next_phase, done, succeeded, num_fails = env.gather_quest_votes(quest_votes)

# Assassination (phase 3) — if Good wins 3 quests
assassin_id = env.get_assassin()
target = 0  # player index to assassinate
_, done, good_wins = env.choose_assassination_target(assassin_id, target)

# Game ends when env.done == True
# env.good_victory: True if Good won, False if Evil won
```

### Avalon: From Presets

```python
env = AvalonGameEnvironment.from_presets({
    'num_players': 5,
    'quest_leader': 0,
    'role_names': ['Merlin', 'Servant', 'Servant', 'Minion', 'Assassin'],
})
```

---

### Diplomacy: Running a Game in Python

```python
import asyncio
from games.games.diplomacy.engine import DiplomacyConfig
from games.games.diplomacy.game import DiplomacyGame, diplomacy_game
from games.agent_factory import create_model_from_config, create_agent_from_config

# 1. Create config
config = DiplomacyConfig(
    power_names=["AUSTRIA", "ENGLAND", "FRANCE", "GERMANY", "ITALY", "RUSSIA", "TURKEY"],
    map_name="standard",
    max_phases=20,
    negotiation_rounds=3,
    seed=42,
    language="en",
)
# Or use defaults from YAML: config = DiplomacyConfig.default()

# 2. Create model
model_config = {
    'model_name': 'qwen-plus',
    'url': 'your_api_url',
    'api_key': 'your_api_key',
    'temperature': 0.7,
    'max_tokens': 2048,
}
model = create_model_from_config(model_config)

# 3. Create agents (one per power)
agent_config = {'type': 'CacheRetrievalAgent', 'kwargs': {'memory': {'type': 'CachedSummarizedMemory', 'kwargs': None}}}
agents = [
    create_agent_from_config(agent_config, model=model, name=f"Player{i}")
    for i in range(7)
]

# 4. Run game (convenience function)
async def main():
    game = await diplomacy_game(
        agents=agents,
        config=config,
        log_dir="logs",
    )
    # game is a diplomacy.Game object
    for power_name, power in game.powers.items():
        print(f"{power_name}: {len(power.centers)} supply centers, eliminated={power.is_eliminated()}")

asyncio.run(main())
```

### Diplomacy Config Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `power_names` | `List[str]` | 7 standard powers | List of power names |
| `map_name` | `str` | `"standard"` | Game map name |
| `max_phases` | `int` | `20` | Maximum number of phases before game ends |
| `negotiation_rounds` | `int` | `3` | Number of negotiation rounds per movement phase |
| `seed` | `int` | `42` | Random seed for game initialization |
| `language` | `str` | `"en"` | Language for prompts (`"en"` or `"zh"`/`"cn"`) |
| `human_power` | `str \| None` | `None` | Power controlled by human (for participate mode) |

---

## 5. Running Evaluations (CLI)

The evaluation script runs multiple games in parallel and aggregates statistics.

### Basic Usage

```bash
# Evaluate Avalon (10 games, 5 parallel workers)
python games/evaluation/run_eval.py \
    --game avalon \
    --config games/games/avalon/configs/task_config.yaml \
    --num-games 10 \
    --max-workers 5

# Evaluate Diplomacy
python games/evaluation/run_eval.py \
    --game diplomacy \
    --config games/games/diplomacy/configs/task_config.yaml \
    --num-games 5 \
    --max-workers 3
```

### CLI Options

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--game` | `-g` | (required) | Game to evaluate: `avalon` or `diplomacy` |
| `--config` | `-c` | (required) | Path to game config YAML file |
| `--num-games` | `-n` | `1` | Number of games to run |
| `--max-workers` | `-w` | `10` | Maximum number of parallel workers |
| `--experiment-name` | | `None` | Experiment name for organizing logs |

### Using Local VLLM Models

Start the VLLM server first, then run evaluation:

```bash
# Terminal 1: Start VLLM server
python games/evaluation/start_vllm.py \
    --model-path /path/to/model \
    --port 8000 \
    --model-name local_model

# Terminal 2: Run evaluation (config must point to localhost:8000/v1)
python games/evaluation/run_eval.py \
    --game avalon \
    --config games/games/avalon/configs/task_config.yaml \
    --num-games 10
```

### Output

Results are displayed in formatted tables:

```
══════════════════════════════════════════════════════════════════
              AVALON - RESULTS (Total Games: 10)
══════════════════════════════════════════════════════════════════

┌─ Game Results ──────────────────────────────────────────────────┐
│  Metric          │ Mean  │ Max   │ Min                         │
│  ────────────────┼───────┼───────┼───────                      │
│  good_win_rate   │ 0.60  │ 1.00  │ 0.00                       │
└─────────────────────────────────────────────────────────────────┘

┌─ Role Statistics ───────────────────────────────────────────────┐
│  Role     │ Metric    │ Mean  │ Max   │ Min                    │
│  ─────────┼───────────┼───────┼───────┼───────                 │
│  Merlin   │ win_rate  │ 0.50  │ 1.00  │ 0.00                  │
│  Servant  │ win_rate  │ 0.55  │ 1.00  │ 0.00                  │
│  Assassin │ win_rate  │ 0.40  │ 1.00  │ 0.00                  │
│  Minion   │ win_rate  │ 0.45  │ 1.00  │ 0.00                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 6. Web Interface

### Starting the Server

```bash
python games/web/server.py
```

Then open your browser at: `http://localhost:8000`

### Available Pages

| URL | Description |
|-----|-------------|
| `/` | Home page — select game and mode |
| `/avalon/observe` | Watch AI agents play Avalon |
| `/avalon/participate` | Play Avalon as a human alongside AI agents |
| `/diplomacy/observe` | Watch AI agents play Diplomacy |
| `/diplomacy/participate` | Play Diplomacy as a human controlling one power |
| `/leaderboard` | View model leaderboard |
| `/leaderboard/avalon` | Avalon-specific leaderboard |
| `/leaderboard/diplomacy` | Diplomacy-specific leaderboard |

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/start-game` | Start a new game (JSON body with game settings) |
| `POST` | `/api/stop-game` | Stop the current running game |
| `GET` | `/api/options?game=avalon` | Get game configuration options for frontend |
| `GET` | `/api/history` | Get Diplomacy game phase history |
| `GET` | `/api/history/{index}` | Get specific Diplomacy history item |
| `GET` | `/api/leaderboard/{game}` | Get leaderboard data |
| `WS` | `/ws` | WebSocket for real-time game state updates |

### Start Game Request Body

```json
{
  "game": "avalon",
  "mode": "observe",
  "language": "en",
  "num_players": 5,
  "user_agent_id": 0,
  "max_phases": 20,
  "negotiation_rounds": 3
}
```

---

## 7. Agent System

### Built-in Agent Types

| Agent | Description | Typical Use |
|-------|-------------|-------------|
| `ThinkingReActAgent` | Think-then-speak agent (default for Avalon) | General-purpose reasoning |
| `CacheRetrievalAgent` | Agent with cached summarized memory (default for Diplomacy) | Long-horizon games with large context |
| `EchoAgent` | Simple echo agent (used as moderator) | Broadcasting messages |

### Creating Agents from Config

```python
from games.agent_factory import create_agent_from_config, create_model_from_config

# Model config
model = create_model_from_config({
    'model_name': 'qwen-plus',
    'temperature': 0.7,
    'max_tokens': 2048,
})

# Agent config with custom memory and formatter
agent = create_agent_from_config(
    agent_config={
        'type': 'ThinkingReActAgent',
        'kwargs': {
            'sys_prompt': 'You are a strategic game player.',
            'memory': {
                'type': 'SlidingWindowMemory',
                'kwargs': {}
            },
            'formatter': {
                'type': 'SecureMultiAgentFormatter',
                'kwargs': {
                    'max_tokens': 1000,
                    'preserved_agent_names': ['Moderator']
                }
            }
        }
    },
    model=model,
    name='Player0',
)
```

### Custom Agents

Implement a custom agent by extending `AgentBase` from AgentScope:

```python
from agentscope.agent import AgentBase
from agentscope.message import Msg

class MyCustomAgent(AgentBase):
    def __init__(self, name, model, sys_prompt="", **kwargs):
        super().__init__(name=name, model=model, sys_prompt=sys_prompt)
        # Custom initialization

    async def __call__(self, msg=None):
        # Process incoming message, generate response
        # Use self.model for LLM calls
        response = await self.model(prompt)
        return Msg(name=self.name, content=response, role="assistant")
```

Then register it in your config:

```yaml
roles:
  merlin:
    agent:
      type: your_module.MyCustomAgent
      kwargs:
        sys_prompt: "You are Merlin..."
```

### Memory Systems

| Memory Type | Module | Description |
|-------------|--------|-------------|
| `InMemoryMemory` | `agentscope.memory` | Simple in-memory storage (default) |
| `SlidingWindowMemory` | `games.agents.memory` | Fixed-window recent memory (last 40 messages) |
| `SummarizedMemory` | `games.agents.memory` | Auto-summarizes old messages when threshold exceeded |
| `CachedSummarizedMemory` | `games.agents.memory` | Summarization + disk persistence for long games |

All memory classes inherit from `agentscope.memory.MemoryBase` and share a common async interface:

```python
await memory.add(msg)           # Add a Msg or list[Msg]
await memory.get_memory()       # Retrieve messages (may trigger summarization)
await memory.clear()            # Clear all content
await memory.size()             # Get message count
await memory.delete(index)      # Delete by index
memory.state_dict()             # Serialize
memory.load_state_dict(d)       # Deserialize
```

**Hierarchy:**

```
MemoryBase (agentscope.memory)
├── InMemoryMemory           — stores everything, no limit
├── SlidingWindowMemory      — returns last 40 messages
├── SummarizedMemory         — summarizes when > max_messages
└── CachedSummarizedMemory   — summarizes + persists chunks to disk
```

#### SlidingWindowMemory

Keeps all messages but `get_memory()` only returns the **last 40**. No LLM calls, no configuration needed.

```python
from games.agents.memory import SlidingWindowMemory

memory = SlidingWindowMemory()
# Use in agent config:
# memory:
#   type: SlidingWindowMemory
#   kwargs: {}
```

#### SummarizedMemory

When message count exceeds `max_messages`, the oldest messages are sent to an LLM for summarization. The summary replaces the old messages, keeping context compact. The last "Moderator" message is always preserved.

```python
from games.agents.memory import SummarizedMemory

memory = SummarizedMemory(
    max_messages=20,           # Trigger summarization threshold (default: 20)
    model_name='qwen-plus',    # Model for summarization
    api_key='...',             # Or use OPENAI_API_KEY env var
    base_url='...',            # Or use OPENAI_BASE_URL env var
    temperature=0.7,
    max_tokens=2048,
)
```

Or via YAML config:

```yaml
agent:
  type: ThinkingReActAgent
  kwargs:
    memory:
      type: SummarizedMemory
      kwargs:
        max_messages: 30
        model_name: qwen-plus
```

Settings can also be loaded from `games/agents/memory/memory_config.yaml` or the path in the `MEMORY_CONFIG_PATH` environment variable.

#### CachedSummarizedMemory

Extends `SummarizedMemory` with **disk persistence**. When the threshold is exceeded, messages are flushed to a numbered cache chunk on disk (with both the raw messages and a summary). `get_memory()` returns `[cache summaries..., live messages...]`.

```python
from games.agents.memory import CachedSummarizedMemory

memory = CachedSummarizedMemory(
    max_messages=20,
    model_name='qwen-plus',
    game_id='game_001',           # Used for cache directory structure
    agent_id='Player0',           # Per-agent cache separation
    log_dir='logs/my_experiment', # Base directory for cache files
)

# Cache is stored as:
#   logs/my_experiment/Player0/cache/chunk_0000/content.json
#   logs/my_experiment/Player0/cache/chunk_0000/summary.txt

# Inspect cached chunks
overview = memory.get_cache_overview()
# [{'chunk_id': 0, 'content_path': '...', 'summary_path': '...', 'message_count': 20}, ...]

# Load a specific cached chunk back into memory
messages = await memory.load_cached_chunk(chunk_id=0)
```

Or via YAML config (contextual parameters `agent_id`, `game_id`, `log_dir` are injected automatically by the agent factory):

```yaml
agent:
  type: CacheRetrievalAgent
  kwargs:
    memory:
      type: CachedSummarizedMemory
      kwargs:
        max_messages: 40
```

#### CacheRetrievalAgent + CachedSummarizedMemory

The `CacheRetrievalAgent` (default for Diplomacy) is specifically designed to work with `CachedSummarizedMemory`. It adds **retrieval** capabilities — the agent can recall relevant cached chunks using a retrieval model:

```python
from games.agents.cache_retrieval_agent import CacheRetrievalAgent
from games.agents.memory import CachedSummarizedMemory

# The agent automatically detects CachedSummarizedMemory
agent = CacheRetrievalAgent(
    name="Player0",
    model=model,
    memory=CachedSummarizedMemory(),
    retrieval_top_k=1,             # Number of chunks to retrieve
    # retrieval_model_config={...}, # Optional separate model for retrieval
    # retrieval_prompt="...",       # Custom retrieval prompt template
)

# Agent methods for cache access:
msg = await agent.recall_cache_chunk(chunk_id=0)          # Load specific chunk
msg = await agent.recall_cache_by_query("alliance plans")  # Retrieve by relevance
overview = agent.cache_overview()                          # List cached chunks
```

#### How Memory Integrates with Agents

Inside every agent, memory is used during reasoning:

```python
# Inside ThinkingReActAgent._reasoning():
prompt = await self.formatter.format(
    msgs=[
        Msg("system", self.sys_prompt, "system"),
        *await self.memory.get_memory(),    # <-- Memory retrieved here
        *reasoning_hints,
    ],
)
response = await self.model(prompt)
```

Messages (including the agent's own responses) are stored back into memory via `await self.memory.add(msg)`.

---

## 8. Diplomacy: Using the Engine Directly

The Diplomacy game engine (`diplomacy.Game`) can be called directly without LLM agents, just like the Avalon engine. This is useful for building custom agents, scripted bots, or gym-style wrappers.

### Creating a Game

```python
from diplomacy import Game

# Create a standard 7-player Diplomacy game
game = Game(map_name='standard', seed=42)

# Inspect powers
for power_name, power in game.powers.items():
    print(f"{power_name}: units={power.units}, centers={power.centers}, homes={power.homes}")
# AUSTRIA: units=['A BUD', 'A VIE', 'F TRI'], centers=['BUD', 'VIE', 'TRI'], homes=['BUD', 'VIE', 'TRI']
# ENGLAND: units=['A LVP', 'F EDI', 'F LON'], centers=['LVP', 'EDI', 'LON'], homes=['LVP', 'EDI', 'LON']
# ...
```

### Phase System

Phases follow the naming convention `[Season][Year][Type]`:

| Example | Season | Year | Type | Description |
|---------|--------|------|------|-------------|
| `S1901M` | Spring | 1901 | Movement | Spring movement phase |
| `S1901R` | Spring | 1901 | Retreat | Spring retreat (if units dislodged) |
| `F1901M` | Fall | 1901 | Movement | Fall movement phase |
| `F1901R` | Fall | 1901 | Retreat | Fall retreat (if units dislodged) |
| `W1901A` | Winter | 1901 | Adjustment | Build/disband units |

Special phases: `FORMING` (not started), `COMPLETED` (game finished).

```python
phase = game.get_current_phase()   # e.g. 'S1901M'
phase_type = phase[-1]             # 'M', 'R', or 'A'
is_done = game.is_game_done        # True when phase == 'COMPLETED'
```

### Querying Possible Orders

```python
# Get locations that need orders for a specific power
orderable_locs = game.get_orderable_locations('FRANCE')
# e.g. ['PAR', 'MAR', 'BRE']

# Get all possible orders for all locations
possible_orders = game.get_all_possible_orders()
# {'PAR': ['A PAR H', 'A PAR - BUR', 'A PAR - PIC', ...],
#  'MAR': ['A MAR H', 'A MAR - SPA', 'A MAR - BUR', ...],
#  'BRE': ['F BRE H', 'F BRE - ENG', 'F BRE - MAO', ...], ...}
```

### Order Formats

| Order Type | Format | Example | Description |
|------------|--------|---------|-------------|
| Hold | `U LOC H` | `A PAR H` | Unit holds position |
| Move | `U LOC - DEST` | `A PAR - BUR` | Unit moves to destination |
| Support Hold | `U LOC S U LOC2` | `A MAR S A PAR` | Support another unit holding |
| Support Move | `U LOC S U LOC2 - DEST` | `A MAR S A PAR - BUR` | Support another unit's move |
| Convoy | `F SEA C U LOC - DEST` | `F ENG C A LON - BRE` | Fleet convoys army across sea |
| Retreat | `U LOC R DEST` | `A PAR R MAR` | Retreat dislodged unit |
| Build | `U LOC B` | `A PAR B` | Build new unit (adjustment phase) |
| Disband | `U LOC D` | `A PAR D` | Disband unit (adjustment phase) |

Where `U` is `A` (Army) or `F` (Fleet), and `LOC`/`DEST` are 3-letter location codes.

### Submitting Orders and Processing

```python
from diplomacy import Game

game = Game(map_name='standard')

# Game loop
while not game.is_game_done:
    current_phase = game.get_current_phase()
    print(f"Phase: {current_phase}")

    # For each power, get possible orders and submit
    for power_name, power in game.powers.items():
        if power.is_eliminated():
            continue

        orderable_locs = game.get_orderable_locations(power_name)
        possible_orders = game.get_all_possible_orders()

        # Example: submit hold orders for all units
        orders = []
        for loc in orderable_locs:
            if loc in possible_orders and possible_orders[loc]:
                orders.append(possible_orders[loc][0])  # Pick first valid order
        
        if orders:
            game.set_orders(power_name, orders)

    # Process the phase (resolve orders, advance state)
    game.process()

    # Check supply centers after processing
    for power_name, power in game.powers.items():
        if not power.is_eliminated():
            print(f"  {power_name}: {len(power.centers)} centers, units={power.units}")

print(f"Game over. Outcome: {game.outcome}")
```

### Accessing Power State

```python
power = game.powers['FRANCE']

power.units          # ['A PAR', 'A MAR', 'F BRE'] — current units
power.centers        # ['PAR', 'MAR', 'BRE'] — controlled supply centers
power.homes          # ['PAR', 'MAR', 'BRE'] — home supply centers
power.retreats       # {'A PAR': ['BUR', 'PIC']} — dislodged units & retreat options
power.is_eliminated()  # True if no units, centers, or retreats
power.orders         # {'A PAR': '- BUR'} — submitted orders
```

### Game History

```python
# Order history — all orders from previous phases
game.order_history   # {'S1901M': {'FRANCE': ['A PAR - BUR', ...], ...}, ...}

# Full game outcome
game.outcome         # e.g. 'FRANCE wins by supply center majority'
```

---

## 9. Gym-Style Wrapping

Currently, neither Avalon nor Diplomacy ships with a Gymnasium-style wrapper (unlike the Overcooked environment). However, both engines expose clean state-transition APIs that make wrapping straightforward. Below is a design sketch for how you could build one.

### Why Wrap?

A gym-style `reset()` / `step()` interface enables:
- Standard RL training loops (PPO, DQN, etc.)
- Compatibility with libraries like Stable-Baselines3, RLlib, CleanRL
- Single-agent perspective (control one player/power, others act via policy)

### Avalon Gym Wrapper (sketch)

```python
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from games.games.avalon.engine import AvalonBasicConfig, AvalonGameEnvironment


class AvalonSingleAgentEnv(gym.Env):
    """Gym wrapper for Avalon, controlling a single player."""

    metadata = {"render_modes": ["human"]}

    def __init__(self, num_players=5, controlled_player=0, partner_policy=None):
        super().__init__()
        self.num_players = num_players
        self.controlled_player = controlled_player
        self.partner_policy = partner_policy  # callable(env_state, player_id) -> action

        # Action space depends on current phase:
        #   Phase 0 (Team Selection): choose a team (combinatorial)
        #   Phase 1 (Team Voting): approve/reject (binary)
        #   Phase 2 (Quest Voting): pass/fail (binary)
        #   Phase 3 (Assassination): choose target (player index)
        # Simplification: use Discrete for voting phases
        self.action_space = spaces.Discrete(2)  # 0=reject/fail, 1=approve/pass

        # Observation: encode game state as a dict or flat array
        self.observation_space = spaces.Dict({
            "phase": spaces.Discrete(4),
            "quest_results": spaces.MultiBinary(5),
            "my_role_id": spaces.Discrete(8),
            "my_side": spaces.Discrete(2),  # 0=evil, 1=good
            "round": spaces.Discrete(5),
            "turn": spaces.Discrete(5),
        })

        self.env = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        config = AvalonBasicConfig.from_num_players(self.num_players)
        self.env = AvalonGameEnvironment(config)
        self.roles = self.env.get_roles()
        return self._get_obs(), self._get_info()

    def step(self, action):
        phase, _ = self.env.get_phase()

        # Let partner agents act for non-controlled players
        if phase == 0:
            # Team selection — controlled player may or may not be leader
            leader = self.env.get_quest_leader()
            if leader == self.controlled_player:
                team = self._action_to_team(action)
            else:
                team = self.partner_policy(self.env, leader, phase)
            self.env.choose_quest_team(team=frozenset(team), leader=leader)

        elif phase == 1:
            votes = self._collect_votes(action, phase)
            _, done, approved = self.env.gather_team_votes(votes)

        elif phase == 2:
            quest_team = self.env.get_current_quest_team()
            if self.controlled_player in quest_team:
                votes = self._collect_quest_votes(action)
            else:
                votes = self._collect_quest_votes(None)
            _, done, succeeded, num_fails = self.env.gather_quest_votes(votes)

        elif phase == 3:
            assassin = self.env.get_assassin()
            if assassin == self.controlled_player:
                target = action  # action encodes target player
            else:
                target = self.partner_policy(self.env, assassin, phase)
            self.env.choose_assassination_target(assassin, target)

        terminated = self.env.done
        reward = self._compute_reward() if terminated else 0.0
        return self._get_obs(), reward, terminated, False, self._get_info()

    def _get_obs(self):
        role_id, _, is_good = self.roles[self.controlled_player]
        results = self.env.quest_results + [0] * (5 - len(self.env.quest_results))
        return {
            "phase": self.env.phase,
            "quest_results": np.array(results, dtype=np.int8),
            "my_role_id": int(role_id),
            "my_side": int(is_good),
            "round": self.env.round,
            "turn": self.env.turn,
        }

    def _get_info(self):
        return {"roles": self.roles, "env": self.env}

    def _compute_reward(self):
        is_good = self.roles[self.controlled_player][2]
        return 1.0 if (is_good == self.env.good_victory) else 0.0

    # ... helper methods for _collect_votes, _action_to_team, etc.
```

### Diplomacy Gym Wrapper (sketch)

```python
import gymnasium as gym
from gymnasium import spaces
from diplomacy import Game


class DiplomacySinglePowerEnv(gym.Env):
    """Gym wrapper for Diplomacy, controlling a single power."""

    metadata = {"render_modes": ["human"]}

    def __init__(self, controlled_power="FRANCE", max_phases=40, partner_policy=None):
        super().__init__()
        self.controlled_power = controlled_power
        self.max_phases = max_phases
        self.partner_policy = partner_policy  # callable(game, power_name) -> list[str]

        # Action space: index into possible orders per unit
        # Since the number of possible orders varies, use a flexible encoding
        # Option A: Discrete per unit (multi-discrete)
        # Option B: Text-based (for LLM agents)
        self.action_space = spaces.Sequence(spaces.Discrete(100))  # simplified

        # Observation: game state for this power
        self.observation_space = spaces.Dict({
            "phase": spaces.Text(min_length=1, max_length=20),
            "my_units": spaces.Sequence(spaces.Text(min_length=1, max_length=10)),
            "my_centers": spaces.Sequence(spaces.Text(min_length=1, max_length=10)),
            "num_centers": spaces.Discrete(35),
        })

        self.game = None
        self.phases_processed = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.game = Game(map_name='standard', seed=seed or 42)
        self.phases_processed = 0
        return self._get_obs(), self._get_info()

    def step(self, action):
        """
        action: list of order strings for the controlled power
               e.g. ['A PAR - BUR', 'F BRE - ENG', 'A MAR H']
        """
        # Set orders for controlled power
        if action:
            self.game.set_orders(self.controlled_power, action)

        # Set orders for all other (non-eliminated) powers via partner_policy
        for power_name, power in self.game.powers.items():
            if power_name == self.controlled_power or power.is_eliminated():
                continue
            if self.partner_policy:
                orders = self.partner_policy(self.game, power_name)
            else:
                orders = self._random_orders(power_name)
            if orders:
                self.game.set_orders(power_name, orders)

        # Process the phase
        self.game.process()
        self.phases_processed += 1

        terminated = self.game.is_game_done
        truncated = self.phases_processed >= self.max_phases
        reward = self._compute_reward()

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def _get_obs(self):
        power = self.game.powers[self.controlled_power]
        return {
            "phase": self.game.get_current_phase(),
            "my_units": power.units,
            "my_centers": list(power.centers),
            "num_centers": len(power.centers),
        }

    def _get_info(self):
        return {
            "game": self.game,
            "possible_orders": self.game.get_all_possible_orders(),
            "orderable_locations": self.game.get_orderable_locations(self.controlled_power),
            "all_powers": {
                name: {"centers": len(p.centers), "units": p.units, "eliminated": p.is_eliminated()}
                for name, p in self.game.powers.items()
            },
        }

    def _compute_reward(self):
        power = self.game.powers[self.controlled_power]
        if power.is_eliminated():
            return -1.0
        return len(power.centers) / 18.0  # Normalized by solo-victory threshold

    def _random_orders(self, power_name):
        """Fallback: random orders for non-controlled powers."""
        import random
        possible = self.game.get_all_possible_orders()
        locs = self.game.get_orderable_locations(power_name)
        orders = []
        for loc in (locs or []):
            if loc in possible and possible[loc]:
                orders.append(random.choice(possible[loc]))
        return orders
```

### Usage Pattern (gym-style loop)

```python
# Diplomacy example
env = DiplomacySinglePowerEnv(controlled_power="FRANCE", max_phases=40)
obs, info = env.reset(seed=42)

while True:
    # Your agent decides orders based on obs and info
    possible = info["possible_orders"]
    locs = info["orderable_locations"]
    orders = [possible[loc][0] for loc in locs if loc in possible and possible[loc]]

    obs, reward, terminated, truncated, info = env.step(orders)
    print(f"Phase: {obs['phase']}, Centers: {obs['num_centers']}, Reward: {reward:.3f}")

    if terminated or truncated:
        break
```

### Design Considerations

| Concern | Avalon | Diplomacy |
|---------|--------|-----------|
| **Action space** | Variable per phase (team selection is combinatorial, voting is binary, assassination is player index) | Variable per phase (list of order strings, number of units changes) |
| **Multi-agent** | 5-10 players with different roles and information | 7 simultaneous powers with private negotiation |
| **Partial observability** | Each role sees different information (Merlin sees evil, Servants see nothing) | All unit positions public, but negotiation messages can be private |
| **Text-based actions** | Natural for LLM agents; need encoding for RL | Order strings are structured; can be tokenized or indexed |
| **Turn structure** | Sequential within a phase (leader proposes, all vote) | Simultaneous orders per phase |

> **Note**: These sketches are starting points. A production wrapper would need to handle the variable action spaces per phase, proper multi-agent support (e.g. PettingZoo), negotiation as a separate channel, and partial observability masking. The Overcooked wrapper in `env_wrappers/` can serve as a reference implementation.

---

## 10. Training Agents

### Step 1: Generate Training Tasks

```bash
python games/generate_train_parquet.py \
    --game avalon \
    --config games/games/avalon/configs/train_config.yaml \
    --output ./train_avalon_tasks.parquet \
    --num_tasks 10
```

In the training config, mark roles as `trainable: true`:

```yaml
roles:
  assassin:
    trainable: true
    model:
      model_name: qwen2.5-14b-instruct
```

### Step 2: Start Training

```bash
# Option 1: One-click script
bash examples/game/avalon/run_train.sh

# Option 2: Python command
python -m agentevolver.main_ppo \
    --config-path="examples/game/avalon" \
    --config-name='config' \
    data.train_files="./train_avalon_tasks.parquet" \
    data.val_files="./train_avalon_tasks.parquet"
```

**Reward Structure:**
- **Avalon**: `1.0` if the agent's team wins, `0.0` otherwise
- **Diplomacy**: `supply_centers / 18` (normalized by solo-victory threshold); `0.0` if eliminated

---

## 11. File Reference

### Core Game Files

| Path | Purpose |
|------|---------|
| `games/games/avalon/engine.py` | `AvalonBasicConfig`, `AvalonGameEnvironment` — game state & transitions |
| `games/games/avalon/game.py` | `AvalonGame`, `avalon_game()` — full game loop with agents |
| `games/games/avalon/prompt.py` | Prompt templates for Avalon (English & Chinese) |
| `games/games/avalon/utils.py` | `Parser`, `GameLogger`, `LanguageFormatter` |
| `games/games/diplomacy/engine.py` | `DiplomacyConfig` — Diplomacy configuration |
| `games/games/diplomacy/game.py` | `DiplomacyGame`, `diplomacy_game()` — full game loop with agents |
| `games/games/diplomacy/utils.py` | Utility functions, prompt loading, order translation |

### Agent & Factory Files

| Path | Purpose |
|------|---------|
| `games/agent_factory.py` | `create_model_from_config()`, `create_agent_from_config()`, `create_memory_from_config()`, `create_formatter_from_config()` |
| `games/agents/thinking_react_agent.py` | `ThinkingReActAgent` — default agent for Avalon |
| `games/agents/cache_retrieval_agent.py` | `CacheRetrievalAgent` — default agent for Diplomacy |
| `games/agents/echo_agent.py` | `EchoAgent` — simple moderator agent |
| `games/agents/memory/` | Custom memory implementations |
| `games/agents/secure_multi_agent_formatter.py` | `SecureMultiAgentFormatter` — message formatting & token management |

### Workflow Files

| Path | Purpose |
|------|---------|
| `games/games/avalon/workflows/rollout_workflow.py` | `AvalonRolloutWorkflow` — training rollout for Avalon |
| `games/games/avalon/workflows/eval_workflow.py` | `EvalAvalonWorkflow` — evaluation workflow for Avalon |
| `games/games/diplomacy/workflows/rollout_workflow.py` | `DiplomacyWorkflow` — training rollout for Diplomacy |
| `games/games/diplomacy/workflows/eval_workflow.py` | `EvalDiplomacyWorkflow` — evaluation workflow for Diplomacy |

### Configuration & Evaluation Files

| Path | Purpose |
|------|---------|
| `games/games/avalon/configs/` | Avalon YAML configs (default, task, train, arena) |
| `games/games/diplomacy/configs/` | Diplomacy YAML configs (default, task, train, arena) |
| `games/evaluation/run_eval.py` | CLI entry point for batch evaluation |
| `games/evaluation/eval_base.py` | Base evaluation framework |
| `games/evaluation/leaderboard/` | Arena leaderboard system |

### Web Interface Files

| Path | Purpose |
|------|---------|
| `games/web/server.py` | FastAPI web server (Avalon + Diplomacy) |
| `games/web/game_state_manager.py` | WebSocket game state management |
| `games/web/run_web_game.py` | Game thread launcher for web mode |
| `games/web/static/avalon/` | Avalon web UI (HTML/CSS/JS) |
| `games/web/static/diplomacy/` | Diplomacy web UI (HTML/CSS/JS) |

---

## 12. Summary

1. **Install**: `pip install -r games/requirements_game.txt` and set `OPENAI_BASE_URL` / `OPENAI_API_KEY`.
2. **Configure**: Edit YAML configs in `games/games/avalon/configs/` or `games/games/diplomacy/configs/`.
3. **Run programmatically**: Use `avalon_game()` or `diplomacy_game()` convenience functions, or instantiate `AvalonGame` / `DiplomacyGame` directly.
4. **Use the engine directly**: `AvalonGameEnvironment` provides step-by-step game state transitions (team selection, voting, quests, assassination). `diplomacy.Game` provides full Diplomacy state transitions (query possible orders, submit orders, process phases).
5. **Evaluate via CLI**: `python games/evaluation/run_eval.py --game avalon --config <config.yaml> --num-games 10`.
6. **Web interface**: `python games/web/server.py` then visit `http://localhost:8000`.
7. **Memory systems**: Choose from `SlidingWindowMemory` (simple), `SummarizedMemory` (auto-summarize), or `CachedSummarizedMemory` (disk-persistent + retrieval). Configure via YAML or Python.
8. **Train agents**: Generate tasks with `generate_train_parquet.py`, then train with `agentevolver.main_ppo`.
9. **Custom agents**: Extend `AgentBase`, register in YAML config via `agent.type`.
10. **Gym wrapping**: Both engines expose clean state-transition APIs suitable for Gymnasium wrapping. See Section 9 for design sketches.
