# Labeling — Episode Annotation with GPT-5.4

This folder contains code and scripts for annotating cold-start episode
trajectories with concise labels suitable for RAG retrieval, the manager
agent, and downstream skill extraction.

## What Gets Labeled

For **each experience step** in an episode:

| Field           | Format | LLM? | Description |
|-----------------|--------|------|-------------|
| `summary_state` | `key=value \| key=value` | No | Deterministic, game-aware structured facts. Optimised for RAG embedding retrieval. |
| `summary`       | `summary_state \| note=<strategic note>` | Yes (≤25 tokens) | Same facts as `summary_state` plus a short LLM-generated threat/opportunity note grounded in what changed since the previous step. |
| `intentions`    | `[TAG] subgoal phrase` | Yes (≤40 tokens) | Tagged subgoal with delta-aware tag evolution. Tag shifts when the game situation changes significantly. |
| `skills`        | `null` | — | Renamed from `sub_tasks`. Populated by the skill pipeline downstream. |

### Subgoal Tags

Intentions use one of these categorical tags for skill segmentation:

```
SETUP | CLEAR | MERGE | ATTACK | DEFEND | NAVIGATE | POSITION |
COLLECT | BUILD | SURVIVE | OPTIMIZE | EXPLORE | EXECUTE
```

### Design Principles

- **Deterministic-first**: `summary_state` is fully deterministic (0 LLM tokens).
  Game-aware extractors pull structured facts (e.g. Tetris holes, 2048 highest
  tile) so key information is never lost even if the LLM call fails.
- **Delta-aware**: Both `summary` and `intentions` receive a state-delta from the
  previous step (e.g. `holes:10->32`), ensuring the LLM note and tag reflect
  what actually changed rather than repeating generic advice.
- **Urgency detection**: Absolute-value thresholds trigger urgency hints for
  critical situations (e.g. Tetris holes>25, 2048 empty<3), so intention tags
  shift to `[SURVIVE]` or `[CLEAR]` even when per-step deltas are small.
- **Concise for RAG**: `summary_state` uses `key=value` pairs that tokenise
  cleanly for embedding models. The keyword scoring in `EpisodicMemoryStore`
  splits on `=|:,;/` so individual keys and values match independently.
- **Tight LLM budget**: The LLM adds only a short strategic note (~10 words,
  25 tokens) and a tagged subgoal phrase (~15 words, 40 tokens) — designed for
  14B models with minimal output budgets.

## Examples

### Tetris (86 steps)

```
step 0/86 (opening):
  summary_state: game=tetris | phase=opening | step=0/86 | stack_h=0 | piece=L | next=S,O,I,J | level=1 | reward=+1
  summary:       game=tetris | phase=opening | step=0/86 | stack_h=0 | piece=L | next=S,O,I,J | level=1 | reward=+1 | note=Flat left stack; avoid right-side overhangs.
  intentions:    [SETUP] Place L flat to preserve a clean, flexible opening stack

step 50/86 (midgame — holes spiked):
  summary_state: game=tetris | phase=midgame | step=50/86 | stack_h=14 | holes=32 | next=T,Z,I,J | level=1
  summary:       game=tetris | phase=midgame | step=50/86 | stack_h=14 | holes=32 | next=T,Z,I,J | level=1 | note=Sharp hole spike from recent stack rise; avoid covering center wells.
  intentions:    [CLEAR] Place S flat to reduce holes and set up immediate line clears

step 85/86 (endgame — near ceiling):
  summary_state: game=tetris | phase=endgame | step=85/86 | stack_h=17 | holes=38 | piece=Z | next=T,L,O,S | level=1 | reward=+1
  summary:       game=tetris | phase=endgame | step=85/86 | stack_h=17 | holes=38 | piece=Z | next=T,L,O,S | level=1 | reward=+1 | note=Z increased holes; avoid center lock, preserve left well.
  intentions:    [SURVIVE] Place Z vertically to avoid overhangs and prevent topping out
```

**Tag evolution**: `[SETUP]` → `[POSITION]` → `[CLEAR]` → `[SURVIVE]` as holes grow and stack nears ceiling.

### 2048 (152 steps)

```
step 0/152 (opening):
  summary_state: game=2048 | phase=opening | step=0/152 | highest=2 | empty=14 | tiles=2,2
  summary:       game=2048 | phase=opening | step=0/152 | highest=2 | empty=14 | tiles=2,2 | note=Two isolated 2s; merge by moving right or down.
  intentions:    [SETUP] Consolidate the two 2-tiles toward one side for an early merge

step 75/152 (midgame):
  summary_state: game=2048 | phase=midgame | step=75/152 | highest=64 | empty=4 | tiles=64,32,16,16,8 | merges=1 | reward=+4
  summary:       game=2048 | phase=midgame | step=75/152 | highest=64 | empty=4 | tiles=64,32,16,16,8 | merges=1 | reward=+4 | note=Rightmost column open; safe right/down merges available
  intentions:    [MERGE] Merge left for space while keeping the 64 anchored and middle chain stable

step 151/152 (endgame — board full):
  summary_state: game=2048 | phase=endgame | step=151/152 | highest=128 | empty=0 | tiles=128,64,32,16,16 | merges=1 | reward=+4
  summary:       game=2048 | phase=endgame | step=151/152 | highest=128 | empty=0 | tiles=128,64,32,16,16 | merges=1 | reward=+4 | note=No empty cells; next move must create one.
  intentions:    [SURVIVE] Merge vertical 2s now to create space and survive
```

**Tag evolution**: `[SETUP]` → `[MERGE]` → `[EXECUTE]` → `[SURVIVE]` as empty cells deplete.

### Candy Crush (50 steps)

```
step 0/50 (opening):
  summary_state: game=candy crush | phase=opening | step=0/50 | score=0 | moves=50 | board=8x8 | pairs=18 | reward=+3
  summary:       game=candy crush | phase=opening | step=0/50 | score=0 | moves=50 | board=8x8 | pairs=18 | reward=+3 | note=No immediate 3-match; look for central column setup.
  intentions:    [ATTACK] Make immediate 3-candy match with swap maximizing score and preserving flexibility

step 32/50 (midgame):
  summary_state: game=candy crush | phase=midgame | step=32/50 | score=366 | moves=18 | board=8x8 | pairs=18 | reward=+6
  summary:       game=candy crush | phase=midgame | step=32/50 | score=366 | moves=18 | board=8x8 | pairs=18 | reward=+6 | note=Center-right red cluster nearly aligned for next-turn match.
  intentions:    [ATTACK] Swap for an immediate match and better cascade potential

step 49/50 (endgame — final move):
  summary_state: game=candy crush | phase=endgame | step=49/50 | score=507 | moves=1 | board=8x8 | pairs=14 | reward=+12
  summary:       game=candy crush | phase=endgame | step=49/50 | score=507 | moves=1 | board=8x8 | pairs=14 | reward=+12 | note=New 4-candy setup available; exploit final move for bigger clear.
  intentions:    [EXECUTE] Use the final move to make the immediate horizontal match on row 3
```

**Tag evolution**: `[ATTACK]` → `[CLEAR]` → `[ATTACK]` → `[EXECUTE]` as moves deplete and focus shifts to maximising remaining turns.

## Files

| File                      | Purpose |
|---------------------------|---------|
| `label_episodes_gpt54.py` | Main labeling script. Reads episode JSONs, calls GPT-5.4, writes labeled output. |
| `run_labeling.sh`         | Convenience shell wrapper (sets PYTHONPATH, runs the script). |
| `readme.md`               | This file. |

## Usage

```bash
# From Game-AI-Agent root
export OPENROUTER_API_KEY="sk-or-..."
export PYTHONPATH="$(pwd):$(pwd)/../GamingAgent:$PYTHONPATH"

# Label all games (reads from cold_start/output/gpt54/, writes to labeling/output/gpt54/)
python labeling/label_episodes_gpt54.py

# Label specific game(s)
python labeling/label_episodes_gpt54.py --games tetris candy_crush

# Label a single episode file
python labeling/label_episodes_gpt54.py --input_file cold_start/output/gpt54/tetris/episode_000.json

# Dry run — preview labeling on one episode without saving
python labeling/label_episodes_gpt54.py --dry_run --games tetris --max_episodes 1

# Process exactly one rollout per game (quick test across all games)
python labeling/label_episodes_gpt54.py --one_per_game -v

# Write labels in-place (overwrite originals)
python labeling/label_episodes_gpt54.py --in_place

# Or use the shell wrapper:
bash labeling/run_labeling.sh --games tetris -v
```

## CLI Options

| Flag              | Default                           | Description |
|-------------------|-----------------------------------|-------------|
| `--input_dir`     | `cold_start/output/gpt54`        | Input directory with `<game>/episode_*.json` |
| `--input_file`    | —                                 | Label a single file instead of scanning a directory |
| `--output_dir`    | `labeling/output/gpt54`          | Output directory for labeled episodes |
| `--games`         | all found                         | Filter to specific game(s) |
| `--model`         | `gpt-5.4`                        | LLM model for labeling |
| `--max_episodes`  | all                               | Cap episodes per game |
| `--one_per_game`  | off                               | Process only the first episode for each game |
| `--delay`         | `0.1`                            | Seconds between API calls (rate limiting) |
| `--overwrite`     | off                               | Re-label already-labeled episodes |
| `--in_place`      | off                               | Write back to original input files |
| `--dry_run`       | off                               | Preview without saving |
| `--verbose / -v`  | off                               | Print per-step details |

## Output Structure

```
labeling/output/gpt54/
├── tetris/
│   ├── episode_000.json          # labeled episode
│   ├── episode_001.json
│   └── labeling_summary.json     # per-game stats
├── candy_crush/
│   └── ...
└── labeling_batch_summary.json   # overall run stats
```

## Architecture

```
label_episode()
  └─ for each step:
       ├─ summary_state = build_rag_summary()       # deterministic, 0 LLM tokens
       │    ├─ extract_game_facts()                  # game-specific parsers
       │    └─ estimate_game_phase()                 # opening/midgame/endgame
       │
       ├─ delta = _compute_state_delta(prev, curr)   # what changed since last step
       │
       ├─ summary = summary_state + LLM note         # 1 LLM call, ≤25 tokens
       │    └─ prompt includes delta → note is specific to the change
       │
       └─ intentions = [TAG] phrase                   # 1 LLM call, ≤40 tokens
            ├─ prompt includes delta + urgency hints
            └─ tag shifts when situation changes significantly
```

## Pipeline Integration

The labeled episodes follow the same `Episode` / `Experience` schema from
`data_structure/experience.py` and can be loaded directly:

```python
from data_structure.experience import Episode
import json

with open("labeling/output/gpt54/tetris/episode_000.json") as f:
    ep = Episode.from_dict(json.load(f))

for exp in ep.experiences:
    print(exp.summary_state)  # structured key=value facts for RAG
    print(exp.summary)        # summary_state + strategic note
    print(exp.intentions)     # [TAG] subgoal phrase
```

## Functions Used from `decision_agents`

| Function | Source | Role in Labeling |
|----------|--------|-----------------|
| `build_rag_summary()` | `agent_helper.py` | Deterministic `key=value` summary from game-aware fact extraction |
| `extract_game_facts()` | `agent_helper.py` | Game-specific parsers (Tetris holes, 2048 tiles, Candy score, etc.) |
| `compact_text_observation()` | `agent_helper.py` | Fallback state pre-compression when game-specific extraction is sparse |
| `get_state_summary()` | `agent_helper.py` | Structured/text state summarisation backbone |
| `infer_intention()` | `agent_helper.py` | Fallback intention inference when GPT-5.4 call fails |
| `strip_think_tags()` | `agent_helper.py` | Strip `<think>` blocks from reasoning model output |
| `SUBGOAL_TAGS` | `agent_helper.py` | Canonical list of 13 subgoal tag categories |
| `HARD_SUMMARY_CHAR_LIMIT` | `agent_helper.py` | Maximum character limit for summary strings |
