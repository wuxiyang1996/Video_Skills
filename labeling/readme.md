# Labeling — Episode Annotation with GPT-5.4

This folder contains code and scripts for annotating cold-start episode
trajectories with concise labels suitable for RAG retrieval, the manager
agent, and downstream skill extraction.

There are **three pipelines** available:

| Script | Purpose |
|--------|---------|
| `label_episodes_gpt54.py` | **Labels only** — annotates episodes with `summary_state`, `summary`, `intentions` (leaves `skills` null). |
| `label_and_extract_skills_gpt54.py` | **Labels + Skills** — same annotations **plus** full skill extraction via the `SkillBankAgent` pipeline. Populates `skills` with named, RAG-optimised skill assignments. |
| `extract_skillbank_gpt54.py` | **Skills only** — reads **already-labeled** rollouts (e.g. from `labeling/output/gpt54/`), runs the full SkillBankAgent pipeline, and writes the skill bank and catalogs. No labeling step. |

## What Gets Labeled

For **each experience step** in an episode:

| Field           | Format | LLM? | Description |
|-----------------|--------|------|-------------|
| `summary_state` | `key=value \| key=value` | No | Deterministic, game-aware structured facts. Optimised for RAG embedding retrieval. |
| `summary`       | `summary_state \| note=<strategic note>` | Yes (≤25 tokens) | Same facts as `summary_state` plus a short LLM-generated threat/opportunity note grounded in what changed since the previous step. |
| `intentions`    | `[TAG] subgoal phrase` | Yes (≤40 tokens) | Tagged subgoal with delta-aware tag evolution. Tag shifts when the game situation changes significantly. |
| `skills`        | `{skill_id, skill_name, skill_summary, ...}` or `null` | Yes (when using skill extraction) | Skill assignment from the SkillBankAgent pipeline. Contains RAG-friendly summaries, effects contracts, and segment boundaries. Null when using labels-only pipeline. |

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

| File                                   | Purpose |
|----------------------------------------|---------|
| `label_episodes_gpt54.py`             | Labels-only script. Reads episode JSONs, calls GPT-5.4, writes labeled output with `skills=null`. |
| `label_and_extract_skills_gpt54.py`   | **Full pipeline**: labels + skill extraction via `SkillBankAgent`. Populates the `skills` field with RAG-friendly skill assignments. |
| `extract_skillbank_gpt54.py`          | **Skills only**: reads already-labeled rollouts, runs SkillBankAgent pipeline, writes skill bank and catalogs. No labeling. |
| `run_labeling.sh`                      | Shell wrapper for `label_episodes_gpt54.py`. |
| `run_skill_labeling.sh`               | Shell wrapper for `label_and_extract_skills_gpt54.py`. |
| `run_extract_skillbank.sh`            | Shell wrapper for `extract_skillbank_gpt54.py`. |
| `readme.md`                            | This file. |

---

## Usage — Labels Only (`label_episodes_gpt54.py`)

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

## Usage — Labels + Skill Extraction (`label_and_extract_skills_gpt54.py`)

```bash
# Full pipeline: label episodes AND extract skills for all games
python labeling/label_and_extract_skills_gpt54.py

# Specific game(s)
python labeling/label_and_extract_skills_gpt54.py --games tetris candy_crush

# Dry run (preview first episode)
python labeling/label_and_extract_skills_gpt54.py --dry_run --games tetris

# One rollout per game (quick test)
python labeling/label_and_extract_skills_gpt54.py --one_per_game -v

# Skip labeling (use already-labeled episodes for skill extraction only)
python labeling/label_and_extract_skills_gpt54.py --skip_labeling --labeled_dir labeling/output/gpt54

# Labels only, skip skill extraction (equivalent to label_episodes_gpt54.py)
python labeling/label_and_extract_skills_gpt54.py --skip_skills

# Or use the shell wrapper:
bash labeling/run_skill_labeling.sh --games tetris -v
bash labeling/run_skill_labeling.sh --skip_labeling --labeled_dir labeling/output/gpt54
```

## Usage — Skills only from labeled rollouts (`extract_skillbank_gpt54.py`)

Use this when you already have labeled episodes (e.g. in `labeling/output/gpt54/`) and want to run **only** the skill extraction pipeline (no labeling, no Phase 1).

**Input:** `labeling/output/gpt54/` (or `--input_dir <path>`) — directory with `<game>/episode_*.json` where each episode has `summary_state`, `summary`, and `intentions` filled in.  
**Output:** `labeling/output/gpt54_skillbank/` (or `--output_dir <path>`) — per-game skill banks, catalogs, sub_episodes, and cross-game archetypes.

```bash
# From Game-AI-Agent root
export OPENROUTER_API_KEY="sk-or-..."
export PYTHONPATH="$(pwd):$(pwd)/../GamingAgent:$PYTHONPATH"

# Extract skills for all games (reads labeling/output/gpt54, writes labeling/output/gpt54_skillbank)
python labeling/extract_skillbank_gpt54.py

# Specific game(s)
python labeling/extract_skillbank_gpt54.py --games tetris super_mario

# Custom input/output
python labeling/extract_skillbank_gpt54.py --input_dir labeling/output/gpt54 --output_dir path/to/out

# Quick test: one episode per game
python labeling/extract_skillbank_gpt54.py --one_per_game -v

# Preview without running (dry run)
python labeling/extract_skillbank_gpt54.py --dry_run

# Re-segment against seeded bank (second pass); save annotated episodes to output
python labeling/extract_skillbank_gpt54.py --resegment --save_annotated

# Skip cross-game archetype aggregation
python labeling/extract_skillbank_gpt54.py --skip_archetypes

# Or use the shell wrapper (sets PYTHONPATH, checks input dir):
bash labeling/run_extract_skillbank.sh
bash labeling/run_extract_skillbank.sh --games tetris -v
bash labeling/run_extract_skillbank.sh --dry_run
```

## CLI Options

### Common Options (both scripts)

| Flag              | Default                           | Description |
|-------------------|-----------------------------------|-------------|
| `--input_dir`     | `cold_start/output/gpt54`        | Input directory with `<game>/episode_*.json` |
| `--input_file`    | —                                 | Label a single file instead of scanning a directory |
| `--output_dir`    | `labeling/output/gpt54` or `gpt54_skills` | Output directory for labeled episodes |
| `--games`         | all found                         | Filter to specific game(s) |
| `--model`         | `gpt-5.4`                        | LLM model for labeling |
| `--max_episodes`  | all                               | Cap episodes per game |
| `--one_per_game`  | off                               | Process only the first episode for each game |
| `--delay`         | `0.1`                            | Seconds between API calls (rate limiting) |
| `--overwrite`     | off                               | Re-label already-labeled episodes |
| `--in_place`      | off                               | Write back to original input files |
| `--dry_run`       | off                               | Preview without saving |
| `--verbose / -v`  | off                               | Print per-step details |

### Skill Extraction Options (`label_and_extract_skills_gpt54.py` only)

| Flag                | Default | Description |
|---------------------|---------|-------------|
| `--skip_labeling`   | off     | Skip Phase 1; use pre-labeled episodes |
| `--labeled_dir`     | —       | Path to pre-labeled episodes (for `--skip_labeling`) |
| `--skip_skills`     | off     | Skip Phase 2; run labels only (like `label_episodes_gpt54.py`) |
| `--skip_archetypes` | off     | Skip Phase 3; skip cross-game archetype aggregation |

### Skills-only script (`extract_skillbank_gpt54.py`)

| Flag                | Default                           | Description |
|---------------------|-----------------------------------|-------------|
| `--input_dir`       | `labeling/output/gpt54`          | Directory with **labeled** game sub-folders (`<game>/episode_*.json`). Episodes must have `summary_state`, `summary`, `intentions`. |
| `--output_dir`      | `labeling/output/gpt54_skillbank`| Root for per-game skill banks, catalogs, sub_episodes, archetypes. |
| `--games`           | all found                         | Only process these games. |
| `--model`           | `gpt-5.4`                        | LLM model for skill naming/description. |
| `--max_episodes`    | all                               | Cap episodes per game. |
| `--one_per_game`    | off                               | Process only the first episode per game. |
| `--resegment`       | off                               | Re-run pipeline against seeded bank (second pass). |
| `--skip_archetypes` | off                               | Skip cross-game archetype aggregation. |
| `--save_annotated`  | off                               | Write episode JSONs with `skills` populated to output. |
| `--dry_run`         | off                               | Preview games/episodes without running extraction. |
| `--verbose / -v`   | off                               | Per-step details. |

## Output Structure

### Labels Only (`labeling/output/gpt54/`)

```
labeling/output/gpt54/
├── tetris/
│   ├── episode_000.json          # labeled episode (skills=null)
│   ├── episode_001.json
│   └── labeling_summary.json     # per-game stats
├── candy_crush/
│   └── ...
└── labeling_batch_summary.json   # overall run stats
```

### Labels + Skills (`labeling/output/gpt54_skills/`)

```
labeling/output/gpt54_skills/
├── tetris/
│   ├── episode_000.json          # labeled episode with skills populated
│   ├── episode_001.json
│   ├── labeling_summary.json     # per-game stats (includes skill counts)
│   ├── skill_bank.jsonl          # persistent skill bank (contracts)
│   ├── skill_catalog.json        # RAG-friendly skill catalog (per-game)
│   └── reports/                  # skill extraction diagnostics
├── candy_crush/
│   └── ...
├── skill_archetypes.json         # cross-game archetype aggregation
├── skill_rag_index.json          # flat RAG index (archetypes + all instances)
├── labeling_batch_summary.json   # overall run stats
└── skill_catalog_all.json        # combined per-game catalog
```

### Skills only — from labeled rollouts (`extract_skillbank_gpt54.py`)

**Input:** Already-labeled episode JSONs with `summary_state`, `summary`, and `intentions` populated (e.g. from `label_episodes_gpt54.py` or the labels-only phase). Default input directory: `labeling/output/gpt54/` with layout `<input_dir>/<game>/episode_*.json`.

**Output:** Default output directory: `labeling/output/gpt54_skillbank/`. No episode JSONs are written unless `--save_annotated` is used.

```
labeling/output/gpt54_skillbank/
├── tetris/
│   ├── skill_bank.jsonl          # persistent skill bank (contracts)
│   ├── skill_catalog.json        # RAG-friendly skill catalog (per-game)
│   ├── sub_episodes.json         # SubTask_Experience instances
│   ├── extraction_summary.json   # per-game stats (episodes, skills, sub_episodes)
│   └── reports/                  # skill extraction diagnostics (if any)
├── candy_crush/
│   └── ...
├── skill_archetypes.json         # cross-game archetype aggregation
├── skill_rag_index.json          # flat RAG index (archetypes + instances)
├── extraction_batch_summary.json # overall run stats
└── skill_catalog_all.json        # combined per-game catalog
```

| Option | Default | Description |
|--------|---------|-------------|
| **Input** | `labeling/output/gpt54` | Directory with game sub-folders; each game folder contains `episode_*.json` (labeled, with `intentions`). |
| **Output** | `labeling/output/gpt54_skillbank` | Root for per-game skill banks, catalogs, sub_episodes, and cross-game archetypes. |
| `--save_annotated` | off | If set, writes episode JSONs with the `skills` field populated into the per-game output folder. |

## Skill Extraction Pipeline

The `label_and_extract_skills_gpt54.py` script runs four phases:

```
Phase 1 — Annotation (same as label_episodes_gpt54.py)
  └─ for each step:
       ├─ summary_state = build_rag_summary()       # deterministic
       ├─ summary = summary_state + LLM note         # 1 LLM call
       └─ intentions = [TAG] phrase                   # 1 LLM call

Phase 2 — Skill Extraction (per game, via SkillBankAgent)
  ├─ Stage 1: Boundary proposal (predicate flips, events, change-points)
  ├─ Stage 2: Skill decoding (preference-learned scorer + Viterbi DP)
  ├─ Stage 3: Contract learning (eff_add / eff_del / eff_event)
  ├─ Materialize NEW: promote __NEW__ segments to named skills
  ├─ GPT-5.4 naming: generate skill_name + RAG summary per skill
  └─ Annotate: populate skills field on each experience

Phase 3 — Cross-Game Archetype Aggregation (runs after all games)
  ├─ Group skills by dominant SUBGOAL_TAG across all games
  ├─ GPT-5.4: generate archetype name, description, transfer summary
  ├─ skill_archetypes.json         # archetype → game instances
  └─ skill_rag_index.json          # flat index for vector store

Phase 4 — Persistence
  ├─ <game>/skill_bank.jsonl       # per-game JSONL skill bank
  ├─ <game>/skill_catalog.json     # per-game RAG catalog
  ├─ skill_catalog_all.json        # combined per-game catalog
  └─ labeling_batch_summary.json   # run statistics
```

### Skill Field Format

When skill extraction is enabled, each experience gets a `skills` dict:

```json
{
  "skill_id": "S_new_1741779200_0",
  "skill_name": "Clear Bottom Rows",
  "skill_summary": "game=tetris | skill=clear_bottom | effects=rows_cleared,stack_lowered | context=when holes accumulate in bottom rows",
  "description": "Targets bottom-row line clears in Tetris when holes accumulate below row 10. Use when stack height exceeds 12 and holes concentrate in the lower half.",
  "segment_start": 30,
  "segment_end": 45,
  "eff_add": ["rows_cleared", "stack_lowered"],
  "eff_del": ["holes_bottom"],
  "eff_event": ["line_clear"]
}
```

### Skill Catalog (for RAG)

The `skill_catalog.json` is designed for easy ingestion into a vector store:

```json
{
  "game": "tetris",
  "model": "gpt-5.4",
  "n_skills": 5,
  "skills": [
    {
      "skill_id": "S_new_1741779200_0",
      "name": "Clear Bottom Rows",
      "summary": "game=tetris | skill=clear_bottom | effects=rows_cleared,stack_lowered | context=holes in lower half",
      "description": "Targets bottom-row line clears when holes accumulate below row 10.",
      "eff_add": ["rows_cleared", "stack_lowered"],
      "eff_del": ["holes_bottom"],
      "eff_event": ["line_clear"],
      "n_instances": 12,
      "version": 1
    }
  ]
}
```

Each skill's `summary` field uses the same `key=value` format as `summary_state`,
optimised for RAG embedding retrieval. The `description` field provides natural
language for human readability and broader semantic matching.

### Cross-Game Archetypes (`skill_archetypes.json`)

Skills are aggregated across games by their dominant `SUBGOAL_TAG`. Each
archetype represents an abstract strategic pattern with game-specific instances:

```json
{
  "n_archetypes": 8,
  "n_total_skills": 23,
  "n_games": 4,
  "archetypes": [
    {
      "archetype_id": "archetype_survive",
      "tag": "SURVIVE",
      "name": "Emergency Resource Recovery",
      "description": "Emergency pattern triggered when a critical resource is nearly depleted. Prioritises immediate relief over long-term optimisation.",
      "transfer_summary": "archetype=survive | pattern=emergency_recovery | trigger=resource_critical | strategy=prioritize_immediate_relief | games=tetris,2048,candy_crush",
      "games": ["tetris", "2048", "candy_crush"],
      "n_skills": 5,
      "instances": [
        {
          "game": "tetris",
          "skill_id": "S_new_...",
          "skill_name": "Clear Bottom Rows",
          "summary": "game=tetris | skill=clear_bottom | effects=rows_cleared,stack_lowered",
          "description": "Targets bottom-row line clears when holes accumulate."
        },
        {
          "game": "2048",
          "skill_id": "S_new_...",
          "skill_name": "Emergency Merge",
          "summary": "game=2048 | skill=emergency_merge | effects=space_created",
          "description": "Force merges to create space when board is nearly full."
        }
      ]
    }
  ]
}
```

**Key design**: Per-game skill banks stay isolated (game-specific predicates don't
mix). Archetypes are a **read-only overlay** that groups skills by strategic intent
for cross-game transfer during RAG retrieval.

### RAG Index (`skill_rag_index.json`)

A flat list of entries ready for vector-store ingestion. Contains two entry types:

- **`archetype`**: cross-game pattern with `transfer_summary` for retrieval
- **`skill_instance`**: game-specific skill linked to its parent archetype

```json
{
  "n_entries": 31,
  "entries": [
    {
      "id": "archetype_survive",
      "type": "archetype",
      "tag": "SURVIVE",
      "name": "Emergency Resource Recovery",
      "text": "archetype=survive | pattern=emergency_recovery | trigger=resource_critical | strategy=prioritize_immediate_relief | games=tetris,2048,candy_crush",
      "description": "Emergency pattern triggered when a critical resource is nearly depleted.",
      "games": ["tetris", "2048", "candy_crush"]
    },
    {
      "id": "archetype_survive_tetris_S_new_...",
      "type": "skill_instance",
      "tag": "SURVIVE",
      "archetype": "archetype_survive",
      "game": "tetris",
      "name": "Clear Bottom Rows",
      "text": "game=tetris | skill=clear_bottom | effects=rows_cleared,stack_lowered",
      "description": "Targets bottom-row line clears when holes accumulate."
    }
  ]
}
```

Use the `text` field as the embedding content and `type` field to filter
between archetypes (cross-game patterns) and instances (game-specific skills).

## Architecture — Labels Only

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

## Architecture — Labels + Skill Extraction

```
main()
  ├─ Phase 1: label_episode()                        # per game, per episode
  │    └─ (same as labels-only architecture above)
  │
  ├─ Phase 2: extract_skills_for_game()              # per game
  │    ├─ _dict_to_episode()                         # convert JSON → Episode objects
  │    ├─ SkillBankAgent.segment_episode()            # Stage 1+2 per episode
  │    │    ├─ boundary_proposal (Stage 1)            # candidate cut points
  │    │    └─ infer_segmentation (Stage 2)           # preference-learned decoding
  │    ├─ SkillBankAgent.run_contract_learning()      # Stage 3 effects contracts
  │    ├─ SkillBankAgent.materialize_new_skills()     # promote __NEW__ → named skills
  │    ├─ _generate_skill_name()                      # GPT-5.4 name + RAG summary
  │    ├─ _generate_skill_description()               # GPT-5.4 description
  │    └─ annotate_episodes_with_skills()             # populate skills field
  │
  ├─ Phase 3: aggregate_cross_game_archetypes()      # runs AFTER all games
  │    ├─ _extract_dominant_tag()                     # classify by SUBGOAL_TAG
  │    ├─ Group skills by tag across games            # SURVIVE, CLEAR, etc.
  │    ├─ GPT-5.4: archetype name + transfer summary  # cross-game RAG text
  │    ├─ skill_archetypes.json                       # structured archetypes
  │    └─ skill_rag_index.json                        # flat vector-store index
  │
  └─ Phase 4: save outputs
       ├─ <game>/episode_NNN.json (with skills)
       ├─ <game>/skill_bank.jsonl
       ├─ <game>/skill_catalog.json
       ├─ skill_catalog_all.json
       └─ labeling_batch_summary.json
```

## Pipeline Integration

The labeled episodes follow the same `Episode` / `Experience` schema from
`data_structure/experience.py` and can be loaded directly:

```python
from data_structure.experience import Episode
import json

# Load labeled episode (with or without skills)
with open("labeling/output/gpt54_skills/tetris/episode_000.json") as f:
    ep = Episode.from_dict(json.load(f))

for exp in ep.experiences:
    print(exp.summary_state)  # structured key=value facts for RAG
    print(exp.summary)        # summary_state + strategic note
    print(exp.intentions)     # [TAG] subgoal phrase
    print(exp.skills)         # skill assignment dict (or None)
```

### Loading the Skill Catalog for RAG

```python
import json

with open("labeling/output/gpt54_skills/tetris/skill_catalog.json") as f:
    catalog = json.load(f)

for skill in catalog["skills"]:
    # Index these into your vector store
    print(skill["skill_id"], skill["name"])
    print(f"  summary: {skill['summary']}")
    print(f"  description: {skill['description']}")
```

### Loading Archetypes for Cross-Game RAG

```python
import json

# Load the flat RAG index — ready for vector store ingestion
with open("labeling/output/gpt54_skills/skill_rag_index.json") as f:
    index = json.load(f)

for entry in index["entries"]:
    if entry["type"] == "archetype":
        # Cross-game pattern — embed entry["text"] for broad retrieval
        print(f"[ARCHETYPE] {entry['name']}: {entry['text']}")
    else:
        # Game-specific instance — embed for precise in-game retrieval
        print(f"  [{entry['game']}] {entry['name']}: {entry['text']}")

# Or load the structured archetypes for programmatic access
with open("labeling/output/gpt54_skills/skill_archetypes.json") as f:
    archetypes = json.load(f)

for arch in archetypes["archetypes"]:
    print(f"\n[{arch['tag']}] {arch['name']} — {arch['n_games']} games, {arch['n_skills']} skills")
    for inst in arch["instances"]:
        print(f"  {inst['game']}: {inst['skill_name']}")
```

### Using with SkillBankAgent

```python
from skill_agents.pipeline import SkillBankAgent, PipelineConfig
from skill_agents.skill_bank.bank import SkillBankMVP

# Load the extracted skill bank
bank = SkillBankMVP(path="labeling/output/gpt54_skills/tetris/skill_bank.jsonl")
bank.load()

agent = SkillBankAgent(bank=bank)
result = agent.query_skill("clear bottom rows to reduce holes")
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

## Functions Used from `skill_agents`

| Function / Class | Source | Role in Skill Extraction |
|-----------------|--------|--------------------------|
| `SkillBankAgent` | `skill_agents/pipeline.py` | Orchestrates the full skill extraction pipeline (Stages 1–4) |
| `PipelineConfig` | `skill_agents/pipeline.py` | Configuration for all pipeline stages |
| `SkillBankMVP` | `skill_agents/skill_bank/bank.py` | Persistent effects-only contract storage (JSONL) |
| `SkillEffectsContract` | `skill_agents/stage3_mvp/schemas.py` | Per-skill effects contract (eff_add, eff_del, eff_event) |
| `SegmentRecord` | `skill_agents/stage3_mvp/schemas.py` | Enriched segment with predicate summaries and effects |
| `SkillQueryEngine` | `skill_agents/query.py` | RAG-based skill retrieval and selection |
