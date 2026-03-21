# Patches

Tracking fixes and patches applied to the codebase for future reference.

---

## 001 — Candy Crush: inject dynamic action mapping into GymEnvAdapter (2026-03-11)

**File:** `GamingAgent/gamingagent/envs/custom_03_candy_crush/candyCrushEnv.py` (line ~266)

**Problem:**
`game_env_config.json` ships with `"action_mapping": {}` because Candy Crush
has 112 valid swap actions on an 8x8 board — too many to enumerate manually
in the config (unlike 2048/Sokoban which have only 4 directional actions).

Because the adapter's `move_to_action_idx` was always empty, every call to
`_parse_agent_action_str` triggered:

1. `adapter.map_agent_action_to_env_action()` → miss → **warning printed**
2. Fallback regex parse of `((r1,c1),(r2,c2))` → lookup in
   `env_move_to_action_idx` → success

Actions executed correctly, but the per-step warning was noisy and the
unnecessary fallback added overhead.

**Fix:**
After `CandyCrushEnv.__init__` dynamically builds `env_move_to_action_idx`
from the board geometry, inject that mapping into the adapter when the
adapter's own mapping (from config) is empty:

```python
if not self.adapter.move_to_action_idx and self.env_move_to_action_idx:
    self.adapter.move_to_action_idx = {k.lower(): v for k, v in self.env_move_to_action_idx.items()}
    self.adapter.action_idx_to_move = {v: k for k, v in self.env_move_to_action_idx.items()}
```

Now the adapter's first lookup succeeds directly — no warning, no regex
fallback needed.

---

## 002 — Orak: disable screenshot storage for Pokemon Red & Super Mario (2026-03-15)

**Files changed (Orak repo):**
- `Orak/src/mcp_game_servers/pokemon_red/game/pyboy_runner.py` — `take_screenshot()`
- `Orak/src/mcp_game_servers/pokemon_red/game/pokemon_red_env.py` — `PokemonRedEnv`
- `Orak/src/mcp_game_servers/super_mario/game/super_mario_env.py` — `SuperMarioEnv`

**Files changed (Game-AI-Agent):**
- `evaluate_orak/orak_nl_wrapper.py` — `make_orak_env()`
- `evaluate_orak/orak_gym_like.py` — `make_orak_gaming_env()`

**Problem:**
Pokemon Red called `PyBoyRunner.take_screenshot()` on every `initial_obs()`
and `step()`, writing a timestamped PNG to `<log_path>/screenshots/`. Over
long evaluation runs this accumulated gigabytes of unused images. Super Mario
had a similar `save_state_image()` (commented out) and was still converting
every frame to a PIL image unnecessarily.

**Fix:**
1. `pyboy_runner.py`: `take_screenshot()` now accepts `save=True`. When
   `save=False` it returns the PIL image from `pyboy.screen.image` without
   writing to disk.
2. `pokemon_red_env.py`: Added `save_screenshots: bool = True` to `Config`.
   Both `initial_obs()` and `step()` pass `save=self.save_screenshots` to
   `take_screenshot()`.
3. `super_mario_env.py`: Added `save_screenshots: bool = True` to `Config`.
   When False, `initial_obs()` and `step()` skip the `to_pil_image()`
   conversion (sets `image=None`). Also added a `save` guard to
   `save_state_image()`.
4. `orak_nl_wrapper.py` and `orak_gym_like.py`: Both wrappers set
   `cfg.env.save_screenshots = False` for `pokemon_red` / `super_mario`
   (resp. `orak_pokemon_red` / `orak_super_mario`) via `omegaconf.open_dict`.

All defaults remain `True` so other games and direct Orak usage are
unaffected.

---

## 003 — GRPO: inf segment reward crashes FSDP training with SIGABRT (2026-03-17)

**Files changed:**
- `skill_agents_grpo/infer_segmentation/diagnostics.py` — `SegmentDiagnostic.margin`, `SegmentationDiagnostics.from_result`
- `skill_agents_grpo/grpo/rewards.py` — `_segmentation_reward_with_decode`
- `trainer/coevolution/grpo_training.py` — `_compute_advantages`
- `skill_agents_grpo/grpo/fsdp_trainer.py` — training loop

**Problem:**
`SegmentDiagnostic.margin` returned `float("inf")` when a segment had fewer
than 2 skill candidates (common during cold start with a small skill bank).
The reward function `_segmentation_reward_with_decode` consumed these margins
without filtering, so `sum([1.8, 1.8, inf])` produced an `inf` reward.

This `inf` propagated through three stages:
1. **Reward** → `inf` returned by `_segmentation_reward_with_decode`
2. **Advantage** → `_compute_advantages` computed `mean = inf`, `var = nan`,
   all advantages became `nan`
3. **FSDP** → `nan` advantage fed into the PPO loss, one rank diverged while
   others stayed finite, NCCL collective communication deadlocked, and
   `torch.multiprocessing.spawn` raised `ProcessExitedException: process 0
   terminated with signal SIGABRT`

The diagnostics *display* path (`SegmentationDiagnostics.from_result`) already
filtered `inf` margins, but the *reward* path did not.

**Fix (4 layers):**
1. **`diagnostics.py`** — `margin` now returns `1e6` instead of `float("inf")`
   as the sentinel for "no comparison possible". `from_result` updated to
   filter `margin < 1e6` accordingly. Eliminates `inf` at the source.
2. **`rewards.py`** — `_segmentation_reward_with_decode` now filters margins
   with `math.isfinite(seg.margin)` before averaging, mirroring the
   diagnostics display path. Prevents `inf` rewards even if margin sentinels
   change.
3. **`grpo_training.py`** — `_compute_advantages` sanitizes non-finite rewards
   by replacing them with the mean of finite rewards before normalization.
   Prevents `nan` advantages.
4. **`fsdp_trainer.py`** — Training loop skips any sample whose advantage or
   loss is non-finite, substituting a zero-gradient contribution. Prevents
   any future numerical divergence from crashing the entire run.

---

## 004 — Sokoban: exempt from stuck detection + reduce step penalty + distance shaping (2026-03-21)

**Files changed:**
- `Game-AI-Agent/trainer/coevolution/episode_runner.py` — stuck detection exempt set
- `GamingAgent/gamingagent/envs/custom_02_sokoban/sokobanEnv.py` — step penalty, distance shaping

**Problem:**
Sokoban reward was stuck at -1.0 to -1.4 across 30 co-evolution steps with
no improvement. Root causes:

1. **Stuck detection killed episodes too early.** Every Sokoban step incurs a
   -0.1 penalty, so `sum(last 15 rewards) = -1.5 <= 0` always triggered at
   step ~20. Episodes never ran long enough for the agent to discover
   positive reward (box-on-target = +1.0, puzzle solved = +10.0).

2. **Step penalty too harsh.** At -0.1/step, a 200-step episode costs -20.0
   just from moving. The +1.0 box placement and +10.0 puzzle bonuses were
   drowned out, giving GRPO no meaningful reward variance.

3. **Reward signal too sparse.** The only positive reward came from pushing a
   box onto a target — but with episodes killed at step 20 and the agent
   wandering aimlessly, this never occurred. GRPO had zero positive examples.

**Fix (3 changes):**

1. **Exempt Sokoban from stuck detection** (`episode_runner.py`):
   Added `"sokoban"` to `_STUCK_EXEMPT_GAMES` (renamed from
   `_SPARSE_REWARD_GAMES`). Episodes now run the full 200 steps, giving the
   agent time to discover box-on-target rewards.

2. **Reduce step penalty -0.1 → -0.02** (`sokobanEnv.py`):
   A full 200-step episode now costs -4.0 instead of -20.0. The +1.0 box
   placement and +10.0 puzzle bonuses are now significant relative to the
   movement cost, creating meaningful reward variance for GRPO.

3. **Add distance-based reward shaping** (`sokobanEnv.py`):
   New `_sum_box_target_distance()` computes total Manhattan distance from
   each free box to its nearest target. Each step receives
   `+0.1 × distance_reduction` — pushing a box 1 cell closer to a target
   gives +0.1 bonus, pushing it away gives -0.1 penalty. This creates a
   smooth gradient toward solving the puzzle instead of a sparse binary
   signal. Distance is initialized in `reset()` and tracked across steps.

---

## 005 — Tetris: board-quality reward shaping for holes and height (2026-03-21)

**Files changed:**
- `GamingAgent/gamingagent/envs/custom_04_tetris/tetrisEnv.py` — hole counting, height penalty, commit reward shaping

**Problem:**
Tetris reward was stuck at ~10-11 across 30 co-evolution steps. The agent
learned exactly one strategy: spam `hard_drop` without positioning pieces
(no left/right/rotate). Every episode looked identical: ~10 pieces dropped
straight down → game over in ~10 steps → total reward ~10-11.

Root cause — **reward trap**: `hard_drop` gives +1.0 per piece placed, but
movement/rotation gives 0.0. The agent only learned the immediately
rewarding action. No line clears ever occurred (+10.0 bonus) because
clearing requires proper positioning first — which pays nothing. With no
reward variance, GRPO advantages were flat and no learning happened.

**Fix (3 additions to `_commit_active_tetromino`):**

1. **`_board_holes()` method**: Counts holes — empty cells with at least one
   filled cell above them in the same column. Measures placement quality.

2. **`_max_col_height()` method**: Returns the height of the tallest column
   in the game area. Measures danger level.

3. **Board-quality shaping in `_commit_active_tetromino()`**:
   - **-0.3 per new hole created** — penalizes sloppy drops that create
     unreachable gaps (e.g. hard_drop creating 3 holes: +1.0 - 0.9 = +0.1)
   - **+0.2 per hole eliminated** — rewards placements that fill gaps
   - **Height penalty when stack > 75% board height** — `-0.5 × (h/20 - 0.75)`
     penalizes dangerous stacking that leads to game-over

   This breaks the reward trap: a blind hard_drop that creates 3 holes now
   gets +0.1 instead of +1.0, while a well-positioned drop gets the full
   +1.0 (or more with line clears + hole reduction). GRPO can now
   distinguish good from bad placements and learn positioning.

---

## 006 — 2048: Game-AI-Agent techniques backported to GamingAgent inference agent (2026-03-21)

**Files changed:**
- `GamingAgent/gamingagent/envs/custom_01_2048/twentyFortyEightEnv.py` — complete observation pipeline rewrite
- `GamingAgent/gamingagent/configs/custom_01_2048/module_prompts.json` — all prompts rewritten
- `GamingAgent/gamingagent/configs/custom_01_2048/config.yaml` — max_memory 10→15

**Problem:**
The GamingAgent 2048 agent sent a raw Python dict as the text observation
(`{'board': [[0, 2, 4, ...]], 'highest_tile': 64, 'analysis': 'Board has
10 empty spaces'}`). The LLM had to mentally simulate tile mechanics for
all 4 moves, guess at board quality, and had no phase-awareness, no recent
action context, and no urgency signals. This left significant performance
on the table compared to Game-AI-Agent's co-evolution agent, which uses
structured state summaries, phase-aware skill protocols, urgency detection,
and anti-repetition guards.

**Fix (8 techniques backported from Game-AI-Agent):**

1. **Phase detection** (from `phase_detector._extract_2048_phases`):
   Classifies board as `opening`/`midgame`/`endgame` using occupancy and
   highest-tile thresholds (`occupancy<0.35 && max<=32` → opening,
   `occupancy>0.7 || max>=256` → endgame). Each phase activates a
   different skill protocol.

2. **Structured state summary** (from `agent_helper.build_rag_summary`):
   Compact `key=value` header line replaces the raw dict:
   `game=2048 | step=42 | phase=midgame | score=5680 | max_tile=128 |
   empty=5/16 | merges=0 | corner=top-left | mono=-2.0 | smooth=-21.0`

3. **Move lookahead** (uses static `apply_action()` to simulate all 4
   moves): Shows the LLM the resulting board, score, empty cells, merge
   count, monotonicity, and smoothness for every legal move. Eliminates
   the need for mental tile simulation.

4. **Heuristic board scoring** (from `gamingagent_nl_wrapper._count_merges`
   + new metrics): Monotonicity (how well tiles are ordered in rows/cols),
   smoothness (adjacent tile similarity), corner proximity (which corner
   the max tile is nearest), and merge count (adjacent same-value pairs).

5. **Urgency detection** (from `episode_runner._detect_urgency`):
   Fires `!! URGENCY: board nearly full — must MERGE now !!` when
   `empty < 3`. Softer warning at `empty < 5` in endgame.

6. **State delta tracking** (from `episode_runner._compute_state_delta`):
   Shows `Changed: max_tile 64→128, empty 8→6` between consecutive steps
   so the LLM sees what its last move accomplished.

7. **Recent action/reward context** (from `episode_runner._build_recent_context`):
   Shows last 5 actions with their rewards. Warns `Recent actions got
   0 reward. Try a DIFFERENT strategy!` when 3+ moves scored nothing.

8. **Anti-repetition guard** (from `episode_runner._apply_anti_repetition`):
   If the same action is repeated `MAX_REPEAT_ZERO_REWARD` (2) consecutive
   times with zero total reward, a random alternative is forced. Prevents
   the agent from getting stuck in fruitless loops.

**Prompt changes (from Game-AI-Agent episode_runner prompt structure):**

- **Compound skill protocols** (from `pipeline._GAME_DEFAULT_SEEDS` +
  `skill_enrichment`): Each phase has a named skill with strategy, plan
  steps, success criteria, and abort criteria. Injected as `=== ACTIVE
  SKILL: midgame:POSITION+OPTIMIZE ===` into the observation.

- **Intention/subgoal system** (from `_generate_intention` + `SUBGOAL_TAGS`):
  Added `subgoal: [TAG] objective` to the output format. Tags: SETUP,
  MERGE, POSITION, OPTIMIZE, SURVIVE. Forces strategic planning before
  action selection.

- **Data-driven reasoning instructions**: Prompts tell the LLM to use
  the pre-computed lookahead scores as primary decision data — not to
  mentally re-simulate tile mechanics.

- **Reflection prompt** rewritten to focus on corner anchor integrity,
  phase alignment, merge efficiency, and concrete next-step priorities
  (under 80 words, must cite tile values and positions).

**Config change:**
- `max_memory`: 10 → 15 (more trajectory context for richer reflections)
