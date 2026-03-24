# Adaptive Pipeline — Unified Plan

**Created:** 2026-03-22  
**Status:** Open  
**Depends on:** `CO_EVOLUTION_PLAN.md`, `SKILL_EVALUATION_REDESIGN.md`  
**Existing code:** `trainer/coevolution/orchestrator.py`, `trainer/coevolution/config.py`, `trainer/coevolution/episode_runner.py`, `trainer/coevolution/grpo_training.py`

---

## Overview

The co-evolution loop currently uses **static hyperparameters** for most
per-step decisions: temperature, episode counts, max steps, UCB
exploration, reward weights, and skill bank thresholds.  These are tuned
once and baked into `CoEvolutionConfig` or hard-coded constants.

This plan replaces six static knobs with **data-driven adaptive
controllers** that react to per-game reward trends, bank maturity, and
training progress.  Each controller is simple (no learned parameters),
cheap (O(1) per step), and independently toggleable via env vars.

**Design principle:** every adaptive controller observes the same metric
history that already flows through W&B logging — no new data collection
is needed.  Each has a sensible static fallback when disabled.

---

## Architecture

```
                     ┌──────────────────────────────────┐
                     │     AdaptiveController (new)      │
                     │                                    │
                     │  Inputs:                           │
                     │    per_game_reward_history[game][]  │
                     │    bank_maturity[game]              │
                     │    step / total_steps               │
                     │                                    │
                     │  Outputs (per step):               │
                     │    temperature[game]                │
                     │    episodes[game]                   │
                     │    max_steps[game]                  │
                     │    ucb_c                            │
                     │    reward_weights                   │
                     │    bank_thresholds                  │
                     └──────┬───────────────────────────┘
                            │
          ┌─────────────────┼──────────────────┐
          │                 │                  │
          ▼                 ▼                  ▼
    episode_runner      grpo_training     skillbank_pipeline
    (temp, max_steps,   (per-game LR      (materialization,
     UCB coeff)          weighting)        retirement thresholds)
```

A single `AdaptiveController` instance lives on the orchestrator and is
called once per co-evolution step.  It reads the W&B-logged
`per_game_reward_history` dict and computes all adaptive outputs.
Controllers are pure functions with no internal learned state.

---

## Part A: Adaptive Per-Game Temperature

### Problem

Temperature is globally scheduled via `grpo_schedule()`.  Diplomacy
(negotiation, high entropy needed) and 2048 (convergent, low entropy
preferred) share the same temperature.  This means either Diplomacy
under-explores or 2048 over-explores.

### Design

Compute per-game temperature based on **reward trend** (improving →
lower temp, stagnant → higher temp) and **reward variance** (high
variance → lower temp, low variance → higher temp):

```python
def adaptive_temperature(
    game: str,
    reward_history: list[float],
    base_temp: float,
    step: int,
    total_steps: int,
) -> float:
    if len(reward_history) < 3:
        return base_temp  # not enough data

    recent = reward_history[-5:]
    trend = np.polyfit(range(len(recent)), recent, 1)[0]
    variance = np.var(recent)

    # Improving → cool down for exploitation
    # Stagnant/declining → heat up for exploration
    trend_factor = np.clip(-trend * 10, -0.15, 0.15)

    # High variance → cool down (already diverse enough)
    var_factor = -0.1 * min(1.0, variance / max(np.mean(recent) ** 2, 1e-6))

    # Global annealing: gentle cooldown over training
    anneal = 0.1 * (step / max(1, total_steps))

    adjusted = base_temp + trend_factor + var_factor - anneal
    return float(np.clip(adjusted, 0.3, 1.2))
```

Per-game temperatures are passed into `run_episode_async()` and override
the global `config.temperature` for LLM calls within that game's
episodes.

### Files

| File | Change |
|------|--------|
| `trainer/coevolution/adaptive.py` | New file: `AdaptiveController` with `adaptive_temperature()` |
| `trainer/coevolution/episode_runner.py` | Accept `temperature` override in `EpisodeConfig` |
| `trainer/coevolution/rollout_collector.py` | Pass per-game temperature from controller |
| `trainer/coevolution/orchestrator.py` | Instantiate `AdaptiveController`, call per step |

---

## Part B: Adaptive Episode Budget

### Problem

Every game gets the same `episodes_per_game` (default 4).  Games that
are improving benefit from more data; games that have plateaued waste
compute.  The total episode budget is fixed.

### Design

Redistribute a fixed total budget based on reward trends:

```python
def adaptive_episodes(
    games: list[str],
    reward_history: dict[str, list[float]],
    total_budget: int,
    min_per_game: int = 2,
    max_per_game: int = 8,
) -> dict[str, int]:
    scores = {}
    for game in games:
        history = reward_history.get(game, [])
        if len(history) < 3:
            scores[game] = 1.0  # neutral
            continue

        recent = history[-5:]
        trend = np.polyfit(range(len(recent)), recent, 1)[0]

        # Improving games get more budget
        # Declining games still get minimum (they need exploration)
        # Flat games get slightly below average
        if trend > 0.01:
            scores[game] = 1.5   # improving
        elif trend < -0.01:
            scores[game] = 0.8   # declining but still needs attempts
        else:
            scores[game] = 0.9   # flat

    # Softmax-like allocation
    total_score = sum(scores.values())
    allocation = {}
    remaining = total_budget
    for game in games:
        raw = total_budget * scores[game] / max(total_score, 1e-6)
        count = int(np.clip(round(raw), min_per_game, max_per_game))
        allocation[game] = count
        remaining -= count

    # Distribute surplus to highest-trend games
    if remaining > 0:
        ranked = sorted(games, key=lambda g: scores[g], reverse=True)
        for g in ranked:
            if remaining <= 0:
                break
            if allocation[g] < max_per_game:
                allocation[g] += 1
                remaining -= 1

    return allocation
```

The `total_budget` is `len(games) * config.episodes_per_game` (same
total compute).  Only the distribution changes.

### Files

| File | Change |
|------|--------|
| `trainer/coevolution/adaptive.py` | Add `adaptive_episodes()` |
| `trainer/coevolution/rollout_collector.py` | Use per-game episode counts from controller |
| `trainer/coevolution/config.py` | Add `adaptive_episode_budget: bool = True` flag |

---

## Part C: Adaptive Episode Length (Max Steps)

### Problem

`GAME_MAX_STEPS` is static.  Early in training, the agent wastes many
steps achieving nothing (200 steps of random actions in Sokoban).
Later, competent agents might benefit from longer horizons to attempt
harder strategies.

### Design

Scale max steps based on **agent competency** (mean reward relative to
the game's max observed reward):

```python
def adaptive_max_steps(
    game: str,
    reward_history: list[float],
    base_max_steps: int,
    step: int,
    total_steps: int,
) -> int:
    # Warm-up: start at 50% of base, ramp to 100% over first 15 steps
    warmup_fraction = min(1.0, 0.5 + 0.5 * step / 15)

    # Competency bonus: if recent rewards are high, allow up to 130%
    if len(reward_history) >= 3:
        recent_mean = np.mean(reward_history[-5:])
        best_ever = max(reward_history) if reward_history else 1.0
        competency = recent_mean / max(abs(best_ever), 1e-6)
        competency_bonus = 0.3 * np.clip(competency, 0.0, 1.0)
    else:
        competency_bonus = 0.0

    scale = warmup_fraction + competency_bonus
    scaled = int(base_max_steps * np.clip(scale, 0.4, 1.3))

    # Round to nearest 10 for cleanliness
    return max(10, (scaled // 10) * 10)
```

This means:
- Step 0, Sokoban: `200 * 0.5 = 100` max steps (faster iteration)
- Step 15, Sokoban improving: `200 * 1.1 = 220` max steps
- Step 40, Sokoban plateaued at low reward: `200 * 0.7 = 140` max steps

### Files

| File | Change |
|------|--------|
| `trainer/coevolution/adaptive.py` | Add `adaptive_max_steps()` |
| `trainer/coevolution/episode_runner.py` | Use dynamic `max_steps` per episode |
| `trainer/coevolution/config.py` | Add `adaptive_max_steps: bool = True` flag |

---

## Part D: Adaptive UCB Exploration Decay

### Problem

The UCB exploration coefficient `c=0.15` in `_compute_confidence()` is
fixed.  Early in training, the agent should explore many skills; later,
it should exploit proven ones.  Fixed `c` means the agent either
over-explores late or under-explores early.

### Design

Decay `c` from `c_init` to `c_final` using a cosine schedule:

```python
def adaptive_ucb_coeff(
    step: int,
    total_steps: int,
    c_init: float = 0.25,
    c_final: float = 0.05,
) -> float:
    progress = min(1.0, step / max(1, total_steps))
    # Cosine decay: slow decay early, faster near end
    c = c_final + 0.5 * (c_init - c_final) * (1 + math.cos(math.pi * progress))
    return c
```

Timeline for 60-step training:
- Step 0: `c = 0.25` (heavy exploration, try many skills)
- Step 15: `c = 0.21` (still exploring but favoring good skills)
- Step 30: `c = 0.15` (balanced, matches current static value)
- Step 45: `c = 0.09` (exploitation-heavy)
- Step 60: `c = 0.05` (mostly exploit)

### Files

| File | Change |
|------|--------|
| `trainer/coevolution/adaptive.py` | Add `adaptive_ucb_coeff()` |
| `trainer/coevolution/episode_runner.py` | Pass UCB coeff into skill selection |
| `skill_agents_grpo/query.py` | Accept `ucb_c` parameter in `_compute_confidence()` |

---

## Part E: Adaptive Skill Bank Thresholds

### Problem

Materialization, retirement, and promotion thresholds are fixed.  The
existing `ProtoSkill.set_relaxed(step <= 15)` is binary — relaxed for
15 steps, then hard cutoff.  As the bank matures, quality standards
should rise smoothly.

### Design

Three thresholds adapt to **bank maturity** (n_skills per game):

```python
def adaptive_bank_thresholds(
    n_skills: int,
    step: int,
    total_steps: int,
) -> dict[str, float]:
    # Bank maturity: 0.0 (empty) → 1.0 (>= 20 skills)
    maturity = min(1.0, n_skills / 20.0)

    # Training progress: 0.0 (start) → 1.0 (end)
    progress = min(1.0, step / max(1, total_steps))

    # Combined signal: 60% maturity, 40% training progress
    combined = 0.6 * maturity + 0.4 * progress

    return {
        # Materialization: low bar early (accept speculative skills),
        # high bar late (only well-evidenced skills)
        "materialize_min_instances": max(1, int(1 + 4 * combined)),
        "materialize_min_pass_rate": 0.3 + 0.4 * combined,

        # Retirement: lenient early (give skills time), strict late
        "retire_min_score": 0.1 + 0.2 * combined,
        "retire_min_episodes": max(3, int(3 + 5 * combined)),

        # Protocol refinement: frequent early, less often late
        "refine_interval": max(2, int(2 + 4 * combined)),
    }
```

Timeline for a game reaching 20 skills by step 30:

| Step | n_skills | Materialize (min instances) | Retire threshold | Refine interval |
|------|----------|-----------------------------|------------------|-----------------|
| 0    | 0        | 1 instance, 30% pass rate   | score < 0.10, 3+ eps | every 2 steps |
| 10   | 5        | 2 instances, 40% pass rate  | score < 0.15, 4+ eps | every 3 steps |
| 30   | 20       | 4 instances, 58% pass rate  | score < 0.24, 7+ eps | every 5 steps |
| 50   | 20       | 5 instances, 66% pass rate  | score < 0.28, 8+ eps | every 6 steps |

### Files

| File | Change |
|------|--------|
| `trainer/coevolution/adaptive.py` | Add `adaptive_bank_thresholds()` |
| `trainer/coevolution/skillbank_pipeline.py` | Pass thresholds to Stage 4 |
| `skill_agents_grpo/pipeline.py` | Accept threshold overrides in materialization/retirement |
| `skill_agents_grpo/stage3_mvp/schemas.py` | Remove binary `set_relaxed()`, accept threshold dict |

---

## Part F: Adaptive Reward Weighting

### Problem

The decision agent reward has fixed component weights:
`r = r_env + 0.1 * r_follow + r_cost`.  Early in training, the skill
bank's contracts are unreliable — `r_follow` is noisy.  As contracts
improve, `r_follow` becomes a stronger signal and deserves more weight.

### Design

Scale `w_follow` based on **bank contract quality**:

```python
def adaptive_reward_weights(
    n_skills: int,
    mean_pass_rate: float,
    step: int,
) -> dict[str, float]:
    # Contract reliability: low when bank is small or pass rates are low
    reliability = min(1.0, n_skills / 15.0) * min(1.0, mean_pass_rate / 0.7)

    return {
        # r_follow: ramp up as contracts become reliable
        "w_follow": 0.05 + 0.25 * reliability,

        # r_cost: reduce penalty as agent learns to use skills efficiently
        "w_cost": max(0.01, 0.05 * (1 - reliability)),
    }
```

Timeline:
- Step 0 (empty bank): `w_follow=0.05`, `w_cost=0.05` (almost ignore follow)
- Step 15 (10 skills, 50% pass rate): `w_follow=0.14`, `w_cost=0.03`
- Step 40 (20 skills, 80% pass rate): `w_follow=0.29`, `w_cost=0.01`

### Files

| File | Change |
|------|--------|
| `trainer/coevolution/adaptive.py` | Add `adaptive_reward_weights()` |
| `decision_agents/reward_func.py` | Accept weight overrides, currently hard-coded `w_follow=0.1` |
| `trainer/coevolution/episode_runner.py` | Pass weights from controller to `RewardComputer` |

---

## AdaptiveController — Unified Interface

All six sub-controllers live in a single class:

```python
@dataclass
class AdaptiveController:
    """Computes all adaptive hyperparameters from reward history.

    Instantiated once in the orchestrator.  Call ``update()`` at the
    start of each co-evolution step.
    """
    total_steps: int
    base_temperature: float = 0.7
    base_episodes_per_game: int = 4
    base_max_steps: dict[str, int] = field(default_factory=lambda: dict(GAME_MAX_STEPS))

    # Feature toggles — each controller can be independently disabled
    enable_adaptive_temp: bool = True
    enable_adaptive_episodes: bool = True
    enable_adaptive_max_steps: bool = True
    enable_adaptive_ucb: bool = True
    enable_adaptive_bank_thresholds: bool = True
    enable_adaptive_reward_weights: bool = True

    # Internal state: per-game reward history
    _reward_history: dict[str, list[float]] = field(default_factory=dict)

    def record_rewards(self, per_game_rewards: dict[str, float]) -> None:
        """Append per-game mean rewards from the latest step."""
        for game, reward in per_game_rewards.items():
            self._reward_history.setdefault(game, []).append(reward)

    def update(
        self,
        step: int,
        games: list[str],
        skill_counts: dict[str, int],
        mean_pass_rate: float = 0.5,
    ) -> "AdaptiveStepConfig":
        """Compute all adaptive parameters for this step."""
        temperatures = {}
        episodes = {}
        max_steps = {}

        for game in games:
            history = self._reward_history.get(game, [])

            temperatures[game] = (
                adaptive_temperature(game, history, self.base_temperature,
                                     step, self.total_steps)
                if self.enable_adaptive_temp
                else self.base_temperature
            )

            max_steps[game] = (
                adaptive_max_steps(game, history,
                                   self.base_max_steps.get(game, 200),
                                   step, self.total_steps)
                if self.enable_adaptive_max_steps
                else self.base_max_steps.get(game, 200)
            )

        total_budget = len(games) * self.base_episodes_per_game
        if self.enable_adaptive_episodes:
            episodes = adaptive_episodes(
                games, self._reward_history, total_budget,
            )
        else:
            episodes = {g: self.base_episodes_per_game for g in games}

        ucb_c = (
            adaptive_ucb_coeff(step, self.total_steps)
            if self.enable_adaptive_ucb
            else 0.15
        )

        n_skills = sum(skill_counts.values())
        bank_thresholds = (
            adaptive_bank_thresholds(n_skills, step, self.total_steps)
            if self.enable_adaptive_bank_thresholds
            else None
        )

        reward_weights = (
            adaptive_reward_weights(n_skills, mean_pass_rate, step)
            if self.enable_adaptive_reward_weights
            else {"w_follow": 0.1, "w_cost": 0.05}
        )

        return AdaptiveStepConfig(
            temperatures=temperatures,
            episodes=episodes,
            max_steps=max_steps,
            ucb_c=ucb_c,
            bank_thresholds=bank_thresholds,
            reward_weights=reward_weights,
        )


@dataclass
class AdaptiveStepConfig:
    """Output of AdaptiveController.update() — consumed by the orchestrator."""
    temperatures: dict[str, float]
    episodes: dict[str, int]
    max_steps: dict[str, int]
    ucb_c: float
    bank_thresholds: dict[str, float] | None
    reward_weights: dict[str, float]
```

---

## Orchestrator Integration

In `orchestrator.py`, add these lines to the per-step loop:

```python
# Before existing code at the top of the step loop:
adaptive = AdaptiveController(
    total_steps=config.total_steps,
    base_temperature=config.temperature,
    base_episodes_per_game=config.episodes_per_game,
)

# Inside the step loop, after computing episode metrics:
adaptive.record_rewards({
    game: metrics["mean_reward"]
    for game, metrics in episode_metrics["per_game"].items()
})

# At the start of each step, before rollout:
step_config = adaptive.update(
    step=step,
    games=config.games,
    skill_counts=sb_manager.skill_counts(),
)

# Pass into rollout collection:
rollout_results = await collect_rollouts(
    config, vllm_client,
    skill_banks=skill_banks,
    per_game_temperature=step_config.temperatures,
    per_game_episodes=step_config.episodes,
    per_game_max_steps=step_config.max_steps,
    ucb_c=step_config.ucb_c,
    reward_weights=step_config.reward_weights,
)
```

W&B logging of all adaptive parameters (per step, per game):

```python
for game in config.games:
    wandb.log({
        f"adaptive/{game}/temperature": step_config.temperatures[game],
        f"adaptive/{game}/episodes": step_config.episodes[game],
        f"adaptive/{game}/max_steps": step_config.max_steps[game],
    }, step=step)
wandb.log({
    "adaptive/ucb_c": step_config.ucb_c,
    "adaptive/w_follow": step_config.reward_weights["w_follow"],
    "adaptive/w_cost": step_config.reward_weights["w_cost"],
}, step=step)
if step_config.bank_thresholds:
    for k, v in step_config.bank_thresholds.items():
        wandb.log({f"adaptive/bank/{k}": v}, step=step)
```

---

## Environment Variables

Each controller is independently toggleable via env vars, defaulting to
**enabled**:

| Variable | Default | Description |
|----------|---------|-------------|
| `ADAPTIVE_TEMPERATURE` | `1` | Per-game temperature scaling |
| `ADAPTIVE_EPISODES` | `1` | Budget redistribution across games |
| `ADAPTIVE_MAX_STEPS` | `1` | Episode length scaling |
| `ADAPTIVE_UCB_DECAY` | `1` | UCB exploration coefficient decay |
| `ADAPTIVE_BANK_THRESHOLDS` | `1` | Smooth bank quality gating |
| `ADAPTIVE_REWARD_WEIGHTS` | `1` | Follow/cost weight scaling |

Set any to `0` to disable that controller and fall back to the existing
static value.

---

## Implementation Checklist

### P0 — Core adaptive infrastructure

- [ ] **Create `trainer/coevolution/adaptive.py`** — all six controller
  functions + `AdaptiveController` + `AdaptiveStepConfig` dataclasses
- [ ] **Unit tests** — test each controller function with synthetic
  reward histories (improving, declining, flat, empty)
- [ ] **Wire into orchestrator** — instantiate controller, call
  `update()` per step, pass outputs to rollout/training

### P1 — Per-game temperature (highest impact)

- [ ] **`episode_runner.py`** — accept `temperature` override in episode
  config, pass to `vllm_client.generate()` calls
- [ ] **`rollout_collector.py`** — thread per-game temperature through
  `build_interleaved_configs()`

### P2 — Episode budget + max steps

- [ ] **`rollout_collector.py`** — use `per_game_episodes` dict instead
  of uniform `config.episodes_per_game`
- [ ] **`episode_runner.py`** — use `max_steps` from step config instead
  of `GAME_MAX_STEPS[game]`

### P3 — UCB decay + reward weights

- [ ] **`skill_agents_grpo/query.py`** — accept `ucb_c` parameter in
  `_compute_confidence()`, currently hard-coded 0.15
- [ ] **`decision_agents/reward_func.py`** — accept `w_follow` and
  `w_cost` overrides in `RewardComputer`

### P4 — Bank thresholds

- [ ] **`skill_agents_grpo/pipeline.py`** — accept threshold dict in
  materialization/retirement logic
- [ ] **`skill_agents_grpo/stage3_mvp/schemas.py`** — replace binary
  `set_relaxed()` with continuous threshold parameters
- [ ] **`trainer/coevolution/skillbank_pipeline.py`** — pass thresholds
  from `AdaptiveStepConfig`

### P5 — Logging + observability

- [ ] **W&B logging** — log all adaptive parameters per step per game
  (temperatures, episodes, max_steps, ucb_c, reward_weights, bank
  thresholds)
- [ ] **TensorBoard** — mirror W&B adaptive scalars
- [ ] **step_log.jsonl** — include adaptive config in step summary

---

## Validation

### A/B comparison

Run two 30-step training runs on the same seed:

1. **Baseline**: all `ADAPTIVE_*` env vars set to `0` (current static behavior)
2. **Adaptive**: all `ADAPTIVE_*` env vars set to `1` (full adaptive pipeline)

Compare on:
- Per-game reward curves (expect adaptive to converge faster on easy games)
- Total training wall time (expect ~same since budget is constant)
- Skill bank size and quality at step 30
- Reward variance across games (expect adaptive to be more uniform)

### Ablation

Disable one controller at a time to measure individual contribution.
Expected ranking by impact:

1. **Temperature** — directly affects LLM output quality
2. **Episode budget** — concentrates compute where it helps most
3. **UCB decay** — prevents late-training exploration waste
4. **Max steps** — speeds up early steps significantly
5. **Reward weights** — improves signal quality for skill selection
6. **Bank thresholds** — smoother skill promotion curve

---

## What Was Cut (intentionally)

- **Per-game learning rate**: requires per-game GRPO batching, which
  conflicts with the current FSDP setup (all games in one batch).
  Revisit if per-game GRPO becomes available.
- **Adaptive GRPO group size**: marginal impact; standard group size 8
  works well across games.
- **Cross-game skill transfer**: fundamentally different architecture
  (shared skill embedding space), too large for this plan.
- **Learned adaptive parameters**: meta-learning the controller weights
  is interesting but adds complexity for uncertain gains.  Static
  formulas with hand-tuned coefficients are sufficient.
- **Online (within-step) adaptation**: temperature/budget changes within
  a single rollout phase.  Overhead of recomputing mid-step outweighs
  benefit since episodes run concurrently.

---

## Key Files

| File | Role |
|------|------|
| `trainer/coevolution/adaptive.py` | **New:** all adaptive controllers |
| `trainer/coevolution/orchestrator.py` | Wire controller into main loop |
| `trainer/coevolution/config.py` | Feature toggle flags, env var parsing |
| `trainer/coevolution/episode_runner.py` | Consume per-game temp, max_steps, UCB, reward weights |
| `trainer/coevolution/rollout_collector.py` | Per-game episode counts and temperatures |
| `skill_agents_grpo/query.py` | Parameterized UCB coefficient |
| `decision_agents/reward_func.py` | Parameterized w_follow, w_cost |
| `skill_agents_grpo/pipeline.py` | Parameterized bank thresholds |
| `skill_agents_grpo/stage3_mvp/schemas.py` | Continuous threshold API replacing binary relaxed mode |
