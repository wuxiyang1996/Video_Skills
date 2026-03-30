"""Generate training reward curves for each game.

Plots the training curve up to a per-game cutoff and annotates the best
checkpoint chosen from the second half.  Rewards are ego-player-only
(single-agent training mode).  Curves use EMA smoothing (α=0.6) matching
the TensorBoard / W&B style.
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

RUNS_DIR = Path(__file__).parent / "runs"
OUTPUT_DIR = Path(__file__).parent / "training_curves"
OUTPUT_DIR.mkdir(exist_ok=True)

# (display_name, run_dir, game_key, max_step_to_plot or None for all)
GAMES = [
    ("2048",           "Qwen3-8B_2048_20260322_071227",          "twenty_forty_eight", None),
    ("Candy Crush",    "Qwen3-8B_20260321_213813_(Candy_crush)", "candy_crush",        10),
    ("Tetris",         "Qwen3-8B_tetris_20260322_170438",        "tetris",             7),
    ("Super Mario",    "Qwen3-8B_super_mario_20260323_030839",   "super_mario",        10),
    ("Avalon",         "Qwen3-8B_avalon_20260322_200424",        "avalon",             20),
    ("Diplomacy",      "Qwen3-8B_diplomacy_20260322_234548",     "diplomacy",          None),
    ("Avalon (DA)",    "Qwen3-8B_20260326_215431",               "avalon",             None),
    ("Diplomacy (DA)", "Qwen3-8B_20260327_062035",               "diplomacy",          None),
]

COLORS = {
    "2048":            "#E63946",
    "Candy Crush":     "#F4A261",
    "Tetris":          "#2A9D8F",
    "Super Mario":     "#E76F51",
    "Avalon":          "#457B9D",
    "Diplomacy":       "#6D6875",
    "Avalon (DA)":     "#264653",
    "Diplomacy (DA)":  "#B5838D",
}


EMA_WEIGHT = 0.6

def _ema(y, weight=EMA_WEIGHT):
    """Exponential moving average, same algorithm as TensorBoard / W&B smoothing."""
    smoothed = np.empty_like(y, dtype=float)
    smoothed[0] = y[0]
    for i in range(1, len(y)):
        smoothed[i] = weight * smoothed[i - 1] + (1 - weight) * y[i]
    return smoothed


def load_rewards(run_dir: Path, game_key: str, max_step=None):
    """Load reward data from step_log.jsonl + reward JSONL files."""
    by_step: dict[int, dict] = {}
    log_file = run_dir / "step_log.jsonl"
    if log_file.exists():
        with open(log_file) as f:
            for line in f:
                entry = json.loads(line)
                s = entry["step"]
                if max_step is not None and s > max_step:
                    continue
                game_stats = entry.get("reward_per_game", {}).get(game_key)
                if game_stats:
                    by_step[s] = game_stats

    rewards_dir = run_dir / "rewards"
    if rewards_dir.is_dir():
        for rf in sorted(rewards_dir.glob("step_*.jsonl")):
            s = int(rf.stem.split("_")[1])
            if max_step is not None and s > max_step:
                continue
            eps = []
            with open(rf) as f:
                for raw_line in f:
                    line = raw_line.strip()
                    if not line:
                        continue
                    idx = line.find("{")
                    if idx == -1:
                        continue
                    try:
                        eps.append(json.loads(line[idx:]))
                    except json.JSONDecodeError:
                        continue
            rews = [ep["total_reward"] for ep in eps if ep["game"] == game_key]
            if not rews:
                continue
            arr = np.array(rews)
            by_step[s] = {
                "mean_reward": float(arr.mean()),
                "max_reward": float(arr.max()),
                "min_reward": float(arr.min()),
                "std_reward": float(arr.std()),
            }

    if not by_step:
        print(f"  WARNING: no reward data found in {run_dir}")
        return None

    sorted_steps = sorted(by_step.keys())
    steps, means, maxs, mins, stds = [], [], [], [], []
    for s in sorted_steps:
        gs = by_step[s]
        steps.append(s)
        means.append(gs["mean_reward"])
        maxs.append(gs["max_reward"])
        mins.append(gs["min_reward"])
        stds.append(gs.get("std_reward", 0))

    return dict(
        steps=np.array(steps, dtype=float),
        means=np.array(means),
        maxs=np.array(maxs),
        mins=np.array(mins),
        stds=np.array(stds),
    )


def find_best_second_half(data):
    steps, means = data["steps"], data["means"]
    max_step = steps[-1]
    half_step = max_step / 2
    mask = steps >= half_step
    second_half_idx = np.where(mask)[0]
    best_local = int(np.argmax(means[mask]))
    best_idx = second_half_idx[best_local]
    return int(steps[best_idx]), float(means[best_idx]), int(best_idx)


def _draw_curve(ax, data, color, marker_size=4):
    """Draw an EMA-smoothed reward curve with shaded bands."""
    steps, means = data["steps"], data["means"]
    stds, mins, maxs = data["stds"], data["mins"], data["maxs"]

    mean_s = _ema(means)
    upper_s = _ema(means + stds)
    lower_s = _ema(means - stds)
    min_s = _ema(mins)
    max_s = _ema(maxs)

    # Shaded bands
    ax.fill_between(steps, lower_s, upper_s, alpha=0.18, color=color)
    ax.fill_between(steps, min_s, max_s, alpha=0.07, color=color)

    # Raw data as faint dots
    ax.scatter(steps, means, s=marker_size**2, color=color, alpha=0.25, zorder=3, edgecolors="none")

    # Smoothed line
    ax.plot(steps, mean_s, linewidth=2.4, color=color, label="Mean reward (EMA)", zorder=4)


def plot_single_game(name, run_dir_name, game_key, max_step):
    run_dir = RUNS_DIR / run_dir_name
    data = load_rewards(run_dir, game_key, max_step)
    if data is None or len(data["steps"]) == 0:
        print(f"  Skipping {name}: no data")
        return

    color = COLORS[name]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    _draw_curve(ax, data, color)

    ax.set_xlabel("Co-evolution Step", fontsize=18)
    ax.set_ylabel("Ego-Player Reward", fontsize=18)
    ax.set_title(f"{name} — Training Reward Curve", fontsize=20, fontweight="bold")
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.legend(loc="best", framealpha=0.9)
    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    out_path = OUTPUT_DIR / f"{name.lower().replace(' ', '_').replace('(', '').replace(')', '')}.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_combined():
    coevo_games = GAMES[:6]
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for idx, (name, run_dir_name, game_key, max_step) in enumerate(coevo_games):
        ax = axes[idx]
        run_dir = RUNS_DIR / run_dir_name
        data = load_rewards(run_dir, game_key, max_step)
        if data is None or len(data["steps"]) == 0:
            ax.set_title(f"{name} — No Data")
            continue

        color = COLORS[name]
        _draw_curve(ax, data, color, marker_size=3)

        ax.set_xlabel("Co-evolution Step", fontsize=15)
        ax.set_ylabel("Ego-Player Reward", fontsize=15)
        ax.set_title(name, fontsize=18, fontweight="bold")
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax.grid(True, alpha=0.25)

    fig.tight_layout()
    out_path = OUTPUT_DIR / "all_games_combined.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_da_combined():
    da_games = GAMES[6:]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for idx, (name, run_dir_name, game_key, max_step) in enumerate(da_games):
        ax = axes[idx]
        run_dir = RUNS_DIR / run_dir_name
        data = load_rewards(run_dir, game_key, max_step)
        if data is None or len(data["steps"]) == 0:
            ax.set_title(f"{name} — No Data")
            continue

        color = COLORS[name]
        _draw_curve(ax, data, color, marker_size=3)

        ax.set_xlabel("Co-evolution Step", fontsize=15)
        ax.set_ylabel("Ego-Player Reward", fontsize=15)
        ax.set_title(name, fontsize=18, fontweight="bold")
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax.grid(True, alpha=0.25)

    fig.tight_layout()
    out_path = OUTPUT_DIR / "da_games_combined.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


if __name__ == "__main__":
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 14,
        "axes.titlesize": 18,
        "axes.labelsize": 16,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 13,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    print("Generating individual game reward curves...")
    for name, run_dir_name, game_key, max_step in GAMES:
        plot_single_game(name, run_dir_name, game_key, max_step)

    print("\nGenerating combined co-evolution plot...")
    plot_combined()

    print("\nGenerating combined DA plot...")
    plot_da_combined()

    print("\nDone! All plots saved to:", OUTPUT_DIR)
