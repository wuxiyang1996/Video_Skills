#!/usr/bin/env python3
"""Analyze Avalon results: win rate & 95% CI per character role and per good/evil group."""

import json
import numpy as np
from collections import defaultdict
from pathlib import Path
from scipy import stats


def win_rate_ci(wins_array):
    """Compute win rate and 95% CI for a binary array (1=win, 0=loss).

    Uses Wilson score interval for small-sample binary proportions.
    """
    n = len(wins_array)
    if n == 0:
        return float('nan'), float('nan'), float('nan')
    p = np.mean(wins_array)
    if n == 1:
        return p, float('nan'), float('nan')
    z = 1.96
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denom
    return p, center - margin, center + margin


def load_episodes(summary_path):
    with open(summary_path) as f:
        data = json.load(f)
    return data.get('episode_stats', []), data


MODEL_CONFIGS = {}

ablation_base = Path("/workspace/game_agent/Game-AI-Agent/ablation_study/output")
ablation_models = {
    "Base Model (Qwen3-8B)": "base_model_avalon_da_",
    "SFT (no bank)": "sft_no_bank_avalon_da_",
    "RL (no bank)": "no_bank_avalon_da_",
    "SFT + Best Bank": "sft_best_bank_avalon_da_",
    "SFT + First Bank": "sft_first_bank_avalon_da_",
    "RL + Skill Bank (Full)": "with_bank_avalon_da_",
}

for label, prefix in ablation_models.items():
    dirs = sorted(ablation_base.glob(f"{prefix}*"))
    summaries = []
    for d in dirs:
        found = list(d.rglob("rollout_summary.json"))
        if found:
            summaries.append(found[0])
    if summaries:
        MODEL_CONFIGS[label] = summaries

baselines_base = Path("/workspace/game_agent/Game-AI-Agent/baselines/output")
baseline_models = {
    "GPT-5.4": [baselines_base / "gpt54_avalon_single_20260323_185147/avalon/rollout_summary.json"],
    "Claude-4.6": [
        baselines_base / "claude46_avalon_20260326_063751/avalon/rollout_summary.json",
        baselines_base / "claude46_avalon_20260326_074558/avalon/rollout_summary.json",
    ],
    "Gemini-3.1-Pro": [baselines_base / "gemini31pro_avalon_20260326_093347/avalon/rollout_summary.json"],
    "GPT-o-ss-120B": [baselines_base / "gptoss120b_avalon_20260326_113706/avalon/rollout_summary.json"],
}

for label, paths in baseline_models.items():
    valid = [p for p in paths if p.exists()]
    if valid:
        MODEL_CONFIGS[label] = valid

inference_best = Path("/workspace/game_agent/Game-AI-Agent/output/infer_avalon_da_vs_gpt54_20260327_073838/avalon/20260327_073929/rollout_summary.json")
if inference_best.exists():
    MODEL_CONFIGS["Qwen3-8B DA (Best, vs GPT-5.4)"] = [inference_best]


def compute_win(ep):
    """Determine if the controlled player won (binary 1/0).

    For decision-agent runs (role_side present): player wins iff good_victory
    matches their side. For baseline single-model runs: use good_victory directly
    since the model controls a single player with known side.
    """
    gv = ep.get('good_victory')
    if gv is None:
        return None
    side = ep.get('role_side')
    if side is not None:
        is_good = (side == 'good')
        return 1 if (is_good == gv) else 0
    return 1 if gv else 0


def analyze_model(label, summary_paths):
    all_episodes = []
    for sp in summary_paths:
        eps, _ = load_episodes(str(sp))
        all_episodes.extend(eps)

    has_roles = len(all_episodes) > 0 and 'role_name' in all_episodes[0]

    by_role = defaultdict(list)
    by_side = defaultdict(list)
    all_wins = []

    for ep in all_episodes:
        w = compute_win(ep)
        if w is None:
            continue
        all_wins.append(w)

        if has_roles:
            role = ep.get('role_name', 'Unknown')
            side = ep.get('role_side', 'unknown')
            by_role[role].append(w)
            by_side[side].append(w)

    return all_wins, by_role, by_side, has_roles, len(all_episodes)


def fmt_wr(data, label=""):
    wr, lo, hi = win_rate_ci(data)
    n = len(data)
    wins = sum(data)
    if n == 0:
        return f"  {label:20s}  n={n:3d}  -- no data --"
    if np.isnan(lo):
        return f"  {label:20s}  n={n:3d}  WR={wr:.1%} ({wins}W)  CI=N/A"
    return f"  {label:20s}  n={n:3d}  WR={wr:.1%} ({wins}W/{n-wins}L)  95%CI=[{lo:.1%}, {hi:.1%}]"


print("=" * 95)
print("AVALON RESULTS: Win Rate & 95% CI (Wilson) by Character Role and Good/Evil Group")
print("  Win = controlled player's side won the game")
print("=" * 95)

ROLE_ORDER = ["Servant", "Merlin", "Assassin", "Minion"]
SIDE_ORDER = ["good", "evil"]

display_order = [
    "GPT-5.4",
    "Claude-4.6",
    "Gemini-3.1-Pro",
    "GPT-o-ss-120B",
    "Base Model (Qwen3-8B)",
    "SFT (no bank)",
    "SFT + First Bank",
    "SFT + Best Bank",
    "RL (no bank)",
    "RL + Skill Bank (Full)",
    "Qwen3-8B DA (Best, vs GPT-5.4)",
]

for model_label in display_order:
    if model_label not in MODEL_CONFIGS:
        continue
    paths = MODEL_CONFIGS[model_label]
    all_wins, by_role, by_side, has_roles, n_eps = analyze_model(model_label, paths)

    print(f"\n{'─' * 95}")
    print(f"  Model: {model_label}")
    print(f"  Total episodes: {n_eps}  |  Sources: {len(paths)} run(s)")
    print(f"{'─' * 95}")

    print(fmt_wr(all_wins, "OVERALL"))

    if has_roles:
        print()
        print("  --- By Character Role ---")
        for role in ROLE_ORDER:
            if role in by_role:
                print(fmt_wr(by_role[role], role))
        for role in sorted(by_role.keys()):
            if role not in ROLE_ORDER:
                print(fmt_wr(by_role[role], role))

        print()
        print("  --- By Side (Good / Evil) ---")
        for side in SIDE_ORDER:
            if side in by_side:
                print(fmt_wr(by_side[side], side.upper()))
    else:
        print("  (No per-character role data available for this model)")

print(f"\n{'=' * 95}")

# ── Compact cross-model tables ──────────────────────────────────────────

all_results = {}
for model_label in display_order:
    if model_label not in MODEL_CONFIGS:
        continue
    paths = MODEL_CONFIGS[model_label]
    all_wins, by_role, by_side, has_roles, n_eps = analyze_model(model_label, paths)
    all_results[model_label] = {
        "overall": all_wins,
        "by_role": by_role,
        "by_side": by_side,
        "has_roles": has_roles,
    }

def fmt_cell(data):
    """Format a compact WR cell: '42.0% [29,56] (n=50)'."""
    n = len(data)
    if n == 0:
        return "  --  "
    wr, lo, hi = win_rate_ci(data)
    if np.isnan(lo):
        return f"{wr:5.1%} (n={n})"
    return f"{wr:5.1%} [{lo:.0%},{hi:.0%}] (n={n})"

SHORT_NAMES = {
    "GPT-5.4": "GPT-5.4",
    "Claude-4.6": "Claude-4.6",
    "Gemini-3.1-Pro": "Gemini-3.1",
    "GPT-o-ss-120B": "o-ss-120B",
    "Base Model (Qwen3-8B)": "Base",
    "SFT (no bank)": "SFT",
    "SFT + First Bank": "SFT+1st",
    "SFT + Best Bank": "SFT+Best",
    "RL (no bank)": "RL",
    "RL + Skill Bank (Full)": "RL+Bank",
    "Qwen3-8B DA (Best, vs GPT-5.4)": "DA-Best",
}

col_w = 28
name_w = 12

print()
print("=" * 95)
print("CROSS-MODEL COMPARISON: Win Rate by Character Role")
print("=" * 95)

for role in ROLE_ORDER:
    print(f"\n  ── {role} {'(Good)' if role in ('Servant','Merlin') else '(Evil)'} ──")
    print(f"  {'Model':<{name_w}}  {'WR [95%CI] (n)':>{col_w}}")
    print(f"  {'─'*name_w}  {'─'*col_w}")
    for model_label in display_order:
        if model_label not in all_results:
            continue
        r = all_results[model_label]
        if not r["has_roles"]:
            continue
        data = r["by_role"].get(role, [])
        short = SHORT_NAMES.get(model_label, model_label[:name_w])
        print(f"  {short:<{name_w}}  {fmt_cell(data):>{col_w}}")

print()
print("=" * 95)
print("CROSS-MODEL COMPARISON: Win Rate by Side")
print("=" * 95)

for side_label, side_key in [("Good", "good"), ("Evil", "evil")]:
    print(f"\n  ── {side_label} Side ──")
    print(f"  {'Model':<{name_w}}  {'WR [95%CI] (n)':>{col_w}}")
    print(f"  {'─'*name_w}  {'─'*col_w}")
    for model_label in display_order:
        if model_label not in all_results:
            continue
        r = all_results[model_label]
        if not r["has_roles"]:
            continue
        data = r["by_side"].get(side_key, [])
        short = SHORT_NAMES.get(model_label, model_label[:name_w])
        print(f"  {short:<{name_w}}  {fmt_cell(data):>{col_w}}")

print()
print("=" * 95)
print("CROSS-MODEL COMPARISON: Overall Win Rate")
print("=" * 95)
print(f"\n  {'Model':<{name_w}}  {'WR [95%CI] (n)':>{col_w}}")
print(f"  {'─'*name_w}  {'─'*col_w}")
for model_label in display_order:
    if model_label not in all_results:
        continue
    data = all_results[model_label]["overall"]
    short = SHORT_NAMES.get(model_label, model_label[:name_w])
    print(f"  {short:<{name_w}}  {fmt_cell(data):>{col_w}}")

print(f"\n{'=' * 95}")
print("Done.")
