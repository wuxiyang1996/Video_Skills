#!/usr/bin/env python3
"""
P2 — Reward Decomposition by Ablation Condition

Addresses: "We don't show WHY it improves performance."

Analyses:
 A. Per-step reward distribution across ablation conditions
 B. Step efficiency (reward / steps)
 C. Reward trajectory shape (early vs late game reward accrual)
 D. Training reward curve from step_log.jsonl
"""

import json
import os
from collections import defaultdict

from utils import (
    ABLATION_BASE, RUNS,
    load_all_ablation_episodes, load_step_log, load_rewards_at_step,
    episode_total_reward, mean_std, print_header, print_subheader,
    get_checkpoint_steps,
)


def per_condition_reward_stats():
    """Compare reward distributions across ablation conditions per game."""
    print_header("SECTION A — Reward Statistics by Ablation Condition", 100)

    all_eps = load_all_ablation_episodes()

    games = sorted(set(g for _, g in all_eps.keys()))

    for game in games:
        print_subheader(f"Game: {game}")

        conditions = {}
        for (label, g), eps in all_eps.items():
            if g != game:
                continue
            rewards = [episode_total_reward(e) for e in eps]
            if rewards:
                conditions[label] = rewards

        if not conditions:
            print("  [No data]")
            continue

        print(f"  {'Condition':<25} | {'N':>4} | {'Mean':>8} | {'Std':>8} | {'Min':>8} | {'Max':>8} | {'Median':>8}")
        print(f"  {'-'*25}-+-{'-'*4}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")

        sorted_conds = sorted(conditions.items(), key=lambda kv: -sum(kv[1])/len(kv[1]))
        for label, rewards in sorted_conds:
            m, s = mean_std(rewards)
            rewards_sorted = sorted(rewards)
            median = rewards_sorted[len(rewards_sorted)//2]
            print(f"  {label:<25} | {len(rewards):>4} | {m:>8.2f} | {s:>8.2f} | {min(rewards):>8.2f} | {max(rewards):>8.2f} | {median:>8.2f}")

        # Delta from base model
        base_rewards = conditions.get("Base Model", [])
        if base_rewards:
            base_mean = sum(base_rewards) / len(base_rewards)
            print(f"\n  Improvement over Base Model (mean={base_mean:.2f}):")
            for label, rewards in sorted_conds:
                if label == "Base Model":
                    continue
                m = sum(rewards) / len(rewards)
                delta = m - base_mean
                pct = (delta / abs(base_mean) * 100) if base_mean != 0 else 0
                print(f"    {label:<25}  Δ = {delta:>+8.2f}  ({pct:>+6.1f}%)")


def step_efficiency():
    """Compare reward-per-step across conditions."""
    print_header("SECTION B — Step Efficiency (Reward / Steps)", 100)

    all_eps = load_all_ablation_episodes()
    games = sorted(set(g for _, g in all_eps.keys()))

    for game in games:
        print_subheader(f"Game: {game}")

        rows = []
        for (label, g), eps in all_eps.items():
            if g != game:
                continue
            efficiencies = []
            step_counts = []
            for ep in eps:
                exps = ep.get("experiences", [])
                n_steps = len(exps)
                total_r = episode_total_reward(ep)
                if n_steps > 0:
                    efficiencies.append(total_r / n_steps)
                    step_counts.append(n_steps)
            if efficiencies:
                me, se = mean_std(efficiencies)
                ms, ss = mean_std(step_counts)
                rows.append((label, len(efficiencies), me, se, ms, ss))

        if not rows:
            print("  [No data]")
            continue

        rows.sort(key=lambda r: -r[2])
        print(f"  {'Condition':<25} | {'N':>4} | {'Rew/Step':>8} | {'±':>6} | {'Avg Steps':>9} | {'±':>6}")
        print(f"  {'-'*25}-+-{'-'*4}-+-{'-'*8}-+-{'-'*6}-+-{'-'*9}-+-{'-'*6}")
        for label, n, me, se, ms, ss in rows:
            print(f"  {label:<25} | {n:>4} | {me:>8.3f} | {se:>6.3f} | {ms:>9.1f} | {ss:>6.1f}")


def reward_trajectory_shape():
    """Analyze where in the episode reward is accrued (early vs late)."""
    print_header("SECTION C — Reward Accrual Shape (Early vs Late)", 100)

    all_eps = load_all_ablation_episodes()
    games = sorted(set(g for _, g in all_eps.keys()))

    for game in games:
        rows = []
        for (label, g), eps in all_eps.items():
            if g != game:
                continue

            first_half_rewards = []
            second_half_rewards = []

            for ep in eps:
                exps = ep.get("experiences", [])
                n = len(exps)
                if n < 4:
                    continue
                mid = n // 2
                r1 = sum(e.get("reward", 0) for e in exps[:mid])
                r2 = sum(e.get("reward", 0) for e in exps[mid:])
                first_half_rewards.append(r1)
                second_half_rewards.append(r2)

            if first_half_rewards:
                m1, _ = mean_std(first_half_rewards)
                m2, _ = mean_std(second_half_rewards)
                ratio = m2 / m1 if m1 != 0 else float("inf")
                rows.append((label, len(first_half_rewards), m1, m2, ratio))

        if not rows:
            continue

        print_subheader(f"Game: {game}")
        rows.sort(key=lambda r: -r[4])
        print(f"  {'Condition':<25} | {'N':>4} | {'1st Half':>8} | {'2nd Half':>8} | {'Ratio':>6} | {'Shape':>12}")
        print(f"  {'-'*25}-+-{'-'*4}-+-{'-'*8}-+-{'-'*8}-+-{'-'*6}-+-{'-'*12}")
        for label, n, m1, m2, ratio in rows:
            shape = "accelerating" if ratio > 1.2 else "decelerating" if ratio < 0.8 else "steady"
            print(f"  {label:<25} | {n:>4} | {m1:>8.2f} | {m2:>8.2f} | {ratio:>6.2f} | {shape:>12}")


def training_reward_curves():
    """Plot training reward progression from step_log.jsonl."""
    print_header("SECTION D — Training Reward Curves (step_log.jsonl)", 100)

    for run_name, cfg in RUNS.items():
        base = cfg["base"]
        best_step = cfg["best_step"]
        step_log = load_step_log(base)
        if not step_log:
            continue

        print_subheader(f"{run_name} ({len(step_log)} steps, best={best_step})")

        print(f"  {'Step':>5} | {'Mode':<12} | {'Mean Rew':>9} | {'Max Rew':>8} | {'Min Rew':>8} | {'Std':>7} | {'#Ep':>4} | {'#Skills':>7} | {'New':>4} | {'Wall(s)':>8}")
        print(f"  {'-'*5}-+-{'-'*12}-+-{'-'*9}-+-{'-'*8}-+-{'-'*8}-+-{'-'*7}-+-{'-'*4}-+-{'-'*7}-+-{'-'*4}-+-{'-'*8}")

        for entry in step_log:
            step = entry["step"]
            mode = entry.get("mode", "?")
            mr = entry.get("mean_reward", 0)
            n_sk = entry.get("n_skills", 0)
            n_new = entry.get("n_new_skills", 0)
            n_ep = entry.get("n_episodes", 0)
            wt = entry.get("wall_time_s", 0)

            rpg = entry.get("reward_per_game", {})
            max_r, min_r, std_r = 0, 0, 0
            for gdata in rpg.values():
                max_r = max(max_r, gdata.get("max_reward", 0))
                min_r = min(min_r, gdata.get("min_reward", 0)) if min_r == 0 else min(min_r, gdata.get("min_reward", 0))
                std_r = gdata.get("std_reward", 0)

            marker = " ◀ BEST" if step == best_step else ""
            print(f"  {step:>5} | {mode:<12} | {mr:>9.2f} | {max_r:>8.1f} | {min_r:>8.1f} | {std_r:>7.2f} | {n_ep:>4} | {n_sk:>7} | {n_new:>4} | {wt:>8.1f}{marker}")

        # Compute improvement
        if len(step_log) >= 2:
            first_r = step_log[0].get("mean_reward", 0)
            best_entry = next((e for e in step_log if e["step"] == best_step), step_log[-1])
            best_r = best_entry.get("mean_reward", 0)
            improvement = best_r - first_r
            pct = (improvement / abs(first_r) * 100) if first_r != 0 else 0
            print(f"\n  Step 0 → Best: {first_r:.2f} → {best_r:.2f}  (Δ = {improvement:+.2f}, {pct:+.1f}%)")


def main():
    per_condition_reward_stats()
    step_efficiency()
    reward_trajectory_shape()
    training_reward_curves()


if __name__ == "__main__":
    main()
