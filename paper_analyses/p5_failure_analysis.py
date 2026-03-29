#!/usr/bin/env python3
"""
P5 + P8 — Failure Analysis

Addresses: "No failure analysis" and "Contract failure signatures."

Analyses:
 A. Low-reward episode autopsy (bottom-10% vs top-10%)
 B. Episode termination analysis (goal achieved, game over, timeout)
 C. Repeated-state / stuck detection
 D. Skill contract failure signatures aggregation
 E. Cross-condition failure rate comparison
"""

import json
import os
from collections import defaultdict, Counter

from utils import (
    LABELED_BASE, LABELED_GAMES, RUNS,
    load_all_labeled_episodes, load_all_ablation_episodes,
    episode_total_reward, extract_intention_tag, parse_summary_state,
    mean_std, print_header, print_subheader,
    parse_skill_bank, load_final_bank, load_checkpoint_bank,
    get_checkpoint_steps,
)


def low_reward_autopsy():
    """Compare bottom-10% vs top-10% episodes across labeled data."""
    print_header("SECTION A — Low vs High Reward Episode Comparison", 100)

    for game in LABELED_GAMES:
        episodes = load_all_labeled_episodes(game)
        if len(episodes) < 10:
            continue

        episodes.sort(key=episode_total_reward)
        n = len(episodes)
        bottom_k = max(1, n // 10)
        top_k = max(1, n // 10)

        bottom = episodes[:bottom_k]
        top = episodes[-top_k:]

        print_subheader(f"Game: {game} ({n} episodes, comparing bottom-{bottom_k} vs top-{top_k})")

        for label, group in [("BOTTOM 10%", bottom), ("TOP 10%", top)]:
            rewards = [episode_total_reward(e) for e in group]
            steps = [len(e.get("experiences", [])) for e in group]
            mr, sr = mean_std(rewards)
            ms, ss = mean_std(steps)

            # Skill diversity
            skill_sets = []
            for ep in group:
                sids = set()
                for exp in ep.get("experiences", []):
                    sk = exp.get("skills")
                    if sk:
                        sids.add(sk.get("skill_id", "?"))
                skill_sets.append(len(sids))
            msk, ssk = mean_std(skill_sets)

            # Intention diversity
            intention_sets = []
            for ep in group:
                tags = set()
                for exp in ep.get("experiences", []):
                    tag = extract_intention_tag(exp.get("intentions", ""))
                    if tag:
                        tags.add(tag)
                intention_sets.append(len(tags))
            mit, sit = mean_std(intention_sets)

            # Confidence
            confs = []
            for ep in group:
                for exp in ep.get("experiences", []):
                    sk = exp.get("skills")
                    if sk and sk.get("confidence") is not None:
                        confs.append(sk["confidence"])
            mc, sc = mean_std(confs)

            # Switch rate
            switch_rates = []
            for ep in group:
                exps = ep.get("experiences", [])
                prev = None
                switches, total = 0, 0
                for exp in exps:
                    sk = exp.get("skills")
                    if not sk:
                        continue
                    sid = sk.get("skill_id", "?")
                    if prev is not None:
                        total += 1
                        if sid != prev:
                            switches += 1
                    prev = sid
                if total > 0:
                    switch_rates.append(switches / total)
            msw, ssw = mean_std(switch_rates)

            print(f"\n  {label}:")
            print(f"    Reward:           {mr:>8.2f} ± {sr:.2f}")
            print(f"    Steps:            {ms:>8.1f} ± {ss:.1f}")
            print(f"    Unique skills:    {msk:>8.1f} ± {ssk:.1f}")
            print(f"    Unique intentions:{mit:>8.1f} ± {sit:.1f}")
            print(f"    Confidence:       {mc:>8.4f} ± {sc:.4f}")
            print(f"    Switch rate:      {msw:>8.3f} ± {ssw:.3f}")

        # Intention tag comparison
        bottom_tags = Counter()
        top_tags = Counter()
        for ep in bottom:
            for exp in ep.get("experiences", []):
                tag = extract_intention_tag(exp.get("intentions", ""))
                if tag:
                    bottom_tags[tag] += 1
        for ep in top:
            for exp in ep.get("experiences", []):
                tag = extract_intention_tag(exp.get("intentions", ""))
                if tag:
                    top_tags[tag] += 1

        all_tags = sorted(set(list(bottom_tags.keys()) + list(top_tags.keys())))
        if all_tags:
            total_b = sum(bottom_tags.values()) or 1
            total_t = sum(top_tags.values()) or 1
            print(f"\n  Intention tag distribution (% of steps):")
            print(f"  {'Tag':<15} | {'Bottom':>8} | {'Top':>8} | {'Δ':>8}")
            print(f"  {'-'*15}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")
            for tag in all_tags:
                bp = bottom_tags.get(tag, 0) / total_b * 100
                tp = top_tags.get(tag, 0) / total_t * 100
                delta = tp - bp
                print(f"  {tag:<15} | {bp:>7.1f}% | {tp:>7.1f}% | {delta:>+7.1f}%")


def termination_analysis():
    """Classify how episodes end: success, failure, timeout."""
    print_header("SECTION B — Episode Termination Analysis", 100)

    for game in LABELED_GAMES:
        episodes = load_all_labeled_episodes(game)
        if not episodes:
            continue

        termination_counts = Counter()
        termination_rewards = defaultdict(list)

        for ep in episodes:
            meta = ep.get("metadata", {})
            if not isinstance(meta, dict):
                meta = {}
            reason = meta.get("termination_reason", "unknown")
            terminated = meta.get("terminated", False)
            truncated = meta.get("truncated", False)

            if reason == "unknown":
                if truncated:
                    reason = "max_steps"
                elif terminated:
                    reason = "terminated"

            termination_counts[reason] += 1
            termination_rewards[reason].append(episode_total_reward(ep))

        print_subheader(f"Game: {game} ({len(episodes)} episodes)")
        total = sum(termination_counts.values())
        print(f"  {'Reason':<20} | {'Count':>6} | {'%':>6} | {'Mean Rew':>9} | {'Std':>7}")
        print(f"  {'-'*20}-+-{'-'*6}-+-{'-'*6}-+-{'-'*9}-+-{'-'*7}")
        for reason, count in termination_counts.most_common():
            pct = count / total * 100
            mr, sr = mean_std(termination_rewards[reason])
            print(f"  {reason:<20} | {count:>6} | {pct:>5.1f}% | {mr:>9.2f} | {sr:>7.2f}")


def stuck_detection():
    """Detect episodes where the agent gets stuck (repeated states or zero reward runs)."""
    print_header("SECTION C — Stuck / Loop Detection", 100)

    for game in LABELED_GAMES:
        episodes = load_all_labeled_episodes(game)
        if not episodes:
            continue

        stuck_episodes = []
        for ep in episodes:
            exps = ep.get("experiences", [])
            if len(exps) < 5:
                continue

            # Detect zero-reward streaks
            max_zero_streak = 0
            current_streak = 0
            for exp in exps:
                if exp.get("reward", 0) == 0:
                    current_streak += 1
                    max_zero_streak = max(max_zero_streak, current_streak)
                else:
                    current_streak = 0

            # Detect repeated actions
            actions = [exp.get("action", "") for exp in exps]
            action_counts = Counter(str(a) for a in actions)
            most_common_action, most_common_count = action_counts.most_common(1)[0]
            action_repeat_ratio = most_common_count / len(actions)

            # Detect repeated skills
            skill_ids = [
                (exp.get("skills") or {}).get("skill_id", "?")
                for exp in exps if exp.get("skills")
            ]
            if skill_ids:
                skill_counts = Counter(skill_ids)
                _, most_common_skill_count = skill_counts.most_common(1)[0]
                skill_repeat_ratio = most_common_skill_count / len(skill_ids)
            else:
                skill_repeat_ratio = 0

            is_stuck = (
                max_zero_streak >= len(exps) * 0.5
                or action_repeat_ratio > 0.7
            )

            if is_stuck:
                stuck_episodes.append({
                    "reward": episode_total_reward(ep),
                    "steps": len(exps),
                    "zero_streak": max_zero_streak,
                    "action_repeat": action_repeat_ratio,
                    "skill_repeat": skill_repeat_ratio,
                    "top_action": most_common_action[:60],
                })

        print_subheader(f"Game: {game} — {len(stuck_episodes)}/{len(episodes)} stuck episodes")

        if stuck_episodes:
            stuck_episodes.sort(key=lambda x: x["reward"])
            print(f"  {'Reward':>8} | {'Steps':>5} | {'0-streak':>8} | {'Act-rep':>7} | {'Sk-rep':>6} | Top Action")
            print(f"  {'-'*8}-+-{'-'*5}-+-{'-'*8}-+-{'-'*7}-+-{'-'*6}-+-{'-'*40}")
            for se in stuck_episodes[:10]:
                print(f"  {se['reward']:>8.2f} | {se['steps']:>5} | {se['zero_streak']:>8} | {se['action_repeat']:>7.2f} | {se['skill_repeat']:>6.2f} | {se['top_action']}")


def contract_failure_signatures():
    """Aggregate failure_signatures from skill report objects across all banks."""
    print_header("SECTION D — Contract Failure Signatures", 100)

    all_failures = defaultdict(lambda: defaultdict(int))
    all_pass_rates = defaultdict(list)

    for run_name, cfg in RUNS.items():
        base = cfg["base"]
        game_dirs = cfg["game_dirs"]
        best_step = cfg["best_step"]

        skills = load_final_bank(base, game_dirs)
        if not skills:
            ckpts = get_checkpoint_steps(base)
            if ckpts:
                step = min(ckpts, key=lambda s: abs(s - best_step))
                skills = load_checkpoint_bank(base, step, game_dirs)

        for sk in skills:
            report = sk.get("_report", {}) or {}
            sid = sk.get("skill_id", "?")

            pass_rate = report.get("overall_pass_rate")
            if pass_rate is not None:
                all_pass_rates[run_name].append((sid, pass_rate))

            failure_sigs = report.get("failure_signatures", {})
            for sig, count in failure_sigs.items():
                all_failures[run_name][(sid, sig)] += count

    # Overall pass rates
    print_subheader("Skill contract pass rates by game")
    for run_name in RUNS:
        rates = all_pass_rates.get(run_name, [])
        if not rates:
            continue
        values = [r for _, r in rates]
        m, s = mean_std(values)
        n_fail = sum(1 for _, r in rates if r < 0.5)
        print(f"\n  {run_name}:")
        print(f"    Mean pass rate: {m:.3f} ± {s:.3f}  |  Skills with <50% pass: {n_fail}/{len(rates)}")
        for sid, pr in sorted(rates, key=lambda x: x[1]):
            status = "FAIL" if pr < 0.5 else "WARN" if pr < 0.8 else "OK"
            print(f"      {sid:<35}  pass_rate={pr:.3f}  [{status}]")

    # Failure signature taxonomy
    print_subheader("Most common failure signatures (across all games)")

    all_sigs = Counter()
    for run_name, sigs in all_failures.items():
        for (sid, sig), count in sigs.items():
            sig_type = sig.split(":")[0] if ":" in sig else sig[:30]
            all_sigs[sig_type] += count

    if all_sigs:
        for sig_type, count in all_sigs.most_common(20):
            print(f"    {sig_type:<50}  count={count}")
    else:
        print("  [No failure signatures recorded in current bank snapshots]")


def cross_condition_failure_rates():
    """Compare failure indicators across ablation conditions."""
    print_header("SECTION E — Cross-Condition Failure Rates", 100)

    all_eps = load_all_ablation_episodes()
    games = sorted(set(g for _, g in all_eps.keys()))

    for game in games:
        rows = []
        for (label, g), eps in all_eps.items():
            if g != game:
                continue

            n = len(eps)
            rewards = [episode_total_reward(e) for e in eps]
            m, s = mean_std(rewards)

            # Zero-reward episodes
            n_zero = sum(1 for r in rewards if r <= 0)
            # Negative reward episodes
            n_neg = sum(1 for r in rewards if r < 0)
            # Episodes with very short runs (< 5 steps)
            n_short = sum(1 for e in eps if len(e.get("experiences", [])) < 5)

            rows.append((label, n, m, s, n_zero, n_neg, n_short))

        if not rows:
            continue

        print_subheader(f"Game: {game}")
        rows.sort(key=lambda r: -r[2])
        print(f"  {'Condition':<25} | {'N':>4} | {'Mean Rew':>8} | {'Zero-R':>6} | {'Neg-R':>5} | {'Short':>5}")
        print(f"  {'-'*25}-+-{'-'*4}-+-{'-'*8}-+-{'-'*6}-+-{'-'*5}-+-{'-'*5}")
        for label, n, m, s, nz, nn, ns in rows:
            print(f"  {label:<25} | {n:>4} | {m:>8.2f} | {nz:>6} | {nn:>5} | {ns:>5}")


def main():
    low_reward_autopsy()
    termination_analysis()
    stuck_detection()
    contract_failure_signatures()
    cross_condition_failure_rates()


if __name__ == "__main__":
    main()
