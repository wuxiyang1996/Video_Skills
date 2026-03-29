#!/usr/bin/env python3
"""
P7 — Intention Tag / Strategy Distribution Analysis

Addresses: "No analysis of the strategies used by the models."

Analyses:
 A. Intention tag distribution per game (labeled data)
 B. Strategy diversity comparison across ablation conditions
 C. Strategy sequences — common N-gram patterns in intention tags
 D. Strategy shift over episode progress (early/mid/late)
 E. Avalon role-specific strategy analysis
 F. Diplomacy power-specific strategy analysis
"""

import json
import os
from collections import defaultdict, Counter

from utils import (
    LABELED_BASE, LABELED_GAMES, ABLATION_BASE,
    load_all_labeled_episodes, load_all_ablation_episodes,
    episode_total_reward, extract_intention_tag, parse_summary_state,
    mean_std, print_header, print_subheader, gini_coefficient,
)


def intention_distribution_labeled():
    """Intention tag distribution from labeled episodes per game."""
    print_header("SECTION A — Intention Tag Distribution (Labeled Data)", 100)

    for game in LABELED_GAMES:
        episodes = load_all_labeled_episodes(game)
        if not episodes:
            continue

        tag_counts = Counter()
        tag_rewards = defaultdict(list)
        n_steps = 0

        for ep in episodes:
            for exp in ep.get("experiences", []):
                tag = extract_intention_tag(exp.get("intentions", ""))
                if tag:
                    tag_counts[tag] += 1
                    tag_rewards[tag].append(exp.get("reward", 0))
                    n_steps += 1

        if not tag_counts:
            continue

        print_subheader(f"Game: {game} ({len(episodes)} episodes, {n_steps} tagged steps)")

        gini = gini_coefficient(list(tag_counts.values()))
        print(f"  Strategy diversity: {len(tag_counts)} unique tags, Gini={gini:.3f}")

        print(f"\n  {'Tag':<20} | {'Count':>6} | {'%':>6} | {'Mean Rew':>8} | {'Std':>7}")
        print(f"  {'-'*20}-+-{'-'*6}-+-{'-'*6}-+-{'-'*8}-+-{'-'*7}")
        for tag, count in tag_counts.most_common():
            pct = count / n_steps * 100
            mr, sr = mean_std(tag_rewards[tag])
            print(f"  {tag:<20} | {count:>6} | {pct:>5.1f}% | {mr:>8.3f} | {sr:>7.3f}")


def strategy_diversity_ablation():
    """Compare strategy diversity across ablation conditions."""
    print_header("SECTION B — Strategy Diversity Across Ablation Conditions", 100)

    all_eps = load_all_ablation_episodes()
    games = sorted(set(g for _, g in all_eps.keys()))

    for game in games:
        rows = []
        for (label, g), eps in all_eps.items():
            if g != game:
                continue

            all_tags = Counter()
            per_ep_diversity = []

            for ep in eps:
                ep_tags = set()
                for exp in ep.get("experiences", []):
                    tag = extract_intention_tag(exp.get("intentions", ""))
                    if tag:
                        all_tags[tag] += 1
                        ep_tags.add(tag)
                per_ep_diversity.append(len(ep_tags))

            if not all_tags:
                continue

            n_unique = len(all_tags)
            gini = gini_coefficient(list(all_tags.values()))
            md, sd = mean_std(per_ep_diversity)
            mr, sr = mean_std([episode_total_reward(e) for e in eps])

            rows.append((label, len(eps), n_unique, gini, md, sd, mr))

        if not rows:
            continue

        print_subheader(f"Game: {game}")
        rows.sort(key=lambda r: -r[6])
        print(f"  {'Condition':<25} | {'N':>4} | {'Uniq Tags':>9} | {'Gini':>6} | {'Div/Ep':>6} | {'Mean Rew':>8}")
        print(f"  {'-'*25}-+-{'-'*4}-+-{'-'*9}-+-{'-'*6}-+-{'-'*6}-+-{'-'*8}")
        for label, n, nu, gi, md, sd, mr in rows:
            print(f"  {label:<25} | {n:>4} | {nu:>9} | {gi:>6.3f} | {md:>6.1f} | {mr:>8.2f}")


def strategy_ngrams():
    """Common 2-gram and 3-gram sequences in intention tags."""
    print_header("SECTION C — Strategy Sequence Patterns (N-grams)", 100)

    for game in LABELED_GAMES:
        episodes = load_all_labeled_episodes(game)
        if not episodes:
            continue

        bigrams = Counter()
        trigrams = Counter()

        for ep in episodes:
            tags = []
            for exp in ep.get("experiences", []):
                tag = extract_intention_tag(exp.get("intentions", ""))
                if tag:
                    tags.append(tag)

            for i in range(len(tags) - 1):
                bigrams[(tags[i], tags[i+1])] += 1
            for i in range(len(tags) - 2):
                trigrams[(tags[i], tags[i+1], tags[i+2])] += 1

        if not bigrams:
            continue

        print_subheader(f"Game: {game}")

        print(f"  Top-10 bigrams:")
        for (a, b), count in bigrams.most_common(10):
            print(f"    {a:>15} → {b:<15}  count={count}")

        if trigrams:
            print(f"\n  Top-10 trigrams:")
            for (a, b, c), count in trigrams.most_common(10):
                print(f"    {a:>15} → {b:>15} → {c:<15}  count={count}")


def strategy_by_phase():
    """How intention tag distribution shifts over episode progress."""
    print_header("SECTION D — Strategy Shift Over Episode Progress", 100)

    for game in LABELED_GAMES:
        episodes = load_all_labeled_episodes(game)
        if not episodes:
            continue

        phase_tags = {"early": Counter(), "mid": Counter(), "late": Counter()}

        for ep in episodes:
            exps = ep.get("experiences", [])
            n = len(exps)
            if n < 3:
                continue

            for i, exp in enumerate(exps):
                tag = extract_intention_tag(exp.get("intentions", ""))
                if not tag:
                    continue
                progress = i / n
                if progress < 0.33:
                    phase_tags["early"][tag] += 1
                elif progress < 0.67:
                    phase_tags["mid"][tag] += 1
                else:
                    phase_tags["late"][tag] += 1

        all_tags = sorted(set(
            t for counts in phase_tags.values() for t in counts
        ))
        if not all_tags:
            continue

        print_subheader(f"Game: {game}")

        print(f"  {'Tag':<20} | {'Early %':>8} | {'Mid %':>8} | {'Late %':>8} | {'Shift':>10}")
        print(f"  {'-'*20}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*10}")

        total_e = sum(phase_tags["early"].values()) or 1
        total_m = sum(phase_tags["mid"].values()) or 1
        total_l = sum(phase_tags["late"].values()) or 1

        for tag in all_tags:
            pe = phase_tags["early"].get(tag, 0) / total_e * 100
            pm = phase_tags["mid"].get(tag, 0) / total_m * 100
            pl = phase_tags["late"].get(tag, 0) / total_l * 100
            if pe > pl + 5:
                shift = "early→"
            elif pl > pe + 5:
                shift = "→late"
            else:
                shift = "steady"
            print(f"  {tag:<20} | {pe:>7.1f}% | {pm:>7.1f}% | {pl:>7.1f}% | {shift:>10}")


def avalon_role_strategy():
    """Avalon role-specific strategy: intention distribution per character role."""
    print_header("SECTION E — Avalon Role-Specific Strategy", 100)

    episodes = load_all_labeled_episodes("avalon")
    if not episodes:
        print("  [No labeled Avalon episodes]")
        return

    role_tags = defaultdict(Counter)
    role_rewards = defaultdict(list)

    for ep in episodes:
        meta = ep.get("metadata", {})
        if not isinstance(meta, dict):
            continue
        role = meta.get("role", meta.get("character", "unknown"))
        side = meta.get("side", "unknown")
        key = f"{role} ({side})" if side != "unknown" else role

        role_rewards[key].append(episode_total_reward(ep))

        for exp in ep.get("experiences", []):
            tag = extract_intention_tag(exp.get("intentions", ""))
            if tag:
                role_tags[key][tag] += 1

    if not role_tags:
        # Try extracting role from summary_state
        for ep in episodes:
            for exp in ep.get("experiences", []):
                ss = parse_summary_state(exp.get("summary_state", ""))
                role = ss.get("role", ss.get("character", None))
                if role:
                    tag = extract_intention_tag(exp.get("intentions", ""))
                    if tag:
                        role_tags[role][tag] += 1
            if role_tags:
                break

    for role in sorted(role_tags.keys()):
        tags = role_tags[role]
        total = sum(tags.values())
        rewards = role_rewards.get(role, [])
        mr = sum(rewards) / len(rewards) if rewards else 0

        print(f"\n  {role} ({len(rewards)} episodes, mean_reward={mr:.2f}):")
        for tag, count in tags.most_common(8):
            pct = count / total * 100
            print(f"    {tag:<20}  {count:>4}  ({pct:>5.1f}%)")


def diplomacy_power_strategy():
    """Diplomacy power-specific strategy from labeled or ablation data."""
    print_header("SECTION F — Diplomacy Power-Specific Strategy", 100)

    all_eps = load_all_ablation_episodes()

    diplomacy_eps = []
    for (label, game), eps in all_eps.items():
        if game == "diplomacy":
            for ep in eps:
                ep["_condition_label"] = label
                diplomacy_eps.append(ep)

    if not diplomacy_eps:
        diplomacy_eps = load_all_labeled_episodes("diplomacy")

    if not diplomacy_eps:
        print("  [No diplomacy episodes found]")
        return

    power_tags = defaultdict(Counter)
    power_rewards = defaultdict(list)

    for ep in diplomacy_eps:
        exps = ep.get("experiences", [])
        power = None

        for exp in exps:
            ss = parse_summary_state(exp.get("summary_state", ""))
            if "self" in ss:
                self_str = ss["self"]
                if "power:" in self_str:
                    power = self_str.split("power:")[1].split()[0]
                    break

        if not power:
            meta = ep.get("metadata", {})
            if isinstance(meta, dict):
                power = meta.get("controlled_power", "unknown")

        power = power or "unknown"
        power_rewards[power].append(episode_total_reward(ep))

        for exp in exps:
            tag = extract_intention_tag(exp.get("intentions", ""))
            if tag:
                power_tags[power][tag] += 1

    for power in sorted(power_tags.keys()):
        tags = power_tags[power]
        total = sum(tags.values())
        rewards = power_rewards.get(power, [])
        mr, sr = mean_std(rewards)

        print(f"\n  {power} ({len(rewards)} episodes, mean_reward={mr:.2f}±{sr:.2f}):")
        for tag, count in tags.most_common(8):
            pct = count / total * 100
            print(f"    {tag:<20}  {count:>4}  ({pct:>5.1f}%)")


def main():
    intention_distribution_labeled()
    strategy_diversity_ablation()
    strategy_ngrams()
    strategy_by_phase()
    avalon_role_strategy()
    diplomacy_power_strategy()


if __name__ == "__main__":
    main()
