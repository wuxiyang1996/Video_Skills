#!/usr/bin/env python3
"""
P4 + P9 — Skill-State Association & Retrieval Confidence Analysis

Addresses:
 - "Skill retrieval could be stress tested more"
 - "In what states do models retrieve different skills"

Analyses:
 A. Skill-state association: which skills are retrieved in which game states
 B. Confidence distribution: histogram of retrieval confidence scores
 C. Candidate override analysis: how often the agent picks #2 or #3 over #1
 D. Skill switching frequency and its correlation with episode reward
 E. Phase-correctness: do phase-prefixed skills get retrieved in the right phase?
"""

import json
import os
from collections import defaultdict, Counter

from utils import (
    LABELED_BASE, LABELED_GAMES,
    load_all_labeled_episodes, episode_total_reward,
    parse_skill_id, parse_summary_state, extract_intention_tag,
    mean_std, print_header, print_subheader,
)


def skill_state_association():
    """Cross-tabulate chosen skills × state features."""
    print_header("SECTION A — Skill × State Association", 100)

    for game in LABELED_GAMES:
        episodes = load_all_labeled_episodes(game)
        if not episodes:
            continue

        skill_by_phase = defaultdict(Counter)
        skill_by_intention = defaultdict(Counter)
        skill_state_features = defaultdict(lambda: defaultdict(list))

        n_steps = 0
        for ep in episodes:
            for exp in ep.get("experiences", []):
                skills = exp.get("skills")
                if not skills:
                    continue
                sid = skills.get("skill_id", "?")
                n_steps += 1

                ss = parse_summary_state(exp.get("summary_state", ""))
                phase = ss.get("phase", "?")
                skill_by_phase[sid][phase] += 1

                tag = extract_intention_tag(exp.get("intentions", ""))
                if tag:
                    skill_by_intention[sid][tag] += 1

                step_idx = exp.get("idx", 0)
                reward = exp.get("reward", 0)
                skill_state_features[sid]["step_indices"].append(step_idx)
                skill_state_features[sid]["rewards"].append(reward)

        if not skill_by_phase:
            continue

        print_subheader(f"Game: {game} ({n_steps} steps with skills)")

        # Skill × Phase
        all_skills = sorted(skill_by_phase.keys())
        all_phases = sorted(set(p for counts in skill_by_phase.values() for p in counts))

        if len(all_phases) <= 15:
            print(f"\n  Skill × Phase:")
            print(f"  {'Skill ID':<30}", end="")
            for phase in all_phases:
                pname = phase[:10]
                print(f" | {pname:>10}", end="")
            print(f" | {'TOTAL':>6}")
            print(f"  {'-'*30}" + "".join(f"-+-{'-'*10}" for _ in all_phases) + f"-+-{'-'*6}")

            for sid in all_skills:
                row = f"  {sid:<30}"
                total = 0
                for phase in all_phases:
                    n = skill_by_phase[sid].get(phase, 0)
                    total += n
                    row += f" | {n:>10}" if n > 0 else f" | {'·':>10}"
                row += f" | {total:>6}"
                print(row)
        else:
            print(f"\n  [Too many phases ({len(all_phases)}) for tabular display; showing top-3 per skill]")
            for sid in all_skills:
                top = skill_by_phase[sid].most_common(3)
                phases_str = ", ".join(f"{p}={c}" for p, c in top)
                print(f"    {sid:<30}  {phases_str}")

        # Skill × Intention tag
        print(f"\n  Skill × Intention tag (top-3 per skill):")
        for sid in all_skills:
            top = skill_by_intention[sid].most_common(3)
            tag_str = ", ".join(f"{t}={c}" for t, c in top)
            total = sum(skill_by_intention[sid].values())
            print(f"    {sid:<30}  ({total:>4} total)  {tag_str}")

        # Skill temporal distribution
        print(f"\n  Skill temporal distribution (mean step index ± std):")
        for sid in all_skills:
            steps = skill_state_features[sid]["step_indices"]
            rewards = skill_state_features[sid]["rewards"]
            ms, ss = mean_std(steps)
            mr, sr = mean_std(rewards)
            print(f"    {sid:<30}  step={ms:>5.1f}±{ss:<5.1f}  reward={mr:>6.3f}±{sr:<6.3f}  n={len(steps)}")


def confidence_distribution():
    """Analyze the distribution of skill retrieval confidence scores."""
    print_header("SECTION B — Retrieval Confidence Distribution", 100)

    for game in LABELED_GAMES:
        episodes = load_all_labeled_episodes(game)
        if not episodes:
            continue

        confidences = []
        conf_by_skill = defaultdict(list)

        for ep in episodes:
            for exp in ep.get("experiences", []):
                skills = exp.get("skills")
                if not skills:
                    continue
                conf = skills.get("confidence")
                if conf is not None:
                    confidences.append(conf)
                    conf_by_skill[skills.get("skill_id", "?")].append(conf)

        if not confidences:
            continue

        print_subheader(f"Game: {game} ({len(confidences)} confidence scores)")

        m, s = mean_std(confidences)
        sorted_c = sorted(confidences)
        p10 = sorted_c[int(len(sorted_c) * 0.1)]
        p25 = sorted_c[int(len(sorted_c) * 0.25)]
        p50 = sorted_c[int(len(sorted_c) * 0.5)]
        p75 = sorted_c[int(len(sorted_c) * 0.75)]
        p90 = sorted_c[int(len(sorted_c) * 0.9)]

        print(f"  mean={m:.4f}  std={s:.4f}  min={min(confidences):.4f}  max={max(confidences):.4f}")
        print(f"  P10={p10:.4f}  P25={p25:.4f}  P50={p50:.4f}  P75={p75:.4f}  P90={p90:.4f}")

        # Histogram buckets
        buckets = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        hist = Counter()
        for c in confidences:
            for i in range(len(buckets) - 1):
                if buckets[i] <= c < buckets[i+1]:
                    hist[f"[{buckets[i]:.1f},{buckets[i+1]:.1f})"] += 1
                    break
            else:
                hist[f"[{buckets[-2]:.1f},{buckets[-1]:.1f}]"] += 1

        print(f"\n  Histogram:")
        for bucket in sorted(hist.keys()):
            n = hist[bucket]
            bar = "█" * (n * 40 // max(hist.values())) if hist.values() else ""
            print(f"    {bucket:>12}  {n:>5} ({n/len(confidences)*100:>5.1f}%)  {bar}")

        # Per-skill mean confidence
        print(f"\n  Mean confidence per skill:")
        for sid in sorted(conf_by_skill.keys()):
            vals = conf_by_skill[sid]
            m, s = mean_std(vals)
            print(f"    {sid:<30}  mean={m:.4f}  std={s:.4f}  n={len(vals)}")


def candidate_override_analysis():
    """How often does the agent override the top retrieval candidate?"""
    print_header("SECTION C — Candidate Override Analysis", 100)

    for game in LABELED_GAMES:
        episodes = load_all_labeled_episodes(game)
        if not episodes:
            continue

        chosen_idx_counts = Counter()
        n_candidates_total = Counter()
        override_rewards = defaultdict(list)
        total_steps = 0

        for ep in episodes:
            for exp in ep.get("experiences", []):
                candidates = exp.get("skill_candidates")
                chosen_idx = exp.get("skill_chosen_idx")
                if candidates is None or chosen_idx is None:
                    continue

                total_steps += 1
                chosen_idx_counts[chosen_idx] += 1
                n_candidates_total[len(candidates)] += 1
                override_rewards[chosen_idx].append(exp.get("reward", 0))

        if total_steps == 0:
            continue

        print_subheader(f"Game: {game} ({total_steps} steps)")

        print(f"  Chosen candidate position:")
        for idx in sorted(chosen_idx_counts.keys()):
            n = chosen_idx_counts[idx]
            pct = n / total_steps * 100
            mr, sr = mean_std(override_rewards[idx])
            bar = "█" * int(pct / 2)
            print(f"    #{idx}:  {n:>5} ({pct:>5.1f}%)  mean_reward={mr:>7.3f}  {bar}")

        top1_pct = chosen_idx_counts.get(0, 0) / total_steps * 100
        override_pct = 100 - top1_pct
        print(f"\n  Override rate: {override_pct:.1f}% (agent chose #2+ over #1)")

        mr_top1, _ = mean_std(override_rewards.get(0, []))
        mr_override = []
        for idx, rewards in override_rewards.items():
            if idx > 0:
                mr_override.extend(rewards)
        mr_ov, _ = mean_std(mr_override) if mr_override else (0, 0)
        print(f"  Mean reward when choosing #1: {mr_top1:.3f}")
        print(f"  Mean reward when overriding:  {mr_ov:.3f}")


def skill_switching_frequency():
    """Track how often the agent switches skills between consecutive steps."""
    print_header("SECTION D — Skill Switching Frequency", 100)

    for game in LABELED_GAMES:
        episodes = load_all_labeled_episodes(game)
        if not episodes:
            continue

        switch_rates = []
        switch_reward_corr = []

        for ep in episodes:
            exps = ep.get("experiences", [])
            prev_skill = None
            switches = 0
            total = 0
            for exp in exps:
                skills = exp.get("skills")
                if not skills:
                    continue
                sid = skills.get("skill_id", "?")
                if prev_skill is not None:
                    total += 1
                    if sid != prev_skill:
                        switches += 1
                prev_skill = sid

            if total > 0:
                rate = switches / total
                switch_rates.append(rate)
                switch_reward_corr.append((rate, episode_total_reward(ep)))

        if not switch_rates:
            continue

        print_subheader(f"Game: {game} ({len(switch_rates)} episodes)")

        m, s = mean_std(switch_rates)
        print(f"  Mean switch rate: {m:.3f} ± {s:.3f}")

        # Correlation between switch rate and reward
        if len(switch_reward_corr) >= 3:
            rates = [r for r, _ in switch_reward_corr]
            rewards = [rw for _, rw in switch_reward_corr]
            n = len(rates)
            mr, _ = mean_std(rates)
            mrw, _ = mean_std(rewards)
            cov = sum((r - mr) * (rw - mrw) for r, rw in zip(rates, rewards)) / (n - 1) if n > 1 else 0
            sr = mean_std(rates)[1]
            srw = mean_std(rewards)[1]
            corr = cov / (sr * srw) if sr > 0 and srw > 0 else 0
            print(f"  Correlation(switch_rate, episode_reward): r = {corr:.3f}")

        # Bucket by switch rate
        low = [rw for r, rw in switch_reward_corr if r < 0.3]
        mid = [rw for r, rw in switch_reward_corr if 0.3 <= r < 0.6]
        high = [rw for r, rw in switch_reward_corr if r >= 0.6]
        for label, bucket in [("Low (<0.3)", low), ("Mid (0.3-0.6)", mid), ("High (≥0.6)", high)]:
            if bucket:
                mb, sb = mean_std(bucket)
                print(f"    {label:<15}  n={len(bucket):>3}  mean_reward={mb:>8.2f} ± {sb:.2f}")


def phase_correctness():
    """Check if phase-prefixed skills are actually retrieved in the correct game phase."""
    print_header("SECTION E — Phase Correctness (Phase-Prefixed Skills)", 100)

    for game in LABELED_GAMES:
        episodes = load_all_labeled_episodes(game)
        if not episodes:
            continue

        phase_skills = defaultdict(lambda: Counter())
        total_phased = 0
        total_unphased = 0

        for ep in episodes:
            exps = ep.get("experiences", [])
            n_steps = len(exps)
            for exp in exps:
                skills = exp.get("skills")
                if not skills:
                    continue
                sid = skills.get("skill_id", "?")
                prefix, intent = parse_skill_id(sid)
                if not prefix:
                    total_unphased += 1
                    continue

                total_phased += 1
                step_idx = exp.get("idx", 0)
                if n_steps > 0:
                    progress = step_idx / n_steps
                    if progress < 0.33:
                        actual_phase = "early"
                    elif progress < 0.67:
                        actual_phase = "mid"
                    else:
                        actual_phase = "late"
                    phase_skills[prefix][actual_phase] += 1

        if not phase_skills:
            continue

        print_subheader(f"Game: {game} ({total_phased} phased, {total_unphased} unphased)")

        print(f"  {'Prefix':<15} | {'early':>8} | {'mid':>8} | {'late':>8} | {'Total':>6} | {'Match?':>8}")
        print(f"  {'-'*15}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*6}-+-{'-'*8}")

        for prefix in sorted(phase_skills.keys()):
            counts = phase_skills[prefix]
            total = sum(counts.values())
            e = counts.get("early", 0)
            m = counts.get("mid", 0)
            l = counts.get("late", 0)

            match_pct = 0
            if prefix in ("early", "opening", "start"):
                match_pct = e / total * 100 if total else 0
            elif prefix in ("mid", "midgame", "middle"):
                match_pct = m / total * 100 if total else 0
            elif prefix in ("late", "endgame", "end"):
                match_pct = l / total * 100 if total else 0
            else:
                match_pct = -1  # unknown mapping

            match_str = f"{match_pct:.0f}%" if match_pct >= 0 else "N/A"
            print(f"  {prefix:<15} | {e:>8} | {m:>8} | {l:>8} | {total:>6} | {match_str:>8}")


def main():
    skill_state_association()
    confidence_distribution()
    candidate_override_analysis()
    skill_switching_frequency()
    phase_correctness()


if __name__ == "__main__":
    main()
