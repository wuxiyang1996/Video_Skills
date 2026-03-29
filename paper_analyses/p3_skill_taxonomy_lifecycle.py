#!/usr/bin/env python3
"""
P3 + P6 — Skill Taxonomy & Lifecycle Analysis

Addresses: "No training dynamics / skill taxonomy / change over time."

Analyses:
 A. Cross-game skill taxonomy: strategic_intent × game_phase matrix
 B. Skill lifecycle tracking across checkpoint steps (birth, persistence, retirement)
 C. Skill churn rate per training step
 D. Contract refinement over training (version, clause counts, n_instances)
 E. Jaccard similarity of skill sets across consecutive checkpoints
"""

import json
import os
from collections import defaultdict, Counter

from utils import (
    RUNS, print_header, print_subheader, mean_std, gini_coefficient,
    parse_skill_id, parse_skill_bank, load_checkpoint_bank, load_final_bank,
    get_checkpoint_steps, load_step_log,
)


def build_taxonomy():
    """Build a strategic_intent × game_phase × game taxonomy from all available skill banks."""
    print_header("SECTION A — Skill Taxonomy (Intent × Phase × Game)", 100)

    taxonomy = defaultdict(lambda: defaultdict(list))
    game_intents = defaultdict(lambda: Counter())
    game_phases = defaultdict(lambda: Counter())

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
            sid = sk.get("skill_id", "unknown")
            phase, intent = parse_skill_id(sid)
            phase = phase or "unphased"
            taxonomy[(intent, phase)][run_name].append(sid)
            game_intents[run_name][intent] += 1
            game_phases[run_name][phase] += 1

    all_intents = sorted(set(i for i, _ in taxonomy.keys()))
    all_phases = sorted(set(p for _, p in taxonomy.keys()))
    all_games = list(RUNS.keys())

    # Intent × Phase heatmap
    print_subheader("Intent × Phase count matrix (all games)")
    header = f"  {'Intent':<15}"
    for phase in all_phases:
        header += f" | {phase:>10}"
    header += f" | {'TOTAL':>6}"
    print(header)
    print(f"  {'-'*15}" + "".join(f"-+-{'-'*10}" for _ in all_phases) + f"-+-{'-'*6}")

    for intent in all_intents:
        row = f"  {intent:<15}"
        total = 0
        for phase in all_phases:
            n = sum(len(v) for v in taxonomy[(intent, phase)].values())
            total += n
            row += f" | {n:>10}" if n > 0 else f" | {'·':>10}"
        row += f" | {total:>6}"
        print(row)

    # Per-game intent distribution
    print_subheader("Intent distribution per game")
    print(f"  {'Game':<15}", end="")
    for intent in all_intents:
        print(f" | {intent:>10}", end="")
    print(f" | {'TOTAL':>6}")
    print(f"  {'-'*15}" + "".join(f"-+-{'-'*10}" for _ in all_intents) + f"-+-{'-'*6}")

    for game in all_games:
        counts = game_intents[game]
        total = sum(counts.values())
        row = f"  {game:<15}"
        for intent in all_intents:
            n = counts.get(intent, 0)
            row += f" | {n:>10}" if n > 0 else f" | {'·':>10}"
        row += f" | {total:>6}"
        print(row)

    # Per-game phase distribution
    print_subheader("Phase distribution per game")
    print(f"  {'Game':<15}", end="")
    for phase in all_phases:
        print(f" | {phase:>10}", end="")
    print(f" | {'TOTAL':>6}")
    print(f"  {'-'*15}" + "".join(f"-+-{'-'*10}" for _ in all_phases) + f"-+-{'-'*6}")

    for game in all_games:
        counts = game_phases[game]
        total = sum(counts.values())
        row = f"  {game:<15}"
        for phase in all_phases:
            n = counts.get(phase, 0)
            row += f" | {n:>10}" if n > 0 else f" | {'·':>10}"
        row += f" | {total:>6}"
        print(row)


def skill_lifecycle():
    """Track individual skill birth, persistence, and retirement across checkpoints."""
    print_header("SECTION B — Skill Lifecycle Across Training Steps", 100)

    for run_name, cfg in RUNS.items():
        base = cfg["base"]
        game_dirs = cfg["game_dirs"]
        best_step = cfg["best_step"]

        if cfg.get("no_checkpoints"):
            continue

        ckpt_steps = get_checkpoint_steps(base)
        if len(ckpt_steps) < 2:
            continue

        print_subheader(f"{run_name} (checkpoints: {ckpt_steps}, best={best_step})")

        step_skills = {}
        for step in ckpt_steps:
            skills = load_checkpoint_bank(base, step, game_dirs)
            step_skills[step] = {
                sk.get("skill_id", "?"): sk for sk in skills
            }

        all_skill_ids = set()
        for sk_dict in step_skills.values():
            all_skill_ids.update(sk_dict.keys())

        # Presence matrix
        print(f"\n  Skill presence across checkpoints:")
        print(f"  {'Skill ID':<35}", end="")
        for step in ckpt_steps:
            print(f" | step_{step:04d}", end="")
        print(f" | {'Lifespan':>8}")
        print(f"  {'-'*35}" + "".join(f"-+-{'-'*9}" for _ in ckpt_steps) + f"-+-{'-'*8}")

        for sid in sorted(all_skill_ids):
            row = f"  {sid:<35}"
            present_count = 0
            for step in ckpt_steps:
                if sid in step_skills[step]:
                    sk = step_skills[step][sid]
                    retired = sk.get("retired", False)
                    row += f" | {'RETIRED' if retired else '  ✓  ':>9}"
                    present_count += 1
                else:
                    row += f" | {'·':>9}"
            row += f" | {present_count:>4}/{len(ckpt_steps):>2}"
            print(row)


def skill_churn():
    """Compute churn between consecutive checkpoints."""
    print_header("SECTION C — Skill Churn Between Checkpoints", 100)

    for run_name, cfg in RUNS.items():
        base = cfg["base"]
        game_dirs = cfg["game_dirs"]

        if cfg.get("no_checkpoints"):
            continue

        ckpt_steps = get_checkpoint_steps(base)
        if len(ckpt_steps) < 2:
            continue

        print_subheader(f"{run_name}")

        step_log = load_step_log(base)
        reward_at = {e["step"]: e.get("mean_reward", 0) for e in step_log}

        print(f"  {'Transition':<20} | {'Added':>6} | {'Removed':>7} | {'Kept':>5} | {'Jaccard':>7} | {'Mean Rew':>8}")
        print(f"  {'-'*20}-+-{'-'*6}-+-{'-'*7}-+-{'-'*5}-+-{'-'*7}-+-{'-'*8}")

        prev_ids = None
        for step in ckpt_steps:
            skills = load_checkpoint_bank(base, step, game_dirs)
            curr_ids = set(sk.get("skill_id", "?") for sk in skills)

            if prev_ids is not None:
                added = curr_ids - prev_ids
                removed = prev_ids - curr_ids
                kept = curr_ids & prev_ids
                union = curr_ids | prev_ids
                jaccard = len(kept) / len(union) if union else 1.0
                mr = reward_at.get(step, 0)

                trans = f"→ step_{step:04d}"
                print(f"  {trans:<20} | {len(added):>6} | {len(removed):>7} | {len(kept):>5} | {jaccard:>7.3f} | {mr:>8.2f}")
                if added:
                    print(f"    + Added: {sorted(added)}")
                if removed:
                    print(f"    - Removed: {sorted(removed)}")

            prev_ids = curr_ids


def contract_refinement():
    """Track how skill contracts grow more specific over training."""
    print_header("SECTION D — Contract Refinement Over Training", 100)

    for run_name, cfg in RUNS.items():
        base = cfg["base"]
        game_dirs = cfg["game_dirs"]

        if cfg.get("no_checkpoints"):
            continue

        ckpt_steps = get_checkpoint_steps(base)
        if len(ckpt_steps) < 2:
            continue

        print_subheader(f"{run_name}")

        print(f"  {'Step':>6} | {'#Skills':>7} | {'Avg ver':>7} | {'Avg eff_add':>11} | {'Avg eff_del':>11} | {'Tot clauses':>11} | {'Avg n_inst':>10}")
        print(f"  {'-'*6}-+-{'-'*7}-+-{'-'*7}-+-{'-'*11}-+-{'-'*11}-+-{'-'*11}-+-{'-'*10}")

        for step in ckpt_steps:
            skills = load_checkpoint_bank(base, step, game_dirs)
            if not skills:
                continue

            versions = []
            eff_adds = []
            eff_dels = []
            n_instances = []

            for sk in skills:
                c = sk.get("contract", {})
                versions.append(c.get("version", sk.get("version", 1)))
                eff_adds.append(len(c.get("eff_add", [])))
                eff_dels.append(len(c.get("eff_del", [])))
                n_instances.append(c.get("n_instances", sk.get("n_instances", 0)))

            n = len(skills)
            avg_ver = sum(versions) / n
            avg_ea = sum(eff_adds) / n
            avg_ed = sum(eff_dels) / n
            tot_clauses = sum(eff_adds) + sum(eff_dels)
            avg_ni = sum(n_instances) / n

            print(f"  {step:>6} | {n:>7} | {avg_ver:>7.1f} | {avg_ea:>11.1f} | {avg_ed:>11.1f} | {tot_clauses:>11} | {avg_ni:>10.1f}")

        # Per-skill detail at last checkpoint
        last_step = ckpt_steps[-1]
        skills = load_checkpoint_bank(base, last_step, game_dirs)
        if skills:
            print(f"\n  Per-skill detail at step {last_step}:")
            print(f"  {'Skill ID':<35} | {'Ver':>4} | {'+eff':>4} | {'-eff':>4} | {'inst':>5} | {'sub_ep':>6}")
            print(f"  {'-'*35}-+-{'-'*4}-+-{'-'*4}-+-{'-'*4}-+-{'-'*5}-+-{'-'*6}")
            for sk in sorted(skills, key=lambda s: s.get("skill_id", "")):
                c = sk.get("contract", {})
                sid = sk.get("skill_id", "?")
                ver = c.get("version", sk.get("version", 1))
                ea = len(c.get("eff_add", []))
                ed = len(c.get("eff_del", []))
                ni = c.get("n_instances", sk.get("n_instances", 0))
                ns = len(sk.get("sub_episodes", []))
                print(f"  {sid:<35} | {ver:>4} | {ea:>4} | {ed:>4} | {ni:>5} | {ns:>6}")


def reuse_concentration():
    """Gini coefficient showing whether a few skills dominate or usage is spread evenly."""
    print_header("SECTION E — Skill Reuse Concentration (Gini)", 100)

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

        if not skills:
            continue

        instances = [sk.get("n_instances", sk.get("contract", {}).get("n_instances", 0)) for sk in skills]
        sub_eps = [len(sk.get("sub_episodes", [])) for sk in skills]

        gini_inst = gini_coefficient(instances)
        gini_sub = gini_coefficient(sub_eps)

        m_inst, s_inst = mean_std(instances)
        m_sub, s_sub = mean_std(sub_eps)

        print(f"  {run_name:<15} | Gini(instances)={gini_inst:.3f}, mean={m_inst:.1f}±{s_inst:.1f} | Gini(sub_eps)={gini_sub:.3f}, mean={m_sub:.1f}±{s_sub:.1f} | #skills={len(skills)}")


def main():
    build_taxonomy()
    skill_lifecycle()
    skill_churn()
    contract_refinement()
    reuse_concentration()


if __name__ == "__main__":
    main()
