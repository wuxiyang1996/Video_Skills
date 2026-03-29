#!/usr/bin/env python3
"""
P1 — Qualitative Skill Retrieval Walkthrough

Produces paper-ready examples showing how the skill bank guides decision-making.
For each game, selects the highest-reward episode and prints a multi-step
walkthrough: state summary → skill candidates → reasoning → chosen skill → action → reward.

Also produces a "with-bank vs without-bank" side-by-side comparison using
ablation episodes for overlapping game/phase pairs.
"""

import json
import os
import textwrap
from collections import defaultdict

from utils import (
    LABELED_BASE, LABELED_GAMES, ABLATION_BASE,
    load_all_labeled_episodes, load_all_ablation_episodes,
    episode_total_reward, extract_intention_tag, parse_summary_state,
    print_header, print_subheader, condition_label, detect_game_from_dir,
    iter_episodes,
)


def format_state_compact(summary_state: str, max_len: int = 120) -> str:
    if not summary_state:
        return "(no summary)"
    if len(summary_state) <= max_len:
        return summary_state
    return summary_state[:max_len] + "..."


def print_step_walkthrough(exp: dict, step_idx: int, indent: str = "    "):
    """Print one step of the walkthrough."""
    skills = exp.get("skills", {}) or {}
    candidates = exp.get("skill_candidates", []) or []
    reasoning = exp.get("skill_reasoning", "") or ""
    chosen_idx = exp.get("skill_chosen_idx", "?")
    confidence = skills.get("confidence", "?")
    action = exp.get("action", "?")
    reward = exp.get("reward", 0)
    intention = exp.get("intentions", "") or ""
    summary = format_state_compact(exp.get("summary_state", ""))

    print(f"{indent}┌─ Step {step_idx}")
    print(f"{indent}│ State:      {summary}")
    print(f"{indent}│ Intention:  {intention[:120]}")
    print(f"{indent}│ Candidates: {candidates}")
    print(f"{indent}│ Chosen:     #{chosen_idx} → {skills.get('skill_id', '?')} (conf={confidence})")
    if reasoning:
        wrapped = textwrap.fill(reasoning[:300], width=80, initial_indent=f"{indent}│   ", subsequent_indent=f"{indent}│   ")
        print(f"{indent}│ Reasoning:")
        print(wrapped)
    print(f"{indent}│ Action:     {str(action)[:120]}")
    print(f"{indent}│ Reward:     {reward}")
    print(f"{indent}└{'─' * 60}")


def select_interesting_window(experiences: list, window_size: int = 5) -> int:
    """Find the most interesting consecutive window: max cumulative reward slope."""
    if len(experiences) <= window_size:
        return 0
    best_start, best_score = 0, float("-inf")
    for i in range(len(experiences) - window_size + 1):
        window = experiences[i:i + window_size]
        rewards = [e.get("reward", 0) for e in window]
        has_skills = all(e.get("skills") for e in window)
        if not has_skills:
            continue
        n_unique_skills = len(set(
            (e.get("skills") or {}).get("skill_id", "?") for e in window
        ))
        score = sum(rewards) + n_unique_skills * 2
        if score > best_score:
            best_score = score
            best_start = i
    return best_start


def walkthrough_best_episode(game: str, top_k: int = 1, window: int = 5):
    """Print walkthrough for the top-k highest-reward episodes of a game."""
    episodes = load_all_labeled_episodes(game)
    if not episodes:
        print(f"  [No labeled episodes for {game}]")
        return

    episodes.sort(key=episode_total_reward, reverse=True)

    for rank, ep in enumerate(episodes[:top_k]):
        total_r = episode_total_reward(ep)
        n_steps = len(ep.get("experiences", []))
        meta = ep.get("metadata", {})
        model = meta.get("model", "?") if isinstance(meta, dict) else "?"
        print(f"\n  Episode rank #{rank+1}  |  game={game}  |  total_reward={total_r:.2f}  |  steps={n_steps}  |  model={model}")

        exps = ep.get("experiences", [])
        has_skills = [e for e in exps if e.get("skills")]
        if not has_skills:
            print("    [No skill retrieval data in this episode]")
            continue

        start = select_interesting_window(exps, window)
        print(f"  Showing steps {start}–{start + window - 1} (most interesting window):\n")

        for i in range(start, min(start + window, len(exps))):
            print_step_walkthrough(exps[i], i)


def bank_vs_no_bank_comparison():
    """Side-by-side comparison of with-bank vs no-bank episodes."""
    print_header("WITH-BANK vs NO-BANK SIDE-BY-SIDE COMPARISON")

    all_eps = load_all_ablation_episodes()

    for game in ["diplomacy", "avalon"]:
        bank_eps = all_eps.get(("RL + Bank", game), [])
        no_bank_eps = all_eps.get(("RL (no bank)", game), [])
        sft_no_bank_eps = all_eps.get(("SFT (no bank)", game), [])

        if not bank_eps:
            continue

        comparison_eps = no_bank_eps or sft_no_bank_eps
        comp_label = "RL (no bank)" if no_bank_eps else "SFT (no bank)"

        print_subheader(f"{game.upper()}: RL+Bank ({len(bank_eps)} eps) vs {comp_label} ({len(comparison_eps)} eps)")

        bank_rewards = [episode_total_reward(e) for e in bank_eps]
        comp_rewards = [episode_total_reward(e) for e in comparison_eps]

        bank_mean = sum(bank_rewards) / len(bank_rewards) if bank_rewards else 0
        comp_mean = sum(comp_rewards) / len(comp_rewards) if comp_rewards else 0

        print(f"  Mean reward:  RL+Bank = {bank_mean:.2f}  |  {comp_label} = {comp_mean:.2f}  |  Δ = {bank_mean - comp_mean:+.2f}")

        # Show best bank episode and best no-bank episode intentions side by side
        best_bank = max(bank_eps, key=episode_total_reward)
        exps_bank = best_bank.get("experiences", [])

        print(f"\n  Best RL+Bank episode (reward={episode_total_reward(best_bank):.2f}, steps={len(exps_bank)}):")
        print(f"  Intention sequence:")
        for i, e in enumerate(exps_bank[:10]):
            tag = extract_intention_tag(e.get("intentions", ""))
            r = e.get("reward", 0)
            summ = format_state_compact(e.get("summary_state", ""), 80)
            print(f"    Step {i:>2}: [{tag or '?':>10}]  r={r:>6.2f}  state={summ}")

        if comparison_eps:
            best_comp = max(comparison_eps, key=episode_total_reward)
            exps_comp = best_comp.get("experiences", [])
            print(f"\n  Best {comp_label} episode (reward={episode_total_reward(best_comp):.2f}, steps={len(exps_comp)}):")
            print(f"  Intention sequence:")
            for i, e in enumerate(exps_comp[:10]):
                tag = extract_intention_tag(e.get("intentions", ""))
                r = e.get("reward", 0)
                summ = format_state_compact(e.get("summary_state", ""), 80)
                print(f"    Step {i:>2}: [{tag or '?':>10}]  r={r:>6.2f}  state={summ}")


def skill_bank_entry_showcase():
    """Print 2-3 representative skill entries for the paper."""
    print_header("SKILL BANK ENTRY SHOWCASE")

    from utils import parse_skill_bank, RUNS

    for run_name, cfg in list(RUNS.items())[:3]:
        base = cfg["base"]
        game_dirs = cfg["game_dirs"]

        skills = []
        for gd in game_dirs:
            fp = os.path.join(base, "skillbank", gd, "skill_bank.jsonl")
            skills.extend(parse_skill_bank(fp))

        if not skills:
            best_step = cfg["best_step"]
            from utils import load_checkpoint_bank, get_checkpoint_steps
            ckpts = get_checkpoint_steps(base)
            if ckpts:
                step = min(ckpts, key=lambda s: abs(s - best_step))
                skills = load_checkpoint_bank(base, step, game_dirs)

        if not skills:
            continue

        print_subheader(f"{run_name}: {len(skills)} skills")

        for sk in skills[:2]:
            sid = sk.get("skill_id", "?")
            name = sk.get("name", sk.get("strategic_description", "")[:60])
            desc = sk.get("strategic_description", "")[:200]
            protocol = sk.get("protocol", {})
            contract = sk.get("contract", {})
            hint = sk.get("execution_hint", {}) or {}
            n_inst = sk.get("n_instances", contract.get("n_instances", 0))
            n_sub = len(sk.get("sub_episodes", []))
            retired = sk.get("retired", False)

            print(f"\n  ┌─ Skill: {sid}  (v{sk.get('version', contract.get('version', '?'))})")
            print(f"  │ Name: {name}")
            if desc:
                print(f"  │ Description: {desc}")
            if protocol:
                pre = protocol.get("preconditions", [])[:3]
                steps = protocol.get("steps", [])[:4]
                success = protocol.get("success_criteria", [])[:3]
                if pre:
                    print(f"  │ Preconditions: {pre}")
                if steps:
                    print(f"  │ Protocol steps: {steps}")
                if success:
                    print(f"  │ Success criteria: {success}")
            eff_add = contract.get("eff_add", [])[:5]
            eff_del = contract.get("eff_del", [])[:5]
            print(f"  │ Contract: +{len(contract.get('eff_add', []))} effects, -{len(contract.get('eff_del', []))} effects")
            if eff_add:
                print(f"  │   eff_add sample: {eff_add}")
            if eff_del:
                print(f"  │   eff_del sample: {eff_del}")
            if hint:
                failures = hint.get("common_failure_modes", [])[:2]
                if failures:
                    print(f"  │ Common failures: {failures}")
            print(f"  │ Instances: {n_inst}  |  Sub-episodes: {n_sub}  |  Retired: {retired}")
            print(f"  └{'─' * 60}")


def main():
    print_header("P1: QUALITATIVE SKILL RETRIEVAL WALKTHROUGH", 90)

    print_subheader("SECTION A — Best-episode walkthrough per game (labeled data)")
    for game in LABELED_GAMES:
        print_subheader(f"Game: {game}")
        walkthrough_best_episode(game, top_k=1, window=5)

    print("\n")
    bank_vs_no_bank_comparison()

    print("\n")
    skill_bank_entry_showcase()


if __name__ == "__main__":
    main()
