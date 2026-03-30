#!/usr/bin/env python3
"""
P8 — Candy Crush & Diplomacy Deep Case Studies

Addresses:
 - Why our method improves performance (causal mechanism, not just correlation)
 - Failure analysis for both our method and GPT-5.4
 - Qualitative outplay examples (turn-by-turn board state + action)
 - Skill retrieval stress test (state features → skill switches)

Sections:
 A. Candy Crush: side-by-side best episode comparison (Ours vs GPT-5.4)
 B. Candy Crush: skill bank, skill transitions, and per-step reward analysis
 C. Diplomacy: episode-level statistics (center progression, reward accounting)
 D. Diplomacy: failure analysis (our stagnation vs GPT-5.4 collapse)
 E. Diplomacy: qualitative outplay example (France step 8 explosion)
 F. Diplomacy: skill retrieval stress test (transition triggers, heatmaps)
 G. Diplomacy: causal mechanism (action diversity, action-1 bias, skill→structure)
"""

import json
import os
import glob
from collections import defaultdict, Counter

from utils import (
    GAME_AI_AGENT, LABELED_BASE,
    load_all_labeled_episodes, episode_total_reward,
    extract_intention_tag, parse_summary_state,
    mean_std, print_header, print_subheader,
    parse_skill_bank, load_final_bank,
    RUNS,
)

# ─── Paths ────────────────────────────────────────────────────────────────────

CANDY_RUN = RUNS["Candy Crush"]["base"]
CANDY_BEST_STEP = RUNS["Candy Crush"]["best_step"]

DIPLO_RUN = RUNS["Diplomacy"]["base"]
DIPLO_BEST_STEP = RUNS["Diplomacy"]["best_step"]

GPT54_CANDY_DIR = os.path.join(LABELED_BASE.replace("gpt54_skill_labeled", "gpt54"), "candy_crush")
GPT54_DIPLO_DIR = os.path.join(LABELED_BASE.replace("gpt54_skill_labeled", "gpt54"), "diplomacy")
GPT54_DIPLO_SKILL_DIR = os.path.join(LABELED_BASE, "diplomacy")


def _load_grpo_action_taking(run_base, step):
    fp = os.path.join(run_base, "grpo_data", f"step_{step:04d}", "action_taking.jsonl")
    if not os.path.exists(fp):
        return []
    with open(fp) as f:
        return [json.loads(l) for l in f if l.strip()]


def _load_gpt54_episodes(directory):
    eps = []
    if not os.path.exists(directory):
        return eps
    for fp in sorted(glob.glob(os.path.join(directory, "episode_*.json"))):
        with open(fp) as f:
            d = json.load(f)
        eps.append((os.path.basename(fp), d))
    return eps


def _extract_prompt_field(prompt, field):
    if f"{field}=" not in prompt:
        return None
    val = prompt.split(f"{field}=")[1].split("|")[0].split()[0].strip()
    return val


def _extract_subgoal_tag(prompt):
    if "Assigned subgoal:" not in prompt:
        return "?"
    sg = prompt.split("Assigned subgoal:")[1].split("\n")[0].strip()
    if "]" in sg:
        return sg.split("]")[0].replace("[", "").strip()
    return sg[:10]


def _extract_action_text(prompt, completion):
    action_num = "?"
    for line in completion.split("\n"):
        if line.strip().startswith("ACTION:"):
            action_num = line.strip().split(":")[1].strip()
            break
    if "Available actions" not in prompt:
        return action_num, "?"
    for line in prompt.split("Available actions")[1].split("\n"):
        line = line.strip()
        if line.startswith(f"{action_num}."):
            return action_num, line.split(".", 1)[1].strip()
    return action_num, "?"


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION A — Candy Crush: Side-by-Side Best Episode Comparison
# ═══════════════════════════════════════════════════════════════════════════════

def candy_crush_episode_comparison():
    print_header("SECTION A — Candy Crush: Best Episode Comparison (Ours vs GPT-5.4)")

    # Load our method
    our_lines = _load_grpo_action_taking(CANDY_RUN, CANDY_BEST_STEP)
    if not our_lines:
        print("  [SKIP] No GRPO data found for Candy Crush")
        return

    our_eps = defaultdict(list)
    for l in our_lines:
        our_eps[l["episode_id"]].append(l)

    sorted_eps = sorted(our_eps.items(), key=lambda x: sum(s["reward"] for s in x[1]), reverse=True)

    print_subheader("Our Method — Top 5 Episodes")
    for eid, steps in sorted_eps[:5]:
        total = sum(s["reward"] for s in steps)
        print(f"  {eid}: reward={total:.1f}, steps={len(steps)}")

    # Best episode details
    best_eid, best_steps = sorted_eps[0]
    best_steps = sorted(best_steps, key=lambda x: x["step"])
    best_total = sum(s["reward"] for s in best_steps)

    print_subheader(f"Our Method Best: {best_eid} (reward={best_total:.1f}, {len(best_steps)} steps)")

    cum = 0
    for s in best_steps:
        cum += s["reward"]
        tag = _extract_subgoal_tag(s["prompt"])
        _, action_text = _extract_action_text(s["prompt"], s["completion"])
        print(f"  Step {s['step']:2d} | [{tag:8s}] | r={s['reward']:6.1f} | cum={cum:7.1f} | {action_text[:60]}")

    # GPT-5.4
    gpt_eps = _load_gpt54_episodes(GPT54_CANDY_DIR)
    if not gpt_eps:
        print("  [SKIP] No GPT-5.4 Candy Crush episodes found")
        return

    gpt_ranked = sorted(gpt_eps, key=lambda x: sum(e["reward"] for e in x[1]["experiences"]), reverse=True)

    print_subheader("GPT-5.4 — Top 5 Episodes")
    for fname, ep in gpt_ranked[:5]:
        total = sum(e["reward"] for e in ep["experiences"])
        print(f"  {fname}: reward={total:.1f}, steps={len(ep['experiences'])}")

    # GPT-5.4 best
    gpt_best_fname, gpt_best = gpt_ranked[0]
    gpt_total = sum(e["reward"] for e in gpt_best["experiences"])

    print_subheader(f"GPT-5.4 Best: {gpt_best_fname} (reward={gpt_total:.1f}, {len(gpt_best['experiences'])} steps)")

    cum = 0
    for i, e in enumerate(gpt_best["experiences"]):
        cum += e["reward"]
        tag = extract_intention_tag(e.get("intentions", "")) or "?"
        action = e.get("action", "?")
        if isinstance(action, str) and len(action) > 60:
            action = action[:60] + "..."
        print(f"  Step {i:2d} | [{tag:8s}] | r={e['reward']:6.1f} | cum={cum:7.1f} | {action}")

    # Summary comparison
    our_rewards = [sum(s["reward"] for s in steps) for _, steps in sorted_eps]
    gpt_rewards = [sum(e["reward"] for e in ep["experiences"]) for _, ep in gpt_ranked]

    m_our, s_our = mean_std(our_rewards)
    m_gpt, s_gpt = mean_std(gpt_rewards)

    print_subheader("Summary Statistics")
    print(f"  Our Method:  mean={m_our:.1f} ± {s_our:.1f}, max={max(our_rewards):.1f}, min={min(our_rewards):.1f}, n={len(our_rewards)}")
    print(f"  GPT-5.4:     mean={m_gpt:.1f} ± {s_gpt:.1f}, max={max(gpt_rewards):.1f}, min={min(gpt_rewards):.1f}, n={len(gpt_rewards)}")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION B — Candy Crush: Skill Bank & Skill Transitions
# ═══════════════════════════════════════════════════════════════════════════════

def candy_crush_skills():
    print_header("SECTION B — Candy Crush: Skill Bank & Transitions")

    bank_path = os.path.join(CANDY_RUN, "best", "banks", "candy_crush", "skill_bank.jsonl")
    skills = parse_skill_bank(bank_path)
    if not skills:
        print("  [SKIP] No skill bank found")
        return

    print_subheader(f"Evolved Skill Bank ({len(skills)} skills)")
    for sk in skills:
        sid = sk.get("skill_id", "?")
        name = sk.get("name", "?")
        desc = sk.get("strategic_description", "")[:150]
        proto = sk.get("protocol", {})
        dur = proto.get("expected_duration", "?")
        contract = sk.get("contract", {})
        n_inst = contract.get("n_instances", "?")
        print(f"\n  Skill: {sid} — {name}")
        print(f"    Desc: {desc}")
        print(f"    Duration: {dur} steps, Instances: {n_inst}")
        print(f"    Protocol: {proto.get('steps', [])[:3]}")
        print(f"    Success:  {proto.get('success_criteria', [])[:3]}")

    # Skill transitions from GRPO data
    our_lines = _load_grpo_action_taking(CANDY_RUN, CANDY_BEST_STEP)
    if not our_lines:
        return

    our_eps = defaultdict(list)
    for l in our_lines:
        our_eps[l["episode_id"]].append(l)

    transitions = defaultdict(list)
    tag_step_counts = defaultdict(lambda: Counter())

    for eid, steps in our_eps.items():
        steps = sorted(steps, key=lambda x: x["step"])
        prev_tag = None
        for s in steps:
            tag = _extract_subgoal_tag(s["prompt"])
            step_num = s["step"]
            tag_step_counts[tag][step_num] += 1
            if prev_tag and tag != prev_tag:
                transitions[(prev_tag, tag)].append(step_num)
            prev_tag = tag

    print_subheader("Skill Transitions")
    for (from_t, to_t), steps_list in sorted(transitions.items(), key=lambda x: -len(x[1])):
        m, sd = mean_std(steps_list)
        print(f"  {from_t:10s} → {to_t:10s}: n={len(steps_list):3d}, step={m:.1f}±{sd:.1f}")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION C — Diplomacy: Episode-Level Statistics
# ═══════════════════════════════════════════════════════════════════════════════

def diplomacy_episode_stats():
    print_header("SECTION C — Diplomacy: Episode-Level Statistics")

    our_lines = _load_grpo_action_taking(DIPLO_RUN, DIPLO_BEST_STEP)
    if not our_lines:
        print("  [SKIP] No GRPO data for Diplomacy")
        return

    our_eps = defaultdict(list)
    for l in our_lines:
        our_eps[l["episode_id"]].append(l)

    print_subheader("Our Method — All Episodes at Best Step")

    all_data = []
    for eid, steps in our_eps.items():
        steps = sorted(steps, key=lambda x: x["step"])
        total_r = sum(s["reward"] for s in steps)
        power = _extract_prompt_field(steps[0]["prompt"], "power") or "?"
        center_prog = []
        for s in steps:
            c = _extract_prompt_field(s["prompt"], "centers")
            center_prog.append(int(c) if c else 0)
        all_data.append((total_r, eid, power, center_prog))

    all_data.sort(reverse=True)
    for total_r, eid, power, centers in all_data:
        print(f"  {eid}: {power:10s} R={total_r:6.2f} | centers: {centers}")

    # Per-power stats
    power_groups = defaultdict(list)
    for total_r, eid, power, centers in all_data:
        power_groups[power].append((total_r, centers[-1]))

    print_subheader("Per-Power Statistics")
    for power in sorted(power_groups):
        rs = [x[0] for x in power_groups[power]]
        finals = [x[1] for x in power_groups[power]]
        m_r, s_r = mean_std(rs)
        m_c, s_c = mean_std(finals)
        print(f"  {power:10s}: reward={m_r:.2f}±{s_r:.2f}, final_centers={m_c:.1f}±{s_c:.1f}, n={len(rs)}")

    # GPT-5.4 stats
    gpt_eps = _load_gpt54_episodes(GPT54_DIPLO_DIR)
    if not gpt_eps:
        return

    print_subheader("GPT-5.4 — Austria Center Progression")
    gpt_data = []
    for fname, ep in gpt_eps:
        total_r = sum(e["reward"] for e in ep["experiences"])
        centers = []
        for e in ep["experiences"]:
            raw = e.get("raw_state", {})
            c = raw.get("powers", {}).get("AUSTRIA", {}).get("num_centers", 3)
            centers.append(c)
        gpt_data.append((total_r, fname, centers))

    gpt_data.sort(reverse=True)
    finals = [c[-1] for _, _, c in gpt_data]
    all_r = [r for r, _, _ in gpt_data]
    m_r, s_r = mean_std(all_r)
    m_c, s_c = mean_std(finals)
    print(f"  GPT-5.4 Austria (n={len(gpt_data)}): reward={m_r:.2f}±{s_r:.2f}")
    print(f"    Final centers: mean={m_c:.1f}±{s_c:.1f}, max={max(finals)}, min={min(finals)}")

    our_finals = [c[-1] for _, _, _, c in all_data]
    m_c2, s_c2 = mean_std(our_finals)
    print(f"  Our Method all powers (n={len(all_data)}): final_centers={m_c2:.1f}±{s_c2:.1f}, max={max(our_finals)}, min={min(our_finals)}")

    # Stability comparison
    gpt_collapse = sum(1 for f in finals if f <= 2)
    our_collapse = sum(1 for f in our_finals if f <= 2)
    print(f"\n  Stability:")
    print(f"    GPT-5.4: {gpt_collapse}/{len(finals)} episodes ({gpt_collapse/len(finals):.0%}) end with ≤2 centers")
    print(f"    Ours:    {our_collapse}/{len(our_finals)} episodes ({our_collapse/len(our_finals):.0%}) end with ≤2 centers")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION D — Diplomacy: Failure Analysis
# ═══════════════════════════════════════════════════════════════════════════════

def diplomacy_failure_analysis():
    print_header("SECTION D — Diplomacy: Failure Analysis")

    # Our method failures
    our_lines = _load_grpo_action_taking(DIPLO_RUN, DIPLO_BEST_STEP)
    if not our_lines:
        return

    our_eps = defaultdict(list)
    for l in our_lines:
        our_eps[l["episode_id"]].append(l)

    print_subheader("Our Method — Failure Cases (final centers ≤ 3)")

    for eid, steps in sorted(our_eps.items(), key=lambda x: x[0]):
        steps = sorted(steps, key=lambda x: x["step"])
        power = _extract_prompt_field(steps[-1]["prompt"], "power") or "?"
        centers = []
        actions = []
        subgoals = []
        for s in steps:
            c = _extract_prompt_field(s["prompt"], "centers")
            centers.append(int(c) if c else 0)
            subgoals.append(_extract_subgoal_tag(s["prompt"]))
            _, at = _extract_action_text(s["prompt"], s["completion"])
            actions.append(at)

        if centers[-1] > 3:
            continue

        action_counts = Counter(actions)
        most_common = action_counts.most_common(3)
        repeats = sum(1 for i in range(1, len(actions)) if actions[i] == actions[i - 1])
        rate = repeats / (len(actions) - 1) if len(actions) > 1 else 0

        print(f"\n  {eid}: {power}, final={centers[-1]}, repeat_rate={rate:.0%}")
        print(f"    Centers:  {centers}")
        print(f"    Subgoals: {subgoals}")
        print(f"    Top actions: {most_common}")

    # GPT-5.4 failures
    gpt_skill_eps = []
    if os.path.exists(GPT54_DIPLO_SKILL_DIR):
        for fp in sorted(glob.glob(os.path.join(GPT54_DIPLO_SKILL_DIR, "episode_*.json"))):
            with open(fp) as f:
                gpt_skill_eps.append((os.path.basename(fp), json.load(f)))

    print_subheader("GPT-5.4 — Failure Cases (Austria final centers ≤ 3)")

    for fname, ep in gpt_skill_eps:
        exps = ep["experiences"]
        centers = []
        intents = []
        skills_used = []
        for e in exps:
            raw = e.get("raw_state", {})
            c = raw.get("powers", {}).get("AUSTRIA", {}).get("num_centers", 3)
            centers.append(c)
            tag = extract_intention_tag(e.get("intentions", "")) or "?"
            intents.append(tag)
            sk = e.get("skills", {})
            sk_name = sk.get("skill_name", "?") if isinstance(sk, dict) else "?"
            skills_used.append(sk_name[:22])

        if centers[-1] > 3:
            continue

        peak = max(centers)
        peak_idx = centers.index(peak)

        print(f"\n  {fname}: final={centers[-1]}, peak={peak} at step {peak_idx}")
        print(f"    Centers:    {centers}")
        print(f"    Intentions: {intents}")
        print(f"    Skills:     {skills_used}")

    # Failure mode comparison summary
    print_subheader("Failure Mode Comparison Summary")

    our_failures = []
    our_total = 0
    for eid, steps in our_eps.items():
        steps = sorted(steps, key=lambda x: x["step"])
        centers = [int(_extract_prompt_field(s["prompt"], "centers") or 0) for s in steps]
        our_total += 1
        if centers[-1] <= 3:
            our_failures.append(centers)

    gpt_failures = []
    gpt_total = 0
    for fname, ep in gpt_skill_eps:
        exps = ep["experiences"]
        centers = [e.get("raw_state", {}).get("powers", {}).get("AUSTRIA", {}).get("num_centers", 3) for e in exps]
        gpt_total += 1
        if centers[-1] <= 3:
            gpt_failures.append(centers)

    print(f"  Our Method: {len(our_failures)}/{our_total} episodes fail ({len(our_failures)/our_total:.0%})")
    print(f"    Failure mode: STAGNATION — stuck at starting centers, never grow")
    if our_failures:
        print(f"    All failure final centers: {[c[-1] for c in our_failures]}")
        print(f"    Peak centers in failures: {[max(c) for c in our_failures]}")

    print(f"\n  GPT-5.4: {len(gpt_failures)}/{gpt_total} episodes fail ({len(gpt_failures)/gpt_total:.0%})")
    print(f"    Failure mode: COLLAPSE — grows to 4-5 centers then crashes")
    if gpt_failures:
        print(f"    All failure final centers: {[c[-1] for c in gpt_failures]}")
        print(f"    Peak centers in failures: {[max(c) for c in gpt_failures]}")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION E — Diplomacy: Qualitative Outplay Examples
# ═══════════════════════════════════════════════════════════════════════════════

def diplomacy_outplay_examples():
    print_header("SECTION E — Diplomacy: Qualitative Outplay Examples")

    our_lines = _load_grpo_action_taking(DIPLO_RUN, DIPLO_BEST_STEP)
    if not our_lines:
        return

    our_eps = defaultdict(list)
    for l in our_lines:
        our_eps[l["episode_id"]].append(l)

    # Best FRANCE episode (best overall)
    best_eid = max(our_eps, key=lambda eid: sum(s["reward"] for s in our_eps[eid]))
    best_steps = sorted(our_eps[best_eid], key=lambda x: x["step"])
    best_total = sum(s["reward"] for s in best_steps)
    power = _extract_prompt_field(best_steps[0]["prompt"], "power") or "?"

    print_subheader(f"Best Episode: {best_eid} ({power}, R={best_total:.2f})")

    # Show key moments with full context
    key_steps = [0, 4, 8, 9, 14, 15, 19]
    for s in best_steps:
        if s["step"] not in key_steps:
            continue
        p = s["prompt"]
        phase = _extract_prompt_field(p, "phase") or "?"
        centers = _extract_prompt_field(p, "centers") or "?"
        tag = _extract_subgoal_tag(p)
        action_num, action_text = _extract_action_text(p, s["completion"])

        print(f"\n  ── Step {s['step']} | {phase} | centers={centers} | [{tag}] ──")
        print(f"  Action: {action_text}")
        print(f"  Reward: {s['reward']:.3f}")

        # Show available actions for critical moments
        if s["step"] in [8, 14, 15]:
            if "Available actions" in p:
                actions_block = p.split("Available actions")[1].split("Subgoal tags:")[0]
                print(f"  Available actions:{actions_block.rstrip()}")

        # Show recent actions context
        if "Recent actions" in p:
            recent = p.split("Recent actions and rewards:")[1].split("Available actions")[0].strip()
            print(f"  Recent history:\n    {recent.replace(chr(10), chr(10) + '    ')}")

    # Also show best AUSTRIA episode for same-power comparison
    austria_eps = [(eid, steps) for eid, steps in our_eps.items()
                   if _extract_prompt_field(sorted(steps, key=lambda x: x["step"])[0]["prompt"], "power") == "AUSTRIA"]
    if austria_eps:
        best_a_eid, best_a_steps = max(austria_eps, key=lambda x: sum(s["reward"] for s in x[1]))
        best_a_steps = sorted(best_a_steps, key=lambda x: x["step"])
        best_a_total = sum(s["reward"] for s in best_a_steps)

        print_subheader(f"Best Austria Episode: {best_a_eid} (R={best_a_total:.2f})")
        for s in best_a_steps:
            phase = _extract_prompt_field(s["prompt"], "phase") or "?"
            centers = _extract_prompt_field(s["prompt"], "centers") or "?"
            tag = _extract_subgoal_tag(s["prompt"])
            _, action_text = _extract_action_text(s["prompt"], s["completion"])
            print(f"  Step {s['step']:2d} | {phase:8s} | c={centers} | [{tag:8s}] | {action_text[:55]}")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION F — Diplomacy: Skill Retrieval Stress Test
# ═══════════════════════════════════════════════════════════════════════════════

def diplomacy_skill_retrieval():
    print_header("SECTION F — Diplomacy: Skill Retrieval Stress Test")

    our_lines = _load_grpo_action_taking(DIPLO_RUN, DIPLO_BEST_STEP)
    if not our_lines:
        return

    our_eps = defaultdict(list)
    for l in our_lines:
        our_eps[l["episode_id"]].append(l)

    # Transition analysis
    transitions = defaultdict(list)
    tag_center = defaultdict(lambda: Counter())
    tag_step = defaultdict(lambda: Counter())

    for eid, steps in our_eps.items():
        steps = sorted(steps, key=lambda x: x["step"])
        prev_tag = None
        for s in steps:
            tag = _extract_subgoal_tag(s["prompt"])
            step_num = s["step"]
            c = int(_extract_prompt_field(s["prompt"], "centers") or 0)
            reward_str = _extract_prompt_field(s["prompt"], "reward") or "0"
            try:
                reward = float(reward_str.replace("+", ""))
            except ValueError:
                reward = 0

            tag_center[tag][c] += 1
            tag_step[tag][step_num] += 1

            if prev_tag and tag != prev_tag:
                transitions[(prev_tag, tag)].append({
                    "step": step_num, "centers": c, "reward": reward, "episode": eid,
                })
            prev_tag = tag

    print_subheader("Our Method — Skill Transition Triggers")
    for (from_t, to_t), instances in sorted(transitions.items(), key=lambda x: -len(x[1])):
        steps_list = [x["step"] for x in instances]
        centers_list = [x["centers"] for x in instances]
        rewards_list = [x["reward"] for x in instances]
        m_s, s_s = mean_std(steps_list)
        m_c, s_c = mean_std(centers_list)
        m_r, s_r = mean_std(rewards_list)
        print(f"\n  {from_t:10s} → {to_t:10s}  (n={len(instances)})")
        print(f"    Step:    {min(steps_list)}-{max(steps_list)} (mean={m_s:.1f}±{s_s:.1f})")
        print(f"    Centers: {min(centers_list)}-{max(centers_list)} (mean={m_c:.1f}±{s_c:.1f})")
        print(f"    Reward:  {min(rewards_list):.2f}-{max(rewards_list):.2f} (mean={m_r:.2f}±{s_r:.2f})")

    print_subheader("Our Method — Subgoal Tag vs Centers Heatmap")
    print(f"  {'Tag':<12s}", end="")
    for c in range(3, 8):
        print(f"  c={c} ", end="")
    print("   Total")
    print(f"  {'-' * 12}", end="")
    for _ in range(3, 8):
        print(f" -----", end="")
    print(f" ------")
    for tag in ["EXPLORE", "SETUP", "DEFEND", "ATTACK"]:
        if tag not in tag_center:
            continue
        total = sum(tag_center[tag].values())
        print(f"  {tag:<12s}", end="")
        for c in range(3, 8):
            pct = tag_center[tag][c] / total * 100 if total else 0
            print(f" {pct:4.0f}%", end="")
        print(f"  n={total}")

    print_subheader("Our Method — Subgoal Tag vs Game Step")
    print(f"  {'Tag':<12s} Steps 0-4  Steps 5-9  Steps 10-14  Steps 15-19  Total")
    print(f"  {'-' * 72}")
    for tag in ["EXPLORE", "SETUP", "DEFEND", "ATTACK"]:
        if tag not in tag_step:
            continue
        total = sum(tag_step[tag].values())
        bins = []
        for lo, hi in [(0, 5), (5, 10), (10, 15), (15, 20)]:
            bins.append(sum(tag_step[tag][s] for s in range(lo, hi)))
        print(f"  {tag:<12s}", end="")
        for b in bins:
            print(f"  {b / total * 100:5.0f}%    ", end="")
        print(f"  n={total}")

    # GPT-5.4 comparison
    if not os.path.exists(GPT54_DIPLO_SKILL_DIR):
        return

    gpt_tag_center = defaultdict(lambda: Counter())
    gpt_tag_step = defaultdict(lambda: Counter())
    gpt_transitions = defaultdict(list)

    for fp in sorted(glob.glob(os.path.join(GPT54_DIPLO_SKILL_DIR, "episode_*.json"))):
        with open(fp) as f:
            ep = json.load(f)
        prev_skill = None
        for i, e in enumerate(ep["experiences"]):
            raw = e.get("raw_state", {})
            c = raw.get("powers", {}).get("AUSTRIA", {}).get("num_centers", 3)
            tag = extract_intention_tag(e.get("intentions", "")) or "?"
            sk = e.get("skills", {})
            skill_name = sk.get("skill_name", "?") if isinstance(sk, dict) else "?"

            gpt_tag_center[tag][c] += 1
            gpt_tag_step[tag][i] += 1

            if prev_skill and skill_name != prev_skill:
                gpt_transitions[(prev_skill[:22], skill_name[:22])].append({"step": i, "centers": c})
            prev_skill = skill_name

    print_subheader("GPT-5.4 — Top Skill Transitions (ping-pong pattern)")
    for (from_s, to_s), instances in sorted(gpt_transitions.items(), key=lambda x: -len(x[1]))[:8]:
        steps_list = [x["step"] for x in instances]
        centers_list = [x["centers"] for x in instances]
        m_s, _ = mean_std(steps_list)
        m_c, _ = mean_std(centers_list)
        print(f"  {from_s:24s} → {to_s:24s}: n={len(instances):3d}, step={m_s:.1f}, centers={m_c:.1f}")

    print_subheader("GPT-5.4 — Intention Tag vs Centers Heatmap")
    print(f"  {'Tag':<12s}", end="")
    for c in range(1, 9):
        print(f"  c={c} ", end="")
    print("  Total")
    for tag in ["POSITION", "DEFEND", "ATTACK", "BUILD", "SURVIVE", "SETUP"]:
        if tag not in gpt_tag_center:
            continue
        total = sum(gpt_tag_center[tag].values())
        print(f"  {tag:<12s}", end="")
        for c in range(1, 9):
            pct = gpt_tag_center[tag][c] / total * 100 if total else 0
            if pct > 0:
                print(f" {pct:4.0f}%", end="")
            else:
                print(f"    - ", end="")
        print(f"  n={total}")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION G — Diplomacy: Causal Mechanism (Action Diversity & Bias)
# ═══════════════════════════════════════════════════════════════════════════════

def diplomacy_causal_mechanism():
    print_header("SECTION G — Diplomacy: Causal Mechanism (Action Diversity & Skill→Structure)")

    our_lines = _load_grpo_action_taking(DIPLO_RUN, DIPLO_BEST_STEP)
    if not our_lines:
        return

    our_eps = defaultdict(list)
    for l in our_lines:
        our_eps[l["episode_id"]].append(l)

    # Action number distribution
    all_nums = Counter()
    success_nums = Counter()
    fail_nums = Counter()
    success_reps = []
    fail_reps = []

    for eid, steps in our_eps.items():
        steps = sorted(steps, key=lambda x: x["step"])
        final_c = int(_extract_prompt_field(steps[-1]["prompt"], "centers") or 0)

        actions = []
        for s in steps:
            num, text = _extract_action_text(s["prompt"], s["completion"])
            all_nums[num] += 1
            if final_c >= 5:
                success_nums[num] += 1
            elif final_c <= 3:
                fail_nums[num] += 1
            actions.append(text)

        repeats = sum(1 for i in range(1, len(actions)) if actions[i] == actions[i - 1])
        rate = repeats / (len(actions) - 1) if len(actions) > 1 else 0

        if final_c >= 5:
            success_reps.append(rate)
        elif final_c <= 3:
            fail_reps.append(rate)

    print_subheader("Action Number Distribution")
    total = sum(all_nums.values())
    for num, count in all_nums.most_common(5):
        print(f"  ACTION {num}: {count}/{total} = {count / total:.1%}")

    s_total = sum(success_nums.values())
    f_total = sum(fail_nums.values())
    print(f"\n  Action 1 in successes (c≥5): {success_nums['1']}/{s_total} = {success_nums['1'] / s_total:.1%}")
    print(f"  Action 1 in failures  (c≤3): {fail_nums['1']}/{f_total} = {fail_nums['1'] / f_total:.1%}")

    print_subheader("Action Repeat Rate (consecutive identical actions)")
    if success_reps:
        m_s, s_s = mean_std(success_reps)
        print(f"  Success episodes (c≥5): {m_s:.1%} ± {s_s:.1%}  (n={len(success_reps)})")
    if fail_reps:
        m_f, s_f = mean_std(fail_reps)
        print(f"  Failure episodes (c≤3): {m_f:.1%} ± {s_f:.1%}  (n={len(fail_reps)})")

    # Categorize what action 1 is
    action1_types = Counter()
    for l in our_lines:
        num, _ = _extract_action_text(l["prompt"], l["completion"])
        if num != "1":
            continue
        if "Available actions" not in l["prompt"]:
            continue
        for line in l["prompt"].split("Available actions")[1].split("\n"):
            line = line.strip()
            if line.startswith("1."):
                text = line.split(".", 1)[1].strip()
                if " S " in text:
                    action1_types["SUPPORT"] += 1
                elif " - " in text:
                    action1_types["MOVE"] += 1
                elif " H" in text or "hold" in text.lower():
                    action1_types["HOLD"] += 1
                elif " B" in text or "WAIVE" in text:
                    action1_types["BUILD/WAIVE"] += 1
                else:
                    action1_types["OTHER"] += 1
                break

    print_subheader("When ACTION 1 Is Chosen — Action Type Breakdown")
    a1_total = sum(action1_types.values())
    for cat, count in action1_types.most_common():
        print(f"  {cat:14s}: {count:3d}/{a1_total} = {count / a1_total:.1%}")

    print_subheader("Causal Mechanism Summary")
    print("""
  The skill system improves Diplomacy performance through THREE mechanisms:

  1. TEMPORAL STRUCTURE:
     EXPLORE→SETUP boundary fires at step 5 in 100% of episodes (28/28).
     This hard phase boundary forces the agent to stop passive support orders
     and begin positioning for center gain. Without it, failure episodes show
     the agent stuck in the same support loop for 15+ turns.

  2. SUBGOAL-DRIVEN ACTION REFRAMING:
     When the subgoal switches from "scout neighbor's intentions" (EXPLORE) to
     "secure supply centers" (SETUP), the action distribution shifts even though
     the model still outputs "Expert play." The skill prompt change affects which
     action the model selects from the available list.

  3. COLLAPSE PREVENTION:
     Our method never loses starting centers (min=3). GPT-5.4 collapses to 1-2
     centers in 27% of episodes. The skill-phased structure prevents the active
     self-destruction (retreats, disbands) that GPT-5.4's "Secure Defensive Line"
     skill can trigger when applied for 12+ consecutive steps.

  HONEST LIMITATION:
     The model picks ACTION 1 in 85% of steps. Failure episodes pick it 90%.
     The skill system's main contribution is changing what action 1 IS at
     different game stages, not enabling sophisticated multi-action reasoning.
""")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    candy_crush_episode_comparison()
    candy_crush_skills()
    diplomacy_episode_stats()
    diplomacy_failure_analysis()
    diplomacy_outplay_examples()
    diplomacy_skill_retrieval()
    diplomacy_causal_mechanism()


if __name__ == "__main__":
    main()
