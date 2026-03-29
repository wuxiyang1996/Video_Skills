"""Shared utilities for paper analysis scripts."""

import json
import os
import re
from collections import defaultdict, Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# ─── Paths ────────────────────────────────────────────────────────────────────

GAME_AI_AGENT = "/workspace/game_agent/Game-AI-Agent"
MULTI_HOP = "/workspace/game_agent/Multi-hop-Reasoning-VLM-Agent"
LABELED_BASE = f"{GAME_AI_AGENT}/labeling/output/gpt54_skill_labeled"
ABLATION_BASE = f"{GAME_AI_AGENT}/ablation_study/output"
BASELINES_BASE = f"{GAME_AI_AGENT}/baselines/output"

LABELED_GAMES = [
    "candy_crush", "tetris", "avalon", "diplomacy",
    "super_mario", "twenty_forty_eight", "sokoban", "pokemon_red",
]

RUNS = {
    "2048": {
        "base": f"{MULTI_HOP}/runs/Qwen3-8B_2048_20260322_071227",
        "game_dirs": ["twenty_forty_eight"],
        "best_step": 9,
    },
    "Candy Crush": {
        "base": f"{GAME_AI_AGENT}/runs/Qwen3-8B_20260321_213813_(Candy_crush)",
        "game_dirs": ["candy_crush"],
        "best_step": 9,
    },
    "Avalon": {
        "base": f"{GAME_AI_AGENT}/runs/Qwen3-8B_avalon_20260322_200424",
        "game_dirs": ["avalon/good", "avalon/evil"],
        "best_step": 5,
        "no_checkpoints": True,
    },
    "Tetris": {
        "base": f"{GAME_AI_AGENT}/runs/Qwen3-8B_tetris_20260322_170438",
        "game_dirs": ["tetris"],
        "best_step": 12,
    },
    "Diplomacy": {
        "base": f"{GAME_AI_AGENT}/runs/Qwen3-8B_diplomacy_20260322_234548",
        "game_dirs": [f"diplomacy/{n}" for n in
                      ["AUSTRIA", "ENGLAND", "FRANCE", "GERMANY", "ITALY", "RUSSIA", "TURKEY"]],
        "best_step": 22,
    },
    "Super Mario": {
        "base": f"{GAME_AI_AGENT}/runs/Qwen3-8B_super_mario_20260323_030839",
        "game_dirs": ["super_mario"],
        "best_step": 11,
    },
}

# Ablation condition pattern → human-readable label
ABLATION_CONDITION_PATTERNS = [
    ("with_bank_",       "RL + Bank"),
    ("with_skillbank_",  "RL + Bank"),
    ("sft_best_bank_",   "SFT + Best Bank"),
    ("sft_first_bank_",  "SFT + First Bank"),
    ("sft_no_bank_",     "SFT (no bank)"),
    ("sft_no_skillbank_","SFT (no bank)"),
    ("no_bank_",         "RL (no bank)"),
    ("no_skillbank_",    "RL (no bank)"),
    ("base_model_",      "Base Model"),
]


def condition_label(dirname: str) -> str:
    for pat, label in ABLATION_CONDITION_PATTERNS:
        if dirname.startswith(pat):
            return label
    return "Unknown"


def detect_game_from_dir(dirname: str) -> str:
    """Heuristic to detect game name from ablation directory name."""
    dirname_lower = dirname.lower()
    if "diplomacy" in dirname_lower:
        return "diplomacy"
    if "avalon" in dirname_lower:
        return "avalon"
    if "super_mario" in dirname_lower:
        return "super_mario"
    if "2048" in dirname_lower or "twenty" in dirname_lower:
        return "2048"
    if "tetris" in dirname_lower:
        return "tetris"
    if "candy" in dirname_lower:
        return "candy_crush"
    if "sokoban" in dirname_lower:
        return "sokoban"
    if "pokemon" in dirname_lower:
        return "pokemon_red"
    return "mixed"


# ─── Episode loading ──────────────────────────────────────────────────────────

def load_episode(filepath: str) -> Optional[dict]:
    try:
        with open(filepath) as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return None


def iter_episodes(directory: str):
    """Yield (filepath, episode_dict) for all episode_*.json under directory."""
    if not os.path.exists(directory):
        return
    for root, _, files in os.walk(directory):
        for f in sorted(files):
            if f.startswith("episode_") and f.endswith(".json"):
                ep = load_episode(os.path.join(root, f))
                if ep:
                    yield os.path.join(root, f), ep


def load_all_labeled_episodes(game: str) -> List[dict]:
    gdir = os.path.join(LABELED_BASE, game)
    return [ep for _, ep in iter_episodes(gdir)]


def load_ablation_episodes(condition_prefix: str, game_filter: Optional[str] = None):
    """Load episodes from ablation conditions matching prefix."""
    results = []
    if not os.path.exists(ABLATION_BASE):
        return results
    for d in sorted(os.listdir(ABLATION_BASE)):
        if not d.startswith(condition_prefix):
            continue
        if game_filter and game_filter not in d.lower():
            continue
        dpath = os.path.join(ABLATION_BASE, d)
        if not os.path.isdir(dpath):
            continue
        for fp, ep in iter_episodes(dpath):
            ep["_condition_dir"] = d
            ep["_condition_label"] = condition_label(d)
            ep["_game_detected"] = detect_game_from_dir(d)
            results.append(ep)
    return results


def load_all_ablation_episodes():
    """Load all ablation episodes grouped by (condition_label, game)."""
    grouped = defaultdict(list)
    if not os.path.exists(ABLATION_BASE):
        return grouped
    for d in sorted(os.listdir(ABLATION_BASE)):
        dpath = os.path.join(ABLATION_BASE, d)
        if not os.path.isdir(dpath):
            continue
        label = condition_label(d)
        game = detect_game_from_dir(d)
        for fp, ep in iter_episodes(dpath):
            ep["_condition_label"] = label
            ep["_game_detected"] = game
            grouped[(label, game)].append(ep)
    return grouped


# ─── Skill bank loading ──────────────────────────────────────────────────────

def parse_skill_bank(filepath: str) -> List[dict]:
    skills = []
    if not os.path.exists(filepath):
        return skills
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "skill" in entry:
                sk = entry["skill"]
                sk["_report"] = entry.get("report")
                skills.append(sk)
            elif "contract" in entry:
                skills.append({
                    "skill_id": entry["contract"].get("skill_id", "unknown"),
                    "contract": entry["contract"],
                    "_report": entry.get("report"),
                })
    return skills


def load_checkpoint_bank(run_base: str, step: int, game_dirs: List[str]) -> List[dict]:
    all_skills = []
    for gd in game_dirs:
        fp = os.path.join(run_base, "checkpoints", f"step_{step:04d}", "banks", gd, "skill_bank.jsonl")
        all_skills.extend(parse_skill_bank(fp))
    return all_skills


def load_final_bank(run_base: str, game_dirs: List[str]) -> List[dict]:
    all_skills = []
    for gd in game_dirs:
        fp = os.path.join(run_base, "skillbank", gd, "skill_bank.jsonl")
        all_skills.extend(parse_skill_bank(fp))
    return all_skills


def get_checkpoint_steps(run_base: str) -> List[int]:
    ckpt_dir = os.path.join(run_base, "checkpoints")
    if not os.path.exists(ckpt_dir):
        return []
    steps = []
    for name in os.listdir(ckpt_dir):
        m = re.match(r"step_(\d+)", name)
        if m:
            step_num = int(m.group(1))
            if step_num != 99999:
                steps.append(step_num)
    return sorted(steps)


# ─── Step log / rewards ──────────────────────────────────────────────────────

def load_step_log(run_base: str) -> List[dict]:
    fp = os.path.join(run_base, "step_log.jsonl")
    if not os.path.exists(fp):
        return []
    entries = []
    with open(fp) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return entries


def load_rewards_at_step(run_base: str, step: int) -> List[dict]:
    fp = os.path.join(run_base, "rewards", f"step_{step:04d}.jsonl")
    if not os.path.exists(fp):
        return []
    entries = []
    with open(fp) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return entries


# ─── Skill ID parsing ────────────────────────────────────────────────────────

def parse_skill_id(skill_id: str) -> Tuple[str, str]:
    """Split 'phase:ACTION' → (phase, action). Plain 'ACTION' → ('', action)."""
    if ":" in skill_id:
        parts = skill_id.split(":", 1)
        return parts[0], parts[1]
    return "", skill_id


def categorize_skill(skill_id: str) -> str:
    _, action = parse_skill_id(skill_id)
    return action


# ─── Summary state parsing ───────────────────────────────────────────────────

def parse_summary_state(summary_state: str) -> Dict[str, str]:
    """Parse 'key=val | key=val' summary_state into dict."""
    if not summary_state:
        return {}
    result = {}
    for part in summary_state.split("|"):
        part = part.strip()
        if "=" in part:
            k, v = part.split("=", 1)
            result[k.strip()] = v.strip()
    return result


# ─── Intention tag parsing ────────────────────────────────────────────────────

def extract_intention_tag(intentions: str) -> Optional[str]:
    """Extract [TAG] from intention string like '[ATTACK] Do something'."""
    if not intentions:
        return None
    m = re.match(r"\[([A-Z_]+)\]", intentions.strip())
    return m.group(1) if m else None


# ─── Statistics ───────────────────────────────────────────────────────────────

def wilson_ci(successes: int, total: int, z: float = 1.96) -> Tuple[float, float, float]:
    """Wilson score interval for a binomial proportion."""
    if total == 0:
        return 0.0, 0.0, 0.0
    p = successes / total
    denom = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / denom
    spread = z * ((p * (1 - p) / total + z**2 / (4 * total**2)) ** 0.5) / denom
    return p, max(0, center - spread), min(1, center + spread)


def mean_std(values):
    if not values:
        return 0.0, 0.0
    m = sum(values) / len(values)
    if len(values) < 2:
        return m, 0.0
    var = sum((x - m) ** 2 for x in values) / (len(values) - 1)
    return m, var ** 0.5


def gini_coefficient(values):
    if not values or sum(values) == 0:
        return 0.0
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    cumulative = sum((2 * (i + 1) - n - 1) * v for i, v in enumerate(sorted_vals))
    return cumulative / (n * sum(sorted_vals))


# ─── Display helpers ──────────────────────────────────────────────────────────

def print_header(title: str, width: int = 90):
    print(f"\n{'=' * width}")
    print(f"  {title}")
    print(f"{'=' * width}")


def print_subheader(title: str, width: int = 80):
    print(f"\n  {'─' * width}")
    print(f"  {title}")
    print(f"  {'─' * width}")


def episode_total_reward(ep: dict) -> float:
    meta = ep.get("metadata", {})
    if isinstance(meta, dict) and "total_reward" in meta:
        return meta["total_reward"]
    return sum(e.get("reward", 0) for e in ep.get("experiences", []))


def episode_game(ep: dict) -> str:
    return ep.get("game_name", ep.get("_game_detected", "unknown"))
