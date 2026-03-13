#!/usr/bin/env python
"""
Cold-start rollout generation for Orak games (Super Mario, StarCraft II)
using ALFWorld-7B checkpoints as a baseline agent (mirrors
generate_cold_start_orak.py).

Output structure (cold_start/output/alfworld7b_orak/<game_name>/):
  - episode_NNN.json        Individual episode (Episode.to_dict())
  - episode_buffer.json     All episodes in Episode_Buffer format
  - rollouts.jsonl          Append-friendly JSONL (one Episode per line)
  - rollout_summary.json    Per-game run stats

Usage (from Game-AI-Agent root):

    # Super Mario (activate orak-mario env first)
    source evaluate_orak/setup_orak_mario.sh
    python cold_start/ALFWORLD-7B/generate_cold_start_orak_alfworld7b.py \\
        --model_path Jianwen/Alfworld-7B-SFT --games super_mario --episodes 10

    # RL baseline
    python cold_start/ALFWORLD-7B/generate_cold_start_orak_alfworld7b.py \\
        --model_path Jianwen/Alfworld-7B-RL --checkpoint_type rl \\
        --games super_mario --episodes 10
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent          # cold_start/ALFWORLD-7B/
COLD_START_DIR = SCRIPT_DIR.parent                     # cold_start/
CODEBASE_ROOT = COLD_START_DIR.parent                  # Game-AI-Agent/
ORAK_SRC = CODEBASE_ROOT.parent / "Orak" / "src"

for _p in [str(CODEBASE_ROOT), str(ORAK_SRC), str(SCRIPT_DIR)]:
    if Path(_p).exists() and _p not in sys.path:
        sys.path.insert(0, _p)

logging.getLogger("sc2").setLevel(logging.WARNING)
try:
    import loguru
    loguru.logger.disable("sc2")
except Exception:
    pass

from evaluate_orak.orak_nl_wrapper import make_orak_env, ORAK_GAMES  # type: ignore
from data_structure.experience import Experience, Episode, Episode_Buffer  # type: ignore
from policy_alfworld7b import Alfworld7BConfig, Alfworld7BPolicy  # type: ignore

try:
    from cold_start.generate_cold_start import label_trajectory  # type: ignore
except ImportError:
    label_trajectory = None

ORAK_COLD_START_GAMES: Dict[str, Dict[str, Any]] = {
    "super_mario": {
        "max_steps": 100,
        "display_name": "Super Mario Bros",
        "task": ORAK_GAMES["super_mario"]["task"],
    },
    "star_craft": {
        "max_steps": 1000,
        "display_name": "StarCraft II (1v1 Protoss)",
        "task": ORAK_GAMES["star_craft"]["task"],
    },
}

# ---------------------------------------------------------------------------
# Global policy (set in main)
# ---------------------------------------------------------------------------
_POLICY: Optional[Alfworld7BPolicy] = None


# ---------------------------------------------------------------------------
# Agent action helpers
# ---------------------------------------------------------------------------

def alfworld7b_mario_action(
    state_nl: str,
    action_names: List[str],
) -> Tuple[str, Optional[str]]:
    prompt = (
        f"Super Mario game state:\n{state_nl}\n\n"
        f"Valid actions: {', '.join(action_names)}\n\n"
        "Choose one action."
    )
    action = _POLICY.choose_action(prompt, action_names)
    return action, None


def alfworld7b_sc2_action(
    state_nl: str,
    action_names: List[str],
) -> Tuple[str, Optional[str]]:
    default = "EMPTY ACTION"
    prompt = (
        f"StarCraft II game state:\n{state_nl}\n\n"
        f"Valid actions: {', '.join(action_names)}\n\n"
        "Choose exactly 5 sequential macro actions."
    )
    acts = []
    remaining = list(action_names)
    for _ in range(5):
        if not remaining:
            remaining = [default]
        a = _POLICY.choose_action(prompt, remaining)
        acts.append(a)
    action_str = "\n".join(f"{i}: {a}" for i, a in enumerate(acts, 1))
    return action_str, None


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_orak_episode(
    game_name: str,
    max_steps: int = 100,
    verbose: bool = False,
) -> Tuple[Episode, Dict[str, Any]]:
    cfg = ORAK_COLD_START_GAMES[game_name]
    task = cfg["task"]
    is_sc2 = game_name in ("star_craft", "star_craft_multi")
    agent_fn = alfworld7b_sc2_action if is_sc2 else alfworld7b_mario_action

    env = make_orak_env(game_name, max_steps=max_steps)
    action_names = env.action_names
    obs, info = env.reset()

    experiences: List[Experience] = []
    total_reward = 0.0
    step_count = 0
    terminated = False
    truncated = False

    while step_count < max_steps:
        action, _ = agent_fn(state_nl=obs, action_names=action_names)

        next_obs, reward, terminated, truncated, next_info = env.step(action)
        done = terminated or truncated
        total_reward += reward

        exp = Experience(
            state=obs,
            action=str(action),
            reward=float(reward),
            next_state=next_obs,
            done=done,
            intentions=None,
            tasks=task,
        )
        exp.idx = step_count
        exp.action_type = "primitive"
        exp.raw_state = obs
        exp.raw_next_state = next_obs
        exp.available_actions = list(action_names)

        step_info = next_info if isinstance(next_info, dict) else {}
        exp.interface = {
            "env_name": "orak",
            "game_name": game_name,
            "step": step_count,
            "terminated": terminated,
            "truncated": truncated,
            "score": step_info.get("score"),
            "cumulative_reward": total_reward,
        }
        experiences.append(exp)

        if verbose:
            act_short = str(action)[:60]
            term_label = "TERM" if terminated else ("TRUNC" if truncated else "")
            print(f"  step {step_count}: action={act_short}, reward={reward:.3f}, "
                  f"cum={total_reward:.3f} {term_label}")

        obs = next_obs
        step_count += 1
        if done:
            break

    env.close()

    episode = Episode(experiences=experiences, task=task, env_name="orak", game_name=game_name)
    episode.set_outcome()

    final_info = next_info if (step_count > 0 and isinstance(next_info, dict)) else {}
    stats = {
        "game": game_name,
        "display_name": cfg["display_name"],
        "steps": step_count,
        "total_reward": total_reward,
        "final_score": final_info.get("score", 0),
        "terminated": terminated,
        "truncated": truncated,
        "model": _POLICY.cfg.model_path,
        "checkpoint_type": _POLICY.cfg.checkpoint_type,
        "agent_type": "alfworld7b_orak",
    }
    return episode, stats


# ---------------------------------------------------------------------------
# Batch rollout helpers
# ---------------------------------------------------------------------------

def count_existing_episodes(game_dir: Path) -> int:
    if not game_dir.exists():
        return 0
    return sum(1 for f in game_dir.glob("episode_*.json") if f.name != "episode_buffer.json")


def save_episode_jsonl(episode: Episode, jsonl_path: Path, stats: Dict[str, Any]):
    record = episode.to_dict()
    record["rollout_metadata"] = stats
    with open(jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def save_game_summary(
    game_name: str,
    game_dir: Path,
    all_stats: List[Dict[str, Any]],
    elapsed: float,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    good = [s for s in all_stats if "error" not in s]
    summary: Dict[str, Any] = {
        "game": game_name,
        "display_name": ORAK_COLD_START_GAMES[game_name]["display_name"],
        "timestamp": datetime.now().isoformat(),
        "model": args.model_path,
        "checkpoint_type": args.checkpoint_type,
        "agent_type": "alfworld7b_orak",
        "total_episodes": len(good),
        "target_episodes": args.episodes,
        "max_steps": args.max_steps,
        "labeled": args.label and not args.no_label,
        "elapsed_seconds": round(elapsed, 2),
        "episode_stats": all_stats,
    }
    if good:
        rewards = [s["total_reward"] for s in good]
        steps = [s["steps"] for s in good]
        summary["mean_reward"] = sum(rewards) / len(rewards)
        summary["mean_steps"] = sum(steps) / len(steps)
        summary["max_reward"] = max(rewards)
        summary["min_reward"] = min(rewards)

    summary_path = game_dir / "rollout_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    return summary


def run_game_rollouts(
    game_name: str,
    args: argparse.Namespace,
    output_dir: Path,
) -> Dict[str, Any]:
    game_dir = output_dir / game_name
    game_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = game_dir / "rollouts.jsonl"

    max_steps = args.max_steps or ORAK_COLD_START_GAMES[game_name]["max_steps"]

    start_idx = 0
    if args.resume:
        start_idx = count_existing_episodes(game_dir)
        if start_idx >= args.episodes:
            print(f"  [SKIP] {game_name}: {start_idx}/{args.episodes} episodes already done")
            return {"game": game_name, "skipped": True, "existing": start_idx}
        if start_idx > 0:
            print(f"  [RESUME] {game_name}: resuming from episode {start_idx}")

    episode_buffer = Episode_Buffer(buffer_size=args.episodes + 10)
    all_stats: List[Dict[str, Any]] = []
    t0 = time.time()

    for ep_idx in range(start_idx, args.episodes):
        display = ORAK_COLD_START_GAMES[game_name]["display_name"]
        print(f"\n  [{display}] Episode {ep_idx + 1}/{args.episodes}")

        try:
            episode, stats = run_orak_episode(
                game_name=game_name,
                max_steps=max_steps,
                verbose=args.verbose,
            )
            stats["episode_index"] = ep_idx
            print(f"    Steps: {stats['steps']}, Reward: {stats['total_reward']:.3f}")

            if args.label and not args.no_label and label_trajectory is not None:
                episode = label_trajectory(episode, args.label_model)

            episode_buffer.add_episode(episode)
            all_stats.append(stats)

            ep_data = episode.to_dict()
            ep_data["metadata"] = stats
            ep_path = game_dir / f"episode_{ep_idx:03d}.json"
            with open(ep_path, "w", encoding="utf-8") as f:
                json.dump(ep_data, f, indent=2, ensure_ascii=False, default=str)

            save_episode_jsonl(episode, jsonl_path, stats)

        except Exception as e:
            print(f"    [ERROR] Episode {ep_idx + 1} failed: {e}")
            traceback.print_exc()
            all_stats.append({
                "game": game_name, "episode_index": ep_idx,
                "error": str(e), "steps": 0, "total_reward": 0.0,
            })
            continue

    elapsed = time.time() - t0

    buffer_path = game_dir / "episode_buffer.json"
    episode_buffer.save_to_json(str(buffer_path))
    print(f"\n  Saved {len(episode_buffer)} episodes to {buffer_path}")

    summary = save_game_summary(game_name, game_dir, all_stats, elapsed, args)
    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global _POLICY

    parser = argparse.ArgumentParser(
        description="ALFWorld-7B cold-start rollouts for Orak games (Super Mario, StarCraft II)",
    )
    parser.add_argument("--games", type=str, nargs="+", default=None,
                        choices=list(ORAK_COLD_START_GAMES.keys()),
                        help="Orak games to run (default: all)")
    parser.add_argument("--episodes", type=int, default=10,
                        help="Number of episodes per game (default: 10)")
    parser.add_argument("--max_steps", type=int, default=None,
                        help="Max steps per episode (None = game default: Mario=100, SC2=1000)")
    parser.add_argument("--model_path", type=str, required=True,
                        help="HF model id or local path for ALFWorld-7B.")
    parser.add_argument("--checkpoint_type", type=str, default="sft",
                        choices=["sft", "rl"],
                        help="Checkpoint type: 'sft' or 'rl'.")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="Sampling temperature (default: 0.8)")
    parser.add_argument("--label", action="store_true",
                        help="Label trajectories with LLM.")
    parser.add_argument("--no_label", action="store_true",
                        help="Skip trajectory labeling.")
    parser.add_argument("--label_model", type=str, default="gpt-5-mini")
    parser.add_argument("--resume", action="store_true",
                        help="Resume interrupted run.")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: cold_start/output/alfworld7b_orak)")

    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else COLD_START_DIR / "output" / f"alfworld7b_{args.checkpoint_type}_orak"
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = Alfworld7BConfig(
        model_path=args.model_path,
        checkpoint_type=args.checkpoint_type,
        temperature=args.temperature,
    )
    _POLICY = Alfworld7BPolicy(cfg)

    requested = args.games or list(ORAK_COLD_START_GAMES.keys())

    print("=" * 78)
    print("  ALFWorld-7B Cold-Start — Orak Games (Super Mario & StarCraft II)")
    print("=" * 78)
    print(f"  Games:       {', '.join(requested)}")
    print(f"  Episodes:    {args.episodes} per game")
    print(f"  Max steps:   {args.max_steps or 'per-game default'}")
    print(f"  Model:       {args.model_path} ({args.checkpoint_type})")
    print(f"  Temperature: {args.temperature}")
    print(f"  Labeling:    {args.label and not args.no_label}")
    print(f"  Resume:      {args.resume}")
    print(f"  Output:      {output_dir}")
    print("=" * 78)

    overall_t0 = time.time()
    game_summaries: List[Dict[str, Any]] = []

    for game_name in requested:
        display = ORAK_COLD_START_GAMES[game_name]["display_name"]
        print(f"\n{'━' * 78}")
        print(f"  GAME: {display} ({args.episodes} episodes)")
        print(f"{'━' * 78}")
        summary = run_game_rollouts(game_name, args, output_dir)
        game_summaries.append(summary)

    overall_elapsed = time.time() - overall_t0

    master_summary = {
        "timestamp": datetime.now().isoformat(),
        "source": "orak",
        "model": args.model_path,
        "checkpoint_type": args.checkpoint_type,
        "agent_type": "alfworld7b_orak",
        "episodes_per_game": args.episodes,
        "max_steps": args.max_steps,
        "temperature": args.temperature,
        "labeled": args.label and not args.no_label,
        "total_elapsed_seconds": round(overall_elapsed, 2),
        "games_completed": requested,
        "per_game_summaries": game_summaries,
    }
    master_path = output_dir / "batch_rollout_summary.json"
    with open(master_path, "w", encoding="utf-8") as f:
        json.dump(master_summary, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n{'=' * 78}")
    print("  ALFWorld-7B ORAK COLD-START COMPLETE")
    print(f"{'=' * 78}")
    total_eps = sum(
        s.get("total_episodes", 0) for s in game_summaries if not s.get("skipped")
    )
    print(f"  Games:          {len(requested)}")
    print(f"  Total episodes: {total_eps}")
    print(f"  Elapsed:        {overall_elapsed:.1f}s")
    print(f"  Output:         {output_dir}")
    print(f"  Summary:        {master_path}")
    print(f"{'=' * 78}\n")


if __name__ == "__main__":
    main()
