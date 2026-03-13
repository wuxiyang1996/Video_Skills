#!/usr/bin/env python
"""
Cold-start base agent for LM-Game Bench using ALFWorld-7B checkpoints.

This mirrors `generate_cold_start_gpt54.py` but replaces GPT-5.4 API calls
with a local HF checkpoint (Alfworld-7B-SFT or Alfworld-7B-RL) loaded via
`transformers`. The environment side (GamingAgent) is unchanged.

Output structure (cold_start/output/alfworld7b_lmgame/<game_name>/):
  - episode_NNN.json        Individual episode (Episode.to_dict())
  - episode_buffer.json     All episodes in Episode_Buffer format
  - rollouts.jsonl          Append-friendly JSONL (one Episode per line)
  - rollout_summary.json    Per-game run stats

Usage (from Game-AI-Agent root):

    # SFT baseline
    python cold_start/ALFWORLD-7B/generate_cold_start_lmgame_alfworld7b.py \\
        --model_path Jianwen/Alfworld-7B-SFT

    # RL baseline
    python cold_start/ALFWORLD-7B/generate_cold_start_lmgame_alfworld7b.py \\
        --model_path Jianwen/Alfworld-7B-RL --checkpoint_type rl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent          # cold_start/ALFWORLD-7B/
COLD_START_DIR = SCRIPT_DIR.parent                     # cold_start/
CODEBASE_ROOT = COLD_START_DIR.parent                  # Game-AI-Agent/
GAMINGAGENT_ROOT = CODEBASE_ROOT.parent / "GamingAgent"

for p in [str(CODEBASE_ROOT), str(GAMINGAGENT_ROOT), str(SCRIPT_DIR)]:
    if Path(p).exists() and p not in sys.path:
        sys.path.insert(0, p)

from data_structure.experience import Experience, Episode, Episode_Buffer  # type: ignore

from cold_start.generate_cold_start import (  # type: ignore
    GAME_REGISTRY,
    ColdStartEnvWrapper,
    get_cold_start_max_steps,
    label_trajectory,
)

from policy_alfworld7b import Alfworld7BConfig, Alfworld7BPolicy  # type: ignore


def run_alfworld7b_episode(
    env: ColdStartEnvWrapper,
    game_name: str,
    policy: Alfworld7BPolicy,
    max_steps: int = 50,
    verbose: bool = False,
) -> Tuple[Episode, Dict[str, Any]]:
    """
    Run one LM-Game Bench episode with the ALFWorld-7B policy as the agent.
    """
    task = GAME_REGISTRY[game_name]["task"]
    action_names = GAME_REGISTRY[game_name]["action_names"]

    obs, info = env.reset()
    raw_obs = info.get("raw_obs")
    curr_available_actions = info.get("available_actions") or action_names
    experiences: List[Experience] = []
    total_reward = 0.0
    step_count = 0
    terminated = False
    truncated = False

    while step_count < max_steps:
        step_actions = curr_available_actions if curr_available_actions else action_names
        prompt_state = obs + f"\n\nValid actions: {', '.join(step_actions)}. Choose one."

        action = policy.choose_action(prompt_state, list(step_actions))

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
        exp.raw_state = str(raw_obs) if raw_obs is not None else None
        next_raw_obs = next_info.get("raw_obs")
        exp.raw_next_state = str(next_raw_obs) if next_raw_obs is not None else None
        exp.available_actions = list(step_actions) if step_actions else None
        exp.interface = {"env_name": "gamingagent", "game_name": game_name}
        experiences.append(exp)

        if verbose:
            print(
                f"  step {step_count}: action={action}, "
                f"reward={reward:.2f}, cum={total_reward:.2f}"
            )

        obs = next_obs
        raw_obs = next_raw_obs
        curr_available_actions = next_info.get("available_actions") or action_names
        step_count += 1

        if done:
            break

    episode = Episode(
        experiences=experiences,
        task=task,
        env_name="gamingagent",
        game_name=game_name,
    )
    episode.set_outcome()

    stats = {
        "game": game_name,
        "steps": step_count,
        "total_reward": total_reward,
        "terminated": terminated,
        "truncated": truncated,
        "model": policy.cfg.model_path,
        "checkpoint_type": policy.cfg.checkpoint_type,
        "agent_type": "alfworld7b_lmgame",
    }
    return episode, stats


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
    max_steps_used: Optional[int] = None,
) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "game": game_name,
        "timestamp": datetime.now().isoformat(),
        "model": args.model_path,
        "checkpoint_type": args.checkpoint_type,
        "agent_type": "alfworld7b_lmgame",
        "total_episodes": len(all_stats),
        "target_episodes": args.episodes,
        "max_steps": max_steps_used if max_steps_used is not None else args.max_steps,
        "labeled": args.label and not args.no_label,
        "elapsed_seconds": elapsed,
        "episode_stats": all_stats,
    }
    if all_stats:
        rewards = [s["total_reward"] for s in all_stats]
        steps = [s["steps"] for s in all_stats]
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
    policy: Alfworld7BPolicy,
) -> Dict[str, Any]:
    game_dir = output_dir / game_name
    game_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = game_dir / "rollouts.jsonl"

    start_idx = 0
    if args.resume:
        start_idx = count_existing_episodes(game_dir)
        if start_idx >= args.episodes:
            print(f"  [SKIP] {game_name}: {start_idx}/{args.episodes} episodes already done")
            return {"game": game_name, "skipped": True, "existing": start_idx}
        if start_idx > 0:
            print(f"  [RESUME] {game_name}: resuming from episode {start_idx}")

    effective_max_steps = (
        args.max_steps
        if args.max_steps is not None
        else get_cold_start_max_steps(game_name)
    )

    episode_buffer = Episode_Buffer(buffer_size=args.episodes + 10)
    all_stats: List[Dict[str, Any]] = []
    t0 = time.time()

    for ep_idx in range(start_idx, args.episodes):
        print(f"\n  [{game_name}] Episode {ep_idx + 1}/{args.episodes}")

        try:
            env = ColdStartEnvWrapper(game_name, max_steps=effective_max_steps)
            episode, stats = run_alfworld7b_episode(
                env=env,
                game_name=game_name,
                policy=policy,
                max_steps=effective_max_steps,
                verbose=args.verbose,
            )
            env.close()

            stats["episode_index"] = ep_idx
            print(f"    Steps: {stats['steps']}, Reward: {stats['total_reward']:.2f}")

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
            all_stats.append(
                {
                    "game": game_name,
                    "episode_index": ep_idx,
                    "error": str(e),
                    "steps": 0,
                    "total_reward": 0.0,
                }
            )
            continue

    elapsed = time.time() - t0

    buffer_path = game_dir / "episode_buffer.json"
    episode_buffer.save_to_json(str(buffer_path))
    print(f"\n  Saved {len(episode_buffer)} episodes to {buffer_path}")

    summary = save_game_summary(game_name, game_dir, all_stats, elapsed, args, max_steps_used=effective_max_steps)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ALFWorld-7B base agent cold-start rollouts for LM-Game Bench",
    )
    parser.add_argument(
        "--games",
        type=str,
        nargs="+",
        default=None,
        help="Games to generate rollouts for (default: all available in GAME_REGISTRY)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
        help="Number of episodes per game (default: 100)",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Max steps per episode (default: per-game natural end)",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="HF model id or local path for ALFWorld-7B (SFT or RL).",
    )
    parser.add_argument(
        "--checkpoint_type",
        type=str,
        default="sft",
        choices=["sft", "rl"],
        help="Checkpoint type: 'sft' or 'rl' (for logging only).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature for ALFWorld-7B (default: 0.8)",
    )
    parser.add_argument(
        "--label",
        action="store_true",
        help="Label trajectories with LLM (if label_trajectory is available).",
    )
    parser.add_argument(
        "--no_label",
        action="store_true",
        help="Skip trajectory labeling.",
    )
    parser.add_argument(
        "--label_model",
        type=str,
        default="gpt-5-mini",
        help="Model used for trajectory labeling (default: gpt-5-mini).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume interrupted run (skip completed episodes).",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print step-by-step details.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (default: cold_start/output/alfworld7b_lmgame)",
    )

    args = parser.parse_args()

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else COLD_START_DIR / "output" / f"alfworld7b_{args.checkpoint_type}_lmgame"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = Alfworld7BConfig(
        model_path=args.model_path,
        checkpoint_type=args.checkpoint_type,
        temperature=args.temperature,
    )
    policy = Alfworld7BPolicy(cfg)

    if args.games:
        requested = args.games
    else:
        requested = list(GAME_REGISTRY.keys())

    available_games: List[str] = []
    skipped_games: List[str] = []
    for g in requested:
        if g not in GAME_REGISTRY:
            print(f"[WARNING] Game '{g}' not in registry, skipping.")
            skipped_games.append(g)
            continue
        if GAME_REGISTRY[g]["env_class"] is None:
            print(f"[WARNING] Game '{g}' env class not importable, skipping.")
            skipped_games.append(g)
            continue
        available_games.append(g)

    if not available_games:
        print("[ERROR] No games available. Ensure GamingAgent is installed.")
        sys.exit(1)

    print("=" * 78)
    print("  ALFWorld-7B Base Agent — LM-Game Bench Cold-Start Rollout Generation")
    print("=" * 78)
    print(f"  Games:       {', '.join(available_games)}")
    if skipped_games:
        print(f"  Skipped:     {', '.join(skipped_games)}")
    print(f"  Episodes:    {args.episodes} per game")
    print(
        f"  Max steps:   "
        f"{'per-game (natural end)' if args.max_steps is None else args.max_steps}"
    )
    print(f"  Model path:  {args.model_path} ({args.checkpoint_type})")
    print(f"  Temperature: {args.temperature}")
    print(f"  Labeling:    {args.label and not args.no_label} (label model: {args.label_model})")
    print(f"  Resume:      {args.resume}")
    print(f"  Output:      {output_dir}")
    print("=" * 78)

    overall_t0 = time.time()
    game_summaries: List[Dict[str, Any]] = []

    for game_name in available_games:
        print(f"\n{'━' * 78}")
        print(f"  GAME: {game_name} ({args.episodes} episodes)")
        print(f"{'━' * 78}")

        summary = run_game_rollouts(game_name, args, output_dir, policy)
        game_summaries.append(summary)

    overall_elapsed = time.time() - overall_t0

    master_summary = {
        "timestamp": datetime.now().isoformat(),
        "model": args.model_path,
        "checkpoint_type": args.checkpoint_type,
        "agent_type": "alfworld7b_lmgame",
        "episodes_per_game": args.episodes,
        "max_steps": args.max_steps,
        "temperature": args.temperature,
        "labeled": args.label and not args.no_label,
        "label_model": args.label_model,
        "total_elapsed_seconds": overall_elapsed,
        "games_completed": list(available_games),
        "games_skipped": skipped_games,
        "per_game_summaries": game_summaries,
    }
    master_path = output_dir / "batch_rollout_summary.json"
    with open(master_path, "w", encoding="utf-8") as f:
        json.dump(master_summary, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n{'=' * 78}")
    print("  ALFWorld-7B LM-Game BASE AGENT — BATCH ROLLOUT COMPLETE")
    print(f"{'=' * 78}")
    print(f"  Games processed: {len(available_games)}")
    total_eps = sum(
        s.get("total_episodes", 0) for s in game_summaries if not s.get("skipped")
    )
    print(f"  Total episodes:  {total_eps}")
    print(f"  Elapsed:         {overall_elapsed:.1f}s")
    print(f"  Output:          {output_dir}")
    print(f"  Master summary:  {master_path}")
    print(f"{'=' * 78}\n")


if __name__ == "__main__":
    main()

