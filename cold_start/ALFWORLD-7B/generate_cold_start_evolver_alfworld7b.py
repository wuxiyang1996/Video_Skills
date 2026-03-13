#!/usr/bin/env python
"""
Cold-start rollout generation for Avalon and Diplomacy using ALFWorld-7B
checkpoints as a baseline agent (mirrors generate_cold_start_evolver.py).

Output structure (cold_start/output/alfworld7b_evolver/<game_name>/):
  - episode_NNN.json        Individual episode (Episode.to_dict())
  - episode_buffer.json     All episodes in Episode_Buffer format
  - rollouts.jsonl          Append-friendly JSONL (one Episode per line)
  - rollout_summary.json    Per-game run stats

Usage (from Game-AI-Agent root):

    export PYTHONPATH="$(pwd):$(pwd)/../AgentEvolver:$PYTHONPATH"

    # SFT baseline, both games
    python cold_start/ALFWORLD-7B/generate_cold_start_evolver_alfworld7b.py \\
        --model_path Jianwen/Alfworld-7B-SFT

    # RL baseline, Avalon only
    python cold_start/ALFWORLD-7B/generate_cold_start_evolver_alfworld7b.py \\
        --model_path Jianwen/Alfworld-7B-RL --checkpoint_type rl --games avalon
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

SCRIPT_DIR = Path(__file__).resolve().parent          # cold_start/ALFWORLD-7B/
COLD_START_DIR = SCRIPT_DIR.parent                     # cold_start/
CODEBASE_ROOT = COLD_START_DIR.parent                  # Game-AI-Agent/
_workspace = CODEBASE_ROOT.parent

for p in [str(CODEBASE_ROOT), str(SCRIPT_DIR)]:
    if Path(p).exists() and p not in sys.path:
        sys.path.insert(0, p)

for _candidate in [_workspace / "AgentEvolver", CODEBASE_ROOT / "AgentEvolver"]:
    if _candidate.exists() and str(_candidate) not in sys.path:
        sys.path.insert(0, str(_candidate))
        break

for _candidate in [_workspace / "AI_Diplomacy", CODEBASE_ROOT / "AI_Diplomacy"]:
    if _candidate.exists() and str(_candidate) not in sys.path:
        sys.path.insert(0, str(_candidate))
        break

from data_structure.experience import Experience, Episode, Episode_Buffer  # type: ignore
from policy_alfworld7b import Alfworld7BConfig, Alfworld7BPolicy  # type: ignore

try:
    from env_wrappers.avalon_nl_wrapper import AvalonNLWrapper  # type: ignore
except ImportError:
    AvalonNLWrapper = None  # type: ignore

try:
    from env_wrappers.diplomacy_nl_wrapper import DiplomacyNLWrapper  # type: ignore
except ImportError:
    DiplomacyNLWrapper = None  # type: ignore

try:
    from cold_start.generate_cold_start import label_trajectory  # type: ignore
except ImportError:
    label_trajectory = None

EVOLVER_GAMES: Dict[str, Dict[str, Any]] = {
    "avalon": {
        "env_class": AvalonNLWrapper,
        "task": "Win a game of Avalon through social deduction, strategic voting, and deception.",
        "env_kwargs": {"num_players": 5},
    },
    "diplomacy": {
        "env_class": DiplomacyNLWrapper,
        "task": "Gain the most supply centres in Diplomacy through strategic orders and alliances.",
        "env_kwargs": {},
    },
}

DIPLOMACY_MAX_PHASES = 20

# ---------------------------------------------------------------------------
# Global policy (set in main, used by agent helpers via thread pool)
# ---------------------------------------------------------------------------
_POLICY: Optional[Alfworld7BPolicy] = None


# ---------------------------------------------------------------------------
# Per-agent action helpers (ALFWorld-7B replaces GPT-5.4)
# ---------------------------------------------------------------------------

AVALON_PHASES = {
    0: {"type": "team_selection", "hint": "Provide comma-separated player IDs for the team."},
    1: {"type": "team_vote", "choices": ["approve", "reject"]},
    2: {"type": "quest_vote", "choices": ["pass", "fail"]},
    3: {"type": "assassination", "hint": "Provide the player ID to assassinate."},
}


def avalon_agent_action(
    state_nl: str,
    phase_id: int = 0,
    num_players: int = 5,
) -> Tuple[str, Optional[str]]:
    phase_info = AVALON_PHASES.get(phase_id, {})
    if phase_info.get("choices"):
        action_names = phase_info["choices"]
    elif phase_id == 0:
        action_names = [", ".join(str(i) for i in combo)
                        for i in range(num_players)
                        for combo in [[i]]]
        action_names = [str(i) for i in range(num_players)]
    elif phase_id == 3:
        action_names = [str(i) for i in range(num_players)]
    else:
        action_names = ["wait"]

    prompt = f"Avalon game state:\n{state_nl}\n\nChoose your action."
    action = _POLICY.choose_action(prompt, action_names)
    return action, None


def diplomacy_agent_action(
    state_nl: str,
    possible_orders: Optional[List[str]] = None,
) -> Tuple[List[str], Optional[str]]:
    if possible_orders:
        action_names = list(possible_orders)
    else:
        action_names = ["HOLD", "WAIVE"]

    prompt = (
        f"Diplomacy game state:\n{state_nl}\n\n"
        "Choose orders for your units. Pick one order at a time."
    )
    orders: List[str] = []
    for _ in range(min(len(action_names), 10)):
        order = _POLICY.choose_action(prompt, action_names)
        orders.append(order)
        if order in action_names:
            action_names.remove(order)
        if not action_names:
            break
    return orders, None


# ---------------------------------------------------------------------------
# Sanitize helper (same as original)
# ---------------------------------------------------------------------------

def _sanitize_keys(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _sanitize_keys(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_keys(v) for v in obj]
    return obj


# ---------------------------------------------------------------------------
# Episode runners
# ---------------------------------------------------------------------------

def run_avalon_episode(
    num_players: int = 5,
    seed: int = 42,
    verbose: bool = False,
) -> Tuple[Episode, Dict[str, Any]]:
    if AvalonNLWrapper is None:
        raise ImportError("AvalonNLWrapper not available. Check AgentEvolver install.")

    task = EVOLVER_GAMES["avalon"]["task"]
    env = AvalonNLWrapper(num_players=num_players, seed=seed)
    obs, info = env.reset()

    experiences: List[Experience] = []
    total_reward = 0.0
    step_count = 0

    while not env.done:
        active = info.get("active_players", [])
        actions: Dict[int, Any] = {}
        phase_id = info.get("phase", 0)

        for pid in active:
            state_nl = obs.get(pid, "")
            if not state_nl:
                continue
            action, _ = avalon_agent_action(state_nl, phase_id, num_players)
            actions[pid] = action
            if verbose:
                print(f"    Player {pid} action={action!r}", flush=True)

        next_obs, rewards, terminated, truncated, next_info = env.step(actions)
        done = terminated or truncated

        reward_val = sum(rewards.values()) if isinstance(rewards, dict) and rewards else 0.0
        total_reward += reward_val

        exp = Experience(
            state=json.dumps({str(k): v for k, v in obs.items()}, ensure_ascii=False, default=str),
            action=json.dumps({str(k): v for k, v in actions.items()}, ensure_ascii=False, default=str),
            reward=float(reward_val),
            next_state=json.dumps(
                {str(k): v for k, v in next_obs.items()}, ensure_ascii=False, default=str
            ) if isinstance(next_obs, dict) else str(next_obs),
            done=done,
            intentions=None,
            tasks=task,
        )
        exp.idx = step_count
        exp.action_type = "primitive"
        exp.raw_state = info
        exp.raw_next_state = next_info
        exp.interface = {"env_name": "avalon", "game_name": "avalon", "num_players": num_players}
        experiences.append(exp)

        phase_name = next_info.get("phase_name", next_info.get("phase", ""))
        print(f"    step {step_count}: phase={phase_name}, players={len(active)}, reward={reward_val:.2f}", flush=True)
        if verbose:
            print(f"    cum_reward={total_reward:.2f}", flush=True)

        obs = next_obs
        info = next_info
        step_count += 1

        if done:
            break

    episode = Episode(experiences=experiences, task=task, env_name="avalon", game_name="avalon")
    episode.set_outcome()

    stats = {
        "game": "avalon",
        "steps": step_count,
        "total_reward": total_reward,
        "terminated": env.done,
        "truncated": False,
        "model": _POLICY.cfg.model_path,
        "checkpoint_type": _POLICY.cfg.checkpoint_type,
        "agent_type": "alfworld7b_evolver",
        "num_players": num_players,
        "seed": seed,
        "good_victory": info.get("good_victory"),
    }
    return episode, stats


def run_diplomacy_episode(
    seed: int = 42,
    verbose: bool = False,
) -> Tuple[Episode, Dict[str, Any]]:
    if DiplomacyNLWrapper is None:
        raise ImportError("DiplomacyNLWrapper not available. Check AI_Diplomacy install.")

    task = EVOLVER_GAMES["diplomacy"]["task"]
    env = DiplomacyNLWrapper(seed=seed, max_phases=DIPLOMACY_MAX_PHASES)
    obs, info = env.reset()

    experiences: List[Experience] = []
    total_reward = 0.0
    step_count = 0
    terminated = False
    truncated = False

    while not env.done:
        actions: Dict[str, Union[List[str], str]] = {}
        active_powers = info.get("active_powers", {})

        for pname in obs:
            if not obs.get(pname) or pname not in active_powers:
                continue
            possible = info.get("possible_orders", {}).get(pname, [])
            orders, _ = diplomacy_agent_action(obs[pname], possible)
            actions[pname] = orders
            if verbose:
                print(f"    {pname}: {len(orders)} orders", flush=True)

        next_obs, rewards, terminated, truncated, next_info = env.step(actions)
        done = terminated or truncated

        reward_val = sum(rewards.values()) if isinstance(rewards, dict) and rewards else 0.0
        total_reward += reward_val

        exp = Experience(
            state=json.dumps(dict(obs), ensure_ascii=False, default=str),
            action=json.dumps(dict(actions), ensure_ascii=False, default=str),
            reward=float(reward_val),
            next_state=json.dumps(
                dict(next_obs), ensure_ascii=False, default=str
            ) if isinstance(next_obs, dict) else str(next_obs),
            done=done,
            intentions=None,
            tasks=task,
        )
        exp.idx = step_count
        exp.action_type = "primitive"
        exp.raw_state = _sanitize_keys(info)
        exp.raw_next_state = _sanitize_keys(next_info)
        exp.available_actions = _sanitize_keys(info.get("possible_orders", {}))
        exp.interface = {"env_name": "diplomacy", "game_name": "diplomacy"}
        experiences.append(exp)

        phase = next_info.get("phase", "")
        n_powers = len([p for p in obs if obs.get(p) and p in active_powers])
        print(f"    step {step_count}: phase={phase}, powers={n_powers}, reward={reward_val:.2f}", flush=True)
        if verbose:
            print(f"    cum_reward={total_reward:.2f}", flush=True)

        obs = next_obs
        info = next_info
        step_count += 1

        if done:
            break

    episode = Episode(experiences=experiences, task=task, env_name="diplomacy", game_name="diplomacy")
    episode.set_outcome()

    final_rewards = {}
    if isinstance(rewards, dict):
        final_rewards = {str(k): float(v) for k, v in rewards.items()}

    stats = {
        "game": "diplomacy",
        "steps": step_count,
        "total_reward": total_reward,
        "terminated": terminated,
        "truncated": truncated,
        "model": _POLICY.cfg.model_path,
        "checkpoint_type": _POLICY.cfg.checkpoint_type,
        "agent_type": "alfworld7b_evolver",
        "seed": seed,
        "max_phases": DIPLOMACY_MAX_PHASES,
        "final_sc_rewards": final_rewards,
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
    record = _sanitize_keys(episode.to_dict())
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
    summary: Dict[str, Any] = {
        "game": game_name,
        "timestamp": datetime.now().isoformat(),
        "model": args.model_path,
        "checkpoint_type": args.checkpoint_type,
        "agent_type": "alfworld7b_evolver",
        "total_episodes": len(all_stats),
        "target_episodes": args.episodes,
        "end_condition": "natural",
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

    episode_buffer = Episode_Buffer(buffer_size=args.episodes + 10)
    all_stats: List[Dict[str, Any]] = []
    t0 = time.time()

    for ep_idx in range(start_idx, args.episodes):
        print(f"\n  [{game_name}] Episode {ep_idx + 1}/{args.episodes}", flush=True)
        seed = args.seed + ep_idx

        try:
            if game_name == "avalon":
                episode, stats = run_avalon_episode(
                    num_players=args.num_players,
                    seed=seed,
                    verbose=args.verbose,
                )
            elif game_name == "diplomacy":
                episode, stats = run_diplomacy_episode(
                    seed=seed,
                    verbose=args.verbose,
                )
            else:
                print(f"    [ERROR] Unknown game: {game_name}")
                continue

            stats["episode_index"] = ep_idx
            print(f"    ✓ Done — Steps: {stats['steps']}, Reward: {stats['total_reward']:.2f}", flush=True)

            if args.label and not args.no_label and label_trajectory is not None:
                episode = label_trajectory(episode, args.label_model)

            episode_buffer.add_episode(episode)
            all_stats.append(stats)

            ep_data = _sanitize_keys(episode.to_dict())
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
        description="ALFWorld-7B cold-start rollouts for Avalon and Diplomacy (evolver wrappers)",
    )
    parser.add_argument("--games", type=str, nargs="+", default=None,
                        choices=list(EVOLVER_GAMES.keys()),
                        help="Games to generate rollouts for (default: all)")
    parser.add_argument("--episodes", type=int, default=20,
                        help="Number of episodes per game (default: 20)")
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
    parser.add_argument("--label_model", type=str, default="gpt-5-mini",
                        help="Model used for trajectory labeling.")
    parser.add_argument("--resume", action="store_true",
                        help="Resume interrupted run.")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: cold_start/output/alfworld7b_evolver)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_players", type=int, default=5,
                        help="Number of Avalon players (default: 5)")

    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else COLD_START_DIR / "output" / f"alfworld7b_{args.checkpoint_type}_evolver"
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = Alfworld7BConfig(
        model_path=args.model_path,
        checkpoint_type=args.checkpoint_type,
        temperature=args.temperature,
    )
    _POLICY = Alfworld7BPolicy(cfg)

    requested = args.games if args.games else list(EVOLVER_GAMES.keys())
    available_games: List[str] = []
    skipped_games: List[str] = []

    for g in requested:
        if g not in EVOLVER_GAMES:
            print(f"[WARNING] Game '{g}' not recognised, skipping.")
            skipped_games.append(g)
            continue
        if EVOLVER_GAMES[g]["env_class"] is None:
            print(f"[WARNING] Game '{g}' env wrapper not importable, skipping.")
            skipped_games.append(g)
            continue
        available_games.append(g)

    if not available_games:
        print("[ERROR] No games available. Ensure AgentEvolver / AI_Diplomacy are installed.")
        sys.exit(1)

    print("=" * 78)
    print("  ALFWorld-7B Cold-Start — Avalon & Diplomacy Rollout Generation")
    print("=" * 78)
    print(f"  Games:       {', '.join(available_games)}")
    if skipped_games:
        print(f"  Skipped:     {', '.join(skipped_games)}")
    print(f"  Episodes:    {args.episodes} per game")
    print(f"  End cond:    natural (avalon=engine done, diplomacy=max_phases {DIPLOMACY_MAX_PHASES})")
    print(f"  Model:       {args.model_path} ({args.checkpoint_type})")
    print(f"  Temperature: {args.temperature}")
    print(f"  Labeling:    {args.label and not args.no_label}")
    print(f"  Resume:      {args.resume}")
    print(f"  Seed:        {args.seed}")
    if "avalon" in available_games:
        print(f"  Avalon:      {args.num_players} players")
    print(f"  Output:      {output_dir}")
    print("=" * 78)

    overall_t0 = time.time()
    game_summaries: List[Dict[str, Any]] = []

    for game_name in available_games:
        print(f"\n{'━' * 78}")
        print(f"  GAME: {game_name.upper()} ({args.episodes} episodes)")
        print(f"{'━' * 78}")
        summary = run_game_rollouts(game_name, args, output_dir)
        game_summaries.append(summary)

    overall_elapsed = time.time() - overall_t0

    master_summary = {
        "timestamp": datetime.now().isoformat(),
        "model": args.model_path,
        "checkpoint_type": args.checkpoint_type,
        "agent_type": "alfworld7b_evolver",
        "episodes_per_game": args.episodes,
        "end_condition": "natural",
        "temperature": args.temperature,
        "labeled": args.label and not args.no_label,
        "total_elapsed_seconds": overall_elapsed,
        "games_completed": list(available_games),
        "games_skipped": skipped_games,
        "per_game_summaries": game_summaries,
    }
    master_path = output_dir / "batch_rollout_summary.json"
    with open(master_path, "w", encoding="utf-8") as f:
        json.dump(master_summary, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n{'=' * 78}")
    print("  ALFWorld-7B AVALON & DIPLOMACY — BATCH ROLLOUT COMPLETE")
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
