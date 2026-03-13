#!/usr/bin/env python
"""
Cold-start rollout generation for ALFWorld using the released 7B checkpoints
from SkillRL (Alfworld-7B-SFT and Alfworld-7B-RL).

This mirrors the structure of the existing cold-start generators in this
repository, but instead of calling GPT-5.4 over an API, it runs a local
HF checkpoint for ALFWorld.

Output structure (cold_start/output/<suite>/<task_name>/):
  - episode_NNN.json        Individual episode (Episode.to_dict())
  - episode_buffer.json     All episodes in Episode_Buffer format
  - rollouts.jsonl          Append-friendly JSONL (one Episode per line)
  - rollout_summary.json    Per-task run stats

Suites:
  - alfworld7b_sft/  (for Alfworld-7B-SFT checkpoints)
  - alfworld7b_rl/   (for Alfworld-7B-RL checkpoints)

Usage (from Game-AI-Agent root, after installing ALFWorld + SkillRL deps):

    # SFT checkpoint (HF or local path)
    python cold_start/ALFWORLD-7B/generate_cold_start_alfworld_7b.py \
        --checkpoint_type sft \
        --model_path Jianwen/Alfworld-7B-SFT \
        --tasks train.idx \
        --episodes 10

    # RL checkpoint
    python cold_start/ALFWORLD-7B/generate_cold_start_alfworld_7b.py \
        --checkpoint_type rl \
        --model_path Jianwen/Alfworld-7B-RL \
        --tasks train.idx \
        --episodes 10

You are responsible for:
  - Installing ALFWorld and its assets (see SkillRL and ALFWorld docs).
  - Providing a valid --model_path that transformers can load.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent          # cold_start/ALFWORLD-7B/
COLD_START_DIR = SCRIPT_DIR.parent                     # cold_start/
CODEBASE_ROOT = COLD_START_DIR.parent                  # Game-AI-Agent/

for _p in [str(CODEBASE_ROOT), str(SCRIPT_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from data_structure.experience import Experience, Episode, Episode_Buffer  # type: ignore

try:
    from cold_start.generate_cold_start import label_trajectory  # type: ignore
except Exception:
    label_trajectory = None

try:
    # Standard ALFWorld gym-style API (you must have alfworld installed)
    import alfworld.agents.environment.alfworld_env as alfworld_env  # type: ignore
except Exception as exc:  # pragma: no cover - optional dependency
    alfworld_env = None  # type: ignore

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
except Exception as exc:  # pragma: no cover - optional dependency
    AutoModelForCausalLM = None  # type: ignore
    AutoTokenizer = None  # type: ignore


@dataclass
class AlfworldConfig:
    """Minimal config describing the SkillRL ALFWorld checkpoint."""

    model_path: str
    checkpoint_type: str  # "sft" or "rl"
    max_steps: int = 50
    temperature: float = 0.8


class AlfworldEnvWrapper:
    """
    Thin wrapper around ALFWorld environment to match the cold_start interface.

    This assumes the standard ALFWorld 'alfworld_env.AlfworldEnv' API:
      - reset() -> (obs_str, info_dict)
      - step(action_str) -> (next_obs_str, reward, done, info_dict)
    If your local ALFWorld build uses a slightly different API, adapt here.
    """

    def __init__(self, task_config: str, max_steps: int = 50):
        if alfworld_env is None:
            raise ImportError(
                "alfworld is not installed. Install ALFWorld and its assets before "
                "running ALFWORLD-7B cold-start generation."
            )
        # Standard ALFWorld initialization; user must prepare configs separately.
        self.env = alfworld_env.AlfworldEnv(task_config)  # type: ignore[attr-defined]
        self.max_steps = max_steps
        self.steps = 0

    def reset(self) -> Tuple[str, Dict[str, Any]]:
        self.steps = 0
        obs, info = self.env.reset()
        info = info or {}
        return str(obs), dict(info)

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        self.steps += 1
        next_obs, reward, done, info = self.env.step(action)
        info = info or {}
        terminated = bool(done)
        truncated = bool(self.steps >= self.max_steps and not terminated)
        if truncated:
            done = True
        return str(next_obs), float(reward), terminated, truncated, dict(info)

    def close(self) -> None:
        try:
            self.env.close()
        except Exception:
            pass


class AlfworldPolicy:
    """
    HF-based policy wrapper over a SkillRL ALFWorld 7B checkpoint.

    This is intentionally simple: it treats the problem as next-token prediction
    conditioned on the textual observation and valid actions.
    """

    def __init__(self, cfg: AlfworldConfig, device: Optional[str] = None):
        if AutoModelForCausalLM is None or AutoTokenizer is None:
            raise ImportError(
                "transformers is not installed. Install transformers and re-run "
                "to use ALFWORLD-7B checkpoints."
            )
        self.cfg = cfg
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.model_path,
            device_map="auto" if device is None else device,
            torch_dtype="auto",
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _build_prompt(self, obs: str, action_space: List[str]) -> str:
        # Simple generic prompt; tune as needed to better match SkillRL SFT format.
        actions_str = ", ".join(action_space)
        return (
            "You are an ALFWorld household task agent. "
            "You must choose EXACTLY one next action from the allowed action list.\n\n"
            f"Observation:\n{obs}\n\n"
            f"Allowed actions: {actions_str}\n\n"
            "Respond with ONLY the chosen action string, nothing else.\n"
        )

    def act(self, obs: str, action_space: List[str]) -> str:
        if not action_space:
            # Fallback to a generic 'look' if nothing is provided.
            action_space = ["look around"]
        prompt = self._build_prompt(obs, action_space)
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=True,
            temperature=self.cfg.temperature,
            pad_token_id=self.tokenizer.eos_token_id,
        )[0]

        full_text = self.tokenizer.decode(output_ids, skip_special_tokens=True)
        generated = full_text[len(prompt) :].strip()

        # Heuristic: find the first allowed action that appears in the generated text.
        for a in action_space:
            if a.lower() in generated.lower():
                return a

        # Fallback 1: exact line match
        first_line = generated.splitlines()[0].strip()
        for a in action_space:
            if first_line.lower() == a.lower():
                return a

        # Fallback 2: default to first action
        return action_space[0]


def run_alfworld_episode(
    env: AlfworldEnvWrapper,
    policy: AlfworldPolicy,
    task_name: str,
    max_steps: int,
    verbose: bool = False,
) -> Tuple[Episode, Dict[str, Any]]:
    """
    Run one ALFWorld episode with an ALFWORLD-7B checkpoint.

    Since ALFWorld exposes a large textual action space, we assume the environment
    encodes valid actions inside the observation text (or you can adapt this
    to read them from info if available).
    """
    # For now we use a fixed generic action space that is reasonable across many ALFWorld tasks.
    # If your env exposes a richer set of admissible actions per step, adapt this.
    base_action_space = [
        "look around",
        "inventory",
        "examine",
        "open",
        "close",
        "pick up",
        "drop",
        "go north",
        "go south",
        "go east",
        "go west",
    ]

    obs, info = env.reset()
    experiences: List[Experience] = []
    total_reward = 0.0
    step_count = 0
    terminated = False
    truncated = False

    while step_count < max_steps:
        action_space = list(base_action_space)
        action = policy.act(obs, action_space)
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
            tasks=f"ALFWorld task: {task_name}",
        )
        exp.idx = step_count
        exp.action_type = "primitive"
        exp.raw_state = info
        exp.raw_next_state = next_info
        exp.available_actions = action_space
        exp.interface = {
            "env_name": "alfworld",
            "game_name": task_name,
        }
        experiences.append(exp)

        if verbose:
            print(
                f"  step {step_count}: action={action!r}, "
                f"reward={reward:.3f}, cum={total_reward:.3f}"
            )

        obs = next_obs
        info = next_info
        step_count += 1
        if done:
            break

    env.close()

    episode = Episode(
        experiences=experiences,
        task=f"ALFWorld task: {task_name}",
        env_name="alfworld",
        game_name=task_name,
    )
    episode.set_outcome()

    stats = {
        "game": task_name,
        "steps": step_count,
        "total_reward": total_reward,
        "terminated": terminated,
        "truncated": truncated,
        "model": policy.cfg.model_path,
        "checkpoint_type": policy.cfg.checkpoint_type,
        "agent_type": "alfworld7b",
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
    task_name: str,
    game_dir: Path,
    all_stats: List[Dict[str, Any]],
    elapsed: float,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "game": task_name,
        "timestamp": datetime.now().isoformat(),
        "model": args.model_path,
        "checkpoint_type": args.checkpoint_type,
        "agent_type": "alfworld7b",
        "total_episodes": len(all_stats),
        "target_episodes": args.episodes,
        "max_steps": args.max_steps,
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


def run_task_rollouts(
    task_name: str,
    args: argparse.Namespace,
    output_dir: Path,
    cfg: AlfworldConfig,
) -> Dict[str, Any]:
    """
    Run all episodes for one ALFWorld task and save outputs.

    We treat each ALFWorld task (from a task config file) as a "game_name" for
    compatibility with the rest of the cold_start pipeline.
    """
    game_dir = output_dir / task_name
    game_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = game_dir / "rollouts.jsonl"

    start_idx = 0
    if args.resume:
        start_idx = count_existing_episodes(game_dir)
        if start_idx >= args.episodes:
            print(f"  [SKIP] {task_name}: {start_idx}/{args.episodes} episodes already done")
            return {"game": task_name, "skipped": True, "existing": start_idx}
        if start_idx > 0:
            print(f"  [RESUME] {task_name}: resuming from episode {start_idx}")

    episode_buffer = Episode_Buffer(buffer_size=args.episodes + 10)
    all_stats: List[Dict[str, Any]] = []
    t0 = time.time()

    policy = AlfworldPolicy(cfg)

    for ep_idx in range(start_idx, args.episodes):
        print(f"\n  [{task_name}] Episode {ep_idx + 1}/{args.episodes}")
        try:
            env = AlfworldEnvWrapper(task_config=task_name, max_steps=cfg.max_steps)
            episode, stats = run_alfworld_episode(
                env=env,
                policy=policy,
                task_name=task_name,
                max_steps=cfg.max_steps,
                verbose=args.verbose,
            )
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
                    "game": task_name,
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

    summary = save_game_summary(task_name, game_dir, all_stats, elapsed, args)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ALFWORLD-7B cold-start rollouts for ALFWorld tasks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        required=True,
        help=(
            "ALFWorld task config(s) to run. This can be a path to a single "
            "config file, or multiple entries. Typically you will pass a "
            "train/valid index file or a specific .json task description."
        ),
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Number of episodes per task (default: 10)",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=50,
        help="Max steps per episode (default: 50)",
    )
    parser.add_argument(
        "--checkpoint_type",
        type=str,
        default="sft",
        choices=["sft", "rl"],
        help="Checkpoint type: 'sft' (Alfworld-7B-SFT) or 'rl' (Alfworld-7B-RL).",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help=(
            "HF model identifier or local path for the ALFWorld 7B checkpoint, "
            "e.g. 'Jianwen/Alfworld-7B-SFT' or a local directory."
        ),
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature for the 7B model (default: 0.8)",
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
        help=(
            "Output directory (default: cold_start/output/alfworld7b_sft or "
            "cold_start/output/alfworld7b_rl depending on checkpoint_type)."
        ),
    )

    args = parser.parse_args()

    suite_name = "alfworld7b_sft" if args.checkpoint_type == "sft" else "alfworld7b_rl"
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else COLD_START_DIR / "output" / suite_name
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    if label_trajectory is None and args.label and not args.no_label:
        print("[INFO] label_trajectory not available; skipping labeling")
        args.no_label = True

    cfg = AlfworldConfig(
        model_path=args.model_path,
        checkpoint_type=args.checkpoint_type,
        max_steps=args.max_steps,
        temperature=args.temperature,
    )

    print("=" * 78)
    print(f"  ALFWORLD-7B Cold-Start — ALFWorld Tasks")
    print("=" * 78)
    print(f"  Tasks:        {', '.join(args.tasks)}")
    print(f"  Episodes:     {args.episodes} per task")
    print(f"  Max steps:    {args.max_steps}")
    print(f"  Checkpoint:   {args.model_path} ({args.checkpoint_type})")
    print(f"  Temperature:  {args.temperature}")
    print(f"  Labeling:     {args.label and not args.no_label}")
    print(f"  Resume:       {args.resume}")
    print(f"  Output:       {output_dir}")
    print("=" * 78)

    overall_t0 = time.time()
    task_summaries: List[Dict[str, Any]] = []

    for task_name in args.tasks:
        print(f"\n{'━' * 78}")
        print(f"  TASK: {task_name} ({args.episodes} episodes)")
        print(f"{'━' * 78}")
        summary = run_task_rollouts(task_name, args, output_dir, cfg)
        task_summaries.append(summary)

    overall_elapsed = time.time() - overall_t0

    master_summary = {
        "timestamp": datetime.now().isoformat(),
        "model": args.model_path,
        "checkpoint_type": args.checkpoint_type,
        "agent_type": "alfworld7b",
        "episodes_per_task": args.episodes,
        "max_steps": args.max_steps,
        "temperature": args.temperature,
        "labeled": args.label and not args.no_label,
        "total_elapsed_seconds": overall_elapsed,
        "tasks_completed": args.tasks,
        "per_task_summaries": task_summaries,
    }
    master_path = output_dir / "batch_rollout_summary.json"
    with open(master_path, "w", encoding="utf-8") as f:
        json.dump(master_summary, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n{'=' * 78}")
    print("  ALFWORLD-7B COLD-START COMPLETE")
    print(f"{'=' * 78}")
    print(f"  Tasks:          {len(args.tasks)}")
    total_eps = sum(
        s.get("total_episodes", 0) for s in task_summaries if not s.get("skipped")
    )
    print(f"  Total episodes: {total_eps}")
    print(f"  Elapsed:        {overall_elapsed:.1f}s")
    print(f"  Output:         {output_dir}")
    print(f"  Summary:        {master_path}")
    print(f"{'=' * 78}\n")


if __name__ == "__main__":
    main()

