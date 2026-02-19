"""
Run inference (decision agent) and store rollouts in data_structure format.

Fatal hyperparameters and game env list: see inference_defaults.py.

Usage:
  python -m scripts.run_inference --game overcooked --task "Complete level" --max-steps 500
  python -m scripts.run_inference --game gamingagent --save-path rollouts/episodes.jsonl --verbose
  python -m scripts.run_inference --print-envs
  python -m scripts.run_inference --print-defaults
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add repo root
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.inference_defaults import (
    INFERENCE_GAME_ENVS,
    INFERENCE_FATAL,
    INFERENCE_DEFAULT_SAVE_DIR,
    INFERENCE_DEFAULT_SAVE_FILE,
)


def _make_env_for_game(game: str, seed: int = 0, max_steps: int = 500):
    """Build an env instance for the given game. Requires corresponding deps (overcooked_ai, gamingagent, videogamebench)."""
    game = (game or "").strip().lower()
    if game == "gamingagent":
        try:
            from env_wrappers.gamingagent_nl_wrapper import GamingAgentNLWrapper
            from gamingagent.gym_like import make_gaming_env
            base = make_gaming_env(game="twenty_forty_eight", max_steps=max_steps)
            return GamingAgentNLWrapper(base)
        except Exception as e:
            raise RuntimeError(f"GamingAgent env not available (install GamingAgent): {e}") from e
    if game == "videogamebench_dos":
        try:
            from env_wrappers.videogamebench_dos_nl_wrapper import VideoGameBenchDOSNLWrapper
            from videogamebench.gym_like import make_videogamebench_env
            base = make_videogamebench_env(game="doom2", max_steps=max_steps)
            return VideoGameBenchDOSNLWrapper(base)
        except Exception as e:
            raise RuntimeError(f"VideoGameBench DOS env not available (install videogamebench): {e}") from e
    if game == "overcooked":
        try:
            from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
            from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
            from env_wrappers.overcooked_nl_wrapper import OvercookedNLWrapper
            mdp = OvercookedGridworld.from_layout_name("cramped_room")
            base = OvercookedEnv(mdp, horizon=max_steps)
            return OvercookedNLWrapper(base, multi_agent=False)
        except Exception as e:
            raise RuntimeError(f"Overcooked env not available (install overcooked_ai): {e}") from e
    raise ValueError(
        f"Unknown game for auto env: '{game}'. Supported: {', '.join(INFERENCE_GAME_ENVS)}. "
        "Use inference.run_inference(env=...) with your own env from env_wrappers or evaluate_*."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run inference and store rollouts. Fatal hyperparameters in scripts/inference_defaults.py.",
    )
    parser.add_argument(
        "--game",
        type=str,
        default="",
        choices=INFERENCE_GAME_ENVS + [""],
        help="Game env name (required for auto env; use env_wrappers for others)",
    )
    parser.add_argument(
        "--task",
        type=str,
        default=INFERENCE_FATAL["task"],
        help="Task description for the episode",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=INFERENCE_FATAL["max_steps"],
        help="Max steps per episode (default: %(default)s)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=INFERENCE_FATAL["model"],
        help="Model name for VLM agent (default: %(default)s)",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default=None,
        help="Append rollouts to this path (JSONL). Default: rollouts/episodes.jsonl if --save",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save rollouts to default path rollouts/episodes.jsonl",
    )
    parser.add_argument(
        "--episode-buffer-size",
        type=int,
        default=INFERENCE_FATAL["episode_buffer_size"],
        help="Episode buffer size if using buffers (default: %(default)s)",
    )
    parser.add_argument(
        "--experience-buffer-size",
        type=int,
        default=INFERENCE_FATAL["experience_buffer_size"],
        help="Experience buffer size if using buffers (default: %(default)s)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=INFERENCE_FATAL["verbose"],
        help="Verbose output",
    )
    parser.add_argument(
        "--print-envs",
        action="store_true",
        help="Print supported game envs and exit",
    )
    parser.add_argument(
        "--print-defaults",
        action="store_true",
        help="Print fatal hyperparameter defaults and exit",
    )
    args = parser.parse_args()

    if args.print_envs:
        print("Inference game envs (inference_defaults.INFERENCE_GAME_ENVS):")
        for e in INFERENCE_GAME_ENVS:
            print(f"  - {e}")
        return

    if args.print_defaults:
        print("Inference (fatal) defaults (inference_defaults.INFERENCE_FATAL):")
        for k, v in INFERENCE_FATAL.items():
            print(f"  {k}: {v}")
        return

    if not args.game:
        parser.error("--game is required to run inference (or use inference.run_inference(env=...) directly)")

    from inference import run_inference
    from data_structure.experience import Episode_Buffer, Experience_Replay_Buffer

    env = _make_env_for_game(args.game, max_steps=args.max_steps)
    save_path = args.save_path
    if save_path is None and args.save:
        Path(INFERENCE_DEFAULT_SAVE_DIR).mkdir(parents=True, exist_ok=True)
        save_path = str(Path(INFERENCE_DEFAULT_SAVE_DIR) / INFERENCE_DEFAULT_SAVE_FILE)

    ep_buffer = Episode_Buffer(buffer_size=args.episode_buffer_size)
    exp_buffer = Experience_Replay_Buffer(buffer_size=args.experience_buffer_size)

    episode = run_inference(
        env,
        agent=None,
        task=args.task,
        model=args.model,
        skill_bank=None,
        memory=None,
        reward_config=None,
        max_steps=args.max_steps,
        verbose=args.verbose,
        episode_buffer=ep_buffer,
        experience_buffer=exp_buffer,
        save_path=save_path,
    )
    print(f"Episode: {len(episode.experiences)} steps, total reward: {episode.get_reward():.2f}")


if __name__ == "__main__":
    main()
