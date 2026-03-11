#!/usr/bin/env python
"""
Cold-start data generation for VideoGameBench DOS games.

Generates trajectories using the dummy language agent (GPT-5-mini) against
VideoGameBench DOS games (JS-DOS in browser), then optionally labels them
with the same model. Output format matches cold_start/output/ so data can
be ingested by the skill pipeline and trainer.

Usage (from Game-AI-Agent root):

    export OPENROUTER_API_KEY="sk-or-..."  # or set open_router_api_key in api_keys.py
    export PYTHONPATH="$(pwd):$(pwd)/../videogamebench:$PYTHONPATH"

    # 5 episodes per DOS game, GPT-5-mini (default)
    python cold_start/generate_cold_start_videogamebench_dos.py

    # Specific games, fewer episodes
    python cold_start/generate_cold_start_videogamebench_dos.py --games doom2 civ --episodes 2

    # Skip LLM labeling (faster)
    python cold_start/generate_cold_start_videogamebench_dos.py --episodes 5 --no_label
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import socket
import socketserver
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
CODEBASE_ROOT = SCRIPT_DIR.parent
VIDEOGAMEBENCH_ROOT = CODEBASE_ROOT.parent / "videogamebench"

for p in [str(CODEBASE_ROOT), str(VIDEOGAMEBENCH_ROOT)]:
    if Path(p).exists() and p not in sys.path:
        sys.path.insert(0, p)

# ---------------------------------------------------------------------------
# Imports from Game-AI-Agent
# ---------------------------------------------------------------------------
from data_structure.experience import Experience, Episode, Episode_Buffer
from decision_agents.dummy_agent import (
    language_agent_action,
    _default_action,
    GAME_VIDEOGAMEBENCH_DOS,
)

try:
    from env_wrappers.videogamebench_dos_nl_wrapper import (
        VideoGameBenchDOSNLWrapper,
        VIDEOGAMEBENCH_DOS_VALID_KEYS,
        list_dos_games,
    )
except ImportError as e:
    print(f"[ERROR] Cannot import env_wrappers: {e}")
    print("  Run from Game-AI-Agent root with PYTHONPATH including this repo.")
    sys.exit(1)

# Optional: trajectory labeling (reuse from GamingAgent cold-start)
try:
    from cold_start.generate_cold_start import label_trajectory
except ImportError:
    label_trajectory = None

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_MODEL = "gpt-5-mini"
DEFAULT_EPISODES_PER_GAME = 5
DEFAULT_MAX_STEPS = 50
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "output"


def _patch_browser_controller_for_headless():
    """Patch BrowserController.start() to force headless + --no-sandbox for servers without a display."""
    try:
        from src.emulators.dos.browser_controller import BrowserController
        from playwright.async_api import async_playwright
        import platform as _platform

        _orig_start = BrowserController.start

        async def _headless_start(self):
            self.playwright = await async_playwright().start()
            self.browser = await self.playwright.chromium.launch(
                headless=True,
                args=["--no-sandbox", "--disable-web-security", "--disable-dev-shm-usage"],
            )
            vp = {"width": 640, "height": 400} if _platform.system() == "Darwin" else {"width": 700, "height": 475}
            self.context = await self.browser.new_context(
                viewport=vp,
                user_agent="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36",
            )
            self.page = await self.context.new_page()
            self.current_mouse_position = (0, 0)

        BrowserController.start = _headless_start
    except ImportError:
        pass


def _get_dos_env():
    """Import VideoGameBench DOS server/interface; return (server_cls, interface_cls, url_map) or (None, None, None)."""
    try:
        from src.consts import GAME_URL_MAP
        from src.emulators.dos.interface import DOSGameInterface
        from src.emulators.dos.website_server import DOSGameServer
        socketserver.TCPServer.allow_reuse_address = True
        _patch_browser_controller_for_headless()
        return DOSGameServer, DOSGameInterface, GAME_URL_MAP
    except ImportError:
        return None, None, None


def _find_free_port(start: int = 8000) -> int:
    """Return a port that is currently free, starting the search from *start*."""
    for candidate in range(start, start + 200):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("", candidate))
                return candidate
            except OSError:
                continue
    raise RuntimeError(f"No free port found in range {start}–{start + 200}")


async def run_one_episode_dos(
    game_name: str,
    max_steps: int,
    model: str,
    episode_id: int,
    port: int,
    headless: bool,
    verbose: bool,
) -> Tuple[Episode, Dict[str, Any]]:
    """
    Run a single episode for one DOS game using the dummy language agent.
    Returns (Episode, stats_dict).
    """
    DOSGameServer, DOSGameInterface, GAME_URL_MAP = _get_dos_env()
    if DOSGameServer is None or GAME_URL_MAP is None:
        raise RuntimeError("videogamebench DOS modules not available. Add videogamebench to PYTHONPATH.")

    if game_name not in GAME_URL_MAP:
        raise ValueError(f"Unknown DOS game '{game_name}'. Use --list-games.")
    game_url = GAME_URL_MAP[game_name]
    if not isinstance(game_url, str) or not game_url.startswith("http"):
        raise ValueError(f"Game '{game_name}' has no JS-DOS URL.")

    actual_port = _find_free_port(port)
    server = DOSGameServer(port=actual_port, lite=False)
    interface = None
    experiences: List[Experience] = []
    total_reward = 0.0
    step_count = 0
    terminated = False

    try:
        url = server.start(game_url)
        if verbose:
            print(f"    Server started: {url} (port {actual_port})")

        interface = DOSGameInterface(game=game_name, headless=headless, lite=False)
        await interface.load_game(initial_url=url)
        await asyncio.sleep(2.0)

        nl_wrapper = VideoGameBenchDOSNLWrapper(game_name=game_name)
        task = f"Play the DOS game {game_name}."
        obs_nl = nl_wrapper.build_state_nl()

        while step_count < max_steps:
            try:
                action = language_agent_action(
                    state_nl=obs_nl + f"\n\nValid keys: {', '.join(VIDEOGAMEBENCH_DOS_VALID_KEYS)}. Choose one.",
                    game=GAME_VIDEOGAMEBENCH_DOS,
                    model=model,
                    use_function_call=True,
                    temperature=0.3,
                )
                if not isinstance(action, str) or not action:
                    action = _default_action(GAME_VIDEOGAMEBENCH_DOS)
            except Exception as e:
                if verbose:
                    print(f"    [WARNING] LLM action failed: {e}")
                action = _default_action(GAME_VIDEOGAMEBENCH_DOS)

            key = nl_wrapper.parse_action(action)
            try:
                info, frames = await interface.step("press_key", key)
            except Exception as e:
                if verbose:
                    print(f"    [ERROR] step failed: {e}")
                break

            next_obs = await interface.get_observation()
            nl_wrapper.advance_step()
            next_obs_nl = nl_wrapper.build_state_nl()
            reward = 0.0
            done = False

            exp = Experience(
                state=obs_nl,
                action=str(action),
                reward=float(reward),
                next_state=next_obs_nl,
                done=done,
                tasks=task,
            )
            exp.idx = step_count
            exp.raw_state = obs_nl
            exp.raw_next_state = next_obs_nl
            exp.available_actions = list(VIDEOGAMEBENCH_DOS_VALID_KEYS)
            exp.interface = {"env_name": "videogamebench_dos", "game_name": game_name}
            experiences.append(exp)

            total_reward += reward
            step_count += 1
            obs_nl = next_obs_nl

            if verbose:
                print(f"    Step {step_count}: key={key}")

        if experiences:
            experiences[-1].done = True

    finally:
        # Guarantee cleanup regardless of success or failure
        if interface is not None:
            try:
                await interface.close()
            except Exception:
                pass
        try:
            server.stop()
        except Exception:
            pass
        # Brief cooldown so the OS can fully release the socket
        await asyncio.sleep(1.0)

    task = f"Play the DOS game {game_name}."
    episode = Episode(
        experiences=experiences,
        task=task,
        env_name="videogamebench_dos",
        game_name=game_name,
    )
    episode.set_outcome()

    stats = {
        "game": game_name,
        "steps": step_count,
        "total_reward": total_reward,
        "terminated": terminated,
        "truncated": step_count >= max_steps,
        "model": model,
        "agent_type": "dummy",
    }
    return episode, stats


def run_episode_sync(*args, **kwargs) -> Tuple[Episode, Dict[str, Any]]:
    return asyncio.run(run_one_episode_dos(*args, **kwargs))


def count_existing_episodes(game_dir: Path) -> int:
    if not game_dir.exists():
        return 0
    return sum(1 for f in game_dir.glob("episode_*.json") if f.name != "episode_buffer.json")


def save_episode_jsonl(episode: Episode, jsonl_path: Path, stats: Dict[str, Any]) -> None:
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
    summary = {
        "game": game_name,
        "timestamp": datetime.now().isoformat(),
        "model": args.model,
        "agent_type": "dummy",
        "total_episodes": len(all_stats),
        "target_episodes": args.episodes,
        "max_steps": args.max_steps,
        "labeled": not args.no_label,
        "elapsed_seconds": elapsed,
        "episode_stats": all_stats,
    }
    if all_stats and "total_reward" in all_stats[0]:
        rewards = [s.get("total_reward", 0) for s in all_stats]
        steps = [s.get("steps", 0) for s in all_stats]
        summary["mean_reward"] = sum(rewards) / len(rewards)
        summary["mean_steps"] = sum(steps) / len(steps)
    summary_path = game_dir / "rollout_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Generate cold-start data from VideoGameBench DOS games using GPT-5-mini",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--games", type=str, nargs="+", default=None,
                        help="DOS games to run (default: all from list_dos_games())")
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES_PER_GAME,
                        help=f"Episodes per game (default: {DEFAULT_EPISODES_PER_GAME})")
    parser.add_argument("--max_steps", type=int, default=DEFAULT_MAX_STEPS,
                        help=f"Max steps per episode (default: {DEFAULT_MAX_STEPS})")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help=f"LLM model for agent and labeling (default: {DEFAULT_MODEL})")
    parser.add_argument("--no_label", action="store_true",
                        help="Skip trajectory labeling (faster)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume: skip games/episodes that already have output")
    parser.add_argument("--port", type=int, default=8000,
                        help="Local server port for JS-DOS")
    parser.add_argument("--headless", action="store_true",
                        help="Run browser headless")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print step-by-step details")
    parser.add_argument("--output_dir", type=str, default=None,
                        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--list-games", action="store_true",
                        help="List available DOS games and exit")

    args = parser.parse_args()

    if args.list_games:
        games = list_dos_games()
        print("Available VideoGameBench DOS games:")
        for g in games:
            print(f"  {g}")
        return

    output_dir = Path(args.output_dir) if args.output_dir else DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        from api_keys import open_router_api_key
        has_key = bool(os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY") or (open_router_api_key and open_router_api_key.strip()))
    except Exception:
        has_key = bool(os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY"))
    if not has_key:
        print("[WARNING] No API key set. LLM calls will fail.")
        print("  Prefer: open_router_api_key in api_keys.py or export OPENROUTER_API_KEY='sk-or-...'")
        print("  Or: export OPENAI_API_KEY='sk-...'")

    DOSGameServer, DOSGameInterface, GAME_URL_MAP = _get_dos_env()
    if DOSGameServer is None or GAME_URL_MAP is None:
        print("[ERROR] VideoGameBench DOS not available.")
        print("  Clone videogamebench as sibling and add to PYTHONPATH:")
        print("    export PYTHONPATH=\"$(pwd):$(pwd)/../videogamebench:$PYTHONPATH\"")
        print("  Install: pip install playwright && playwright install")
        sys.exit(1)

    if args.games:
        requested = [g.strip() for g in args.games]
    else:
        requested = list_dos_games()

    available = [g for g in requested if g in GAME_URL_MAP and isinstance(GAME_URL_MAP.get(g), str) and str(GAME_URL_MAP[g]).startswith("http")]
    skipped = [g for g in requested if g not in available]
    if skipped:
        print(f"[WARNING] Skipping (no JS-DOS URL or unknown): {skipped}")
    if not available:
        print("[ERROR] No DOS games available. Use --list-games.")
        sys.exit(1)

    if label_trajectory is None and not args.no_label:
        print("[WARNING] label_trajectory not available; running with --no_label")
        args.no_label = True

    print("=" * 78)
    print("  Cold-Start: VideoGameBench DOS (GPT-5-mini)")
    print("=" * 78)
    print(f"  Games:      {', '.join(available)}")
    print(f"  Episodes:   {args.episodes} per game")
    print(f"  Max steps:  {args.max_steps}")
    print(f"  Model:      {args.model}")
    print(f"  Labeling:   {not args.no_label}")
    print(f"  Output:     {output_dir}")
    print("=" * 78)

    overall_t0 = time.time()
    game_summaries: List[Dict[str, Any]] = []

    for game_name in available:
        game_dir = output_dir / "videogamebench_dos" / game_name
        game_dir.mkdir(parents=True, exist_ok=True)
        jsonl_path = game_dir / "rollouts.jsonl"

        start_idx = 0
        if args.resume:
            start_idx = count_existing_episodes(game_dir)
            if start_idx >= args.episodes:
                print(f"\n  [SKIP] {game_name}: already {start_idx}/{args.episodes} episodes")
                game_summaries.append({"game": game_name, "skipped": True, "existing": start_idx})
                continue
            if start_idx > 0:
                print(f"  [RESUME] {game_name}: from episode {start_idx}")

        print(f"\n{'─' * 78}")
        print(f"  Game: {game_name} ({args.episodes} episodes)")
        print(f"{'─' * 78}")

        episode_buffer = Episode_Buffer(buffer_size=args.episodes + 10)
        all_stats: List[Dict[str, Any]] = []
        t0 = time.time()

        for ep_idx in range(start_idx, args.episodes):
            print(f"\n  Episode {ep_idx + 1}/{args.episodes}")
            try:
                episode, stats = run_episode_sync(
                    game_name=game_name,
                    max_steps=args.max_steps,
                    model=args.model,
                    episode_id=ep_idx,
                    port=args.port,
                    headless=args.headless,
                    verbose=args.verbose,
                )
                stats["episode_index"] = ep_idx
                print(f"    Steps: {stats['steps']}, Reward: {stats['total_reward']:.2f}")

                if not args.no_label and label_trajectory:
                    episode = label_trajectory(episode, args.model)

                episode_buffer.add_episode(episode)
                all_stats.append(stats)

                ep_path = game_dir / f"episode_{ep_idx:03d}.json"
                ep_data = episode.to_dict()
                ep_data["metadata"] = stats
                with open(ep_path, "w", encoding="utf-8") as f:
                    json.dump(ep_data, f, indent=2, ensure_ascii=False, default=str)

                save_episode_jsonl(episode, jsonl_path, stats)

            except Exception as e:
                print(f"    [ERROR] Episode {ep_idx + 1} failed: {e}")
                traceback.print_exc()
                all_stats.append({
                    "game": game_name,
                    "episode_index": ep_idx,
                    "error": str(e),
                    "steps": 0,
                    "total_reward": 0.0,
                })

        elapsed = time.time() - t0
        buffer_path = game_dir / "episode_buffer.json"
        episode_buffer.save_to_json(str(buffer_path))
        print(f"\n  Saved {len(episode_buffer)} episodes to {buffer_path}")
        summary = save_game_summary(game_name, game_dir, all_stats, elapsed, args)
        game_summaries.append(summary)

    overall_elapsed = time.time() - overall_t0

    master = {
        "timestamp": datetime.now().isoformat(),
        "source": "videogamebench_dos",
        "model": args.model,
        "episodes_per_game": args.episodes,
        "max_steps": args.max_steps,
        "labeled": not args.no_label,
        "total_elapsed_seconds": overall_elapsed,
        "games": available,
        "per_game_summaries": game_summaries,
    }
    master_path = output_dir / "videogamebench_dos_batch_summary.json"
    with open(master_path, "w", encoding="utf-8") as f:
        json.dump(master, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n{'=' * 78}")
    print("  VIDEOGAMEBENCH DOS COLD-START COMPLETE")
    print(f"{'=' * 78}")
    total_eps = sum(s.get("total_episodes", 0) for s in game_summaries if not s.get("skipped"))
    print(f"  Games:       {len(available)}")
    print(f"  Episodes:    {total_eps}")
    print(f"  Elapsed:     {overall_elapsed:.1f}s")
    print(f"  Output:      {output_dir}")
    print(f"  Summary:     {master_path}")
    print(f"{'=' * 78}\n")


if __name__ == "__main__":
    main()
