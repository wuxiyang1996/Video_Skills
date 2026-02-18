#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test script: run dummy_agent in VideoGameBench via DOS games only.

Uses DOS games (JS-DOS in browser) via env_wrappers.VideoGameBenchDOSNLWrapper.
Game Boy / PyBoy games are excluded (no ROMs required).

Usage (from the Game-AI-Agent codebase root):

    set PYTHONPATH=%CD%;%CD%\..\videogamebench
    python evaluate_videogamebench/test_videogamebench_dummy.py --game doom2

Options:
    --game        DOS game name (default: doom2). Use --list-games to see options.
    --max_steps   Episode length (default: 20)
    --episodes    Number of episodes (default: 1)
    --mode        Agent mode: "llm" | "random_nl" | "fallback" (default: fallback)
    --model       LLM model for "llm" mode (default: gpt-4o-mini)
    --port        Local server port (default: 8000)
    --headless    Run browser headless
    --verbose     Print state/action per step
    --list-games  List available DOS games and exit
"""

import argparse
import asyncio
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

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
# Imports
# ---------------------------------------------------------------------------
try:
    from env_wrappers.videogamebench_dos_nl_wrapper import (
        VideoGameBenchDOSNLWrapper,
        VIDEOGAMEBENCH_DOS_VALID_KEYS,
        list_dos_games,
    )
except ImportError as e:
    print(f"[ERROR] Cannot import env_wrappers: {e}")
    print("  Run from Game-AI-Agent root. Ensure env_wrappers is on PYTHONPATH.")
    sys.exit(1)

try:
    from agents.dummy_agent import (
        language_agent_action,
        GAME_VIDEOGAMEBENCH_DOS,
        _default_action,
    )
except ImportError:
    GAME_VIDEOGAMEBENCH_DOS = "videogamebench_dos"
    language_agent_action = None
    _default_action = lambda g: "Space"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def random_nl_action() -> str:
    return random.choice(VIDEOGAMEBENCH_DOS_VALID_KEYS)


def choose_action(obs_nl: str, mode: str, model: str) -> str:
    if mode == "llm" and language_agent_action:
        try:
            action = language_agent_action(
                state_nl=obs_nl,
                game=GAME_VIDEOGAMEBENCH_DOS,
                model=model,
                use_function_call=True,
                temperature=0.3,
            )
            if isinstance(action, str) and action:
                return action
        except Exception as e:
            print(f"  [WARNING] LLM action failed: {e}")
    elif mode == "random_nl":
        return random_nl_action()
    return "Space"


# ---------------------------------------------------------------------------
# Async episode runner
# ---------------------------------------------------------------------------

async def run_episode_async(
    game: str,
    max_steps: int,
    mode: str,
    model: str,
    verbose: bool,
    episode_id: int,
    port: int,
    headless: bool,
) -> Dict[str, Any]:
    try:
        from src.consts import GAME_URL_MAP
        from src.emulators.dos.interface import DOSGameInterface
        from src.emulators.dos.website_server import DOSGameServer
    except ImportError as e:
        print(f"[ERROR] videogamebench DOS modules not found: {e}")
        print("  Add videogamebench to PYTHONPATH and install: pip install playwright && playwright install")
        return {"error": "videogamebench DOS not available"}

    if game not in GAME_URL_MAP:
        return {"error": f"Unknown game '{game}'. Use --list-games for options."}

    game_url = GAME_URL_MAP[game]
    if not isinstance(game_url, str) or not game_url.startswith("http"):
        return {"error": f"Game '{game}' has no JS-DOS URL. Use --list-games."}

    server = DOSGameServer(port=port, lite=False)
    url = server.start(game_url)
    if verbose:
        print(f"  Server started: {url}")

    interface = DOSGameInterface(game=game, headless=headless, lite=False)
    await interface.load_game(initial_url=url)
    await asyncio.sleep(2.0)

    nl_wrapper = VideoGameBenchDOSNLWrapper()
    obs_nl = nl_wrapper.build_state_nl()

    total_reward = 0.0
    step_count = 0
    actions_log: List[str] = []

    print(f"\n{'='*78}")
    print(f"  Episode {episode_id + 1}  |  Game: {game}  |  Max steps: {max_steps}  |  Mode: {mode}")
    print(f"{'='*78}")

    if verbose:
        print(f"\nInitial state:\n{obs_nl}")
    else:
        print(f"  Valid keys: {VIDEOGAMEBENCH_DOS_VALID_KEYS}")

    for _ in range(max_steps - 1):
        step_count += 1
        action = choose_action(obs_nl, mode, model)
        key = nl_wrapper.parse_action(action)
        actions_log.append(key)

        try:
            info, frames = await interface.step("press_key", key)
        except Exception as e:
            print(f"\n  [ERROR at step {step_count}] {e}")
            break

        obs = await interface.get_observation()
        nl_wrapper.advance_step()
        obs_nl = nl_wrapper.build_state_nl()
        reward = 0.0

        if verbose:
            print(f"\n  Step {step_count}: key={key}, obs_size={len(obs) if obs else 0}")
        else:
            print(f"  Step {step_count}: {key}")

    await interface.close()
    server.stop()

    action_counts: Dict[str, int] = {}
    for a in actions_log:
        action_counts[a] = action_counts.get(a, 0) + 1

    result = {
        "episode_id": episode_id,
        "game": game,
        "max_steps": max_steps,
        "mode": mode,
        "steps": step_count,
        "total_reward": total_reward,
        "action_distribution": action_counts,
    }

    print(f"\n{'-'*78}")
    print(f"  Episode {episode_id + 1} Summary")
    print(f"{'-'*78}")
    print(f"  Steps:         {step_count}")
    print(f"  Actions:       {action_counts}")
    return result


def run_episode(*args, **kwargs) -> Dict[str, Any]:
    return asyncio.run(run_episode_async(*args, **kwargs))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Test dummy agent in VideoGameBench (DOS games only, no Game Boy)",
    )
    parser.add_argument("--game", type=str, default="doom2", help="DOS game name")
    parser.add_argument("--max_steps", type=int, default=20, help="Max steps per episode")
    parser.add_argument("--episodes", type=int, default=1, help="Number of episodes")
    parser.add_argument("--mode", type=str, default="fallback",
                        choices=["llm", "random_nl", "fallback"], help="Agent mode")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="LLM model for llm mode")
    parser.add_argument("--port", type=int, default=8000, help="Local server port")
    parser.add_argument("--headless", action="store_true", help="Run browser headless")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print full state each step")
    parser.add_argument("--list-games", action="store_true", help="List available DOS games and exit")

    args = parser.parse_args()

    if args.list_games:
        games = list_dos_games()
        print("Available DOS games (no ROMs required):")
        for g in games:
            print(f"  {g}")
        return

    if args.mode == "llm":
        if not os.environ.get("OPENAI_API_KEY"):
            print("[WARNING] OPENAI_API_KEY not set. LLM mode may fall back to default actions.")

    print("VideoGameBench Dummy Agent Test (DOS games only)")
    print(f"  Game:       {args.game}")
    print(f"  Max steps:  {args.max_steps}")
    print(f"  Episodes:   {args.episodes}")
    print(f"  Mode:       {args.mode}")
    if args.mode == "llm":
        print(f"  Model:      {args.model}")

    all_results = []
    t0 = time.time()

    for ep in range(args.episodes):
        r = run_episode(
            game=args.game,
            max_steps=args.max_steps,
            mode=args.mode,
            model=args.model,
            verbose=args.verbose,
            episode_id=ep,
            port=args.port,
            headless=args.headless,
        )
        if "error" not in r:
            all_results.append(r)

    elapsed = time.time() - t0

    if all_results:
        print(f"\n{'='*78}")
        print("  OVERALL RESULTS")
        print(f"{'='*78}")
        total_steps = [r["steps"] for r in all_results]
        print(f"  Mean Steps:   {sum(total_steps) / len(total_steps):.1f}")
        print(f"  Total Time:   {elapsed:.2f}s")
        print(f"{'='*78}\n")


if __name__ == "__main__":
    main()
