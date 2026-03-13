#!/usr/bin/env python
"""
Pokemon Red cold-start rollout generation using ALFWorld-7B checkpoints as
a baseline agent (mirrors generate_cold_start_pokemon_red.py).

Uses Orak's PokemonRedEnv (PyBoyRunner) directly with high-level tools.
The ALFWorld-7B model replaces GPT-5.4 for all action decisions.

Output structure (cold_start/output/alfworld7b_pokemon_red/pokemon_red/):
  - episode_NNN.json        Individual episode
  - episode_buffer.json     Episode_Buffer
  - rollouts.jsonl          JSONL
  - rollout_summary.json    Per-game stats

Usage (from Game-AI-Agent root):

    export PYTHONPATH="$(pwd):$(pwd)/../GamingAgent:$(pwd)/../Orak/src:$PYTHONPATH"

    python cold_start/ALFWORLD-7B/generate_cold_start_pokemon_red_alfworld7b.py \\
        --model_path Jianwen/Alfworld-7B-SFT --episodes 3 --verbose

    python cold_start/ALFWORLD-7B/generate_cold_start_pokemon_red_alfworld7b.py \\
        --model_path Jianwen/Alfworld-7B-RL --checkpoint_type rl --episodes 3
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent          # cold_start/ALFWORLD-7B/
COLD_START_DIR = SCRIPT_DIR.parent                     # cold_start/
CODEBASE_ROOT = COLD_START_DIR.parent                  # Game-AI-Agent/
GAMINGAGENT_ROOT = CODEBASE_ROOT.parent / "GamingAgent"
ORAK_SRC = CODEBASE_ROOT.parent / "Orak" / "src"

for p in [str(CODEBASE_ROOT), str(GAMINGAGENT_ROOT), str(ORAK_SRC), str(SCRIPT_DIR)]:
    if Path(p).exists() and p not in sys.path:
        sys.path.insert(0, p)

from mcp_game_servers.pokemon_red.game.pokemon_red_env import PokemonRedEnv, PokemonRedObs  # type: ignore
from mcp_game_servers.pokemon_red.game.pyboy_runner import PyBoyRunner  # type: ignore
from mcp_game_servers.pokemon_red.game.utils.pokemon_tools import (  # type: ignore
    PokemonToolset,
    process_state_tool,
    execute_action_response,
)

from data_structure.experience import Experience, Episode, Episode_Buffer  # type: ignore
from policy_alfworld7b import Alfworld7BConfig, Alfworld7BPolicy  # type: ignore

GAME_NAME = "pokemon_red"
MAX_MEMORY_STEPS = 10
NO_PROGRESS_THRESHOLD = 80

VALID_ACTIONS = {"a", "b", "start", "select", "up", "down", "left", "right"}

ACTION_CANDIDATES = [
    "use_tool(continue_dialog, ())",
    "use_tool(move_to, (x_dest=3, y_dest=3))",
    "use_tool(warp_with_warp_point, (x_dest=3, y_dest=3))",
    "use_tool(run_away, ())",
    "use_tool(select_move_in_battle, (move_name=TACKLE))",
    "a", "b", "up", "down", "left", "right", "start",
]

# ---------------------------------------------------------------------------
# Lightweight agent wrapper (same as original)
# ---------------------------------------------------------------------------

@dataclass
class AgentMemory:
    state_dict: Dict[str, Any] = field(default_factory=dict)
    map_memory_dict: Dict[str, Any] = field(default_factory=dict)
    dialog_buffer: List[str] = field(default_factory=list)


class AgentShell:
    def __init__(self, env: PokemonRedEnv):
        self.env = env
        self.memory = AgentMemory()


# ---------------------------------------------------------------------------
# Orak env helper (reused from original)
# ---------------------------------------------------------------------------

def _suppress_map_warnings():
    import mcp_game_servers.pokemon_red.game.pyboy_runner as _pbr
    import io

    _orig_load_map = _pbr.load_map_module
    def _quiet_load_map(map_name):
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            return _orig_load_map(map_name)
        finally:
            sys.stdout = old_stdout
    _pbr.load_map_module = _quiet_load_map

    _orig_parse = _pbr.parse_object_sprites
    def _quiet_parse(asm_path):
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            return _orig_parse(asm_path)
        finally:
            sys.stdout = old_stdout
    _pbr.parse_object_sprites = _quiet_parse


def make_orak_env(rom_path: str, log_path: str, task: str = "DefeatBrock") -> PokemonRedEnv:
    import types

    _suppress_map_warnings()

    orak_root = str(ORAK_SRC.parent)
    os.chdir(orak_root)

    rom_abs = os.path.abspath(rom_path)
    sav_path = os.path.splitext(rom_abs)[0] + ".sav"
    if os.path.exists(sav_path):
        os.remove(sav_path)

    cfg = types.SimpleNamespace(
        log_path=log_path,
        task=task,
        rom_path=rom_path,
        success_condition="get_boulder_badge",
        input_modality="text",
    )
    env = PokemonRedEnv.__new__(PokemonRedEnv)
    env.cfg = cfg
    env.configure()
    return env


# ---------------------------------------------------------------------------
# Intro skip (same as original)
# ---------------------------------------------------------------------------

def skip_intro(runner: PyBoyRunner, max_presses: int = 350, verbose: bool = False, fast: bool = False):
    last_text = ""
    same = 0
    field_confirm = 0
    FIELD_CONFIRM_NEEDED = 3

    for i in range(max_presses):
        state = runner.get_battle_state()
        if state == "Field":
            mem = runner.pyboy.memory
            map_h = mem[0xD368]
            party_count = mem[0xD163]
            if map_h > 0 or party_count > 0:
                field_confirm += 1
                if field_confirm >= FIELD_CONFIRM_NEEDED:
                    if verbose:
                        print(f"  [intro] Field confirmed after {i} presses")
                    return
                time.sleep(0.02 if fast else 0.2)
                continue
            else:
                field_confirm = 0
        else:
            field_confirm = 0

        dialog = runner.get_dialog()
        text_portion = dialog.split("[Selection Box Text]")[0] if dialog else ""

        if text_portion == last_text:
            same += 1
        else:
            same = 0
            last_text = text_portion

        if same >= 5:
            runner.send_input("down")
            time.sleep(0.02 if fast else 0.15)
            runner.send_input("a")
            time.sleep(0.02 if fast else 0.15)
            same = 0
        else:
            runner.send_input("a")
            time.sleep(0.02 if fast else 0.15)

    if verbose:
        print(f"  [intro] WARNING: hit max presses ({max_presses})")


# ---------------------------------------------------------------------------
# Termination checks (same as original)
# ---------------------------------------------------------------------------

class ProgressTracker:
    def __init__(self, threshold: int = NO_PROGRESS_THRESHOLD):
        self.threshold = threshold
        self.last_location: Optional[str] = None
        self.last_badge_count: int = 0
        self.steps_at_location: int = 0
        self.had_dialog: bool = False

    def update(self, state_dict: Dict) -> bool:
        location = state_dict.get("map_info", {}).get("map_name")
        badge_text = state_dict.get("badge_list", "")
        badge_count = badge_text.count(",") + 1 if badge_text and badge_text != "N/A" else 0

        dialog_text = state_dict.get("filtered_screen_text", "N/A")
        has_dialog = dialog_text and dialog_text != "N/A"
        if has_dialog:
            self.had_dialog = True

        if badge_count > self.last_badge_count:
            self.last_badge_count = badge_count
            self.steps_at_location = 0
            self.had_dialog = False
            return False

        if location and location != self.last_location:
            self.last_location = location
            self.steps_at_location = 0
            self.had_dialog = False
            return False

        self.steps_at_location += 1
        return self.steps_at_location >= self.threshold and not self.had_dialog


def check_whiteout(state_dict: Dict) -> bool:
    party_text = state_dict.get("your_party", "")
    if not party_text or party_text == "N/A":
        return False
    hp_matches = re.findall(r"HP:\s*(\d+)/(\d+)", party_text)
    if not hp_matches:
        return False
    return all(int(cur) == 0 for cur, _ in hp_matches)


# ---------------------------------------------------------------------------
# Action execution (same as original)
# ---------------------------------------------------------------------------

_NAV_TOOLS = {"move_to", "warp_with_warp_point", "interact_with_object",
              "overworld_map_transition"}


def _has_map_data(toolset: PokemonToolset) -> bool:
    try:
        map_name = toolset.agent.memory.state_dict["map_info"]["map_name"]
        explored = toolset.agent.memory.map_memory_dict.get(map_name, {}).get("explored_map")
        return explored is not None and len(explored) > 0 and len(explored[0]) > 0
    except (KeyError, TypeError):
        return False


def execute_action(action_str: str, toolset: PokemonToolset,
                   env: PokemonRedEnv, fast: bool = False) -> str:
    action_str = action_str.split("\n")[0].strip()
    parts = [p.strip() for p in re.split(r'\s*\|\s*', action_str) if p.strip()]
    parts = parts[:5]

    results = []
    for part in parts:
        if part.startswith("use_tool("):
            tool_match = re.match(r"use_tool\((\w+)", part)
            tool_name = tool_match.group(1) if tool_match else ""
            if tool_name in _NAV_TOOLS and not _has_map_data(toolset):
                results.append(f"{part} -> (False, 'No map data')")
                continue
            result = execute_action_response(toolset, part)
            results.append(f"{part} -> {result}")
        elif part.lower() in VALID_ACTIONS:
            env.send_action_set([part.lower()])
            time.sleep(0.05 if fast else 0.3)
            results.append(part.lower())
        else:
            results.append(f"(ignored: {part})")

    return " | ".join(results)


# ---------------------------------------------------------------------------
# Simple rolling memory (no LLM reflection)
# ---------------------------------------------------------------------------

class RollingMemory:
    def __init__(self, max_steps: int = MAX_MEMORY_STEPS):
        self.max_steps = max_steps
        self.history: List[Dict[str, str]] = []

    def add(self, step: int, action: str, state_summary: str):
        self.history.append({"step": step, "action": action, "summary": state_summary})
        if len(self.history) > self.max_steps:
            self.history = self.history[-self.max_steps:]

    def format_history(self) -> str:
        if not self.history:
            return "(no previous actions)"
        return "\n".join(
            f"  Step {h['step']}: {h['action']} -> {h['summary'][:80]}"
            for h in self.history
        )


# ---------------------------------------------------------------------------
# ALFWorld-7B decision function
# ---------------------------------------------------------------------------

def alfworld7b_decide(
    state_text: str,
    memory: RollingMemory,
    policy: Alfworld7BPolicy,
    state_dict: Dict,
) -> str:
    game_state = state_dict.get("state", "Field")

    if game_state == "Battle":
        candidates = [
            "use_tool(select_move_in_battle, (move_name=TACKLE))",
            "use_tool(select_move_in_battle, (move_name=SCRATCH))",
            "use_tool(select_move_in_battle, (move_name=POUND))",
            "use_tool(run_away, ())",
            "a", "b", "up", "down",
        ]
    elif game_state == "Dialog":
        candidates = [
            "use_tool(continue_dialog, ())",
            "a", "b", "up", "down",
        ]
    else:
        candidates = list(ACTION_CANDIDATES)

    prompt = (
        f"Pokemon Red game state:\n{state_text[:3000]}\n\n"
        f"Recent history:\n{memory.format_history()}\n\n"
        "Choose your next action."
    )

    action = policy.choose_action(prompt, candidates)
    return action


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_pokemon_episode(
    rom_path: str,
    policy: Alfworld7BPolicy,
    max_steps: int = 200,
    verbose: bool = False,
    log_path: str = "/tmp/pokemon_cold_start",
    fast: bool = False,
) -> Tuple[Episode, Dict[str, Any]]:
    os.makedirs(log_path, exist_ok=True)

    env = make_orak_env(rom_path, log_path)
    agent = AgentShell(env)
    toolset = PokemonToolset(agent)

    if verbose:
        print("  [env] Skipping intro...")
    skip_intro(env.runner, verbose=verbose, fast=fast)

    state_text = env._receive_state()
    agent.memory.state_dict = env.parse_game_state(state_text)
    agent.memory.map_memory_dict = toolset.get_map_memory_dict(
        agent.memory.state_dict, agent.memory.map_memory_dict
    )

    memory = RollingMemory()
    progress = ProgressTracker()
    experiences: List[Experience] = []
    total_reward = 0.0
    step_count = 0
    termination_reason = "max_steps"

    task_desc = "Play Pokemon Red: explore, battle, earn badges."
    action_names = list(VALID_ACTIONS)

    state_text = env._receive_state()
    env.state_dict = env.parse_game_state(state_text)
    env.prev_state_dict = dict(env.state_dict)
    score_str = "0.0 (0/12)"

    while step_count < max_steps:
        state_text = env._receive_state()
        state_dict = env.parse_game_state(state_text)
        agent.memory.state_dict = state_dict

        env.prev_state_dict = dict(env.state_dict)
        env.state_dict = state_dict
        env.state_text = state_text

        agent.memory.map_memory_dict = toolset.get_map_memory_dict(
            state_dict, agent.memory.map_memory_dict
        )

        processed_text, state_dict, agent.memory.map_memory_dict, _, agent.memory.dialog_buffer = \
            process_state_tool(
                env, toolset, agent.memory.map_memory_dict,
                step_count, agent.memory.dialog_buffer, state_text
            )
        agent.memory.state_dict = state_dict

        if check_whiteout(state_dict):
            if verbose:
                print(f"    [TERM] Whiteout at step {step_count}")
            termination_reason = "whiteout"
            break

        if progress.update(state_dict):
            if verbose:
                print(f"    [TERM] No progress for {NO_PROGRESS_THRESHOLD} steps")
            termination_reason = "no_progress"
            break

        try:
            score_str, score_done = env.evaluate(PokemonRedObs(state_text=state_text))
        except (KeyError, TypeError):
            score_done = False
        if score_done:
            if verbose:
                print(f"    [TERM] Score completion: {score_str}")
            termination_reason = "score_complete"
            break

        action_str = alfworld7b_decide(processed_text, memory, policy, state_dict)

        feedback = execute_action(action_str, toolset, env, fast=fast)
        step_count += 1

        next_state_text = env._receive_state()
        next_state_dict = env.parse_game_state(next_state_text)
        agent.memory.state_dict = next_state_dict

        env.prev_state_dict = dict(env.state_dict)
        env.state_dict = next_state_dict
        env.state_text = next_state_text

        try:
            score_str, score_done = env.evaluate(PokemonRedObs(state_text=next_state_text))
        except (KeyError, TypeError):
            score_done = False
        reward = 0.0
        try:
            score_val = float(score_str.split("(")[0].strip())
            reward = score_val / 100.0
        except Exception:
            pass
        total_reward += reward

        summary = (f"State={next_state_dict.get('state','?')} "
                   f"Map={next_state_dict.get('map_info',{}).get('map_name','?')} "
                   f"Score={score_str}")
        memory.add(step_count, action_str, summary)

        exp = Experience(
            state=processed_text,
            action=action_str,
            reward=float(reward),
            next_state=next_state_text,
            done=score_done,
            intentions=None,
            tasks=task_desc,
        )
        exp.idx = step_count - 1
        exp.action_type = "tool" if "use_tool" in action_str else "primitive"
        exp.available_actions = action_names
        exp.interface = {"env_name": "orak", "game_name": GAME_NAME}
        exp.raw_state = str(state_text) if state_text else None
        exp.raw_next_state = str(next_state_text) if next_state_text else None
        experiences.append(exp)

        if verbose:
            loc = next_state_dict.get("map_info", {}).get("map_name", "?")
            act_short = action_str[:50]
            fb = feedback if len(feedback) <= 200 else feedback[:197] + "..."
            print(f"  step {step_count}: {act_short} @ {loc}  score={score_str}  feedback={fb}")

        if score_done:
            termination_reason = "score_complete"
            break

    try:
        env.runner.running = False
        time.sleep(0.1 if fast else 0.5)
    except Exception:
        pass

    episode = Episode(experiences=experiences, task=task_desc, env_name="orak", game_name=GAME_NAME)
    episode.set_outcome()

    terminated = termination_reason in ("score_complete", "whiteout")
    truncated = termination_reason in ("max_steps", "no_progress")
    stats = {
        "game": GAME_NAME,
        "steps": step_count,
        "total_reward": total_reward,
        "termination_reason": termination_reason,
        "terminated": terminated,
        "truncated": truncated,
        "model": policy.cfg.model_path,
        "checkpoint_type": policy.cfg.checkpoint_type,
        "agent_type": "alfworld7b_orak_toolset",
        "final_location": state_dict.get("map_info", {}).get("map_name"),
        "final_score": score_str,
    }
    return episode, stats


# ---------------------------------------------------------------------------
# Batch rollout helpers
# ---------------------------------------------------------------------------

def count_existing_episodes(game_dir: Path) -> int:
    if not game_dir.exists():
        return 0
    return sum(1 for f in game_dir.glob("episode_*.json") if f.name != "episode_buffer.json")


def save_episode_jsonl(episode: Episode, jsonl_path: Path, stats: Dict):
    record = episode.to_dict()
    record["rollout_metadata"] = stats
    with open(jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def run_all_episodes(args, output_dir: Path, policy: Alfworld7BPolicy) -> Dict:
    game_dir = output_dir / GAME_NAME
    game_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = game_dir / "rollouts.jsonl"

    start_idx = 0
    if args.resume:
        start_idx = count_existing_episodes(game_dir)
        if start_idx >= args.episodes:
            print(f"  [SKIP] {start_idx}/{args.episodes} episodes already done")
            return {"game": GAME_NAME, "skipped": True}
        if start_idx:
            print(f"  [RESUME] from episode {start_idx}")

    effective_max_steps = args.max_steps or 200

    episode_buffer = Episode_Buffer(buffer_size=args.episodes + 10)
    all_stats: List[Dict] = []
    t0 = time.time()

    for ep_idx in range(start_idx, args.episodes):
        print(f"\n  [pokemon_red] Episode {ep_idx + 1}/{args.episodes}")
        ep_log = str(output_dir / "logs" / f"ep_{ep_idx:03d}")

        try:
            episode, stats = run_pokemon_episode(
                rom_path=args.rom_path,
                policy=policy,
                max_steps=effective_max_steps,
                verbose=args.verbose,
                log_path=ep_log,
                fast=args.fast,
            )

            stats["episode_index"] = ep_idx
            print(f"    Steps: {stats['steps']}, Reward: {stats['total_reward']:.2f}, "
                  f"End: {stats['termination_reason']}, "
                  f"Location: {stats.get('final_location', '?')}, "
                  f"Score: {stats.get('final_score', '?')}")

            if args.label and not args.no_label:
                try:
                    from cold_start.generate_cold_start import label_trajectory
                    episode = label_trajectory(episode, args.label_model)
                except ImportError:
                    pass

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
            all_stats.append({"game": GAME_NAME, "episode_index": ep_idx,
                              "error": str(e), "steps": 0, "total_reward": 0.0})

    elapsed = time.time() - t0
    buffer_path = game_dir / "episode_buffer.json"
    episode_buffer.save_to_json(str(buffer_path))
    print(f"\n  Saved {len(episode_buffer)} episodes to {buffer_path}")

    summary: Dict = {
        "game": GAME_NAME,
        "timestamp": datetime.now().isoformat(),
        "model": args.model_path,
        "checkpoint_type": args.checkpoint_type,
        "agent_type": "alfworld7b_orak_toolset",
        "total_episodes": len(all_stats),
        "target_episodes": args.episodes,
        "max_steps": effective_max_steps,
        "labeled": args.label and not args.no_label,
        "elapsed_seconds": elapsed,
        "episode_stats": all_stats,
    }
    good = [s for s in all_stats if "error" not in s]
    if good:
        rewards = [s.get("total_reward", 0) for s in good]
        steps = [s.get("steps", 0) for s in good]
        summary["mean_reward"] = sum(rewards) / len(rewards)
        summary["mean_steps"] = sum(steps) / len(steps)
        summary["max_reward"] = max(rewards)
        summary["min_reward"] = min(rewards)
        reasons = {}
        for s in good:
            r = s.get("termination_reason", "?")
            reasons[r] = reasons.get(r, 0) + 1
        summary["termination_reasons"] = reasons

    with open(game_dir / "rollout_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Pokemon Red cold-start using ALFWorld-7B + Orak env + toolset",
    )
    parser.add_argument("--rom_path", type=str, default=None,
                        help="Path to Pokemon Red .gb/.gbc ROM")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--model_path", type=str, required=True,
                        help="HF model id or local path for ALFWorld-7B.")
    parser.add_argument("--checkpoint_type", type=str, default="sft",
                        choices=["sft", "rl"])
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--label", action="store_true")
    parser.add_argument("--no_label", action="store_true")
    parser.add_argument("--label_model", type=str, default="gpt-5-mini")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--fast", action="store_true",
                        help="Max speed: no emulator throttle, shorter sleeps")

    args = parser.parse_args()

    if args.fast:
        os.environ["PYBOY_FRAME_TIME"] = "0"
        os.environ["POKEMON_STARTUP_DELAY"] = "1"

    if not args.rom_path:
        candidates = [
            CODEBASE_ROOT.parent / "GamingAgent" / "gamingagent" / "configs" / "custom_06_pokemon_red" / "rom" / "pokemon.gb",
            CODEBASE_ROOT.parent / "ROMs" / "Pokemon - Red Version (USA, Europe).gb",
        ]
        for c in candidates:
            if c.exists():
                args.rom_path = str(c)
                break
        if not args.rom_path:
            print("[ERROR] ROM not found. Provide --rom_path or place ROM at:")
            for c in candidates:
                print(f"  {c}")
            sys.exit(1)

    output_dir = Path(args.output_dir) if args.output_dir else COLD_START_DIR / "output" / f"alfworld7b_{args.checkpoint_type}_pokemon_red"
    output_dir.mkdir(parents=True, exist_ok=True)

    effective_max = args.max_steps or 200

    cfg = Alfworld7BConfig(
        model_path=args.model_path,
        checkpoint_type=args.checkpoint_type,
        temperature=args.temperature,
    )
    policy = Alfworld7BPolicy(cfg)

    print("=" * 72)
    print("  Pokemon Red Cold-Start (ALFWorld-7B + Orak Env + Toolset)")
    print("=" * 72)
    print(f"  ROM:             {args.rom_path}")
    print(f"  Episodes:        {args.episodes}")
    print(f"  Max steps:       {effective_max}")
    print(f"  Model:           {args.model_path} ({args.checkpoint_type})")
    print(f"  Temp:            {args.temperature}")
    print(f"  No-progress cap: {NO_PROGRESS_THRESHOLD}")
    print(f"  Fast mode:       {args.fast}")
    print(f"  Labeling:        {args.label and not args.no_label}")
    print(f"  Output:          {output_dir}")
    print("=" * 72)

    summary = run_all_episodes(args, output_dir, policy)

    print(f"\n{'=' * 72}")
    print("  COMPLETE")
    print(f"{'=' * 72}")
    if not summary.get("skipped"):
        print(f"  Episodes:  {summary.get('total_episodes', 0)}")
        print(f"  Elapsed:   {summary.get('elapsed_seconds', 0):.1f}s")
        if "mean_reward" in summary:
            print(f"  Avg reward: {summary['mean_reward']:.2f}")
            print(f"  Avg steps:  {summary['mean_steps']:.1f}")
        if "termination_reasons" in summary:
            print(f"  End reasons: {summary['termination_reasons']}")
    print(f"  Output:    {output_dir / GAME_NAME}")
    print(f"{'=' * 72}\n")


if __name__ == "__main__":
    main()
