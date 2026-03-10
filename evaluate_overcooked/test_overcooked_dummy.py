#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test script: run dummy_agent in Overcooked_ai via OvercookedNLWrapper.

Both agents (Chef 0 and Chef 1) are driven by the dummy language agent.
Each step prints the natural-language state and chosen action for every agent.
Optionally renders a live pygame GUI.

Usage (from the codebase root):
    conda activate overcooked_eval
    python evaluate_overcooked/test_overcooked_dummy.py

Options:
    --layout        Layout name (default: cramped_room)
    --horizon       Episode length (default: 100)
    --episodes      Number of episodes to run (default: 1)
    --mode          Agent mode: "llm" | "random_nl" | "fallback" (default: fallback)
    --model         LLM model name for "llm" mode (default: gpt-4o-mini)
    --gui           Show live pygame GUI
    --gui-delay     Milliseconds to pause after each GUI frame (default: 200)
    --verbose       Print full NL observations (otherwise prints a compact summary)

Examples:
    python evaluate_overcooked/test_overcooked_dummy.py --mode random_nl --episodes 2 --gui
    python evaluate_overcooked/test_overcooked_dummy.py --mode random_nl --verbose
    python evaluate_overcooked/test_overcooked_dummy.py --mode llm --model gpt-4o-mini --gui --gui-delay 500
"""

import argparse
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
CODEBASE_ROOT = SCRIPT_DIR.parent
OVERCOOKED_SRC = CODEBASE_ROOT / "overcooked_ai" / "src"

for p in [str(CODEBASE_ROOT), str(OVERCOOKED_SRC)]:
    if p not in sys.path:
        sys.path.insert(0, p)

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.actions import Action, Direction
from env_wrappers.overcooked_nl_wrapper import (
    OvercookedNLWrapper,
    state_to_natural_language,
    state_to_natural_language_for_all_agents,
)
from agents.dummy_agent import (
    language_agent_action,
    GAME_OVERCOOKED,
    OVERCOOKED_VALID_ACTIONS,
    _default_action,
    run_episode_with_experience_collection,
    AgentBufferManager,
)


# ---------------------------------------------------------------------------
# Action index <-> human-readable name
# ---------------------------------------------------------------------------
ACTION_INDEX_TO_NAME = {
    0: "north",
    1: "south",
    2: "east",
    3: "west",
    4: "stay",
    5: "interact",
}
ACTION_NAME_TO_INDEX = {v: k for k, v in ACTION_INDEX_TO_NAME.items()}


COMMON_LAYOUTS = [
    "cramped_room",
    "asymmetric_advantages",
    "coordination_ring",
    "forced_coordination",
    "counter_circuit",
    "cramped_room_single",
]


# ---------------------------------------------------------------------------
# Thin adapter: make OvercookedEnv usable by OvercookedNLWrapper
# ---------------------------------------------------------------------------
class OvercookedMultiAgentAdapter:
    """
    Minimal gymnasium-like adapter around the raw OvercookedEnv so that
    OvercookedNLWrapper(multi_agent=True) can call reset() and step().

    reset() -> (obs, info)         where info contains 'overcooked_state'
    step()  -> (state, r, done, info)  (4-tuple handled by NLWrapper)
    """

    def __init__(self, layout_name: str = "cramped_room", horizon: int = 400):
        mdp = OvercookedGridworld.from_layout_name(layout_name)
        self.base_env = OvercookedEnv.from_mdp(mdp, horizon=horizon, info_level=0)
        self.horizon = horizon

    def reset(self, seed=None, options=None):
        self.base_env.reset()
        state = self.base_env.state
        info = {"overcooked_state": state}
        return None, info

    def step(self, joint_action):
        """joint_action: tuple of two Action objects (or index tuples)."""
        next_state, reward, done, env_info = self.base_env.step(joint_action)
        env_info["overcooked_state"] = next_state
        # Return 4-tuple; NLWrapper's multi_agent branch handles this
        return next_state, reward, done, env_info

    @property
    def state(self):
        return self.base_env.state


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def random_nl_action() -> str:
    """Pick a random valid Overcooked NL action."""
    return random.choice(list(OVERCOOKED_VALID_ACTIONS))


def choose_action(obs_nl: str, mode: str, model: str) -> str:
    """Choose an NL action string for one agent based on mode."""
    if mode == "llm":
        try:
            action = language_agent_action(
                state_nl=obs_nl,
                game=GAME_OVERCOOKED,
                model=model,
                use_function_call=True,
                temperature=0.3,
            )
            return action if action else _default_action(GAME_OVERCOOKED)
        except Exception as e:
            print(f"  [WARNING] LLM action failed: {e}, using fallback action")
            return _default_action(GAME_OVERCOOKED)
    elif mode == "random_nl":
        return random_nl_action()
    else:  # fallback
        return _default_action(GAME_OVERCOOKED)


def format_player_line(state, player_idx: int) -> str:
    """One-line compact summary of a player's position/orientation/held item."""
    players = getattr(state, "players", ())
    if player_idx >= len(players):
        return "N/A"
    p = players[player_idx]
    pos = getattr(p, "position", "?")
    ori_tuple = getattr(p, "orientation", (0, -1))
    ori_name = {(0, -1): "N", (0, 1): "S", (1, 0): "E", (-1, 0): "W"}.get(
        tuple(ori_tuple), "?"
    )
    held = getattr(p, "held_object", None)
    held_str = getattr(held, "name", "nothing") if held else "nothing"
    return f"pos={pos} facing={ori_name} holding={held_str}"


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_episode(
    layout_name: str,
    horizon: int,
    mode: str,
    model: str,
    gui: bool,
    gui_delay: int,
    verbose: bool,
    episode_id: int,
    use_experience_collection: bool = False,
    buffer_manager: Optional[AgentBufferManager] = None,
) -> dict:
    """
    Run one multi-agent episode.  Both Chef 0 and Chef 1 are driven by the
    dummy agent.  Prints state + action for every agent each step.
    
    Args:
        use_experience_collection: If True, use experience/episode collection.
        buffer_manager: Optional buffer manager for experience collection.
    """

    # Use experience collection if requested
    if use_experience_collection and buffer_manager is not None:
        return _run_episode_with_experience_collection(
            layout_name=layout_name,
            horizon=horizon,
            mode=mode,
            model=model,
            gui=gui,
            gui_delay=gui_delay,
            verbose=verbose,
            episode_id=episode_id,
            buffer_manager=buffer_manager,
        )

    # 1. Build multi-agent env
    adapter = OvercookedMultiAgentAdapter(layout_name=layout_name, horizon=horizon)
    env = OvercookedNLWrapper(
        env=adapter,
        horizon=horizon,
        multi_agent=True,
        show_gui=gui,
        gui_delay_ms=gui_delay if gui else None,
    )

    # 2. Reset
    obs_list, info = env.reset()  # obs_list = [nl_agent0, nl_agent1]

    total_reward = 0.0
    step_count = 0
    actions_log: List[Tuple[str, str]] = []  # (action_agent0, action_agent1)

    print(f"\n{'='*78}")
    print(f"  Episode {episode_id + 1}  |  Layout: {layout_name}  |  "
          f"Horizon: {horizon}  |  Mode: {mode}")
    print(f"{'='*78}")

    # Print initial state for both agents
    state = info.get("overcooked_state")
    _print_step_header(0, state, obs_list, None, None, 0.0, total_reward, verbose)

    # 3. Episode loop
    terminated = False
    truncated = False

    while not (terminated or truncated):
        step_count += 1

        # Each agent chooses its own action from its own NL observation
        action_0 = choose_action(obs_list[0], mode, model)
        action_1 = choose_action(obs_list[1], mode, model)
        actions_log.append((action_0, action_1))

        # Step with joint action
        try:
            obs_list, reward, terminated, truncated, info = env.step(
                [action_0, action_1]
            )
        except Exception as e:
            print(f"\n  [ERROR at step {step_count}] {e}")
            break

        total_reward += reward
        state = info.get("overcooked_state")

        # Always print state and actions (not just in verbose mode)
        _print_step_header(
            step_count, state, obs_list,
            action_0, action_1,
            reward, total_reward, verbose,
        )

    # 4. Episode summary
    action_counts_0: Dict[str, int] = {}
    action_counts_1: Dict[str, int] = {}
    for a0, a1 in actions_log:
        action_counts_0[a0] = action_counts_0.get(a0, 0) + 1
        action_counts_1[a1] = action_counts_1.get(a1, 0) + 1

    result = {
        "episode_id": episode_id,
        "layout": layout_name,
        "horizon": horizon,
        "mode": mode,
        "steps": step_count,
        "total_reward": total_reward,
        "terminated": terminated,
        "truncated": truncated,
        "action_distribution_agent0": action_counts_0,
        "action_distribution_agent1": action_counts_1,
    }

    print(f"\n{'-'*78}")
    print(f"  Episode {episode_id + 1} Summary")
    print(f"{'-'*78}")
    print(f"  Steps:         {step_count}")
    print(f"  Total Reward:  {total_reward:.2f}")
    print(f"  Terminated:    {terminated}")
    print(f"  Truncated:     {truncated}")
    print(f"  Chef 0 actions: {action_counts_0}")
    print(f"  Chef 1 actions: {action_counts_1}")

    if gui:
        env.close_gui()

    return result


def _run_episode_with_experience_collection(
    layout_name: str,
    horizon: int,
    mode: str,
    model: str,
    gui: bool,
    gui_delay: int,
    verbose: bool,
    episode_id: int,
    buffer_manager: AgentBufferManager,
) -> dict:
    """
    Run one episode with experience collection.
    Manually collects experiences since Overcooked uses list-based observations.
    """
    from data_structure.experience import Experience, Episode
    
    # Build multi-agent env
    adapter = OvercookedMultiAgentAdapter(layout_name=layout_name, horizon=horizon)
    env = OvercookedNLWrapper(
        env=adapter,
        horizon=horizon,
        multi_agent=True,
        show_gui=gui,
        gui_delay_ms=gui_delay if gui else None,
    )
    
    # Reset
    obs_list, info = env.reset()
    experiences = []
    total_reward = 0.0
    step_count = 0
    action_counts_0: Dict[str, int] = {}
    action_counts_1: Dict[str, int] = {}
    
    task = f"Overcooked episode {episode_id + 1} - Layout: {layout_name}"
    
    # Always print episode header (not just verbose)
    print(f"\n{'='*78}")
    print(f"  Episode {episode_id + 1}  |  Layout: {layout_name}  |  "
          f"Horizon: {horizon}  |  Mode: {mode} (with experience collection)")
    print(f"{'='*78}")
    
    prev_state = str(obs_list) if isinstance(obs_list, list) else str(obs_list)
    terminated = False
    truncated = False
    
    while not (terminated or truncated) and step_count < horizon:
        step_count += 1
        
        # Choose actions
        action_0 = choose_action(obs_list[0], mode, model)
        action_1 = choose_action(obs_list[1], mode, model)
        joint_action = [action_0, action_1]
        
        action_counts_0[action_0] = action_counts_0.get(action_0, 0) + 1
        action_counts_1[action_1] = action_counts_1.get(action_1, 0) + 1
        
        # Step environment
        try:
            next_obs_list, reward, terminated, truncated, next_info = env.step(joint_action)
        except Exception as e:
            if verbose:
                print(f"\n  [ERROR at step {step_count}] {e}")
            break
        
        total_reward += reward
        next_state = str(next_obs_list) if isinstance(next_obs_list, list) else str(next_obs_list)
        done = terminated or truncated
        
        # Create experience
        experience = Experience(
            state=prev_state,
            action=joint_action,  # Store as list for multi-agent
            reward=float(reward),
            next_state=next_state,
            done=done,
            tasks=task,
            sub_tasks=None,
        )
        experience.idx = step_count - 1
        experiences.append(experience)
        
        # Always print state and actions for experience collection mode (not just verbose)
        state_obj = next_info.get("overcooked_state")
        print(f"\n  Step {step_count}:")
        print(f"    State: {format_player_line(state_obj, 0) if state_obj else 'N/A'} | {format_player_line(state_obj, 1) if state_obj else 'N/A'}")
        print(f"    Actions: Chef 0 -> {action_0}, Chef 1 -> {action_1}")
        print(f"    Reward: {reward:.2f} | Cumulative: {total_reward:.2f}")
        
        # Update for next iteration
        obs_list = next_obs_list
        info = next_info
        prev_state = next_state
    
    # Create episode
    episode = Episode(
        experiences=experiences,
        task=task,
        env_name="overcooked",
        game_name="overcooked",
    )
    episode.set_outcome()
    
    # Add to buffers
    buffer_manager.experience_buffer.add_experience(episode)
    buffer_manager.episode_buffer.add_episode(episode)
    
    if gui:
        env.close_gui()
    
    result = {
        "episode_id": episode_id,
        "layout": layout_name,
        "horizon": horizon,
        "mode": mode,
        "steps": len(experiences),
        "total_reward": total_reward,
        "terminated": terminated,
        "truncated": truncated,
        "action_distribution_agent0": action_counts_0,
        "action_distribution_agent1": action_counts_1,
        "episode": episode,
    }
    
    if verbose:
        print(f"\n{'-'*78}")
        print(f"  Episode {episode_id + 1} Summary (with experience collection)")
        print(f"{'-'*78}")
        print(f"  Steps:         {len(experiences)}")
        print(f"  Total Reward:  {total_reward:.2f}")
        print(f"  Terminated:    {terminated}")
        print(f"  Truncated:     {truncated}")
        print(f"  Chef 0 actions: {action_counts_0}")
        print(f"  Chef 1 actions: {action_counts_1}")
    
    return result


def _print_step_header(
    step: int,
    state: Any,
    obs_list: List[str],
    action_0: Optional[str],
    action_1: Optional[str],
    reward: float,
    cumulative_reward: float,
    verbose: bool,
) -> None:
    """Print per-step info for both agents with detailed state and actions."""

    tag = f"Step {step}"
    if step == 0:
        tag = "Initial State"

    print(f"\n{'='*78}")
    print(f"  {tag}")
    print(f"{'='*78}")

    # Print state information
    if state is not None:
        print(f"\n  STATE:")
        print(f"    Chef 0: {format_player_line(state, 0)}")
        print(f"    Chef 1: {format_player_line(state, 1)}")
        
        # Print additional state details if available
        if hasattr(state, "objects"):
            print(f"    Objects in environment: {len(state.objects) if state.objects else 0}")
        if hasattr(state, "terrain"):
            print(f"    Terrain: {type(state.terrain).__name__}")

    # Print actions taken (not shown at step 0)
    if action_0 is not None:
        print(f"\n  ACTIONS TAKEN:")
        print(f"    Chef 0 -> {action_0}")
        print(f"    Chef 1 -> {action_1}")

    # Print reward information
    if step > 0:
        print(f"\n  REWARD:")
        print(f"    Step reward: {reward:.2f}")
        print(f"    Cumulative reward: {cumulative_reward:.2f}")

    # Full NL observations (verbose mode)
    if verbose and obs_list:
        print(f"\n  OBSERVATIONS:")
        print(f"    -- Chef 0 observation --")
        for line in obs_list[0].split("\n"):
            if line.strip():
                print(f"    {line}")
        print(f"    -- Chef 1 observation --")
        for line in obs_list[1].split("\n"):
            if line.strip():
                print(f"    {line}")

    print(f"{'='*78}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Test dummy agent in Overcooked (multi-agent, both chefs visible)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available layouts (common ones):
  {', '.join(COMMON_LAYOUTS)}

Modes:
  fallback   - Default action ("stay") every step. No API key needed.
  random_nl  - Random valid NL action each step. No API key needed.
  llm        - LLM-driven actions (e.g. GPT-4o-mini). Needs OPENAI_API_KEY.
        """,
    )

    parser.add_argument("--layout", type=str, default="cramped_room",
                        help="Layout name (default: cramped_room)")
    parser.add_argument("--horizon", type=int, default=100,
                        help="Episode length (default: 100)")
    parser.add_argument("--episodes", type=int, default=1,
                        help="Number of episodes (default: 1)")
    parser.add_argument("--mode", type=str, default="fallback",
                        choices=["llm", "random_nl", "fallback"],
                        help="Agent mode (default: fallback)")
    parser.add_argument("--model", type=str, default="gpt-4o-mini",
                        help="LLM model name for llm mode (default: gpt-4o-mini)")
    parser.add_argument("--gui", action="store_true",
                        help="Show live pygame GUI window")
    parser.add_argument("--gui-delay", type=int, default=200,
                        help="Milliseconds to pause per GUI frame (default: 200)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print full NL observations each step")
    parser.add_argument("--use_experience_collection", action="store_true",
                        help="Use experience/episode collection with buffers")
    parser.add_argument("--experience_buffer_size", type=int, default=10000,
                        help="Size of experience buffer (default: 10000)")
    parser.add_argument("--episode_buffer_size", type=int, default=1000,
                        help="Size of episode buffer (default: 1000)")
    parser.add_argument("--save_episode_buffer", type=str, default=None,
                        help="Path to save episode buffer as JSON file after all episodes")

    args = parser.parse_args()

    # Warn if LLM mode but no API key
    if args.mode == "llm":
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            print("[WARNING] Mode is 'llm' but OPENAI_API_KEY is not set.")
            print("          The agent will fall back to default actions.")
            print("          Set OPENAI_API_KEY or use --mode fallback/random_nl.\n")

    print(f"Overcooked Dummy Agent Test  (multi-agent, both chefs)")
    print(f"  Layout:    {args.layout}")
    print(f"  Horizon:   {args.horizon}")
    print(f"  Episodes:  {args.episodes}")
    print(f"  Mode:      {args.mode}")
    if args.mode == "llm":
        print(f"  Model:     {args.model}")
    print(f"  GUI:       {'ON' if args.gui else 'OFF'}"
          + (f"  (delay {args.gui_delay}ms)" if args.gui else ""))

    # Create buffer manager if using experience collection
    buffer_manager = None
    if args.use_experience_collection:
        buffer_manager = AgentBufferManager(
            experience_buffer_size=args.experience_buffer_size,
            episode_buffer_size=args.episode_buffer_size,
        )
    
    # Run episodes
    all_results = []
    t0 = time.time()

    for ep in range(args.episodes):
        result = run_episode(
            layout_name=args.layout,
            horizon=args.horizon,
            mode=args.mode,
            model=args.model,
            gui=args.gui,
            gui_delay=args.gui_delay,
            verbose=args.verbose,
            episode_id=ep,
            use_experience_collection=args.use_experience_collection,
            buffer_manager=buffer_manager,
        )
        all_results.append(result)
    
    # Save episode buffer if requested
    if args.use_experience_collection and buffer_manager and args.save_episode_buffer:
        buffer_manager.save_episode_buffer(args.save_episode_buffer)
        print(f"\nSaved episode buffer to {args.save_episode_buffer}")
        print(f"Buffer stats: {buffer_manager.get_buffer_stats()}")

    elapsed = time.time() - t0

    # Aggregate
    print(f"\n{'='*78}")
    print(f"  OVERALL RESULTS  ({args.episodes} episode(s))")
    print(f"{'='*78}")

    total_rewards = [r["total_reward"] for r in all_results]
    total_steps = [r["steps"] for r in all_results]

    print(f"  Mean Reward:  {sum(total_rewards) / len(total_rewards):.2f}")
    print(f"  Max Reward:   {max(total_rewards):.2f}")
    print(f"  Min Reward:   {min(total_rewards):.2f}")
    print(f"  Mean Steps:   {sum(total_steps) / len(total_steps):.1f}")
    print(f"  Total Time:   {elapsed:.2f}s")
    print(f"{'='*78}\n")

    return all_results


if __name__ == "__main__":
    main()
