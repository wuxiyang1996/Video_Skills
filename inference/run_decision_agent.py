# Run VLM decision agent and store rollouts as Episode (data_structure format).
# Converts run_episode_vlm_agent result into Experience list + Episode; optional buffer/store.

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from decision_agents import run_episode_vlm_agent, VLMDecisionAgent

try:
    from data_structure.experience import Experience, Episode, Episode_Buffer, Experience_Replay_Buffer
except ImportError:
    Experience = None
    Episode = None
    Episode_Buffer = None
    Experience_Replay_Buffer = None


def rollout_to_episode(
    rollout: Dict[str, Any],
    task: str = "",
):
    """Convert run_episode_vlm_agent result to Episode (data_structure format)."""
    """
    Convert a decision-agent rollout (run_episode_vlm_agent result) into an Episode
    using the data_structure format (Experience list + Episode).

    rollout must have keys: observations (list of state strings), actions, rewards,
    and optionally done. Step i uses state=observations[i], action=actions[i],
    reward=rewards[i], next_state=observations[i+1], done only on the last step.
    """
    if Episode is None or Experience is None:
        raise ImportError(
            "Experience and Episode are required from data_structure.experience. "
            "Install or add the data_structure package."
        )
    observations: List[str] = list(rollout.get("observations", []))
    actions: List[Any] = list(rollout.get("actions", []))
    rewards: List[float] = list(rollout.get("rewards", []))
    done_flag: bool = bool(rollout.get("done", False))

    experiences: List[Experience] = []
    n = len(actions)
    if n == 0:
        ep = Episode(experiences=experiences, task=task or "Unspecified task")
        ep.set_outcome()
        return ep

    for i in range(n):
        state = observations[i] if i < len(observations) else ""
        next_state = observations[i + 1] if i + 1 < len(observations) else ""
        action = actions[i] if i < len(actions) else None
        reward = float(rewards[i]) if i < len(rewards) else 0.0
        done = (i == n - 1) and done_flag

        exp = Experience(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
            intentions=None,
            tasks=task if task else None,
            sub_tasks=None,
        )
        exp.idx = i
        experiences.append(exp)

    episode = Episode(experiences=experiences, task=task or "Unspecified task")
    episode.set_outcome()
    return episode


def run_inference(
    env,
    agent: Optional[VLMDecisionAgent] = None,
    *,
    task: str = "",
    model: Optional[str] = None,
    skill_bank: Any = None,
    memory: Any = None,
    reward_config: Any = None,
    max_steps: int = 1000,
    verbose: bool = False,
    episode_buffer: Optional[Episode_Buffer] = None,
    experience_buffer: Optional[Experience_Replay_Buffer] = None,
    save_path: Optional[str] = None,
):
    """
    Run one episode with the VLM decision agent and return/store the rollout
    in data_structure format (Episode with Experience list).

    - Runs run_episode_vlm_agent(env, agent=..., max_steps=..., verbose=...).
    - Converts the result to Episode via rollout_to_episode(..., task=task).
    - If episode_buffer is provided, adds the episode to it.
    - If experience_buffer is provided, adds the episode's experiences to it.
    - If save_path is provided, appends the episode to a JSON file (one JSON object
      per line, or a single JSON list if the file is new and you want one episode
      per file; here we write episode.to_dict() as one JSON object per line for
      append-friendly storage).

    Returns:
        Episode instance (data_structure.experience.Episode).
    """
    if Episode is None or Experience is None:
        raise ImportError(
            "Experience and Episode are required from data_structure.experience."
        )

    rollout = run_episode_vlm_agent(
        env,
        agent=agent,
        model=model,
        skill_bank=skill_bank,
        memory=memory,
        reward_config=reward_config,
        max_steps=max_steps,
        verbose=verbose,
    )

    episode = rollout_to_episode(rollout, task=task)

    if episode_buffer is not None:
        episode_buffer.add_episode(episode)
        if verbose:
            print(f"Added episode to episode buffer (size: {len(episode_buffer)})")

    if experience_buffer is not None:
        experience_buffer.add_experience(episode)
        if verbose:
            print(f"Added {len(episode.experiences)} experiences to experience buffer")

    if save_path:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            line = json.dumps(episode.to_dict(), ensure_ascii=False) + "\n"
            f.write(line)
        if verbose:
            print(f"Appended episode to {path}")

    return episode
