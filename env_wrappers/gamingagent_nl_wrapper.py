"""
GamingAgent (LMGame-Bench) env wrapper: state <-> natural language.

Wraps GamingAgentEnv (from GamingAgent/gym_like) so that:
- Observation is the textual state as natural language (obs["text"]).
- step() accepts string actions (e.g. "up", "left", "hard_drop").

The underlying GamingAgentEnv already provides text observations and accepts
string actions; this wrapper presents a consistent NL interface for the
dummy_agent and other language-model policies.

Usage:

    from env_wrappers.gamingagent_nl_wrapper import GamingAgentNLWrapper
    from GamingAgent.gym_like import make_gaming_env

    base_env = make_gaming_env("sokoban", max_steps=200)
    env = GamingAgentNLWrapper(base_env)
    obs, info = env.reset()   # obs: str (text state)
    obs, reward, term, trunc, info = env.step("push up")
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


def state_to_natural_language(obs: Union[Dict[str, Any], str]) -> str:
    """
    Convert GamingAgent observation to natural language string.

    Args:
        obs: Observation from GamingAgentEnv (dict with "text", "screen", "raw")
             or already a string.

    Returns:
        Natural language state description for LLM consumption.
    """
    if isinstance(obs, str):
        return obs
    if isinstance(obs, dict):
        text = obs.get("text") or obs.get("textual_representation", "")
        if text:
            return str(text)
        if "screen" in obs and obs["screen"] is not None:
            return "You see the game screen. (Text representation not available.)"
        return "Game state. (No text representation.)"
    return str(obs)


class GamingAgentNLWrapper:
    """
    Wraps GamingAgentEnv so observations are natural language strings
    and step() accepts string actions.

    Compatible with agents.dummy_agent.language_agent_action when game=GAME_GAMINGAGENT.
    """

    def __init__(
        self,
        env: Any,
        include_action_hint: bool = True,
    ):
        """
        Args:
            env: GamingAgentEnv (or any env with obs["text"] and string-action step).
            include_action_hint: If True, append valid action names to the NL state.
        """
        self._env = env
        self._include_action_hint = include_action_hint
        self._action_names: List[str] = []
        self._step_count = 0

    @property
    def env(self):
        return self._env

    def _obs_to_nl(self, obs: Union[Dict[str, Any], str], info: Dict[str, Any]) -> str:
        nl = state_to_natural_language(obs)
        if self._include_action_hint and self._action_names:
            nl += f"\n\nValid actions: {', '.join(self._action_names)}. Choose one."
        return nl

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        obs, info = self._env.reset(seed=seed, options=options)
        self._step_count = 0
        self._action_names = info.get("action_names", getattr(self._env, "action_names", []))
        nl = self._obs_to_nl(obs, info)
        info["state_natural_language"] = nl
        info["action_names"] = self._action_names
        return nl, info

    def step(
        self,
        action: Union[str, int, np.integer],
    ) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        obs, reward, terminated, truncated, info = self._env.step(action)
        self._step_count += 1
        nl = self._obs_to_nl(obs, info)
        info["state_natural_language"] = nl
        info["action_names"] = self._action_names
        info["step"] = self._step_count
        return nl, float(reward), bool(terminated), bool(truncated), info

    def close(self) -> None:
        if hasattr(self._env, "close"):
            self._env.close()

    @property
    def action_space(self):
        return self._env.action_space

    @property
    def observation_space(self):
        return getattr(self._env, "observation_space", None)

    @property
    def action_names(self) -> List[str]:
        return self._action_names or getattr(self._env, "action_names", [])
