"""
Orak game environment NL wrapper.

Wraps Orak game environments (BaseEnv subclasses from krafton-ai/Orak) so that:
- Observations are natural language strings (via obs2text).
- step() accepts string actions (parsed via text2action).

Orak benchmark covers 11 games across 6 genres:
  Action:      street_fighter, super_mario
  Adventure:   pwaat (Ace Attorney), her_story
  RPG:         darkest_dungeon
  Simulation:  minecraft, stardew_valley
  Strategy:    star_craft, star_craft_multi, slay_the_spire
  Puzzle:      baba_is_you, twenty_fourty_eight

Usage:
    from evaluate_orak.orak_nl_wrapper import OrakNLWrapper, make_orak_env

    env = make_orak_env("star_craft")
    obs, info = env.reset()
    obs, reward, term, trunc, info = env.step(...)
"""

from __future__ import annotations

import atexit
import contextlib
import logging
import os
import random
import re
import shutil
import sys
import tempfile
import time as _time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


_SCRIPT_DIR = Path(__file__).resolve().parent
_CODEBASE_ROOT = _SCRIPT_DIR.parent

# Primary: cloned Orak repo
_ORAK_REPO = _CODEBASE_ROOT.parent / "Orak"
_ORAK_SRC = _ORAK_REPO / "src"
_ORAK_MCP_GAMES = _ORAK_SRC / "mcp_game_servers"
_ORAK_MCP_AGENTS = _ORAK_SRC / "mcp_agent_client"

for _p in [str(_ORAK_SRC), str(_ORAK_MCP_GAMES)]:
    if Path(_p).exists() and _p not in sys.path:
        sys.path.insert(0, _p)


@contextlib.contextmanager
def _orak_cwd():
    """Temporarily chdir to the Orak repo root for games that use relative paths."""
    prev = os.getcwd()
    try:
        os.chdir(str(_ORAK_REPO))
        yield
    finally:
        os.chdir(prev)


def _cfg_path(game: str) -> str:
    """Resolve config.yaml path from the Orak repo."""
    return str(_ORAK_MCP_AGENTS / "configs" / game / "config.yaml")



ORAK_GAMES: Dict[str, Dict[str, Any]] = {
    # ── Puzzle ──────────────────────────────────────────────────────────
    "twenty_fourty_eight": {
        "config_yaml": _cfg_path("twenty_fourty_eight"),
        "action_names": ["up", "down", "left", "right"],
        "task": "Merge tiles to reach 2048. Score = min(score/20000*100, 100).",
        "genre": "puzzle",
    },
    "baba_is_you": {
        "config_yaml": _cfg_path("baba_is_you"),
        "action_names": ["idle", "left", "right", "up", "down"],
        "task": "Solve the Baba Is You puzzle by manipulating rules. 100=win, 40=WIN exists, 20=WALL broken, 0=fail.",
        "genre": "puzzle",
    },
    # ── Action ──────────────────────────────────────────────────────────
    "super_mario": {
        "config_yaml": _cfg_path("super_mario"),
        "action_names": [f"Jump Level: {i}" for i in range(7)],
        "task": "Advance Mario as far right as possible. Score = x_pos / 3161 * 100.",
        "genre": "action",
    },
    "street_fighter": {
        "config_yaml": _cfg_path("street_fighter"),
        "action_names": [
            "Move Closer", "Move Away", "Fireball", "Megapunch", "Hurricane",
            "Low Kick", "Medium Kick", "High Kick", "Jump Closer", "Jump Away",
            "Crouch", "Block", "Low Punch", "Medium Punch", "High Punch",
        ],
        "task": "Defeat the opponent in Street Fighter III. Score = stages cleared.",
        "genre": "action",
    },
    # ── Strategy ────────────────────────────────────────────────────────
    "star_craft": {
        "config_yaml": _cfg_path("star_craft"),
        "action_names": [
            "TRAIN PROBE", "TRAIN ZEALOT", "TRAIN STALKER", "BUILD PYLON",
            "BUILD GATEWAY", "BUILD ASSIMILATOR", "BUILD NEXUS",
            "BUILD CYBERNETICSCORE", "RESEARCH WARPGATERESEARCH",
            "RESEARCH CHARGE", "SCOUTING PROBE", "MULTI-ATTACK",
            "MULTI-RETREAT", "CHRONOBOOST NEXUS", "EMPTY ACTION",
        ],
        "task": "Win 1v1 as Protoss vs Zerg bot. Provide 5 sequential macro actions per step.",
        "genre": "strategy",
    },
    "star_craft_multi": {
        "config_yaml": _cfg_path("star_craft_multi"),
        "action_names": [],
        "task": "Win 1v1 StarCraft II (multi-player mode). Provide 5 actions per step.",
        "genre": "strategy",
    },
    "slay_the_spire": {
        "config_yaml": _cfg_path("slay_the_spire"),
        "action_names": ["PLAY", "END", "CHOOSE", "SKIP"],
        "task": "Climb the Spire, defeat enemies with card combos. Score = floor reached (max 50).",
        "genre": "strategy",
    },
    # ── RPG ─────────────────────────────────────────────────────────────
    "darkest_dungeon": {
        "config_yaml": _cfg_path("darkest_dungeon"),
        "action_names": ["attack", "heal", "swap", "idle", "skip"],
        "task": "Survive dungeon raids. Score = 0.4*combat + 0.3*survival + 0.3*(1-stress).",
        "genre": "rpg",
    },
    # ── Adventure ───────────────────────────────────────────────────────
    "pwaat": {
        "config_yaml": _cfg_path("pwaat"),
        "action_names": [
            "Ok", "Back", "Down", "Up", "Left", "Right",
            "Present evidence", "Press",
        ],
        "task": "Solve cases in Ace Attorney. Score = milestone rewards.",
        "genre": "adventure",
    },
    "her_story": {
        "config_yaml": _cfg_path("her_story"),
        "action_names": ["Search", "Play Video"],
        "task": "Uncover the story by searching keywords and watching videos. Score = videos viewed / 272.",
        "genre": "adventure",
    },
    # ── Simulation ──────────────────────────────────────────────────────
    "minecraft": {
        "config_yaml": _cfg_path("minecraft"),
        "action_names": [],
        "task": "Craft target items in Minecraft. Actions are JavaScript async functions.",
        "genre": "simulation",
    },
    "stardew_valley": {
        "config_yaml": _cfg_path("stardew_valley"),
        "action_names": [
            "till_soil", "plant_seeds", "water_seeds", "harvest_crops",
            "sell_item", "buy_item", "get_out_of_house", "go_house_and_sleep",
        ],
        "task": "Complete farming tasks in Stardew Valley (cleanup, cultivation, shopping, earn money).",
        "genre": "simulation",
    },
}


def _obs_to_text(obs_obj: Any, env: Any) -> str:
    """Convert an Orak Obs dataclass to text via env.obs2text."""
    text = env.obs2text(obs_obj)
    if text is None:
        if hasattr(obs_obj, "to_text"):
            text = obs_obj.to_text()
        else:
            text = str(obs_obj)
    return text or ""


class OrakNLWrapper:
    """
    Wraps an Orak BaseEnv so observations are NL strings and step()
    accepts string actions. Presents the same interface as other
    Game-AI-Agent NL wrappers.
    """

    def __init__(
        self,
        env: Any,
        game_name: str,
        include_action_hint: bool = True,
        max_steps: int = 1000,
    ):
        self._env = env
        self._game_name = game_name
        self._include_action_hint = include_action_hint
        self._max_steps = max_steps
        self._step_count = 0
        self._last_reward: Optional[float] = None
        self._prev_score_val: float = 0.0

        game_info = ORAK_GAMES.get(game_name, {})
        self._action_names: List[str] = game_info.get("action_names", [])
        self._task: str = game_info.get("task", "")


    @property
    def env(self):
        return self._env

    @property
    def action_names(self) -> List[str]:
        return self._action_names

    def _format_obs(self, obs_text: str) -> str:
        nl = obs_text
        if self._include_action_hint and self._action_names:
            if self._game_name in ("star_craft", "star_craft_multi"):
                nl += "\n\nProvide exactly 5 actions in the format:\n1: <ACTION>\n2: <ACTION>\n...\n5: <ACTION>"
                nl += f"\n\nValid actions: {', '.join(a for a in self._action_names if a != 'EMPTY ACTION')}, EMPTY ACTION"
            elif self._game_name == "slay_the_spire":
                nl += "\n\nChoose: PLAY <card_idx> [target_idx], END, CHOOSE <idx>, or SKIP."
            elif self._game_name == "minecraft":
                nl += "\n\nWrite a JavaScript async function with bot parameter."
            elif self._game_name == "stardew_valley":
                nl += f"\n\nReturn a Python list of skill calls. Available: {', '.join(self._action_names)}"
            else:
                nl += f"\n\nValid actions: {', '.join(self._action_names[:20])}. Choose one."
        return nl

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        self._step_count = 0
        self._last_reward = None
        self._prev_score_val = 0.0

        with _orak_cwd():
            # Orak BaseEnv uses initial_obs() rather than Gymnasium's reset()
            if hasattr(self._env, "initial_obs"):
                obs_obj = self._env.initial_obs()
            else:
                kw = {}
                if seed is not None:
                    kw["seed"] = seed
                if options is not None:
                    kw["options"] = options
                obs_obj = self._env.reset(**kw) if kw else self._env.reset()
                if isinstance(obs_obj, tuple):
                    obs_obj = obs_obj[0]

            obs_text = _obs_to_text(obs_obj, self._env)

        # SC2: the bot's first on_step() hasn't run yet, so the initial
        # observation is empty.  Do one warmup step with EMPTY ACTIONs to
        # get the real first game state from the running SC2 process.
        if not obs_text.strip() and self._game_name in ("star_craft", "star_craft_multi"):
            n_actions = getattr(self._env, "num_actions", 5)
            empty_actions = "\n".join(f"{i}: EMPTY ACTION" for i in range(1, n_actions + 1))
            warmup_nl, _, _, _, warmup_info = self.step(empty_actions)
            self._step_count = 0
            return warmup_nl, warmup_info

        nl = self._format_obs(obs_text)

        game_info = {}
        if hasattr(self._env, "get_game_info"):
            game_info = self._env.get_game_info() or {}

        info: Dict[str, Any] = {
            "state_natural_language": nl,
            "action_names": self._action_names,
            "env_name": "orak",
            "game_name": self._game_name,
            "task": self._task,
            **game_info,
        }
        return nl, info

    def step(
        self,
        action: Union[str, int],
    ) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        action_str = str(action).strip()

        with _orak_cwd():
            action_obj = self._env.text2action(action_str)

            result = self._env.step(action_obj)
            obs_obj, reward_raw, terminated, truncated, step_info = (
                result[0], result[1], result[2], result[3], result[4]
            )

            score, done = self._env.evaluate(obs_obj)
            obs_text = _obs_to_text(obs_obj, self._env)

        # Convert evaluate() score → numeric reward (delta-based).
        #
        # evaluate() return formats vary by game:
        #   SC2:         ("Victory"|"Defeat"|"Tie"|None, done)
        #   Mario:       (float, done)
        #   Others:      (float|str|None, done)
        _SC2_RESULT_MAP = {"Victory": 100.0, "Defeat": 0.0, "Tie": 50.0}
        score_val = 0.0
        if isinstance(score, str):
            if score in _SC2_RESULT_MAP:
                score_val = _SC2_RESULT_MAP[score]
            else:
                try:
                    score_val = float(score.split("(")[0].strip())
                except (ValueError, AttributeError):
                    score_val = 0.0
        elif score is not None:
            try:
                score_val = float(score)
            except (ValueError, TypeError):
                score_val = 0.0

        reward = score_val - self._prev_score_val
        self._prev_score_val = score_val

        self._step_count += 1
        self._last_reward = reward

        if self._step_count >= self._max_steps and not (terminated or truncated):
            truncated = True

        nl = self._format_obs(obs_text)

        game_info = {}
        if hasattr(self._env, "get_game_info"):
            game_info = self._env.get_game_info() or {}

        info: Dict[str, Any] = {
            "state_natural_language": nl,
            "action_names": self._action_names,
            "env_name": "orak",
            "game_name": self._game_name,
            "step": self._step_count,
            "score": score,
            "task": self._task,
            **game_info,
        }
        return nl, reward, bool(terminated or done), bool(truncated), info

    def close(self) -> None:
        if hasattr(self._env, "close"):
            self._env.close()

    @property
    def action_space(self):
        return getattr(self._env, "action_space", None)

    @property
    def observation_space(self):
        return getattr(self._env, "observation_space", None)


def make_orak_env(
    game_name: str,
    max_steps: int = 1000,
    config_override: Optional[str] = None,
) -> OrakNLWrapper:
    """Create a wrapped Orak game environment.

    Args:
        game_name: One of the keys in ORAK_GAMES (e.g. "star_craft", "baba_is_you").
        max_steps: Max steps before truncation.
        config_override: Optional path to a custom config YAML.

    Returns:
        OrakNLWrapper with standard reset()/step() Gymnasium interface.
    """
    if game_name not in ORAK_GAMES:
        raise ValueError(f"Unknown Orak game '{game_name}'. Available: {sorted(ORAK_GAMES.keys())}")

    config_path = config_override or ORAK_GAMES[game_name]["config_yaml"]

    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Config not found: {config_path}. "
            f"Ensure the Orak repo is cloned at {_ORAK_REPO}"
        )

    import omegaconf
    from mcp_game_servers.utils.module_creator import EnvCreator

    cfg = omegaconf.OmegaConf.load(config_path)

    log_dir = str(_CODEBASE_ROOT / "orak_logs" / game_name)
    os.makedirs(log_dir, exist_ok=True)

    _NO_SCREENSHOT_GAMES = {"super_mario"}

    with omegaconf.open_dict(cfg):
        if "env" in cfg:
            cfg.env.log_path = log_dir
            if hasattr(cfg.env, "show_graphic"):
                cfg.env.show_graphic = False
            if game_name in _NO_SCREENSHOT_GAMES:
                cfg.env.save_screenshots = False
            # Resolve rom_path relative to Orak repo root so it works
            # regardless of the caller's cwd.
            if hasattr(cfg.env, "rom_path") and not os.path.isabs(cfg.env.rom_path):
                abs_rom = os.path.normpath(os.path.join(str(_ORAK_REPO), cfg.env.rom_path))
                cfg.env.rom_path = abs_rom
        cfg.log_path = log_dir

    with _orak_cwd():
        env = EnvCreator(cfg).create()

    wrapper = OrakNLWrapper(env, game_name=game_name, max_steps=max_steps)

    # Use the real action space from the env when available (e.g. SC2 has 72
    # Protoss actions vs the 15-action subset in ORAK_GAMES).
    if hasattr(env, "action_dict") and env.action_dict:
        wrapper._action_names = sorted(
            env.action_dict.keys(), key=lambda a: env.action_dict[a]
        )

    return wrapper
