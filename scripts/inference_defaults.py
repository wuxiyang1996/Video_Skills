"""
Inference: fatal hyperparameters and game environments.

Used by run_inference.py. All critical (fatal) hyperparameters and supported
game env names are declared here. See inference/run_decision_agent.py.
"""

from __future__ import annotations

from typing import Any, Dict, List

# -----------------------------------------------------------------------------
# Game environments used by inference (same as trainer; must match env_wrappers)
# -----------------------------------------------------------------------------
INFERENCE_GAME_ENVS: List[str] = [
    "overcooked",
    "avalon",
    "diplomacy",
    "gamingagent",
    "videogamebench",
    "videogamebench_dos",
]

# -----------------------------------------------------------------------------
# Fatal hyperparameters: run_inference / run_episode_vlm_agent
# Changing these significantly affects rollout length, storage, and quality.
# -----------------------------------------------------------------------------
INFERENCE_FATAL: Dict[str, Any] = {
    # Episode
    "max_steps": 1000,
    "task": "",
    # Model (passed to VLMDecisionAgent)
    "model": "gpt-4o-mini",
    # Agent (optional; if None, created with model/skill_bank/memory)
    "skill_bank": None,
    "memory": None,
    "reward_config": None,
    # Buffers (optional)
    "episode_buffer_size": 100,
    "experience_buffer_size": 10_000,
    # Storage
    "save_path": None,  # e.g. "rollouts/episodes.jsonl"
    "verbose": False,
}

# Default rollout output path when saving
INFERENCE_DEFAULT_SAVE_DIR = "rollouts"
INFERENCE_DEFAULT_SAVE_FILE = "episodes.jsonl"
