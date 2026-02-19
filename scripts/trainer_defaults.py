"""
Trainer: fatal hyperparameters and game environments.

Used by run_trainer.py and co-evolution. All critical (fatal) hyperparameters
and supported game env names are declared here for a single source of truth.
See trainer/common/configs/decision_grpo.yaml and skillbank_em.yaml for full YAML.
"""

from __future__ import annotations

from typing import Any, Dict, List

# -----------------------------------------------------------------------------
# Game environments used by the trainer (must match decision_agents.dummy_agent)
# -----------------------------------------------------------------------------
TRAINER_GAME_ENVS: List[str] = [
    "overcooked",        # Overcooked AI — cooperative cooking
    "avalon",            # AgentEvolver Avalon — hidden-role deduction
    "diplomacy",         # AgentEvolver Diplomacy — strategic negotiation
    "gamingagent",       # LMGame-Bench — 2048, Sokoban, Tetris, etc.
    "videogamebench",    # VideoGameBench (Game Boy)
    "videogamebench_dos", # VideoGameBench DOS — primary for evaluate_videogamebench
]

# -----------------------------------------------------------------------------
# Fatal hyperparameters: Decision Agent GRPO (trainer/decision, launch_coevolution)
# Changing these significantly affects training stability and results.
# -----------------------------------------------------------------------------
TRAINER_DECISION_FATAL: Dict[str, Any] = {
    # Model
    "model_name": "gpt-4o-mini",
    "temperature": 0.7,
    "eval_temperature": 0.0,
    "max_tokens": 400,
    # GRPO
    "group_size": 8,
    "clip_ratio": 0.2,
    "kl_coeff": 0.01,
    "lr": 1.0e-5,
    "epochs_per_batch": 4,
    "max_grad_norm": 1.0,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "normalize_advantages": True,
    "entropy_coeff": 0.01,
    # Rollout
    "batch_size": 32,
    "max_steps": 500,
    "num_workers": 4,
    "retrieval_budget_n": 10,
    "skill_abort_k": 5,
    # Replay
    "replay_capacity": 10000,
    "priority_alpha": 0.6,
    "priority_beta": 0.4,
    "min_episodes": 64,
    # Reward costs (decision_agents.reward_func)
    "c_mem": -0.05,
    "c_skill": -0.05,
    "c_call": -0.02,
    "c_switch": -0.10,
    # Follow shaping
    "w_follow": 0.1,
    "predicate_bonus": 0.05,
    "completion_bonus": 0.20,
    "no_progress_penalty": -0.01,
    # Eval
    "eval_interval_episodes": 50,
    "num_eval_episodes": 10,
    "timeout_steps": 1000,
    "eval_seeds": [42, 137, 256, 512, 1024, 2048, 4096, 8192],
    # Schedule
    "total_episodes": 50000,
    "warmup_episodes": 100,
    "bank_update_cadence": 500,
}

# -----------------------------------------------------------------------------
# Fatal hyperparameters: SkillBank Hard-EM (trainer/skillbank)
# -----------------------------------------------------------------------------
TRAINER_SKILLBANK_FATAL: Dict[str, Any] = {
    # Propose cuts (Stage 1)
    "propose_w": 5,
    "merge_radius": 5,
    "surprisal_weight": 0.5,
    "predicate_change_weight": 0.5,
    # Decode (Stage 2)
    "top_m_candidates": 10,
    "segment_min_len": 3,
    "segment_max_len": 100,
    "new_skill_penalty": 5.0,
    "eff_freq": 0.8,
    "tau_create": 0.7,
    # Contracts (Stage 3)
    "min_instances_per_skill": 5,
    "pass_rate_threshold": 0.6,
    # Update (Stage 4)
    "min_new_cluster_size": 5,
    "split_pass_rate_threshold": 0.7,
    "merge_jaccard_threshold": 0.85,
    # Gating
    "min_pass_rate": 0.6,
    "min_support": 3,
    "max_new_rate": 0.3,
    # EM loop
    "max_iterations": 3,
    "convergence_new_rate": 0.05,
}

# Default config paths (relative to repo root)
TRAINER_DECISION_CONFIG_PATH = "trainer/common/configs/decision_grpo.yaml"
TRAINER_SKILLBANK_CONFIG_PATH = "trainer/common/configs/skillbank_em.yaml"
