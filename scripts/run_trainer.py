"""
Run trainer (Decision GRPO and/or Co-evolution) with declared fatal hyperparameters and game envs.

Preferred: VERL (verl-agent) for distributed GiGPO/PPO training with vLLM/sglang.
Standalone mode uses in-repo GRPO for debugging or when VERL is not installed.

Fatal hyperparameters and game env list: see trainer_defaults.py.

Usage:
  # VERL training (recommended): uses https://github.com/verl-project/verl via verl-agent
  python -m scripts.run_trainer --verl
  python -m scripts.run_trainer --verl algorithm.adv_estimator=gigpo trainer.nnodes=2

  # Standalone (no VERL)
  python -m scripts.run_trainer --config trainer/common/configs/decision_grpo.yaml
  python -m scripts.run_trainer --coevolution \\
    --decision-config trainer/common/configs/decision_grpo.yaml \\
    --skillbank-config trainer/common/configs/skillbank_em.yaml
  python -m scripts.run_trainer --print-envs
  python -m scripts.run_trainer --print-defaults
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

# Add repo root so trainer/ and config paths resolve
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# verl-agent is expected at sibling path (e.g. ICML2026/verl-agent next to ICML2026/Game-AI-Agent)
_VERL_AGENT_ROOT = _REPO_ROOT.parent / "verl-agent"

from scripts.trainer_defaults import (
    TRAINER_GAME_ENVS,
    TRAINER_DECISION_FATAL,
    TRAINER_SKILLBANK_FATAL,
    TRAINER_DECISION_CONFIG_PATH,
    TRAINER_SKILLBANK_CONFIG_PATH,
)


def _run_verl_trainer(extra_overrides: list[str]) -> None:
    """Run VERL Game-AI training via verl-agent's main_gameai."""
    if not _VERL_AGENT_ROOT.is_dir():
        raise FileNotFoundError(
            f"verl-agent not found at {_VERL_AGENT_ROOT}. "
            "Clone verl-agent next to Game-AI-Agent (e.g. ICML2026/verl-agent) and install: pip install -e ."
        )
    env = os.environ.copy()
    path_parts = [str(_REPO_ROOT), str(_VERL_AGENT_ROOT)]
    if env.get("PYTHONPATH"):
        path_parts.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(path_parts)

    default_overrides = [
        "algorithm.adv_estimator=gigpo",
        "env.env_name=gameai",
        "env.seed=42",
        "env.rollout.n=8",
        "reward_model.reward_manager=gameai",
        "costs.c_mem=-0.05",
        "costs.c_skill=-0.05",
        "costs.c_call=-0.02",
        "costs.c_switch=-0.10",
        "costs.w_follow=0.1",
        "coevolution.enable=True",
        "coevolution.bank_dir=runs/skillbank",
    ]
    cmd = [
        sys.executable,
        "-m",
        "verl.trainer.main_gameai",
        *default_overrides,
        *extra_overrides,
    ]
    logging.info("Running VERL trainer: %s", " ".join(cmd))
    subprocess.run(cmd, env=env, cwd=str(_REPO_ROOT.parent))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Decision Agent GRPO or Co-evolution. Use --verl for VERL/verl-agent training.",
    )
    parser.add_argument(
        "--verl",
        action="store_true",
        help="Use VERL (verl-agent) for training. Requires verl-agent at ../verl-agent.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=TRAINER_DECISION_CONFIG_PATH,
        help="Path to Decision GRPO config YAML (default: %(default)s). Ignored if --verl.",
    )
    parser.add_argument(
        "--coevolution",
        action="store_true",
        help="Run co-evolution (Decision + SkillBank) instead of Decision-only",
    )
    parser.add_argument(
        "--decision-config",
        type=str,
        default=TRAINER_DECISION_CONFIG_PATH,
        help="Decision config YAML when using --coevolution",
    )
    parser.add_argument(
        "--skillbank-config",
        type=str,
        default=TRAINER_SKILLBANK_CONFIG_PATH,
        help="SkillBank EM config YAML when using --coevolution",
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
    args, verl_overrides = parser.parse_known_args()

    if args.print_envs:
        print("Trainer game envs (trainer_defaults.TRAINER_GAME_ENVS):")
        for e in TRAINER_GAME_ENVS:
            print(f"  - {e}")
        return

    if args.print_defaults:
        print("Decision (fatal) defaults (trainer_defaults.TRAINER_DECISION_FATAL):")
        for k, v in TRAINER_DECISION_FATAL.items():
            print(f"  {k}: {v}")
        print("\nSkillBank (fatal) defaults (trainer_defaults.TRAINER_SKILLBANK_FATAL):")
        for k, v in TRAINER_SKILLBANK_FATAL.items():
            print(f"  {k}: {v}")
        return

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    if args.verl:
        _run_verl_trainer(verl_overrides)
        return

    if args.coevolution:
        from trainer.launch_coevolution import load_config, run_coevolution
        decision_cfg = load_config(args.decision_config)
        skillbank_cfg = load_config(args.skillbank_config)
        run_coevolution(
            decision_cfg=decision_cfg,
            skillbank_cfg=skillbank_cfg,
            env_factory=None,
        )
    else:
        from trainer.decision.launch_train import load_config, train_decision_agent
        cfg = load_config(args.config)
        train_decision_agent(cfg)


if __name__ == "__main__":
    main()
