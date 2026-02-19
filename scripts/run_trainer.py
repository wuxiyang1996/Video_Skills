"""
Run trainer (Decision GRPO and/or Co-evolution) with declared fatal hyperparameters and game envs.

Fatal hyperparameters and game env list: see trainer_defaults.py.

Usage:
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
import sys
from pathlib import Path

# Add repo root so trainer/ and config paths resolve
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.trainer_defaults import (
    TRAINER_GAME_ENVS,
    TRAINER_DECISION_FATAL,
    TRAINER_SKILLBANK_FATAL,
    TRAINER_DECISION_CONFIG_PATH,
    TRAINER_SKILLBANK_CONFIG_PATH,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Decision Agent GRPO or Co-evolution. Fatal hyperparameters in scripts/trainer_defaults.py.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=TRAINER_DECISION_CONFIG_PATH,
        help="Path to Decision GRPO config YAML (default: %(default)s)",
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
    args = parser.parse_args()

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
