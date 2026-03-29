#!/usr/bin/env python3
"""
Run all paper analysis scripts and produce a combined report.

Usage:
  python run_all.py              # Run all analyses
  python run_all.py p1 p2        # Run specific analyses
  python run_all.py --list       # List available analyses
"""

import importlib
import sys
import time
import traceback

ANALYSES = {
    "p1": ("p1_qualitative_walkthrough", "Qualitative skill retrieval walkthrough"),
    "p2": ("p2_reward_decomposition", "Reward decomposition by ablation condition"),
    "p3": ("p3_skill_taxonomy_lifecycle", "Skill taxonomy & lifecycle / churn analysis"),
    "p4": ("p4_skill_retrieval_analysis", "Skill-state association & retrieval confidence"),
    "p5": ("p5_failure_analysis", "Failure analysis & contract failure signatures"),
    "p7": ("p7_strategy_analysis", "Intention tag / strategy distribution"),
}


def main():
    args = sys.argv[1:]

    if "--list" in args:
        print("Available analyses:")
        for key, (module, desc) in sorted(ANALYSES.items()):
            print(f"  {key}: {desc} ({module}.py)")
        return

    to_run = []
    if args:
        for a in args:
            if a in ANALYSES:
                to_run.append(a)
            else:
                print(f"Unknown analysis: {a}")
                return
    else:
        to_run = sorted(ANALYSES.keys())

    for key in to_run:
        module_name, desc = ANALYSES[key]
        print(f"\n{'#' * 100}")
        print(f"# {key.upper()}: {desc}")
        print(f"{'#' * 100}")

        t0 = time.time()
        try:
            mod = importlib.import_module(module_name)
            mod.main()
        except Exception:
            print(f"\n  ERROR in {module_name}:")
            traceback.print_exc()
        elapsed = time.time() - t0
        print(f"\n  [{key} completed in {elapsed:.1f}s]")


if __name__ == "__main__":
    main()
