#!/usr/bin/env python3
"""Check top-level repository layout against the clean L1/L2 map."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

KNOWN_TOP_LEVEL = {
    ".gitignore",
    "README.md",
    "REPO_STRUCTURE.md",
    "atomic-skill-decomposition-and-assembly",
    "atomic_skills",
    "dataset_clip_wrapper",
    "docs",
    "experiments",
    "rag",
    "reflection",
    "schemas",
    "scripts",
    "tests",
    "video_skills",
    "visual_grounding",
}

GENERATED_SHOULD_NOT_BE_TRACKED = {
    ".env",
    ".pytest_cache",
    "__pycache__",
}

LEGACY_TOP_LEVEL = {
    "cold_start",
    "data_structure",
    "decision_agents",
    "dataset_examples",
    "skill_agents",
    "trainer",
}


def _tracked_files(*paths: str) -> list[str]:
    cmd = ["git", "ls-files", *paths]
    result = subprocess.run(cmd, cwd=ROOT, check=True, text=True, capture_output=True)
    return [line for line in result.stdout.splitlines() if line.strip()]


def main() -> int:
    problems: list[str] = []
    entries = {
        path.name
        for path in ROOT.iterdir()
        if path.name != ".git" and not path.name.endswith(".egg-info")
    }

    unexpected = sorted(entries - KNOWN_TOP_LEVEL - GENERATED_SHOULD_NOT_BE_TRACKED)
    if unexpected:
        problems.append(
            "Unexpected top-level paths: "
            + ", ".join(unexpected)
            + ". Update REPO_STRUCTURE.md and scripts/check_repo_layout.py if intentional."
        )

    legacy_present = sorted(entries & LEGACY_TOP_LEVEL)
    if legacy_present:
        problems.append(
            "Legacy top-level paths are present in the clean branch: "
            + ", ".join(legacy_present)
        )

    tracked_generated = _tracked_files(*GENERATED_SHOULD_NOT_BE_TRACKED)
    if tracked_generated:
        problems.append("Generated/local-only paths are tracked: " + ", ".join(tracked_generated))

    tracked_dataset_outputs = [
        path
        for path in _tracked_files("dataset_clip_wrapper/output")
        if path != "dataset_clip_wrapper/output/.gitkeep"
    ]
    if tracked_dataset_outputs:
        problems.append(
            "dataset_clip_wrapper/output should track only .gitkeep: "
            + ", ".join(tracked_dataset_outputs[:10])
        )

    if problems:
        for problem in problems:
            print(f"repo layout check failed: {problem}", file=sys.stderr)
        return 1

    print("repo layout check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
