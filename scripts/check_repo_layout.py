#!/usr/bin/env python3
"""Check top-level repository layout against the integration cleanup map."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

KNOWN_TOP_LEVEL = {
    ".env.example",
    ".gitattributes",
    ".gitignore",
    "API_func.py",
    "INSTALL.md",
    "LICENSE",
    "README.md",
    "REPO_STRUCTURE.md",
    "atomic-skill-decomposition-and-assembly",
    "atomic_skills",
    "backups",
    "cold_start",
    "configs",
    "data_structure",
    "dataset_clip_wrapper",
    "dataset_examples",
    "decision_agents",
    "docs",
    "experiments",
    "inference",
    "infra_plans",
    "install",
    "labeling",
    "motif",
    "out",
    "plans",
    "pyproject.toml",
    "rag",
    "readme.md",
    "reflection",
    "requirements.txt",
    "schemas",
    "scripts",
    "skill_agents",
    "tests",
    "trainer",
    "video_skills",
    "visual_grounding",
}

GENERATED_SHOULD_NOT_BE_TRACKED = {
    ".env",
    ".pytest_cache",
    "__pycache__",
}


def _is_local_or_generated(name: str) -> bool:
    if name in GENERATED_SHOULD_NOT_BE_TRACKED:
        return True
    if name == ".venv" or name.startswith(".venv-") or name == "venv":
        return True
    return False


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

    unexpected = sorted(
        name
        for name in entries
        if name not in KNOWN_TOP_LEVEL and not _is_local_or_generated(name)
    )
    if unexpected:
        problems.append(
            "Unexpected top-level paths: "
            + ", ".join(unexpected)
            + ". Update REPO_STRUCTURE.md and scripts/check_repo_layout.py if intentional."
        )

    tracked_generated = _tracked_files(*sorted(GENERATED_SHOULD_NOT_BE_TRACKED))
    if tracked_generated:
        problems.append("Generated/local-only paths are tracked: " + ", ".join(tracked_generated))

    tracked_venvs = [
        path
        for path in _tracked_files()
        if path.startswith(".venv/")
        or path.startswith(".venv-")
        or path.startswith("venv/")
        or path == ".venv"
        or path == "venv"
    ]
    if tracked_venvs:
        problems.append(
            "Local virtualenv paths are tracked: " + ", ".join(tracked_venvs[:10])
        )

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
