#!/usr/bin/env python3
"""Preflight checks for official Dispider, M3-Agent, and StreamBridge runs.

This script intentionally distinguishes an official model/evaluation pipeline
from dataset adapters and protocol adaptations in this repository.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any


PROJECT = Path("/mnt/is_data/xwu/video_skills")
LOCAL_REPO = Path("/home/xwu/atomic_skills_for_video")
EXPECTED_REMOTES = {
    "dispider": "https://github.com/Mark12Ding/Dispider.git",
    "m3_agent": "https://github.com/ByteDance-Seed/m3-agent.git",
    "streambridge": "https://github.com/apple/ml-streambridge.git",
}


def _git_value(repo: Path, *args: str) -> str | None:
    if not (repo / ".git").is_dir():
        return None
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _python_packages(python: Path, packages: list[str]) -> dict[str, str | None]:
    if not python.is_file():
        return {package: None for package in packages}
    code = (
        "import importlib.metadata,json;"
        f"names={packages!r};"
        "print(json.dumps({n:(importlib.metadata.version(n) "
        "if any(d.metadata['Name'].lower()==n.lower() for d in importlib.metadata.distributions()) "
        "else None) for n in names}))"
    )
    result = subprocess.run(
        [str(python), "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return {package: None for package in packages}
    return json.loads(result.stdout)


def _repo_check(name: str, path: Path) -> dict[str, Any]:
    remote = _git_value(path, "remote", "get-url", "origin")
    expected = EXPECTED_REMOTES[name]
    return {
        "path": str(path),
        "exists": path.is_dir(),
        "origin": remote,
        "expected_origin": expected,
        "origin_matches": bool(remote and remote.rstrip("/").removesuffix(".git").lower() == expected.rstrip("/").removesuffix(".git").lower()),
        "commit": _git_value(path, "rev-parse", "HEAD"),
        "dirty": bool(_git_value(path, "status", "--porcelain")),
    }


def _checkpoint_check(path: Path, *, config_name: str = "config.json") -> dict[str, Any]:
    config_path = path / config_name
    shards = sorted(path.glob("*.safetensors"))
    index_path = path / "model.safetensors.index.json"
    result: dict[str, Any] = {
        "path": str(path),
        "exists": path.is_dir(),
        "config": config_path.is_file(),
        "safetensors_files": len(shards),
        "safetensors_index": index_path.is_file(),
    }
    if config_path.is_file():
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
            compressor = config.get("mm_compressor")
            if compressor:
                compressor_path = Path(str(compressor))
                result["mm_compressor"] = str(compressor_path)
                result["mm_compressor_exists"] = compressor_path.is_dir()
        except (OSError, json.JSONDecodeError) as exc:
            result["config_error"] = f"{type(exc).__name__}: {exc}"
    return result


def _nonempty_api_config(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {"path": str(path), "exists": path.is_file(), "configured_models": []}
    if not path.is_file():
        return result
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        return result
    configured = []
    for model, values in payload.items():
        if isinstance(values, dict) and any(str(value).strip() for value in values.values()):
            configured.append(model)
    result["configured_models"] = configured
    return result


def _first_existing(paths: list[Path]) -> Path:
    return next((path for path in paths if path.exists()), paths[0])


def build_report(project: Path, local_repo: Path) -> dict[str, Any]:
    code = project / "code"
    models = project / "data" / "models"
    datasets = project / "data" / "datasets"
    dispider_repo = code / "Dispider"
    m3_repo = code / "m3-agent"
    streambridge_repo = code / "ml-streambridge"

    dispider_model = models / "dispider" / "Mar2Ding_Dispider"
    dispider_python = code / "dispider_venv" / "bin" / "python"
    dispider_packages = _python_packages(
        dispider_python,
        ["torch", "transformers", "decord", "flash-attn"],
    )
    m3_mem_model = _first_existing(
        [
            models / "M3-Agent-Memorization",
            project / "models" / "M3-Agent-Memorization",
            m3_repo / "models" / "M3-Agent-Memorization",
        ]
    )
    m3_control_model = _first_existing(
        [
            models / "M3-Agent-Control",
            project / "models" / "M3-Agent-Control",
            m3_repo / "models" / "M3-Agent-Control",
        ]
    )

    report = {
        "scope": {
            "project": str(project),
            "local_repo": str(local_repo),
            "meaning": "Readiness for official upstream execution, not merely schema compatibility.",
        },
        "dispider": {
            "upstream": _repo_check("dispider", dispider_repo),
            "environment": {
                "path": str(code / "dispider_venv"),
                "python": dispider_python.is_file(),
                "packages": dispider_packages,
                "official_flash_attention_available": dispider_packages["flash-attn"] is not None,
            },
            "checkpoint": _checkpoint_check(dispider_model),
            "official_videomme_template": (dispider_repo / "playground" / "data" / "videomme_template.json").is_file(),
            "local_integration": {
                "runner": str(local_repo / "baseline" / "dispider_streaming_eval.py"),
                "runner_exists": (local_repo / "baseline" / "dispider_streaming_eval.py").is_file(),
                "classification": "protocol adaptation: quick-start inference over independently materialized visible prefixes",
                "official_benchmark_equivalent": False,
            },
        },
        "m3_agent": {
            "upstream": _repo_check("m3_agent", m3_repo),
            "memorization_checkpoint": _checkpoint_check(m3_mem_model),
            "control_checkpoint": _checkpoint_check(m3_control_model),
            "api_config": _nonempty_api_config(m3_repo / "configs" / "api_config.json"),
            "m3_bench_present": (datasets / "M3-Bench").is_dir(),
            "local_integration": {
                "classification": "design borrowing only; no M3-Agent baseline runner",
                "official_benchmark_equivalent": False,
            },
        },
        "streambridge": {
            "upstream": _repo_check("streambridge", streambridge_repo),
            "checkpoint_note": "Upstream README does not publish trained checkpoints; CKPT must be supplied separately.",
            "ovo_annotations": (streambridge_repo / "assets" / "ovo_bench.json").is_file(),
            "videomme_annotations": (streambridge_repo / "assets" / "videomme.json").is_file(),
            "ovo_original_video_layout": (Path("/net/mlfs01/export/users/dpatel/OVO-Bench") / "data").is_dir(),
            "ovo_question_prefix_layout": (Path("/net/mlfs01/export/users/dpatel/OVO-Bench") / "chunked_videos").is_dir(),
            "videomme_video_layout": (Path("/net/nj-storage02/mnt/tank/datasets/WHB139426-Grounded-VideoLLM/videomme") / "videos").is_dir(),
            "local_integration": {
                "adapter": str(local_repo / "dataset_clip_wrapper" / "adapters" / "streaming_video.py"),
                "adapter_exists": (local_repo / "dataset_clip_wrapper" / "adapters" / "streaming_video.py").is_file(),
                "classification": "dataset adapter only; it does not execute StreamBridge memory or activation models",
                "official_benchmark_equivalent": False,
            },
        },
    }

    dispider_code_and_weights = bool(
        report["dispider"]["upstream"]["origin_matches"]
        and report["dispider"]["environment"]["python"]
        and report["dispider"]["checkpoint"]["config"]
        and report["dispider"]["checkpoint"]["safetensors_files"]
        and report["dispider"]["checkpoint"].get("mm_compressor_exists", False)
    )
    report["ready"] = {
        "dispider_code_and_weights": dispider_code_and_weights,
        "dispider_official_pipeline": bool(
            dispider_code_and_weights
            and report["dispider"]["environment"]["official_flash_attention_available"]
        ),
        "m3_agent_official_pipeline": bool(
            report["m3_agent"]["upstream"]["origin_matches"]
            and report["m3_agent"]["memorization_checkpoint"]["config"]
            and report["m3_agent"]["control_checkpoint"]["config"]
            and report["m3_agent"]["m3_bench_present"]
        ),
        "streambridge_official_pipeline": False,
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, default=PROJECT)
    parser.add_argument("--local-repo", type=Path, default=LOCAL_REPO)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report = build_report(args.project.resolve(), args.local_repo.resolve())
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
