#!/usr/bin/env python3
"""Download pre-generated cold-start data from HuggingFace.

Usage:
    python labeling/download_cold_start.py
    python labeling/download_cold_start.py --games tetris candy_crush
    python labeling/download_cold_start.py --output-dir labeling/output/gpt54_skill_labeled
"""
import argparse
import json
from pathlib import Path

from huggingface_hub import snapshot_download

REPO_ID = "IntelligenceLab/Cos-Play-Cold-Start"
ALL_GAMES = [
    "avalon", "candy_crush", "diplomacy", "pokemon_red",
    "sokoban", "super_mario", "tetris", "twenty_forty_eight",
]
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "output" / "gpt54_skill_labeled"


def expand_episodes(download_dir: Path, games: list[str]):
    """Expand consolidated JSONL back into per-episode JSON files."""
    episodes_dir = download_dir / "data" / "episodes"
    for game in games:
        jsonl = episodes_dir / f"{game}.jsonl"
        if not jsonl.exists():
            continue
        out_dir = download_dir / game
        out_dir.mkdir(parents=True, exist_ok=True)
        count = 0
        with open(jsonl) as f:
            for line in f:
                ep = json.loads(line)
                ep_id = ep.get("episode_id", f"episode_{count:03d}")
                out_file = out_dir / f"episode_{count:03d}.json"
                with open(out_file, "w") as fout:
                    json.dump(ep, fout, indent=2, ensure_ascii=False)
                count += 1
        print(f"  {game}: expanded {count} episodes -> {out_dir}")

    grpo_src = download_dir / "data" / "grpo_coldstart"
    grpo_dst = download_dir / "grpo_coldstart"
    if grpo_src.is_dir():
        import shutil
        if grpo_dst.exists():
            shutil.rmtree(grpo_dst)
        shutil.copytree(grpo_src, grpo_dst)
        print(f"  grpo_coldstart -> {grpo_dst}")


def main():
    parser = argparse.ArgumentParser(description="Download COS-PLAY cold-start data")
    parser.add_argument(
        "--games", nargs="+", default=ALL_GAMES, choices=ALL_GAMES,
        help="Games to download (default: all)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT,
        help=f"Output directory (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--no-expand", action="store_true",
        help="Skip expanding JSONL into individual episode JSON files",
    )
    args = parser.parse_args()

    print(f"Downloading cold-start data from {REPO_ID} ...")
    snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        local_dir=str(args.output_dir),
    )
    print(f"Downloaded to {args.output_dir}\n")

    if not args.no_expand:
        print("Expanding episodes to individual JSON files...")
        expand_episodes(args.output_dir, args.games)

    print("\nDone!")


if __name__ == "__main__":
    main()
