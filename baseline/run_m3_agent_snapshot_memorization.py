#!/usr/bin/env python3
"""Run official M3 memorization once per source and save causal snapshots."""

from __future__ import annotations

import argparse
import copy
import glob
import json
import os
import pickle
import sys
from pathlib import Path
from typing import Any

from .m3_local_backends import install_into_m3


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _save_graph(graph: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    snapshot = copy.deepcopy(graph)
    snapshot.refresh_equivalences()
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(snapshot, handle)
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-file", type=Path, required=True)
    parser.add_argument("--m3-repo", type=Path, default=Path("/mnt/is_data/xwu/video_skills/code/m3-agent"))
    parser.add_argument(
        "--speakerlab-repo",
        type=Path,
        default=Path("/mnt/is_data/xwu/video_skills/code/3D-Speaker"),
    )
    parser.add_argument("--fps", type=float, default=5.0)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument(
        "--attention-implementation",
        choices=["flash_attention_2", "sdpa"],
        default="flash_attention_2",
    )
    parser.add_argument("--disable-face", action="store_true")
    parser.add_argument("--disable-speaker", action="store_true")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.fps <= 0:
        parser.error("--fps must be positive")
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be positive")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")

    repo = args.m3_repo.resolve()
    os.chdir(repo)
    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(args.speakerlab_repo.resolve()))
    if args.disable_face and args.disable_speaker:
        os.environ["M3_LIGHTWEIGHT_PACKAGE"] = "1"
    backend = install_into_m3()

    from transformers import Qwen2_5OmniThinkerForConditionalGeneration

    original_generate = Qwen2_5OmniThinkerForConditionalGeneration.generate
    original_from_pretrained = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained

    @classmethod
    def configured_from_pretrained(
        cls: type[Any],
        pretrained_model_name_or_path: str | Path,
        *model_args: Any,
        **model_kwargs: Any,
    ) -> Any:
        del cls
        model_kwargs["attn_implementation"] = args.attention_implementation
        return original_from_pretrained(pretrained_model_name_or_path, *model_args, **model_kwargs)

    def capped_generate(self: Any, *generate_args: Any, **generate_kwargs: Any) -> Any:
        generate_kwargs["max_new_tokens"] = min(
            int(generate_kwargs.get("max_new_tokens") or args.max_new_tokens),
            args.max_new_tokens,
        )
        return original_generate(self, *generate_args, **generate_kwargs)

    Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained = configured_from_pretrained
    Qwen2_5OmniThinkerForConditionalGeneration.generate = capped_generate

    from mmagent.memory_processing_qwen import generate_memories, process_memories
    from mmagent.utils.video_processing import process_video_clip
    from mmagent.videograph import VideoGraph

    process_faces = None
    process_voices = None
    if not args.disable_face:
        from mmagent.face_processing import process_faces as _process_faces

        process_faces = _process_faces
    if not args.disable_speaker:
        from mmagent.voice_processing import process_voices as _process_voices

        process_voices = _process_voices

    memory_config = json.loads((repo / "configs" / "memory_config.json").read_text(encoding="utf-8"))
    print(
        json.dumps(
            {
                "local_backends": backend,
                "data_file": str(args.data_file.resolve()),
                "profile": {
                    "fps": args.fps,
                    "max_new_tokens": args.max_new_tokens,
                    "attention_implementation": args.attention_implementation,
                    "face_processing": not args.disable_face,
                    "speaker_processing": not args.disable_speaker,
                },
            }
        ),
        flush=True,
    )

    rows = _rows(args.data_file.resolve())
    if args.limit is not None:
        rows = rows[: args.limit]
    for row in rows:
        final_path = Path(row["mem_path"])
        snapshots = {int(key): Path(value) for key, value in (row.get("snapshot_paths") or {}).items()}
        expected = [final_path, *snapshots.values()]
        if expected and all(path.is_file() for path in expected):
            print(json.dumps({"id": row["id"], "status": "reused"}), flush=True)
            continue

        Path(row["intermediate_outputs"]).mkdir(parents=True, exist_ok=True)
        graph = VideoGraph(**memory_config)
        if -1 in snapshots:
            _save_graph(graph, snapshots[-1])

        clip_paths = sorted(
            glob.glob(str(Path(row["clip_path"]) / "*.mp4")),
            key=lambda value: int(Path(value).stem),
        )
        for clip_path in clip_paths:
            clip_id = int(Path(clip_path).stem)
            base64_video, base64_frames, base64_audio = process_video_clip(clip_path, fps=args.fps)
            if base64_frames:
                voices = (
                    process_voices(
                        graph,
                        base64_audio,
                        base64_video,
                        save_path=str(Path(row["intermediate_outputs"]) / f"clip_{clip_id}_voices.json"),
                        preprocessing=[],
                    )
                    if process_voices is not None
                    else {}
                )
                faces = (
                    process_faces(
                        graph,
                        base64_frames,
                        save_path=str(Path(row["intermediate_outputs"]) / f"clip_{clip_id}_faces.json"),
                        preprocessing=[],
                    )
                    if process_faces is not None
                    else {}
                )
                episodic, semantic = generate_memories(
                    base64_frames,
                    faces,
                    voices,
                    clip_path,
                )
                process_memories(graph, episodic, clip_id, type="episodic")
                process_memories(graph, semantic, clip_id, type="semantic")
            if clip_id in snapshots:
                _save_graph(graph, snapshots[clip_id])

        missing = [clip_id for clip_id, path in snapshots.items() if not path.is_file()]
        if missing:
            raise RuntimeError(f"snapshot clip ids were not materialized for {row['id']}: {missing}")
        _save_graph(graph, final_path)
        print(
            json.dumps(
                {
                    "id": row["id"],
                    "status": "generated",
                    "clips": len(clip_paths),
                    "snapshots": len(snapshots),
                }
            ),
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
