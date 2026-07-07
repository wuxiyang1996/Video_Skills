#!/usr/bin/env python3
"""Build a FAISS clip index from wrapper canonical examples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .embeddings import CLIPVideoTextEmbedder, HashingTextEmbedder
from .faiss_store import FaissClipStore
from .schemas import clip_records_from_canonical


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--canonical-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--videomme-observation-end-s", type=float, default=60.0)
    parser.add_argument("--embedding-backend", default="hashing_text", choices=["hashing_text", "clip"])
    parser.add_argument("--clip-model", default="openai/clip-vit-base-patch32")
    parser.add_argument("--frames-per-clip", type=int, default=1)
    parser.add_argument("--max-clips", type=int, default=None)
    parser.add_argument("--hash-dim", type=int, default=384)
    parser.add_argument(
        "--store-embeddings",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Store clip embeddings directly in clips.jsonl schema records.",
    )
    args = parser.parse_args()

    clips = []
    for example in iter_jsonl(args.canonical_jsonl):
        clips.extend(
            clip_records_from_canonical(
                example,
                start_row_id=len(clips),
                default_videomme_cutoff_s=args.videomme_observation_end_s,
            )
        )
        if args.max_clips is not None and len(clips) >= args.max_clips:
            clips = clips[: args.max_clips]
            break
    if args.embedding_backend == "clip":
        embedder = CLIPVideoTextEmbedder(model_name=args.clip_model, frames_per_clip=args.frames_per_clip)
        embeddings = embedder.encode_clip_records(clips)
    else:
        embedder = HashingTextEmbedder(dim=args.hash_dim)
        embeddings = embedder.encode([clip.text for clip in clips])
    if args.store_embeddings:
        for clip, embedding in zip(clips, embeddings.tolist()):
            object.__setattr__(clip, "embedding", embedding)
    store = FaissClipStore.build(
        embeddings,
        clips,
        embedding_model=embedder.name,
        embedding_backend=args.embedding_backend,
    )
    store.save(args.output_dir)
    print(
        json.dumps(
            {
                "clips": len(clips),
                "dim": embedder.dim,
                "embedding_backend": args.embedding_backend,
                "store_embeddings": args.store_embeddings,
                "output_dir": str(args.output_dir),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
