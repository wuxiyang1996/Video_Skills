"""FAISS index persistence for baseline video clip retrieval."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .schemas import RetrievedClip, VideoClipRecord


def _import_faiss() -> Any:
    try:
        import faiss  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "FAISS is not installed in this Python environment. Install faiss-cpu/faiss-gpu "
            "or run in a cluster environment that provides FAISS."
        ) from exc
    return faiss


class FaissClipStore:
    """Cosine/IP FAISS store with row-aligned JSONL clip metadata."""

    def __init__(self, index: Any, clips: list[VideoClipRecord], manifest: dict[str, Any]):
        self.index = index
        self.clips = clips
        self.manifest = manifest

    @classmethod
    def build(
        cls,
        embeddings: np.ndarray,
        clips: list[VideoClipRecord],
        *,
        embedding_model: str,
        embedding_backend: str | None = None,
    ) -> FaissClipStore:
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)
        if embeddings.ndim != 2:
            raise ValueError(f"expected 2D embeddings, got shape={embeddings.shape}")
        if embeddings.shape[0] != len(clips):
            raise ValueError(f"embedding rows {embeddings.shape[0]} != clips {len(clips)}")
        faiss = _import_faiss()
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        manifest = {
            "index_type": "IndexFlatIP",
            "metric": "cosine_inner_product",
            "embedding_model": embedding_model,
            "embedding_backend": embedding_backend,
            "dim": embeddings.shape[1],
            "count": len(clips),
        }
        return cls(index, clips, manifest)

    def save(self, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        faiss = _import_faiss()
        faiss.write_index(self.index, str(output_dir / "index.faiss"))
        with (output_dir / "clips.jsonl").open("w", encoding="utf-8") as handle:
            for clip in self.clips:
                handle.write(json.dumps(clip.to_dict(), ensure_ascii=False) + "\n")
        (output_dir / "manifest.json").write_text(
            json.dumps(self.manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, index_dir: Path) -> FaissClipStore:
        faiss = _import_faiss()
        index = faiss.read_index(str(index_dir / "index.faiss"))
        clips = []
        with (index_dir / "clips.jsonl").open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    clips.append(VideoClipRecord(**json.loads(line)))
        manifest = json.loads((index_dir / "manifest.json").read_text(encoding="utf-8"))
        return cls(index, clips, manifest)

    def search(self, query_embedding: np.ndarray, *, topk: int) -> list[RetrievedClip]:
        if query_embedding.dtype != np.float32:
            query_embedding = query_embedding.astype(np.float32)
        if query_embedding.ndim == 1:
            query_embedding = query_embedding[None, :]
        scores, row_ids = self.index.search(query_embedding, topk)
        results: list[RetrievedClip] = []
        for rank, (score, row_id) in enumerate(zip(scores[0].tolist(), row_ids[0].tolist()), start=1):
            if row_id < 0:
                continue
            results.append(RetrievedClip(rank=rank, score=float(score), row_id=int(row_id), clip=self.clips[row_id]))
        return results
