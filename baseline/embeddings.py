"""Small embedding interfaces for the FAISS baseline."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


class EmbeddingModel:
    name: str
    dim: int

    def encode(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError


@dataclass
class HashingTextEmbedder(EmbeddingModel):
    """Deterministic no-download text embedder for plumbing smoke tests.

    This is not a strong semantic model. It exists so schema, metadata, and FAISS
    mechanics are runnable before choosing the production embedding model.
    """

    dim: int = 384
    name: str = "hashing-text-v0"

    def encode(self, texts: list[str]) -> np.ndarray:
        vectors = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in _tokens(text):
                digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
                bucket = int.from_bytes(digest[:4], "little") % self.dim
                sign = 1.0 if int.from_bytes(digest[4:], "little") % 2 == 0 else -1.0
                vectors[row, bucket] += sign
            norm = float(np.linalg.norm(vectors[row]))
            if norm > 0:
                vectors[row] /= norm
        return vectors


class CLIPVideoTextEmbedder(EmbeddingModel):
    """Cross-modal CLIP embedder for text queries and video clip records.

    Video clips are represented by one or more sampled frames, averaged in CLIP
    image-embedding space. Questions are encoded in CLIP text space, so FAISS can
    do cross-modal retrieval.
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        *,
        device: str | None = None,
        frames_per_clip: int = 1,
    ) -> None:
        import torch
        from transformers import CLIPModel, CLIPProcessor

        self.name = model_name
        self.frames_per_clip = frames_per_clip
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.dim = int(self.model.config.projection_dim)

    def encode(self, texts: list[str]) -> np.ndarray:
        import torch

        inputs = self.processor(text=texts, return_tensors="pt", padding=True, truncation=True)
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.inference_mode():
            vectors = self.model.get_text_features(**inputs)
        vectors = _feature_tensor(vectors)
        return cosine_normalize(vectors.detach().cpu().numpy())

    def encode_clip_records(self, clips: list[Any]) -> np.ndarray:
        import torch

        vectors = []
        for clip in clips:
            images = sample_clip_frames(
                clip.video_path,
                clip.start_s,
                clip.end_s,
                frames_per_clip=self.frames_per_clip,
            )
            inputs = self.processor(images=images, return_tensors="pt")
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            with torch.inference_mode():
                image_vectors = self.model.get_image_features(**inputs)
            image_vectors = _feature_tensor(image_vectors)
            image_vector = image_vectors.mean(dim=0, keepdim=True)
            vectors.append(image_vector.detach().cpu().numpy()[0])
        return cosine_normalize(np.asarray(vectors, dtype=np.float32))


def sample_clip_frames(video_path: str, start_s: float, end_s: float, *, frames_per_clip: int) -> list[Image.Image]:
    import cv2  # type: ignore

    if frames_per_clip <= 0:
        raise ValueError("frames_per_clip must be positive")
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(video_path)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open video: {video_path}")
    if frames_per_clip == 1:
        timestamps = [(start_s + end_s) / 2.0]
    else:
        span = max(end_s - start_s, 0.001)
        timestamps = [start_s + span * i / (frames_per_clip - 1) for i in range(frames_per_clip)]
    images = []
    try:
        for timestamp_s in timestamps:
            cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, timestamp_s) * 1000.0)
            ok, frame = cap.read()
            if not ok:
                continue
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            images.append(Image.fromarray(frame))
    finally:
        cap.release()
    if not images:
        raise RuntimeError(f"could not sample frames from {video_path} [{start_s}, {end_s}]")
    return images


def _tokens(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9_]+", text.lower())


def _feature_tensor(features: Any) -> Any:
    for attr in ("image_embeds", "text_embeds", "pooler_output"):
        value = getattr(features, attr, None)
        if value is not None:
            return value
    return features


def cosine_normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (vectors / norms).astype(np.float32)
