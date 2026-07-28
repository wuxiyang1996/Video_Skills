"""Local replacements for M3-Agent's hosted embedding and Whisper calls."""

from __future__ import annotations

import os
import json
import sys
import types
import urllib.request
from functools import lru_cache
from pathlib import Path
from typing import Any


DEFAULT_EMBEDDING_MODEL = "Qwen/Qwen3-VL-Embedding-2B"
DEFAULT_WHISPER_MODEL = "openai/whisper-large-v3-turbo"
DEFAULT_INSTRUCTION = "Represent the input for retrieving relevant video memories for a question."


@lru_cache(maxsize=1)
def _embedding_model() -> Any:
    from sentence_transformers import SentenceTransformer

    model_name = os.environ.get("M3_LOCAL_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
    device = os.environ.get("M3_LOCAL_EMBEDDING_DEVICE", "cpu")
    return SentenceTransformer(
        model_name,
        trust_remote_code=True,
        device=device,
        local_files_only=os.environ.get("M3_LOCAL_FILES_ONLY", "1") == "1",
    )


def encode_texts(texts: list[str]) -> list[list[float]]:
    backend_url = os.environ.get("M3_LOCAL_BACKEND_URL")
    if backend_url:
        payload = _post_json(backend_url + "/embed", {"texts": texts})
        return payload["embeddings"]

    import numpy as np

    if not texts:
        return []
    vectors = _embedding_model().encode(
        texts,
        prompt=os.environ.get("M3_LOCAL_EMBEDDING_INSTRUCTION", DEFAULT_INSTRUCTION),
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return np.asarray(vectors, dtype=np.float32).tolist()


def get_embedding_with_retry(_model: str, text: str, timeout: int = 15) -> tuple[list[float], int]:
    del timeout
    return encode_texts([text])[0], 0


def parallel_get_embedding(
    _model: str,
    texts: list[str],
    timeout: int = 15,
) -> tuple[list[list[float]], int]:
    del timeout
    return encode_texts(list(texts)), 0


@lru_cache(maxsize=1)
def _whisper_pipeline() -> Any:
    import torch
    from transformers import pipeline

    model_name = os.environ.get("M3_LOCAL_WHISPER_MODEL", DEFAULT_WHISPER_MODEL)
    device_name = os.environ.get("M3_LOCAL_WHISPER_DEVICE", "cpu")
    device = -1 if device_name == "cpu" else int(device_name.removeprefix("cuda:"))
    dtype = torch.float16 if device >= 0 else torch.float32
    return pipeline(
        "automatic-speech-recognition",
        model=model_name,
        torch_dtype=dtype,
        device=device,
        model_kwargs={
            "local_files_only": os.environ.get("M3_LOCAL_FILES_ONLY", "1") == "1",
        },
    )


def get_whisper_with_retry(_model: str, file_path: str) -> str:
    backend_url = os.environ.get("M3_LOCAL_BACKEND_URL")
    if backend_url:
        payload = _post_json(backend_url + "/transcribe", {"file_path": file_path})
        return str(payload["text"])
    result = _whisper_pipeline()(file_path, return_timestamps=True)
    return str(result.get("text") or "").strip()


def parallel_get_whisper(_model: str, file_paths: list[str]) -> list[str]:
    return [get_whisper_with_retry(_model, path) for path in file_paths]


def install_into_m3() -> dict[str, str]:
    """Patch M3's imported API module before other M3 modules are loaded."""

    if os.environ.get("M3_LIGHTWEIGHT_PACKAGE") == "1" and "mmagent" not in sys.modules:
        package_root = Path.cwd() / "mmagent"
        package = types.ModuleType("mmagent")
        package.__path__ = [str(package_root)]  # type: ignore[attr-defined]
        package.__package__ = "mmagent"
        sys.modules["mmagent"] = package

    from mmagent.utils import chat_api

    chat_api.get_embedding_with_retry = get_embedding_with_retry
    chat_api.parallel_get_embedding = parallel_get_embedding
    chat_api.get_whisper_with_retry = get_whisper_with_retry
    chat_api.parallel_get_whisper = parallel_get_whisper
    return {
        "embedding_model": os.environ.get("M3_LOCAL_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
        "embedding_device": os.environ.get("M3_LOCAL_EMBEDDING_DEVICE", "cpu"),
        "whisper_model": os.environ.get("M3_LOCAL_WHISPER_MODEL", DEFAULT_WHISPER_MODEL),
        "whisper_device": os.environ.get("M3_LOCAL_WHISPER_DEVICE", "cpu"),
    }


def validate_local_models() -> dict[str, Any]:
    embedding_name = os.environ.get("M3_LOCAL_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
    whisper_name = os.environ.get("M3_LOCAL_WHISPER_MODEL", DEFAULT_WHISPER_MODEL)
    return {
        "embedding_model": embedding_name,
        "embedding_cache_present": _cache_present(embedding_name),
        "whisper_model": whisper_name,
        "whisper_cache_present": _cache_present(whisper_name),
        "local_files_only": os.environ.get("M3_LOCAL_FILES_ONLY", "1") == "1",
    }


def _cache_present(model_name: str) -> bool:
    if Path(model_name).exists():
        return True
    cache_root = Path(
        os.environ.get(
            "HUGGINGFACE_HUB_CACHE",
            str(Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "hub"),
        )
    )
    return (cache_root / f"models--{model_name.replace('/', '--')}").is_dir()


def _post_json(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=3600) as response:
        return json.loads(response.read().decode("utf-8"))
