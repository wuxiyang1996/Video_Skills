"""Concrete :class:`VLMCallable` backends for the grounding pipeline.

The grounding pipeline (``local_grounder.ground_window``) is model-agnostic:
it takes any ``vlm_fn(prompt, *, frames=...) -> str``. This module supplies
real backends that plug into that contract. They are optional — the rest of
the package stays importable even when no API keys / vision models are
available.

Available backends (today, scaffolding on hosted APIs):
    * :func:`make_gpt4o_vlm` — OpenAI GPT-4o / GPT-4o-mini with vision,
      via OpenAI or OpenRouter. ``chat.completions`` + ``image_url``.
    * :func:`make_claude_vlm` — Anthropic Claude 3.x with vision
      (``image`` content blocks, base64 source).

Available backend (production target):
    * :func:`make_vllm_vlm` — any OpenAI-compatible ``chat.completions``
      endpoint. This is how the plan swaps in a local Qwen3-VL-32B /
      Qwen3-VL-72B served by vLLM: same message schema as GPT-4o
      (``image_url`` data URLs), so the grounder code never changes. Only
      ``base_url`` + ``model`` move.

Dispatcher:
    * :func:`make_vlm` — pick a backend by ``provider`` name. Callers can
      parameterize the pipeline by provider (``"gpt4o"`` today,
      ``"qwen3vl"`` once the vLLM server is up) with zero other edits.

All backends share the same frame-encoding path (``_encode_frame_as_data_url``)
and the same defensive error envelope (returning ``"__VLM_ERROR__: ..."``
on failure). The grounder already treats any unparseable VLM response as
an empty :class:`GroundedWindow`, so a dead provider degrades gracefully.
"""

from __future__ import annotations

import base64
import os
from typing import Any, Callable, Dict, List, Optional, Sequence


__all__ = [
    "make_gpt4o_vlm",
    "make_claude_vlm",
    "make_vllm_vlm",
    "make_vlm",
]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _load_dotenv_once() -> None:
    """Best-effort load of ``Video_Skills/.env`` (no-op if missing).

    Respects pre-set env vars; only fills in what's missing. Used so that
    ``python scripts/test_gpt4o_grounding.py`` works without an explicit
    ``set -a && source .env && set +a``.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    candidate = os.path.join(os.path.dirname(here), ".env")
    if not os.path.isfile(candidate):
        return
    try:
        for line in open(candidate, "r", encoding="utf-8"):
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k, v = k.strip(), v.strip().strip('"').strip("'")
            if k and v and os.environ.get(k, "") == "":
                os.environ[k] = v
    except Exception:
        return


def _read_frame_bytes(path: str) -> Optional[bytes]:
    try:
        with open(path, "rb") as fp:
            return fp.read()
    except Exception:
        return None


def _image_mime(path: str) -> str:
    ext = os.path.splitext(path)[1].lower().lstrip(".") or "jpeg"
    if ext == "jpg":
        ext = "jpeg"
    return f"image/{ext}"


def _encode_frame_as_data_url(path: str) -> Optional[str]:
    blob = _read_frame_bytes(path)
    if blob is None:
        return None
    b64 = base64.b64encode(blob).decode("ascii")
    return f"data:{_image_mime(path)};base64,{b64}"


def _select_frame_paths(
    frames: Optional[Sequence[Dict[str, Any]]],
    max_frames: int,
) -> List[str]:
    """Return up to ``max_frames`` on-disk image paths from ``frames``."""
    out: List[str] = []
    for f in frames or []:
        if len(out) >= max_frames:
            break
        path = f.get("path") if isinstance(f, dict) else None
        if path and os.path.isfile(path):
            out.append(path)
    return out


# ---------------------------------------------------------------------------
# OpenAI-compatible (GPT-4o + vLLM + OpenRouter) backend
# ---------------------------------------------------------------------------


def _make_openai_compatible_vlm(
    *,
    base_url: Optional[str],
    api_key: str,
    model: str,
    temperature: float,
    max_tokens: int,
    max_frames: int,
    image_detail: str,
    request_json: bool,
    timeout: float,
    tag: str = "openai-compat",
) -> Callable[..., str]:
    """Shared implementation for any OpenAI ``chat.completions`` endpoint.

    Used by :func:`make_gpt4o_vlm` (OpenAI / OpenRouter) and
    :func:`make_vllm_vlm` (local Qwen3-VL-32B/72B via vLLM). Both accept
    the same ``image_url`` content-part schema.
    """

    def _call(prompt: str, *, frames: Optional[Sequence[Dict[str, Any]]] = None,
              **_kwargs: Any) -> str:
        try:
            import openai  # type: ignore
        except Exception as exc:  # pragma: no cover
            return f"__VLM_ERROR__[{tag}]: openai SDK unavailable: {exc}"

        if not api_key:
            return f"__VLM_ERROR__[{tag}]: no API key supplied"

        client_kwargs: Dict[str, Any] = {"api_key": api_key, "timeout": timeout}
        if base_url:
            client_kwargs["base_url"] = base_url
        client = openai.OpenAI(**client_kwargs)

        content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
        for path in _select_frame_paths(frames, max_frames):
            url = _encode_frame_as_data_url(path)
            if not url:
                continue
            content.append({
                "type": "image_url",
                "image_url": {"url": url, "detail": image_detail},
            })

        req: Dict[str, Any] = dict(
            model=model,
            messages=[{"role": "user", "content": content}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if request_json:
            req["response_format"] = {"type": "json_object"}

        try:
            resp = client.chat.completions.create(**req)
        except Exception as exc:
            if request_json:
                # Some providers (OpenRouter, vLLM) reject response_format.
                try:
                    req.pop("response_format", None)
                    resp = client.chat.completions.create(**req)
                except Exception as exc2:
                    return f"__VLM_ERROR__[{tag}]: {exc2}"
            else:
                return f"__VLM_ERROR__[{tag}]: {exc}"

        try:
            return resp.choices[0].message.content or ""
        except Exception as exc:
            return f"__VLM_ERROR__[{tag}]: malformed response: {exc}"

    return _call


def make_gpt4o_vlm(
    *,
    model: str = "gpt-4o",
    temperature: float = 0.2,
    max_tokens: int = 1500,
    max_frames: int = 4,
    prefer_openrouter: bool = True,
    image_detail: str = "low",
    request_json: bool = True,
    timeout: float = 60.0,
) -> Callable[..., str]:
    """Return a :class:`VLMCallable` backed by GPT-4o with vision.

    Args:
        model: Model name. Use ``gpt-4o`` (default), ``gpt-4o-mini``, or an
            OpenRouter-prefixed form such as ``openai/gpt-4o``.
        temperature: Sampling temperature. Grounding wants low temperature
            so the §3.2 JSON stays stable (default 0.2).
        max_tokens: Upper bound on completion size.
        max_frames: Cap on images attached per call. Combine with
            ``max_frames_per_window`` in :func:`pipeline.build_grounded_context`
            to control cost.
        prefer_openrouter: If True and ``OPENROUTER_API_KEY`` is set, route
            through OpenRouter; otherwise use ``OPENAI_API_KEY`` directly.
        image_detail: Forwarded on ``image_url`` parts — ``"low"`` is
            usually enough for grounding and keeps spend bounded.
        request_json: Ask the API for JSON-object responses when supported.
        timeout: Per-request timeout in seconds.

    Notes:
        This is scaffolding for the production plan. Once a Qwen3-VL
        vLLM endpoint is available, callers should switch to
        :func:`make_vllm_vlm` (or pass ``provider="qwen3vl"`` to
        :func:`make_vlm`). Both emit the same ``chat.completions`` +
        ``image_url`` payload, so :mod:`local_grounder` is unchanged.
    """
    _load_dotenv_once()

    openrouter_key = (os.environ.get("OPENROUTER_API_KEY") or "").strip()
    openai_key = (os.environ.get("OPENAI_API_KEY") or "").strip()

    if prefer_openrouter and openrouter_key:
        base_url = "https://openrouter.ai/api/v1"
        api_key = openrouter_key
        model_name = model if "/" in model else f"openai/{model}"
        tag = "openrouter"
    elif openai_key:
        base_url = None
        api_key = openai_key
        model_name = model.split("/", 1)[-1] if model.startswith("openai/") else model
        tag = "openai"
    elif openrouter_key:  # last-chance fallback
        base_url = "https://openrouter.ai/api/v1"
        api_key = openrouter_key
        model_name = model if "/" in model else f"openai/{model}"
        tag = "openrouter"
    else:
        # Defer the error to call time so import stays side-effect free.
        base_url, api_key, model_name, tag = None, "", model, "openai-missing"

    return _make_openai_compatible_vlm(
        base_url=base_url,
        api_key=api_key,
        model=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        max_frames=max_frames,
        image_detail=image_detail,
        request_json=request_json,
        timeout=timeout,
        tag=tag,
    )


def make_vllm_vlm(
    *,
    model: str = "Qwen/Qwen3-VL-32B-Instruct",
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    temperature: float = 0.2,
    max_tokens: int = 1500,
    max_frames: int = 4,
    image_detail: str = "low",
    request_json: bool = False,
    timeout: float = 120.0,
) -> Callable[..., str]:
    """Return a :class:`VLMCallable` that talks to a vLLM server.

    This is the production target for swapping in Qwen3-VL-32B or
    Qwen3-VL-72B. vLLM exposes an OpenAI-compatible ``chat.completions``
    endpoint that accepts ``image_url`` content parts (base64 data URLs),
    so callers can migrate from :func:`make_gpt4o_vlm` to this backend
    without touching the grounding pipeline.

    Args:
        model: Served model name (must match the name vLLM was launched
            with, e.g. ``Qwen/Qwen3-VL-32B-Instruct`` or
            ``Qwen/Qwen3-VL-72B-Instruct``).
        base_url: Endpoint URL. Defaults to ``VLLM_BASE_URL`` env var
            (``http://localhost:8000/v1`` when unset).
        api_key: Bearer token. Defaults to ``VLLM_API_KEY`` or ``"EMPTY"``.
        request_json: Default ``False`` because older vLLM builds reject
            ``response_format``. The grounder parses JSON from free text
            defensively, so leaving this off is safe.

    Notes:
        Per ``infra_plans/03_controller/actors_reasoning_model.md`` §2.4 the VLM stays
        a frozen tool; the only knob that changes between scaffolding
        (GPT-4o) and production (Qwen3-VL) is the provider binding.
    """
    _load_dotenv_once()
    base_url = base_url or os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
    api_key = (api_key if api_key is not None
               else os.environ.get("VLLM_API_KEY", "EMPTY"))

    return _make_openai_compatible_vlm(
        base_url=base_url,
        api_key=api_key or "EMPTY",
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        max_frames=max_frames,
        image_detail=image_detail,
        request_json=request_json,
        timeout=timeout,
        tag="vllm",
    )


# ---------------------------------------------------------------------------
# Anthropic Claude backend
# ---------------------------------------------------------------------------


def make_claude_vlm(
    *,
    model: str = "claude-sonnet-4-5-20250929",
    temperature: float = 0.2,
    max_tokens: int = 1500,
    max_frames: int = 4,
    timeout: float = 60.0,
) -> Callable[..., str]:
    """Return a :class:`VLMCallable` backed by Anthropic Claude with vision.

    Uses the native Anthropic SDK (``messages.create``) and attaches each
    frame as an ``image`` content block with a base64 ``source`` (the
    Anthropic vision schema, which differs from OpenAI's ``image_url``).

    This is scaffolding alongside :func:`make_gpt4o_vlm`; production will
    switch to :func:`make_vllm_vlm` against a local Qwen3-VL server.
    """
    _load_dotenv_once()

    def _call(prompt: str, *, frames: Optional[Sequence[Dict[str, Any]]] = None,
              **_kwargs: Any) -> str:
        try:
            from anthropic import Anthropic  # type: ignore
        except Exception as exc:  # pragma: no cover
            return f"__VLM_ERROR__[claude]: anthropic SDK unavailable: {exc}"

        api_key = (os.environ.get("ANTHROPIC_API_KEY") or "").strip()
        if not api_key:
            return "__VLM_ERROR__[claude]: ANTHROPIC_API_KEY not set"

        content: List[Dict[str, Any]] = []
        for path in _select_frame_paths(frames, max_frames):
            blob = _read_frame_bytes(path)
            if blob is None:
                continue
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": _image_mime(path),
                    "data": base64.b64encode(blob).decode("ascii"),
                },
            })
        content.append({"type": "text", "text": prompt})

        try:
            client = Anthropic(api_key=api_key, timeout=timeout)
            msg = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": content}],
            )
        except Exception as exc:
            return f"__VLM_ERROR__[claude]: {exc}"

        try:
            # Concatenate any text blocks in the response.
            parts = []
            for blk in msg.content or []:
                t = getattr(blk, "text", None)
                if t:
                    parts.append(t)
            return "".join(parts)
        except Exception as exc:
            return f"__VLM_ERROR__[claude]: malformed response: {exc}"

    return _call


# ---------------------------------------------------------------------------
# Unified dispatcher
# ---------------------------------------------------------------------------


def make_vlm(provider: str = "gpt4o", **kwargs: Any) -> Callable[..., str]:
    """Factory that returns a :class:`VLMCallable` for the named provider.

    Args:
        provider: One of
            * ``"gpt4o"`` / ``"openai"`` / ``"openrouter"``
              → :func:`make_gpt4o_vlm`.
            * ``"claude"`` / ``"anthropic"``
              → :func:`make_claude_vlm`.
            * ``"qwen3vl"`` / ``"vllm"`` / ``"qwen"``
              → :func:`make_vllm_vlm` (production target).
        **kwargs: Forwarded to the chosen factory. See each factory for
            the accepted parameters.

    The point of this dispatcher is that the pipeline driver only needs
    a string to switch between the GPT-4o scaffold and the Qwen3-VL
    production backend — no other code changes.
    """
    p = (provider or "").strip().lower()
    if p in ("gpt4o", "gpt-4o", "openai", "openrouter"):
        return make_gpt4o_vlm(**kwargs)
    if p in ("claude", "anthropic"):
        return make_claude_vlm(**kwargs)
    if p in ("qwen3vl", "qwen-vl", "qwen", "vllm", "local"):
        return make_vllm_vlm(**kwargs)
    raise ValueError(
        f"Unknown VLM provider '{provider}'. "
        "Expected one of: gpt4o | claude | qwen3vl."
    )
