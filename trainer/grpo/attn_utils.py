"""Attention backend selection — prefer FlashAttention-2 for GRPO / LoRA."""

from __future__ import annotations

from typing import Any


def flash_attn_available() -> bool:
    try:
        import flash_attn  # noqa: F401

        return True
    except Exception:
        return False


def resolve_attn_implementation(
    preferred: str = "flash_attention_2",
    *,
    allow_sdpa_fallback: bool = False,
) -> str:
    """Return a transformers ``attn_implementation`` string.

    GRPO / LoRA training should use FlashAttention-2 on A6000. SDPA is only
    allowed when ``allow_sdpa_fallback=True`` (debug / install missing).
    """
    pref = (preferred or "flash_attention_2").strip().lower()
    if pref in {"flash_attention_2", "flash-attn", "flash_attn", "fa2"}:
        if flash_attn_available():
            return "flash_attention_2"
        if allow_sdpa_fallback:
            return "sdpa"
        raise RuntimeError(
            "flash_attn is required but not importable. "
            "Install on a GPU node via scripts/grpo/install_flash_attn.sh "
            "or pass allow_sdpa_fallback=True for debug only."
        )
    if pref == "sdpa":
        if not allow_sdpa_fallback:
            raise RuntimeError("sdpa requested but allow_sdpa_fallback=False")
        return "sdpa"
    if pref in {"eager", "flex_attention"}:
        return pref
    raise ValueError(f"unsupported attn implementation: {preferred}")


def assert_model_uses_flash_attn(model: Any) -> None:
    """Best-effort check that the loaded model config requests FA2."""
    cfg = getattr(model, "config", None)
    attn = None
    if cfg is not None:
        attn = getattr(cfg, "_attn_implementation", None) or getattr(cfg, "attn_implementation", None)
    if attn != "flash_attention_2":
        # Some PEFT wrappers nest the base model.
        base = getattr(model, "get_base_model", None)
        if callable(base):
            cfg = getattr(base(), "config", None)
            if cfg is not None:
                attn = getattr(cfg, "_attn_implementation", None) or getattr(cfg, "attn_implementation", None)
    if attn != "flash_attention_2":
        raise RuntimeError(
            f"expected flash_attention_2 on model config, got {attn!r}. "
            "Refusing to start GRPO without FlashAttention-2."
        )
