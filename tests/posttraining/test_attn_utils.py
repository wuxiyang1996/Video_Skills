from __future__ import annotations

import pytest

from trainer.grpo.attn_utils import flash_attn_available, resolve_attn_implementation


def test_resolve_flash_or_fail_closed() -> None:
    if flash_attn_available():
        assert resolve_attn_implementation("flash_attention_2") == "flash_attention_2"
    else:
        with pytest.raises(RuntimeError, match="flash_attn is required"):
            resolve_attn_implementation("flash_attention_2", allow_sdpa_fallback=False)
        assert (
            resolve_attn_implementation("flash_attention_2", allow_sdpa_fallback=True) == "sdpa"
        )


def test_sdpa_requires_explicit_fallback_flag() -> None:
    with pytest.raises(RuntimeError):
        resolve_attn_implementation("sdpa", allow_sdpa_fallback=False)
    assert resolve_attn_implementation("sdpa", allow_sdpa_fallback=True) == "sdpa"
