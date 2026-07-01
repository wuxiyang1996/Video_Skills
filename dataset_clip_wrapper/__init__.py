"""Dataset clip wrappers for the four core video benchmarks."""

from .pipeline import build_canonical_example, iter_canonical_examples
from .schemas import (
    BackboneConfig,
    ClipPolicyConfig,
    RuntimeMode,
    VideoRegime,
    WrapperConfig,
)

__all__ = [
    "BackboneConfig",
    "ClipPolicyConfig",
    "RuntimeMode",
    "VideoRegime",
    "WrapperConfig",
    "build_canonical_example",
    "iter_canonical_examples",
]
