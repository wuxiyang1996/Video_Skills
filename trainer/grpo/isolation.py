"""K-sample rollout isolation helpers.

Each sample in a GRPO group must get an independent environment copy so mutations
in one trajectory cannot leak into another.
"""

from __future__ import annotations

import copy
import json
from typing import Any, Mapping


def deep_isolate(example: Mapping[str, Any]) -> dict[str, Any]:
    """Return a deep copy safe for one rollout sample."""
    return copy.deepcopy(dict(example))


def assert_rollout_isolation(rollouts: list[Mapping[str, Any]]) -> None:
    """Fail if any two rollouts share nested object identity (shallow leak check)."""
    if len(rollouts) < 2:
        return
    for i in range(len(rollouts)):
        for j in range(i + 1, len(rollouts)):
            a = rollouts[i]
            b = rollouts[j]
            if a is b:
                raise AssertionError(f"rollouts[{i}] and rollouts[{j}] share root identity")
            meta_a = a.get("metadata") if isinstance(a, Mapping) else None
            meta_b = b.get("metadata") if isinstance(b, Mapping) else None
            if meta_a is not None and meta_a is meta_b:
                raise AssertionError(
                    f"rollouts[{i}] and rollouts[{j}] share metadata object identity"
                )


def fingerprint_example(example: Mapping[str, Any]) -> str:
    """Stable fingerprint used to detect accidental in-place mutation across samples."""
    return json.dumps(example, sort_keys=True, ensure_ascii=False, default=str)
