"""Persistent exact-request cache (Multi-hop ExactRequestCache pattern)."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping


def stable_hash(payload: Mapping[str, Any] | list[Any] | str | int | float | bool | None) -> str:
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


class ExactRequestCache:
    """Persistent common-randomness cache for structured model responses."""

    def __init__(self, path: str | Path, identity: Mapping[str, Any]) -> None:
        self.path = Path(path)
        self.identity = dict(identity)
        self.identity_sha256 = stable_hash(self.identity)
        self.entries: dict[str, dict[str, Any]] = {}
        if self.path.exists():
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            if payload.get("identity_sha256") != self.identity_sha256:
                raise ValueError("exact-request cache belongs to a different backend identity")
            self.entries = {
                str(key): dict(value) for key, value in (payload.get("entries") or {}).items()
            }

    @staticmethod
    def request_key(request: Mapping[str, Any]) -> str:
        return stable_hash(request)

    def get(self, request: Mapping[str, Any]) -> dict[str, Any] | None:
        row = self.entries.get(self.request_key(request))
        return dict(row) if row is not None else None

    def put(self, request: Mapping[str, Any], response: Mapping[str, Any]) -> None:
        key = self.request_key(request)
        value = dict(response)
        existing = self.entries.get(key)
        if existing is not None and existing != value:
            raise ValueError("attempted to overwrite an exact request with a different response")
        self.entries[key] = value
        payload = {
            "schema_version": 1,
            "identity": self.identity,
            "identity_sha256": self.identity_sha256,
            "entries": dict(sorted(self.entries.items())),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, self.path)
