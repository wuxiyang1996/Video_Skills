"""JSONL-backed motif bank."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from .schemas import MotifLifecycleStatus, MotifRecord


class MotifBank:
    """Persistent registry for reusable L1/L2 graph motifs."""

    def __init__(self, records: Iterable[MotifRecord] | None = None) -> None:
        self._records: dict[str, MotifRecord] = {}
        for record in records or ():
            self.add(record)

    def __len__(self) -> int:
        return len(self._records)

    def __contains__(self, motif_id: str) -> bool:
        return motif_id in self._records

    @property
    def motif_ids(self) -> list[str]:
        return sorted(self._records)

    @property
    def records(self) -> list[MotifRecord]:
        return [self._records[motif_id] for motif_id in self.motif_ids]

    def add(self, record: MotifRecord) -> None:
        self._records[record.motif_id] = record

    def get(self, motif_id: str) -> MotifRecord | None:
        return self._records.get(motif_id)

    def require(self, motif_id: str) -> MotifRecord:
        record = self.get(motif_id)
        if record is None:
            raise KeyError(f"Unknown motif_id: {motif_id}")
        return record

    def active_records(self) -> list[MotifRecord]:
        return [
            record
            for record in self.records
            if record.status in {
                MotifLifecycleStatus.VERIFIED,
                MotifLifecycleStatus.ACTIVE,
            }
        ]

    def save_jsonl(self, path: Path | str) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as handle:
            for record in self.records:
                handle.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")

    @classmethod
    def load_jsonl(cls, path: Path | str) -> "MotifBank":
        input_path = Path(path)
        bank = cls()
        if not input_path.exists():
            return bank
        with input_path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                bank.add(MotifRecord.from_dict(json.loads(line)))
        return bank

    def summary(self) -> dict[str, object]:
        status_counts: dict[str, int] = {}
        type_counts: dict[str, int] = {}
        for record in self.records:
            status_counts[record.status.value] = status_counts.get(record.status.value, 0) + 1
            type_counts[record.motif_type] = type_counts.get(record.motif_type, 0) + 1
        return {
            "motif_count": len(self),
            "status_counts": status_counts,
            "motif_type_counts": type_counts,
        }
