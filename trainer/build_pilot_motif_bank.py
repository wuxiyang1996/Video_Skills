#!/usr/bin/env python3
"""Build a small ACTIVE pilot bank from expandable mined L2 skill sequences."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from motif import MotifBank, MotifLifecycleStatus, MotifRecord  # noqa: E402
from motif.online_expand import expand_motif_record  # noqa: E402


PREFERRED = {
    "l2_sequence:df0e6d8116",
    "l2_sequence:267191d154",
    "l2_sequence:3ab7a8f183",
    "l2_sequence:ca8525c105",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        default="motif/output/mined_l1_l2_motif_bank.jsonl",
    )
    parser.add_argument(
        "--output",
        default="motif/output/pilot_online_motif_bank.jsonl",
    )
    parser.add_argument("--max-motifs", type=int, default=6)
    args = parser.parse_args(argv)

    source = MotifBank.load_jsonl(args.source)
    selected: list[MotifRecord] = []
    # Prefer curated short sequences first.
    for motif_id in PREFERRED:
        record = source.get(motif_id)
        if record is None:
            continue
        if expand_motif_record(record).expansion_valid:
            selected.append(record)
    for record in source.records:
        if record.motif_id in {r.motif_id for r in selected}:
            continue
        if record.motif_type != "l2_skill_sequence":
            continue
        if not expand_motif_record(record).expansion_valid:
            continue
        selected.append(record)
        if len(selected) >= int(args.max_motifs):
            break
    selected = selected[: int(args.max_motifs)]

    out_bank = MotifBank()
    for record in selected:
        record.status = MotifLifecycleStatus.ACTIVE
        # Prefer compressed sequence for online expand.
        l2 = dict(record.l2_template or {})
        compact = l2.get("compressed_skill_sequence") or l2.get("skill_sequence")
        if compact:
            l2["skill_sequence"] = list(compact)
            l2["compressed_skill_sequence"] = list(compact)
        record.l2_template = l2
        out_bank.add(record)

    out = Path(args.output)
    out_bank.save_jsonl(out)
    summary = {
        "source": str(args.source),
        "output": str(out),
        "motif_ids": out_bank.motif_ids,
        "summary": out_bank.summary(),
    }
    out.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
