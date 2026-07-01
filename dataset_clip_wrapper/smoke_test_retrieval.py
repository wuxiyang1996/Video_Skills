#!/usr/bin/env python3
"""Offline tests for M3-style retrieve-gated hierarchical segmentation."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PKG_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PKG_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))

from dataset_clip_wrapper.clip_policy import segment_coarse_index, segment_perception_clips, segment_video
from dataset_clip_wrapper.clip_retrieval import retrieve_coarse_clips
from dataset_clip_wrapper.pipeline import iter_canonical_examples
from dataset_clip_wrapper.llm_pipeline import _resolve_perception_spans
from dataset_clip_wrapper.schemas import (
    ClipPolicyConfig,
    ClipRetrievalConfig,
    RuntimeMode,
    VideoRegime,
    WrapperConfig,
)


def main() -> int:
    long_policy = ClipPolicyConfig.for_regime(VideoRegime.LONG, duration_s=2741.0)
    coarse = segment_coarse_index(2741.0, long_policy)
    full = segment_video(2741.0, long_policy, fine_expansion="all")
    retrieval = retrieve_coarse_clips(
        coarse_spans=coarse,
        query_text="Where did the person hide the key before leaving the kitchen?",
        segments=[],
        topk=2,
    )
    perception = segment_perception_clips(
        2741.0,
        long_policy,
        selected_coarse_indices=retrieval["selected_coarse_indices"],
    )
    fine_only = [s for s in perception if s.granularity == "fine"]

    cg_config = WrapperConfig(
        dataset_root="/fs/gamma-projects/vlm-robot/datasets",
        dataset="cg_bench",
        regime=VideoRegime.LONG,
        mode=RuntimeMode.EXPERT_DEMO,
        split="train",
        limit=1,
    )
    cg = next(iter_canonical_examples(cg_config))
    duration_s = float(cg["video"]["duration_s"])
    policy = cg_config.resolved_clip_policy(duration_s)
    _, perception_meta = _resolve_perception_spans(
        duration_s=duration_s,
        clip_policy=policy,
        regime=VideoRegime.LONG,
        retrieval_config=ClipRetrievalConfig(topk=2),
        question_text=cg["question"].get("question_text", ""),
        visible_segments=cg["video"]["segments"],
    )

    report = {
        "synthetic_2741s": {
            "coarse_count": len(coarse),
            "full_hierarchical_count": len(full),
            "retrieved_coarse": retrieval["selected_coarse_indices"],
            "gated_fine_count": len(fine_only),
            "passed": len(coarse) < len(full) and len(fine_only) < 50,
        },
        "cg_bench_sample": {
            "index_clip_count": cg["metadata"]["index_clip_count"],
            "coarse_clip_count": cg["metadata"]["coarse_clip_count"],
            "fine_clip_count": cg["metadata"]["fine_clip_count"],
            "perception_fine_count": perception_meta.get("perception_clip_count"),
            "retrieved_coarse": perception_meta.get("retrieval", {}).get("selected_coarse_indices"),
            "passed": cg["metadata"]["fine_clip_count"] == 0 and cg["metadata"]["coarse_clip_count"] > 0,
        },
    }
    report["all_passed"] = report["synthetic_2741s"]["passed"] and report["cg_bench_sample"]["passed"]
    print(json.dumps(report, indent=2))
    return 0 if report["all_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
