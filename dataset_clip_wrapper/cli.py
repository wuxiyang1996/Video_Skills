#!/usr/bin/env python3
"""CLI for dataset clip wrappers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .pipeline import iter_canonical_examples
from .schemas import BackboneConfig, ClipPolicyConfig, ClipRetrievalConfig, RuntimeMode, VideoRegime, WrapperConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Wrap video datasets into canonical clip schema.")
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["video_holmes", "cg_bench", "vrbench", "siv_bench"],
    )
    parser.add_argument(
        "--dataset-root",
        default="/fs/gamma-projects/vlm-robot/datasets",
        help="Root directory containing Video-Holmes, CG-Bench, VRBench, SIV-Bench",
    )
    parser.add_argument("--split", default="train", choices=["train", "test"])
    parser.add_argument(
        "--regime",
        default=None,
        choices=["short", "long", "streaming"],
        help="Video regime; defaults per dataset if omitted",
    )
    parser.add_argument("--mode", default="expert_demo", choices=["expert_demo", "video_only"])
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--output", default="dataset_clip_wrapper/output.jsonl")

    parser.add_argument("--clip-strategy", default=None, help="Override clip policy strategy")
    parser.add_argument("--window-s", type=float, default=None)
    parser.add_argument("--overlap-s", type=float, default=None)
    parser.add_argument("--coarse-window-s", type=float, default=None)
    parser.add_argument("--fine-window-s", type=float, default=None)
    parser.add_argument("--observation-end-s", type=float, default=None, help="Streaming visibility cutoff")
    parser.add_argument(
        "--index-fine-expansion",
        default=None,
        choices=["none", "all", "retrieval_gated"],
        help="Long-video index: coarse only, all fine, or retrieve-gated fine",
    )
    parser.add_argument("--retrieval-topk", type=int, default=None)
    parser.add_argument("--retrieval-mode", default=None, choices=["lexical", "sequential"])
    parser.add_argument("--no-retrieval", action="store_true", help="Disable coarse retrieval gate")

    parser.add_argument(
        "--backbone",
        default="annotation_only",
        choices=["annotation_only", "openrouter"],
        help="Perception backbone hyperparameter",
    )
    parser.add_argument(
        "--backbone-model",
        default="openai/gpt-5-mini",
        help="OpenRouter model id when --backbone=openrouter",
    )
    parser.add_argument("--keys-py", default="/fs/gamma-projects/vlm-robot/keys.py")
    parser.add_argument("--backbone-max-clips", type=int, default=None)
    parser.add_argument("--run-backbone", action="store_true", help="Call backbone to caption clip spans")
    parser.add_argument("--backbone-temperature", type=float, default=0.0)
    parser.add_argument("--backbone-request-frames", type=int, default=4)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    regime = VideoRegime(args.regime) if args.regime else None
    dataset_regime = regime or {
        "video_holmes": VideoRegime.SHORT,
        "siv_bench": VideoRegime.SHORT,
        "cg_bench": VideoRegime.LONG,
        "vrbench": VideoRegime.LONG,
    }[args.dataset]

    clip_policy = ClipPolicyConfig.dataset_default(args.dataset, dataset_regime)
    if args.clip_strategy:
        clip_policy.strategy = args.clip_strategy
    if args.window_s is not None:
        clip_policy.window_s = args.window_s
    if args.overlap_s is not None:
        clip_policy.overlap_s = args.overlap_s
    if args.coarse_window_s is not None:
        clip_policy.coarse_window_s = args.coarse_window_s
    if args.fine_window_s is not None:
        clip_policy.fine_window_s = args.fine_window_s
    if args.observation_end_s is not None:
        clip_policy.observation_end_s = args.observation_end_s
    if args.index_fine_expansion:
        clip_policy.index_fine_expansion = args.index_fine_expansion  # type: ignore[assignment]

    retrieval = ClipRetrievalConfig()
    if args.no_retrieval:
        retrieval.enabled = False
    if args.retrieval_topk is not None:
        retrieval.topk = args.retrieval_topk
    if args.retrieval_mode:
        retrieval.mode = args.retrieval_mode  # type: ignore[assignment]

    config = WrapperConfig(
        dataset_root=args.dataset_root,
        dataset=args.dataset,
        regime=dataset_regime,
        mode=RuntimeMode(args.mode),
        clip_policy=clip_policy,
        retrieval=retrieval,
        backbone=BackboneConfig(
            name=args.backbone,
            model=args.backbone_model,
            keys_py_path=args.keys_py,
            max_clips=args.backbone_max_clips,
            temperature=args.backbone_temperature,
            request_frames=args.backbone_request_frames,
        ),
        split=args.split,
        limit=args.limit,
        run_backbone=args.run_backbone,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for example in iter_canonical_examples(config):
            handle.write(json.dumps(example, ensure_ascii=False) + "\n")
            count += 1

    print(
        json.dumps(
            {
                "dataset": args.dataset,
                "regime": dataset_regime.value,
                "mode": args.mode,
                "backbone": config.backbone.to_dict(),
                "written": count,
                "output": str(output_path),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
