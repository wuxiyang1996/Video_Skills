"""Dataset × video-regime presets for layer-1 clue-memory indexing."""

from __future__ import annotations

from .schemas import BenchmarkProfile, ClipPolicyConfig, ClipRetrievalConfig, DatasetName, VideoRegime

DATASET_DEFAULT_REGIME: dict[DatasetName, VideoRegime] = {
    "video_holmes": VideoRegime.SHORT,
    "siv_bench": VideoRegime.SHORT,
    "cg_bench": VideoRegime.LONG,
    "vrbench": VideoRegime.LONG,
    "ovo_bench": VideoRegime.STREAMING,
    "videomme": VideoRegime.SHORT,
}

DATASET_TASK_FAMILY: dict[DatasetName, str] = {
    "video_holmes": "short_video_social_causal_qa",
    "siv_bench": "short_video_social_interaction_qa",
    "cg_bench": "long_video_clue_grounding_qa",
    "vrbench": "long_video_temporal_chain_qa",
    "ovo_bench": "streaming_video_realtime_qa",
    "videomme": "short_video_whole_video_qa",
}

SHORT_MULTI_HOP_DATASETS: tuple[DatasetName, ...] = ("video_holmes", "videomme", "ovo_bench")

PROFILE_TASK_FAMILY: dict[BenchmarkProfile, str] = {
    BenchmarkProfile.SHORT_MULTI_HOP: "short_video_multi_hop_qa",
}

# Hidden supervision fields available per dataset (expert_demo only).
DATASET_HIDDEN_SOURCES: dict[DatasetName, list[str]] = {
    "video_holmes": ["official_answer", "segment_annotations", "inference_shots", "key_relationships"],
    "siv_bench": ["official_answer"],
    "cg_bench": ["official_answer", "clue_intervals", "clue_clips"],
    "vrbench": ["official_answer", "reasoning_process", "video_summary"],
    "ovo_bench": ["official_answer"],
    "videomme": ["official_answer"],
}

# Layer-1 index characteristics per dataset under each regime.
DATASET_LAYER1_PROFILE: dict[DatasetName, dict[str, str]] = {
    "video_holmes": {
        "short": "whole_video + 4s fine windows; lightweight index; segment/inference seeds in expert_demo",
        "long": "hierarchical 30s coarse optional; rarely used",
        "streaming": "fixed_window 4s online; observation_end_s enforces causal clip visibility",
    },
    "siv_bench": {
        "short": "whole_video + subtitle-aligned spans; weak evidence; very short clips",
        "long": "same as short; videos are typically <2 min",
        "streaming": "fixed_window online; subtitle chunks filtered by observation_end_s",
    },
    "cg_bench": {
        "short": "hierarchical if video <10 min else same as long",
        "long": "hierarchical 30s coarse index + retrieval_gated 8s fine perception",
        "streaming": "hierarchical 30s coarse online index only (no 4s fine); observation_end_s enforced",
    },
    "vrbench": {
        "short": "hierarchical for clips <10 min",
        "long": "hierarchical 30s coarse + retrieval_gated fine; reasoning_process timestamps as expert seeds",
        "streaming": "hierarchical 30s coarse online index only; reasoning steps after observation_end_s hidden",
    },
    "ovo_bench": {
        "short": "fixed_window streaming clips around realtime anchors; mainly for coding smoke",
        "long": "not the default; OVO is evaluated as timestamped streaming QA",
        "streaming": "fixed_window online index with realtime question anchors from StreamBridge records",
    },
    "videomme": {
        "short": "whole_video + 4s fine windows for short VideoMME-style records",
        "long": "hierarchical if full VideoMME long videos are supplied",
        "streaming": "not the default; VideoMME is single-turn/offline in StreamBridge",
    },
}


def default_regime_for_dataset(dataset: DatasetName) -> VideoRegime:
    return DATASET_DEFAULT_REGIME[dataset]


def regime_for_dataset(dataset: DatasetName, profile: BenchmarkProfile = BenchmarkProfile.DEFAULT) -> VideoRegime:
    if profile == BenchmarkProfile.SHORT_MULTI_HOP and dataset in SHORT_MULTI_HOP_DATASETS:
        return VideoRegime.SHORT
    return default_regime_for_dataset(dataset)


def task_family_for(
    dataset: DatasetName,
    *,
    regime: VideoRegime,
    profile: BenchmarkProfile = BenchmarkProfile.DEFAULT,
    adapter_task_family: str | None = None,
) -> str:
    if profile == BenchmarkProfile.SHORT_MULTI_HOP and dataset in SHORT_MULTI_HOP_DATASETS and regime == VideoRegime.SHORT:
        return PROFILE_TASK_FAMILY[BenchmarkProfile.SHORT_MULTI_HOP]
    return adapter_task_family or DATASET_TASK_FAMILY[dataset]


def clip_policy_for(dataset: DatasetName, regime: VideoRegime, *, duration_s: float | None = None) -> ClipPolicyConfig:
    """Return regime-aware clip policy defaults for a dataset."""
    observation_end_s = duration_s if regime == VideoRegime.STREAMING and duration_s else None
    if regime == VideoRegime.SHORT:
        if dataset == "siv_bench":
            return ClipPolicyConfig.for_regime(
                VideoRegime.SHORT,
                duration_s=duration_s,
                observation_end_s=None,
            )
        return ClipPolicyConfig.for_regime(VideoRegime.SHORT, duration_s=duration_s)

    if regime == VideoRegime.STREAMING:
        # Long-form datasets: coarse 30s online index (M3-style), not 4s fine windows.
        if dataset in {"cg_bench", "vrbench"}:
            policy = ClipPolicyConfig.for_regime(VideoRegime.LONG, duration_s=duration_s, observation_end_s=observation_end_s)
            policy.online = True
            policy.index_fine_expansion = "none"
            if duration_s is not None and policy.observation_end_s is None:
                policy.observation_end_s = duration_s
            return policy
        if dataset == "ovo_bench":
            policy = ClipPolicyConfig.for_regime(
                VideoRegime.STREAMING,
                duration_s=duration_s,
                observation_end_s=observation_end_s,
            )
            policy.window_s = 4.0
            policy.overlap_s = 1.0
            return policy
        policy = ClipPolicyConfig.for_regime(VideoRegime.STREAMING, duration_s=duration_s, observation_end_s=observation_end_s)
        if duration_s is not None and policy.observation_end_s is None:
            policy.observation_end_s = duration_s
        return policy

    # long
    policy = ClipPolicyConfig.for_regime(VideoRegime.LONG, duration_s=duration_s)
    if dataset in {"cg_bench", "vrbench"}:
        policy.index_fine_expansion = "retrieval_gated"
    return policy


def retrieval_for(regime: VideoRegime) -> ClipRetrievalConfig:
    if regime == VideoRegime.SHORT:
        return ClipRetrievalConfig(enabled=False, topk=1)
    return ClipRetrievalConfig(enabled=True, topk=2, mode="lexical")
