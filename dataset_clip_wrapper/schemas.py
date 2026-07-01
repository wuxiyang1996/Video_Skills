"""Canonical schema builders for dataset clip wrappers."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal

SCHEMA_VERSION = "video-skills-relaunch/v0.1"


class VideoRegime(str, Enum):
    SHORT = "short"
    LONG = "long"
    STREAMING = "streaming"


class RuntimeMode(str, Enum):
    EXPERT_DEMO = "expert_demo"
    VIDEO_ONLY = "video_only"


DatasetName = Literal["video_holmes", "cg_bench", "vrbench", "siv_bench"]


@dataclass
class ClipRetrievalConfig:
    """Coarse-clip retrieval gate (M3-style top-k before fine expansion)."""

    enabled: bool = True
    topk: int = 2
    threshold: float = 0.0
    mode: Literal["lexical", "sequential"] = "lexical"

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "topk": self.topk,
            "threshold": self.threshold,
            "mode": self.mode,
        }


@dataclass
class ClipPolicyConfig:
    """Clip segmentation hyperparameters."""

    strategy: str = "whole_video"
    window_s: float = 4.0
    overlap_s: float = 1.0
    coarse_window_s: float = 45.0
    fine_window_s: float = 8.0
    index_fine_expansion: Literal["none", "all", "retrieval_gated"] = "retrieval_gated"
    online: bool = False
    observation_end_s: float | None = None
    duration_s: float | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "strategy": self.strategy,
            "window_s": self.window_s,
            "overlap_s": self.overlap_s,
            "online": self.online,
        }
        if self.coarse_window_s is not None:
            payload["coarse_window_s"] = self.coarse_window_s
        if self.fine_window_s is not None:
            payload["fine_window_s"] = self.fine_window_s
        payload["index_fine_expansion"] = self.index_fine_expansion
        if self.observation_end_s is not None:
            payload["observation_end_s"] = self.observation_end_s
        if self.duration_s is not None:
            payload["duration_s"] = self.duration_s
        return payload

    @classmethod
    def for_regime(
        cls,
        regime: VideoRegime,
        *,
        observation_end_s: float | None = None,
        duration_s: float | None = None,
    ) -> ClipPolicyConfig:
        if regime == VideoRegime.SHORT:
            return cls(
                strategy="whole_video",
                window_s=4.0,
                overlap_s=1.0,
                online=False,
                observation_end_s=observation_end_s,
                duration_s=duration_s,
            )
        if regime == VideoRegime.LONG:
            return cls(
                strategy="hierarchical",
                coarse_window_s=30.0,
                fine_window_s=8.0,
                overlap_s=2.0,
                index_fine_expansion="retrieval_gated",
                online=False,
                observation_end_s=observation_end_s,
                duration_s=duration_s,
            )
        return cls(
            strategy="fixed_window",
            window_s=4.0,
            overlap_s=1.0,
            online=True,
            observation_end_s=observation_end_s,
            duration_s=duration_s,
        )

    @classmethod
    def dataset_default(cls, dataset: DatasetName, regime: VideoRegime | None = None) -> ClipPolicyConfig:
        inferred = regime or {
            "video_holmes": VideoRegime.SHORT,
            "siv_bench": VideoRegime.SHORT,
            "cg_bench": VideoRegime.LONG,
            "vrbench": VideoRegime.LONG,
        }[dataset]
        return cls.for_regime(inferred)


@dataclass
class BackboneConfig:
    """Perception backbone hyperparameters."""

    name: str = "annotation_only"
    model: str | None = None
    api_base: str = "https://openrouter.ai/api/v1/chat/completions"
    api_key_env: str = "OPENROUTER_API_KEY"
    keys_py_path: str | None = None
    max_clips: int | None = None
    temperature: float = 0.0
    request_frames: int = 4

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "model": self.model,
            "api_base": self.api_base,
            "api_key_env": self.api_key_env,
            "keys_py_path": self.keys_py_path,
            "max_clips": self.max_clips,
            "temperature": self.temperature,
            "request_frames": self.request_frames,
        }


@dataclass
class ClipSchemaConfig:
    """Multimodal clip-schema producer hyperparameters (Qwen via OpenRouter)."""

    backend: Literal["qwen", "video_tools"] = "qwen"
    model: str = "qwen/qwen3.5-9b"
    api_base: str = "https://openrouter.ai/api/v1/chat/completions"
    api_key_env: str = "OPENROUTER_API_KEY"
    keys_py_path: str | None = None
    temperature: float = 0.0
    request_frames: int = 4
    max_clips: int | None = None
    max_tokens: int | None = 1200
    reasoning_effort: str | None = "none"

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "model": self.model,
            "api_base": self.api_base,
            "api_key_env": self.api_key_env,
            "keys_py_path": self.keys_py_path,
            "temperature": self.temperature,
            "request_frames": self.request_frames,
            "max_clips": self.max_clips,
            "max_tokens": self.max_tokens,
            "reasoning_effort": self.reasoning_effort,
        }


@dataclass
class GraphComposerConfig:
    """Graph composer hyperparameters (gpt-oss-120B via OpenRouter)."""

    model: str = "openai/gpt-oss-120b"
    api_base: str = "https://openrouter.ai/api/v1/chat/completions"
    api_key_env: str = "OPENROUTER_API_KEY"
    keys_py_path: str | None = None
    temperature: float = 0.0
    use_llm_planner: bool = True
    max_tokens: int | None = 1800
    reasoning_effort: str | None = "minimal"

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "api_base": self.api_base,
            "api_key_env": self.api_key_env,
            "keys_py_path": self.keys_py_path,
            "temperature": self.temperature,
            "use_llm_planner": self.use_llm_planner,
            "max_tokens": self.max_tokens,
            "reasoning_effort": self.reasoning_effort,
        }


@dataclass
class ClipSpan:
    start_s: float
    end_s: float
    granularity: Literal["whole", "coarse", "fine"] = "whole"
    parent_index: int | None = None
    clip_index: int = 0

    def to_dict(self) -> dict[str, float]:
        return {"start_s": self.start_s, "end_s": self.end_s}


@dataclass
class WrapperConfig:
    dataset_root: str
    dataset: DatasetName
    regime: VideoRegime = VideoRegime.SHORT
    mode: RuntimeMode = RuntimeMode.EXPERT_DEMO
    clip_policy: ClipPolicyConfig | None = None
    retrieval: ClipRetrievalConfig = field(default_factory=ClipRetrievalConfig)
    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    clip_schema: ClipSchemaConfig = field(default_factory=ClipSchemaConfig)
    graph_composer: GraphComposerConfig = field(default_factory=GraphComposerConfig)
    split: str = "train"
    limit: int | None = None
    run_backbone: bool = False
    run_clip_schema: bool = False
    run_graph_compose: bool = False

    def resolved_clip_policy(self, duration_s: float | None = None) -> ClipPolicyConfig:
        from .dataset_graph_presets import clip_policy_for

        policy = self.clip_policy or clip_policy_for(self.dataset, self.regime, duration_s=duration_s)
        if duration_s is not None:
            policy.duration_s = duration_s
        if self.regime == VideoRegime.STREAMING and policy.observation_end_s is None and duration_s:
            policy.observation_end_s = duration_s
        policy.online = self.regime == VideoRegime.STREAMING or policy.online
        return policy


def make_canonical_shell(
    *,
    example_id: str,
    dataset: DatasetName,
    task_family: str,
    split: str,
    video: dict[str, Any],
    question: dict[str, Any],
    mode: RuntimeMode,
    clip_policy: ClipPolicyConfig,
    backbone: BackboneConfig,
    hidden_sources: list[str] | None = None,
    video_regime: VideoRegime | None = None,
    retrieval: ClipRetrievalConfig | None = None,
) -> dict[str, Any]:
    hidden = hidden_sources or []
    visible = ["video", "question"]
    if mode == RuntimeMode.EXPERT_DEMO:
        visible.extend(["subtitles", "captions", "dataset_annotations", "ground_truth_clues"])
    else:
        visible.extend(["automatic_clips", "automatic_segments"])

    return {
        "schema_version": SCHEMA_VERSION,
        "example_id": example_id,
        "dataset": dataset,
        "split": split,
        "task_family": task_family,
        "video": video,
        "question": question,
        "evidence_candidates": [],
        "available_inputs": {
            "mode": mode.value,
            "visible_to_agent": visible,
            "notes": "Generated by dataset_clip_wrapper",
        },
        "hidden_supervision": {
            "available_for_training": True,
            "available_for_inference": False,
            "sources": hidden,
        },
        "evidence_index": {
            "index_id": f"{dataset}:{example_id}:clip_index:v0",
            "index_type": "clip_memory_graph",
            "layer": "clue_memory",
            "visible_in_modes": ["expert_demo", "video_only"],
            "clip_policy": {**clip_policy.to_dict(), "video_regime": (video_regime or VideoRegime.SHORT).value},
            "retrieval": (retrieval or ClipRetrievalConfig()).to_dict(),
            "backbone": backbone.to_dict(),
            "node_types": ["clip", "observation", "subtitle_span", "caption_span", "dialogue_span", "event", "entity"],
            "edge_types": ["temporal_next", "derived_from", "entity_mention", "same_entity"],
            "nodes": [],
            "edges": [],
        },
        "raw_source_refs": [],
        "trust_policy": {
            "gold_sources": [],
            "strong_sources": [],
            "weak_sources": [],
            "model_labeled_sources": [],
        },
        "metadata": {
            "video_regime": (video_regime or VideoRegime.SHORT).value,
            "retrieval": (retrieval or ClipRetrievalConfig()).to_dict(),
            "wrapper_version": "dataset_clip_wrapper/v0.1",
        },
    }
