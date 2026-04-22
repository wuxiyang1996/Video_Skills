"""Visual grounding layer for Video_Skills / COS-PLAY.

Implements the unified visual-grounding pipeline defined across
``infra_plans/``:

- ``video_benchmarks_grounding.md`` — 5-step pipeline, ``SocialVideoGraph``
  schema, benchmark adapters.
- ``agentic_memory_design.md`` — episodic / semantic / state stores with
  evidence as an attachment (not a top-level memory product).
- ``actors_reasoning_model.md`` — direct vs. retrieval execution paths,
  frozen tool contracts.
- ``atomic_skills_hop_refactor_execution_plan.md`` — infrastructure
  primitives live here (``observe_segment`` etc.), not in the skill bank.

Short videos (Video-Holmes, SIV-Bench) → :class:`DirectContext`.
Long videos (VRBench, LongVideoBench, CG-Bench, M3-Bench)
  → :class:`SocialVideoGraph`.

The public entry points are :func:`build_grounded_context` plus the
``build_for_*`` benchmark presets.
"""

from visual_grounding.schemas import (
    DirectContext,
    Entity,
    EntityProfile,
    EvidenceRef,
    Event,
    GroundedWindow,
    GroundingMode,
    GroundingNode,
    Interaction,
    NodeType,
    SocialHypothesis,
)
from visual_grounding.segmenter import (
    Window,
    adaptive_segment,
    probe_duration,
    DEFAULT_SHORT_THRESHOLD_SECONDS,
)
from visual_grounding.perception import (
    SampledFrame,
    align_subtitles_to_window,
    apply_subtitle_mode,
    detect_entities,
    parse_subtitle_file,
    sample_frames,
)
from visual_grounding.local_grounder import (
    GROUNDING_PROMPT_TEMPLATE,
    VLMCallable,
    ground_window,
    ground_windows_batch,
)
from visual_grounding.vlm_backends import (
    make_claude_vlm,
    make_gpt4o_vlm,
    make_vllm_vlm,
    make_vlm,
)
from visual_grounding.consolidator import (
    AttributeEntityResolver,
    EmbeddingEntityResolver,
    distill_semantic_summaries,
    merge_adjacent_windows,
    resolve_entities,
    windows_to_nodes,
)
from visual_grounding.social_video_graph import SocialVideoGraph
from visual_grounding.pipeline import (
    GroundedContext,
    build_for_cg_bench,
    build_for_long_video_bench,
    build_for_m3_bench,
    build_for_siv_bench,
    build_for_video_holmes,
    build_for_vrbench,
    build_grounded_context,
)

# --- Typed grounding stack --------------------------------------------------
from visual_grounding.grounding_schemas import (
    BeliefCandidate,
    EntityState,
    EventSpan,
    GroundedClip,
    InteractionEdge,
    MemoryKind,
    MemoryRecord,
    RawObservation,
    SourceType,
    SubtitleSpan,
    TemporalRelation,
    VideoSegment,
    VisibilityState,
    new_grounding_id,
)
from visual_grounding.segmenter import VideoSegmenter
from visual_grounding.subtitle_aligner import SubtitleAligner, load_subtitles
from visual_grounding.observation_extractor import ObservationExtractor
from visual_grounding.entity_tracker import EntityTracker
from visual_grounding.event_grounder import EventGrounder
from visual_grounding.social_state_grounder import SocialStateGrounder
from visual_grounding.temporal_grounder import TemporalGrounder
from visual_grounding.grounding_normalizer import GroundingNormalizer
from visual_grounding.memory_projection import MemoryProjection
from visual_grounding.grounding_runtime import GroundingRuntime
from visual_grounding.benchmark_adapters import (
    AdapterConfig,
    BENCHMARK_CONFIGS,
    BenchmarkAdapter,
    adapter_for_cg_bench,
    adapter_for_long_video_bench,
    adapter_for_m3_bench,
    adapter_for_siv_bench,
    adapter_for_video_holmes,
    adapter_for_vrbench,
    build_runtime,
)


__all__ = [
    # schemas
    "DirectContext",
    "Entity",
    "EntityProfile",
    "EvidenceRef",
    "Event",
    "GroundedWindow",
    "GroundingMode",
    "GroundingNode",
    "Interaction",
    "NodeType",
    "SocialHypothesis",
    # segmenter
    "Window",
    "adaptive_segment",
    "probe_duration",
    "DEFAULT_SHORT_THRESHOLD_SECONDS",
    # perception
    "SampledFrame",
    "align_subtitles_to_window",
    "apply_subtitle_mode",
    "detect_entities",
    "parse_subtitle_file",
    "sample_frames",
    # local grounding
    "GROUNDING_PROMPT_TEMPLATE",
    "VLMCallable",
    "ground_window",
    "ground_windows_batch",
    "make_gpt4o_vlm",
    "make_claude_vlm",
    "make_vllm_vlm",
    "make_vlm",
    # consolidation
    "AttributeEntityResolver",
    "EmbeddingEntityResolver",
    "distill_semantic_summaries",
    "merge_adjacent_windows",
    "resolve_entities",
    "windows_to_nodes",
    # graph
    "SocialVideoGraph",
    # pipeline
    "GroundedContext",
    "build_for_cg_bench",
    "build_for_long_video_bench",
    "build_for_m3_bench",
    "build_for_siv_bench",
    "build_for_video_holmes",
    "build_for_vrbench",
    "build_grounded_context",
    # typed grounding schemas
    "BeliefCandidate",
    "EntityState",
    "EventSpan",
    "GroundedClip",
    "InteractionEdge",
    "MemoryKind",
    "MemoryRecord",
    "RawObservation",
    "SourceType",
    "SubtitleSpan",
    "TemporalRelation",
    "VideoSegment",
    "VisibilityState",
    "new_grounding_id",
    # typed grounding modules
    "VideoSegmenter",
    "SubtitleAligner",
    "load_subtitles",
    "ObservationExtractor",
    "EntityTracker",
    "EventGrounder",
    "SocialStateGrounder",
    "TemporalGrounder",
    "GroundingNormalizer",
    "MemoryProjection",
    "GroundingRuntime",
    # benchmark adapters
    "AdapterConfig",
    "BENCHMARK_CONFIGS",
    "BenchmarkAdapter",
    "adapter_for_cg_bench",
    "adapter_for_long_video_bench",
    "adapter_for_m3_bench",
    "adapter_for_siv_bench",
    "adapter_for_video_holmes",
    "adapter_for_vrbench",
    "build_runtime",
]
