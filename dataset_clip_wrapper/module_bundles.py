"""Module bundle registry for the video-skill relaunch codebase.

This registry is documentation with teeth: it names the intended ownership
boundary for wrapper modules and keeps new files from drifting into the wrong
place.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModuleBundle:
    name: str
    purpose: str
    modules: tuple[str, ...]


MODULE_BUNDLES: tuple[ModuleBundle, ...] = (
    ModuleBundle(
        name="core_schema_and_config",
        purpose="Canonical example schema, config objects, CLI glue, and dataset profile defaults.",
        modules=(
            "cli",
            "dataset_graph_presets",
            "pipeline",
            "schemas",
        ),
    ),
    ModuleBundle(
        name="dataset_adapters",
        purpose="Dataset-specific conversion into canonical video QA examples.",
        modules=(
            "adapters.__init__",
            "adapters.base",
            "adapters.cg_bench",
            "adapters.siv_bench",
            "adapters.streaming_video",
            "adapters.video_holmes",
            "adapters.vrbench",
        ),
    ),
    ModuleBundle(
        name="perception_and_clip_tools",
        purpose="Video segmentation, clip-schema production, subtitles, local video probes, and model client plumbing.",
        modules=(
            "perception.__init__",
            "perception.backbone",
            "perception.clip_policy",
            "perception.clip_schema",
            "perception.openrouter_client",
            "perception.subtitles",
            "perception.video_probe",
            "perception.video_tool_backend",
        ),
    ),
    ModuleBundle(
        name="l1_clue_graph",
        purpose="Layer-1 clue-memory graph construction, composition, retrieval, and L1-to-L2 gating.",
        modules=(
            "l1_clue_graph.__init__",
            "l1_clue_graph.clip_retrieval",
            "l1_clue_graph.clue_memory",
            "l1_clue_graph.gate_l1_for_l2",
            "l1_clue_graph.graph_composer",
            "l1_clue_graph.graph_plan_validator",
            "l1_clue_graph.skill_graph_bridge",
        ),
    ),
    ModuleBundle(
        name="l2_reasoning_graph",
        purpose="Layer-2 reasoning rollout construction, GPT-OSS planning, recursive trajectory logging, and local repair traces.",
        modules=(
            "l2_reasoning_graph.__init__",
            "l2_reasoning_graph.fault_repair",
            "l2_reasoning_graph.l2_recursive_trace",
            "l2_reasoning_graph.reasoning_planner",
            "l2_reasoning_graph.reasoning_rollout",
        ),
    ),
    ModuleBundle(
        name="l2_repair_and_verification",
        purpose="Strict repair orchestration, runtime verification, quality reports, and final acceptance merging.",
        modules=(
            "verification.__init__",
            "verification.evaluate_l1_query_memory",
            "verification.evaluate_vrbench_video_only_graph",
            "verification.report_evidence_audit",
            "verification.report_final_acceptance",
            "verification.report_failure_taxonomy",
            "verification.report_l1_l2_quality",
            "verification.retrofit_l2_trajectory",
            "verification.run_repair_protocol",
            "verification.runtime_verifier",
            "report_evidence_audit",
            "report_final_acceptance",
            "report_failure_taxonomy",
            "report_l1_l2_quality",
            "retrofit_l2_trajectory",
            "run_repair_protocol",
        ),
    ),
    ModuleBundle(
        name="expert_demo_exports",
        purpose="Export accepted and abstaining L1/L2/repair trajectories as training-ready expert-demo candidates.",
        modules=(
            "expert_demos.__init__",
            "expert_demos.export_expert_demos",
            "export_expert_demos",
        ),
    ),
    ModuleBundle(
        name="training_manifests",
        purpose="Build split-aware train/dev/test manifests for expert-demo gathering and evaluation isolation.",
        modules=(
            "manifests.__init__",
            "manifests.build_training_manifests",
            "build_training_manifests",
        ),
    ),
    ModuleBundle(
        name="controller_training_exports",
        purpose="Adapt verified L1/L2 expert demos into canonical ReasoningTrace and compact SFT chat records.",
        modules=(
            "training.__init__",
            "training.trace_adapter",
            "export_reasoning_traces",
        ),
    ),
    ModuleBundle(
        name="training_controllers",
        purpose="Controller SFT adapters, LoRA training entrypoints, snapshot collection, and root export_*_sft shims.",
        modules=(
            "training.build_sft_splits",
            "training.collect_sft_snapshot",
            "training.l1_builder_sft_adapter",
            "training.l1_patch_sft_adapter",
            "training.l2_retrieval_sft_adapter",
            "training.l2_specialist_sft_adapter",
            "training.motif_evidence_sft_adapter",
            "training.motif_sft_adapter",
            "training.repair_report_stepwise_sft_adapter",
            "training.sft_common",
            "training.stepwise_sft_adapter",
            "training.train_lora_sft",
            "training.verifier_sft_adapter",
            "export_l1_builder_sft",
            "export_l1_patch_sft",
            "export_l2_retrieval_sft",
            "export_motif_sft",
            "export_stepwise_sft",
            "export_verifier_sft",
            "collect_sft_snapshot",
        ),
    ),
    ModuleBundle(
        name="motif_management",
        purpose="Mine, store, retrieve, and evaluate optional L1/L2 graph motif priors, including Qwen/GPT-OSS motif agents.",
        modules=(
            "motifs.__init__",
            "motifs.agent",
            "motifs.bank",
            "motifs.build_motif_bank",
            "motifs.canonicalize",
            "motifs.expansion",
            "motifs.instance_miner",
            "motifs.lifecycle",
            "motifs.llm_agent",
            "motifs.mine_existing_l1_l2",
            "motifs.miner",
            "motifs.promotion",
            "motifs.registry",
            "motifs.retrieval",
            "motifs.schemas",
            "motifs.transfer",
            "build_motif_bank",
        ),
    ),
    ModuleBundle(
        name="pipeline_runners",
        purpose="End-to-end runners that connect adapters, perception, L1 graph building, L2 rollout, and reports.",
        modules=(
            "runners.__init__",
            "runners.llm_pipeline",
            "runners.run_llm_pipeline",
            "runners.run_staged_llm_pipeline",
            "run_llm_pipeline",
            "run_staged_llm_pipeline",
        ),
    ),
    ModuleBundle(
        name="smoke_tests",
        purpose="Small executable checks for schema boundaries, graph builders, repair, and pipeline contracts.",
        modules=(
            "tests.__init__",
            "tests.core.__init__",
            "tests.core.smoke_test",
            "tests.core.smoke_test_module_bundles",
            "tests.core.smoke_test_two_layer_schema",
            "tests.l1.__init__",
            "tests.l1.smoke_test_coarse_fine_graph_crafting",
            "tests.l1.smoke_test_graph_compose",
            "tests.l1.smoke_test_graph_plan_validator",
            "tests.l1.smoke_test_long_coarse_fine_profile",
            "tests.l1.smoke_test_neighbor_vlm_l1_graph_compose",
            "tests.l1.smoke_test_retrieval",
            "tests.l1.smoke_test_short_multi_hop_profile",
            "tests.l1.smoke_test_vlm_l1_graph_compose",
            "tests.l2.__init__",
            "tests.l2.smoke_test_fault_repair",
            "tests.l2.smoke_test_l2_recursive_trace",
            "tests.l2.smoke_test_multi_hop_reasoning_skills",
            "tests.l2.smoke_test_reasoning_rollout",
            "tests.l2.smoke_test_skill_executor",
            "tests.motifs.__init__",
            "tests.motifs.smoke_test_motif_agent",
            "tests.perception.__init__",
            "tests.perception.smoke_test_video_only_takein",
            "tests.perception.smoke_test_video_tools",
            "tests.training.__init__",
            "tests.training.smoke_test_export_expert_demos",
            "tests.training.smoke_test_training_manifests",
            "tests.verification.__init__",
            "tests.verification.smoke_test_long_retrieval_repair",
            "tests.test_entity_reference_contract",
            # Root compatibility shims (python -m dataset_clip_wrapper.tests.smoke_test_*)
            "tests.smoke_test",
            "tests.smoke_test_coarse_fine_graph_crafting",
            "tests.smoke_test_export_expert_demos",
            "tests.smoke_test_fault_repair",
            "tests.smoke_test_graph_compose",
            "tests.smoke_test_graph_plan_validator",
            "tests.smoke_test_l2_recursive_trace",
            "tests.smoke_test_long_coarse_fine_profile",
            "tests.smoke_test_long_retrieval_repair",
            "tests.smoke_test_module_bundles",
            "tests.smoke_test_motif_agent",
            "tests.smoke_test_multi_hop_reasoning_skills",
            "tests.smoke_test_neighbor_vlm_l1_graph_compose",
            "tests.smoke_test_reasoning_rollout",
            "tests.smoke_test_retrieval",
            "tests.smoke_test_short_multi_hop_profile",
            "tests.smoke_test_skill_executor",
            "tests.smoke_test_trace_adapter",
            "tests.smoke_test_training_manifests",
            "tests.smoke_test_two_layer_schema",
            "tests.smoke_test_video_only_takein",
            "tests.smoke_test_video_tools",
            "tests.smoke_test_vlm_l1_graph_compose",
        ),
    ),
)


def bundle_by_module() -> dict[str, ModuleBundle]:
    out: dict[str, ModuleBundle] = {}
    for bundle in MODULE_BUNDLES:
        for module in bundle.modules:
            out[module] = bundle
    return out


def classify_module(module_name: str) -> str:
    """Return the bundle name for a dataset_clip_wrapper module path."""
    return bundle_by_module().get(module_name, ModuleBundle("other", "Unclassified module.", ())).name


def python_modules(package_dir: Path) -> set[str]:
    modules: set[str] = set()
    for path in package_dir.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        rel = path.relative_to(package_dir).with_suffix("")
        modules.add(".".join(rel.parts))
    return modules
