"""Terminal-reward on-policy GRPO for the L2 retrieval specialist.

The local L2 LoRA samples retrieval actions on unseen ``grpo_pool`` states.
Each action filters the visible clue graph; a fixed planner/executor then
produces the answer. Only evaluator-side answer correctness plus runtime
verification yields terminal success.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
import copy
import fcntl
import glob
import hashlib
import json
import os
import random
import signal
import shutil
import time
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from atomic_skills.skill_executor import SkillExecutor
from atomic_skills.skill_model_client import SkillModelClient
from dataset_clip_wrapper.l2_reasoning_graph.reasoning_planner import build_llm_reasoning_rollout
from dataset_clip_wrapper.perception.openrouter_client import OpenRouterClient, load_openrouter_api_key
from dataset_clip_wrapper.training.l2_oracle_retrieval_v5 import policy_catalog
from dataset_clip_wrapper.training.l2_pointwise_reranker_v8 import (
    TASK as POINTWISE_TASK,
    pointwise_state,
    relevance_action,
)
from dataset_clip_wrapper.training.l2_specialist_sft_adapter import SYSTEM as POINTWISE_SYSTEM
from dataset_clip_wrapper.training.sft_common import apply_chat_template_no_think, compact_visibility, strip_think_tags
from trainer.closed_loop_harness import load_frozen_l1_examples
from trainer.grpo.live_rollout import _grpo_skill_backend_config, probe_skill_model
from trainer.grpo.l2_dataset_rewards import (
    RELATIONSHIP_SUPPORT_VERSION,
    lexical_support,
    load_dataset_reward_supervision,
    supervision_key,
    temporal_retrieval_metrics,
)
from trainer.grpo.objective import (
    centered_group_advantages,
    clipped_grpo_loss,
    extract_json_object,
    plackett_luce_logprob,
)
from trainer.grpo.train_specialist_on_policy import _token_logprobs, _trim_generated
from trainer.split_filter import assert_role_exclusive, filter_examples_by_role, load_split_manifest
from trainer.artifact_hash import adapter_weight_sha256
from trainer.posttraining_manifest import build_posttraining_manifest, save_posttraining_manifest


SYSTEM = (
    "You are the Video_Skills L2 retrieval controller. Choose only from the "
    "visible retrieval state and return the requested tool action as JSON. "
    "Catalog entries are hypotheses, not verified facts. Return exactly one action with "
    'tool_name="select_coarse_clips" and arguments.coarse_indices as an integer list. '
    "Do not invent tool names or argument keys, and never select more entries than "
    "budget_state.topk."
)
ACTION_CONTRACT_VERSION = "select-coarse-clips-exact-v1"
POINTWISE_ACTION_CONTRACT_VERSION = "pointwise-logodds-set-sampling-v1"
DATASET_ROUTED_ACTION_CONTRACT_VERSION = "dataset-routed-cg-set-vh-pointwise-v1"
EXECUTOR_ISOLATION_VERSION = "selected-window-closure-v1"
EXECUTOR_FALLBACK_VERSION = "dataset-routed-cg-rule-vh-relative-mcq-typed-plan-v5"
EXECUTOR_CACHE_VERSION = "shared-locked-rollout-typed-plan-v2"
DATASET_BALANCING_VERSION = "equal-groups-cyclic-repeats-v1"
PROCESS_WARMUP_REWARD_VERSION = "dataset-routed-process-hit-aligned-v2"
TERMINAL_REWARD_VERSION = (
    f"dataset-aware-terminal-reward:{RELATIONSHIP_SUPPORT_VERSION}:"
    "verified-query-finalizer-repair-v1"
)
REFERENCE_RUNTIME_VERSION = "shared-base-frozen-reference-adapters-v1"
POINTWISE_GRADIENT_CONTRACT_VERSION = "score-space-vjp-candidate-recompute-v1"
RESUME_CHECKPOINT_VERSION = "group-boundary-optimizer-resume-v1"
OOM_OFFLOAD_FALLBACK_VERSION = "exact-group-recompute-save-on-cpu-inactive-state-offload-v3"


def retry_exact_backward_after_oom(
    normal_backward: Callable[[], Any],
    offloaded_backward: Callable[[], Any],
    *,
    optimizer: Any,
    empty_cache: Callable[[], None],
    oom_type: type[BaseException],
    prepare_retry: Callable[[], None] | None = None,
) -> tuple[Any, bool]:
    """Retry an exact group backward after discarding partial OOM gradients."""
    try:
        return normal_backward(), False
    except oom_type:
        # Do not invoke the fallback from inside this except block: Python keeps
        # the OOM traceback (and therefore the failed autograd graph) alive until
        # the block exits.  Clearing the cache while that graph is referenced
        # leaves almost all CUDA allocations live and makes the retry fail in its
        # first forward pass.
        optimizer.zero_grad(set_to_none=True)
    if prepare_retry is not None:
        prepare_retry()
    empty_cache()
    return offloaded_backward(), True


def _file_sha256(path: Path | None) -> str | None:
    if path is None:
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resume_run_signature(
    args: argparse.Namespace,
    *,
    dataset_adapter_paths: Mapping[str, Path],
    pointwise_temperature: float,
) -> str:
    """Fingerprint every input that may change group order or optimizer math."""
    payload = {
        "checkpoint_contract": RESUME_CHECKPOINT_VERSION,
        "source_adapter_sha256": adapter_weight_sha256(args.adapter),
        "dataset_adapter_sha256": {
            dataset: adapter_weight_sha256(path)
            for dataset, path in sorted(dataset_adapter_paths.items())
        },
        "split_manifest_sha256": _file_sha256(args.split_manifest),
        "allowlist_sha256": _file_sha256(args.example_id_allowlist),
        "frozen_l1_glob": list(args.frozen_l1_glob),
        "split_role": args.split_role,
        "datasets": args.datasets,
        "max_groups": int(args.max_groups),
        "repeats_per_example": int(args.repeats_per_example),
        "repeat_start_index": int(args.repeat_start_index),
        "k": int(args.k),
        "cg_topk": int(args.cg_topk),
        "video_holmes_topk": int(args.video_holmes_topk),
        "ppo_epochs": int(args.ppo_epochs),
        "learning_rate": float(args.learning_rate),
        "kl_coef": float(args.kl_coef),
        "clip_eps": float(args.clip_eps),
        "temperature": float(args.temperature),
        "pointwise_temperature": float(pointwise_temperature),
        "pointwise_train_batch_size": int(args.pointwise_train_batch_size),
        "seed": int(args.seed),
        "boundary_anchor_index0": bool(args.boundary_anchor_index0),
        "preserve_allowlist_order": bool(args.preserve_allowlist_order),
        "dataset_balanced_sampling": bool(args.dataset_balanced_sampling),
        "require_process_supervision": bool(args.require_process_supervision),
        "process_reward_warmup": bool(args.process_reward_warmup),
        "terminal_on_process_hit": bool(args.terminal_on_process_hit),
        "pointwise_action_policy": bool(args.pointwise_action_policy),
        "pointwise_action_datasets": args.pointwise_action_datasets,
        "pointwise_gradient_contract": POINTWISE_GRADIENT_CONTRACT_VERSION,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_resume_checkpoint(
    output_dir: Path, *, expected_signature: str
) -> tuple[Path | None, dict[str, Any] | None]:
    """Load the newest complete checkpoint, including a crash-safe backup slot."""
    for checkpoint_dir in (
        output_dir / "resume_checkpoint",
        output_dir / "resume_checkpoint.backup",
    ):
        state_path = checkpoint_dir / "state.json"
        adapter_path = checkpoint_dir / "adapter"
        optimizer_path = checkpoint_dir / "optimizer.pt"
        if not (state_path.is_file() and adapter_path.is_dir() and optimizer_path.is_file()):
            continue
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("schema_version") != RESUME_CHECKPOINT_VERSION:
            raise RuntimeError(f"unsupported resume checkpoint: {state_path}")
        if state.get("run_signature") != expected_signature:
            raise RuntimeError(
                "resume checkpoint does not match the current data/optimizer protocol: "
                f"{state_path}"
            )
        return checkpoint_dir, state
    return None, None


def _truncate_to_checkpoint(path: Path, expected_size: int) -> None:
    if not path.is_file():
        raise RuntimeError(f"resume log is missing: {path}")
    actual_size = path.stat().st_size
    if actual_size < expected_size:
        raise RuntimeError(
            f"resume log is shorter than its committed offset: {path} "
            f"actual={actual_size} expected={expected_size}"
        )
    with path.open("r+b") as handle:
        handle.truncate(expected_size)


def executor_backend_for_dataset(dataset: str, semantic_skill_model: str) -> str:
    """Return the fixed, dataset-appropriate answer executor backend."""
    return (
        str(semantic_skill_model)
        if str(dataset) == "video_holmes"
        else "deterministic-rule-assembly-v1"
    )


def is_trainable_reward_group(
    samples: Sequence[Mapping[str, Any]], *, process_reward_warmup: bool = False
) -> bool:
    rewards = [float(sample.get("reward") or 0.0) for sample in samples]
    has_variance = len({round(value, 8) for value in rewards}) > 1
    if not samples or not has_variance:
        return False
    if process_reward_warmup:
        return any(bool(sample.get("process_supported")) for sample in samples)
    return any(bool(sample.get("terminal_success")) for sample in samples)


def aligned_process_warmup_reward(outcome: Mapping[str, Any], *, dataset: str) -> float:
    """Reward the process-hit contract more strongly than incidental overlap.

    The ordinary terminal reward intentionally keeps process components small
    beside answer correctness.  In retrieval-only warm-up there is no answer
    component, so reusing those weights can rank a non-hit with broad segment
    overlap above the inference/relationship hit that unlocks the executor.
    """
    components = outcome.get("reward_components") or {}
    hit = 1.0 if bool(outcome.get("process_supported")) else 0.0
    if dataset == "video_holmes":
        reward = (
            0.55 * hit
            + 0.20 * float(components.get("inference_shot_recall") or 0.0)
            + 0.15 * float(components.get("relationship_support") or 0.0)
            + 0.10 * float(components.get("segment_recall") or 0.0)
        )
    elif dataset == "cg_bench":
        reward = (
            0.60 * hit
            + 0.25 * float(components.get("clue_recall") or 0.0)
            + 0.15 * float(components.get("evidence_precision") or 0.0)
        )
    else:
        reward = hit
    return max(0.0, min(1.0, float(reward)))


class RolloutTimeoutError(TimeoutError):
    pass


@contextmanager
def rollout_timeout(seconds: int):
    """Raise after a per-rollout wall-clock timeout on Unix main threads."""
    seconds = int(seconds)
    if seconds <= 0:
        yield
        return
    previous_handler = signal.getsignal(signal.SIGALRM)

    def _raise_timeout(signum, frame):  # noqa: ANN001
        raise RolloutTimeoutError(f"rollout exceeded {seconds}s")

    signal.signal(signal.SIGALRM, _raise_timeout)
    signal.setitimer(signal.ITIMER_REAL, float(seconds))
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)


def _overlaps(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    try:
        return float(left.get("start_s", 0)) < float(right.get("end_s", 0)) and float(
            right.get("start_s", 0)
        ) < float(left.get("end_s", 0))
    except (TypeError, ValueError):
        return False


def retrieval_catalog(example: Mapping[str, Any]) -> tuple[list[dict[str, Any]], str]:
    """Return the dataset-appropriate visible retrieval catalog.

    Long-video CG examples expose 30s ``coarse_clip_schemas``. Short-video
    Video-Holmes examples intentionally have no coarse catalog and expose their
    4s evidence windows as ``clip_schemas`` instead.
    """
    metadata = example.get("metadata") if isinstance(example.get("metadata"), Mapping) else {}
    coarse = metadata.get("coarse_clip_schemas")
    if isinstance(coarse, list) and coarse:
        return [row for row in coarse if isinstance(row, dict)], "coarse_clip_schemas"
    clips = metadata.get("clip_schemas")
    if isinstance(clips, list) and clips:
        return [row for row in clips if isinstance(row, dict)], "clip_schemas"
    return [], "none"


def build_retrieval_state(example: Mapping[str, Any], *, topk: int | None = None) -> dict[str, Any]:
    schemas, catalog_source = retrieval_catalog(example)
    if topk is None:
        topk = 4 if str(example.get("dataset") or "") == "video_holmes" else 2
    return {
        "schema_version": "video-skills/l2-retrieval-state-v0.3",
        "process_model": "mdp_style_l2_retrieval_controller",
        "dataset": example.get("dataset"),
        "example_id": example.get("example_id"),
        "question": compact_visibility(example.get("question") or {}),
        "l1_coarse_summary_catalog": policy_catalog(schemas),
        "retrieval_catalog_source": catalog_source,
        "partial_l1_summary": {
            "coarse_summary_count": len(schemas),
            "fine_observation_count": 0,
        },
        "budget_state": {"topk": max(1, int(topk)), "retrieval_round": 0},
    }


def retrieval_prompt(
    tokenizer: Any, example: Mapping[str, Any], *, topk: int | None = None
) -> tuple[str, dict[str, Any]]:
    state = build_retrieval_state(example, topk=topk)
    messages = [
        {"role": "system", "content": SYSTEM},
        {
            "role": "user",
            "content": json.dumps(
                {"task": "select_coarse_set", "state_t": state},
                ensure_ascii=False,
                separators=(",", ":"),
            ),
        },
    ]
    return (
        apply_chat_template_no_think(tokenizer, messages, add_generation_prompt=True, tokenize=False),
        state,
    )


def sample_pointwise_set(
    scores: Sequence[float], *, topk: int, seed: int, temperature: float,
    boundary_anchor_index0: bool = False,
) -> list[int]:
    """Sample a candidate set from pointwise log-odds with Gumbel top-k."""
    if not scores:
        return []
    width = min(max(1, int(topk)), len(scores))
    rng = random.Random(int(seed))
    scale = max(float(temperature), 1e-6)
    ranked = sorted(
        range(len(scores)),
        key=lambda index: -(
            float(scores[index]) / scale
            - math.log(-math.log(min(max(rng.random(), 1e-12), 1.0 - 1e-12)))
        ),
    )
    selected = ranked[:width]
    if boundary_anchor_index0 and width >= 2 and 0 not in selected:
        selected[-1] = 0
    return selected


def pointwise_policy_scores_tensor(
    policy: Any, tokenizer: Any, example: Mapping[str, Any], *, device: Any,
    batch_size: int = 8, requires_grad: bool = False,
    candidate_indices: Sequence[int] | None = None,
) -> Any:
    """Score the live retrieval catalog with the exact SFT/OPD prompt."""
    import torch
    import torch.nn.functional as F

    source_catalog, _ = retrieval_catalog(example)
    visible_catalog = policy_catalog(source_catalog)
    selected_candidates = (
        list(range(len(visible_catalog)))
        if candidate_indices is None
        else [int(index) for index in candidate_indices]
    )
    if len(set(selected_candidates)) != len(selected_candidates):
        raise ValueError("pointwise candidate indices must be unique")
    if selected_candidates and (
        min(selected_candidates) < 0 or max(selected_candidates) >= len(visible_catalog)
    ):
        raise ValueError("pointwise candidate index outside retrieval catalog")
    variants: list[tuple[int, bool, list[int], int]] = []
    for candidate_index in selected_candidates:
        candidate = visible_catalog[candidate_index]
        state = {
            "dataset": example.get("dataset"),
            "example_id": example.get("example_id"),
            "question": compact_visibility(example.get("question") or {}),
            "candidate_retrieval": {"rank": candidate_index + 1},
        }
        messages = [
            {"role": "system", "content": POINTWISE_SYSTEM},
            {
                "role": "user",
                "content": json.dumps(
                    {"task": POINTWISE_TASK, "state_t": pointwise_state(state, candidate)},
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
            },
        ]
        for relevant in (True, False):
            assistant = json.dumps(
                relevance_action(relevant), ensure_ascii=False, separators=(",", ":")
            )
            prompt = apply_chat_template_no_think(
                tokenizer, messages, add_generation_prompt=True, tokenize=False
            )
            full = apply_chat_template_no_think(
                tokenizer,
                messages + [{"role": "assistant", "content": assistant}],
                add_generation_prompt=False,
                tokenize=False,
            )
            prompt_ids = list(tokenizer(prompt, add_special_tokens=False)["input_ids"])
            full_ids = list(tokenizer(full, add_special_tokens=False)["input_ids"])
            if full_ids[: len(prompt_ids)] != prompt_ids:
                raise ValueError("pointwise policy prompt is not a completion prefix")
            variants.append((candidate_index, relevant, full_ids, len(prompt_ids)))

    causal_lm = policy.get_base_model()
    likelihoods: dict[tuple[int, bool], Any] = {}
    if requires_grad:
        # Transformers only applies gradient checkpointing while the model is in
        # training mode.  Keep that mode for 24GB GPUs, but disable Dropout
        # modules explicitly so policy log-probabilities remain deterministic.
        # Calling policy.eval() here silently disables checkpointing and makes
        # even a one-candidate Qwen3.5 graph exceed 24GB.
        policy.train()
        for module in policy.modules():
            if isinstance(module, torch.nn.Dropout):
                module.eval()
    else:
        policy.eval()
    policy.config.use_cache = False
    context = torch.enable_grad() if requires_grad else torch.no_grad()
    with context:
        for start in range(0, len(variants), max(1, int(batch_size))):
            batch = variants[start : start + max(1, int(batch_size))]
            width = max(len(item[2]) for item in batch)
            input_ids = torch.full(
                (len(batch), width), tokenizer.pad_token_id, dtype=torch.long, device=device
            )
            attention_mask = torch.zeros_like(input_ids)
            for position, (_, _, ids, _) in enumerate(batch):
                input_ids[position, : len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)
                attention_mask[position, : len(ids)] = 1
            hidden = causal_lm.model(
                input_ids=input_ids, attention_mask=attention_mask, use_cache=False
            ).last_hidden_state
            for position, (candidate_index, relevant, ids, prompt_length) in enumerate(batch):
                labels = input_ids[position, prompt_length : len(ids)]
                token_hidden = hidden[position, prompt_length - 1 : len(ids) - 1]
                score = token_hidden.new_zeros((), dtype=torch.float32)
                for offset in range(0, labels.numel(), 16):
                    logits = F.linear(
                        token_hidden[offset : offset + 16], causal_lm.lm_head.weight
                    )
                    score = score + (
                        F.log_softmax(logits.float(), dim=-1)
                        .gather(1, labels[offset : offset + 16, None])
                        .sum()
                    )
                likelihoods[(candidate_index, relevant)] = score
    return torch.stack([
        likelihoods[(index, True)] - likelihoods[(index, False)]
        for index in selected_candidates
    ])


def pointwise_policy_scores(
    policy: Any, tokenizer: Any, example: Mapping[str, Any], *, device: Any,
    batch_size: int = 8,
) -> list[float]:
    """Compatibility wrapper returning detached scores for rollout sampling."""
    scores = pointwise_policy_scores_tensor(
        policy, tokenizer, example, device=device, batch_size=batch_size, requires_grad=False
    )
    return [float(value) for value in scores.detach().cpu().tolist()]


def selected_indices(
    payload: dict[str, Any] | None,
    *,
    catalog_size: int,
    topk: int = 2,
    boundary_anchor_index0: bool = False,
) -> list[int]:
    if not payload:
        return []
    # The current SFT seed is a pointwise L2 relevance scorer, not a native
    # coarse-set action policy. Accept its single-candidate action shape and
    # map it into the retrieval-controller action space so GRPO can train on
    # useful terminal feedback instead of labeling format-compatible pointwise
    # outputs as invalid.
    allowed_tools = {
        "select_coarse_clips",
        "select_coarse_set",
        "select_next_coarse_clip",
        "score_coarse_candidate",
        "score_coarse_candidate_relevance",
        "choose_best_coarse_candidate",
        "choose_better_coarse_candidate",
    }
    index_keys = {
        "selected_coarse_indices",
        "selected_coarse_clips",
        "selected_indices",
        "coarse_indices",
        "candidate_indices",
        "clip_indices",
        "selected_clips",
        "clips",
        "select",
        "coarse_index",
        "candidate_index",
        "clip_index",
        "index",
    }
    tool_name = payload.get("tool_name")
    if tool_name is not None and tool_name not in allowed_tools:
        return []
    arguments = payload.get("arguments") if isinstance(payload.get("arguments"), dict) else {}
    if not arguments and any(key in payload for key in index_keys):
        arguments = payload
    if tool_name is None and not any(key in arguments for key in index_keys):
        return []
    values = None
    for key in (
        "selected_coarse_indices",
        "selected_coarse_clips",
        "selected_indices",
        "coarse_indices",
        "candidate_indices",
        "clip_indices",
        "selected_clips",
        "clips",
        "select",
    ):
        if arguments.get(key) is not None:
            values = arguments.get(key)
            break
    if values is None:
        for key in ("coarse_index", "candidate_index", "clip_index", "index"):
            if arguments.get(key) is not None:
                values = [arguments.get(key)]
                break
    if not isinstance(values, list):
        return []
    result = []
    for value in values:
        try:
            index = int(value)
        except (TypeError, ValueError):
            continue
        if 0 <= index < int(catalog_size) and index not in result:
            result.append(index)
    topk = max(1, int(topk))
    pointwise_tools = {
        "score_coarse_candidate",
        "score_coarse_candidate_relevance",
        "choose_best_coarse_candidate",
        "choose_better_coarse_candidate",
    }
    # Boundary anchoring is a compatibility bridge for a pointwise/singleton
    # seed action.  Never overwrite a native multi-index set action.
    if (
        boundary_anchor_index0 and catalog_size > 0 and topk >= 2
        and (tool_name in pointwise_tools or len(result) <= 1)
    ):
        anchored = []
        for index in [result[0] if result else 0, 0, *result[1:]]:
            if 0 <= index < int(catalog_size) and index not in anchored:
                anchored.append(index)
            if len(anchored) >= topk:
                break
        return anchored
    return result[:topk]


def expand_temporal_neighbors(
    indices: Sequence[int], *, catalog_size: int, topk: int
) -> list[int]:
    """Expand a short-video point prediction into a bounded local evidence chain."""
    result = [int(index) for index in indices if 0 <= int(index) < int(catalog_size)]
    if not result or len(result) >= max(1, int(topk)):
        return result[: max(1, int(topk))]
    center = result[0]
    distance = 1
    while len(result) < int(topk) and distance < int(catalog_size):
        for candidate in (center - distance, center + distance):
            if 0 <= candidate < int(catalog_size) and candidate not in result:
                result.append(candidate)
                if len(result) >= int(topk):
                    break
        distance += 1
    return result


def action_budget_compliant(
    payload: dict[str, Any] | None, *, catalog_size: int, topk: int
) -> bool:
    if not payload:
        return False
    raw = selected_indices(
        payload,
        catalog_size=catalog_size,
        topk=max(1, int(catalog_size)),
        boundary_anchor_index0=False,
    )
    return bool(raw) and len(raw) <= max(1, int(topk))


def _filter_graph_to_selection(
    graph: Mapping[str, Any],
    *,
    spans: Sequence[Mapping[str, Any]],
    selected_source_ids: set[str],
) -> dict[str, Any]:
    """Keep only selected-window nodes and their explicit source records."""
    filtered = copy.deepcopy(dict(graph))
    kept_nodes = []
    kept_ids: set[str] = set()
    for node in filtered.get("nodes") or []:
        if not isinstance(node, dict):
            continue
        span = node.get("time_span")
        source_ids = {str(value) for value in node.get("source_ids") or [] if value}
        keep = bool(
            node.get("node_type") == "question_requirement"
            or str(node.get("node_id") or "") in selected_source_ids
            or source_ids.intersection(selected_source_ids)
            or (isinstance(span, dict) and any(_overlaps(span, selected) for selected in spans))
        )
        if keep:
            kept_nodes.append(node)
            if node.get("node_id"):
                kept_ids.add(str(node["node_id"]))
    filtered["nodes"] = kept_nodes
    filtered["edges"] = [
        edge for edge in filtered.get("edges") or []
        if str(edge.get("src")) in kept_ids and str(edge.get("dst")) in kept_ids
    ]
    return filtered


def filter_example_for_retrieval(example: Mapping[str, Any], indices: Sequence[int]) -> tuple[dict[str, Any], dict[str, Any]]:
    isolated = copy.deepcopy(dict(example))
    metadata = isolated.get("metadata") if isinstance(isolated.get("metadata"), dict) else {}
    schemas, catalog_source = retrieval_catalog(isolated)
    spans = [
        schemas[index].get("time_span")
        for index in indices
        if 0 <= index < len(schemas) and isinstance(schemas[index], dict)
        and isinstance(schemas[index].get("time_span"), dict)
    ]
    selected = [schemas[index] for index in indices if 0 <= index < len(schemas)]
    selected_source_ids = {
        str(row.get("clip_id") or row.get("node_id") or "") for row in selected
        if row.get("clip_id") or row.get("node_id")
    }
    graph = _filter_graph_to_selection(
        metadata.get("clue_memory_graph") or {},
        spans=spans,
        selected_source_ids=selected_source_ids,
    )
    graph.setdefault("retrieval", {})["selected_coarse_indices"] = list(indices)
    graph["retrieval"]["policy"] = "local_l2_on_policy"
    graph["retrieval"]["catalog_source"] = catalog_source
    metadata["clue_memory_graph"] = graph
    coarse_fine = copy.deepcopy(metadata.get("coarse_fine_graph") or {})
    for graph_name in ("coarse_graph", "fine_graph"):
        if isinstance(coarse_fine.get(graph_name), Mapping):
            coarse_fine[graph_name] = _filter_graph_to_selection(
                coarse_fine[graph_name], spans=spans, selected_source_ids=selected_source_ids
            )
    if coarse_fine:
        coarse_fine["selected_coarse_indices"] = list(indices)
        metadata["coarse_fine_graph"] = coarse_fine
    metadata[catalog_source] = selected
    # The downstream planner historically reads coarse_clip_schemas. Mirror the
    # selected short-video windows there without mutating the source example.
    metadata["coarse_clip_schemas"] = selected
    metadata["motif_enabled"] = False
    # Cached full-video reasoning products must not bypass the selected-window boundary.
    for key in ("reasoning_rollout", "reasoning_rollout_shell", "explanation"):
        metadata.pop(key, None)
    isolated["metadata"] = metadata
    if isinstance(isolated.get("evidence_index"), Mapping):
        isolated["evidence_index"] = _filter_graph_to_selection(
            isolated["evidence_index"], spans=spans, selected_source_ids=selected_source_ids
        )
    return isolated, graph


def terminal_reward(
    rollout: Mapping[str, Any],
    gold: Mapping[str, Any],
    *,
    dataset: str = "",
    selected_entries: Sequence[Mapping[str, Any]] = (),
    supervision: Mapping[str, Any] | None = None,
    question_type: str = "",
) -> dict[str, Any]:
    final = rollout.get("final_answer") if isinstance(rollout.get("final_answer"), Mapping) else {}
    predicted = str(final.get("label") or "")
    correct = bool(predicted and predicted == str(gold.get("label") or ""))
    runtime = ((rollout.get("metadata") or {}).get("runtime_verifier") or {})
    verifier_passed = bool(runtime.get("passed"))
    status = str(rollout.get("acceptance_status") or "")
    strong = status in {"accepted_strong", "resolved_strong"}
    evidence_pack = (
        rollout.get("verified_evidence_pack")
        if isinstance(rollout.get("verified_evidence_pack"), Mapping)
        else {}
    )
    support_ref_count = int(evidence_pack.get("support_ref_count") or 0)
    min_support_refs = int(evidence_pack.get("min_support_refs") or 0)
    trace_fail = int(evidence_pack.get("trace_fail") or 0)
    metadata = rollout.get("metadata") if isinstance(rollout.get("metadata"), Mapping) else {}
    llm_plan = metadata.get("llm_plan") if isinstance(metadata.get("llm_plan"), Mapping) else {}
    query_finalizer = (
        llm_plan.get("query_memory_finalizer")
        if isinstance(llm_plan.get("query_memory_finalizer"), Mapping)
        else {}
    )
    finalizer_commit_verified = bool(query_finalizer.get("committed")) or any(
        isinstance(node, Mapping)
        and node.get("step_id") == "query_memory_commit_final"
        and node.get("skill_id") == "commit_answer"
        and str(node.get("status") or "").lower() in {"verified", "ok", "success"}
        for node in (rollout.get("nodes") or [])
    )
    repaired_minimum_verified_acceptance = bool(
        query_finalizer.get("verified")
        and finalizer_commit_verified
        and min_support_refs > 0
        and support_ref_count >= min_support_refs
    )
    minimum_verified_acceptance = bool(
        status in {"accepted_weak", "accepted_bridge"}
        and verifier_passed
        and min_support_refs > 0
        and support_ref_count >= min_support_refs
        and (trace_fail == 0 or repaired_minimum_verified_acceptance)
    )
    answer_terminal_success = bool(
        correct and verifier_passed and (strong or minimum_verified_acceptance)
    )
    spans = [row["time_span"] for row in selected_entries if isinstance(row.get("time_span"), Mapping)]
    answer_component = 1.0 if correct else (-1.0 if predicted else 0.0)
    verifier_component = (
        1.0 if verifier_passed and predicted else (-1.0 if predicted and not verifier_passed else 0.0)
    )
    commit_component = 1.0 if strong else (0.25 if status in {"accepted_weak", "accepted_bridge"} else 0.0)
    components: dict[str, float] = {
        "answer": answer_component,
        "verifier": verifier_component,
        "commit": commit_component,
    }
    process_supported = True
    supervision = supervision or {}
    if dataset == "cg_bench" and supervision.get("clue_spans"):
        clue = temporal_retrieval_metrics(spans, supervision.get("clue_spans") or [])
        components.update({
            "clue_recall": clue["recall"],
            "evidence_precision": clue["precision"],
            "clue_mean_best_iou": clue["mean_best_iou"],
        })
        process_supported = clue["recall"] > 0.0
        reward = (
            0.45 * answer_component
            + 0.20 * verifier_component
            + 0.20 * clue["recall"]
            + 0.10 * clue["precision"]
            + 0.05 * commit_component
        )
    elif dataset == "video_holmes" and supervision:
        segment = temporal_retrieval_metrics(spans, supervision.get("segment_spans") or [])
        inference = temporal_retrieval_metrics(spans, supervision.get("inference_spans") or [])
        relationship = lexical_support(selected_entries, supervision.get("relationship_texts") or [])
        components.update({
            "segment_recall": segment["recall"],
            "segment_precision": segment["precision"],
            "inference_shot_recall": inference["recall"],
            "relationship_support": relationship,
        })
        normalized_question_type = str(question_type or "").upper()
        if normalized_question_type == "SR":
            process_supported = inference["recall"] > 0.0 and relationship >= 0.25
        elif normalized_question_type in {"MHR", "IMC"}:
            process_supported = inference["recall"] > 0.0
        else:
            process_supported = inference["recall"] > 0.0 or relationship >= 0.25
        reward = (
            0.40 * answer_component
            + 0.20 * verifier_component
            + 0.15 * segment["recall"]
            + 0.15 * inference["recall"]
            + 0.05 * relationship
            + 0.05 * commit_component
        )
    elif answer_terminal_success:
        reward = 1.0
    elif correct:
        reward = 0.35
    elif predicted:
        reward = -0.25
    else:
        reward = 0.0
    success = answer_terminal_success and process_supported
    if success:
        reward = 1.0
    elif answer_terminal_success and not process_supported:
        reward = min(float(reward), 0.35)
    elif predicted and not correct:
        # A plausible evidence selection must not make a wrong strong commit
        # preferable to abstention on these answerable benchmarks.
        reward = min(float(reward), -0.25)
    reward = max(-1.0, min(1.0, float(reward)))
    return {
        "reward": reward,
        "reward_components": components,
        "reward_dataset": dataset or None,
        "terminal_success": success,
        "answer_terminal_success": answer_terminal_success,
        "minimum_verified_acceptance": minimum_verified_acceptance,
        "repaired_minimum_verified_acceptance": repaired_minimum_verified_acceptance,
        "evidence_acceptance_level": "strong" if strong else (
            "minimum_verified" if minimum_verified_acceptance else "insufficient"
        ),
        "process_supported": process_supported,
        "answer_correct": correct,
        "verifier_passed": verifier_passed,
        "acceptance_status": status,
        "predicted_label": predicted or None,
    }


def compact_rollout_diagnostic(rollout: Mapping[str, Any] | None) -> dict[str, Any]:
    """Keep enough executor evidence to diagnose terminal failures in JSONL logs."""
    rollout = rollout or {}
    metadata = rollout.get("metadata") if isinstance(rollout.get("metadata"), Mapping) else {}
    runtime = metadata.get("runtime_verifier") if isinstance(metadata.get("runtime_verifier"), Mapping) else {}
    evidence = rollout.get("verified_evidence_pack") if isinstance(rollout.get("verified_evidence_pack"), Mapping) else {}
    llm_plan = metadata.get("llm_plan") if isinstance(metadata.get("llm_plan"), Mapping) else {}
    query_finalizer = (
        llm_plan.get("query_memory_finalizer")
        if isinstance(llm_plan.get("query_memory_finalizer"), Mapping)
        else {}
    )
    return {
        "failure_reasons": list(rollout.get("failure_reasons") or runtime.get("failure_reasons") or []),
        "failed_skill_ids": list(metadata.get("failed_skill_ids") or []),
        "failed_skill_codes": list(metadata.get("failed_skill_codes") or []),
        "support_ref_count": int(evidence.get("support_ref_count") or 0),
        "min_support_refs": int(evidence.get("min_support_refs") or 0),
        "trace_ok": int(metadata.get("llm_trace_ok") or 0),
        "trace_fail": int(metadata.get("llm_trace_fail") or 0),
        "query_memory_finalizer": dict(query_finalizer),
    }


def compact_executor_trace(rollout: Mapping[str, Any] | None) -> dict[str, Any]:
    """Persist the plan and failed execution steps without duplicating the clue graph."""
    rollout = rollout or {}
    metadata = rollout.get("metadata") if isinstance(rollout.get("metadata"), Mapping) else {}
    skill_nodes = []
    for node in rollout.get("nodes") or []:
        if not isinstance(node, Mapping) or not node.get("skill_id"):
            continue
        skill_nodes.append({
            "skill_id": node.get("skill_id"),
            "step_id": node.get("step_id"),
            "status": node.get("status"),
            "failure_code": node.get("failure_code"),
            "evidence_refs": list(node.get("evidence_refs") or []),
        })
    return {
        "final_answer": rollout.get("final_answer") or {},
        "acceptance_status": rollout.get("acceptance_status"),
        "failure_reasons": list(rollout.get("failure_reasons") or []),
        "verified_evidence_pack": rollout.get("verified_evidence_pack") or {},
        "llm_plan": metadata.get("llm_plan") or {},
        "failed_skill_ids": list(metadata.get("failed_skill_ids") or []),
        "failed_skill_codes": list(metadata.get("failed_skill_codes") or []),
        "skill_trace": skill_nodes,
    }


def executor_cache_key(
    *,
    example: Mapping[str, Any],
    indices: Sequence[int],
    graph: Mapping[str, Any],
    planner_model: str,
    skill_model: str,
) -> str:
    payload = {
        "example_id": example.get("example_id"),
        "selected_indices": [int(value) for value in indices],
        "question": compact_visibility(example.get("question") or {}),
        "graph": graph,
        "planner_model": planner_model,
        "skill_model": skill_model,
        "action_contract": ACTION_CONTRACT_VERSION,
        "isolation_contract": EXECUTOR_ISOLATION_VERSION,
        "fallback_contract": EXECUTOR_FALLBACK_VERSION,
        "cache_contract": EXECUTOR_CACHE_VERSION,
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def cached_executor_rollout(
    *,
    cache_dir: Path,
    key: str,
    build: Callable[[], dict[str, Any]],
) -> tuple[dict[str, Any], bool]:
    """Share one fixed executor result across matched policy runs."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{key}.json"
    lock_path = cache_dir / f"{key}.lock"
    with lock_path.open("a+", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        if cache_path.exists():
            return json.loads(cache_path.read_text(encoding="utf-8")), True
        rollout = build()
        temporary = cache_dir / f"{key}.tmp"
        temporary.write_text(
            json.dumps(rollout, ensure_ascii=False, default=str), encoding="utf-8"
        )
        temporary.replace(cache_path)
        return rollout, False


def read_example_id_allowlist(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    if not path.exists():
        raise FileNotFoundError(path)
    allowed: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        value = line.strip()
        if not value or value.startswith("#"):
            continue
        allowed.add(value.split("\t", 1)[0].strip())
    if not allowed:
        raise ValueError(f"empty example id allowlist: {path}")
    return allowed


def read_example_id_order(path: Path | None) -> list[str]:
    if path is None:
        return []
    return [
        line.strip().split("\t", 1)[0].strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def read_exact_group_allowlist(path: Path | None) -> list[tuple[str, int]] | None:
    """Read an exact ``example_id<TAB>repeat_index`` mined-group allowlist.

    Plain example-id allowlists remain supported.  Mixed formats are rejected:
    silently dropping a repeat index would break the rollout-seed provenance
    established by reward-variance mining.
    """
    if path is None:
        return None
    values = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    has_tabs = ["\t" in value for value in values]
    if not any(has_tabs):
        return None
    if not all(has_tabs):
        raise ValueError("cannot mix example-id and exact-group allowlist rows")
    groups: list[tuple[str, int]] = []
    seen: set[tuple[str, int]] = set()
    for value in values:
        example_id, repeat_text = value.rsplit("\t", 1)
        example_id = example_id.strip()
        try:
            repeat_index = int(repeat_text.strip())
        except ValueError as error:
            raise ValueError(f"invalid repeat index in exact-group allowlist: {value!r}") from error
        if not example_id or repeat_index < 0:
            raise ValueError(f"invalid exact-group allowlist row: {value!r}")
        key = (example_id, repeat_index)
        if key in seen:
            raise ValueError(f"duplicate exact group in allowlist: {key}")
        seen.add(key)
        groups.append(key)
    return groups


def exact_group_pool(
    examples: Sequence[Mapping[str, Any]], groups: Sequence[tuple[str, int]]
) -> list[dict[str, Any]]:
    by_id = {str(example.get("example_id") or ""): example for example in examples}
    missing = sorted({example_id for example_id, _ in groups if example_id not in by_id})
    if missing:
        raise ValueError(f"exact-group allowlist references missing examples: {missing[:5]}")
    output = []
    for example_id, repeat_index in groups:
        row = dict(by_id[example_id])
        row["_grpo_repeat_index"] = int(repeat_index)
        output.append(row)
    return output


def filtered_grpo_pool(
    examples: Sequence[Mapping[str, Any]],
    *,
    datasets: set[str] | None = None,
    example_id_allowlist: set[str] | None = None,
    min_catalog_size: int = 1,
    max_catalog_size: int | None = None,
) -> list[dict[str, Any]]:
    pool: list[dict[str, Any]] = []
    for example in examples:
        example_id = str(example.get("example_id") or "")
        dataset = str(example.get("dataset") or "")
        if datasets is not None and dataset not in datasets:
            continue
        if example_id_allowlist is not None and example_id not in example_id_allowlist:
            continue
        schemas, _ = retrieval_catalog(example)
        catalog_size = len(schemas)
        if catalog_size < int(min_catalog_size):
            continue
        if max_catalog_size is not None and catalog_size > int(max_catalog_size):
            continue
        pool.append(dict(example))
    return pool


def dataset_balanced_order(examples: Sequence[Mapping[str, Any]], *, seed: int) -> list[dict[str, Any]]:
    """Round-robin datasets after deterministic within-dataset shuffling."""
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for example in examples:
        buckets[str(example.get("dataset") or "unknown")].append(dict(example))
    rng = random.Random(seed)
    for rows in buckets.values():
        rng.shuffle(rows)
    ordered: list[dict[str, Any]] = []
    names = sorted(buckets)
    while any(buckets.values()):
        for name in names:
            if buckets[name]:
                ordered.append(buckets[name].pop())
    return ordered


def process_supervised_pool(
    examples: Sequence[Mapping[str, Any]], supervision_index: Mapping[str, Mapping[str, Any]]
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for example in examples:
        supervision = supervision_index.get(supervision_key(example)) or {}
        dataset = str(example.get("dataset") or "")
        if dataset == "cg_bench":
            usable = bool(supervision.get("clue_spans"))
        elif dataset == "video_holmes":
            question_type = str((example.get("question") or {}).get("question_type") or "SR").upper()
            usable = bool(supervision.get("inference_spans"))
            if question_type not in {"MHR", "IMC"}:
                usable = usable and bool(supervision.get("relationship_texts"))
        else:
            usable = False
        if usable:
            output.append(dict(example))
    return output


def repeat_grpo_pool(
    examples: Sequence[Mapping[str, Any]], *, repeats_per_example: int, repeat_start_index: int = 0
) -> list[dict[str, Any]]:
    output = []
    # Traverse the full (possibly dataset-balanced) pool once per repeat so
    # max-groups truncation does not destroy its ordering.
    start = max(0, int(repeat_start_index))
    for repeat_index in range(start, start + max(1, int(repeats_per_example))):
        for example in examples:
            row = dict(example)
            row["_grpo_repeat_index"] = repeat_index
            output.append(row)
    return output


def balanced_repeat_grpo_pool(
    examples: Sequence[Mapping[str, Any]],
    *,
    repeats_per_example: int,
    repeat_start_index: int = 0,
) -> list[dict[str, Any]]:
    """Build an equal-size, round-robin group stream for every dataset.

    A plain round-robin only balances an early prefix; once the smaller bucket is
    exhausted the larger dataset dominates again.  Here every dataset contributes
    ``max_bucket_size * repeats_per_example`` groups.  Smaller buckets cycle through
    their examples and advance each example's repeat index, so the stable rollout
    seed remains distinct instead of silently duplicating a group.
    """
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for example in examples:
        buckets[str(example.get("dataset") or "unknown")].append(dict(example))
    if not buckets:
        return []
    names = sorted(buckets)
    target_per_dataset = max(len(rows) for rows in buckets.values()) * max(
        1, int(repeats_per_example)
    )
    start = max(0, int(repeat_start_index))
    output: list[dict[str, Any]] = []
    for position in range(target_per_dataset):
        for name in names:
            rows = buckets[name]
            row = dict(rows[position % len(rows)])
            row["_grpo_repeat_index"] = start + position // len(rows)
            output.append(row)
    return output


def stable_rollout_seed(
    base_seed: int, *, example_id: str, repeat_index: int, sample_index: int
) -> int:
    payload = f"{int(base_seed)}|{example_id}|{int(repeat_index)}|{int(sample_index)}".encode()
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "big") % (2**31 - 1)


def parse_dataset_adapters(values: Sequence[str]) -> dict[str, Path]:
    output: dict[str, Path] = {}
    for value in values:
        dataset, separator, path = str(value).partition("=")
        dataset, path = dataset.strip(), path.strip()
        if not separator or not dataset or not path:
            raise ValueError(f"dataset adapter must use DATASET=PATH, got {value!r}")
        if dataset in output:
            raise ValueError(f"duplicate dataset adapter route: {dataset}")
        output[dataset] = Path(path)
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument(
        "--dataset-adapter",
        action="append",
        default=[],
        help=(
            "Optional DATASET=ADAPTER route. This keeps SFT for high-resource datasets "
            "while activating OPD calibration only for a low-sample dataset."
        ),
    )
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument(
        "--frozen-l1-glob",
        action="append",
        required=True,
        help="Repeat to union multiple frozen-L1 trees; examples are deduplicated by example_id.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/fs/gamma-projects/vlm-robot/datasets"),
        help="Evaluator-only benchmark annotations; never inserted into policy prompts.",
    )
    parser.add_argument("--keys-py", default="/fs/gamma-projects/vlm-robot/keys.py")
    parser.add_argument("--planner-model", default="openai/gpt-oss-120b")
    parser.add_argument("--skill-model", default="qwen/qwen3.5-9b")
    parser.add_argument("--planner-timeout-s", type=int, default=180)
    parser.add_argument("--skill-timeout-s", type=int, default=90)
    parser.add_argument(
        "--skip-skill-model-probe",
        action="store_true",
        help="Skip the one-call startup check that the skill model returns content.",
    )
    parser.add_argument(
        "--executor-cache-dir",
        type=Path,
        help="Shared locked cache making the fixed executor identical for matching actions.",
    )
    parser.add_argument("--max-groups", type=int, default=4)
    parser.add_argument("--repeats-per-example", type=int, default=1)
    parser.add_argument("--repeat-start-index", type=int, default=0)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--cg-topk", type=int, default=2)
    parser.add_argument("--video-holmes-topk", type=int, default=4)
    parser.add_argument("--ppo-epochs", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument(
        "--pointwise-temperature",
        type=float,
        help="Gumbel top-k temperature for pointwise-routed datasets (defaults to --temperature).",
    )
    parser.add_argument(
        "--pointwise-train-batch-size",
        type=int,
        default=1,
        help=(
            "Micro-batch size for differentiable pointwise candidate scoring. "
            "Keep this at 1 on 24GB GPUs; rollout/reference scoring remains batched."
        ),
    )
    parser.add_argument(
        "--checkpoint-every-groups",
        type=int,
        default=10,
        help=(
            "For training runs, atomically save policy/optimizer/statistics at this "
            "group interval and resume automatically after Slurm requeue. Set 0 to disable."
        ),
    )
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument(
        "--generation-timeout-s",
        type=float,
        default=90.0,
        help="Best-effort wall-clock cap passed to transformers.generate(max_time=...).",
    )
    parser.add_argument(
        "--rollout-timeout-s",
        type=int,
        default=240,
        help="Per sampled terminal rollout timeout; timed-out samples receive a negative reward.",
    )
    parser.add_argument("--learning-rate", type=float, default=2e-6)
    parser.add_argument("--kl-coef", type=float, default=0.05)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-sdpa-fallback", action="store_true")
    parser.add_argument("--boundary-anchor-index0", action="store_true")
    parser.add_argument(
        "--datasets",
        default="",
        help="Optional comma-separated dataset allowlist after split filtering, e.g. cg_bench.",
    )
    parser.add_argument(
        "--split-role",
        choices=("sft_seed", "opd_pool", "grpo_pool", "dev_tune", "heldout_test"),
        default="grpo_pool",
        help="Frozen video-level split to evaluate. Training is restricted to grpo_pool.",
    )
    parser.add_argument(
        "--example-id-allowlist",
        type=Path,
        help="Optional newline-delimited example_id allowlist after split filtering.",
    )
    parser.add_argument(
        "--preserve-allowlist-order",
        action="store_true",
        help="When --example-id-allowlist is set, order the pool by that file and skip shuffling.",
    )
    parser.add_argument(
        "--dataset-balanced-sampling",
        action="store_true",
        help=(
            "Emit equal numbers of groups per dataset via round-robin cyclic repeats "
            "before max-groups truncation."
        ),
    )
    parser.add_argument(
        "--require-process-supervision",
        action="store_true",
        help="Exclude examples without evaluator-only clue/segment/inference supervision.",
    )
    parser.add_argument("--min-catalog-size", type=int, default=1)
    parser.add_argument("--max-catalog-size", type=int)
    parser.add_argument(
        "--eval-only", action="store_true",
        help="Run the fixed terminal executor without optimizer updates or adapter output.",
    )
    parser.add_argument(
        "--retrieval-only", action="store_true",
        help="Mine policy groups with evaluator-side process rewards; skip remote terminal execution.",
    )
    parser.add_argument(
        "--process-reward-warmup", action="store_true",
        help=(
            "Train retrieval actions on evaluator-side process reward variance in grpo_pool only; "
            "does not use answer labels or count as terminal GRPO."
        ),
    )
    parser.add_argument(
        "--terminal-on-process-hit", action="store_true",
        help="Run the remote executor only when the sampled action hits evaluator-side process evidence.",
    )
    parser.add_argument(
        "--pointwise-action-policy",
        action="store_true",
        help=(
            "Use the exact SFT/OPD pointwise prompt to score candidates, then sample "
            "sets with Gumbel top-k and optimize their Plackett-Luce set probability."
        ),
    )
    parser.add_argument(
        "--pointwise-action-datasets",
        default="video_holmes",
        help=(
            "Comma-separated datasets routed through the pointwise action policy. "
            "The default keeps the higher-resource CG direct set policy and routes "
            "only low-sample Video-Holmes through OPD pointwise scores."
        ),
    )
    args = parser.parse_args(argv)
    if args.retrieval_only and not args.eval_only:
        parser.error("--retrieval-only requires --eval-only")
    if args.retrieval_only and args.terminal_on_process_hit:
        parser.error("--retrieval-only and --terminal-on-process-hit are mutually exclusive")
    if args.process_reward_warmup and args.eval_only:
        parser.error("--process-reward-warmup is a training mode and cannot use --eval-only")
    if args.process_reward_warmup and (args.retrieval_only or args.terminal_on_process_hit):
        parser.error("--process-reward-warmup is mutually exclusive with retrieval-only/terminal execution")
    if not args.eval_only and args.split_role != "grpo_pool":
        parser.error("optimizer updates are restricted to --split-role grpo_pool")
    try:
        dataset_adapter_paths = parse_dataset_adapters(args.dataset_adapter)
    except ValueError as error:
        parser.error(str(error))
    pointwise_action_datasets = {
        value.strip()
        for value in str(args.pointwise_action_datasets or "").split(",")
        if value.strip()
    }
    if args.pointwise_action_policy and not pointwise_action_datasets:
        parser.error("--pointwise-action-policy requires a non-empty dataset route")
    pointwise_temperature = (
        float(args.pointwise_temperature)
        if args.pointwise_temperature is not None
        else float(args.temperature)
    )
    if pointwise_temperature <= 0:
        parser.error("--pointwise-temperature must be positive")
    if int(args.pointwise_train_batch_size) <= 0:
        parser.error("--pointwise-train-batch-size must be positive")
    if int(args.checkpoint_every_groups) < 0:
        parser.error("--checkpoint-every-groups must be non-negative")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    resume_signature = _resume_run_signature(
        args,
        dataset_adapter_paths=dataset_adapter_paths,
        pointwise_temperature=pointwise_temperature,
    )
    resume_checkpoint_dir = None
    resume_state = None
    if not args.eval_only and int(args.checkpoint_every_groups) > 0:
        resume_checkpoint_dir, resume_state = _load_resume_checkpoint(
            args.output_dir, expected_signature=resume_signature
        )

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from trainer.grpo.attn_utils import resolve_attn_implementation
    from trainer.grpo.model_runtime import _disable_torchao_peft_probes

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    _disable_torchao_peft_probes()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    attn = resolve_attn_implementation(
        "flash_attention_2", allow_sdpa_fallback=bool(args.allow_sdpa_fallback)
    )
    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    def base() -> Any:
        return AutoModelForCausalLM.from_pretrained(
            args.model, local_files_only=True, dtype=torch.bfloat16, attn_implementation=attn
        )

    policy_adapter_path = (
        resume_checkpoint_dir / "adapter" if resume_checkpoint_dir else args.adapter
    )
    policy = PeftModel.from_pretrained(
        base(), policy_adapter_path, is_trainable=not args.eval_only
    ).to(device)
    dataset_adapter_names: dict[str, str] = {}
    for route_index, (dataset, adapter_path) in enumerate(sorted(dataset_adapter_paths.items())):
        adapter_name = f"dataset_route_{route_index}_{dataset}"
        policy_route_path = (
            policy_adapter_path / adapter_name if resume_checkpoint_dir else adapter_path
        )
        policy.load_adapter(
            str(policy_route_path), adapter_name=adapter_name, is_trainable=not args.eval_only
        )
        dataset_adapter_names[dataset] = adapter_name
    reference_adapter_names: dict[str, str] = {}
    optimizer = None
    if not args.eval_only:
        # Policy and reference share immutable base weights.  Keeping frozen
        # reference LoRAs on the same base is exactly equivalent to a second
        # PeftModel for log-prob evaluation, while avoiding another ~18 GB copy
        # of the 9B base model.
        policy.load_adapter(
            str(args.adapter), adapter_name="reference_default", is_trainable=False
        )
        reference_adapter_names["default"] = "reference_default"
        for route_index, (dataset, adapter_path) in enumerate(sorted(dataset_adapter_paths.items())):
            reference_name = f"reference_dataset_route_{route_index}_{dataset}"
            policy.load_adapter(
                str(adapter_path), adapter_name=reference_name, is_trainable=False
            )
            reference_adapter_names[dataset] = reference_name
        policy.set_adapter("default")
        policy.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        policy.enable_input_require_grads()
        policy.config.use_cache = False
        # PEFT toggles requires_grad when set_adapter() changes the active route.
        # Register every LoRA tensor up front so both the CG default and the VH
        # OPD route remain optimizable during dataset-balanced training.
        trainable_adapter_parameters = [
            parameter
            for name, parameter in policy.named_parameters()
            if "lora_" in name and ".reference_" not in name
        ]
        if not trainable_adapter_parameters:
            raise RuntimeError("no LoRA parameters found for optimizer")
        optimizer = torch.optim.AdamW(trainable_adapter_parameters, lr=float(args.learning_rate))
        if resume_checkpoint_dir is not None:
            optimizer.load_state_dict(
                torch.load(
                    resume_checkpoint_dir / "optimizer.pt",
                    map_location=device,
                    weights_only=False,
                )
            )

    paths: list[Path] = []
    for pattern in args.frozen_l1_glob:
        paths.extend(Path(path) for path in sorted(glob.glob(pattern, recursive=True)))
    examples = load_frozen_l1_examples(paths)
    deduped = {
        str(example.get("example_id") or ""): example
        for example in examples
        if example.get("example_id")
    }
    examples = list(deduped.values())
    manifest = load_split_manifest(args.split_manifest)
    pool = filter_examples_by_role(
        examples, manifest=manifest, role=args.split_role, strict=False
    )
    assert_role_exclusive(pool, manifest=manifest, allowed_roles=(args.split_role,))
    datasets = {
        value.strip()
        for value in str(args.datasets or "").split(",")
        if value.strip()
    } or None
    example_id_allowlist = read_example_id_allowlist(args.example_id_allowlist)
    example_id_order = read_example_id_order(args.example_id_allowlist)
    exact_groups = read_exact_group_allowlist(args.example_id_allowlist)
    pre_filter_pool_size = len(pool)
    pool = filtered_grpo_pool(
        pool,
        datasets=datasets,
        example_id_allowlist=example_id_allowlist,
        min_catalog_size=int(args.min_catalog_size),
        max_catalog_size=args.max_catalog_size,
    )
    pre_supervision_pool_size = len(pool)
    reward_supervision = load_dataset_reward_supervision(args.dataset_root)
    if args.require_process_supervision:
        pool = process_supervised_pool(pool, reward_supervision)
    if not pool:
        raise RuntimeError(
            "empty GRPO pool after filters "
            f"datasets={sorted(datasets) if datasets else None} "
            f"allowlist={args.example_id_allowlist} "
            f"min_catalog_size={args.min_catalog_size} max_catalog_size={args.max_catalog_size}"
        )
    if exact_groups is not None:
        pool = exact_group_pool(pool, exact_groups)
    elif args.preserve_allowlist_order and example_id_order:
        rank = {example_id: index for index, example_id in enumerate(example_id_order)}
        pool.sort(key=lambda row: rank.get(str(row.get("example_id") or ""), len(rank)))
    elif args.dataset_balanced_sampling:
        pool = dataset_balanced_order(pool, seed=args.seed)
    else:
        random.Random(args.seed).shuffle(pool)
    unique_pool_examples_before_repeats = len(
        {str(row.get("example_id") or "") for row in pool}
    )
    if exact_groups is not None:
        if args.dataset_balanced_sampling and not args.preserve_allowlist_order:
            pool = dataset_balanced_order(pool, seed=args.seed)
    elif args.dataset_balanced_sampling:
        pool = balanced_repeat_grpo_pool(
            pool,
            repeats_per_example=args.repeats_per_example,
            repeat_start_index=args.repeat_start_index,
        )
    else:
        pool = repeat_grpo_pool(
            pool,
            repeats_per_example=args.repeats_per_example,
            repeat_start_index=args.repeat_start_index,
        )
    pool = pool[: max(1, int(args.max_groups))]

    planner = None
    executor = None
    if not args.retrieval_only and not args.process_reward_warmup:
        key = load_openrouter_api_key(keys_py_path=args.keys_py)
        planner = OpenRouterClient(
            model=args.planner_model,
            api_key=key,
            max_tokens=1800,
            temperature=0.0,
            reasoning={"effort": "minimal", "exclude": True},
            timeout_s=int(args.planner_timeout_s),
        )
        skill_client = SkillModelClient(
            model=args.skill_model,
            api_key=key,
            max_tokens=768,
            temperature=0.0,
            timeout_s=int(args.skill_timeout_s),
        )
        if not args.skip_skill_model_probe:
            problem = probe_skill_model(skill_client)
            if problem:
                raise RuntimeError(
                    f"refusing to start: {problem}. Pick a working --skill-model "
                    "(openai/gpt-oss-120b or qwen/qwen3-30b-a3b-instruct-2507 have been verified) "
                    "or pass --skip-skill-model-probe."
                )
        executor = SkillExecutor(
            llm_client=skill_client,
            vlm_client=None,
            config=_grpo_skill_backend_config(),
        )

    metrics_path = args.output_dir / "terminal_metrics.jsonl"
    sample_events_path = args.output_dir / "terminal_samples.jsonl"
    executor_traces_path = args.output_dir / "terminal_executor_traces.jsonl"
    log_paths = {
        "metrics": metrics_path,
        "samples": sample_events_path,
        "executor_traces": executor_traces_path,
    }
    if resume_state is None:
        for path in log_paths.values():
            path.write_text("", encoding="utf-8")
    else:
        committed_offsets = resume_state.get("log_offsets") or {}
        for name, path in log_paths.items():
            _truncate_to_checkpoint(path, int(committed_offsets[name]))
    started = time.time()
    previous_elapsed_s = float((resume_state or {}).get("elapsed_s") or 0.0)
    resume_next_group_index = int((resume_state or {}).get("next_group_index") or 0)
    trained = int((resume_state or {}).get("trained") or 0)
    skipped = int((resume_state or {}).get("skipped") or 0)
    skipped_no_process_hit = int((resume_state or {}).get("skipped_no_process_hit") or 0)
    skipped_no_terminal_success = int(
        (resume_state or {}).get("skipped_no_terminal_success") or 0
    )
    total_samples = int((resume_state or {}).get("total_samples") or 0)
    terminal_successes = int((resume_state or {}).get("terminal_successes") or 0)
    executor_cache_hits = int((resume_state or {}).get("executor_cache_hits") or 0)
    executor_cache_misses = int((resume_state or {}).get("executor_cache_misses") or 0)
    oom_offload_retries = int((resume_state or {}).get("oom_offload_retries") or 0)
    kl_values: list[float] = [
        float(value) for value in (resume_state or {}).get("kl_values", [])
    ]
    dataset_stats: dict[str, dict[str, int]] = defaultdict(
        lambda: {
            "groups_seen": 0,
            "groups_trainable": 0,
            "groups_reward_variance": 0,
            "groups_process_hit": 0,
            "groups_trained": 0,
            "samples": 0,
            "terminal_successes": 0,
            "valid_retrieval_actions": 0,
            "answer_correct": 0,
            "verifier_passed": 0,
            "format_compliant_actions": 0,
        }
    )
    dataset_status_counts: dict[str, Counter[str]] = defaultdict(Counter)
    dataset_component_sums: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for dataset, values in ((resume_state or {}).get("dataset_stats") or {}).items():
        dataset_stats[str(dataset)].update(
            {str(name): int(value) for name, value in values.items()}
        )
    for dataset, values in ((resume_state or {}).get("dataset_status_counts") or {}).items():
        dataset_status_counts[str(dataset)].update(
            {str(name): int(value) for name, value in values.items()}
        )
    for dataset, values in ((resume_state or {}).get("dataset_component_sums") or {}).items():
        dataset_component_sums[str(dataset)].update(
            {str(name): float(value) for name, value in values.items()}
        )

    def checkpoint_after_group(group_index: int) -> None:
        interval = int(args.checkpoint_every_groups)
        if args.eval_only or interval <= 0 or (group_index + 1) % interval != 0:
            return
        assert optimizer is not None
        checkpoint_root = args.output_dir / "resume_checkpoint"
        checkpoint_backup = args.output_dir / "resume_checkpoint.backup"
        checkpoint_temp = args.output_dir / f"resume_checkpoint.tmp.{os.getpid()}"
        if checkpoint_temp.exists():
            shutil.rmtree(checkpoint_temp)
        checkpoint_temp.mkdir(parents=True)
        policy.save_pretrained(
            checkpoint_temp / "adapter",
            selected_adapters=["default", *sorted(dataset_adapter_names.values())],
        )
        torch.save(optimizer.state_dict(), checkpoint_temp / "optimizer.pt")
        checkpoint_state = {
            "schema_version": RESUME_CHECKPOINT_VERSION,
            "run_signature": resume_signature,
            "next_group_index": group_index + 1,
            "trained": trained,
            "skipped": skipped,
            "skipped_no_process_hit": skipped_no_process_hit,
            "skipped_no_terminal_success": skipped_no_terminal_success,
            "total_samples": total_samples,
            "terminal_successes": terminal_successes,
            "executor_cache_hits": executor_cache_hits,
            "executor_cache_misses": executor_cache_misses,
            "oom_offload_retries": oom_offload_retries,
            "kl_values": kl_values,
            "dataset_stats": {name: dict(values) for name, values in dataset_stats.items()},
            "dataset_status_counts": {
                name: dict(values) for name, values in dataset_status_counts.items()
            },
            "dataset_component_sums": {
                name: dict(values) for name, values in dataset_component_sums.items()
            },
            "log_offsets": {
                name: path.stat().st_size for name, path in log_paths.items()
            },
            "elapsed_s": previous_elapsed_s + (time.time() - started),
        }
        (checkpoint_temp / "state.json").write_text(
            json.dumps(checkpoint_state, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if checkpoint_backup.exists():
            shutil.rmtree(checkpoint_backup)
        if checkpoint_root.exists():
            os.replace(checkpoint_root, checkpoint_backup)
        os.replace(checkpoint_temp, checkpoint_root)
        if checkpoint_backup.exists():
            shutil.rmtree(checkpoint_backup)
        print(
            json.dumps(
                {
                    "event": "terminal_grpo_checkpoint",
                    "next_group_index": group_index + 1,
                    "checkpoint": str(checkpoint_root),
                }
            ),
            flush=True,
        )

    for group_index, example in enumerate(pool):
        if group_index < resume_next_group_index:
            continue
        dataset = str(example.get("dataset") or "unknown")
        train_adapter_name = dataset_adapter_names.get(dataset, "default")
        reference_adapter_name = reference_adapter_names.get(dataset, "reference_default")
        policy.set_adapter(train_adapter_name)
        dataset_stats[dataset]["groups_seen"] += 1
        prompt, state = retrieval_prompt(
            tokenizer,
            example,
            topk=(args.video_holmes_topk if dataset == "video_holmes" else args.cg_topk),
        )
        encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        prompt_ids = encoded["input_ids"].to(device)
        prompt_len = int(prompt_ids.shape[1])
        samples = []
        policy.eval()
        policy.config.use_cache = True
        live_pointwise_scores_tensor = (
            pointwise_policy_scores_tensor(
                policy, tokenizer, example, device=device, requires_grad=False
            ).detach()
            if args.pointwise_action_policy and dataset in pointwise_action_datasets
            else None
        )
        live_pointwise_scores = (
            [float(value) for value in live_pointwise_scores_tensor.cpu().tolist()]
            if live_pointwise_scores_tensor is not None else None
        )
        reference_pointwise_scores_tensor = None
        if live_pointwise_scores_tensor is not None and not args.eval_only:
            policy.set_adapter(reference_adapter_name)
            reference_pointwise_scores_tensor = pointwise_policy_scores_tensor(
                policy, tokenizer, example, device=device, requires_grad=False
            ).detach()
            policy.set_adapter(train_adapter_name)
        for sample_index in range(int(args.k)):
            rollout_seed = stable_rollout_seed(
                args.seed,
                example_id=str(example.get("example_id") or ""),
                repeat_index=int(example.get("_grpo_repeat_index") or 0),
                sample_index=sample_index,
            )
            action_topk = int((state.get("budget_state") or {}).get("topk") or 2)
            if live_pointwise_scores is not None:
                sampled_ordered_indices = sample_pointwise_set(
                    live_pointwise_scores,
                    topk=action_topk,
                    seed=rollout_seed,
                    temperature=pointwise_temperature,
                    boundary_anchor_index0=(
                        bool(args.boundary_anchor_index0) and dataset == "cg_bench"
                    ),
                )
                indices = list(sampled_ordered_indices)
                payload = {
                    "schema_version": "video-skills/l2-retrieval-action-v0.1",
                    "tool_name": "select_coarse_clips",
                    "arguments": {"coarse_indices": indices},
                }
                completion = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
                completion_ids = tokenizer(
                    completion, add_special_tokens=False
                )["input_ids"]
                full_ids = torch.tensor(
                    [prompt_ids[0].tolist() + list(completion_ids)],
                    dtype=torch.long,
                    device=device,
                )
                format_compliant = True
            else:
                torch.manual_seed(rollout_seed)
                with torch.no_grad():
                    sequence = policy.generate(
                        input_ids=prompt_ids,
                        attention_mask=encoded["attention_mask"].to(device),
                        max_new_tokens=args.max_new_tokens,
                        max_time=float(args.generation_timeout_s),
                        do_sample=True,
                        temperature=args.temperature,
                        top_p=0.95,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )[0]
                completion_ids = _trim_generated(
                    sequence[prompt_len:].tolist(), tokenizer.eos_token_id, tokenizer.pad_token_id
                )
                if not completion_ids:
                    continue
                full_ids = torch.tensor(
                    [prompt_ids[0].tolist() + completion_ids], dtype=torch.long, device=device
                )
                completion = strip_think_tags(
                    tokenizer.decode(completion_ids, skip_special_tokens=True)
                )
                payload = extract_json_object(completion)
                format_compliant = action_budget_compliant(
                    payload,
                    catalog_size=len(state["l1_coarse_summary_catalog"]),
                    topk=action_topk,
                )
                indices = selected_indices(
                    payload,
                    catalog_size=len(state["l1_coarse_summary_catalog"]),
                    topk=action_topk,
                    boundary_anchor_index0=bool(args.boundary_anchor_index0),
                )
            if dataset == "video_holmes":
                indices = expand_temporal_neighbors(
                    indices,
                    catalog_size=len(state["l1_coarse_summary_catalog"]),
                    topk=int((state.get("budget_state") or {}).get("topk") or 4),
                )
            if indices:
                isolated, graph = filter_example_for_retrieval(example, indices)
                source_catalog, _ = retrieval_catalog(example)
                selected_entries = [source_catalog[index] for index in indices if 0 <= index < len(source_catalog)]
                process_outcome = terminal_reward(
                    {
                        "final_answer": {},
                        "acceptance_status": "retrieval_only",
                        "metadata": {"runtime_verifier": {"passed": False}},
                    },
                    (example.get("question") or {}).get("answer") or {},
                    dataset=dataset,
                    selected_entries=selected_entries,
                    supervision=reward_supervision.get(supervision_key(example)),
                    question_type=str((example.get("question") or {}).get("question_type") or ""),
                )
                should_screen_out = bool(
                    args.terminal_on_process_hit
                    and not process_outcome.get("process_supported")
                )
                if args.retrieval_only or args.process_reward_warmup or should_screen_out:
                    outcome = process_outcome
                    if args.process_reward_warmup:
                        outcome["reward"] = aligned_process_warmup_reward(
                            outcome, dataset=dataset
                        )
                        if isinstance(outcome.get("reward_components"), dict):
                            outcome["reward_components"]["process_hit_bonus"] = float(
                                bool(outcome.get("process_supported"))
                            )
                    outcome["acceptance_status"] = (
                        "process_reward_warmup"
                        if args.process_reward_warmup
                        else "retrieval_only" if args.retrieval_only else "process_screened_out"
                    )
                else:
                    assert planner is not None and executor is not None
                    try:
                        def build_rollout() -> dict[str, Any]:
                            with rollout_timeout(int(args.rollout_timeout_s)):
                                return build_llm_reasoning_rollout(
                                    isolated,
                                    graph,
                                    client=planner,
                                    # CG multiple-choice clue grounding is
                                    # reliably handled by the deterministic
                                    # evidence assembler; VH implicit social /
                                    # causal options require semantic scoring.
                                    skill_executor=(executor if dataset == "video_holmes" else None),
                                    motif_enabled=False,
                                )

                        cache_hit = False
                        cache_key = None
                        if args.executor_cache_dir:
                            cache_key = executor_cache_key(
                                example=isolated,
                                indices=indices,
                                graph=graph,
                                planner_model=args.planner_model,
                                skill_model=executor_backend_for_dataset(dataset, args.skill_model),
                            )
                            rollout, cache_hit = cached_executor_rollout(
                                cache_dir=args.executor_cache_dir,
                                key=cache_key,
                                build=build_rollout,
                            )
                            executor_cache_hits += int(cache_hit)
                            executor_cache_misses += int(not cache_hit)
                        else:
                            rollout = build_rollout()
                        outcome = terminal_reward(
                            rollout,
                            (example.get("question") or {}).get("answer") or {},
                            dataset=dataset,
                            selected_entries=selected_entries,
                            supervision=reward_supervision.get(supervision_key(example)),
                            question_type=str((example.get("question") or {}).get("question_type") or ""),
                        )
                        outcome["executor_cache_hit"] = cache_hit
                        outcome["executor_cache_key"] = cache_key
                        outcome["rollout_diagnostic"] = compact_rollout_diagnostic(rollout)
                        with executor_traces_path.open("a", encoding="utf-8") as handle:
                            handle.write(json.dumps({
                                "group": group_index,
                                "sample": sample_index,
                                "example_id": example.get("example_id"),
                                "repeat_index": int(example.get("_grpo_repeat_index") or 0),
                                "dataset": dataset,
                                "selected_indices": indices,
                                **compact_executor_trace(rollout),
                            }) + "\n")
                    except RolloutTimeoutError:
                        outcome = {
                            "reward": -1.0,
                            "terminal_success": False,
                            "answer_correct": False,
                            "verifier_passed": False,
                            "acceptance_status": "rollout_timeout",
                            "predicted_label": None,
                        }
            else:
                outcome = {
                    "reward": -1.0,
                    "terminal_success": False,
                    "answer_correct": False,
                    "verifier_passed": False,
                    "acceptance_status": "invalid_retrieval_action",
                    "predicted_label": None,
                }
            outcome["format_budget_compliant"] = bool(format_compliant)
            if not format_compliant:
                outcome["terminal_success"] = False
                outcome["process_supported"] = False
                outcome["reward"] = min(float(outcome["reward"]), 0.20)
                if isinstance(outcome.get("reward_components"), dict):
                    outcome["reward_components"]["format_budget"] = 0.0
            sample = {
                    "input_ids": full_ids,
                    "prompt_len": prompt_len,
                    "completion": completion,
                    "selected_indices": indices,
                    **outcome,
            }
            if not args.eval_only:
                if live_pointwise_scores_tensor is not None:
                    assert reference_pointwise_scores_tensor is not None
                    sample["pointwise_ordered_indices"] = list(sampled_ordered_indices)
                    sample["old_logprobs"] = plackett_luce_logprob(
                        live_pointwise_scores_tensor,
                        sampled_ordered_indices,
                        temperature=pointwise_temperature,
                    ).detach().reshape(1)
                    sample["ref_logprobs"] = plackett_luce_logprob(
                        reference_pointwise_scores_tensor,
                        sampled_ordered_indices,
                        temperature=pointwise_temperature,
                    ).detach().reshape(1)
                else:
                    policy.eval()
                    sample["old_logprobs"] = _token_logprobs(
                        policy, full_ids, prompt_len, requires_grad=False
                    ).detach()
                    policy.set_adapter(reference_adapter_name)
                    sample["ref_logprobs"] = _token_logprobs(
                        policy, full_ids, prompt_len, requires_grad=False
                    ).detach()
                    policy.set_adapter(train_adapter_name)
            samples.append(sample)
            status_name = str(outcome.get("acceptance_status") or "unknown")
            dataset_status_counts[dataset][status_name] += 1
            dataset_stats[dataset]["valid_retrieval_actions"] += int(
                status_name != "invalid_retrieval_action"
            )
            dataset_stats[dataset]["answer_correct"] += int(bool(outcome.get("answer_correct")))
            dataset_stats[dataset]["verifier_passed"] += int(bool(outcome.get("verifier_passed")))
            dataset_stats[dataset]["format_compliant_actions"] += int(format_compliant)
            for name, value in (outcome.get("reward_components") or {}).items():
                dataset_component_sums[dataset][str(name)] += float(value)
            sample_event = {
                "event": "terminal_sample",
                "group": group_index,
                "sample": sample_index,
                "example_id": example.get("example_id"),
                "repeat_index": int(example.get("_grpo_repeat_index") or 0),
                "dataset": dataset,
                "question_type": str(
                    (example.get("question") or {}).get("question_type") or ""
                ),
                "selected_indices": indices,
                "completion_snippet": completion[:240],
                **outcome,
            }
            with sample_events_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(sample_event) + "\n")
            print(json.dumps(sample_event), flush=True)
        policy.config.use_cache = False
        rewards = [float(sample["reward"]) for sample in samples]
        advantages = centered_group_advantages(rewards)
        trainable_group = is_trainable_reward_group(
            samples, process_reward_warmup=bool(args.process_reward_warmup)
        )
        reward_variance_group = len({round(value, 8) for value in rewards}) > 1
        process_hit_group = any(bool(sample.get("process_supported")) for sample in samples)
        dataset_stats[dataset]["groups_trainable"] += int(trainable_group)
        dataset_stats[dataset]["groups_reward_variance"] += int(reward_variance_group)
        dataset_stats[dataset]["groups_process_hit"] += int(process_hit_group)
        total_samples += len(samples)
        terminal_successes += sum(bool(sample["terminal_success"]) for sample in samples)
        dataset_stats[dataset]["samples"] += len(samples)
        dataset_stats[dataset]["terminal_successes"] += sum(
            bool(sample["terminal_success"]) for sample in samples
        )
        if args.eval_only:
            metric = {
                "group": group_index,
                "example_id": example.get("example_id"),
                "repeat_index": int(example.get("_grpo_repeat_index") or 0),
                "dataset": dataset,
                "question_type": str(
                    (example.get("question") or {}).get("question_type") or ""
                ),
                "rewards": rewards,
                "terminal_successes": sum(bool(sample["terminal_success"]) for sample in samples),
                "process_supported_samples": sum(
                    bool(sample.get("process_supported")) for sample in samples
                ),
                "format_compliant_samples": sum(
                    bool(sample.get("format_budget_compliant")) for sample in samples
                ),
                "reward_variance": reward_variance_group,
            }
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(metric) + "\n")
            checkpoint_after_group(group_index)
            continue
        # Process warm-up is allowed to optimize retrieval differences without a
        # terminal answer, but only when at least one rollout actually reaches
        # evaluator-side process evidence.  Merely having unequal weak rewards
        # must not trigger an optimizer step.
        if args.process_reward_warmup and not process_hit_group:
            skipped_no_process_hit += 1
            checkpoint_after_group(group_index)
            continue
        # On answerable benchmarks, abstention/weak progress is diagnostic only.
        # Never update a terminal-GRPO group unless at least one rollout reaches
        # verified, correct terminal success; otherwise GRPO would optimize among
        # failures.
        if not args.process_reward_warmup and not any(
            bool(sample["terminal_success"]) for sample in samples
        ):
            skipped_no_terminal_success += 1
            checkpoint_after_group(group_index)
            continue
        if not samples or not any(abs(value) > 1e-8 for value in advantages):
            skipped += 1
            checkpoint_after_group(group_index)
            continue
        assert trainable_group, "optimizer update attempted for a non-trainable reward group"
        for _ in range(max(1, int(args.ppo_epochs))):
            assert optimizer is not None
            policy.set_adapter(train_adapter_name)
            optimizer.zero_grad(set_to_none=True)
            group_kls = []
            group_losses = []
            new_pointwise_scores = None
            if live_pointwise_scores_tensor is not None:
                # First evaluate the complete score vector without retaining the
                # 9B-model graph.  The small leaf tensor is sufficient to compute
                # the exact derivative of the joint PL/GRPO loss with respect to
                # every candidate score.
                new_pointwise_scores = pointwise_policy_scores_tensor(
                    policy,
                    tokenizer,
                    example,
                    device=device,
                    requires_grad=False,
                ).detach().requires_grad_(True)
            else:
                # Release generation/reference cache blocks before the first
                # differentiable token completion on 24GB devices.
                torch.cuda.empty_cache()
            if new_pointwise_scores is not None:
                for sample, advantage in zip(samples, advantages):
                    new_lp = plackett_luce_logprob(
                        new_pointwise_scores,
                        sample["pointwise_ordered_indices"],
                        temperature=pointwise_temperature,
                    ).reshape(1)
                    loss, _, kl = clipped_grpo_loss(
                        new_lp,
                        sample["old_logprobs"],
                        sample["ref_logprobs"],
                        advantage,
                        clip_eps=args.clip_eps,
                        kl_coef=args.kl_coef,
                    )
                    # The PL loss couples every candidate score, so its small
                    # score-space graph is differentiated once below.
                    group_losses.append(loss / len(samples))
                    group_kls.append(float(kl.detach().cpu()))
                total_group_loss = torch.stack(group_losses).sum()
                # Applying the score-space VJP one candidate at a time is exactly
                # the chain rule for the joint PL loss, but releases each Qwen3.5
                # graph immediately.  Holding all candidate graphs until one
                # backward exceeds 24GB even when their forwards are micro-batched.
                score_gradients = torch.autograd.grad(
                    total_group_loss, new_pointwise_scores, retain_graph=False
                )[0].detach()
                del total_group_loss, group_losses, new_pointwise_scores
                torch.cuda.empty_cache()
                for candidate_index, score_gradient in enumerate(score_gradients):
                    if abs(float(score_gradient)) <= 1e-12:
                        continue
                    candidate_score = pointwise_policy_scores_tensor(
                        policy,
                        tokenizer,
                        example,
                        device=device,
                        batch_size=int(args.pointwise_train_batch_size),
                        requires_grad=True,
                        candidate_indices=[candidate_index],
                    )[0]
                    candidate_score.backward(gradient=score_gradient)
                    del candidate_score
                del score_gradients
            else:
                optimizer_state_offloaded_bytes = 0
                inactive_adapter_offloaded_bytes = 0
                inactive_adapter_parameters: list[tuple[Any, Any]] = []

                def move_optimizer_state(target_device: Any) -> int:
                    moved_bytes = 0
                    for optimizer_entry in optimizer.state.values():
                        for state_name, state_value in list(optimizer_entry.items()):
                            if not torch.is_tensor(state_value):
                                continue
                            if state_value.device == torch.device(target_device):
                                continue
                            moved_bytes += state_value.numel() * state_value.element_size()
                            optimizer_entry[state_name] = state_value.to(target_device)
                    return moved_bytes

                def prepare_offloaded_retry() -> None:
                    nonlocal optimizer_state_offloaded_bytes, inactive_adapter_offloaded_bytes
                    # Adam moments are not read until optimizer.step().  Moving
                    # them to CPU during the exact backward recomputation frees
                    # ~330 MiB on this run without changing any tensor values or
                    # optimizer math.  They are restored before clipping/step.
                    optimizer_state_offloaded_bytes = move_optimizer_state("cpu")
                    inactive_names = {
                        reference_adapter_name,
                        *(
                            adapter_name
                            for adapter_name in dataset_adapter_names.values()
                            if adapter_name != train_adapter_name
                        ),
                    }
                    for parameter_name, parameter in policy.named_parameters():
                        if parameter.device.type != "cuda":
                            continue
                        if not any(
                            f".{adapter_name}." in parameter_name
                            for adapter_name in inactive_names
                        ):
                            continue
                        inactive_adapter_parameters.append((parameter, parameter.device))
                        inactive_adapter_offloaded_bytes += (
                            parameter.numel() * parameter.element_size()
                        )
                        parameter.data = parameter.data.to("cpu")

                def backward_token_group(*, offload_saved_tensors: bool) -> list[float]:
                    local_kls = []
                    context = (
                        torch.autograd.graph.save_on_cpu(pin_memory=True)
                        if offload_saved_tensors
                        else nullcontext()
                    )
                    with context:
                        for sample, advantage in zip(samples, advantages):
                            policy.train()
                            new_lp = _token_logprobs(
                                policy,
                                sample["input_ids"],
                                sample["prompt_len"],
                                requires_grad=True,
                            )
                            loss, _, kl = clipped_grpo_loss(
                                new_lp,
                                sample["old_logprobs"],
                                sample["ref_logprobs"],
                                advantage,
                                clip_eps=args.clip_eps,
                                kl_coef=args.kl_coef,
                            )
                            # Token completions are conditionally independent once
                            # group advantages are fixed. Immediate scaled backward
                            # is exactly the summed group objective.
                            (loss / len(samples)).backward()
                            local_kls.append(float(kl.detach().cpu()))
                    return local_kls

                group_kls, used_oom_fallback = retry_exact_backward_after_oom(
                    lambda: backward_token_group(offload_saved_tensors=False),
                    lambda: backward_token_group(offload_saved_tensors=True),
                    optimizer=optimizer,
                    empty_cache=torch.cuda.empty_cache,
                    oom_type=torch.OutOfMemoryError,
                    prepare_retry=prepare_offloaded_retry,
                )
                if used_oom_fallback:
                    for parameter, original_device in inactive_adapter_parameters:
                        parameter.data = parameter.data.to(original_device)
                    move_optimizer_state(device)
                    oom_offload_retries += 1
                    print(json.dumps({
                        "event": "terminal_grpo_oom_offload_retry",
                        "group": group_index,
                        "dataset": dataset,
                        "contract": OOM_OFFLOAD_FALLBACK_VERSION,
                        "optimizer_state_offloaded_bytes": optimizer_state_offloaded_bytes,
                        "inactive_adapter_offloaded_bytes": inactive_adapter_offloaded_bytes,
                    }), flush=True)
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
            kl_values.extend(group_kls)
        trained += 1
        dataset_stats[dataset]["groups_trained"] += 1
        metric = {
            "group": group_index,
            "example_id": example.get("example_id"),
            "repeat_index": int(example.get("_grpo_repeat_index") or 0),
            "rewards": rewards,
            "advantages": advantages,
            "terminal_successes": sum(bool(sample["terminal_success"]) for sample in samples),
            "mean_kl": sum(group_kls) / max(1, len(group_kls)),
        }
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(metric) + "\n")
        checkpoint_after_group(group_index)

    # A run without an optimizer step is a diagnostic, not a trained adapter.
    # Do not emit a loadable artifact that can be mistaken for a GRPO result.
    adapter_out = args.output_dir / "adapter"
    artifact_status = (
        "eval_only" if args.eval_only
        else "trained" if trained > 0
        else "withheld_no_successful_update"
    )
    if trained > 0:
        policy.save_pretrained(
            adapter_out,
            selected_adapters=["default", *sorted(dataset_adapter_names.values())],
        )
        tokenizer.save_pretrained(adapter_out)
    trained_adapter_outputs: dict[str, dict[str, str]] = {}
    if trained > 0:
        primary_path = adapter_out
        trained_adapter_outputs["default"] = {
            "adapter": str(primary_path),
            "adapter_weight_sha256": adapter_weight_sha256(primary_path),
        }
        for dataset, adapter_name in sorted(dataset_adapter_names.items()):
            route_path = adapter_out / adapter_name
            if not route_path.is_dir():
                raise RuntimeError(f"trained routed adapter was not saved: {route_path}")
            trained_adapter_outputs[dataset] = {
                "adapter": str(route_path),
                "adapter_weight_sha256": adapter_weight_sha256(route_path),
            }
    report = {
        "schema_version": "video-skills/l2-terminal-on-policy-grpo-v1",
        "split_role": args.split_role,
        "source_adapter": str(args.adapter),
        "source_adapter_weight_sha256": adapter_weight_sha256(args.adapter),
        "dataset_adapter_backends": {
            dataset: {
                "adapter": str(path),
                "adapter_weight_sha256": adapter_weight_sha256(path),
            }
            for dataset, path in sorted(dataset_adapter_paths.items())
        },
        "adapter_out": str(adapter_out) if trained > 0 else None,
        "sample_events": str(sample_events_path),
        "trained_adapter_outputs": trained_adapter_outputs,
        "artifact_status": artifact_status,
        "reference_runtime_contract": (
            REFERENCE_RUNTIME_VERSION if not args.eval_only else None
        ),
        "eval_only": bool(args.eval_only),
        "retrieval_only": bool(args.retrieval_only),
        "process_reward_warmup": bool(args.process_reward_warmup),
        "process_warmup_reward_contract": (
            PROCESS_WARMUP_REWARD_VERSION if args.process_reward_warmup else None
        ),
        "terminal_on_process_hit": bool(args.terminal_on_process_hit),
        "controller_action_contract": (
            (
                POINTWISE_ACTION_CONTRACT_VERSION
                if pointwise_action_datasets == {"cg_bench", "video_holmes"}
                else DATASET_ROUTED_ACTION_CONTRACT_VERSION
            )
            if args.pointwise_action_policy
            else ACTION_CONTRACT_VERSION
        ),
        "pointwise_action_datasets": (
            sorted(pointwise_action_datasets) if args.pointwise_action_policy else []
        ),
        "sampling_protocol": {
            "generation_temperature": float(args.temperature),
            "pointwise_temperature": (
                pointwise_temperature if args.pointwise_action_policy else None
            ),
            "pointwise_train_batch_size": (
                int(args.pointwise_train_batch_size) if args.pointwise_action_policy else None
            ),
            "pointwise_sampler": (
                "gumbel-top-k-without-replacement-v1"
                if args.pointwise_action_policy else None
            ),
            "pointwise_gradient_contract": (
                POINTWISE_GRADIENT_CONTRACT_VERSION
                if args.pointwise_action_policy and not args.eval_only else None
            ),
        },
        "resume_checkpoint_contract": (
            RESUME_CHECKPOINT_VERSION
            if not args.eval_only and int(args.checkpoint_every_groups) > 0
            else None
        ),
        "checkpoint_every_groups": (
            int(args.checkpoint_every_groups) if not args.eval_only else 0
        ),
        "resumed_from_group_index": resume_next_group_index,
        "relationship_support_contract": RELATIONSHIP_SUPPORT_VERSION,
        "terminal_reward_contract": TERMINAL_REWARD_VERSION,
        "executor_isolation_contract": EXECUTOR_ISOLATION_VERSION,
        "executor_fallback_contract": EXECUTOR_FALLBACK_VERSION,
        "dataset_executor_backends": {
            "cg_bench": executor_backend_for_dataset("cg_bench", args.skill_model),
            "video_holmes": executor_backend_for_dataset("video_holmes", args.skill_model),
        },
        "executor_cache_contract": EXECUTOR_CACHE_VERSION if args.executor_cache_dir else None,
        "executor_cache_dir": str(args.executor_cache_dir) if args.executor_cache_dir else None,
        "executor_cache_hits": executor_cache_hits,
        "executor_cache_misses": executor_cache_misses,
        "pre_filter_pool_size": pre_filter_pool_size,
        "frozen_l1_paths": len(paths),
        "frozen_l1_unique_examples": len(examples),
        "post_filter_pool_size": len(pool),
        "unique_pool_examples_before_repeats": unique_pool_examples_before_repeats,
        "repeats_per_example": max(1, int(args.repeats_per_example)),
        "repeat_start_index": max(0, int(args.repeat_start_index)),
        "pre_supervision_pool_size": pre_supervision_pool_size,
        "pool_filters": {
            "datasets": sorted(datasets) if datasets else None,
            "example_id_allowlist": str(args.example_id_allowlist) if args.example_id_allowlist else None,
            "example_id_allowlist_sha256": _file_sha256(args.example_id_allowlist),
            "exact_mined_group_allowlist": bool(exact_groups is not None),
            "preserve_allowlist_order": bool(args.preserve_allowlist_order),
            "dataset_balanced_sampling": bool(args.dataset_balanced_sampling),
            "dataset_balancing_contract": (
                DATASET_BALANCING_VERSION if args.dataset_balanced_sampling else None
            ),
            "require_process_supervision": bool(args.require_process_supervision),
            "min_catalog_size": int(args.min_catalog_size),
            "max_catalog_size": int(args.max_catalog_size) if args.max_catalog_size is not None else None,
        },
        "split_manifest": str(args.split_manifest),
        "split_manifest_sha256": _file_sha256(args.split_manifest),
        "groups_seen": len(pool),
        "groups_trainable": sum(stats["groups_trainable"] for stats in dataset_stats.values()),
        "groups_trained": trained,
        "groups_skipped_equal_reward": skipped,
        "groups_skipped_no_process_hit": skipped_no_process_hit,
        "groups_skipped_no_terminal_success": skipped_no_terminal_success,
        "oom_offload_retries": oom_offload_retries,
        "oom_offload_fallback_contract": OOM_OFFLOAD_FALLBACK_VERSION,
        "samples": total_samples,
        "terminal_successes": terminal_successes,
        "terminal_success_rate": terminal_successes / max(1, total_samples),
        "dataset_metrics": {
            dataset: {
                **stats,
                "terminal_success_rate": stats["terminal_successes"] / max(1, stats["samples"]),
                "trainable_group_rate": stats["groups_trainable"] / max(1, stats["groups_seen"]),
                "reward_variance_group_rate": stats["groups_reward_variance"] / max(1, stats["groups_seen"]),
                "process_hit_group_rate": stats["groups_process_hit"] / max(1, stats["groups_seen"]),
                "valid_retrieval_action_rate": stats["valid_retrieval_actions"] / max(1, stats["samples"]),
                "answer_accuracy": stats["answer_correct"] / max(1, stats["samples"]),
                "verifier_pass_rate": stats["verifier_passed"] / max(1, stats["samples"]),
                "format_compliance_rate": stats["format_compliant_actions"] / max(1, stats["samples"]),
                "acceptance_status_counts": dict(dataset_status_counts[dataset]),
                "mean_reward_components": {
                    name: value / max(1, stats["samples"])
                    for name, value in sorted(dataset_component_sums[dataset].items())
                },
            }
            for dataset, stats in sorted(dataset_stats.items())
        },
        "mean_kl": sum(kl_values) / max(1, len(kl_values)),
        "mock_semantic_judge": False,
        "remote_rollout_policy": False,
        "fixed_remote_environment_executor": True,
        "boundary_anchor_index0": bool(args.boundary_anchor_index0),
        "elapsed_s": previous_elapsed_s + (time.time() - started),
        "peak_gpu_memory_gb": torch.cuda.max_memory_allocated() / (1024**3),
    }
    run_manifest_path = args.output_dir / "posttraining_run_manifest.json"
    run_stage = (
        "grpo_retrieval_mining" if args.retrieval_only
        else "grpo_process_warmup" if args.process_reward_warmup
        else "grpo_terminal_eval" if args.eval_only
        else "grpo_terminal_train"
    )
    run_manifest = build_posttraining_manifest(
        stage=run_stage,
        split_manifest_path=args.split_manifest,
        reward_spec_version=TERMINAL_REWARD_VERSION,
        grpo_mode=str(report["controller_action_contract"]),
        update_modules=["default", *sorted(dataset_adapter_names.values())],
        policy_checkpoint=str(args.adapter),
        reference_checkpoint=str(args.adapter),
        k_samples=int(args.k),
        candidate_order_seeds=[int(args.seed)],
        extras={
            "split_role": args.split_role,
            "source_adapter_weight_sha256": report["source_adapter_weight_sha256"],
            "dataset_adapter_backends": report["dataset_adapter_backends"],
            "sampling_protocol": report["sampling_protocol"],
            "pointwise_action_datasets": report["pointwise_action_datasets"],
            "dataset_balanced_sampling": bool(args.dataset_balanced_sampling),
            "learning_rate": float(args.learning_rate),
            "clip_eps": float(args.clip_eps),
            "kl_coef": float(args.kl_coef),
            "ppo_epochs": int(args.ppo_epochs),
            "groups_seen": report["groups_seen"],
            "groups_trained": report["groups_trained"],
            "artifact_status": artifact_status,
            "trained_adapter_outputs": trained_adapter_outputs,
        },
    )
    save_posttraining_manifest(run_manifest_path, run_manifest)
    report["posttraining_run_manifest"] = str(run_manifest_path)
    report["posttraining_run_manifest_content_hash"] = run_manifest.content_hash()
    (args.output_dir / "terminal_grpo_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"event": "terminal_grpo_complete", **report}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
