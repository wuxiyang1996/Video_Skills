"""Live Motif-gated rollout_fn for GRPO collection."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from dataset_clip_wrapper.l2_reasoning_graph.reasoning_planner import build_llm_reasoning_rollout
from dataset_clip_wrapper.perception.openrouter_client import OpenRouterClient, load_openrouter_api_key

RolloutFn = Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]]


def make_motif_gated_rollout_fn(
    *,
    motif_bank_path: str | Path,
    planner_model: str = "openai/gpt-oss-120b",
    keys_py: str = "/fs/gamma-projects/vlm-robot/keys.py",
    api_key: str | None = None,
    timeout_s: int = 180,
    motif_candidate_sink_path: str | Path | None = None,
) -> RolloutFn:
    """Build a Motif-first rollout function using the frozen OpenRouter planner client.

    Episode planning still uses the external planner for structure; the local LoRA
    policy is optimized on recorded controller JSON via sequence logprobs.
    """
    key = api_key or load_openrouter_api_key(keys_py_path=keys_py)
    client = OpenRouterClient(
        model=planner_model,
        api_key=key,
        max_tokens=1800,
        reasoning={"effort": "minimal", "exclude": True},
        timeout_s=timeout_s,
    )
    bank = str(motif_bank_path)
    sink = str(motif_candidate_sink_path) if motif_candidate_sink_path else None

    def _rollout(example: dict[str, Any], clue: dict[str, Any]) -> dict[str, Any]:
        meta = dict(example.get("metadata") or {})
        meta["motif_enabled"] = True
        meta["motif_bank_path"] = bank
        if sink:
            meta["motif_candidate_sink_path"] = sink
        example = {**example, "metadata": meta}
        return build_llm_reasoning_rollout(
            example,
            clue,
            client=client,
            skill_executor=None,
            motif_enabled=True,
            motif_bank_path=bank,
            motif_candidate_sink_path=sink,
        )

    return _rollout
