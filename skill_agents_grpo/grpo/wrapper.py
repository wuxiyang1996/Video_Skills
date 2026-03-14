"""
Generic GRPO call wrapper: intercepts an LLM function, samples G times,
evaluates via a reward function, stores results in a buffer, and returns
the best sample to the calling pipeline.

The EM pipeline runs unchanged — it just gets better LLM outputs as
GRPO training progresses.

Usage::

    from skill_agents_grpo.grpo import GRPOCallWrapper, GRPOBuffer
    from skill_agents_grpo.lora import SkillFunction

    buffer = GRPOBuffer()
    wrapper = GRPOCallWrapper(
        adapter=SkillFunction.CONTRACT,
        reward_fn=contract_reward,
        buffer=buffer,
        group_size=4,
        temperature=0.7,
    )

    # Monkey-patch or inject
    original_fn = llm_summarize_contract
    llm_summarize_contract = wrapper.wrap(original_fn)
"""

from __future__ import annotations

import logging
from functools import wraps
from typing import Any, Callable, Dict, List, Optional

from skill_agents_grpo.grpo.buffer import GRPOBuffer, GRPOSample
from skill_agents_grpo.lora.skill_function import SkillFunction

logger = logging.getLogger(__name__)


class GRPOCallWrapper:
    """Wrap an LLM call to generate G samples, evaluate, store, return best.

    Parameters
    ----------
    adapter : SkillFunction
        Which LoRA adapter this wrapper targets.
    reward_fn : callable
        ``reward_fn(sample_output, *original_args, **original_kwargs) -> float``
        Evaluates a single sample. Must be CPU-only and deterministic.
    buffer : GRPOBuffer
        Shared buffer that accumulates samples for deferred training.
    group_size : int
        Number of samples per call (G).
    temperature : float
        Sampling temperature injected during rollout.
    prompt_extractor : callable, optional
        ``prompt_extractor(*args, **kwargs) -> str``
        Extracts the prompt string from the original function's arguments.
        Needed for log-prob recomputation during training. If None, the
        prompt is stored as empty (training will skip these samples).
    metadata_extractor : callable, optional
        ``metadata_extractor(*args, **kwargs) -> dict``
        Extracts context metadata (e.g. skill_id) for diagnostics.
    """

    def __init__(
        self,
        adapter: SkillFunction,
        reward_fn: Callable[..., float],
        buffer: GRPOBuffer,
        group_size: int = 4,
        temperature: float = 0.7,
        prompt_extractor: Optional[Callable[..., str]] = None,
        metadata_extractor: Optional[Callable[..., Dict[str, Any]]] = None,
    ) -> None:
        self.adapter = adapter
        self.reward_fn = reward_fn
        self.buffer = buffer
        self.group_size = group_size
        self.temperature = temperature
        self.prompt_extractor = prompt_extractor
        self.metadata_extractor = metadata_extractor
        self._call_count = 0
        self._total_samples = 0

    def wrap(self, original_fn: Callable) -> Callable:
        """Return a wrapped version of *original_fn* that does GRPO sampling.

        The wrapped function has the same signature as the original.
        It generates G samples, evaluates rewards, stores the group in
        the buffer, and returns the best sample to the caller.
        """
        wrapper_self = self

        @wraps(original_fn)
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            return wrapper_self._run(original_fn, args, kwargs)

        wrapped._grpo_wrapper = wrapper_self  # type: ignore[attr-defined]
        return wrapped

    def _run(
        self,
        original_fn: Callable,
        args: tuple,
        kwargs: dict,
    ) -> Any:
        """Core sampling logic."""
        self._call_count += 1

        # Inject higher temperature for diversity
        kwargs_sample = {**kwargs, "temperature": self.temperature}

        samples: List[Any] = []
        completions: List[str] = []

        for _ in range(self.group_size):
            try:
                out = original_fn(*args, **kwargs_sample)
                samples.append(out)
                completions.append(str(out) if out is not None else "")
            except Exception:
                samples.append(None)
                completions.append("")

        # Evaluate rewards
        rewards: List[float] = []
        for sample in samples:
            try:
                r = self.reward_fn(sample, *args, **kwargs)
                rewards.append(float(r))
            except Exception:
                rewards.append(0.0)

        # Extract prompt for training
        prompt = ""
        if self.prompt_extractor is not None:
            try:
                prompt = self.prompt_extractor(*args, **kwargs)
            except Exception:
                pass

        # Extract metadata
        metadata: Dict[str, Any] = {}
        if self.metadata_extractor is not None:
            try:
                metadata = self.metadata_extractor(*args, **kwargs)
            except Exception:
                pass

        # Store in buffer
        grpo_sample = GRPOSample(
            adapter=self.adapter,
            prompt=prompt,
            completions=completions,
            rewards=rewards,
            metadata=metadata,
        )
        self.buffer.add(grpo_sample)
        self._total_samples += self.group_size

        logger.debug(
            "GRPO[%s] call #%d: rewards=%s best_idx=%d",
            self.adapter.value, self._call_count,
            [f"{r:.3f}" for r in rewards], grpo_sample.best_index,
        )

        # Return the best sample to the pipeline
        best_idx = grpo_sample.best_index
        return samples[best_idx]

    @property
    def stats(self) -> Dict[str, Any]:
        return {
            "adapter": self.adapter.value,
            "calls": self._call_count,
            "total_samples": self._total_samples,
            "buffer_size": self.buffer.size(self.adapter),
        }
