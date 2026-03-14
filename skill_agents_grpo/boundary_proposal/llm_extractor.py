"""
LLM-based signal extraction for boundary proposal.

Instead of hardcoded keywords, this module uses the framework's LLM
(via ``ask_model``) to extract structured predicates from natural-language
state descriptions.  This makes boundary proposal **environment-agnostic**:
no per-env keyword lists needed.

Design choices
--------------
- **Batch processing**: States are grouped into chunks (default 30) and
  sent in a single LLM call per chunk.  One trajectory of T=1000 steps
  requires ~33 cheap LLM calls, not 1000.
- **Structured JSON output**: The LLM is prompted to return a JSON array
  of predicate dicts.  Parsing is robust (regex fallback for partial JSON).
- **Two-pass option**: (1) extract predicates, (2) optionally ask the LLM
  which predicate *changes* are boundary-significant (filters noise).
- **Fallback**: If an LLM call fails, that chunk falls back to empty
  predicates (hard events still fire from the rule-based layer).
"""

from __future__ import annotations

import json
import re
import logging
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

# Allow importing from repo root
_repo_root = Path(__file__).resolve().parent.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from skill_agents_grpo.boundary_proposal.signal_extractors import SignalExtractorBase

logger = logging.getLogger(__name__)


def _make_boundary_ask_fn():
    """Build an ask_model callable routed through the LoRA boundary adapter.

    Returns None if the lora module is not configured, letting callers
    fall back to the default API-based ``ask_model``.
    """
    try:
        from skill_agents_grpo.lora import MultiLoraSkillBankLLM, SkillFunction
        llm = MultiLoraSkillBankLLM.get_shared_instance()
        if llm is not None:
            return llm.as_ask_fn(SkillFunction.BOUNDARY)
    except Exception:
        pass
    return None

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_PREDICATE_EXTRACTION_PROMPT = """\
You are analyzing a game agent's trajectory to identify key state facts (predicates) at each timestep.

For each timestep below, extract the important discrete state facts as key-value pairs.
Focus on facts that, when they CHANGE between consecutive steps, would signal a meaningful transition:
- Location or area the agent is in
- Items held, inventory changes
- Game phase, menu/UI mode
- Objectives completed or active
- Interaction targets (NPCs, objects)
- Agent status (alive, health level, role)
- Team composition or alliances

Return ONLY a JSON array with one object per timestep, in order.
Each object should have string keys and string/boolean/number values.
Keep predicate names consistent across timesteps (same key = same concept).

Timesteps:
{states_block}

Return JSON array (length {num_states}):"""

_BOUNDARY_SIGNIFICANCE_PROMPT = """\
You are analyzing state transitions in a game trajectory to identify which changes represent meaningful skill/subtask boundaries.

Below are consecutive pairs of predicate sets where something changed.
For each pair, decide: is this change a SIGNIFICANT boundary (a shift in what the agent is doing)?

Answer ONLY with a JSON array of booleans, one per pair. true = significant boundary, false = minor change.

Consider significant:
- Area/location changes
- Phase or mode transitions
- Objective completion or switch
- Major inventory changes (gaining key item, not routine pickups)
- Role or team changes

Consider NOT significant:
- Minor position shifts within the same area
- Routine repeated actions
- Small numeric changes (score +1)

Pairs (timestep, before → after):
{pairs_block}

Return JSON array of booleans (length {num_pairs}):"""


# ---------------------------------------------------------------------------
# JSON parsing helpers
# ---------------------------------------------------------------------------


def _parse_json_array(text: str) -> Optional[list]:
    """
    Try to parse a JSON array from LLM output.
    Handles common issues: markdown fences, trailing commas, partial output.
    """
    # Strip markdown code fences
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = text.strip().rstrip("`")

    # Try direct parse
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        pass

    # Try to find the outermost [ ... ] bracket pair
    match = re.search(r"\[[\s\S]*\]", text)
    if match:
        candidate = match.group(0)
        # Fix trailing commas before ] or }
        candidate = re.sub(r",\s*([}\]])", r"\1", candidate)
        try:
            result = json.loads(candidate)
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass

    return None


# ---------------------------------------------------------------------------
# LLM Signal Extractor
# ---------------------------------------------------------------------------


class LLMSignalExtractor(SignalExtractorBase):
    """
    Environment-agnostic signal extractor that uses an LLM to extract
    structured predicates from natural-language state descriptions.

    Parameters
    ----------
    ask_model_fn : callable, optional
        The LLM call function.  Signature: ``ask_model_fn(prompt, **kwargs) -> str``.
        If None, imports ``ask_model`` from ``API_func`` at call time.
    model : str, optional
        Model name passed to ask_model (e.g. "gpt-4o-mini", "gemini-2.5-flash").
        Cheap/fast models are recommended since this is Stage 1.
    chunk_size : int
        Number of timesteps per LLM call (default 30).
    temperature : float
        LLM temperature (low for structured output).
    filter_significance : bool
        If True, run a second LLM pass to filter which predicate changes
        are boundary-significant (reduces false positives).
    max_state_chars : int
        Truncate each state string to this length to control prompt size.
    reward_spike_std : float
        Std factor for reward spike detection (rule-based, always on).
    """

    def __init__(
        self,
        ask_model_fn: Optional[Callable] = None,
        model: Optional[str] = None,
        chunk_size: int = 30,
        temperature: float = 0.2,
        filter_significance: bool = False,
        max_state_chars: int = 500,
        reward_spike_std: float = 2.0,
    ):
        self._ask_model_fn = ask_model_fn
        self._model = model
        self._chunk_size = chunk_size
        self._temperature = temperature
        self._filter_significance = filter_significance
        self._max_state_chars = max_state_chars
        self._reward_spike_std = reward_spike_std

    def _get_ask_model(self) -> Callable:
        from skill_agents_grpo._llm_compat import wrap_ask_for_reasoning_models

        if self._ask_model_fn is not None:
            return wrap_ask_for_reasoning_models(
                self._ask_model_fn, model_hint=self._model,
            )
        lora_fn = _make_boundary_ask_fn()
        if lora_fn is not None:
            return wrap_ask_for_reasoning_models(lora_fn, model_hint=self._model)
        from API_func import ask_model
        return wrap_ask_for_reasoning_models(ask_model, model_hint=self._model)

    def _state_to_text(self, state: Any) -> str:
        """Convert a state (str, dict, or other) to a concise text repr."""
        if isinstance(state, str):
            text = state
        elif isinstance(state, dict):
            text = json.dumps(state, default=str, ensure_ascii=False)
        else:
            text = str(state)
        if len(text) > self._max_state_chars:
            text = text[: self._max_state_chars] + "..."
        return text

    def _extract_predicates_chunk(
        self,
        states: List[str],
        chunk_offset: int,
    ) -> List[dict]:
        """Extract predicates for one chunk of states via LLM."""
        ask = self._get_ask_model()

        states_block = "\n".join(
            f"  t={chunk_offset + i}: {s}" for i, s in enumerate(states)
        )
        prompt = _PREDICATE_EXTRACTION_PROMPT.format(
            states_block=states_block,
            num_states=len(states),
        )

        kwargs: dict = {"temperature": self._temperature, "max_tokens": 3000}
        if self._model is not None:
            kwargs["model"] = self._model

        try:
            response = ask(prompt, **kwargs)
            parsed = _parse_json_array(response)
            if parsed is not None and len(parsed) == len(states):
                return [p if isinstance(p, dict) else {} for p in parsed]
            elif parsed is not None:
                # Length mismatch — pad or truncate
                result = [p if isinstance(p, dict) else {} for p in parsed]
                while len(result) < len(states):
                    result.append({})
                return result[: len(states)]
            else:
                logger.warning(
                    "LLM predicate extraction: failed to parse JSON for chunk at t=%d",
                    chunk_offset,
                )
                return [{} for _ in states]
        except Exception as e:
            logger.warning("LLM predicate extraction failed for chunk at t=%d: %s", chunk_offset, e)
            return [{} for _ in states]

    def extract_predicates(self, experiences: list) -> List[Optional[dict]]:
        """
        Extract predicates from all experiences using batched LLM calls.

        States are chunked into groups of ``chunk_size``.  Each chunk is
        sent as one LLM call that returns a JSON array of predicate dicts.
        """
        T = len(experiences)
        if T == 0:
            return []

        state_texts = [self._state_to_text(exp.state) for exp in experiences]
        all_predicates: List[dict] = []

        for start in range(0, T, self._chunk_size):
            end = min(start + self._chunk_size, T)
            chunk_states = state_texts[start:end]
            chunk_preds = self._extract_predicates_chunk(chunk_states, start)
            all_predicates.extend(chunk_preds)

        # Ensure consistent predicate keys across the trajectory
        all_predicates = _normalize_predicate_keys(all_predicates)

        # Optional: filter significance
        if self._filter_significance:
            all_predicates = self._filter_significant_changes(all_predicates)

        return all_predicates

    def _filter_significant_changes(
        self,
        predicates: List[dict],
    ) -> List[dict]:
        """
        Second LLM pass: for each predicate change, ask whether it's a
        significant boundary.  Non-significant changes are smoothed out
        (set to match the previous step so they don't trigger a flip).
        """
        ask = self._get_ask_model()
        T = len(predicates)

        # Find change points
        change_pairs: List[Tuple[int, dict, dict]] = []
        for t in range(1, T):
            prev, curr = predicates[t - 1], predicates[t]
            if prev != curr:
                change_pairs.append((t, prev, curr))

        if not change_pairs:
            return predicates

        # Batch the significance query
        pairs_block = "\n".join(
            f"  t={t}: {json.dumps(prev, default=str)} → {json.dumps(curr, default=str)}"
            for t, prev, curr in change_pairs
        )
        prompt = _BOUNDARY_SIGNIFICANCE_PROMPT.format(
            pairs_block=pairs_block,
            num_pairs=len(change_pairs),
        )

        kwargs: dict = {"temperature": 0.1, "max_tokens": 2000}
        if self._model is not None:
            kwargs["model"] = self._model

        try:
            response = ask(prompt, **kwargs)
            parsed = _parse_json_array(response)
            if parsed and len(parsed) >= len(change_pairs):
                significances = [bool(v) for v in parsed[: len(change_pairs)]]
            else:
                # Can't parse — keep all changes (safe default)
                return predicates
        except Exception:
            return predicates

        # Smooth out non-significant changes
        result = [dict(p) for p in predicates]
        for (t, prev, curr), is_sig in zip(change_pairs, significances):
            if not is_sig:
                result[t] = dict(result[t - 1])

        return result

    def extract_event_times(self, experiences: list) -> List[int]:
        """
        Rule-based hard event detection (cheap, always-on).
        This does NOT use the LLM — hard events are structural.
        """
        events = []
        for t, exp in enumerate(experiences):
            # Done / episode end
            if exp.done:
                events.append(t)

            # Reward: any non-zero reward is potentially interesting
            r = exp.reward if exp.reward is not None else 0.0
            if isinstance(r, dict):
                r = sum(r.values())
            if isinstance(r, (tuple, list)):
                r = sum(r)

        # Add reward spikes
        events.extend(self.detect_reward_spike_events(experiences, self._reward_spike_std))
        return sorted(set(events))


# ---------------------------------------------------------------------------
# Predicate key normalization
# ---------------------------------------------------------------------------


def _normalize_predicate_keys(predicates: List[dict]) -> List[dict]:
    """
    Ensure predicate dicts use consistent keys across the trajectory.

    The LLM might use slightly different key names across chunks
    (e.g. "location" vs "area").  This pass collects the union of all
    keys and fills missing keys with None so that flip detection works.
    """
    if not predicates:
        return predicates

    all_keys = set()
    for p in predicates:
        all_keys.update(p.keys())

    normalized = []
    for p in predicates:
        normed = {k: p.get(k) for k in all_keys}
        normalized.append(normed)
    return normalized
