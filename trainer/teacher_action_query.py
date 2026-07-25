"""Teacher letter top-logprob query over complete-action candidates."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from .candidate_action_builder import CandidateAction, CandidateActionSet
from .exact_request_cache import ExactRequestCache, stable_hash

LETTERS = "ABCDEFGH"
MISSING_LOGPROB_FLOOR = -20.0

TeacherFn = Callable[[Mapping[str, Any]], dict[str, Any]]


@dataclass
class TeacherActionDistribution:
    state_id: str
    order_seed: int
    letter_to_action_id: dict[str, str]
    logprobs: dict[str, float]
    probs: dict[str, float]
    action_probs: dict[str, float]
    raw_response: dict[str, Any] = field(default_factory=dict)
    cache_hit: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "state_id": self.state_id,
            "order_seed": self.order_seed,
            "letter_to_action_id": self.letter_to_action_id,
            "logprobs": self.logprobs,
            "probs": self.probs,
            "action_probs": self.action_probs,
            "cache_hit": self.cache_hit,
            "raw_response": self.raw_response,
        }


def _softmax_from_logprobs(logprobs: Mapping[str, float]) -> dict[str, float]:
    if not logprobs:
        return {}
    max_lp = max(logprobs.values())
    exps = {k: math.exp(v - max_lp) for k, v in logprobs.items()}
    total = sum(exps.values()) or 1.0
    return {k: v / total for k, v in exps.items()}


def map_candidates_to_letters(
    candidates: Sequence[CandidateAction],
    *,
    order_seed: int,
) -> tuple[list[CandidateAction], dict[str, str]]:
    rng = random.Random(order_seed)
    ordered = list(candidates)
    rng.shuffle(ordered)
    if len(ordered) > len(LETTERS):
        ordered = ordered[: len(LETTERS)]
    letter_map = {LETTERS[i]: ordered[i].action_id for i in range(len(ordered))}
    return ordered, letter_map


def build_teacher_prompt(
    *,
    state: Mapping[str, Any],
    ordered_candidates: Sequence[CandidateAction],
    letter_map: Mapping[str, str],
) -> list[dict[str, str]]:
    import json

    motif = state.get("motif_online") or {}
    student = state.get("student_action") or {}
    lines = [
        "You are ranking complete executable controller actions for video QA.",
        "Judge only by action semantics (tool_name + arguments), NOT by letter position.",
        "Letter order is randomized; A is not preferred over later letters.",
        "Choose exactly one letter. Do not invent actions outside the list.",
        "State summary:",
        f"- example_id: {state.get('example_id')}",
        f"- task_family: {state.get('task_family')}",
        f"- question: {(state.get('question') or {}).get('question_text')}",
        f"- motif: {motif.get('selected_motif_id')}",
        f"- expansion_valid: {motif.get('expansion_valid')}",
        f"- student_tool: {student.get('tool_name')}",
        f"- student_args: {json.dumps(student.get('arguments') or {}, ensure_ascii=False, sort_keys=True)}",
        "Candidates:",
    ]
    inv = {v: k for k, v in letter_map.items()}
    for cand in ordered_candidates:
        letter = inv[cand.action_id]
        action_json = json.dumps(cand.action, ensure_ascii=False, sort_keys=True)
        flags = []
        if cand.is_stop:
            flags.append("STOP")
        if cand.is_abstain:
            flags.append("ABSTAIN")
        if cand.is_fallback:
            flags.append("FALLBACK")
        if cand.is_hard_negative:
            flags.append("HARD_NEG")
        flag_s = f" flags={','.join(flags)}" if flags else ""
        lines.append(
            f"{letter}. id={cand.action_id} family={cand.family}{flag_s} json={action_json}"
        )
    lines.append("Reply with a single letter only.")
    return [
        {
            "role": "system",
            "content": (
                "Return only one candidate letter. "
                "Ignore letter order; pick the best action content."
            ),
        },
        {"role": "user", "content": "\n".join(lines)},
    ]


def average_action_probs(
    distributions: Sequence[TeacherActionDistribution],
) -> dict[str, float]:
    """Mean action probabilities across order seeds (renormalized)."""
    keys: set[str] = set()
    for dist in distributions:
        keys.update(dist.action_probs)
    if not keys or not distributions:
        return {}
    n = float(len(distributions))
    avg = {k: sum(float(d.action_probs.get(k, 0.0)) for d in distributions) / n for k in keys}
    total = sum(avg.values()) or 1.0
    return {k: v / total for k, v in avg.items()}


def query_teacher_averaged(
    action_set: CandidateActionSet,
    *,
    state: Mapping[str, Any],
    teacher_fn: TeacherFn,
    order_seeds: Sequence[int],
    cache: ExactRequestCache | None = None,
    floor: float = MISSING_LOGPROB_FLOOR,
    require_logprobs: bool = True,
) -> tuple[dict[str, float], list[TeacherActionDistribution]]:
    """Query teacher under multiple letter orders and average action probs."""
    dists = [
        query_teacher_action_distribution(
            action_set,
            state=state,
            teacher_fn=teacher_fn,
            order_seed=int(seed),
            cache=cache,
            floor=floor,
            require_logprobs=require_logprobs,
        )
        for seed in order_seeds
    ]
    return average_action_probs(dists), dists


def normalize_letter_logprobs(
    letter_map: Mapping[str, str],
    raw_logprobs: Mapping[str, float] | None,
    *,
    floor: float = MISSING_LOGPROB_FLOOR,
) -> dict[str, float]:
    out: dict[str, float] = {}
    raw = {str(k).strip().upper()[:1]: float(v) for k, v in (raw_logprobs or {}).items()}
    for letter in letter_map:
        out[letter] = float(raw.get(letter, floor))
    return out


def query_teacher_action_distribution(
    action_set: CandidateActionSet,
    *,
    state: Mapping[str, Any],
    teacher_fn: TeacherFn,
    order_seed: int = 0,
    cache: ExactRequestCache | None = None,
    floor: float = MISSING_LOGPROB_FLOOR,
    require_logprobs: bool = True,
) -> TeacherActionDistribution:
    ordered, letter_map = map_candidates_to_letters(action_set.candidates, order_seed=order_seed)
    messages = build_teacher_prompt(state=state, ordered_candidates=ordered, letter_map=letter_map)
    request = {
        "state_id": action_set.state_id,
        "order_seed": order_seed,
        "letter_map": letter_map,
        "messages": messages,
        "candidate_ids": [c.action_id for c in ordered],
    }
    cache_hit = False
    if cache is not None:
        cached = cache.get(request)
        if cached is not None:
            raw = cached
            cache_hit = True
        else:
            raw = teacher_fn(request)
            cache.put(request, raw)
    else:
        raw = teacher_fn(request)

    raw_letter_logprobs = raw.get("letter_logprobs") or {}
    if require_logprobs and not raw_letter_logprobs:
        raise RuntimeError(
            "teacher response missing letter_logprobs; fail-closed "
            "(do not floor into a uniform soft target). "
            f"finish_reason={raw.get('raw_finish_reason') or raw.get('finish_reason')}"
        )

    letter_logprobs = normalize_letter_logprobs(letter_map, raw_letter_logprobs, floor=floor)
    letter_probs = _softmax_from_logprobs(letter_logprobs)
    action_probs = {
        letter_map[letter]: prob for letter, prob in letter_probs.items() if letter in letter_map
    }
    return TeacherActionDistribution(
        state_id=action_set.state_id,
        order_seed=order_seed,
        letter_to_action_id=dict(letter_map),
        logprobs=letter_logprobs,
        probs=letter_probs,
        action_probs=action_probs,
        raw_response=dict(raw),
        cache_hit=cache_hit,
    )


def mock_teacher_preferring_oracle(request: Mapping[str, Any]) -> dict[str, Any]:
    """Deterministic teacher for unit tests / smoke without API spend."""
    letter_map = request.get("letter_map") or {}
    logprobs = {letter: MISSING_LOGPROB_FLOOR for letter in letter_map}
    # Prefer oracle / student / choose_best if present.
    preferred_ids = ("oracle", "student", "choose_best", "select_next")
    inv = {v: k for k, v in letter_map.items()}
    chosen = None
    for action_id in preferred_ids:
        if action_id in inv:
            chosen = inv[action_id]
            break
    if chosen is None and letter_map:
        chosen = next(iter(letter_map))
    if chosen is not None:
        logprobs[chosen] = -0.1
    return {
        "letter": chosen,
        "letter_logprobs": logprobs,
        "teacher": "mock_prefer_oracle",
        "request_hash": stable_hash(request),
    }


def _extract_letter_logprobs_from_openrouter(payload: Mapping[str, Any]) -> dict[str, float]:
    """Parse OpenAI-style logprobs from an OpenRouter chat completion payload."""
    out: dict[str, float] = {}
    choices = payload.get("choices") or []
    if not choices:
        return out
    choice0 = choices[0] or {}
    logprobs = choice0.get("logprobs") or {}

    # Newer OpenAI shape: logprobs.content[i].top_logprobs = [{token, logprob}, ...]
    content = logprobs.get("content")
    if isinstance(content, list) and content:
        first = content[0] or {}
        for item in first.get("top_logprobs") or []:
            if not isinstance(item, dict):
                continue
            token = str(item.get("token") or "").strip().upper()
            if token[:1] in LETTERS:
                out[token[:1]] = float(item.get("logprob"))
        # Also record the sampled token if present.
        tok = str(first.get("token") or "").strip().upper()
        if tok[:1] in LETTERS and tok[:1] not in out and first.get("logprob") is not None:
            out[tok[:1]] = float(first.get("logprob"))

    # Older shape: top_logprobs = [{ "A": -0.1, ... }, ...]
    top = logprobs.get("top_logprobs")
    if isinstance(top, list) and top and isinstance(top[0], dict):
        for token, lp in top[0].items():
            t = str(token).strip().upper()
            if t[:1] in LETTERS:
                out[t[:1]] = float(lp)

    # Do not invent logprobs from content letters — missing logprobs must fail closed.
    return out


def make_openrouter_letter_teacher(
    *,
    api_key: str,
    model: str = "openai/gpt-4.1-mini",
    api_base: str = "https://openrouter.ai/api/v1/chat/completions",
    timeout_s: int = 60,
    top_logprobs: int = 10,
) -> TeacherFn:
    """TeacherFn that queries OpenRouter with logprobs=True for letter ranking."""
    import requests

    def _teacher(request: Mapping[str, Any]) -> dict[str, Any]:
        messages = list(request.get("messages") or [])
        payload = {
            "model": model,
            "temperature": 0.0,
            "max_tokens": 1,
            "messages": messages,
            "logprobs": True,
            "top_logprobs": int(top_logprobs),
        }
        response = requests.post(
            api_base,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=timeout_s,
        )
        if not response.ok:
            raise RuntimeError(
                f"teacher HTTP {response.status_code}: {response.text[:500]}"
            )
        body = response.json()
        letter_logprobs = _extract_letter_logprobs_from_openrouter(body)
        # Ensure every candidate letter has an entry (floor filled later by normalize).
        content = (((body.get("choices") or [{}])[0].get("message") or {}).get("content")) or ""
        chosen = None
        for ch in str(content).strip().upper():
            if ch in LETTERS:
                chosen = ch
                break
        return {
            "letter": chosen,
            "letter_logprobs": letter_logprobs,
            "teacher": f"openrouter:{model}",
            "request_hash": stable_hash(request),
            "raw_finish_reason": ((body.get("choices") or [{}])[0].get("finish_reason")),
        }

    return _teacher


def order_shuffle_stability(
    action_probs_a: Mapping[str, float],
    action_probs_b: Mapping[str, float],
) -> dict[str, Any]:
    """Compare two teacher distributions after different candidate orders."""
    keys = sorted(set(action_probs_a) | set(action_probs_b))
    if not keys:
        return {"l1": None, "top1_match": None, "n": 0}
    l1 = sum(abs(float(action_probs_a.get(k, 0.0)) - float(action_probs_b.get(k, 0.0))) for k in keys)
    top_a = max(keys, key=lambda k: float(action_probs_a.get(k, 0.0)))
    top_b = max(keys, key=lambda k: float(action_probs_b.get(k, 0.0)))
    return {
        "l1": l1,
        "top1_match": top_a == top_b,
        "top_a": top_a,
        "top_b": top_b,
        "n": len(keys),
    }
