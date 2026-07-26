"""Teacher letter top-logprob / ranking query over complete-action candidates."""

from __future__ import annotations

import json
import math
import random
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from .candidate_action_builder import CandidateAction, CandidateActionSet
from .exact_request_cache import ExactRequestCache, stable_hash

LETTERS = "ABCDEFGH"
MISSING_LOGPROB_FLOOR = -20.0
SOFT_TOP1_MATCH_MIN = 0.90
SOFT_MEAN_L1_MAX = 0.15
DEFAULT_RANK_TEMPERATURE = 1.0

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
    teacher_mode: str = "soft_logprob"
    ranked_action_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "state_id": self.state_id,
            "order_seed": self.order_seed,
            "letter_to_action_id": self.letter_to_action_id,
            "logprobs": self.logprobs,
            "probs": self.probs,
            "action_probs": self.action_probs,
            "cache_hit": self.cache_hit,
            "teacher_mode": self.teacher_mode,
            "ranked_action_ids": list(self.ranked_action_ids),
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
        teacher_mode="soft_logprob",
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


def soft_calibration_gates(
    *,
    top1_match_rate: float,
    mean_l1: float,
    n_rows: int,
    top1_min: float = SOFT_TOP1_MATCH_MIN,
    mean_l1_max: float = SOFT_MEAN_L1_MAX,
) -> dict[str, bool]:
    """Soft letter-logprob shuffle gates from the post-training plan."""
    return {
        "top1_match_ge_0_90": float(top1_match_rate) >= float(top1_min),
        "mean_l1_le_0_15": float(mean_l1) <= float(mean_l1_max),
        "n_rows_ge_1": int(n_rows) >= 1,
    }


def soft_calibration_passed(gates: Mapping[str, bool]) -> bool:
    return bool(gates) and all(bool(v) for v in gates.values())


def build_teacher_ranking_prompt(
    *,
    state: Mapping[str, Any],
    ordered_candidates: Sequence[CandidateAction],
    letter_map: Mapping[str, str],
) -> list[dict[str, str]]:
    motif = state.get("motif_online") or {}
    student = state.get("student_action") or {}
    letters = [LETTERS[i] for i in range(len(ordered_candidates))]
    lines = [
        "You are ranking complete executable controller actions for video QA.",
        "Judge only by action semantics (tool_name + arguments), NOT by letter position.",
        "Letter order is randomized; A is not preferred over later letters.",
        "Return a total ranking of ALL candidate letters, best first.",
        "Do not invent actions outside the list. Do not omit letters.",
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
    lines.append(
        "Reply with JSON only: "
        + json.dumps({"ranked_letters": letters}, ensure_ascii=False)
        + " but ordered best-to-worst."
    )
    return [
        {
            "role": "system",
            "content": (
                "Return strict JSON with key ranked_letters: a permutation of all "
                "candidate letters, best first. No prose."
            ),
        },
        {"role": "user", "content": "\n".join(lines)},
    ]


def _extract_json_object(text: str) -> dict[str, Any] | None:
    blob = str(text or "").strip()
    if not blob:
        return None
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", blob, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        blob = fence.group(1)
    else:
        start = blob.find("{")
        end = blob.rfind("}")
        if start >= 0 and end > start:
            blob = blob[start : end + 1]
    try:
        payload = json.loads(blob)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def parse_ranked_letters(
    raw: Mapping[str, Any],
    letter_map: Mapping[str, str],
) -> list[str]:
    """Parse a total letter ranking; fail closed if incomplete/invalid."""
    expected = set(letter_map)
    ranked: list[str] = []
    direct = raw.get("ranked_letters") or raw.get("ranked_labels")
    if isinstance(direct, list):
        ranked = [str(x).strip().upper()[:1] for x in direct]
    else:
        payload = _extract_json_object(str(raw.get("content") or raw.get("text") or ""))
        if payload is not None:
            vals = payload.get("ranked_letters") or payload.get("ranked_labels") or []
            if isinstance(vals, list):
                ranked = [str(x).strip().upper()[:1] for x in vals]
    ranked = [x for x in ranked if x in expected]
    # Deduplicate preserving order.
    seen: set[str] = set()
    uniq: list[str] = []
    for letter in ranked:
        if letter not in seen:
            seen.add(letter)
            uniq.append(letter)
    missing = expected - set(uniq)
    if missing:
        raise RuntimeError(
            "teacher ranking missing letters; fail-closed "
            f"(missing={sorted(missing)} got={uniq})"
        )
    if len(uniq) != len(expected):
        raise RuntimeError(
            f"teacher ranking length mismatch: got={uniq} expected={sorted(expected)}"
        )
    return uniq


def scores_to_probs(
    scores: Mapping[str, float],
    *,
    temperature: float = DEFAULT_RANK_TEMPERATURE,
) -> dict[str, float]:
    if not scores:
        return {}
    temp = max(float(temperature), 1e-6)
    scaled = {k: float(v) / temp for k, v in scores.items()}
    return _softmax_from_logprobs(scaled)


def borda_scores_from_rankings(
    rankings: Sequence[Sequence[str]],
    *,
    action_ids: Sequence[str],
) -> dict[str, float]:
    """Borda counts: best rank gets n-1, worst gets 0; summed over rankings."""
    ids = list(action_ids)
    n = len(ids)
    scores = {aid: 0.0 for aid in ids}
    for ranking in rankings:
        for rank, aid in enumerate(ranking):
            if aid in scores:
                scores[aid] += float(n - 1 - rank)
    return scores


def bradley_terry_scores_from_rankings(
    rankings: Sequence[Sequence[str]],
    *,
    action_ids: Sequence[str],
    n_iters: int = 64,
) -> dict[str, float]:
    """Fit BT strengths from pairwise outcomes implied by total rankings."""
    ids = list(action_ids)
    wins = {aid: 0.0 for aid in ids}
    pair_n: dict[tuple[str, str], float] = {}
    for ranking in rankings:
        present = [aid for aid in ranking if aid in wins]
        for i, better in enumerate(present):
            for worse in present[i + 1 :]:
                wins[better] += 1.0
                a, b = (better, worse) if better < worse else (worse, better)
                pair_n[(a, b)] = pair_n.get((a, b), 0.0) + 1.0
    strengths = {aid: 1.0 for aid in ids}
    for _ in range(int(n_iters)):
        updated = {}
        for i in ids:
            denom = 0.0
            for j in ids:
                if i == j:
                    continue
                a, b = (i, j) if i < j else (j, i)
                nij = pair_n.get((a, b), 0.0)
                if nij <= 0:
                    continue
                denom += nij / (strengths[i] + strengths[j])
            updated[i] = (wins[i] / denom) if denom > 0 else strengths[i]
        # Geometric-mean normalize for scale invariance.
        log_mean = sum(math.log(max(v, 1e-12)) for v in updated.values()) / max(len(updated), 1)
        scale = math.exp(log_mean) or 1.0
        strengths = {k: max(v / scale, 1e-12) for k, v in updated.items()}
    return {k: math.log(v) for k, v in strengths.items()}


def aggregate_rankings_to_action_probs(
    rankings: Sequence[Sequence[str]],
    *,
    action_ids: Sequence[str],
    method: str = "borda",
    temperature: float = DEFAULT_RANK_TEMPERATURE,
) -> dict[str, float]:
    method_l = str(method or "borda").strip().lower()
    if method_l in {"borda", "majority_borda"}:
        scores = borda_scores_from_rankings(rankings, action_ids=action_ids)
    elif method_l in {"bt", "bradley_terry", "plackett_luce", "pl"}:
        # Plackett-Luce listwise ≈ BT pairwise aggregation for distillation targets.
        scores = bradley_terry_scores_from_rankings(rankings, action_ids=action_ids)
    else:
        raise ValueError(f"unknown ranking aggregation method: {method}")
    return scores_to_probs(scores, temperature=temperature)


def query_teacher_ranking_distribution(
    action_set: CandidateActionSet,
    *,
    state: Mapping[str, Any],
    teacher_fn: TeacherFn,
    order_seed: int = 0,
    cache: ExactRequestCache | None = None,
    method: str = "borda",
    temperature: float = DEFAULT_RANK_TEMPERATURE,
) -> TeacherActionDistribution:
    """Single-order structured ranking → soft action distribution (no letter logprobs)."""
    ordered, letter_map = map_candidates_to_letters(action_set.candidates, order_seed=order_seed)
    messages = build_teacher_ranking_prompt(
        state=state, ordered_candidates=ordered, letter_map=letter_map
    )
    request = {
        "mode": "ranking",
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

    ranked_letters = parse_ranked_letters(raw, letter_map)
    ranked_action_ids = [letter_map[letter] for letter in ranked_letters]
    action_ids = [c.action_id for c in ordered]
    action_probs = aggregate_rankings_to_action_probs(
        [ranked_action_ids],
        action_ids=action_ids,
        method=method,
        temperature=temperature,
    )
    letter_probs = {
        letter: float(action_probs.get(aid, 0.0)) for letter, aid in letter_map.items()
    }
    # Store Borda/BT scores as pseudo-logprobs for debugging only (not API logprobs).
    if str(method).lower() in {"bt", "bradley_terry", "plackett_luce", "pl"}:
        scores = bradley_terry_scores_from_rankings([ranked_action_ids], action_ids=action_ids)
    else:
        scores = borda_scores_from_rankings([ranked_action_ids], action_ids=action_ids)
    letter_scores = {letter: float(scores.get(aid, 0.0)) for letter, aid in letter_map.items()}
    mode_name = f"ranking_{str(method).strip().lower()}"
    return TeacherActionDistribution(
        state_id=action_set.state_id,
        order_seed=order_seed,
        letter_to_action_id=dict(letter_map),
        logprobs=letter_scores,
        probs=letter_probs,
        action_probs=action_probs,
        raw_response=dict(raw),
        cache_hit=cache_hit,
        teacher_mode=mode_name,
        ranked_action_ids=ranked_action_ids,
    )


def query_teacher_ranking_averaged(
    action_set: CandidateActionSet,
    *,
    state: Mapping[str, Any],
    teacher_fn: TeacherFn,
    order_seeds: Sequence[int],
    cache: ExactRequestCache | None = None,
    method: str = "borda",
    temperature: float = DEFAULT_RANK_TEMPERATURE,
) -> tuple[dict[str, float], list[TeacherActionDistribution]]:
    """Multi-order ranking → Borda/BT aggregation into one action distribution."""
    dists = [
        query_teacher_ranking_distribution(
            action_set,
            state=state,
            teacher_fn=teacher_fn,
            order_seed=int(seed),
            cache=cache,
            method=method,
            temperature=temperature,
        )
        for seed in order_seeds
    ]
    action_ids = [c.action_id for c in action_set.candidates]
    # Prefer truncated letter-mapped ids present in any dist.
    if dists:
        action_ids = sorted({aid for d in dists for aid in d.action_probs})
    rankings = [list(d.ranked_action_ids) for d in dists if d.ranked_action_ids]
    if not rankings:
        return {}, dists
    avg = aggregate_rankings_to_action_probs(
        rankings,
        action_ids=action_ids,
        method=method,
        temperature=temperature,
    )
    return avg, dists


def mock_teacher_ranking_preferring_oracle(request: Mapping[str, Any]) -> dict[str, Any]:
    """Deterministic ranking teacher for unit tests (no network)."""
    letter_map = request.get("letter_map") or {}
    preferred_ids = ("oracle", "student", "choose_best", "select_next")
    inv = {v: k for k, v in letter_map.items()}
    ranked_letters: list[str] = []
    for action_id in preferred_ids:
        letter = inv.get(action_id)
        if letter and letter not in ranked_letters:
            ranked_letters.append(letter)
    for letter in sorted(letter_map):
        if letter not in ranked_letters:
            ranked_letters.append(letter)
    return {
        "ranked_letters": ranked_letters,
        "letter": ranked_letters[0] if ranked_letters else None,
        "teacher": "mock_ranking_prefer_oracle",
        "request_hash": stable_hash(request),
    }


def make_openrouter_ranking_teacher(
    *,
    api_key: str,
    model: str = "deepseek/deepseek-v4-pro",
    api_base: str = "https://openrouter.ai/api/v1/chat/completions",
    timeout_s: int = 120,
    max_tokens: int = 256,
) -> TeacherFn:
    """TeacherFn that requests a full letter ranking as strict JSON (no logprobs)."""
    import requests

    def _teacher(request: Mapping[str, Any]) -> dict[str, Any]:
        messages = list(request.get("messages") or [])
        payload = {
            "model": model,
            "temperature": 0.0,
            "max_tokens": int(max_tokens),
            "messages": messages,
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
                f"ranking teacher HTTP {response.status_code}: {response.text[:500]}"
            )
        body = response.json()
        content = (((body.get("choices") or [{}])[0].get("message") or {}).get("content")) or ""
        parsed = _extract_json_object(str(content)) or {}
        ranked = parsed.get("ranked_letters") or parsed.get("ranked_labels") or []
        if not isinstance(ranked, list):
            ranked = []
        ranked_letters = [str(x).strip().upper()[:1] for x in ranked if str(x).strip()]
        chosen = ranked_letters[0] if ranked_letters else None
        return {
            "letter": chosen,
            "ranked_letters": ranked_letters,
            "content": str(content),
            "teacher": f"openrouter_ranking:{model}",
            "request_hash": stable_hash(request),
            "raw_finish_reason": ((body.get("choices") or [{}])[0].get("finish_reason")),
        }

    return _teacher
