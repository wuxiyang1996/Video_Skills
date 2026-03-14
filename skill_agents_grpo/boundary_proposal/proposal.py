"""
Stage 1 boundary proposal: generate candidate cut points C from multiple signals.

Signals: predicate flips, action surprisal spikes, embedding change-points,
         hard events, **intention-tag changes** (tag proposes, Stage 2 decides).
Output: merged and optionally density-controlled candidate set (centers + windows).
"""

from __future__ import annotations

import re
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

# ---------------------------------------------------------------------------
# Types and config
# ---------------------------------------------------------------------------


@dataclass
class BoundaryCandidate:
    """A single candidate cut point, optionally with a window."""

    center: int
    """Center timestep of the candidate."""
    half_window: int = 0
    """Half-width of allowed boundary window [center - half_window, center + half_window]."""
    source: str = "unknown"
    """Which signal produced this: 'predicate' | 'surprisal' | 'changepoint' | 'event'."""


@dataclass
class ProposalConfig:
    """Configuration for the boundary proposal pipeline."""

    # Merge: merge candidates within this radius (timesteps)
    merge_radius: int = 5
    # Window: half-width for each merged candidate
    window_half_width: int = 2

    # Surprisal
    surprisal_std_factor: float = 2.0
    surprisal_local_radius: int = 3
    surprisal_delta_threshold: Optional[float] = None  # if set, use delta spike

    # Change-point (embedding)
    changepoint_threshold: Optional[float] = None
    changepoint_top_k_per_minute: Optional[int] = None
    changepoint_local_radius: int = 5
    steps_per_minute: int = 60  # for top-K per minute

    # Density: for soft signals, keep at most this many per minute (None = no cap)
    soft_max_per_minute: Optional[int] = 20

    # ── Intention-tag boundary signals ─────────────────────────────
    # Tag change → boundary candidate (tag proposes, Stage 2 decides)
    tag_min_segment_len: int = 3
    tag_completion_bonus_weight: float = 1.5  # stronger event for done/completion
    tag_repeated_filter: bool = True  # ignore A→A non-changes


# ---------------------------------------------------------------------------
# Trigger generation
# ---------------------------------------------------------------------------


def _triggers_from_predicate_flips(
    predicates: list[dict],
) -> list[tuple[int, str]]:
    """
    A) Predicate flips: add t when any key predicate changes from t-1 to t.
    predicates[t] = dict of predicate name -> bool (or value).
    """
    out: list[tuple[int, str]] = []
    for t in range(1, len(predicates)):
        prev, curr = predicates[t - 1], predicates[t]
        if prev is None or curr is None:
            continue
        for k in set(prev) | set(curr):
            if prev.get(k) != curr.get(k):
                out.append((t, "predicate"))
                break
    return out


def _triggers_from_surprisal(
    surprisal: np.ndarray,
    config: ProposalConfig,
) -> list[tuple[int, str]]:
    """
    B) Action surprisal spikes: local maximum, or above mean + k*std, or sharp delta.
    """
    out: list[tuple[int, str]] = []
    T = len(surprisal)
    if T == 0:
        return out

    mean_s = float(np.nanmean(surprisal))
    std_s = float(np.nanstd(surprisal))
    if std_s == 0:
        std_s = 1e-9

    r = config.surprisal_local_radius
    for t in range(T):
        s_t = surprisal[t]
        if np.isnan(s_t):
            continue
        # Strong local maximum
        left = max(0, t - r)
        right = min(T, t + r + 1)
        if np.nanmax(surprisal[left:right]) == s_t and (t == 0 or surprisal[t - 1] != s_t or t == right - 1):
            out.append((t, "surprisal"))
            continue
        # Above threshold
        if s_t >= mean_s + config.surprisal_std_factor * std_s:
            out.append((t, "surprisal"))
            continue
        # Sharp delta
        if config.surprisal_delta_threshold is not None and t >= 1:
            delta = s_t - surprisal[t - 1]
            if not np.isnan(delta) and delta >= config.surprisal_delta_threshold:
                out.append((t, "surprisal"))

    return out


def _triggers_from_changepoint(
    changepoint_scores: np.ndarray,
    config: ProposalConfig,
) -> list[tuple[int, str]]:
    """
    C) Embedding change-point: add t when score is local max above threshold or in top-K per minute.
    """
    out: list[tuple[int, str]] = []
    T = len(changepoint_scores)
    if T == 0:
        return out

    r = config.changepoint_local_radius
    step_per_min = config.steps_per_minute

    for t in range(T):
        cp = changepoint_scores[t]
        if np.isnan(cp):
            continue
        left = max(0, t - r)
        right = min(T, t + r + 1)
        is_local_max = np.nanmax(changepoint_scores[left:right]) == cp

        if config.changepoint_threshold is not None and is_local_max and cp >= config.changepoint_threshold:
            out.append((t, "changepoint"))
        elif config.changepoint_threshold is None and is_local_max:
            out.append((t, "changepoint"))

    if config.changepoint_top_k_per_minute is not None and config.changepoint_top_k_per_minute > 0:
        # Restrict to top-K per minute by score
        by_minute: dict[int, list[tuple[int, float]]] = {}
        for t, src in out:
            if src != "changepoint":
                continue
            m = t // step_per_min
            by_minute.setdefault(m, []).append((t, float(changepoint_scores[t])))
        kept_ts = set()
        for m, cands in by_minute.items():
            cands.sort(key=lambda x: -x[1])
            for t, _ in cands[: config.changepoint_top_k_per_minute]:
                kept_ts.add(t)
        out = [(t, s) for t, s in out if s != "changepoint" or t in kept_ts]

    return out


def _triggers_from_events(
    event_times: list[int],
    event_window: int = 1,
) -> list[tuple[int, str]]:
    """
    D) Hard events: always include these times; optionally add t±event_window.
    """
    out: list[tuple[int, str]] = []
    seen = set()
    for t in event_times:
        for dt in range(-event_window, event_window + 1):
            tt = t + dt
            if tt >= 0 and tt not in seen:
                seen.add(tt)
                out.append((tt, "event"))
    return sorted(out, key=lambda x: x[0])


# ── Tag aliases for canonicalization (subset of signal_extractors) ───

_TAG_RE = re.compile(r"\[(\w+)\]")

_TAG_ALIASES: Dict[str, str] = {
    "PLACE": "SETUP", "DROP": "EXECUTE", "MOVE": "NAVIGATE",
    "SWAP": "EXECUTE", "PUSH": "NAVIGATE", "JUMP": "NAVIGATE",
    "MATCH": "CLEAR", "PLAN": "SETUP", "ARRANGE": "SETUP",
    "ROTATE": "SETUP", "ORGANIZE": "OPTIMIZE", "SCORE": "EXECUTE",
    "PROTECT": "DEFEND", "GRAB": "COLLECT", "FLEE": "SURVIVE",
    "RUN": "NAVIGATE", "CREATE": "BUILD", "FIND": "EXPLORE",
    "FIX": "OPTIMIZE", "ALIGN": "POSITION", "TARGET": "ATTACK",
    "SECURE": "DEFEND", "EXPAND": "ATTACK", "RETREAT": "DEFEND",
}


def _canonicalize_tag(raw: str) -> str:
    """Normalize a tag string: strip brackets, upper, resolve aliases."""
    m = _TAG_RE.match((raw or "").strip())
    token = m.group(1).upper() if m else raw.strip().strip("[]").upper()
    return _TAG_ALIASES.get(token, token)


def _triggers_from_intention_tags(
    intention_tags: list[str],
    config: ProposalConfig,
    done_flags: Optional[list[bool]] = None,
) -> list[tuple[int, str]]:
    """
    E) Intention-tag signal: propose boundaries from tag changes and completions.

    Two sub-signals:
      1. Tag change: if the canonical tag at t differs from t-1, propose t.
         Subject to min_segment_len and repeated-tag filtering.
      2. Tag completion / done: if done_flags[t] is True, propose t as a
         stronger "tag_done" event.

    The tag signal is a *proposal prior* — Stage 2 makes the final decision.
    """
    out: list[tuple[int, str]] = []
    if not intention_tags:
        return out

    T = len(intention_tags)
    canonical = [_canonicalize_tag(t) for t in intention_tags]

    last_change_t = 0

    for t in range(1, T):
        prev_tag = canonical[t - 1]
        curr_tag = canonical[t]

        # Skip UNKNOWN tags
        if curr_tag == "UNKNOWN" or not curr_tag:
            continue

        # --- Signal 1: tag change ---
        if curr_tag != prev_tag:
            if config.tag_repeated_filter and prev_tag == "UNKNOWN":
                pass  # don't count transitions from UNKNOWN
            elif (t - last_change_t) < config.tag_min_segment_len:
                pass  # too rapid, skip
            else:
                out.append((t, "tag_change"))
                last_change_t = t

    # --- Signal 2: tag completion / done ---
    if done_flags is not None:
        for t in range(T):
            if done_flags[t]:
                out.append((t, "tag_done"))

    return out


# ---------------------------------------------------------------------------
# Merge and window
# ---------------------------------------------------------------------------


def _merge_and_window(
    triggers: list[tuple[int, str]],
    merge_radius: int,
    window_half_width: int,
) -> list[BoundaryCandidate]:
    """
    Merge nearby triggers into single candidates; represent as center + half_window.
    """
    if not triggers:
        return []

    triggers = sorted(triggers, key=lambda x: x[0])
    merged: list[tuple[int, list[str]]] = []
    current_center = triggers[0][0]
    current_sources = [triggers[0][1]]

    for t, src in triggers[1:]:
        if t <= current_center + merge_radius:
            current_sources.append(src)
            # extend center toward new trigger (weight by keeping first event)
            current_center = (current_center + t) // 2
        else:
            merged.append((current_center, list(dict.fromkeys(current_sources))))
            current_center = t
            current_sources = [src]
    merged.append((current_center, list(dict.fromkeys(current_sources))))

    return [
        BoundaryCandidate(
            center=c,
            half_window=window_half_width,
            source="+".join(sources),
        )
        for c, sources in merged
    ]


def _density_control(
    candidates: list[BoundaryCandidate],
    config: ProposalConfig,
    steps_per_minute: int = 60,
) -> list[BoundaryCandidate]:
    """
    Optional: keep all event and predicate candidates; for surprisal/changepoint
    keep at most soft_max_per_minute per minute (e.g. top by some score).
    Here we do a simple cap per minute for soft sources (surprisal, changepoint).
    """
    if config.soft_max_per_minute is None:
        return candidates

    hard = [c for c in candidates if "event" in c.source or "predicate" in c.source or "tag" in c.source]
    soft = [c for c in candidates if c not in hard]

    by_minute: dict[int, list[BoundaryCandidate]] = {}
    for c in soft:
        m = c.center // steps_per_minute
        by_minute.setdefault(m, []).append(c)

    soft_kept: list[BoundaryCandidate] = []
    for m in sorted(by_minute.keys()):
        soft_kept.extend(by_minute[m][: config.soft_max_per_minute])

    result = hard + soft_kept
    result.sort(key=lambda c: c.center)
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def propose_boundary_candidates(
    T: int,
    *,
    predicates: Optional[list[dict]] = None,
    surprisal: Optional[np.ndarray] = None,
    changepoint_scores: Optional[np.ndarray] = None,
    event_times: Optional[list[int]] = None,
    intention_tags: Optional[list[str]] = None,
    done_flags: Optional[list[bool]] = None,
    config: Optional[ProposalConfig] = None,
    event_window: int = 1,
) -> list[BoundaryCandidate]:
    """
    Stage 1: produce candidate cut points C for trajectory segmentation.

    Parameters
    ----------
    T : int
        Trajectory length (number of timesteps).
    predicates : list[dict], optional
        predicates[t] = dict of predicate name -> value; flips add candidates.
    surprisal : np.ndarray, optional
        Surprisal s_t = -log p(a_t | x_t); shape (T,).
    changepoint_scores : np.ndarray, optional
        Change-point score per timestep; shape (T,).
    event_times : list[int], optional
        Hard event timesteps (reward spike, death, menu toggle, etc.).
    intention_tags : list[str], optional
        Per-timestep intention tag strings (e.g. ``"[MOVE_TO_ONION]"``).
        Tag changes produce boundary candidates; tag completion / done
        produces stronger boundary events.  Subject to
        ``tag_min_segment_len`` and repeated-tag filtering.
    done_flags : list[bool], optional
        Per-timestep completion flags.  When True at timestep *t*, the
        tag-done signal emits a stronger boundary event.
    config : ProposalConfig, optional
        If None, uses ProposalConfig() defaults.
    event_window : int
        For each event time t, also add t±event_window (default 1).

    Returns
    -------
    list[BoundaryCandidate]
        Sorted by center. Each candidate has center, half_window, and source.
        Downstream DP/HSMM/beam search should only place boundaries at
        times in [c.center - c.half_window, c.center + c.half_window] for some c.
    """
    cfg = config or ProposalConfig()
    triggers: list[tuple[int, str]] = []

    if predicates is not None and len(predicates) > 0:
        triggers.extend(_triggers_from_predicate_flips(predicates))

    if surprisal is not None and len(surprisal) > 0:
        triggers.extend(_triggers_from_surprisal(surprisal, cfg))

    if changepoint_scores is not None and len(changepoint_scores) > 0:
        triggers.extend(_triggers_from_changepoint(changepoint_scores, cfg))

    if event_times is not None:
        triggers.extend(_triggers_from_events(event_times, event_window=event_window))

    if intention_tags is not None and len(intention_tags) > 0:
        triggers.extend(
            _triggers_from_intention_tags(intention_tags, cfg, done_flags)
        )

    triggers = [(t, s) for t, s in triggers if 0 <= t < T]
    if not triggers:
        return []

    candidates = _merge_and_window(
        triggers,
        merge_radius=cfg.merge_radius,
        window_half_width=cfg.window_half_width,
    )
    candidates = _density_control(candidates, cfg, steps_per_minute=cfg.steps_per_minute)
    return candidates


def candidate_centers_only(candidates: list[BoundaryCandidate]) -> list[int]:
    """Return just the center times C = {c1, c2, ...} for use as cut set."""
    return sorted(set(c.center for c in candidates))


def candidate_windows(candidates: list[BoundaryCandidate]) -> list[tuple[int, int]]:
    """Return (start, end) inclusive windows for each candidate."""
    return [
        (c.center - c.half_window, c.center + c.half_window)
        for c in candidates
    ]
