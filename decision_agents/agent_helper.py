# This file defines helper functions for the VLM decision agent: state summarization,
# intention inference, episodic memory store, and skill-bank formatting.
#
# The state-summary pipeline produces compact key=value summaries optimised for
# LLM/VLM context windows, retrieval, skill-bank indexing, and trajectory
# segmentation.  Summaries are *not* human-readable paragraphs — they are short
# structured abstractions of the current game state.

from __future__ import annotations

import re
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

try:
    from API_func import ask_model
except ImportError:
    ask_model = None


# ---------------------------------------------------------------------------
# Summary budget constants
# ---------------------------------------------------------------------------

DEFAULT_SUMMARY_CHAR_BUDGET: int = 400
"""Default budget for state summaries (characters).  Prefer ~220-380 when
possible; treat 400 as the hard cap, not a target to fill."""

HARD_SUMMARY_CHAR_LIMIT: int = 400
"""Absolute upper bound.  No summary should exceed this."""


# ---------------------------------------------------------------------------
# Boilerplate patterns to strip from raw observations
# ---------------------------------------------------------------------------

_BOILERPLATE_RE = re.compile(
    r"(?i)"
    r"(choose one action[^\n]*"
    r"|valid actions?[^\n]*"
    r"|possible actions?[^\n]*"
    r"|examples?[:\s][^\n]*"
    r"|respond with[^\n]*"
    r"|reply with[^\n]*"
    r"|submit your orders[^\n]*"
    r"|output format[^\n]*"
    r"|order format[^\n]*"
    r"|--- order format ---[^\n]*"
    r"|  hold:[^\n]*"
    r"|  move:[^\n]*"
    r"|  support hold:[^\n]*"
    r"|  support move:[^\n]*"
    r"|  convoy:[^\n]*"
    r"|  retreat:[^\n]*"
    r"|  disband:[^\n]*"
    r"|  build:[^\n]*"
    r"|example:?\s*\[?\"[A-Z]\s[A-Z]{3}[^\n]*"
    r")"
)

# Keys to prioritise when compressing a structured state dict
_PRIORITY_KEYS = (
    "game", "phase", "subgoal", "objective", "self", "ally", "enemy",
    "critical", "resources", "orders", "inventory", "progress",
    "time_left", "affordance", "delta", "valid_actions",
)


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _safe_str(x: Any) -> str:
    """Coerce *x* to a short string suitable for a summary slot."""
    if x is None:
        return ""
    if isinstance(x, str):
        return x.strip()
    if isinstance(x, (list, tuple)):
        if len(x) == 0:
            return ""
        items = [_safe_str(i) for i in x[:6]]
        out = ",".join(items)
        if len(x) > 6:
            out += f"..+{len(x) - 6}"
        return out
    if isinstance(x, dict):
        parts = [f"{k}:{_safe_str(v)}" for k, v in list(x.items())[:4]]
        return "{" + ",".join(parts) + "}"
    return str(x).strip()


def _remove_boilerplate(obs: str) -> str:
    """Strip action-formatting instructions and boilerplate from *obs*."""
    cleaned = _BOILERPLATE_RE.sub("", obs)
    # collapse blank lines
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _truncate_keep_important(text: str, max_chars: int) -> str:
    """Truncate *text* to *max_chars* keeping the most information-dense prefix.

    Prefers cutting at a sentence / clause boundary when possible.
    """
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    # try to cut at last sentence-end
    for sep in (". ", "| ", "; ", ", ", " "):
        pos = cut.rfind(sep)
        if pos > max_chars // 2:
            return cut[:pos + len(sep)].rstrip()
    return cut.rstrip()


def _join_kv(parts: List[Tuple[str, str]], max_chars: int) -> str:
    """Join ``(key, value)`` pairs as ``key=value | key=value | ...``.

    Drops trailing pairs if the result would exceed *max_chars*.
    """
    segments: List[str] = []
    length = 0
    for key, val in parts:
        if not val:
            continue
        seg = f"{key}={val}"
        added = len(seg) + (3 if segments else 0)  # " | " separator
        if length + added > max_chars:
            break
        segments.append(seg)
        length += added
    return " | ".join(segments)


# ---------------------------------------------------------------------------
# Structured-state compressor
# ---------------------------------------------------------------------------

def compact_structured_state(
    structured_state: Dict[str, Any],
    max_chars: int = DEFAULT_SUMMARY_CHAR_BUDGET,
) -> str:
    """Compress a structured state dict into a compact ``key=value`` summary.

    *structured_state* should be a flat-ish dict produced by an env wrapper's
    ``build_structured_state_summary()``.  Keys listed in ``_PRIORITY_KEYS``
    are emitted first; remaining keys are appended if budget allows.

    Returns:
        A string of at most *max_chars* characters.
    """
    max_chars = min(max_chars, HARD_SUMMARY_CHAR_LIMIT)
    if not structured_state:
        return ""

    ordered: List[Tuple[str, str]] = []
    seen = set()
    for k in _PRIORITY_KEYS:
        if k in structured_state:
            ordered.append((k, _safe_str(structured_state[k])))
            seen.add(k)
    for k, v in structured_state.items():
        if k not in seen:
            ordered.append((k, _safe_str(v)))

    return _join_kv(ordered, max_chars)


# ---------------------------------------------------------------------------
# Text-observation compressor
# ---------------------------------------------------------------------------

def compact_text_observation(
    observation: str,
    max_chars: int = DEFAULT_SUMMARY_CHAR_BUDGET,
) -> str:
    """Deterministically compress a raw text observation into a short summary.

    Steps:
      1. Strip boilerplate / action-format instructions.
      2. Split into clauses.
      3. Keep the most informative clauses that fit within *max_chars*.

    Returns:
        A string of at most *max_chars* characters.  Never returns the raw
        observation verbatim (even if it is already short).
    """
    max_chars = min(max_chars, HARD_SUMMARY_CHAR_LIMIT)
    if not observation or not isinstance(observation, str):
        return ""

    text = _remove_boilerplate(observation)
    if not text:
        return ""

    # Normalise whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)

    # Split into clauses (sentences, newlines, pipe-delimited)
    clauses = re.split(r"[.\n|]+", text)
    clauses = [c.strip() for c in clauses if c.strip() and len(c.strip()) > 2]

    if not clauses:
        return _truncate_keep_important(text, max_chars)

    # Heuristic: drop lines that are purely decorative (=== headers, --- separators)
    clauses = [c for c in clauses if not re.match(r"^[-=]{3,}", c)]

    # Build output greedily, keeping as many clauses as fit
    parts: List[str] = []
    length = 0
    for c in clauses:
        needed = len(c) + (3 if parts else 0)  # " | " separator
        if length + needed > max_chars:
            break
        parts.append(c)
        length += needed

    if not parts:
        # Single long clause — truncate it
        return _truncate_keep_important(clauses[0], max_chars)

    return " | ".join(parts)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def get_state_summary(
    observation: str,
    structured_state: Optional[Dict[str, Any]] = None,
    *,
    max_chars: int = DEFAULT_SUMMARY_CHAR_BUDGET,
    use_llm_fallback: bool = False,
    llm_callable: Optional[Callable[..., str]] = None,
    # Legacy keyword args kept for backward compatibility
    game: Optional[str] = None,
    model: Optional[str] = None,
) -> str:
    """Produce a compact state summary for agent context / retrieval / memory.

    Priority order:
      1. ``structured_state`` → ``compact_structured_state()``
      2. ``observation``      → ``compact_text_observation()``
      3. LLM fallback (only if *use_llm_fallback* is True)

    The result is **never** the raw observation verbatim and always respects
    *max_chars* (capped at ``HARD_SUMMARY_CHAR_LIMIT``).

    Args:
        observation: Raw text observation from the environment.
        structured_state: Optional dict from wrapper's
            ``build_structured_state_summary()``.
        max_chars: Soft character budget (default 220).
        use_llm_fallback: If True and deterministic compression is very lossy,
            attempt an LLM-based summarisation.  Disabled by default.
        llm_callable: Custom LLM callable ``(prompt, **kw) -> str``.
            Falls back to ``ask_model`` if available.
        game: (legacy) Game hint — ignored by deterministic path.
        model: (legacy) Model name for LLM fallback.

    Returns:
        Compact summary string, always ≤ ``HARD_SUMMARY_CHAR_LIMIT`` chars.
    """
    max_chars = min(max_chars, HARD_SUMMARY_CHAR_LIMIT)

    # --- Path 1: structured state ---
    if structured_state:
        summary = compact_structured_state(structured_state, max_chars)
        if summary:
            return summary

    # --- Path 2: deterministic text compression ---
    summary = compact_text_observation(observation, max_chars)
    if summary:
        return summary

    # --- Path 3 (optional): LLM fallback ---
    if use_llm_fallback:
        _llm = llm_callable or ask_model
        if _llm is not None:
            obs_slice = (observation or "")[:3000]
            prompt = (
                "Compress this game state into a compact key=value summary. "
                f"Max {max_chars} characters. No prose. "
                "Format: key=value | key=value | ...\n\n" + obs_slice
            )
            try:
                result = _llm(prompt, model=model or "gpt-4o-mini",
                              temperature=0.0, max_tokens=200)
                if result:
                    return _truncate_keep_important(result.strip(), max_chars)
            except Exception:
                pass

    return ""

# ---------------------------------------------------------------------------
# Intention inference
# ---------------------------------------------------------------------------

def infer_intention(
    summary_or_observation: str,
    game: Optional[str] = None,
    model: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Infer current intention (short objective/subgoal) from state summary or observation.
    context can include: last_actions, progress_notes, task description.
    """
    if not summary_or_observation or not isinstance(summary_or_observation, str):
        return "Explore and survive."
    if ask_model is None:
        return "Complete objective."
    ctx = context or {}
    extra = ""
    if ctx.get("last_actions"):
        extra += "\nRecent actions: " + ", ".join(str(a) for a in ctx["last_actions"][-3:])
    if ctx.get("progress_notes"):
        extra += "\nProgress: " + " | ".join(ctx["progress_notes"])
    if ctx.get("task"):
        extra += "\nEpisode task: " + str(ctx["task"])
    prompt = (
        "Given this game state, output a single short phrase (under 15 words) "
        "describing the agent's current objective or subgoal. Be specific (e.g. 'Reach checkpoint', 'Avoid sniper and heal').\n\n"
        "State:\n" + summary_or_observation[:2500] + extra + "\n\nIntention phrase:"
    )
    out = ask_model(prompt, model=model or "gpt-4o-mini", temperature=0.2, max_tokens=80)
    return (out or "Complete objective.").strip()[:200]


# ---------------------------------------------------------------------------
# Episodic memory store (for query_memory)
# ---------------------------------------------------------------------------

class EpisodicMemoryStore:
    """Episodic memory with RAG-embedding retrieval (cosine similarity) and
    keyword-overlap fallback.

    When an ``embedder`` is provided (a ``TextEmbedderBase`` from
    ``rag.embedding``), every memory is embedded on ``add`` and queries are
    scored via cosine similarity.  The final score is a weighted mix of
    embedding similarity and keyword overlap so the system degrades
    gracefully if the embedding model is unavailable.

    When no embedder is provided, behaviour is identical to the original
    keyword-overlap-only store.
    """

    def __init__(
        self,
        max_entries: int = 500,
        embedder: Any = None,
        embedding_weight: float = 0.7,
    ) -> None:
        """
        Args:
            max_entries: Maximum number of memories to keep (FIFO eviction).
            embedder: Optional ``TextEmbedderBase`` (e.g. from
                ``rag.get_text_embedder()``).  Enables embedding retrieval.
            embedding_weight: Blend weight for embedding vs keyword score
                (0 = keyword only, 1 = embedding only).
        """
        self._entries: List[Dict[str, Any]] = []
        self._max_entries = max_entries
        self._embedder = embedder
        self._embedding_weight = embedding_weight
        self._memory_store: Any = None
        if embedder is not None:
            self._init_memory_store(embedder)

    def _init_memory_store(self, embedder: Any) -> None:
        try:
            from rag.retrieval import MemoryStore
            self._memory_store = MemoryStore(embedder=embedder, top_k=self._max_entries)
        except ImportError:
            self._memory_store = None

    def add(
        self,
        key: str,
        summary: str,
        action: Any = None,
        outcome: Any = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add one memory entry and embed it if an embedder is available."""
        entry = {
            "key": key,
            "summary": summary,
            "action": action,
            "outcome": outcome,
            **(extra or {}),
        }
        self._entries.append(entry)
        if len(self._entries) > self._max_entries:
            self._entries = self._entries[-self._max_entries:]

        if self._memory_store is not None:
            text = (key + " " + summary).strip()
            if text:
                try:
                    self._memory_store.add_texts([text], payloads=[entry])
                except Exception:
                    pass

    def add_experience(self, state_summary: str, action: Any, next_state_summary: str, done: bool) -> None:
        """Convenience: add from a single experience."""
        key = state_summary[:200] if state_summary else ""
        self.add(key=key, summary=state_summary, action=action, outcome=next_state_summary, extra={"done": done})

    def query(self, query_key: str, k: int = 3) -> List[Dict[str, Any]]:
        """Retrieve top-k memories by embedding similarity + keyword overlap.

        If an embedder is available, scores are a weighted blend of cosine
        similarity and keyword overlap.  Otherwise falls back to keyword only.
        """
        if not query_key or not self._entries:
            return []

        keyword_scores = self._keyword_scores(query_key)

        if self._memory_store is not None and len(self._memory_store) > 0:
            try:
                ranked = self._memory_store.rank(query_key, k=len(self._entries))
                emb_scores = {idx: score for idx, score, _ in ranked}
            except Exception:
                emb_scores = {}
        else:
            emb_scores = {}

        w = self._embedding_weight if emb_scores else 0.0
        scored: List[Tuple[float, Dict[str, Any]]] = []
        for i, entry in enumerate(self._entries):
            kw = keyword_scores[i]
            emb = emb_scores.get(i, 0.0)
            combined = w * emb + (1.0 - w) * kw
            scored.append((combined, entry))

        scored.sort(key=lambda x: -x[0])
        return [e for _, e in scored[:k]]

    def _keyword_scores(self, query_key: str) -> List[float]:
        q_lower = query_key.lower()
        q_words = set(w for w in q_lower.split() if len(w) >= 2)
        scores: List[float] = []
        for e in self._entries:
            text = (e.get("key", "") + " " + e.get("summary", "")).lower()
            t_words = set(w for w in text.split() if len(w) >= 2)
            overlap = len(q_words & t_words) / max(len(q_words), 1)
            scores.append(overlap)
        return scores

    @property
    def has_embedder(self) -> bool:
        return self._memory_store is not None

    def __len__(self) -> int:
        return len(self._entries)


# ---------------------------------------------------------------------------
# Skill bank formatting for agent prompt
# ---------------------------------------------------------------------------

def skill_bank_to_text(skill_bank: Any) -> str:
    """
    Format skill bank for inclusion in agent prompt (skill_ids + short effect summary).
    skill_bank can be SkillBankMVP, SkillBankAgent, or any object with
    .skill_ids and .get_contract(skill_id).
    """
    if skill_bank is None:
        return "(no skill bank)"

    # SkillBankAgent wraps a SkillBankMVP; unwrap if needed
    bank = getattr(skill_bank, "bank", skill_bank)

    try:
        ids = list(bank.skill_ids)[:50]
    except AttributeError:
        return "(no skill bank)"
    if not ids:
        return "(empty skill bank)"

    lines = [f"Available skills ({len(ids)}): " + ", ".join(ids)]
    for sid in ids[:15]:
        try:
            c = bank.get_contract(sid)
            if c is not None:
                add = getattr(c, "eff_add", set()) or set()
                dele = getattr(c, "eff_del", set()) or set()
                add_preview = ", ".join(sorted(add)[:3])
                parts = [f"add({len(add)})", f"del({len(dele)})"]
                if add_preview:
                    parts.append(f"e.g. {add_preview}")
                r = bank.get_report(sid) if hasattr(bank, "get_report") else None
                if r is not None:
                    parts.append(f"pass={r.overall_pass_rate:.0%}")
                lines.append(f"  - {sid}: {', '.join(parts)}")
        except Exception:
            lines.append(f"  - {sid}")
    return "\n".join(lines)


def query_skill_bank(skill_bank: Any, key: str, top_k: int = 1) -> Dict[str, Any]:
    """Query the skill bank and return a result compatible with the QUERY_SKILL tool.

    Supports SkillBankAgent (rich query), SkillQueryEngine, and plain SkillBankMVP
    (fallback to name matching).

    Returns ``{"skill_id": str|None, "micro_plan": list[dict], ...}``.
    """
    if skill_bank is None:
        return {"skill_id": None, "micro_plan": []}

    # SkillBankAgent has .query_skill()
    if hasattr(skill_bank, "query_skill"):
        results = skill_bank.query_skill(key, top_k=top_k)
        if results:
            best = results[0]
            return {
                "skill_id": best.get("skill_id"),
                "micro_plan": best.get("micro_plan", []) or [{"action": "proceed"}],
                "contract": best.get("contract", {}),
            }
        return {"skill_id": None, "micro_plan": []}

    # SkillQueryEngine
    if hasattr(skill_bank, "query_for_decision_agent"):
        return skill_bank.query_for_decision_agent(key, top_k=top_k)

    # Fallback: plain SkillBankMVP or similar — name match
    bank = getattr(skill_bank, "bank", skill_bank)
    try:
        ids = list(bank.skill_ids)
    except AttributeError:
        return {"skill_id": None, "micro_plan": []}

    key_lower = key.lower()
    skill_id = None
    for sid in ids:
        if sid.lower() in key_lower or key_lower in sid.lower():
            skill_id = sid
            break
    if skill_id is None and ids:
        skill_id = ids[0]

    if skill_id:
        c = bank.get_contract(skill_id)
        if c:
            add_set = getattr(c, "eff_add", set()) or set()
            steps = [{"action": None, "effect": lit} for lit in sorted(add_set)[:5]]
            return {"skill_id": skill_id, "micro_plan": steps or [{"action": "proceed"}]}

    return {"skill_id": None, "micro_plan": []}
