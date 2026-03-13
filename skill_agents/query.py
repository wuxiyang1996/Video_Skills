"""
Skill query engine — evolving from pure retriever to skill selection policy.

The query layer exposes two conceptual dimensions to the decision agent:

  1. **Retrieval relevance** — how well a skill's description / effects match
     the query (embedding similarity + keyword Jaccard).
  2. **Execution applicability** — how confident we are that the skill can
     actually execute successfully in the current state (contract match,
     verification pass rate, supporting evidence).

Decision agents should consume the rich ``SkillSelectionResult`` returned by
``select()`` rather than the flat dicts from ``query()``.  The old ``query()``
API is preserved for backward compatibility.

Usage::

    from skill_agents.query import SkillQueryEngine
    from skill_agents.skill_bank.bank import SkillBankMVP

    bank = SkillBankMVP("bank.jsonl"); bank.load()
    engine = SkillQueryEngine(bank)

    # Simple retrieval (backward compatible)
    results = engine.query("navigate to pot and place onion", top_k=3)

    # Rich skill selection for decision agents
    selection = engine.select(
        query="navigate to pot",
        current_state={"near_pot": 0.1, "holding_onion": 0.9},
        top_k=3,
    )
    for r in selection:
        print(r.skill_id, r.relevance, r.applicability, r.confidence)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

import numpy as np

from skill_agents.skill_bank.bank import SkillBankMVP, _effects_compat_score
from skill_agents.stage3_mvp.schemas import SkillEffectsContract, VerificationReport


def _tokenize(text: str) -> Set[str]:
    """Lowercase tokenisation for keyword overlap scoring."""
    return {w for w in re.split(r"[^a-zA-Z0-9]+", text.lower()) if len(w) >= 2}


def _jaccard(a: Set[str], b: Set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _effect_set_for_contract(c: SkillEffectsContract) -> Set[str]:
    return (c.eff_add or set()) | (c.eff_del or set()) | (c.eff_event or set())


def _contract_summary(c: SkillEffectsContract) -> Dict[str, Any]:
    return {
        "skill_id": c.skill_id,
        "version": c.version,
        "eff_add": sorted(c.eff_add) if c.eff_add else [],
        "eff_del": sorted(c.eff_del) if c.eff_del else [],
        "eff_event": sorted(c.eff_event) if c.eff_event else [],
        "n_literals": c.total_literals,
        "n_instances": c.n_instances,
    }


def _skill_description(
    sid: str,
    c: Optional[SkillEffectsContract],
    skill: Optional[Any] = None,
) -> str:
    """Build a textual description of a skill for embedding.

    When a ``Skill`` wrapper is available, includes strategic_description
    and protocol preconditions for richer semantic matching.
    """
    parts = [sid.replace("_", " ")]
    if skill is not None:
        if getattr(skill, "name", None):
            parts.append(skill.name)
        if getattr(skill, "strategic_description", None):
            parts.append(skill.strategic_description)
        protocol = getattr(skill, "protocol", None)
        if protocol is not None:
            for pc in getattr(protocol, "preconditions", [])[:5]:
                parts.append(pc)
            for step in getattr(protocol, "steps", [])[:3]:
                parts.append(step)
    if c is not None:
        if getattr(c, "name", None) and not (skill and getattr(skill, "name", None)):
            parts.append(c.name)
        if getattr(c, "description", None) and not (skill and getattr(skill, "strategic_description", None)):
            parts.append(c.description)
        for lit in sorted(c.eff_add or set()):
            parts.append(lit)
        for lit in sorted(c.eff_del or set()):
            parts.append(lit)
        for lit in sorted(c.eff_event or set()):
            parts.append(lit)
    return " ".join(parts)


# ── Rich result type for decision agents ─────────────────────────────

@dataclass
class SkillSelectionResult:
    """Rich result from the skill selection policy.

    Decision agents should use these fields to decide which skill to execute:
      - ``relevance``: how well the skill matches the query (0-1).
      - ``applicability``: how well the current state matches the skill's
        contract / preconditions (approx. -1 to +1).
      - ``confidence``: combined score incorporating pass rate and evidence.
      - ``matched_effects``: which contract effects match the desired outcome.
      - ``missing_effects``: expected effects not matched by the current state.
    """

    skill_id: str
    relevance: float = 0.0
    applicability: float = 0.0
    confidence: float = 0.0
    contract_match_score: float = 0.0
    pass_rate: Optional[float] = None
    n_instances: int = 0
    matched_effects: List[str] = field(default_factory=list)
    missing_effects: List[str] = field(default_factory=list)
    contract: Dict[str, Any] = field(default_factory=dict)
    micro_plan: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "skill_id": self.skill_id,
            "relevance": round(self.relevance, 4),
            "applicability": round(self.applicability, 4),
            "confidence": round(self.confidence, 4),
            "contract_match_score": round(self.contract_match_score, 4),
            "pass_rate": round(self.pass_rate, 3) if self.pass_rate is not None else None,
            "n_instances": self.n_instances,
            "matched_effects": self.matched_effects,
            "missing_effects": self.missing_effects,
            "contract": self.contract,
            "micro_plan": self.micro_plan,
        }


class SkillQueryEngine:
    """Query interface over a SkillBankMVP with RAG embedding support.

    Supports:
    - **embedding query**: cosine similarity over skill descriptions via RAG.
    - **keyword query**: match query tokens against skill ID and effect literals.
    - **effect query**: find skills whose effects overlap with desired state changes.
    - **select**: rich skill selection policy combining relevance + applicability.
    - **list / detail**: enumerate skills or inspect one in depth.

    The ``select()`` method is the preferred API for decision agents.  The old
    ``query()`` API is preserved for backward compatibility.
    """

    def __init__(
        self,
        bank: SkillBankMVP,
        embedder: Any = None,
        embedding_weight: float = 0.6,
    ) -> None:
        self._bank = bank
        self._embedding_weight = embedding_weight
        self._embedder = embedder
        if self._embedder is None:
            try:
                from rag import get_text_embedder
                self._embedder = get_text_embedder()
            except Exception:
                self._embedder = None
        self._skill_embeddings: Optional[np.ndarray] = None
        self._skill_id_order: List[str] = []
        self._build_index()

    def _build_index(self) -> None:
        """Pre-tokenise skill IDs/effects and embed skill descriptions."""
        self._id_tokens: Dict[str, Set[str]] = {}
        self._effect_tokens: Dict[str, Set[str]] = {}
        self._effect_sets: Dict[str, Set[str]] = {}
        self._strategic_tokens: Dict[str, Set[str]] = {}

        descs: List[str] = []
        self._skill_id_order = list(self._bank.skill_ids)

        has_get_skill = hasattr(self._bank, "get_skill")

        for sid in self._skill_id_order:
            self._id_tokens[sid] = _tokenize(sid)
            c = self._bank.get_contract(sid)
            skill = self._bank.get_skill(sid) if has_get_skill else None

            if c is not None:
                eff = _effect_set_for_contract(c)
                self._effect_sets[sid] = eff
                self._effect_tokens[sid] = set()
                for lit in eff:
                    self._effect_tokens[sid] |= _tokenize(lit)
            else:
                self._effect_sets[sid] = set()
                self._effect_tokens[sid] = set()

            strat_text = ""
            if skill is not None:
                strat_text = " ".join(filter(None, [
                    getattr(skill, "name", ""),
                    getattr(skill, "strategic_description", ""),
                ]))
            self._strategic_tokens[sid] = _tokenize(strat_text) if strat_text else set()

            descs.append(_skill_description(sid, c, skill))

        if self._embedder is not None and descs:
            try:
                self._skill_embeddings = self._embedder.encode(
                    descs, prompt_name="passage",
                )
            except Exception:
                self._skill_embeddings = None

    def rebuild_index(self) -> None:
        """Re-index after the skill bank has been mutated."""
        self._build_index()

    @property
    def has_embedder(self) -> bool:
        return self._skill_embeddings is not None

    def _embedding_scores(self, query: str) -> Optional[np.ndarray]:
        """Return per-skill cosine similarity scores for *query*, or None."""
        if self._embedder is None or self._skill_embeddings is None:
            return None
        try:
            q_emb = self._embedder.encode(query, prompt_name="query")
            q_emb = np.atleast_2d(q_emb).astype(np.float32)
            scores = (q_emb @ self._skill_embeddings.T).squeeze(0)
            return scores
        except Exception:
            return None

    # ── Retrieval relevance (internal) ───────────────────────────────

    def _compute_relevance(self, query: str) -> Dict[str, float]:
        """Compute retrieval relevance for all skills given a query.

        Incorporates strategic_description and protocol precondition tokens
        when available for richer semantic matching.
        """
        q_tokens = _tokenize(query)
        emb_scores = self._embedding_scores(query)
        w = self._embedding_weight if emb_scores is not None else 0.0

        scores: Dict[str, float] = {}
        for i, sid in enumerate(self._skill_id_order):
            id_score = _jaccard(q_tokens, self._id_tokens.get(sid, set()))
            eff_score = _jaccard(q_tokens, self._effect_tokens.get(sid, set()))
            strat_score = _jaccard(q_tokens, self._strategic_tokens.get(sid, set()))
            # Blend: strategic > id > effects
            kw_score = 0.35 * strat_score + 0.35 * id_score + 0.30 * eff_score
            emb = float(emb_scores[i]) if emb_scores is not None else 0.0
            scores[sid] = w * emb + (1.0 - w) * kw_score
        return scores

    # ── Execution applicability (internal) ───────────────────────────

    def _compute_applicability(
        self,
        sid: str,
        current_state: Optional[Dict[str, float]] = None,
    ) -> tuple:
        """Compute applicability score and effect match details for a skill.

        Returns (applicability_score, matched_effects, missing_effects).
        """
        c = self._bank.get_contract(sid)
        if c is None:
            return 0.0, [], []

        if current_state is None:
            # Without state info, rely on pass rate as a proxy
            r = self._bank.get_report(sid)
            pass_rate = r.overall_pass_rate if r else 0.5
            return pass_rate - 0.5, [], []  # center around 0

        # Use the effects compat scorer
        compat = _effects_compat_score(c, current_state, current_state)

        matched = []
        missing = []
        for lit in (c.eff_add or set()):
            val = current_state.get(lit)
            if val is not None:
                matched.append(lit)
            else:
                missing.append(lit)
        for lit in (c.eff_del or set()):
            val = current_state.get(lit)
            if val is not None:
                matched.append(lit)
            else:
                missing.append(lit)

        return compat, sorted(matched), sorted(missing)

    def _compute_confidence(
        self,
        sid: str,
        relevance: float,
        applicability: float,
    ) -> float:
        """Combined confidence blending relevance, applicability, and pass rate.

        Confidence = w_rel * relevance + w_app * norm_applicability + w_pr * pass_rate
        """
        r = self._bank.get_report(sid)
        pass_rate = r.overall_pass_rate if r else 0.5

        norm_app = (applicability + 1.0) / 2.0  # map [-1,1] to [0,1]

        w_rel, w_app, w_pr = 0.4, 0.35, 0.25
        return w_rel * relevance + w_app * norm_app + w_pr * pass_rate

    # ── Rich selection API (preferred for decision agents) ───────────

    def select(
        self,
        query: str,
        current_state: Optional[Dict[str, float]] = None,
        top_k: int = 3,
    ) -> List[SkillSelectionResult]:
        """Rich skill selection combining retrieval relevance with execution
        applicability.

        This is the preferred API for decision agents.  It separates "is this
        skill relevant to what I want?" from "can this skill execute now?"
        and provides supporting evidence.

        Parameters
        ----------
        query : str
            Natural-language description of the desired action/goal.
        current_state : dict, optional
            Current predicate state as ``{predicate: probability}``.
            When provided, enables contract-based applicability scoring.
        top_k : int
            Number of results to return.

        Returns
        -------
        list[SkillSelectionResult]
            Sorted by confidence (highest first).
        """
        relevance_scores = self._compute_relevance(query)

        results: List[SkillSelectionResult] = []
        for sid in self._skill_id_order:
            rel = relevance_scores.get(sid, 0.0)
            app, matched, missing = self._compute_applicability(sid, current_state)
            conf = self._compute_confidence(sid, rel, app)

            c = self._bank.get_contract(sid)
            r = self._bank.get_report(sid)

            # Build micro_plan from protocol steps when available
            has_get_skill = hasattr(self._bank, "get_skill")
            skill = self._bank.get_skill(sid) if has_get_skill else None
            if skill is not None and skill.protocol.steps:
                micro_plan = [{"action": step} for step in skill.protocol.steps[:7]]
            elif c:
                micro_plan = [
                    {"action": None, "effect": lit}
                    for lit in sorted(c.eff_add or set())[:7]
                ]
            else:
                micro_plan = []

            result = SkillSelectionResult(
                skill_id=sid,
                relevance=rel,
                applicability=app,
                confidence=conf,
                contract_match_score=app,
                pass_rate=r.overall_pass_rate if r else None,
                n_instances=c.n_instances if c else 0,
                matched_effects=matched,
                missing_effects=missing,
                contract=_contract_summary(c) if c else {},
                micro_plan=micro_plan,
            )
            results.append(result)

        results.sort(key=lambda r: -r.confidence)
        return results[:top_k]

    # ── Backward-compatible query API ────────────────────────────────

    def query(self, key: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Score every skill against a natural-language query key.

        Backward-compatible API.  For richer results, use ``select()``.
        """
        q_tokens = _tokenize(key)
        if not q_tokens:
            return self.list_all()[:top_k]

        emb_scores = self._embedding_scores(key)
        w = self._embedding_weight if emb_scores is not None else 0.0

        scored: List[tuple] = []
        for i, sid in enumerate(self._skill_id_order):
            id_score = _jaccard(q_tokens, self._id_tokens.get(sid, set()))
            eff_score = _jaccard(q_tokens, self._effect_tokens.get(sid, set()))
            kw_score = 0.6 * id_score + 0.4 * eff_score
            emb = float(emb_scores[i]) if emb_scores is not None else 0.0
            total = w * emb + (1.0 - w) * kw_score
            scored.append((total, sid))

        scored.sort(key=lambda x: -x[0])
        results: List[Dict[str, Any]] = []
        for score, sid in scored[:top_k]:
            c = self._bank.get_contract(sid)
            entry: Dict[str, Any] = {
                "skill_id": sid,
                "score": round(score, 4),
            }
            if c is not None:
                entry["contract"] = _contract_summary(c)
                entry["micro_plan"] = [
                    {"action": None, "effect": lit}
                    for lit in sorted(c.eff_add or set())[:7]
                ]
            else:
                entry["contract"] = {}
                entry["micro_plan"] = []
            results.append(entry)
        return results

    # ── Effect-based query ───────────────────────────────────────────

    def query_by_effects(
        self,
        desired_add: Optional[set] = None,
        desired_del: Optional[set] = None,
        top_k: int = 3,
    ) -> List[Dict[str, Any]]:
        """Find skills whose contract effects best match desired state changes."""
        query_set = (desired_add or set()) | (desired_del or set())
        if not query_set:
            return self.list_all()[:top_k]

        query_text = " ".join(sorted(query_set))
        emb_scores = self._embedding_scores(query_text)
        w = self._embedding_weight * 0.5 if emb_scores is not None else 0.0

        scored: List[tuple] = []
        for i, sid in enumerate(self._skill_id_order):
            eff = self._effect_sets.get(sid, set())
            sim = _jaccard(query_set, eff)
            if desired_add:
                c = self._bank.get_contract(sid)
                if c and c.eff_add:
                    add_overlap = len((desired_add or set()) & c.eff_add) / max(len(desired_add or set()), 1)
                    sim = 0.5 * sim + 0.5 * add_overlap
            emb = float(emb_scores[i]) if emb_scores is not None else 0.0
            total = w * emb + (1.0 - w) * sim
            scored.append((total, sid))

        scored.sort(key=lambda x: -x[0])
        results: List[Dict[str, Any]] = []
        for score, sid in scored[:top_k]:
            c = self._bank.get_contract(sid)
            entry: Dict[str, Any] = {
                "skill_id": sid,
                "effect_match_score": round(score, 4),
            }
            if c is not None:
                entry["contract"] = _contract_summary(c)
            results.append(entry)
        return results

    # ── List / detail ────────────────────────────────────────────────

    def list_all(self) -> List[Dict[str, Any]]:
        """Return a compact summary for every skill in the bank."""
        entries: List[Dict[str, Any]] = []
        for sid in self._bank.skill_ids:
            c = self._bank.get_contract(sid)
            r = self._bank.get_report(sid)
            entry: Dict[str, Any] = {"skill_id": sid}
            if c is not None:
                entry["version"] = c.version
                if getattr(c, "name", None):
                    entry["name"] = c.name
                if getattr(c, "description", None):
                    entry["description"] = c.description
                entry["n_add"] = len(c.eff_add) if c.eff_add else 0
                entry["n_del"] = len(c.eff_del) if c.eff_del else 0
                entry["n_event"] = len(c.eff_event) if c.eff_event else 0
                entry["n_instances"] = c.n_instances
            if r is not None:
                entry["pass_rate"] = round(r.overall_pass_rate, 3)
            entries.append(entry)
        return entries

    def get_detail(self, skill_id: str) -> Optional[Dict[str, Any]]:
        """Full detail for one skill: contract + report + quality info."""
        c = self._bank.get_contract(skill_id)
        if c is None:
            return None

        detail: Dict[str, Any] = {"skill_id": skill_id}
        if getattr(c, "name", None):
            detail["name"] = c.name
        if getattr(c, "description", None):
            detail["description"] = c.description
        detail["contract"] = _contract_summary(c)

        r = self._bank.get_report(skill_id)
        if r is not None:
            detail["report"] = {
                "pass_rate": round(r.overall_pass_rate, 3),
                "n_instances": r.n_instances,
                "worst_segments": r.worst_segments[:5],
                "failure_signatures": dict(list(r.failure_signatures.items())[:5]),
            }
            detail["eff_add_rates"] = {
                k: round(v, 3) for k, v in r.eff_add_success_rate.items()
            }
            detail["eff_del_rates"] = {
                k: round(v, 3) for k, v in r.eff_del_success_rate.items()
            }
        return detail

    # ── Decision-agent convenience ───────────────────────────────────

    def query_for_decision_agent(
        self,
        key: str,
        current_state: Optional[Dict[str, float]] = None,
        top_k: int = 1,
    ) -> Dict[str, Any]:
        """Convenience method for decision agents returning a single best-match.

        When ``current_state`` is provided, uses the full selection policy
        (relevance + applicability).  Otherwise falls back to retrieval only.

        Returns a dict with: skill_id, micro_plan, contract, relevance,
        applicability, confidence.
        """
        if current_state is not None:
            results = self.select(key, current_state=current_state, top_k=top_k)
            if not results:
                return {"skill_id": None, "micro_plan": [], "confidence": 0.0}
            best = results[0]
            return best.to_dict()

        # Backward-compatible path
        results = self.query(key, top_k=top_k)
        if not results:
            return {"skill_id": None, "micro_plan": []}

        best = results[0]
        return {
            "skill_id": best.get("skill_id"),
            "micro_plan": best.get("micro_plan", []) or [{"action": "proceed"}],
            "contract": best.get("contract", {}),
        }
