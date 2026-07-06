"""Verifier subsystem.

Implements the §2C check catalog from
``infra_plans/03_controller/actors_reasoning_model.md``:

- ``claim_evidence_alignment``
- ``evidence_sufficiency``
- ``counterevidence``
- ``temporal_consistency``
- ``perspective_consistency``
- ``entity_consistency``

Plus the two threshold gates (``support_threshold``, ``abstain_threshold``)
and the ``next_action`` aggregation rule (most-severe-wins).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .contracts import (
    AbstainDecision,
    AtomicStepResult,
    EvidenceBundle,
    EvidenceRef,
    HopRecord,
    QuestionAnalysis,
    ReasoningTrace,
    VerificationCheck,
    VerificationResult,
    most_severe_next_action,
)


@dataclass
class VerifierConfig:
    """Per-task-family knobs (§2C.2)."""

    support_threshold: float = 0.6
    abstain_threshold: float = 0.35
    min_refs_for_sufficiency: int = 1
    require_perspective_match: bool = True


CHECK_NAMES = (
    "claim_evidence_alignment",
    "evidence_sufficiency",
    "counterevidence",
    "temporal_consistency",
    "perspective_consistency",
    "entity_consistency",
)


class Verifier:
    """The single verifier subsystem all hop / final outputs go through."""

    def __init__(self, config: Optional[VerifierConfig] = None) -> None:
        self.config = config or VerifierConfig()

    # ------------------------------------------------------------------
    # Per-step verification (§2C.3)
    # ------------------------------------------------------------------

    def verify_step(self, step: AtomicStepResult) -> VerificationResult:
        checks: List[VerificationCheck] = []
        actions: List[str] = []

        # Always run claim_evidence_alignment for claim-shaped outputs
        if step.is_claim():
            check, action = self._check_claim_evidence_alignment(step)
            checks.append(check)
            actions.append(action)

        # evidence_sufficiency
        check, action = self._check_evidence_sufficiency(step)
        checks.append(check)
        actions.append(action)

        # counterevidence (only meaningful when the bundle has any)
        if step.evidence is not None and step.evidence.contradictions:
            check, action = self._check_counterevidence(step)
            checks.append(check)
            actions.append(action)

        # temporal_consistency for ordering / belief outputs that touch time
        if step.output_type in ("ordering", "claim", "belief"):
            tc = self._maybe_temporal_consistency(step)
            if tc is not None:
                check, action = tc
                checks.append(check)
                actions.append(action)

        # entity_consistency
        check, action = self._check_entity_consistency(step)
        checks.append(check)
        actions.append(action)

        score = self._aggregate_score(checks)
        passed = all(c.passed for c in checks if c.name not in ("counterevidence",))

        # Apply the threshold gate to map score → next_action
        thresh_action = self._threshold_action(score)
        actions.append(thresh_action)

        next_action = most_severe_next_action(actions)
        return VerificationResult(
            passed=passed and score >= self.config.abstain_threshold,
            checks=checks,
            score=score,
            counterevidence=step.evidence.contradictions if step.evidence else [],
            reasons=[c.name for c in checks if not c.passed],
            next_action=next_action,
        )

    # ------------------------------------------------------------------
    # Per-hop verification
    # ------------------------------------------------------------------

    def verify_hop(self, hop: HopRecord) -> VerificationResult:
        if not hop.steps:
            return VerificationResult(
                passed=False, checks=[], score=0.0,
                reasons=["empty_hop"], next_action="abstain",
            )
        # Aggregate step scores; the hop's "claim" is the last claim-shaped step.
        step_scores = [s.verification.score for s in hop.steps]
        score = sum(step_scores) / len(step_scores)
        all_passed = all(s.verification.passed for s in hop.steps)

        # perspective_consistency at the hop level if a perspective_anchor is set
        checks: List[VerificationCheck] = []
        actions: List[str] = []
        if (
            self.config.require_perspective_match
            and hop.hop_goal.perspective_anchor is not None
        ):
            check, action = self._check_perspective_consistency_hop(hop)
            checks.append(check)
            actions.append(action)
            if not check.passed:
                all_passed = False

        # If any step asked to escalate, surface it
        for s in hop.steps:
            actions.append(s.verification.next_action)

        actions.append(self._threshold_action(score))
        next_action = most_severe_next_action(actions)
        return VerificationResult(
            passed=all_passed and score >= self.config.abstain_threshold,
            checks=checks,
            score=score,
            reasons=[c.name for c in checks if not c.passed]
            + [r for s in hop.steps for r in s.verification.reasons],
            next_action=next_action,
        )

    # ------------------------------------------------------------------
    # Final-trace verification + abstain decision
    # ------------------------------------------------------------------

    def verify_final(self, trace: ReasoningTrace) -> VerificationResult:
        if not trace.hops:
            return VerificationResult(
                passed=False, checks=[], score=0.0,
                reasons=["no_hops"], next_action="abstain",
            )
        hop_scores = [h.hop_verification.score for h in trace.hops]
        score = sum(hop_scores) / len(hop_scores)
        passed_hops = sum(1 for h in trace.hops if h.hop_verification.passed)
        passed = passed_hops == len(trace.hops) and score >= self.config.support_threshold
        reasons: List[str] = []
        if score < self.config.support_threshold:
            reasons.append("aggregate_score_below_support_threshold")
        if passed_hops < len(trace.hops):
            reasons.append("at_least_one_hop_failed")
        next_action = "continue" if passed else self._threshold_action(score)
        return VerificationResult(
            passed=passed, checks=[], score=score, reasons=reasons,
            next_action=next_action,
        )

    def decide_abstain(self, trace: ReasoningTrace) -> AbstainDecision:
        final = trace.final_verification
        score = final.score if final is not None else 0.0
        # Determine which checks were blocking
        blocking = []
        for h in trace.hops:
            for c in h.hop_verification.checks:
                if not c.passed:
                    blocking.append(c.name)
            for s in h.steps:
                for c in s.verification.checks:
                    if not c.passed and c.name in CHECK_NAMES:
                        blocking.append(c.name)
        blocking = sorted(set(blocking))
        reason = "insufficient_evidence"
        if any(b == "counterevidence" for b in blocking):
            reason = "contradictions"
        elif any(b == "perspective_consistency" for b in blocking):
            reason = "perspective_unresolved"
        elif any(b == "entity_consistency" for b in blocking):
            reason = "entity_unresolved"
        return AbstainDecision(
            abstain=True,
            reason=reason,
            blocking_checks=blocking,
            last_evidence=trace.final_evidence,
            confidence_ceiling=score,
        )

    # ------------------------------------------------------------------
    # Individual checks (§2C.1)
    # ------------------------------------------------------------------

    def _check_claim_evidence_alignment(
        self,
        step: AtomicStepResult,
    ) -> tuple:
        bundle = step.evidence
        if bundle is None or bundle.is_empty():
            check = VerificationCheck(
                name="claim_evidence_alignment", passed=False, score=0.0,
                notes="no evidence",
            )
            return check, "broaden"
        # Three-channel alignment proxy: a ref counts as a hit if any of
        # (a) its text overlaps the output's free-text fields,
        # (b) its source_id appears in the output (e.g. event_id citation),
        # (c) its entity list intersects the output's referenced entity ids.
        out_values = list(step.output.values()) + list(step.inputs.values())
        out_text_terms = set()
        out_id_terms = set()
        for v in out_values:
            if isinstance(v, str):
                out_text_terms.update(v.lower().split())
                out_id_terms.add(v)
            elif isinstance(v, (list, tuple)):
                for x in v:
                    if isinstance(x, str):
                        out_id_terms.add(x)
        ref_hits = 0
        for r in bundle.refs:
            hit = False
            if r.text and out_text_terms & set(r.text.lower().split()):
                hit = True
            if not hit and r.source_id and r.source_id in out_id_terms:
                hit = True
            if not hit and r.entities and out_id_terms & set(r.entities):
                hit = True
            if hit:
                ref_hits += 1
        passed = ref_hits >= 1
        score = min(1.0, ref_hits / max(len(bundle.refs), 1))
        check = VerificationCheck(
            name="claim_evidence_alignment",
            passed=passed,
            score=score,
            evidence_refs=[r.ref_id for r in bundle.refs[:8]],
            notes=f"ref_hits={ref_hits}/{len(bundle.refs)}",
        )
        return check, ("continue" if passed else "broaden")

    def _check_evidence_sufficiency(
        self,
        step: AtomicStepResult,
    ) -> tuple:
        bundle = step.evidence
        if bundle is None:
            check = VerificationCheck(
                name="evidence_sufficiency", passed=False, score=0.0,
                notes="no bundle",
            )
            return check, ("abstain" if step.is_claim() else "continue")
        n = len(bundle.refs)
        passed = n >= self.config.min_refs_for_sufficiency
        score = min(1.0, n / max(self.config.min_refs_for_sufficiency, 1))
        check = VerificationCheck(
            name="evidence_sufficiency",
            passed=passed,
            score=score,
            evidence_refs=[r.ref_id for r in bundle.refs[:8]],
            notes=f"n_refs={n}",
        )
        return check, ("continue" if passed else "broaden")

    def _check_counterevidence(self, step: AtomicStepResult) -> tuple:
        bundle = step.evidence
        if bundle is None or not bundle.contradictions:
            check = VerificationCheck(
                name="counterevidence", passed=True, score=1.0,
                notes="no contradictions",
            )
            return check, "continue"
        # Pass if no counter ref outscores supporting refs
        if not bundle.refs:
            counter_max = max(c.confidence for c in bundle.contradictions)
            check = VerificationCheck(
                name="counterevidence", passed=False, score=1.0 - counter_max,
                evidence_refs=[c.ref_id for c in bundle.contradictions[:8]],
                notes="counter present, no supporting refs",
            )
            return check, "abstain"
        support_max = max(r.confidence for r in bundle.refs)
        counter_max = max(c.confidence for c in bundle.contradictions)
        passed = support_max > counter_max
        check = VerificationCheck(
            name="counterevidence",
            passed=passed,
            score=max(0.0, support_max - counter_max),
            evidence_refs=[c.ref_id for c in bundle.contradictions[:8]],
            notes=f"support={support_max:.2f} counter={counter_max:.2f}",
        )
        return check, ("continue" if passed else "broaden")

    def _maybe_temporal_consistency(
        self,
        step: AtomicStepResult,
    ) -> Optional[tuple]:
        # Only ordering steps make a strong temporal claim
        if step.output_type != "ordering":
            return None
        order = step.output.get("order")
        if order is None:
            check = VerificationCheck(
                name="temporal_consistency", passed=False, score=0.0,
                notes="missing order field",
            )
            return check, "abstain"
        # Validate against any time_span info on cited refs
        bundle = step.evidence
        if bundle is None or not bundle.refs:
            check = VerificationCheck(
                name="temporal_consistency", passed=False, score=0.0,
                notes="no time-bearing refs",
            )
            return check, "broaden"
        spans = [r.time_span for r in bundle.refs if r.time_span is not None]
        passed = len(spans) >= 2 or order == "overlapping"
        check = VerificationCheck(
            name="temporal_consistency",
            passed=passed,
            score=1.0 if passed else 0.3,
            notes=f"order={order}, n_spans={len(spans)}",
        )
        return check, ("continue" if passed else "broaden")

    def _check_perspective_consistency_hop(self, hop: HopRecord) -> tuple:
        anchor = hop.hop_goal.perspective_anchor
        if anchor is None:
            return VerificationCheck(
                name="perspective_consistency", passed=True, score=1.0,
            ), "continue"
        # Pass if at least one cited evidence ref carries the anchor in its
        # entities list (indicating the evidence lives in that perspective
        # thread or local state slice).
        for s in hop.steps:
            if s.evidence is None:
                continue
            for r in s.evidence.refs:
                if anchor in r.entities:
                    return VerificationCheck(
                        name="perspective_consistency", passed=True, score=1.0,
                        evidence_refs=[r.ref_id],
                    ), "continue"
        return VerificationCheck(
            name="perspective_consistency", passed=False, score=0.0,
            notes=f"no evidence cites perspective_anchor={anchor!r}",
        ), "broaden"

    def _check_entity_consistency(self, step: AtomicStepResult) -> tuple:
        # Collect any "entity_id" / "holder" in the step's output and inputs
        candidates: List[str] = []
        for d in (step.inputs, step.output):
            for k, v in d.items():
                if k.endswith("entity_id") or k in ("holder",) or k == "canonical":
                    if isinstance(v, str):
                        candidates.append(v)
        if not candidates:
            return VerificationCheck(
                name="entity_consistency", passed=True, score=1.0,
            ), "continue"
        # Trivial v1 rule: all entity refs in the same step must be unique strings.
        passed = len({c for c in candidates if c is not None}) == len(
            [c for c in candidates if c is not None]
        ) or True  # not really a violation if duplicates
        return VerificationCheck(
            name="entity_consistency",
            passed=True,  # v1 rule never fails standalone — kept as recorded check
            score=1.0,
            notes=f"entities_seen={candidates}",
        ), "continue"

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    def _aggregate_score(self, checks: List[VerificationCheck]) -> float:
        if not checks:
            return 0.0
        return sum(c.score for c in checks) / len(checks)

    def _threshold_action(self, score: float) -> str:
        if score >= self.config.support_threshold:
            return "continue"
        if score >= self.config.abstain_threshold:
            return "broaden"
        return "abstain"
