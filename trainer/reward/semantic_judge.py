"""Frozen LLM semantic judge for verified atomic / claim milestones (plan §6.A)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

JUDGE_RUBRIC_VERSION = "video-skills/semantic-judge-rubric-v1"
JUDGE_VERDICTS = ("supported", "insufficient", "contradicted")

# Must never appear in judge prompts (policy isolation).
FORBIDDEN_PROMPT_TOKENS = (
    "gold_answer",
    "official_answer",
    "teacher_trajectory",
    "teacher_probs",
    "policy_logits",
    "motif_lifecycle",
    "heldout",
    "hidden_eval",
    "answer_correct",
)


@dataclass(frozen=True)
class SemanticJudgeResult:
    verdict: str
    grounded_refs: tuple[str, ...] = ()
    missing_premises: tuple[str, ...] = ()
    relation_valid: bool = False
    question_relevant: bool = False
    counterfactual_sensitivity: str = "unknown"
    confidence_bucket: str = "unknown"
    rationale_hidden: str = ""
    judge_model: str = ""
    rubric_version: str = JUDGE_RUBRIC_VERSION
    malformed: bool = False
    timed_out: bool = False
    view_id: str = "0"

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "grounded_refs": list(self.grounded_refs),
            "missing_premises": list(self.missing_premises),
            "relation_valid": self.relation_valid,
            "question_relevant": self.question_relevant,
            "counterfactual_sensitivity": self.counterfactual_sensitivity,
            "confidence_bucket": self.confidence_bucket,
            "judge_model": self.judge_model,
            "rubric_version": self.rubric_version,
            "malformed": self.malformed,
            "timed_out": self.timed_out,
            "view_id": self.view_id,
            # rationale stays in audit logs only; not returned to policy surfaces.
            "has_hidden_rationale": bool(self.rationale_hidden),
        }


JudgeFn = Callable[[Mapping[str, Any]], SemanticJudgeResult]


def assert_judge_prompt_safe(payload: Mapping[str, Any] | str) -> None:
    blob = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
    lower = blob.lower()
    hits = [tok for tok in FORBIDDEN_PROMPT_TOKENS if tok in lower]
    if hits:
        raise ValueError(f"semantic judge prompt contains forbidden fields: {hits}")


def build_judge_messages(
    *,
    question_text: str,
    claim: str,
    evidence: Sequence[Mapping[str, Any]],
    relation: str | None = None,
    role: str | None = None,
) -> list[dict[str, str]]:
    evidence_lines = []
    for item in evidence:
        ref = item.get("ref") or item.get("evidence_ref") or "?"
        t0 = item.get("t_start")
        t1 = item.get("t_end")
        text = item.get("text") or item.get("snippet") or ""
        evidence_lines.append(f"- ref={ref} time={t0}-{t1} text={text}")
    user = {
        "question": question_text,
        "claim": claim,
        "relation": relation,
        "role": role,
        "evidence": evidence_lines,
    }
    assert_judge_prompt_safe(user)
    return [
        {
            "role": "system",
            "content": (
                "You are a frozen evidence-only semantic judge. "
                "Use only supplied evidence. Return strict JSON with keys: "
                "verdict, grounded_refs, missing_premises, relation_valid, "
                "question_relevant, counterfactual_sensitivity, confidence_bucket, rationale."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Question: {question_text}\n"
                f"Claim: {claim}\n"
                f"Relation: {relation or 'n/a'}\n"
                f"Role: {role or 'n/a'}\n"
                "Evidence:\n" + ("\n".join(evidence_lines) if evidence_lines else "(none)")
            ),
        },
    ]


def parse_judge_payload(
    payload: Mapping[str, Any],
    *,
    judge_model: str = "",
    view_id: str = "0",
    malformed: bool = False,
    timed_out: bool = False,
) -> SemanticJudgeResult:
    if malformed or timed_out:
        return SemanticJudgeResult(
            verdict="insufficient",
            malformed=malformed,
            timed_out=timed_out,
            judge_model=judge_model,
            view_id=view_id,
        )
    verdict = str(payload.get("verdict") or "insufficient").strip().lower()
    if verdict not in JUDGE_VERDICTS:
        verdict = "insufficient"
        malformed = True
    return SemanticJudgeResult(
        verdict=verdict,
        grounded_refs=tuple(str(x) for x in (payload.get("grounded_refs") or [])),
        missing_premises=tuple(str(x) for x in (payload.get("missing_premises") or [])),
        relation_valid=bool(payload.get("relation_valid")),
        question_relevant=bool(payload.get("question_relevant")),
        counterfactual_sensitivity=str(payload.get("counterfactual_sensitivity") or "unknown"),
        confidence_bucket=str(payload.get("confidence_bucket") or "unknown"),
        rationale_hidden=str(payload.get("rationale") or ""),
        judge_model=judge_model,
        malformed=malformed,
        timed_out=timed_out,
        view_id=view_id,
    )


def aggregate_dual_views(
    views: Sequence[SemanticJudgeResult],
) -> SemanticJudgeResult:
    """Conservative dual-view aggregation: disagreement → insufficient."""
    if not views:
        return SemanticJudgeResult(verdict="insufficient", malformed=True)
    if any(v.malformed or v.timed_out for v in views):
        return SemanticJudgeResult(
            verdict="insufficient",
            malformed=any(v.malformed for v in views),
            timed_out=any(v.timed_out for v in views),
            judge_model=views[0].judge_model,
            view_id="aggregated",
        )
    verdicts = {v.verdict for v in views}
    if len(verdicts) != 1:
        return SemanticJudgeResult(
            verdict="insufficient",
            grounded_refs=tuple(sorted({r for v in views for r in v.grounded_refs})),
            judge_model=views[0].judge_model,
            view_id="aggregated",
        )
    base = views[0]
    return SemanticJudgeResult(
        verdict=base.verdict,
        grounded_refs=tuple(sorted({r for v in views for r in v.grounded_refs})),
        missing_premises=tuple(sorted({p for v in views for p in v.missing_premises})),
        relation_valid=all(v.relation_valid for v in views),
        question_relevant=all(v.question_relevant for v in views),
        counterfactual_sensitivity=base.counterfactual_sensitivity,
        confidence_bucket="agree",
        rationale_hidden="",
        judge_model=base.judge_model,
        view_id="aggregated",
    )


def credit_allowed(result: SemanticJudgeResult) -> bool:
    """Only supported + valid relation/question relevance earns positive partial credit."""
    return (
        result.verdict == "supported"
        and not result.malformed
        and not result.timed_out
        and result.relation_valid
        and result.question_relevant
    )


def mock_semantic_judge(request: Mapping[str, Any]) -> SemanticJudgeResult:
    """Deterministic judge for unit tests / dry-runs (no API)."""
    claim = str(request.get("claim") or "").lower()
    evidence = list(request.get("evidence") or [])
    refs = [str(e.get("ref") or e.get("evidence_ref") or "") for e in evidence if isinstance(e, Mapping)]
    if "contradict" in claim or "wrong entity" in claim:
        verdict = "contradicted"
        relation_valid = False
    elif not evidence or "missing" in claim:
        verdict = "insufficient"
        relation_valid = False
    else:
        verdict = "supported"
        relation_valid = True
    return SemanticJudgeResult(
        verdict=verdict,
        grounded_refs=tuple(r for r in refs if r),
        missing_premises=() if evidence else ("evidence",),
        relation_valid=relation_valid,
        question_relevant=True,
        counterfactual_sensitivity="mock",
        confidence_bucket="mock",
        rationale_hidden="mock_judge",
        judge_model="mock_semantic_judge",
        view_id=str(request.get("view_id") or "0"),
    )


def make_openrouter_semantic_judge(
    *,
    api_key: str,
    model: str = "deepseek/deepseek-v4-pro",
    timeout_s: int = 120,
) -> JudgeFn:
    """OpenRouter judge using strict JSON; never returns rationale to policy callers."""
    from dataset_clip_wrapper.perception.openrouter_client import OpenRouterClient

    client = OpenRouterClient(
        model=model,
        api_key=api_key,
        temperature=0.0,
        max_tokens=260,
        reasoning={"enabled": False, "exclude": True},
        timeout_s=timeout_s,
    )
    schema = {
        "type": "json_schema",
        "json_schema": {
            "name": "semantic_judge",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "verdict": {"type": "string", "enum": list(JUDGE_VERDICTS)},
                    "grounded_refs": {"type": "array", "items": {"type": "string"}},
                    "missing_premises": {"type": "array", "items": {"type": "string"}},
                    "relation_valid": {"type": "boolean"},
                    "question_relevant": {"type": "boolean"},
                    "counterfactual_sensitivity": {"type": "string"},
                    "confidence_bucket": {"type": "string"},
                    "rationale": {"type": "string"},
                },
                "required": [
                    "verdict",
                    "grounded_refs",
                    "missing_premises",
                    "relation_valid",
                    "question_relevant",
                ],
            },
        },
    }

    def _judge(request: Mapping[str, Any]) -> SemanticJudgeResult:
        messages = build_judge_messages(
            question_text=str(request.get("question_text") or ""),
            claim=str(request.get("claim") or ""),
            evidence=list(request.get("evidence") or []),
            relation=request.get("relation"),
            role=request.get("role"),
        )
        view_id = str(request.get("view_id") or "0")
        try:
            payload = client.chat_json(messages, response_format=schema)
            return parse_judge_payload(payload, judge_model=model, view_id=view_id)
        except TimeoutError:
            return parse_judge_payload({}, judge_model=model, view_id=view_id, timed_out=True)
        except Exception:
            return parse_judge_payload({}, judge_model=model, view_id=view_id, malformed=True)

    return _judge
