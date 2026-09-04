"""Guards against the two silent fallbacks in the answer chain.

Both leave ok=True and errors=0: a withdrawn planner model makes
build_llm_reasoning_rollout fall back to a deterministic plan (only
llm_plan.fallback_reason records it), and an absent or failing skill executor
makes every answer-critical skill run as a lexical rule (rollout nodes carry no
backend field; the marker is metadata.answer_step_diagnostics[step].backend).
Each produced a plausible wrong number before it was caught.
"""

from scripts.eval.measure_answer_chain import degradation, score


def _rollout(*, backend="llm", fallback_reason=None, messages=()):
    """Shaped like a real cached rollout: nodes without backend, diagnostics with it."""
    return {
        "final_answer": {"label": "B"},
        "acceptance_status": "accepted_strong",
        "nodes": [
            {"node_id": "skill:1", "skill_id": "parse_question_target", "step_id": "r1", "status": "ok"},
            {"node_id": "skill:2", "skill_id": "score_hypothesis_support", "step_id": "r7", "status": "verified"},
            {"node_id": "skill:3", "skill_id": "verify_claim_support", "step_id": "r10", "status": "verified"},
        ],
        "metadata": {
            "llm_plan": {"fallback_reason": fallback_reason} if fallback_reason else {"llm_usage": {}},
            "answer_step_diagnostics": {
                "r7": {"scored_hypothesis": {"option_label": "B", "backend": backend}, "messages": list(messages)},
                "r10": {"backend": backend, "verified_claim": {"option_label": "B"}},
            },
        },
    }


def test_llm_backend_is_read_from_step_diagnostics_not_nodes() -> None:
    out = degradation(_rollout(backend="llm"))
    assert out["llm_skill_nodes"] == 2
    assert out["critical_skills_on_llm"] == 2
    assert out["planner_fell_back"] is False


def test_rule_only_execution_reports_zero_critical_llm_skills() -> None:
    # This is the degraded mode the script used to run in silently.
    out = degradation(_rollout(backend="rule"))
    assert out["critical_skills_on_llm"] == 0


def test_planner_fallback_is_detected() -> None:
    out = degradation(_rollout(fallback_reason="planner_http_404"))
    assert out["planner_fell_back"] is True


def test_rule_fallback_message_is_counted() -> None:
    out = degradation(_rollout(messages=["llm_timeout_fallback_to_rule"]))
    assert out["rule_fallbacks"] == 1


def test_score_carries_degradation_markers() -> None:
    row = score(_rollout(fallback_reason="planner_http_404"), "B")
    assert row["correct"] is True  # the label happens to match ...
    assert row["planner_fell_back"] is True  # ... but the row must be excludable


def test_nested_backend_is_found_when_top_level_is_absent() -> None:
    r = _rollout()
    r["metadata"]["answer_step_diagnostics"]["r10"] = {"verified_claim": {"option_label": "B", "backend": "llm"}}
    assert degradation(r)["critical_skills_on_llm"] == 2
