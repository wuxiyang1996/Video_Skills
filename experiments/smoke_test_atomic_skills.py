#!/usr/bin/env python3
"""Smoke-test all 28 executable atomic skill functions."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from atomic_skills import export_skill_ontology  # noqa: E402
from atomic_skills.evidence_graph_construction import (  # noqa: E402
    assign_provenance_trust,
    create_event_node,
    create_state_node,
    detect_entity_mention,
    extract_dialogue_span,
    extract_observation,
    link_graph_relation,
    resolve_entity_coreference,
    segment_video_or_select_clip,
)
from atomic_skills.reasoning_graph_assembly import (  # noqa: E402
    assign_evidence_role,
    commit_answer,
    compose_evidence_chain,
    detect_missing_role,
    extract_claim,
    infer_causal_relation,
    infer_intention_or_motive,
    infer_social_contradiction,
    infer_state_change,
    infer_temporal_relation,
    localize_clue,
    parse_question_target,
    propose_evidence_roles,
    retrieve_by_entity,
    retrieve_by_event,
    retrieve_by_relation,
    retrieve_by_time,
    search_counterevidence,
    verify_claim_support,
)


def require(ok: bool, label: str) -> None:
    if not ok:
        raise AssertionError(label)


def main() -> int:
    graph: dict = {}
    executed: list[str] = []

    seg = segment_video_or_select_clip(
        graph,
        video_id="smoke_video",
        clip_policy={"strategy": "fixed_window", "duration_s": 60, "window_s": 30, "overlap_s": 0},
    )
    require(seg.ok, "segment_video_or_select_clip")
    graph = seg.outputs["graph"]
    executed.append(seg.skill_id)

    obs1 = extract_observation(
        graph,
        clip_or_text_ref=seg.evidence_refs[0],
        modality="subtitle",
        text="Alice tells Bob: I will stay in the library.",
        time_span={"start_s": 2, "end_s": 7},
    )
    require(obs1.ok, "extract_observation obs1")
    graph = obs1.outputs["graph"]
    executed.append(obs1.skill_id)

    obs2 = extract_observation(
        graph,
        clip_or_text_ref=seg.evidence_refs[0],
        modality="visual_caption",
        text="Alice leaves the library and boards a bus.",
        time_span={"start_s": 32, "end_s": 38},
    )
    require(obs2.ok, "extract_observation obs2")
    graph = obs2.outputs["graph"]

    dia = extract_dialogue_span(
        graph,
        subtitle_or_asr_ref="subtitle:1",
        text="Bob: Alice said she would stay, but she left.",
        time_span={"start_s": 41, "end_s": 46},
    )
    require(dia.ok, "extract_dialogue_span")
    graph = dia.outputs["graph"]
    executed.append(dia.skill_id)

    mentions = []
    for ref in [obs1.evidence_refs[0], obs2.evidence_refs[0], dia.evidence_refs[0]]:
        result = detect_entity_mention(graph, observation_ref=ref)
        require(result.ok, f"detect_entity_mention {ref}")
        graph = result.outputs["graph"]
        mentions.extend(node["node_id"] for node in result.outputs["mention_nodes"] if node["surface_form"] == "Alice")
    executed.append("detect_entity_mention")

    entity = resolve_entity_coreference(graph, mention_nodes=mentions)
    require(entity.ok, "resolve_entity_coreference")
    graph = entity.outputs["graph"]
    entity_ref = entity.outputs["entity_node"]["node_id"]
    executed.append(entity.skill_id)

    event1 = create_event_node(
        graph,
        observation_refs=[obs1.evidence_refs[0]],
        event_description="Alice says she will stay in the library",
        time_span={"start_s": 2, "end_s": 7},
    )
    require(event1.ok, "create_event_node event1")
    graph = event1.outputs["graph"]
    executed.append(event1.skill_id)

    event2 = create_event_node(
        graph,
        observation_refs=[obs2.evidence_refs[0]],
        event_description="Alice leaves the library and boards a bus",
        time_span={"start_s": 32, "end_s": 38},
    )
    require(event2.ok, "create_event_node event2")
    graph = event2.outputs["graph"]

    state = create_state_node(
        graph,
        entity_ref=entity_ref,
        state_predicate="location",
        evidence_refs=[obs2.evidence_refs[0]],
        state_value="outside library / bus",
        time_span={"start_s": 32, "end_s": 38},
    )
    require(state.ok, "create_state_node")
    graph = state.outputs["graph"]
    executed.append(state.skill_id)

    rel = link_graph_relation(
        graph,
        source_node=event1.outputs["event_node"]["node_id"],
        target_node=event2.outputs["event_node"]["node_id"],
        edge_type="temporal_next",
        evidence_refs=[event1.outputs["event_node"]["node_id"], event2.outputs["event_node"]["node_id"]],
    )
    require(rel.ok, "link_graph_relation")
    graph = rel.outputs["graph"]
    executed.append(rel.skill_id)

    prov = assign_provenance_trust(
        graph,
        node_or_edge_ref=obs1.evidence_refs[0],
        source_ref="segment_description",
        mode="expert_demo",
        trust_policy={"gold_sources": ["segment_description"], "strong_sources": [], "weak_sources": [], "model_labeled_sources": []},
    )
    require(prov.ok, "assign_provenance_trust")
    graph = prov.outputs["graph"]
    executed.append(prov.skill_id)

    parsed = parse_question_target("Why is Alice's statement inconsistent with what happens later?")
    require(parsed.ok, "parse_question_target")
    executed.append(parsed.skill_id)

    roles = propose_evidence_roles("Why is Alice's statement inconsistent with what happens later?", parsed.outputs)
    require(roles.ok, "propose_evidence_roles")
    executed.append(roles.skill_id)

    by_event = retrieve_by_event(graph, event_description="Alice stay library")
    require(by_event.ok, "retrieve_by_event")
    executed.append(by_event.skill_id)

    by_entity = retrieve_by_entity(graph, entity_id="Alice")
    require(by_entity.ok, "retrieve_by_entity")
    executed.append(by_entity.skill_id)

    by_time = retrieve_by_time(graph, anchor_event_or_time=event1.outputs["event_node"]["node_id"], window_before=5, window_after=40)
    require(by_time.ok, "retrieve_by_time")
    executed.append(by_time.skill_id)

    by_relation = retrieve_by_relation(graph, source_node=event1.outputs["event_node"]["node_id"], relation_type="temporal_next")
    require(by_relation.ok, "retrieve_by_relation")
    executed.append(by_relation.skill_id)

    clue = localize_clue(by_time.outputs["neighbor_events"], role_constraint="contradiction_evidence", question_context="Alice inconsistent")
    require(clue.ok, "localize_clue")
    executed.append(clue.skill_id)

    claim = extract_claim(graph, evidence_ref=obs1.evidence_refs[0])
    require(claim.ok, "extract_claim")
    executed.append(claim.skill_id)

    role1 = assign_evidence_role(graph, evidence_ref=obs1.evidence_refs[0], role_schema="stated_claim", question_context="Alice inconsistent")
    role2 = assign_evidence_role(graph, evidence_ref=obs2.evidence_refs[0], role_schema="contradiction_evidence", question_context="Alice inconsistent")
    require(role1.ok and role2.ok, "assign_evidence_role")
    executed.append(role1.skill_id)

    chain = compose_evidence_chain(
        [role1.outputs["role_labeled_evidence"], role2.outputs["role_labeled_evidence"]],
        dependency_template="stated_claim->contradiction_evidence",
    )
    require(chain.ok, "compose_evidence_chain")
    evidence_chain = chain.outputs["evidence_chain"]
    executed.append(chain.skill_id)

    missing = detect_missing_role(evidence_chain, required_roles=["stated_claim", "contradiction_evidence"])
    require(missing.ok, "detect_missing_role")
    executed.append(missing.skill_id)

    counter = search_counterevidence(
        graph,
        claim=claim.outputs,
        supporting_evidence=[obs1.evidence_refs[0]],
        search_scope="Alice left bus",
    )
    require(counter.ok, "search_counterevidence")
    executed.append(counter.skill_id)

    temporal = infer_temporal_relation(
        [event1.outputs["event_node"]["node_id"], event2.outputs["event_node"]["node_id"]],
        evidence_graph=graph,
    )
    require(temporal.ok, "infer_temporal_relation")
    executed.append(temporal.skill_id)

    state_change = infer_state_change(
        graph,
        entity_or_object="Alice",
        state_predicate="location",
        before_after_refs=[event1.outputs["event_node"]["node_id"], state.outputs["state_node"]["node_id"]],
    )
    require(state_change.ok, "infer_state_change")
    executed.append(state_change.skill_id)

    causal = infer_causal_relation("Alice left", "statement became inconsistent", evidence_chain=evidence_chain)
    require(causal.ok, "infer_causal_relation")
    executed.append(causal.skill_id)

    motive = infer_intention_or_motive("Alice", ["left the library", "boarded a bus"], context_evidence=evidence_chain["evidence_refs"])
    require(motive.ok, "infer_intention_or_motive")
    executed.append(motive.skill_id)

    contradiction = infer_social_contradiction(claim.outputs, evidence_chain=evidence_chain, counterevidence=counter.evidence_refs)
    require(contradiction.ok, "infer_social_contradiction")
    executed.append(contradiction.skill_id)

    verified = verify_claim_support(contradiction.outputs["contradiction_claim"], evidence_chain=evidence_chain, support_policy={"min_evidence_refs": 2})
    require(verified.ok, "verify_claim_support")
    executed.append(verified.skill_id)

    answer = commit_answer(verified.outputs["verified_claim"], answer_format="free_text", support_chain=evidence_chain)
    require(answer.ok, "commit_answer")
    executed.append(answer.skill_id)

    ontology = export_skill_ontology()
    expected = {
        item["skill_id"]
        for group in ontology.values()
        for item in group
    }
    missing_skills = sorted(expected - set(executed))
    report = {
        "passed": not missing_skills,
        "executed_count": len(set(executed)),
        "expected_count": len(expected),
        "missing_skills": missing_skills,
        "graph_nodes": len(graph["nodes"]),
        "graph_edges": len(graph["edges"]),
        "final_answer": answer.outputs["final_answer"],
    }
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
