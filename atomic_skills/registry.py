"""Atomic skill registry for the two documented skill sets."""

from __future__ import annotations

from typing import Any

from .common import SkillSpec
from .evidence_graph_construction import skills as evidence
from .reasoning_graph_assembly import skills as reasoning


EVIDENCE_GRAPH_CONSTRUCTION_SKILLS = [
    SkillSpec("segment_video_or_select_clip", "evidence_graph_construction", "Create clip/window nodes under a clip policy.", ["video_id", "clip_policy", "observation_end_s?"], ["clip_nodes", "time_spans"], "windows are valid and respect visibility constraints", ["invalid_clip_policy"], evidence.segment_video_or_select_clip),
    SkillSpec("extract_observation", "evidence_graph_construction", "Extract observable facts from a clip, caption, ASR span, or annotation.", ["clip_or_text_ref", "modality", "observation_query?", "text", "time_span?"], ["observation_nodes", "evidence_refs"], "observation is grounded to source span", ["empty_observation"], evidence.extract_observation),
    SkillSpec("extract_dialogue_span", "evidence_graph_construction", "Extract speaker, utterance, and timestamp from subtitle/ASR/dialogue annotation.", ["subtitle_or_asr_ref", "speaker_hint?", "text", "time_span"], ["dialogue_span_node", "speaker_mention", "evidence_ref"], "dialogue span has source and timestamp", ["empty_dialogue"], evidence.extract_dialogue_span),
    SkillSpec("detect_entity_mention", "evidence_graph_construction", "Detect person, object, place, and speaker mentions.", ["observation_ref", "entity_type?", "text?"], ["mention_nodes", "surface_forms", "time_spans"], "mention is supported by text/visual/audio evidence", ["no_entity_mentions"], evidence.detect_entity_mention),
    SkillSpec("resolve_entity_coreference", "evidence_graph_construction", "Link mentions across clips or modalities to the same entity.", ["mention_nodes", "context_edges?"], ["entity_node", "same_entity_edges", "confidence"], "linked mentions are compatible and not contradictory", ["missing_mentions"], evidence.resolve_entity_coreference),
    SkillSpec("create_event_node", "evidence_graph_construction", "Convert observations or dialogue spans into timestamped event nodes.", ["observation_refs", "event_description", "time_span"], ["event_node", "event_type", "evidence_refs"], "event is grounded and not a duplicate", ["missing_observation_refs"], evidence.create_event_node),
    SkillSpec("create_state_node", "evidence_graph_construction", "Represent an entity/object state.", ["entity_ref", "state_predicate", "evidence_refs", "state_value", "time_span?"], ["state_node", "state_value", "confidence"], "state is grounded and temporally scoped", ["missing_state_grounding"], evidence.create_state_node),
    SkillSpec("link_graph_relation", "evidence_graph_construction", "Add typed graph relation edges.", ["source_node", "target_node", "edge_type", "evidence_refs?"], ["memory_edge", "confidence"], "edge type is allowed and endpoints exist", ["missing_edge_endpoint", "invalid_edge_type"], evidence.link_graph_relation),
    SkillSpec("assign_provenance_trust", "evidence_graph_construction", "Attach source, trust level, visibility mode, and hidden-supervision status.", ["node_or_edge_ref", "source_ref", "mode", "trust_policy"], ["provenance", "trust_level", "discovery_status"], "provenance and visibility are consistent", ["missing_target"], evidence.assign_provenance_trust),
]


REASONING_GRAPH_ASSEMBLY_SKILLS = [
    SkillSpec("parse_question_target", "reasoning_graph_assembly", "Extract target entities, events, constraints, and answer format.", ["question_text", "options?"], ["target_entities", "target_events", "constraints", "answer_format"], "required targets are present", [], reasoning.parse_question_target),
    SkillSpec("propose_evidence_roles", "reasoning_graph_assembly", "Propose reusable evidence roles needed to answer the question.", ["question_text", "parsed_target", "task_family?"], ["evidence_roles", "role_constraints", "expected_chain_shape"], "roles are typed and relevant", [], reasoning.propose_evidence_roles),
    SkillSpec("retrieve_by_event", "reasoning_graph_assembly", "Retrieve event/evidence nodes matching an event description.", ["event_description", "time_range?", "entity_filter?"], ["event_nodes", "evidence_refs", "retrieval_scores"], "retrieved nodes match event intent", ["no_event_match"], reasoning.retrieve_by_event),
    SkillSpec("retrieve_by_entity", "reasoning_graph_assembly", "Retrieve an entity timeline, history, or related evidence.", ["entity_id", "time_range?", "predicate_filter?"], ["entity_timeline", "evidence_refs"], "evidence refers to the same entity", ["no_entity_match"], reasoning.retrieve_by_entity),
    SkillSpec("retrieve_by_time", "reasoning_graph_assembly", "Retrieve evidence around an anchor event or time window.", ["anchor_event_or_time", "window_before", "window_after"], ["neighbor_events", "evidence_refs"], "timestamps overlap requested window", ["no_time_overlap"], reasoning.retrieve_by_time),
    SkillSpec("retrieve_by_relation", "reasoning_graph_assembly", "Query graph paths or relation edges.", ["source_node", "relation_type", "hop_limit?"], ["related_nodes", "path_edges", "evidence_refs"], "relation path is valid", ["no_relation_path"], reasoning.retrieve_by_relation),
    SkillSpec("localize_clue", "reasoning_graph_assembly", "Select the most relevant clue span/node for a requested role.", ["candidate_evidence", "role_constraint", "question_context"], ["clue_refs", "clue_spans", "confidence"], "clue supports the requested role", ["no_clue_candidate"], reasoning.localize_clue),
    SkillSpec("extract_claim", "reasoning_graph_assembly", "Extract a claim from dialogue, annotation, or evidence text.", ["evidence_ref", "speaker_hint?", "claim_query?"], ["claim_id", "claim_text", "speaker?", "evidence_ref"], "claim is anchored to evidence", ["missing_evidence_ref", "claim_query_not_supported"], reasoning.extract_claim),
    SkillSpec("assign_evidence_role", "reasoning_graph_assembly", "Bind evidence to a semantic role.", ["evidence_ref", "role_schema", "question_context"], ["role_labeled_evidence", "role_confidence"], "role assignment matches content", ["missing_evidence_ref"], reasoning.assign_evidence_role),
    SkillSpec("generate_answer_hypotheses", "reasoning_graph_assembly", "Convert answer options or free-form targets into explicit candidate hypotheses.", ["question_text", "options?", "parsed_target?"], ["hypotheses"], "hypotheses preserve answer choices without adding unsupported facts", [], reasoning.generate_answer_hypotheses),
    SkillSpec("retrieve_evidence_for_hypothesis", "reasoning_graph_assembly", "Retrieve support evidence for a single candidate hypothesis.", ["hypothesis", "max_refs?"], ["support_refs", "weak_refs", "missing_refs", "retrieval_scores"], "retrieved evidence is relevant to the hypothesis", ["no_hypothesis_evidence"], reasoning.retrieve_evidence_for_hypothesis),
    SkillSpec("score_hypothesis_support", "reasoning_graph_assembly", "Score one hypothesis using support evidence and counterevidence.", ["hypothesis", "support_evidence", "counterevidence?"], ["scored_hypothesis"], "support and contradiction scores are comparable across options", ["missing_support_evidence"], reasoning.score_hypothesis_support),
    SkillSpec("compare_hypotheses", "reasoning_graph_assembly", "Compare scored hypotheses and select the best-supported answer candidate.", ["scored_hypotheses", "decision_policy?"], ["best_hypothesis", "eliminated_hypotheses", "decision_reason", "score_margin"], "chosen hypothesis has strongest support under the policy", ["no_hypotheses", "ambiguous_hypotheses"], reasoning.compare_hypotheses),
    SkillSpec("bridge_evidence_hops", "reasoning_graph_assembly", "Construct a small multi-hop evidence bridge from source refs toward a hypothesis.", ["source_evidence", "target_hypothesis", "allowed_hop_types?", "max_hops?"], ["multi_hop_chain"], "bridge connects evidence through graph relations or lexical links", ["no_bridge_path"], reasoning.bridge_evidence_hops),
    SkillSpec("verify_temporal_social_consistency", "reasoning_graph_assembly", "Check generic temporal ordering and social plausibility for a hypothesis support chain.", ["evidence_chain", "hypothesis", "evidence_graph?"], ["temporal_ok", "social_plausibility_ok", "conflicts"], "support chain is temporally and socially consistent", ["consistency_conflict"], reasoning.verify_temporal_social_consistency),
    SkillSpec("compose_evidence_chain", "reasoning_graph_assembly", "Assemble role-labeled evidence into an answer-support chain.", ["role_labeled_evidence", "dependency_template"], ["evidence_chain", "chain_edges", "missing_roles"], "chain covers required roles", ["missing_required_roles"], reasoning.compose_evidence_chain),
    SkillSpec("detect_missing_role", "reasoning_graph_assembly", "Identify missing evidence roles and generate query hints.", ["evidence_chain", "required_roles"], ["missing_roles", "suggested_queries"], "missing roles are truly absent", ["roles_missing"], reasoning.detect_missing_role),
    SkillSpec("search_counterevidence", "reasoning_graph_assembly", "Find evidence that contradicts or weakens a claim.", ["claim", "supporting_evidence", "search_scope"], ["counterevidence_refs", "counter_claims"], "counterevidence is relevant", ["no_counterevidence"], reasoning.search_counterevidence),
    SkillSpec("infer_temporal_relation", "reasoning_graph_assembly", "Infer before/after/overlap/order among events.", ["event_refs", "evidence_graph"], ["temporal_relation", "supporting_evidence"], "timestamps support relation", ["too_few_events"], reasoning.infer_temporal_relation),
    SkillSpec("infer_state_change", "reasoning_graph_assembly", "Infer before/after state change for an entity or object.", ["entity_or_object", "state_predicate", "before_after_refs"], ["state_change_claim", "before_state", "after_state"], "states are grounded and ordered", ["too_few_state_refs"], reasoning.infer_state_change),
    SkillSpec("infer_causal_relation", "reasoning_graph_assembly", "Infer cause-effect support between events or states.", ["candidate_cause", "candidate_effect", "evidence_chain"], ["causal_claim", "supporting_roles"], "cause precedes effect and evidence links them", ["empty_evidence_chain"], reasoning.infer_causal_relation),
    SkillSpec("infer_intention_or_motive", "reasoning_graph_assembly", "Infer agent intention, goal, or motive from actions and context.", ["agent", "actions", "context_evidence"], ["intention_claim", "alternatives", "supporting_roles"], "intention is evidence-supported", ["missing_context_evidence"], reasoning.infer_intention_or_motive),
    SkillSpec("infer_social_contradiction", "reasoning_graph_assembly", "Infer conflict between statement/alibi/promise and later action/evidence.", ["claim_or_alibi", "evidence_chain", "counterevidence?"], ["contradiction_claim", "supporting_evidence"], "claim and evidence cannot both hold", ["missing_contradiction_evidence"], reasoning.infer_social_contradiction),
    SkillSpec("verify_claim_support", "reasoning_graph_assembly", "Verify that an evidence chain supports a claim.", ["claim", "evidence_chain", "support_policy", "evidence_graph?", "question_text?"], ["verification_score", "passed", "failure_code", "messages", "claim_support_score", "target_alignment_score"], "evidence entails or supports claim and aligns with the question target", ["insufficient_evidence"], reasoning.verify_claim_support),
    SkillSpec("commit_answer", "reasoning_graph_assembly", "Map a verified claim to final answer text or MCQ option and record support.", ["verified_claim", "options?", "answer_format", "support_chain"], ["final_answer", "answer_support_chain", "confidence"], "final answer follows from verified claim", ["claim_not_verified", "invalid_answer_commit"], reasoning.commit_answer),
]


SKILL_REGISTRY = {
    spec.skill_id: spec
    for spec in [*EVIDENCE_GRAPH_CONSTRUCTION_SKILLS, *REASONING_GRAPH_ASSEMBLY_SKILLS]
}


def export_skill_ontology() -> dict[str, Any]:
    return {
        "evidence_graph_construction": [spec.as_ontology_record() for spec in EVIDENCE_GRAPH_CONSTRUCTION_SKILLS],
        "reasoning_graph_assembly": [spec.as_ontology_record() for spec in REASONING_GRAPH_ASSEMBLY_SKILLS],
    }
