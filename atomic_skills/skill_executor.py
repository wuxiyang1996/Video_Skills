"""LLM/VLM-backed atomic skill execution.

This module provides `execute_skill()` which dispatches to either:
- rule backend: fast, deterministic (existing pure-Python skills)
- llm backend: real semantic reasoning via SkillModelClient
- vlm backend: perception via VLM with image inputs

Usage:
    from atomic_skills.skill_executor import SkillExecutor

    executor = SkillExecutor(
        llm_client=SkillModelClient.from_openrouter(model="qwen/qwen3.5-9b", api_key=key),
        vlm_client=SkillModelClient.from_openrouter(model="qwen/qwen3.5-9b", api_key=key),
        config=SkillBackendConfig(default_mode=SkillBackendMode.LLM),
    )
    result = executor.execute("infer_causal_relation", args={...}, graph=graph)
"""

from __future__ import annotations

import json
from typing import Any

from .common import make_result, stable_id, lexical_score, find_nodes, normalize_time_span, spans_overlap
from .skill_backends import SkillBackendConfig, SkillBackendMode, format_skill_prompt
from .skill_model_client import SkillModelClient


class SkillExecutor:
    """Dispatches skill execution to rule or LLM/VLM backend."""

    def __init__(
        self,
        *,
        llm_client: SkillModelClient | None = None,
        vlm_client: SkillModelClient | None = None,
        config: SkillBackendConfig | None = None,
    ):
        self.llm_client = llm_client
        self.vlm_client = vlm_client
        self.config = config or SkillBackendConfig()

    def execute(
        self,
        skill_id: str,
        *,
        args: dict[str, Any],
        graph: dict[str, Any] | None = None,
    ) -> Any:
        """Execute a skill with the configured backend."""
        mode = self.config.mode_for(skill_id)

        if mode == SkillBackendMode.RULE:
            return self._execute_rule(skill_id, args=args, graph=graph)

        if mode == SkillBackendMode.VLM:
            if self.vlm_client:
                return self._execute_vlm(skill_id, args=args, graph=graph)
            return self._execute_rule(skill_id, args=args, graph=graph)

        if mode == SkillBackendMode.LLM and not self.llm_client:
            return self._execute_rule(skill_id, args=args, graph=graph)

        if skill_id in self.config.vlm_skills:
            return self._execute_vlm(skill_id, args=args, graph=graph)
        elif skill_id in self.config.retrieval_skills:
            return self._execute_retrieval_llm(skill_id, args=args, graph=graph)
        else:
            return self._execute_llm(skill_id, args=args, graph=graph)

    def _execute_rule(self, skill_id: str, *, args: dict[str, Any], graph: dict[str, Any] | None) -> Any:
        """Fallback to the existing pure-Python rule functions."""
        from .reasoning_graph_assembly import skills as reasoning
        from .evidence_graph_construction import skills as evidence

        rule_fns = {
            "parse_question_target": lambda: reasoning.parse_question_target(args.get("question_text", ""), options=args.get("options")),
            "propose_evidence_roles": lambda: reasoning.propose_evidence_roles(args.get("question_text", ""), args.get("parsed_target", {}), task_family=args.get("task_family")),
            "retrieve_by_event": lambda: reasoning.retrieve_by_event(graph or {}, event_description=args.get("event_description", ""), time_range=args.get("time_range"), entity_filter=args.get("entity_filter")),
            "retrieve_by_entity": lambda: reasoning.retrieve_by_entity(graph or {}, entity_id=args.get("entity_id", ""), time_range=args.get("time_range"), predicate_filter=args.get("predicate_filter")),
            "retrieve_by_time": lambda: reasoning.retrieve_by_time(graph or {}, anchor_event_or_time=args.get("anchor_event_or_time", {"start_s": 0, "end_s": 30}), window_before=args.get("window_before", 30), window_after=args.get("window_after", 30)),
            "retrieve_by_relation": lambda: reasoning.retrieve_by_relation(graph or {}, source_node=args.get("source_node", ""), relation_type=args.get("relation_type", "temporal_next"), hop_limit=args.get("hop_limit", 1)),
            "localize_clue": lambda: reasoning.localize_clue(args.get("candidate_evidence") or (graph or {}).get("nodes", []), role_constraint=args.get("role_constraint", ""), question_context=args.get("question_context", "")),
            "extract_claim": lambda: reasoning.extract_claim(graph or {}, evidence_ref=args.get("evidence_ref", ""), speaker_hint=args.get("speaker_hint"), claim_query=args.get("claim_query")),
            "assign_evidence_role": lambda: reasoning.assign_evidence_role(graph or {}, evidence_ref=args.get("evidence_ref", ""), role_schema=args.get("role_schema", ""), question_context=args.get("question_context", "")),
            "generate_answer_hypotheses": lambda: reasoning.generate_answer_hypotheses(args.get("question_text", ""), options=args.get("options"), parsed_target=args.get("parsed_target")),
            "retrieve_evidence_for_hypothesis": lambda: reasoning.retrieve_evidence_for_hypothesis(graph or {}, hypothesis=args.get("hypothesis", ""), max_refs=args.get("max_refs", 6)),
            "score_hypothesis_support": lambda: reasoning.score_hypothesis_support(args.get("hypothesis", ""), support_evidence=args.get("support_evidence", []), counterevidence=args.get("counterevidence"), evidence_graph=graph),
            "compare_hypotheses": lambda: reasoning.compare_hypotheses(args.get("scored_hypotheses", []), decision_policy=args.get("decision_policy")),
            "bridge_evidence_hops": lambda: reasoning.bridge_evidence_hops(graph or {}, source_evidence=args.get("source_evidence", []), target_hypothesis=args.get("target_hypothesis", ""), allowed_hop_types=args.get("allowed_hop_types"), max_hops=args.get("max_hops", 2)),
            "verify_temporal_social_consistency": lambda: reasoning.verify_temporal_social_consistency(args.get("evidence_chain", {"evidence_refs": []}), hypothesis=args.get("hypothesis", ""), evidence_graph=graph),
            "compose_evidence_chain": lambda: reasoning.compose_evidence_chain(args.get("role_labeled_evidence", []), dependency_template=args.get("dependency_template", "")),
            "detect_missing_role": lambda: reasoning.detect_missing_role(args.get("evidence_chain", {"items": []}), required_roles=args.get("required_roles", [])),
            "search_counterevidence": lambda: reasoning.search_counterevidence(graph or {}, claim=args.get("claim", {}), supporting_evidence=args.get("supporting_evidence", []), search_scope=args.get("search_scope", "")),
            "infer_temporal_relation": lambda: reasoning.infer_temporal_relation(args.get("event_refs", []), evidence_graph=graph or {}),
            "infer_state_change": lambda: reasoning.infer_state_change(graph or {}, entity_or_object=args.get("entity_or_object", ""), state_predicate=args.get("state_predicate", ""), before_after_refs=args.get("before_after_refs", [])),
            "infer_causal_relation": lambda: reasoning.infer_causal_relation(args.get("candidate_cause", ""), args.get("candidate_effect", ""), evidence_chain=args.get("evidence_chain", {"evidence_refs": []})),
            "infer_intention_or_motive": lambda: reasoning.infer_intention_or_motive(args.get("agent", ""), args.get("actions", []), context_evidence=args.get("context_evidence", [])),
            "infer_social_contradiction": lambda: reasoning.infer_social_contradiction(args.get("claim_or_alibi", {}), evidence_chain=args.get("evidence_chain", {"evidence_refs": []}), counterevidence=args.get("counterevidence")),
            "verify_claim_support": lambda: reasoning.verify_claim_support(args.get("claim", ""), evidence_chain=args.get("evidence_chain", {"evidence_refs": []}), support_policy=args.get("support_policy"), evidence_graph=graph, question_text=args.get("question_text")),
            "commit_answer": lambda: reasoning.commit_answer(args.get("verified_claim", {}), options=args.get("options"), answer_format=args.get("answer_format", "free_text"), support_chain=args.get("support_chain", {"evidence_refs": []})),
            "segment_video_or_select_clip": lambda: evidence.segment_video_or_select_clip(graph, video_id=args.get("video_id", ""), clip_policy=args.get("clip_policy", {}), observation_end_s=args.get("observation_end_s")),
            "extract_observation": lambda: evidence.extract_observation(graph, clip_or_text_ref=args.get("clip_or_text_ref", ""), modality=args.get("modality", "text"), text=args.get("text", ""), time_span=args.get("time_span"), observation_query=args.get("observation_query")),
            "extract_dialogue_span": lambda: evidence.extract_dialogue_span(graph, subtitle_or_asr_ref=args.get("subtitle_or_asr_ref", ""), text=args.get("text", ""), time_span=args.get("time_span", {}), speaker_hint=args.get("speaker_hint")),
            "detect_entity_mention": lambda: evidence.detect_entity_mention(graph, observation_ref=args.get("observation_ref", ""), entity_type=args.get("entity_type"), text=args.get("text")),
            "resolve_entity_coreference": lambda: evidence.resolve_entity_coreference(graph, mention_nodes=args.get("mention_nodes", []), context_edges=args.get("context_edges")),
            "create_event_node": lambda: evidence.create_event_node(graph, observation_refs=args.get("observation_refs", []), event_description=args.get("event_description", ""), time_span=args.get("time_span", {})),
            "create_state_node": lambda: evidence.create_state_node(graph, entity_ref=args.get("entity_ref", ""), state_predicate=args.get("state_predicate", ""), evidence_refs=args.get("evidence_refs", []), state_value=args.get("state_value", ""), time_span=args.get("time_span")),
            "link_graph_relation": lambda: evidence.link_graph_relation(graph, source_node=args.get("source_node", ""), target_node=args.get("target_node", ""), edge_type=args.get("edge_type", ""), evidence_refs=args.get("evidence_refs")),
            "assign_provenance_trust": lambda: evidence.assign_provenance_trust(graph, node_or_edge_ref=args.get("node_or_edge_ref", ""), source_ref=args.get("source_ref", ""), mode=args.get("mode", "video_only"), trust_policy=args.get("trust_policy", {})),
        }

        fn = rule_fns.get(skill_id)
        if fn:
            return fn()
        return make_result(skill_id, ok=False, failure_code="unknown_skill_id")

    def _execute_llm(self, skill_id: str, *, args: dict[str, Any], graph: dict[str, Any] | None) -> Any:
        """Execute a reasoning skill via LLM call."""
        evidence_text = self._gather_evidence_text(args, graph)
        prompt_kwargs = self._build_prompt_kwargs(skill_id, args, evidence_text)
        prompt = format_skill_prompt(skill_id, **prompt_kwargs)

        if not prompt:
            return self._execute_rule(skill_id, args=args, graph=graph)

        try:
            response = self.llm_client.reason(prompt)
        except Exception as exc:  # noqa: BLE001 — keep GRPO collect alive on flaky OpenRouter
            rule_result = self._execute_rule(skill_id, args=args, graph=graph)
            rule_result.messages.append(f"llm_exception_fallback_to_rule:{type(exc).__name__}")
            return rule_result

        if response.get("parse_error"):
            rule_result = self._execute_rule(skill_id, args=args, graph=graph)
            msg = "llm_timeout_fallback_to_rule" if response.get("timeout") else "llm_parse_error_fallback_to_rule"
            rule_result.messages.append(msg)
            return rule_result

        return self._llm_response_to_result(skill_id, response, args, graph)

    def _execute_retrieval_llm(self, skill_id: str, *, args: dict[str, Any], graph: dict[str, Any] | None) -> Any:
        """Execute retrieval skills with LLM-based semantic scoring."""
        if not graph or not graph.get("nodes"):
            return self._execute_rule(skill_id, args=args, graph=graph)

        query = args.get("event_description") or args.get("entity_id") or args.get("search_scope") or ""
        if isinstance(args.get("hypothesis"), dict):
            query = args["hypothesis"].get("claim_text") or query
        elif isinstance(args.get("hypothesis"), str):
            query = args["hypothesis"] or query

        nodes_text = []
        for node in graph.get("nodes", [])[:20]:
            text = node.get("text") or node.get("event_description") or node.get("state_value") or ""
            nodes_text.append(f"- [{node.get('node_id')}] {text[:100]}")

        prompt = (
            f"Given query: \"{query}\"\n\n"
            f"Which of these evidence nodes are most relevant? Return the node_ids.\n\n"
            f"Nodes:\n" + "\n".join(nodes_text) + "\n\n"
            f"Answer with JSON: {{\"relevant_ids\": [...], \"scores\": {{\"node_id\": score}}}}"
        )

        response = self.llm_client.reason(prompt)

        if response.get("parse_error"):
            return self._execute_rule(skill_id, args=args, graph=graph)

        relevant_ids = response.get("relevant_ids") or []
        node_map = {n.get("node_id"): n for n in graph.get("nodes", []) if n.get("node_id")}
        refs = [nid for nid in relevant_ids if nid in node_map]

        if not refs:
            return self._execute_rule(skill_id, args=args, graph=graph)

        return make_result(
            skill_id,
            {"evidence_refs": refs, "retrieval_scores": response.get("scores", {}), "backend": "llm"},
            refs,
            ok=True,
            confidence=0.8,
        )

    def _execute_vlm(self, skill_id: str, *, args: dict[str, Any], graph: dict[str, Any] | None) -> Any:
        """Execute perception skills via VLM call with video clip frames.

        VLM perception works on video clips, not static images:
        1. Resolve clip_or_text_ref to find the clip node with time_span
        2. Sample frames from the video within that time_span
        3. Send frames (as base64 data URIs) + prompt to VLM
        4. Parse structured observations from VLM response
        """
        if not self.vlm_client:
            return self._execute_rule(skill_id, args=args, graph=graph)

        clip_ref = args.get("clip_or_text_ref") or args.get("observation_ref") or args.get("subtitle_or_asr_ref") or ""
        time_span = args.get("time_span")
        modality = args.get("modality", "visual")
        observation_query = args.get("observation_query") or args.get("text") or ""
        entity_type = args.get("entity_type")

        clip_node = None
        if graph and clip_ref:
            clip_node = next((n for n in graph.get("nodes", []) if n.get("node_id") == clip_ref), None)
            if clip_node and not time_span:
                time_span = clip_node.get("time_span")

        image_urls = self._sample_clip_frames(args, graph, clip_ref, time_span)

        if skill_id == "extract_observation":
            prompt = (
                f"You are observing a video clip"
                + (f" from {time_span['start_s']:.1f}s to {time_span['end_s']:.1f}s" if time_span else "")
                + ".\n"
                + (f"Focus on: {observation_query}\n" if observation_query else "")
                + f"Modality: {modality}\n\n"
                "Describe ALL observable facts: actions, objects, persons, spatial relations, "
                "state changes, and dialogue/text if visible.\n"
                "Return JSON: {\"observations\": [{\"text\": \"...\", \"modality\": \"visual|subtitle|audio\", "
                "\"entities\": [...], \"confidence\": 0.0-1.0}], \"scene_description\": \"...\"}"
            )
        elif skill_id == "extract_dialogue_span":
            speaker_hint = args.get("speaker_hint") or ""
            prompt = (
                f"You are observing a video clip"
                + (f" from {time_span['start_s']:.1f}s to {time_span['end_s']:.1f}s" if time_span else "")
                + ".\n"
                + (f"Speaker hint: {speaker_hint}\n" if speaker_hint else "")
                + "Identify all dialogue/speech in this clip.\n"
                "Return JSON: {\"dialogue_spans\": [{\"speaker\": \"...\", \"utterance\": \"...\", "
                "\"start_s\": 0.0, \"end_s\": 0.0, \"confidence\": 0.0-1.0}]}"
            )
        elif skill_id == "detect_entity_mention":
            prompt = (
                f"You are observing a video clip"
                + (f" from {time_span['start_s']:.1f}s to {time_span['end_s']:.1f}s" if time_span else "")
                + ".\n"
                + (f"Focus on entity type: {entity_type}\n" if entity_type else "")
                + "Detect all person, object, place, and speaker mentions visible in this clip.\n"
                "Return JSON: {\"entities\": [{\"surface_form\": \"...\", \"entity_type\": "
                "\"person|object|place|speaker\", \"first_appearance_s\": 0.0, \"confidence\": 0.0-1.0}]}"
            )
        else:
            return self._execute_rule(skill_id, args=args, graph=graph)

        response = self.vlm_client.perceive(prompt, image_urls=image_urls)

        if response.get("parse_error"):
            return self._execute_rule(skill_id, args=args, graph=graph)

        return self._vlm_response_to_result(skill_id, response, args, graph, clip_ref, time_span)

    def _sample_clip_frames(
        self,
        args: dict[str, Any],
        graph: dict[str, Any] | None,
        clip_ref: str,
        time_span: dict[str, Any] | None,
    ) -> list[str]:
        """Sample frames from a video clip, returning base64 data URIs or paths.

        Checks for:
        1. Pre-extracted frame paths/URLs in args
        2. representative_frame in clip node
        3. Video path + time_span for dynamic frame sampling
        """
        if args.get("image_urls"):
            return args["image_urls"]
        if args.get("frame_paths"):
            return args["frame_paths"]
        if args.get("frame_data_uris"):
            return args["frame_data_uris"]

        if graph:
            clip_node = next((n for n in graph.get("nodes", []) if n.get("node_id") == clip_ref), None)
            if clip_node:
                rep_frame = clip_node.get("representative_frame")
                if rep_frame and rep_frame.get("image_url"):
                    return [rep_frame["image_url"]]

        video_path = args.get("video_path")
        if not video_path and graph:
            video_path = graph.get("video_path") or graph.get("primary_path")

        if video_path and time_span:
            return self._extract_frames_from_video(video_path, time_span)

        return []

    def _extract_frames_from_video(
        self, video_path: str, time_span: dict[str, Any], num_frames: int = 4
    ) -> list[str]:
        """Extract frames from video at evenly-spaced timestamps within time_span."""
        try:
            import cv2
            import base64
        except ImportError:
            return []

        from pathlib import Path
        path = Path(video_path)
        if not path.exists():
            return []

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            return []

        start_s = float(time_span.get("start_s", 0))
        end_s = float(time_span.get("end_s", start_s + 5))
        span = max(end_s - start_s, 0.1)

        times = [start_s + (span * i / max(num_frames - 1, 1)) for i in range(num_frames)]
        frames: list[str] = []

        for t in times:
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
            ok, frame = cap.read()
            if ok:
                ok_enc, buf = cv2.imencode(".jpg", frame)
                if ok_enc:
                    data = base64.b64encode(buf.tobytes()).decode("ascii")
                    frames.append(f"data:image/jpeg;base64,{data}")

        cap.release()
        return frames

    def _vlm_response_to_result(
        self,
        skill_id: str,
        response: dict[str, Any],
        args: dict[str, Any],
        graph: dict[str, Any] | None,
        clip_ref: str,
        time_span: dict[str, Any] | None,
    ) -> Any:
        """Convert VLM response to SkillResult for perception skills."""
        if skill_id == "extract_observation":
            observations = response.get("observations", [])
            obs_nodes = []
            for i, obs in enumerate(observations):
                obs_text = obs.get("text") if isinstance(obs, dict) else str(obs)
                obs_modality = obs.get("modality", "visual") if isinstance(obs, dict) else "visual"
                conf = float(obs.get("confidence", 0.8)) if isinstance(obs, dict) else 0.8
                node_id = stable_id("evidence.observation", clip_ref, obs_modality, obs_text, time_span)
                node = {
                    "node_id": node_id,
                    "node_type": "observation",
                    "source_ids": [clip_ref],
                    "modality": obs_modality,
                    "text": obs_text,
                    "time_span": time_span,
                    "confidence": conf,
                    "provenance": {"created_by": "extract_observation", "backend": "vlm"},
                }
                obs_nodes.append(node)

            refs = [n["node_id"] for n in obs_nodes]
            return make_result(
                skill_id,
                {
                    "graph": graph,
                    "observation_nodes": obs_nodes,
                    "evidence_refs": refs,
                    "scene_description": response.get("scene_description", ""),
                    "backend": "vlm",
                },
                refs,
                ok=bool(refs),
                confidence=0.8,
            )

        elif skill_id == "extract_dialogue_span":
            dialogue_spans = response.get("dialogue_spans", [])
            if not dialogue_spans:
                return make_result(skill_id, ok=False, failure_code="empty_dialogue", messages=["VLM found no dialogue"])

            first = dialogue_spans[0]
            speaker = first.get("speaker") or args.get("speaker_hint") or "unknown"
            utterance = first.get("utterance", "")
            node_id = stable_id("evidence.dialogue", clip_ref, speaker, utterance)
            node = {
                "node_id": node_id,
                "node_type": "dialogue_span",
                "source_ids": [clip_ref],
                "speaker": speaker,
                "text": utterance,
                "time_span": time_span,
                "provenance": {"created_by": "extract_dialogue_span", "backend": "vlm"},
            }
            return make_result(
                skill_id,
                {"graph": graph, "dialogue_span_node": node, "speaker_mention": speaker, "evidence_ref": node_id, "all_spans": dialogue_spans, "backend": "vlm"},
                [node_id],
                ok=True,
                confidence=float(first.get("confidence", 0.7)),
            )

        elif skill_id == "detect_entity_mention":
            entities = response.get("entities", [])
            if not entities:
                return make_result(skill_id, ok=False, failure_code="no_entity_mentions", messages=["VLM found no entities"])

            mention_nodes = []
            for ent in entities:
                surface = ent.get("surface_form") or str(ent)
                etype = ent.get("entity_type", "unknown")
                node_id = stable_id("evidence.mention", clip_ref, surface, etype)
                mention_nodes.append({
                    "node_id": node_id,
                    "node_type": "entity_mention",
                    "surface_form": surface,
                    "entity_type": etype,
                    "source_ids": [clip_ref],
                    "time_span": time_span,
                    "confidence": float(ent.get("confidence", 0.7)),
                    "provenance": {"created_by": "detect_entity_mention", "backend": "vlm"},
                })

            refs = [n["node_id"] for n in mention_nodes]
            return make_result(
                skill_id,
                {"mention_nodes": mention_nodes, "surface_forms": [n["surface_form"] for n in mention_nodes], "time_spans": [time_span] * len(mention_nodes), "backend": "vlm"},
                refs,
                ok=True,
                confidence=0.75,
            )

        return self._execute_rule(skill_id, args=args, graph=graph)

    def _gather_evidence_text(self, args: dict[str, Any], graph: dict[str, Any] | None) -> str:
        """Collect evidence text from refs in args or graph."""
        refs = args.get("evidence_refs") or args.get("supporting_evidence") or args.get("context_evidence") or []
        if isinstance(refs, str):
            refs = [refs]
        chain = args.get("evidence_chain")
        if isinstance(chain, dict):
            refs = refs or chain.get("evidence_refs", [])

        if not refs or not graph:
            return ""

        node_map = {n.get("node_id"): n for n in graph.get("nodes", []) if n.get("node_id")}
        texts = []
        for ref in refs[:10]:
            node = node_map.get(ref)
            if node:
                texts.append(node.get("text") or node.get("event_description") or node.get("state_value") or "")
        return " | ".join(t for t in texts if t)

    def _build_prompt_kwargs(self, skill_id: str, args: dict[str, Any], evidence_text: str) -> dict[str, Any]:
        """Map skill args to prompt template kwargs."""
        kwargs: dict[str, Any] = {"evidence": evidence_text}

        if skill_id == "infer_causal_relation":
            kwargs["cause"] = args.get("candidate_cause", "")
            kwargs["effect"] = args.get("candidate_effect", "")
        elif skill_id == "infer_temporal_relation":
            kwargs["events"] = args.get("event_refs", [])
        elif skill_id == "infer_state_change":
            kwargs["entity"] = args.get("entity_or_object", "")
            kwargs["predicate"] = args.get("state_predicate", "")
            kwargs["before"] = args.get("before_after_refs", [""])[0] if args.get("before_after_refs") else ""
            kwargs["after"] = args.get("before_after_refs", ["", ""])[1] if len(args.get("before_after_refs", [])) > 1 else ""
        elif skill_id == "infer_intention_or_motive":
            kwargs["agent"] = args.get("agent", "")
            kwargs["actions"] = args.get("actions", [])
            kwargs["context"] = evidence_text
        elif skill_id == "infer_social_contradiction":
            claim = args.get("claim_or_alibi", {})
            kwargs["claim"] = claim.get("claim_text") or claim.get("text") or str(claim)
            kwargs["counter"] = args.get("counterevidence", [])
        elif skill_id == "verify_claim_support":
            claim = args.get("claim", "")
            kwargs["claim"] = claim if isinstance(claim, str) else claim.get("claim_text") or claim.get("text") or str(claim)
            kwargs["question"] = args.get("question_text") or (claim.get("question_text") if isinstance(claim, dict) else "")
        elif skill_id == "score_hypothesis_support":
            h = args.get("hypothesis", "")
            kwargs["hypothesis"] = h if isinstance(h, str) else ((h or {}).get("claim_text") if isinstance(h, dict) else str(h or ""))
            kwargs["support"] = args.get("support_evidence", [])
            kwargs["counter"] = args.get("counterevidence", [])
        elif skill_id == "compare_hypotheses":
            kwargs["hypotheses"] = args.get("scored_hypotheses", [])
        elif skill_id == "generate_answer_hypotheses":
            kwargs["question"] = args.get("question_text", "")
            kwargs["options"] = args.get("options") or []
            kwargs["seed"] = (
                args.get("seed")
                or args.get("grpo_seed")
                or getattr(self.llm_client, "seed", None)
                or 0
            )
        elif skill_id == "localize_clue":
            kwargs["role"] = args.get("role_constraint", "")
            kwargs["question"] = args.get("question_context", "")
            kwargs["candidates"] = args.get("candidate_evidence", [])
        elif skill_id == "extract_claim":
            kwargs["text"] = args.get("text") or ""
            kwargs["query"] = args.get("claim_query") or ""
        elif skill_id == "assign_evidence_role":
            kwargs["role"] = args.get("role_schema", "")
            kwargs["question"] = args.get("question_context", "")
            kwargs["text"] = args.get("text") or ""
        elif skill_id == "verify_temporal_social_consistency":
            h = args.get("hypothesis", "")
            kwargs["hypothesis"] = h if isinstance(h, str) else h.get("claim_text") or str(h)
            kwargs["chain"] = args.get("evidence_chain", {})
        elif skill_id == "bridge_evidence_hops":
            kwargs["sources"] = args.get("source_evidence", [])
            h = args.get("target_hypothesis", "")
            kwargs["target"] = h if isinstance(h, str) else h.get("claim_text") or str(h)
            kwargs["intermediates"] = []

        return kwargs

    def _llm_response_to_result(
        self, skill_id: str, response: dict[str, Any], args: dict[str, Any], graph: dict[str, Any] | None
    ) -> Any:
        """Convert LLM JSON response into a SkillResult."""
        confidence = float(response.get("confidence") or response.get("score") or response.get("support_score") or 0.7)

        if skill_id == "infer_causal_relation":
            ok = bool(response.get("causal", True))
            return make_result(skill_id, {
                "causal_claim": response.get("reasoning") or f"{args.get('candidate_cause')} caused {args.get('candidate_effect')}",
                "supporting_roles": [], "backend": "llm",
            }, confidence=confidence, ok=ok)

        elif skill_id == "infer_temporal_relation":
            return make_result(skill_id, {
                "temporal_relation": response.get("relation", "before"),
                "supporting_evidence": args.get("event_refs", []),
                "backend": "llm",
            }, args.get("event_refs", []), confidence=confidence)

        elif skill_id == "infer_state_change":
            ok = bool(response.get("changed", True))
            return make_result(skill_id, {
                "state_change_claim": response.get("reasoning") or f"state changed",
                "before_state": response.get("before_state", ""),
                "after_state": response.get("after_state", ""),
                "backend": "llm",
            }, confidence=confidence, ok=ok)

        elif skill_id == "infer_intention_or_motive":
            return make_result(skill_id, {
                "intention_claim": response.get("intention", ""),
                "alternatives": response.get("alternatives", []),
                "supporting_roles": [], "backend": "llm",
            }, confidence=confidence, ok=bool(response.get("intention")))

        elif skill_id == "infer_social_contradiction":
            ok = bool(response.get("contradicted", False))
            return make_result(skill_id, {
                "contradiction_claim": response.get("contradiction_claim", ""),
                "supporting_evidence": [], "backend": "llm",
            }, confidence=confidence, ok=ok)

        elif skill_id == "verify_claim_support":
            ok = bool(response.get("supported", False))
            score = float(response.get("score", 0.5))
            claim = args.get("claim", "")
            claim_text = claim if isinstance(claim, str) else claim.get("claim_text") or claim.get("text") or str(claim)
            option_label = None if isinstance(claim, str) else claim.get("option_label")
            refs = []
            chain = args.get("evidence_chain")
            if isinstance(chain, dict):
                refs = chain.get("evidence_refs") or []
            rule_result = self._execute_rule(skill_id, args=args, graph=graph) if graph and refs else None
            if rule_result is not None:
                rule_score = float(rule_result.outputs.get("verification_score", 0.0))
                rule_claim_score = float(rule_result.outputs.get("claim_support_score", 0.0))
                rule_target_score = float(rule_result.outputs.get("target_alignment_score", 0.0))
                support_policy = args.get("support_policy") if isinstance(args.get("support_policy"), dict) else {}
                allow_llm_alignment_override = bool(support_policy.get("allow_llm_target_alignment_override"))
                llm_target_aligned = bool(response.get("target_aligned", ok))
                override_min_score = float(support_policy.get("llm_alignment_override_min_score") or 0.75)
                can_override_alignment = (
                    allow_llm_alignment_override
                    and ok
                    and llm_target_aligned
                    and score >= override_min_score
                    and refs
                )
                if not rule_result.ok and not can_override_alignment:
                    ok = False
                    score = min(score, rule_score)
                elif ok:
                    score = max(score, rule_score)
                if rule_claim_score < 0.05:
                    ok = False
                if rule_target_score < 0.05 and not can_override_alignment:
                    ok = False
                if can_override_alignment:
                    rule_target_score = max(rule_target_score, score)
            return make_result(skill_id, {
                "verification_score": score,
                "passed": ok,
                "failure_code": None if ok else "insufficient_evidence",
                "messages": [response.get("reasoning", "")],
                "claim_support_score": (
                    rule_result.outputs.get("claim_support_score")
                    if rule_result is not None
                    else score
                ),
                "target_alignment_score": (
                    rule_result.outputs.get("target_alignment_score")
                    if rule_result is not None
                    else (score if response.get("target_aligned", ok) else 0.0)
                ),
                "verified_claim": {
                    "claim_text": claim_text,
                    "text": claim_text,
                    "option_label": option_label,
                    "question_text": args.get("question_text") or (claim.get("question_text") if isinstance(claim, dict) else None),
                    "claim_status": "verified" if ok else "insufficient",
                    "supported_by_refs": refs if ok else [],
                },
                "backend": "llm",
            }, refs if ok else [], confidence=score, ok=ok, failure_code=None if ok else "insufficient_evidence")

        elif skill_id == "score_hypothesis_support":
            support_score = float(response.get("support_score", 0.5))
            contradiction_score = float(response.get("contradiction_score", 0.0))
            support_arg = args.get("support_evidence", [])
            if isinstance(support_arg, dict):
                support_refs = support_arg.get("support_refs") or support_arg.get("evidence_refs") or []
            else:
                support_refs = support_arg if isinstance(support_arg, list) else []
            counter_refs = args.get("counterevidence") or []
            hypothesis = args.get("hypothesis")
            claim_text = hypothesis.get("claim_text") if isinstance(hypothesis, dict) else str(hypothesis or "")
            option_label = hypothesis.get("option_label") if isinstance(hypothesis, dict) else None
            rule_result = self._execute_rule(skill_id, args=args, graph=graph) if graph and support_refs else None
            if rule_result is not None:
                rule_scored = rule_result.outputs.get("scored_hypothesis") or {}
                rule_support = float(rule_scored.get("support_score", 0.0))
                if rule_support < 0.05:
                    support_score = min(support_score, 0.05)
                else:
                    support_score = max(support_score, rule_support)
            return make_result(skill_id, {
                "scored_hypothesis": {
                    "hypothesis": hypothesis,
                    "claim_text": claim_text,
                    "option_label": option_label,
                    "support_score": support_score,
                    "contradiction_score": contradiction_score,
                    "overall_score": max(0, support_score - contradiction_score),
                    "support_refs": support_refs,
                    "counterevidence_refs": counter_refs,
                    "backend": "llm",
                },
                "backend": "llm",
            }, support_refs + counter_refs, confidence=support_score, ok=support_score > 0)

        elif skill_id == "compare_hypotheses":
            best_label = str(response.get("best_label") or "").strip()
            margin = float(response.get("margin", 0.0))
            scored = list(args.get("scored_hypotheses") or [])
            policy = args.get("decision_policy") if isinstance(args.get("decision_policy"), dict) else {}
            # If scored options are near-tied, rotate with explore_seed instead of
            # letting a single LLM label collapse all K samples.
            if policy.get("explore_seed") is not None and scored:
                scores = [
                    float(item.get("overall_score") or 0.0)
                    for item in scored
                    if isinstance(item, dict)
                ]
                tie_eps = float(policy.get("tie_epsilon", 0.2))
                if scores and sum(1 for s in scores if max(scores) - s <= tie_eps) > 1:
                    from .reasoning_graph_assembly.skills import compare_hypotheses as _rule_compare

                    explored = _rule_compare(scored, decision_policy=policy)
                    out = dict(explored.outputs or {})
                    out["backend"] = "llm+explore"
                    out["llm_best_label"] = best_label
                    out["llm_margin"] = margin
                    out["llm_reasoning"] = response.get("reasoning", "")
                    best_row = out.get("best_hypothesis") or {}
                    refs = list(best_row.get("support_refs") or [])
                    return make_result(
                        skill_id,
                        out,
                        refs,
                        confidence=float(best_row.get("overall_score") or margin or 0.0),
                        ok=bool(best_row.get("option_label") or best_row.get("claim_text")),
                    )
            best: dict[str, Any] | None = None
            for item in scored:
                if not isinstance(item, dict):
                    continue
                hyp = item.get("hypothesis") if isinstance(item.get("hypothesis"), dict) else {}
                label = str(item.get("option_label") or hyp.get("option_label") or "").strip()
                if best_label and label == best_label:
                    best = dict(item)
                    break
            if best is None and scored:
                # Fall back to highest overall_score among provided scored hypotheses.
                best = max(
                    (item for item in scored if isinstance(item, dict)),
                    key=lambda item: float(item.get("overall_score") or 0.0),
                    default=None,
                )
            if best is None:
                return make_result(
                    skill_id,
                    {
                        "best_hypothesis": {"option_label": best_label, "overall_score": margin},
                        "eliminated_hypotheses": [],
                        "decision_reason": response.get("reasoning", ""),
                        "score_margin": margin,
                        "backend": "llm",
                    },
                    confidence=margin,
                    ok=bool(best_label),
                )
            best = dict(best)
            best["option_label"] = best.get("option_label") or best_label or (
                (best.get("hypothesis") or {}).get("option_label") if isinstance(best.get("hypothesis"), dict) else None
            )
            best["overall_score"] = float(best.get("overall_score") or margin or 0.0)
            best["backend"] = "llm"
            refs = list(best.get("support_refs") or [])
            return make_result(
                skill_id,
                {
                    "best_hypothesis": best,
                    "eliminated_hypotheses": [
                        item
                        for item in scored
                        if isinstance(item, dict) and item is not best
                    ],
                    "decision_reason": response.get("reasoning", ""),
                    "score_margin": margin,
                    "backend": "llm",
                },
                refs,
                confidence=float(best.get("overall_score") or margin or 0.0),
                ok=bool(best.get("option_label") or best.get("claim_text")),
            )

        elif skill_id == "generate_answer_hypotheses":
            # Start from deterministic option→hypothesis mapping, then apply LLM rank/priors.
            rule = self._execute_rule(skill_id, args=args, graph=graph)
            hypotheses = list((rule.outputs or {}).get("hypotheses") or [])
            ranked = [str(x).strip() for x in (response.get("ranked_labels") or []) if str(x).strip()]
            priors = response.get("priors") if isinstance(response.get("priors"), dict) else {}
            by_label = {
                str(h.get("option_label") or "").strip(): h
                for h in hypotheses
                if isinstance(h, dict)
            }
            ordered: list[dict[str, Any]] = []
            for label in ranked:
                if label in by_label:
                    hyp = dict(by_label.pop(label))
                    if label in priors:
                        try:
                            hyp["prior_score"] = float(priors[label])
                        except (TypeError, ValueError):
                            pass
                    ordered.append(hyp)
            ordered.extend(by_label.values())
            if not ordered:
                ordered = [h for h in hypotheses if isinstance(h, dict)]
            return make_result(
                skill_id,
                {
                    "hypotheses": ordered,
                    "ranked_labels": ranked,
                    "priors": priors,
                    "backend": "llm",
                },
                confidence=0.8,
                ok=bool(ordered),
            )

        elif skill_id == "localize_clue":
            refs = response.get("best_refs", [])
            return make_result(skill_id, {
                "clue_refs": refs, "clue_spans": [], "confidence": confidence, "backend": "llm",
            }, refs, confidence=confidence, ok=bool(refs))

        elif skill_id == "extract_claim":
            claim_text = response.get("claim_text", "")
            ref = args.get("evidence_ref", "")
            return make_result(skill_id, {
                "claim_id": stable_id("claim", ref, claim_text),
                "claim_text": claim_text,
                "speaker": response.get("speaker"),
                "evidence_ref": ref,
                "backend": "llm",
            }, [ref] if ref else [], confidence=confidence, ok=bool(claim_text))

        elif skill_id == "assign_evidence_role":
            ok = bool(response.get("fits_role", True))
            ref = args.get("evidence_ref", "")
            return make_result(skill_id, {
                "role_labeled_evidence": {"evidence_ref": ref, "role": args.get("role_schema", ""), "text": "", "confidence": confidence},
                "role_confidence": confidence,
                "backend": "llm",
            }, [ref] if ref else [], confidence=confidence, ok=ok)

        elif skill_id == "verify_temporal_social_consistency":
            temporal_ok = bool(response.get("temporal_ok", True))
            social_ok = bool(response.get("social_ok", True))
            conflicts = response.get("conflicts", [])
            chain = args.get("evidence_chain") if isinstance(args.get("evidence_chain"), dict) else {}
            refs = chain.get("evidence_refs") or []
            return make_result(skill_id, {
                "temporal_ok": temporal_ok,
                "social_plausibility_ok": social_ok,
                "conflicts": conflicts,
                "backend": "llm",
            }, refs, confidence=1.0 if not conflicts else 0.4, ok=not conflicts)

        elif skill_id == "bridge_evidence_hops":
            path = response.get("bridge_path", [])
            return make_result(skill_id, {
                "multi_hop_chain": {"evidence_refs": path, "path_edges": [], "target_claim": ""},
                "backend": "llm",
            }, path, confidence=confidence, ok=len(path) >= 2)

        return self._execute_rule(skill_id, args=args, graph=graph)
