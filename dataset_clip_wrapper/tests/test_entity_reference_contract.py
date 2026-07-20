from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from dataset_clip_wrapper.l1_clue_graph.graph_composer import GraphComposer
from dataset_clip_wrapper.perception.clip_schema import _normalize_clip_schema_payload
from dataset_clip_wrapper.schemas import ClipSpan, GraphComposerConfig, RuntimeMode


class EntityReferenceContractTest(unittest.TestCase):
    def test_clip_mentions_receive_stable_grounded_ids(self) -> None:
        payload = {
            "scene_description": "A person holds a red box.",
            "observable_facts": [],
            "dialogue_spans": [],
            "entity_mentions": [
                {
                    "surface_form": "red box",
                    "entity_type": "OBJECT",
                    "attributes": {"color": "red", "intent": "gift"},
                }
            ],
            "state_assertions": [
                {
                    "subject_entity_index": 0,
                    "attribute": "openness",
                    "value": "closed",
                    "evidence_text": "The red box lid is closed.",
                    "confidence": 0.9,
                }
            ],
            "salient_objects": [],
            "place": {},
            "events": [
                {
                    "description": "A person holds the closed red box.",
                    "participant_entity_indices": [0],
                }
            ],
            "cross_clip_cues": [],
            "searchable_phrases": [],
            "uncertainty": "",
        }
        normalized = _normalize_clip_schema_payload(
            payload,
            clip_id="clip:demo:001",
            clip=ClipSpan(1.0, 2.0),
            model="fixture",
            attempt="full",
        )
        mention = normalized["entity_mentions"][0]

        self.assertEqual(mention["mention_id"], "clip:demo:001:entity:000")
        self.assertEqual(mention["entity_type"], "object")
        self.assertEqual(mention["attributes"], {"color": "red"})
        self.assertEqual(mention["evidence_refs"], ["clip:demo:001"])
        state = normalized["state_assertions"][0]
        self.assertEqual(state["subject_mention_id"], mention["mention_id"])
        self.assertEqual(state["attribute"], "openness")
        self.assertEqual(state["value"], "closed")
        self.assertEqual(
            normalized["events"][0]["participant_refs"],
            [mention["mention_id"]],
        )

    def test_structured_states_and_multi_participant_events_keep_references(self) -> None:
        clip_id = "clip:demo:001"
        box_mention = f"{clip_id}:entity:000"
        person_mention = f"{clip_id}:entity:001"
        state_id = f"{clip_id}:state:000"
        schemas = [
            {
                "clip_id": clip_id,
                "time_span": {"start_s": 0.0, "end_s": 1.0},
                "entity_mentions": [
                    {
                        "mention_id": box_mention,
                        "surface_form": "red box",
                        "entity_type": "object",
                        "attributes": {"color": "red"},
                    },
                    {
                        "mention_id": person_mention,
                        "surface_form": "person",
                        "entity_type": "person",
                        "attributes": {"clothing": "dark shirt"},
                    },
                ],
                "state_assertions": [
                    {
                        "state_id": state_id,
                        "subject_mention_id": box_mention,
                        "attribute": "openness",
                        "value": "closed",
                        "evidence_text": "The box is visibly closed.",
                    }
                ],
            }
        ]
        response = {
            "target_clip_id": clip_id,
            "target_nodes": [
                {
                    "node_id": "event-1",
                    "node_type": "event",
                    "text": "A person holds the red box.",
                    "participant_refs": [person_mention, box_mention],
                }
            ],
            "neighbor_edges": [],
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = Path(temp_dir) / "cache.jsonl"
            cache.write_text(
                json.dumps(
                    {
                        "target_clip_id": clip_id,
                        "model": "fixture",
                        "response": response,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            composer = GraphComposer(
                GraphComposerConfig(
                    model="fixture",
                    neighbor_cache_path=str(cache),
                ),
                client=object(),
            )
            graph, _, _ = composer._compose_neighbor_vlm_l1_graph(
                graph={"schema_version": "test", "nodes": [], "edges": []},
                example_id="example",
                video_id="demo",
                clip_schemas=schemas,
                mode=RuntimeMode.VIDEO_ONLY,
            )

        nodes = {node["node_id"]: node for node in graph["nodes"]}
        state = next(node for node in nodes.values() if node["node_type"] == "state")
        event = next(node for node in nodes.values() if node["node_type"] == "event")
        self.assertEqual(state["attribute"], "openness")
        self.assertEqual(state["value"], "closed")
        self.assertEqual(len(event["participant_refs"]), 2)
        self.assertTrue(
            any(
                edge["src"] == state["node_id"]
                and edge["dst"] == state["subject_ref"]
                and edge["edge_type"] == "state_of"
                for edge in graph["edges"]
            )
        )
        event_mentions = {
            edge["dst"]
            for edge in graph["edges"]
            if edge["src"] == event["node_id"]
            and edge["edge_type"] == "entity_mention"
        }
        self.assertEqual(event_mentions, set(event["participant_refs"]))

    def test_identity_edges_require_grounded_entity_endpoints(self) -> None:
        schemas = [
            {
                "clip_id": "clip:demo:001",
                "time_span": {"start_s": 0.0, "end_s": 1.0},
                "entity_mentions": [
                    {
                        "mention_id": "clip:demo:001:entity:000",
                        "surface_form": "red box",
                        "entity_type": "object",
                        "attributes": {"color": "red"},
                        "evidence_refs": ["clip:demo:001"],
                    }
                ],
            },
            {
                "clip_id": "clip:demo:002",
                "time_span": {"start_s": 1.0, "end_s": 2.0},
                "entity_mentions": [
                    {
                        "mention_id": "clip:demo:002:entity:000",
                        "surface_form": "red box",
                        "entity_type": "object",
                        "attributes": {"color": "red"},
                        "evidence_refs": ["clip:demo:002"],
                    }
                ],
            },
        ]
        responses = [
            {
                "target_clip_id": "clip:demo:001",
                "target_nodes": [
                    {"node_id": "obs", "node_type": "observation", "text": "box visible"}
                ],
                "neighbor_edges": [
                    {
                        "src_clip_id": "clip:demo:001",
                        "dst_clip_id": "clip:demo:002",
                        "src_node_id": "clip:demo:001:entity:000",
                        "dst_node_id": "clip:demo:002:entity:000",
                        "edge_type": "same_object",
                    },
                    {
                        "src_clip_id": "clip:demo:001",
                        "dst_clip_id": "clip:demo:002",
                        "edge_type": "same_object",
                    },
                ],
            },
            {
                "target_clip_id": "clip:demo:002",
                "target_nodes": [
                    {"node_id": "obs", "node_type": "observation", "text": "box visible"}
                ],
                "neighbor_edges": [],
            },
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = Path(temp_dir) / "cache.jsonl"
            cache.write_text(
                "".join(
                    json.dumps(
                        {
                            "target_clip_id": row["target_clip_id"],
                            "model": "fixture",
                            "response": row,
                        }
                    )
                    + "\n"
                    for row in responses
                ),
                encoding="utf-8",
            )
            composer = GraphComposer(
                GraphComposerConfig(model="fixture", neighbor_cache_path=str(cache)),
                client=object(),
            )
            graph, trace, _ = composer._compose_neighbor_vlm_l1_graph(
                graph={"schema_version": "test", "nodes": [], "edges": []},
                example_id="example",
                video_id="demo",
                clip_schemas=schemas,
                mode=RuntimeMode.VIDEO_ONLY,
            )

        identity_edges = [
            edge for edge in graph["edges"] if edge.get("edge_type") == "same_object"
        ]
        nodes = {node["node_id"]: node for node in graph["nodes"]}
        self.assertEqual(len(identity_edges), 1)
        self.assertEqual(nodes[identity_edges[0]["src"]]["node_type"], "entity_mention")
        self.assertEqual(nodes[identity_edges[0]["dst"]]["node_type"], "entity_mention")
        self.assertTrue(
            any(row.get("reason") == "identity_endpoint_not_entity_mention" for row in trace)
        )


if __name__ == "__main__":
    unittest.main()
