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
            "salient_objects": [],
            "place": {},
            "events": [],
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
