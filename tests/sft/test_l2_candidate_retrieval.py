from dataset_clip_wrapper.training.evaluate_l2_candidate_retrieval import (
    boundary_hybrid_candidates,
    candidate_metrics,
    document_text,
    query_text,
)


def test_candidate_metrics_and_visible_text_formatting() -> None:
    metrics = candidate_metrics([3, 1, 2], [2, 4])
    assert metrics["4"] == {"hit": True, "recall": 0.5}
    query = query_text({"question_text": "What?", "options": [{"text": "A"}]}, include_options=True)
    assert "What?" in query and "Answer options: A" in query
    document = document_text({"scene_description": "scene", "observable_facts": ["fact"]})
    assert document == "scene\nfact"
    assert boundary_hybrid_candidates(list(range(10, 42)), 50) == list(range(10, 40)) + [0, 49]
    assert len(boundary_hybrid_candidates(list(range(32)), 32)) == 32
