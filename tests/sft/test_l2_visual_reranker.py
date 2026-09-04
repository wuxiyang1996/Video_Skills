from dataset_clip_wrapper.training.evaluate_l2_visual_reranker import (
    aggregate_rankings,
    process_in_batches,
)


def test_aggregate_rankings_uses_strict_top2_predictions() -> None:
    report = aggregate_rankings([
        {"predicted": [1, 2], "gold": [2]},
        {"predicted": [3, 4], "gold": [5]},
    ])
    assert report["hit_rate"] == 0.5
    assert report["mean_recall"] == 0.5
    assert report["mean_precision"] == 0.25


def test_process_in_batches_preserves_pair_order() -> None:
    class Inputs(list):
        def to(self, device: str) -> "Inputs":
            assert device == "cuda"
            return self

    class Model:
        device = "cuda"

    class Reranker:
        default_instruction = "default"
        fps = 1.0
        max_frames = None
        model = Model()

        def format_mm_instruction(self, *args, **kwargs):
            return args[3]

        def tokenize(self, pairs):
            return Inputs(pairs)

        def compute_scores(self, inputs):
            return [f"score:{value}" for value in inputs]

    payload = {
        "instruction": "rank",
        "query": {"text": "query"},
        "documents": [{"text": str(index)} for index in range(5)],
    }
    assert process_in_batches(Reranker(), payload, 2) == [
        "score:0", "score:1", "score:2", "score:3", "score:4",
    ]
