import json

from trainer.reader.build_sft_data import build_rows


def test_only_correct_teacher_rationales_are_kept(tmp_path, monkeypatch):
    ex = {"example_id": "e1", "question": {"question_text": "Q", "answer": {"label": "B"},
                                          "options": [{"label": "A", "text": "a"}, {"label": "B", "text": "b"}]},
          "metadata": {"clip_schemas": [{"time_span": {"start_s": 0, "end_s": 4}, "scene_description": "x"}]}, "hidden_supervision": {}}
    p = tmp_path / "e1.json"; p.write_text(json.dumps(ex))
    index = {"e1": {"path": str(p)}}
    rollouts = [{"example_id": "e1", "gold_label": "B", "rollout": {"final_answer": {"label": "B"}, "thinking": "clip 1 shows b"}},
                {"example_id": "e1", "gold_label": "B", "rollout": {"final_answer": {"label": "A"}, "thinking": "clip 1 shows a"}},
                {"example_id": "e1", "gold_label": "B", "rollout": {"final_answer": {"label": "B"}, "thinking": ""}},
                {"example_id": "zz", "rollout": {}}]
    rows, stats = build_rows(index, rollouts)
    assert stats == {"rollouts": 4, "no_example": 1, "wrong": 1, "low_precision": 0, "empty_reasoning": 1, "kept": 1}
    assert rows[0]["messages"][0]["role"] == "system" and json.loads(rows[0]["completion"])["label"] == "B"
