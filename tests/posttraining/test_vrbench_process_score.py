from scripts.eval.vrbench_process_score import cited_spans, score_question, step_spans


def test_step_spans_parse_ranges_and_single_timestamps_in_step_order() -> None:
    rp = {"2": "eyes filled with curiosity [00:08:28->00:09:09]", "1": "explore the ruins [00:07:44->00:08:28]",
          "3": "no timestamp here", "4": "a glance at 00:12:05"}
    out = step_spans(rp)
    assert out == [{"start_s": 464.0, "end_s": 508.0}, {"start_s": 508.0, "end_s": 549.0}, {"start_s": 725.0, "end_s": 726.0}]


def test_cited_spans_follow_clip_ranks_of_the_top_option_and_probes() -> None:
    catalog = [{"start_s": i * 30.0, "end_s": i * 30.0 + 30.0} for i in range(6)]
    rollout = {"evidence_chain": [{"label": "D", "clip_ranks": [2, 5]}, {"label": "A", "clip_ranks": [1]}],
               "probe_observations": [{"time_span": {"start_s": 100.0, "end_s": 104.0}}]}
    spans = cited_spans(rollout, indices=[0, 1, 2, 3, 4, 5], catalog_spans=catalog, top_options=1)
    assert spans == [catalog[1], catalog[4], {"start_s": 100.0, "end_s": 104.0}]


def test_score_question_recall_precision_iou() -> None:
    steps = [{"start_s": 0.0, "end_s": 30.0}, {"start_s": 100.0, "end_s": 130.0}]
    cited = [{"start_s": 10.0, "end_s": 40.0}, {"start_s": 500.0, "end_s": 530.0}]
    sc = score_question(steps, cited)
    assert sc["step_recall"] == 0.5 and sc["citation_precision"] == 0.5
    assert abs(sc["mean_best_iou"] - (20 / 40) / 2) < 1e-9
    assert score_question([], cited)["timed_steps"] == 0.0
