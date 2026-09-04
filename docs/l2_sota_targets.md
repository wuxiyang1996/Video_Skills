# L2 SOTA targets

Goal: reach state of the art on **both** CG-Bench and Video-Holmes at the ~9B
scale. Every number below is measured, not assumed; the measurement scripts are
named so each target can be re-checked.

## Where the bar is

Published figures for the comparison points.

| Benchmark | Metric | Best ~7-8B | Best published overall |
| --- | --- | ---: | ---: |
| CG-Bench | mIoU | 1.63 (LLaVA-OneVision-7B) | 3.58 (Qwen2-VL-72B) |
| CG-Bench | rec.@IoU | 2.89 (ViLA-8B) | 5.32 (Qwen2-VL-72B) |
| CG-Bench | acc.@IoU | 1.35 (ViLA-8B) | 3.31 (Qwen2-VL-72B) |
| Video-Holmes | accuracy | 27.8 (Qwen2.5-VL-7B) | 45.0 (Gemini-2.5-Pro) |

## Where we are

CG-Bench numbers are from `dataset_clip_wrapper.training.cg_bench_official_metrics`,
which vendors the benchmark's own `calculate_intervals_iou`. They come from a
67-video subset whose catalog is still 43% placeholder rows, so they are a lower
bound and are **not yet comparable** to the table above.

| Metric | SFT | OPD | Oracle over the same 30s clips |
| --- | ---: | ---: | ---: |
| mIoU (k=3) | 2.79 | 3.64 | 31.56 |
| rec.@IoU (k=3) | 2.39 | 4.18 | 49.25 |
| rec@0.5 | 0.00 | 0.00 | 25.37 |

Video-Holmes has no answer-accuracy evaluation at all yet; the pipeline reports
retrieval proxies only.

## Targets

**CG-Bench**, on the official mini-set (1,118 videos / 3,000 QA), reporting the
official metrics at a budget of at most five intervals:

- `mIoU >= 10` — 2.8x the best published number, and well inside the 31.56 that
  oracle selection already reaches over the current clip granularity.
- `rec.@IoU >= 15`
- `rec@0.5 > 0` — currently zero purely because ranking is bad, not because 30s
  clips are too coarse; oracle reaches 25.37.
- Minimum publishable: `mIoU > 3.58`.

**Video-Holmes**, on all 1,837 questions:

- `accuracy > 27.8` — beat the best open ~7B model.
- Requires an end-to-end answer path that does not exist yet; this is the
  highest-risk target.

## What the gap is made of

The dominant CG-Bench gap is **ranking, not clip granularity**: oracle selection
over the existing 30s clips reaches mIoU 31.56 against our 3.64. Closing a third
of that gap already clears every published number. Finer segmentation raises the
ceiling further (82.80 with 4s clips) but is not on the critical path.

Two things bound ranking today:

1. 43% of CG candidate rows are placeholder text from a failed L1 lane, and 46.6%
   of gold clue spans land on one, so nearly half the golds are invisible to any
   reranker. Tracked by `--max-clip-schema-failure-rate` (see
   `run_staged_llm_pipeline.py`).
2. OPD trains on one positive per example out of ~36 available, with a pointwise
   two-way objective against a top-k ranking metric.

## What the learned reranker has to beat

BM25 between the question and the clip text, with no training at all, reaches
segment_recall 58.12 and inference_shot_recall 10.77 on the Video-Holmes heldout,
against SFT 56.01/5.61 and OPD 59.93/7.92. It beats SFT on both and beats OPD on
inference_shot_recall. On CG-Bench it reaches mIoU 2.97 against the reranker's
3.64, already close to the 3.58 published for Qwen2-VL-72B. Rank-fusing the two
gains nothing over the better of them, so the signals are redundant.

This does not retire the learned controller: BM25 is a scoring function, not a
policy, and the L2 controller has to emit structured tool actions. What it does
is set the bar. Two methods as different as lexical matching and a trained
reranker plateauing together, far below an oracle of 99.75/95.57, points at the
input they share rather than at either model -- the first-pass captions are
written without the question in context, and gold-overlapping clips share only
11.1% of the gold wording against 8.9% for the rest.

So the comparison to make is BM25 against OPD **on the improved input**, once
`--anchor-repass-top-n` re-captions a shortlist with the question in context. A
learned reranker should separate there, because it can use semantics, negation,
entity binding and the option text that lexical overlap cannot. If it does not
separate even then, the contribution is the perception change, not the policy.

**OPD success criterion: beat BM25 by a clear margin on the repassed catalog**,
not merely beat SFT.

## Order of work

1. Rebuild the CG L1 catalog (`--retry-failed-clip-schemas`).
2. Re-caption a per-question shortlist with the question in context
   (`--anchor-repass-top-n`), which is where the shared ceiling sits.
3. **Retrain OPD on the repassed catalog** and compare it against BM25 on that
   same input. Changing the catalog without retraining reads new text with a
   model fitted to the old text and understates the policy.
4. Scale evaluation to the official mini-set.
5. Coarse-to-fine retrieval, only if needed after (3).
6. Video-Holmes answer path.

Measured negatives, kept so they are not re-attempted blind: emitting the
discarded middle band under the pointwise objective changed nothing
(-0.16 dev segment_recall) and hurt when combined with decision-logit training
(-3.33); rank-fusing BM25 with the reranker gains nothing.
