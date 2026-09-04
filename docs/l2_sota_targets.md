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

## Order of work

1. Rebuild the CG L1 catalog (`--retry-failed-clip-schemas`).
2. OPD: use all available positives; replace the pointwise objective with a
   listwise one over each example's candidate set.
3. Scale evaluation to the official mini-set.
4. Coarse-to-fine retrieval, only if needed after (2).
5. Video-Holmes answer path.
