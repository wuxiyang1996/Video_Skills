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

## What has moved the reranker so far (dev, 21 Video-Holmes examples)

| Variant | seg_recall | infer_shot | vs BM25 on the same input |
| --- | ---: | ---: | --- |
| BM25, no learning | 54.60 | 0.00 | — |
| OPD baseline (408 rows, pointwise) | 59.76 | 3.97 | +5.2 / +4.0 |
| + middle band (696 rows) | 59.60 | 3.97 | flat |
| + middle band + decision-logit training | 56.43 | 3.97 | worse |
| **+ middle band + decision-logit + pairwise margin** | **63.73** | **10.32** | **+9.1 / +10.3** |

**The heldout reversed this.** On all 263 Video-Holmes heldout videos, D scores
segment_recall 53.21 against OPD 59.93 (paired bootstrap −6.72 [−9.24, −4.25])
and below SFT (−2.79, CI excludes 0); inference_shot_recall 8.71 vs OPD 7.92
(+0.79 [−1.33, +2.98], not significant) and still under BM25's 10.77.
segment_precision rose to 94.58 (+2.76 vs OPD, significant): the margin objective
learned to pick fewer, surer clips at the cost of recall. The dev gain was a
21-example artifact. **Do not ship D.** This is the third dev→heldout reversal in
this work; 21 dev examples cannot select a checkpoint, and no further training
variant should be judged on them.

Why D lost recall: its top-4 picks are positives as often as OPD's (68.4% vs
68.7%) but their temporal spread halves (median 37s vs 67s). The margin pairs
are ordered by a teacher score dominated by `0.60 × inference_hit`, so every
pair pulls toward the one inference region and nothing in the loss rewards
covering the ~3.4 gold segments per question.

**Diagnostic, not a result — untuned temporal NMS at selection time.** Skipping
any candidate that overlaps an already-chosen pick (no free parameter) on the
same heldout rankings gives segment_recall SFT 56.01→62.33, OPD 59.93→**64.89**,
D 53.21→61.69, with precision unchanged; D's inference_shot 8.71→10.30. So the
top-k selection step, not the scorer, was leaving ~5 points on the table for
every model. Observed first on the inspected heldout, then **confirmed on Video-Holmes dev**
(an independent split for this rule, 21 examples), where the same untuned rule
lifts every model: SFT 47.70→63.33, OPD 59.76→65.32, midband 59.60→67.70,
midband+dl 56.43→69.29, D 63.73→66.11; inference_shot rises too (OPD
3.97→7.94) and precision is flat. The gain is model-agnostic — with NMS on, D is
+0.8 over OPD on dev and −3.2 on heldout, so it does not rescue the pairwise
adapter. The retrieval heldout has no unread portion (all 263 were scored), so
dev is the confirmation available. `--temporal-nms` in the evaluator, default
off, recorded in report provenance.
Measured negatives kept so they are not re-attempted blind: middle band alone,
middle band with decision-logit training, and rank-fusing BM25 with the reranker.

## Where Video-Holmes answers are lost

Both benchmarks score QA accuracy and an abstention counts as wrong. The answer
chain abstains on the majority of questions, and 95.7% of those abstentions
already carry an option_label at the commit step: `verify_claim_support` scores
support by lexical overlap (threshold 0.05), fails `insufficient_evidence`, and
`commit_answer` refuses on `claim_not_verified`. On multiple choice that policy
is strictly dominated by committing the best hypothesis, since a guess scores
~20% and an abstention scores 0.

`always_commit_mcq` (opt-in, default off so frozen GRPO rewards reproduce) emits
the candidate at confidence 0.2 and flags it `committed_unverified`; the
acceptance gate is unchanged, so "has an answer" and "answer is verified" are
now separate facts.

**Measured (2026-09-04, 38 reserved heldout examples, gpt-oss-120b as planner
and skill executor, all rows verified LLM-executed, seed 20260904):** completion
100%, end-to-end accuracy **36.8%**, against published 27.8 (Qwen2.5-VL-7B)
and 45.0 (Gemini-2.5-Pro). With skills run as lexical rules the same setup
scored 15.8%. **Retracted:** an earlier note here said only 3 of 38 commits used the flag.
The flag-off run on the same 38 examples commits 34.2%, and 22 examples that
abstained there are `accepted_strong` with the flag on, with identical
correctness on every example both runs committed — so the flag, not the chain,
produced most commits. The `accepted_strong` label was a bug: the acceptance
gate checked only that refs were present, and an always_commit commit carries
the refs the verifier looked at, so unverified commits were reported verified.
Fixed: `committed_unverified` now forces `commit_ok=False`; the label survives,
the status is `rejected`. The B-run labels on disk predate the fix.

**Verdict on the skill graph (same 38, one model for planner, skills and the
direct control):** direct 44.7% / graph flag-off 13.2% / graph + always_commit
36.8%. Paired bootstrap: graph − direct **−31.6 [−47.4, −15.8]**; always_commit −
graph +23.7 [+10.5, +36.8]; always_commit − direct −7.9 [−23.7, +7.9], not
significant. With model, evidence and budget fixed, removing the decomposition
raises accuracy; the graph's best configuration reaches parity at best.
MHR/IMC are 1–2 examples each. A second seed is in flight; the 190 unread
examples must supply any final number. Still missing before this is a claim: the same-model `direct`
control (no skill graph) and the flag-off `model` run, both in flight; and the
190 unread heldout examples for the final number.

**The skill executor's default model is unusable on OpenRouter.** `qwen/qwen3.5-9b`
(the trainer's `--skill-model` default) now spends its whole completion budget
on hidden reasoning and returns empty content with `finish_reason: length` on
every provider tried (SiliconFlow, DeepInfra), and `reasoning.exclude` /
`chat_template_kwargs.enable_thinking=false` do not stop it. Every LLM-backed
skill then parse-fails and falls back to its lexical rule with `ok=True`. The
cached trainer rollouts predate this and show `backend: llm`; any new GRPO run
with the default would silently train on rule-executed skills. `SkillModelClient`
now records `reasoning_tokens` and `thinking_exhausted` from the usage counter,
accepts a `provider` preference, and `openai/gpt-oss-120b` (75 reasoning tokens
under `effort: minimal`, content returned) or `qwen/qwen3-30b-a3b-instruct-2507`
(0 reasoning tokens) both work. Answer-chain measurements use gpt-oss-120b for
both planner and skills so the direct-vs-skill-graph comparison holds the model
fixed. The frozen GRPO executor caches (last writes 2026-09-02 23:22) show every
answer-critical skill on `backend: llm`, so the paper's GRPO artifacts predate
the breakage and were trained with LLM-executed skills.

Two earlier numbers are retracted: "76.4% when answered" came from 25 examples
selected for reward variance, and a "13.2% clean heldout" figure was measured on
the deterministic fallback planner after the `:free` model was withdrawn — the
fallback is silent, `errors: 0`, and only `llm_plan.fallback_reason` reveals it.
