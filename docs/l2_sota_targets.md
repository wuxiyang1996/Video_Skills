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

- **Stated target (2026-09-04): atomic skills + OPD above Gemini-2.5-Pro, i.e.
  `accuracy > 45.0`**, seven-type average. Direct prompting with BM25 retrieval
  measures 33.5; the skill graph is currently below direct on the SR subset. A
  retrieval-ceiling run (direct with gold-span clips) is in flight to decide
  whether retrieval or the answer model is the binding lever.
- `accuracy > 27.8` — beat the best open ~7B model (already exceeded by direct).
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

## Video-Holmes: is retrieval the lever? (interim, 2026-09-04)

Direct answering (gpt-oss-120b over clip descriptions, no skill graph) on the
full 1,837-question test, paired by question:

| evidence given to the answer model | n paired | acc | vs BM25 top-4 |
|---|---|---|---|
| BM25 top-4 (system, no learning) | 1,827 | 33.6 | — |
| ~~"oracle" top-4~~ **RETRACTED** — was the video's first 4 clips (see below) | 1,827 | 30.6 | −3.0 [−5.3, −0.6] (measures "opening vs BM25", not a ceiling) |
| BM25 top-8 | 1,823 | 34.3 | +0.7 [−1.2, +2.6] |
| no clips at all (`--indices-from none`, 300 ids) | 281 so far | 15.3 (54% abstain; ≈chance when it commits) | |
| **whole catalog** (`--indices-from all`), full test | 1,827 | **35.7** (7-type avg 35.4; SR 50.3, IMC 37.0, MHR 34.9, PAR 34.5, TA 32.5, CTI 30.4, TCI 28.2) | **+2.3 [+0.2, +4.5]** (SR +6.6 is the only per-type CI excluding 0) |

**Retraction (same day).** The "oracle" used Video-Holmes `segment_spans`,
which come from the per-video *Segment Description* rows and cover a median
95% of the video. A median 52 of ~61 clips therefore "overlapped gold", and
`hits[:4]` returned the video's opening four clips (ending at 9% of the video)
— identical for every question of a video. The row above measures "video
opening vs BM25 top-4", not a retrieval ceiling. Fixed in commit after
888c822: the oracle now uses the per-video *Inference Shots* timestamps
(`inference_spans`: 1-s clue moments, median 2 clips hit, 12% overlap with
BM25 top-4; 14% of questions have none and are counted as errors) and ranks
clips by overlap; CG-Bench uses its per-question `clue_intervals`. The
corrected oracle runs on the 300 control ids (`direct_oracle_fixed_300`).

The graph+always_commit run on the same retracted oracle (`full_vh_oracle/graph_commit_oracle_300`, 23.3%) is likewise not a ceiling; the graph-over-whole-catalog run (`full_vh_controls/graph_commit_all_300`) replaces it.

**The trained controller does beat the no-learning baseline on the meaningful
metric.** OPD reranker vs BM25 on the same 1,561 full-test questions that have
Inference Shots, top-4:

| model | segment recall | segment precision | **inference-shot recall** |
|---|---|---|---|
| BM25 (no learning) | 60.49 | 88.02 | 11.00 |
| **OPD reranker** | 58.68 | 90.05 | **14.57** |
| BM25 + temporal NMS | 67.36 | 88.32 | 10.93 |
| OPD + temporal NMS | 58.07 | 89.94 | 12.19 |

OPD − BM25 inference-shot recall = **+3.57 [+1.85, +5.34]**; OPD finds a clue
clip BM25 misses on 242 questions, the reverse on 163. Temporal NMS raises
segment recall (which is near-vacuous) and *lowers* inference recall, so it is
not part of the reported system. This is the honest retrieval claim for
Video-Holmes.

Better grounding does **not** move the answer, though. On the **full test**
with gpt-oss-120b the OPD pointer scores 36.5 against the BM25 pointer's 36.8
(1,835 paired, −0.33 [−2.23, +1.58]; no per-type CI excludes 0); with the
235B reader on 300 ids it is 44.0 against 45.0 (−1.0 [−4.7, +2.7]), and using the reranker to *cut*
to top-4 scores 33.6 against 35.3 for whole catalog + BM25 pointer (1,186
paired, −1.8 [−4.4, +0.8]). So on Video-Holmes retrieval is a grounding
contribution, not an accuracy one — say that plainly rather than implying the
controller drives the QA number.

**Same caveat applies to the paper's VH "segment recall".** `temporal_retrieval_metrics(selected, segment_spans)`
scores recall over those 95%-coverage segments, so it mostly measures how
spread out the top-k picks are, not whether they land on the clue. The
inference-span recall is the meaningful VH retrieval number, and there the
heldout order was BM25 10.77 > OPD 7.92 > SFT 5.61. Report inference recall as
the primary VH retrieval metric.

The `none`/`all` controls remain valid; results land in `full_vh_controls/`.

**Corrected oracle (300 control ids, 257 with Inference Shots):** the clue
clips alone (median 2 clips) score 30.2 vs BM25 top-4 35.3 (−5.1 [−10.2, 0.0])
and vs whole catalog 36.6 (−6.2 [−12.1, −0.4]). Two readings are possible and
the run cannot separate them: the clue moment's *description* does not carry
the answer, or two clips are too little context. The clean test keeps the
evidence fixed and changes only the pointer: `--indices-from all
--highlight-from oracle` vs `all` (running). Whole catalog vs BM25 top-4 on
the full 300: 35.9 vs 35.9, +0.0 [−5.0, +5.4].

**Retrieval as a pointer, not a cut (300 ids):** whole catalog + BM25 top-4
flagged as `likely_key_clips` scores 39.0 — +3.0 [−2.0, +8.0] over the whole
catalog and +3.4 [−1.7, +8.4] over BM25 top-4 (IMC +21 is the only per-type
CI excluding 0). **Full test (1,826 paired): 36.8** (7-type avg 36.6) — +3.4 [+1.2, +5.5]
over BM25 top-4 and +1.1 [−0.7, +2.9] over the whole catalog. So the pointer
itself is worth about one point on top of full context; the whole-catalog
step carries the other two. Current best honest VH number without learning:
**36.8**. The same prompt with the **corrected oracle**
(Inference-Shot clips) as the pointer scores 38.3, +2.3 [−2.0, +6.7] over the
whole catalog — no better than the BM25 pointer. So even a perfect pointer to
the clue moment is worth ≈2–3 points to this answer model over these
descriptions; whatever the reranker learns cannot exceed that on Video-Holmes. If it holds, this is the slot where the OPD
reranker can add value honestly: `--highlight-from report` with the OPD
ranking, same evidence, only the pointer trained.

Caution on interim readouts: rows land in completion order, and whole-catalog
prompts finish first on short videos, so the first 104 `all` rows read 41.3%
before settling to 33.2% on 199 paired ids. No accuracy pattern by catalog size
(≤40 / 41–60 / 61–80 / >80 clips) separates `all` from BM25 top-4. Everything
that keeps the same descriptions and the same answer model converges on ≈33%:
none 15 < oracle 30.6 < BM25 top-4 33.6 ≈ top-8 ≈ whole catalog. The lever has
to be the descriptions or the answer model. Hidden reasoning is not it either
(`--reasoning-effort high`, 8k budget, 300 ids, paired): BM25 top-4 −1.7
[−6.1, +2.4]; whole catalog +1.7 [−3.7, +7.1]. The answer model's own
chain-of-thought extracts nothing more from these descriptions, so every
number stays at effort=minimal.

## Video-Holmes: opening the evidence ceiling (pilots launched 2026-09-04)

Every text-side lever converges at 33–37, so the next step changes the
evidence modality: the answer call now gets **frames of the pointed clips**
(`--frames-per-clip 4 --frame-max-clips 4`, 448 px) next to the whole
catalog, and the answer model can differ from the planner (`--answer-model`).
Five 300-id pilots in `full_vh_mm/`, all over whole catalog + BM25 pointer:

| run | answer model | frames | separates |
|---|---|---|---|
| `vl9b_text_300` | qwen/qwen3.5-9b (the L1 describer) | no | small VLM as reader vs gpt-oss-120b |
| `vl9b_frames_300` | qwen/qwen3.5-9b | 16 | seeing the pointed clips, same small model |
| `vl235b_text_300` | qwen/qwen3-vl-235b-a22b-instruct | no | bigger model, text only |
| `vl235b_frames_300` | qwen/qwen3-vl-235b-a22b-instruct | 16 | bigger model + frames |
| `gptoss_votes5_300` | gpt-oss-120b, 5 samples at T=0.7 | no | self-consistency |

Reference on the same 300 ids: gpt-oss-120b text, whole catalog + BM25 pointer
= 39.0.

**Result (300 ids, paired).** The answer model is the lever, not the modality:

| answer model | frames | acc | vs gpt-oss-120b |
|---|---|---|---|
| gpt-oss-120b | no | 39.0 | — |
| **qwen/qwen3-vl-235b-a22b-instruct** | no | **45.0** | **+6.0 [−0.7, +12.7]** (CTI +25.7 sig, TCI +11.7) |
| qwen/qwen3-vl-235b-a22b-instruct | 16 | 43.6 | +0.0 vs its own text run (236 paired) |

Frames of the pointed clips add nothing over the L1 descriptions, so the clip
schemas already carry what the answer needs; the 120b reader was the
bottleneck. **Full test (1,836 of 1,837, completion 100%): 42.5** (7-type avg 42.1;
SR 53.8, IMC 49.1, PAR 40.2, TCI 39.9, MHR 39.8, CTI 37.8, TA 34.0), +5.8
[+3.4, +8.2] over gpt-oss-120b through the identical pipeline. Against the
published numbers: 27.8 (Qwen2.5-VL-7B) — beaten by 14.7; 45.0
(Gemini-2.5-Pro) — 2.5 short. The 300-id subset read 45.0, so the subset was
optimistic. A size-matched reader (`qwen/qwen3-vl-8b-instruct`) scores **39.1 on the full
test** (SR 51.7, IMC 43.5, CTI 39.3, MHR 36.7, PAR 36.1, TA 34.0, TCI 29.7):
+2.3 [−0.3, +4.9] over gpt-oss-120b, −3.5 [−5.8, −1.3] under the 235B, and
**+11.3 over the 27.8 published for Qwen2.5-VL-7B**. So an 8B-class system
built on this pipeline beats the size-matched baseline by eleven points, and
the pipeline rather than the reader's size carries most of the gain.
Frames still add nothing: 235B 44.7 with frames vs 45.0 without (300 paired,
−0.3 [−4.3, +3.7]); 8B 36.3 vs 39.0. Self-consistency does not help either —
gpt-oss-120b with 5 samples at T=0.7 and a majority vote scores 38.0 against
39.0 for the single greedy call (295 paired, −1.0 [−5.4, +3.7]). The reader is
the only lever that moved. qwen3.5-9b
as answer model is unusable on OpenRouter (hidden-reasoning exhaustion,
ValueError on every call); qwen3-vl-8b-instruct is the size-matched
substitute (`vl8b_*`). If frames help, retrieval matters again (the pointer decides what is
watched) and the atomic skills get a real job: perception skills that turn
frames into verifiable facts, feeding the `hybrid` answer call.

## Why the atomic-skill graph loses (24-question rollout dump, 2026-09-04)

`graph_diag_24` (whole catalog, always_commit, gpt-oss-120b for planner and
skills, every step on the LLM backend): 3/24 correct where direct over the
same evidence gets ≈10/24. Every rollout runs the same chain
(parse_question_target → propose_evidence_roles → generate_answer_hypotheses →
retrieve_evidence_for_hypothesis → score_hypothesis_support ×k →
compare_hypotheses → verify_claim_support → commit_answer). Where the points go:

- `generate_answer_hypotheses` assigns a prior per option: top prior = gold
  5/24. It sees the question, not the evidence.
- `score_hypothesis_support` scores each option **independently** against ~5
  retrieved observations (single-clip facts, not the narrative); plausible
  wrong options get "moderate support" (0.6–0.65) and `compare_hypotheses`
  picks the max: best = gold **3/24**, worse than the prior.
- `verify_claim_support` fails on 21/24; under always_commit the committed
  label is then whatever `commit_answer` falls back to (7× "A"), i.e. arbitrary.

So the graph never lets one model read all the evidence and weigh the options
against each other — which is exactly what direct does.

**The `hybrid` fix does not rescue it (150 ids, paired).** Keeping the skills
as an analysis pass and answering once over the whole catalog *plus* their
notes scores 28.0, against 36.7 for plain direct answering over the same clips
(−8.7 [−16.0, −1.3]) and 40.0 for direct with the BM25 pointer (−12.0
[−19.3, −4.7]). The graph's own answer inside those runs was 22.0. So the
notes are not neutral filler: they drag a capable reader toward the graph's
wrong pick. **It is not the scoring step.** `--findings-mode observations_only` — drop the
per-option scores and the vote, keep only the skills' observations — scores
26.7 on the same 150 ids: no better than the full notes (28.0, −1.3 [−8.0,
+5.3]) and still −13.3 [−20.7, −6.0] against direct answering with the
pointer. The skills' observations are themselves interpretive claims drawn
from the ~12 clips (of ~61) their per-hypothesis retrieval reached, and adding
them to a capable reader hurts regardless of whether verdicts come attached.
The decomposition has no accuracy contribution to make on Video-Holmes in any
form that *re-reads the L1 text*.

**Why that does not contradict the benchmark.** Video-Holmes is built to test
seeing a subtle visual clue and chaining it. Our skills operate on clip
descriptions written at L1 *without the question in view*, so a clue the
describer did not think to mention is already unreachable; re-arranging that
text cannot recover it, and the graph works from a strict subset (~12 of ~61
clips) of what the reader already has. Three measurements agree: the corrected
oracle (clue clips only) scores *below* BM25 top-4, passively attached frames
add nothing, and the reader is the only lever that moved.

The one decomposition that can add information is therefore to go back to the
pixels **with the question**: `--conditions probe` asks a question-conditioned
visual probe of each pointed clip (4 frames, qwen3-vl-235b) and answers over
the whole catalog plus those observations. Running on the 300 control ids
against the same reader's 45.0.

**Summary of every form the decomposition was tested in (Video-Holmes):**

| condition | evidence | acc | vs direct on the same clips |
|---|---|---|---|
| direct | whole catalog | 36.0 (300) | — |
| graph + always_commit | whole catalog | 23.3 (300) | **−12.7 [−19.0, −6.3]** |
| hybrid (graph notes + vote, one answer call) | whole catalog | 28.0 (150) | **−8.7 [−16.0, −1.3]** |
| graph + always_commit | BM25 top-4 | 23.6 (127) | −7.9 [−18.1, +3.2] |
| hybrid, observations only (no scores, no vote) | whole catalog | 26.7 (150) | **−13.3 [−20.7, −6.0]** vs direct+pointer |
| graph (SR subset, earlier) | reranker top-4 | 19.4 (72) | −26.4 [−38.9, −13.9] |

Every form loses, including the two where the graph cannot remove evidence.
Report this as a negative result and place the decomposition's contribution in
verifiable evidence chains, not accuracy.

## CG catalog repair, targeted

Of the 67 heldout CG videos, 28 have placeholder catalogs and they are spread
over all eight shards of the broken lane, so the paper's CG number was gated on
the whole lane (~35 GPU-h, repeatedly preempted on scavenger). The runner now
takes `--example-id-allowlist`; eight per-shard jobs (`cg-held-<start>`) repair
only those 28 examples' stage directories — 1,728 placeholder clips, ~6 GPU-h in
jobs short enough to usually survive preemption. The full-lane jobs were
cancelled so they cannot write the same directories concurrently; the other 172
lane examples (training pool, not on the critical path) can be repaired later.
Do not run `resubmit_preempted_repair.sh` while `cg-held-*` jobs are active.
Targets: `l2_expansion_20260831/heldout_repair_targets.json`.

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

**Verdict on the skill graph (two seeds, n=72 paired, one model for planner,
skills and the direct control):** direct **45.8%** / graph flag-off 19.4% /
graph + always_commit 36.8% (n=38). Paired bootstrap: graph − direct
**−26.4 [−38.9, −13.9]**; always_commit − graph +23.7 [+10.5, +36.8];
always_commit − direct −7.9 [−23.7, +7.9], not significant. The second seed
replicates the direction (direct 48.7, graph 30.8). With model, evidence and
budget fixed, removing the decomposition raises accuracy; the graph's best
configuration reaches parity at best. One qualification: on the second seed the
graph is right 75% of the time *when it commits* (vs direct 48.7%), so its
failure is coverage, not precision — Tested offline: a hybrid that takes the graph's
answer when it commits and direct's otherwise scores 48.6% vs direct 45.8%
(+2.8 [−2.8, +8.3], not significant), and best-of-both caps at 50.0% — the
decomposition's complementary value is at most ~4 points. **Correction:** multi-hop is absent from *our example set*, not from the
benchmark. Video-Holmes test has 1,837 questions across seven types — MHR 332,
IMC 276, TCI 273, CTI 270, SR 292, TA 200, PAR 194 — so MHR+IMC is 33%. Our
heldout L1 was built with one question per video (270 videos → 270 questions)
and that selection is 260/270 SR. Two consequences: every Video-Holmes accuracy
above is on an SR-dominated subset and **is not comparable to the leaderboard's
27.8, which averages all seven types**; and because the L1 catalog is built per
video, the existing 270 catalogs should already cover all 1,837 questions
without new captioning — a full-benchmark evaluation is a matter of
re-deriving examples for every question over the cached stages. Confirmed:
`extract_clue_memory_graph` takes no question ("question-agnostic layer-1"),
and in `video_only` mode clip schemas carry no `question_context`; the skew came
from `--unique-videos` in the L1 worker, which keeps the first question per
video (SR for 260 of 270). Stage directories are keyed by example_id, so the
runner cannot reuse a video's cache for a sibling question on its own;
`scripts/eval/derive_full_question_examples.py` copies each video's frozen L1
into one example per question with no captioning. The 270 videos carry all
1,837 test questions.

**A first "full-benchmark" run measured only 263 of them.** The measurement
script took retrieval indices from the reranker's eval report, which covers only
the 263 original questions; the 1,574 derived siblings got no indices and were
silently skipped, so the run reproduced the SR-subset number (direct 43.0%,
253/5/5 SR/IMC/MHR). Skipped examples are now counted as errors, and
`--indices-from bm25` ranks each example's own captions against its question so
every question is covered; the reranker's own ranking over all 1,837 requires a
pointwise build plus a GPU scoring pass. Built: 1,781 of 1,837 questions
(109,978 candidate rows); 56 excluded by the pre-registered rule that a
question's full catalog must contain both a positive and a negative candidate
(43 no-negative, 13 no-positive). OPD-adapter scoring over it is queued on GPU;
on completion both answer-chain conditions relaunch with those indices. Two
BM25-indexed runs over all 1,837 are in flight now.

**Full Video-Holmes test, direct prompting, gpt-oss-120b, BM25-retrieved top-4 clips (n=1837 of 1,837, all measured; completion 99.5%): accuracy 33.5%.** By type: MHR 32.2 (n=332), SR 43.8 (n=292), IMC 34.1 (n=276), TCI 27.5 (n=273), CTI 30.0 (n=270), TA 36.0 (n=200), PAR 29.9 (n=194). Published: 27.8 (Qwen2.5-VL-7B), 45.0 (Gemini-2.5-Pro). First number here comparable to the leaderboard; the skill-graph condition and the reranker-retrieved variants are in flight. 80 of 270
one-question examples are consumed; the full-benchmark run must be the source
of any final number. Still missing before this is a claim: the same-model `direct`
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
