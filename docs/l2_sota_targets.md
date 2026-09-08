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

## What Video-Holmes actually scores (audit of GT + official eval, 2026-09-04)

- **Construction.** 270 suspense short films, human-annotated as *Segment
  Description* (timed narrative), *Key Relationships*, *Inference Shots*
  (time + clue + conclusion), *Core Theme*. The 1,837 questions and their
  `Explanation` were then **generated by DeepSeek from those text annotations**
  (README: "employ DeepSeek to generate questions"). Six options each, answer
  letters balanced (284–323 per letter). The GT is a narrative synthesis, not a
  frame-level fact: explanation terms overlap SegmentDescription only 18%,
  InferenceShots 11%.
- **Official accuracy eval** (`evaluate.py`): the model sees the video and the
  prompt "reason between <think></think>, answer between <answer></answer>";
  the letter is parsed from the answer tag; score = exact match, reported per
  type and averaged. Nothing else is scored.
- **Official "reasoning process analysis"** (`evaluate_reasoning.py`): DeepSeek
  classifies the model's <think> text against SegmentDescription + Explanation
  into VPE / VOE / RE / TRAW (wrong) or TWAR / TRAR (right). It is a
  *diagnostic taxonomy*, not a metric — a correct evidence chain earns no
  credit, and a wrong answer with a perfect chain is still wrong.

Consequences for the decomposition: (1) the only scored quantity is the
letter, so anything the graph produces beyond the letter (citations, chains,
uncertainty) is invisible to the benchmark; (2) the GT rewards whole-story
synthesis in the annotators' narrative register, which one full-context read
does best; (3) the paper can still *use* the official taxonomy to say *where*
each system fails — that analysis is running on the fresh 300
(`vh_reasoning_analysis.py`, judge `deepseek/deepseek-chat`, direct with the
official <think> protocol vs graph2 chains). The lexical-coverage probe
(explanation terms found in our catalog: correct 27.7% vs wrong 26.6%) cannot
separate omission from reasoning errors, hence the judge.

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
the whole catalog plus those observations. **Unconditional looking is not the answer either (300 ids, paired).** Probing
every pointed clip scores 43.3 against direct's 45.0 (−1.7 [−5.3, +2.0]).
It does beat the text-only skill notes by a wide margin (37.3 vs 28.0 on the
150 hybrid ids, +9.3 [0.0, +18.7]), so visual observations are far better
skill output than re-read text — but spending four visual calls on every
question buys nothing on average, because most questions are already decided
by the descriptions and many Video-Holmes questions are interpretive ("what is
the deep implication of..."), which a single clip's frames cannot answer.

**Which is what makes the repaired graph the real test.** `--conditions graph2`
(commit c6f9e7b) ranks every option against the same whole-catalog evidence in
one call and cites the deciding clips, then looks at the frames of the clips
the top two disagree over *only when its own top-two margin is small*. Two
variants run on the 300 control ids: `--rank-margin 0` (repair only, no
looking) isolates the structural fix, `--rank-margin 0.15` adds the selective
look. **graph2 v1 on the full control 300 (2026-09-04):** 39.7 vs direct 45.0,
**−5.3 [−9.0, −2.0]**; with the verbatim-question look-again 38.7 (−6.3
[−10.3, −2.3]; looking itself −1.0 [−4.0, +2.0]). At 93 paired ids this had
read as parity (−2.2 [−7.5, +3.2]) — the CI was simply too wide; the
structural repair cut the old 9–13 point loss roughly in half, not to zero.
Log inspection of those 93 (direct-right/graph-wrong 4, graph-right/direct-wrong
2, both wrong 55) shows why it still trails:

- 59% of questions fail under both. In those the gold option sits at rank 2
  (20) or 3 (16) of the graph's chain: seen, not decided. They concentrate in
  TCI (15/25) and TA (8/11) — causal chains and event order, which
  question-blind clip descriptions do not record. No re-arrangement of the same
  evidence fixes that.
- The scores are saturated: top ≥0.9 on 92/95 (mean 0.93), runner-up mean
  0.75. The margin is a constant, not a confidence; its apparent
  informativeness tracked question difficulty (direct's accuracy in the same
  buckets was identical), so it cannot decide where to look.
- The look-again probes were handed the abstract question verbatim and 91%
  (107/118) came back "not visible / no evidence"; they changed 7/30 answers,
  1 wrong→right and 4 right→wrong. "Nothing visible" statements push the
  ranker off correct inferential options.
- The 4 losses share one cause: "cite the deciding clips" biases the ranking
  toward literally citable options over inferential ones (e.g. "the
  yellow-clothed woman's physical attack" over "asthma attack cannot be
  treated"). The 2 wins are behaviour-visible IMC/CTI questions.

Two log-justified changes remain untested: probe with *factual sub-questions*
derived from the top-two dispute instead of the abstract question, and score
as a probability distribution over options so the margin means something.
They run on a **fresh** 300 ids (the control 300 have been used for selection
too many times to validate anything). Interim (v2, fresh ids, paired against
the 235B direct run): probability ranking without looking −2.1 [−7.2, +3.4]
(237); with factual sub-question probes −4.9 [−10.7, +0.8] (122), the look
firing on 86% of questions and costing −4.8 on the questions where it fired.
The scores are no longer saturated (top mean 0.36) and the sub-questions are
genuinely factual — only 34% of probe answers are of the "not visible" kind
against 91% in v1 — yet the probes still flip answers the wrong way (on 106
fired questions: 3 wrong→right, 7 right→wrong, 11 wrong→wrong). The
probability margin tracks difficulty for both systems, not the graph's own
reliability (margin <0.10: graph 33.5 vs direct 37.1 on the same 167 ids;
0.10–0.25: 58.3 vs 58.3). Two independent ranking calls on the same question
agree on the label only 12/17 times, so the two-stage rank→look→re-rank
pipeline also compounds sampling noise that a single answer call does not.
The remaining gap is therefore not a prompt defect: the
decomposition does not add accuracy on Video-Holmes even in its best-behaved
form. **Final (fresh 300, paired vs plain direct 235B, 2026-09-05):** probability
ranking without looking 38.7 vs 41.0, −2.3 [−6.7, +2.0]; with factual
sub-question probes 39.7, −1.3 [−5.3, +2.7]; looking vs not +1.0 [−3.0,
+5.0]; on the 250 questions where it looked, +0.8 [−3.2, +4.8] vs direct.
Direct under the official <think> protocol: 41.5 vs 41.1, +0.3 [−3.3, +4.0].
Verdict: the repaired decomposition is at **parity** with a single answer call
on Video-Holmes (every CI includes 0, point estimates 1–2 below) and produces
citations, a probability over options, and — when unsure — targeted visual
observations, none of which the benchmark scores. No further variants.

**In the benchmark's own error taxonomy** (`evaluate_reasoning.py` prompts,
judge `deepseek/deepseek-chat` as the official script uses; the chain's
per-option reasons serve as the "thinking"): graph2's wrong answers are **RE
91%** (reasoning error), VOE 3%, VPE 2%, unclassifiable 4% (n=184); its right
answers are TRAR 91% / TWAR 9%. With the sub-question look: RE 85%, VOE 7%,
VPE 2% (n=181). Paired transitions (300): RE→RIGHT 17, RIGHT→RE 13, RE→VOE 6
— looking fixes and breaks in nearly equal measure. So, as the benchmark sees
it, the decomposition does not fail by *missing* clues (VOE/VPE ≤ 9%); it
fails by drawing the wrong inference from the same descriptions the direct
reader has — exactly the case where re-arranging evidence cannot help. **Side-by-side with direct** (same 300, direct's reasoning elicited as a
JSON field after the 235B reader ignored <think> tags 299/299 times):

| | wrong answers: RE / VOE / VPE | right: TRAR / TWAR |
|---|---|---|
| direct (n wrong 179) | **80% / 17% / 3%** | 84% / 16% |
| graph2 (n wrong 184) | **91% / 3% / 2%** | 91% / 9% |

Paired transitions direct→graph2: RIGHT→RE 30 vs RE→RIGHT 21; **VOE→RE 20 vs
VOE→RIGHT 7**. Read in the benchmark's terms: the decomposition *finds* the
clues (omission errors fall from 17% to 3%, and its right answers are more
often reasoned right, TRAR 91% vs 84%) and then *infers worse over them*
(reasoning errors rise to 91%). The clue-finding gain is exactly the grounding
contribution measured elsewhere (OPD > BM25 on inference-shot recall); the
inference loss is what keeps accuracy at parity. That is the paper's honest
one-line summary of atomic skills on Video-Holmes.

**Why the inference goes wrong — three mechanisms from the rollouts (fresh 300).**

1. *Timeline questions: the ranker ignores the timestamps of its own
   citations.* TA is the **only** type where graph2 is significantly below
   direct (19.4 vs 35.5, −16.1 [−29.0, −3.2]); every other type is within ±3
   (n.s.). In 9 of 23 wrong TA answers the chosen option's cited clips are not
   in time order, while the gold option's are (e.g. q1194: chosen B cites clips
   25, 39, **63, 60**, 79 and narrates "burns the paper, then the candle is
   blown out"; gold F cites 25, 39, 60, 63, 79). Where exactly one option's
   citations are time-ordered, that option is gold 4/7 times. The graph holds
   the information that fixes its worst error and does not use it — a
   deterministic "order events by cited clip time" skill would.
2. *Implication questions (MHR, "what does the shot at 3:09 imply"): the gold
   is film grammar, not content.* Gold explanations cite mirror-image
   reversal, subjective POV, foreshadowing; the descriptions encode none of
   it, and citation pressure pushes the ranker to over-interpret citable
   content (a note → "looped reality"; blur → "memory fragmentation"). MHR
   has the most RE cases (39/61) and the most near-misses (gold at rank 2 with
   p-gap ≤0.2: 15).
3. *Motive questions (IMC): directly-supported generic options beat
   explanatory specific ones.* q104: "visible distress → sudden discomfort"
   (chosen, p=0.35) over "fear of the women in yellow" (gold, rank 2, p=0.25)
   although the chain cites the yellow-dress association. Across all RE
   cases gold sits at rank 2 in 55/167 with mean gap 0.17 — the comparative
   step prefers the option most *directly* supported by visible cues over the
   one that *explains* them.

Judge-reason themes among graph2's RE (multi-label, n=167): causal/temporal
30, missing link 21, literal/surface 15, symbolic over-reading 9.
Mechanism 1 was tested on all 200 TA questions (`--timeline-skill`, commit
7fa2d51; soft assembly commit after). One localisation call per question,
then the order is assembled from clip start times:

| on the 200 TA questions | acc | vs direct 235B (34.0 / 31.8 on the paired ids) |
|---|---|---|
| graph2 base | 25.5 | −8.5 [−14.5, −2.5] |
| + strict timeline (fires 35/151 permutation questions) | 27.0 | −7.0 [−14.5, +0.5]; on fired: base 37.1 → 42.9, direct 40.0 |
| + soft concordance assembly (fires 57/151, offline re-assembly of the same localisations) | 27.2 | −4.6 [−13.9, +4.6]; vs base +4.0 [−3.3, +11.3]; on fired: 35.1 vs base 26.3 vs direct 36.8 |

Why it fires so rarely: of 151 permutation questions, 73 had at least one
event the localiser could not place and 43 produced an order no option
contains (a localisation off by one clip); 49 TA questions are not
permutations at all. So the deterministic assembly removes the ranker's
self-inconsistency and recovers about half of the TA gap, and the limiter
moves to **localisation quality** — the same description ceiling as
everywhere else. It brings the graph to parity with direct on the questions
it decides, not above it.

**Summary of every form the decomposition was tested in (Video-Holmes):**

| condition | evidence | acc | vs direct on the same clips |
|---|---|---|---|
| direct | whole catalog | 36.0 (300) | — |
| graph + always_commit | whole catalog | 23.3 (300) | **−12.7 [−19.0, −6.3]** |
| hybrid (graph notes + vote, one answer call) | whole catalog | 28.0 (150) | **−8.7 [−16.0, −1.3]** |
| graph + always_commit, **full test** | BM25 top-4 | 24.2 (1,823) | **−9.4 [−11.9, −6.9]** (2 rows with no LLM skill, 4 excluded) |
| hybrid, observations only (no scores, no vote) | whole catalog | 26.7 (150) | **−13.3 [−20.7, −6.0]** vs direct+pointer |
| graph (SR subset, earlier) | reranker top-4 | 19.4 (72) | −26.4 [−38.9, −13.9] |

Every form loses, including the two where the graph cannot remove evidence.
Report this as a negative result and place the decomposition's contribution in
verifiable evidence chains, not accuracy.

## VRBench pilot (started 2026-09-04)

Why: VRBench (960 long narrative videos, median 85 min; 8,243 questions; every
question annotated with 2–4 reasoning steps, 18,970 of 25,106 steps with a
time span; seven reasoning types) scores the *reasoning process* as well as
the MCQ letter. A chain that cites clips can therefore earn credit a single
answer call cannot — the accuracy-independent win the decomposition never had
on Video-Holmes. Adapter (`vrbench`) and runner support already existed; the
per-question derivation and the answer chain work unchanged.

Pilot: the 60 shortest readable videos (20–25 min, 1,372 min total, 480
questions; two videos are unreadable and excluded), six scavenger shards, all
outputs on nexus-scratch through the `output/vrbench_pilot_v1` symlink. Smoke
on one 20-min video: L1 in 9 min (44 coarse 30-s clips + 10 fine, 0
placeholders; note the coarse selector puts all 10 fine clips inside one
~1-min window), 8 derived questions, direct 6/8 and graph2 5/8 (n=8, not a
result).

Process scorer (`scripts/eval/vrbench_process_score.py`): cited clip spans of
the top option (+ probed spans) against the annotated timed steps — step
recall, citation precision, mean best IoU (IoU is structurally small for 30-s
clips vs multi-minute steps; recall/precision are the meaningful pair).
Smoke: step recall 37.1, citation precision 22.4 over 7 timed questions.

Pilot L1 finished for 57/60 videos with **0 placeholder clips of 3,375**
(the last shard was preempted at 7/10 and resumes); 459 questions derived
(8.1 per video; Event Attribution 95, Counting 88, Hypothetical 78, Event
Prediction 69, Summarization 58, Implicit Inference 49, Logical Linkage 22).
direct and graph2 (235B, whole catalog + BM25 pointer) are running on all 459.
Interim (completion-order-biased, do not quote): direct 70.7% (140 rows, 11
unparsed labels), graph2 63.2% (38 rows); graph2 process score on 30 timed
questions: step recall 64.4, citation precision 63.1, mean best IoU 18.5 —
far above the smoke video's 37/22. No selection on the pilot ids before a
held-out confirmation.

direct finished on all 459: 30 replies had no parseable label (the 1800-token
plain budget truncates long counting rationales); re-asking those with the
3000-token `--rationale` variant recovered 29. **direct (235B, whole catalog
+ BM25 pointer) on the 459-question pilot: 72.3%** (`direct.merged.rows.jsonl`).
graph2 is at 243/459. Interim paired on 246 (completion-order biased):
direct 72.0 vs graph2 67.1, −4.9 [−8.9, −0.8], driven by Hypothetical
Reasoning (84.4 → 66.7, −17.8 [−28.9, −6.7]) and Logical Linkage (80 → 60,
n=10); Event Attribution, Counting, Summarization and Implicit Inference are
tied. graph2 process score on 201 timed questions: step recall 54.4, citation
precision 60.0, mean best IoU 15.5. Process baselines that need no model, on the 382 timed pilot questions
(what counts as a "citation" → step recall / citation precision): BM25
pointer top-4 **43.3 / 39.4**; BM25 top-2 29.3 / 39.9; first four clips
22.5 / 33.7; all clips 100 / 31.4. graph2's cited clips (interim, 201 timed)
score **54.4 / 60.0** — above the retrieval pointer on both axes, i.e. the
comparative step's citations land on the annotated reasoning steps more often
and more precisely than the retriever that fed it. A single-call direct with
a cited rationale is running as the fair citation-bearing baseline.

**FINAL pilot (480 questions, all 60 videos, paired bootstrap):**

| reasoning type | n | direct | graph2 | graph2 − direct |
|---|---|---|---|---|
| Event Attribution | 99 | 76.8 | 72.7 | −4.0 [−9.1, 0.0] |
| Counting | 94 | 31.9 | 31.9 | 0.0 [−6.4, +6.4] |
| Hypothetical Reasoning | 80 | 82.5 | 66.2 | **−16.3 [−25.0, −7.5]** |
| Event Prediction | 72 | 80.6 | 73.6 | **−6.9 [−13.9, −1.4]** |
| Event Summarization | 61 | 93.4 | 91.8 | −1.6 [−4.9, 0.0] |
| Implicit Inference | 51 | 76.5 | 72.5 | −3.9 [−11.8, +3.9] |
| Logical Linkage | 23 | 87.0 | 78.3 | −8.7 [−26.1, +8.7] |
| **ALL** | 480 | **72.1** | **66.5** | **−5.6 [−8.5, −2.9]** |

Reading: on VRBench the decomposition costs 5.5 accuracy points, concentrated
in counterfactual/predictive questions (the same "infers worse over the same
evidence" pattern as Video-Holmes), and in return produces a reasoning chain
whose citations land on the annotated steps half the time and beat the
retrieval pointer that fed it. **The fair citation-bearing baseline settles it (480 questions, paired).**
A single call asked to reason step by step and cite clip ranks in prose
(`--rationale`) scores **67.3** — the same −4.8 [−8.1, −1.5] against plain
direct as graph2's −5.6, and graph2 vs that baseline is −0.8 [−4.4, +2.9].
So the accuracy cost is not the decomposition's: it is the **cost of making
the reader articulate**; a reader that must justify loses five points
whether it does so in one call or through the graph.

| VRBench pilot, 480 q (process on 399 timed) | accuracy | step recall | citation precision |
|---|---|---|---|
| direct, plain | **72.1** | — | — |
| direct, cited rationale (one call) | 67.3 | **52.7** | 43.8 |
| graph2 (comparative ranking, citations) | 66.5 | 50.3 | **53.5** |
| BM25 pointer top-4 as citations | — | 43.2 | 39.2 |

graph2 − cited rationale: step recall −2.4 [−6.5, +1.6] (tied), citation
precision **+9.7 [+6.0, +13.4]**. graph2 − BM25 pointer: +7.0 [+3.1, +10.9]
and **+14.3 [+10.8, +17.8]**. Both articulated systems recall the annotated
steps about equally; the decomposition's structure is what makes the
citations *precise*.

Paper reading: the decomposition's *structure* buys citation precision
(+9.7 over a single call that also cites, +14.2 over the retriever), at no
accuracy cost relative to any articulated answer and at the same five-point
cost every articulated answer pays relative to a bare letter.

## Where the remaining accuracy is (2026-09-05)

Every answer-step lever on Video-Holmes measured null — frames +0.0, 5-vote
self-consistency −1.0, reasoning effort −1.7, perfect retrieval −3.0, OPD
pointer −0.3, top-8 +0.7 — and only a stronger reader moved the number
(120b → 235b, +5.8). That leaves the L1 descriptions, whose ceiling had never
been measured. `scripts/eval/build_oracle_description_examples.py` swaps our
9B-written catalog for the benchmark's **own human annotations** (timed
Segment Description rows plus Inference Shot clue/conclusion rows, ~5 per
video) and re-answers with the same 235B reader.

| catalog given to the same reader (fresh 300) | accuracy | vs ours |
|---|---|---|
| ours (Qwen3.5-9B, question-agnostic, ~61 clips) | 41.0 | — |
| human narrative rows only (~3.4 rows) | 53.3 | **+12.3 [+7.0, +17.7]** |
| human narrative + clue rows (~5 rows) | **62.0** | **+21.0 [+15.3, +26.7]** |

By type the gap is largest exactly where the graph's reasoning errors
concentrated: MHR +34.4, IMC +29.4, TCI +29.0; TA is the only type that does
not improve (−6.5, timeline questions need the fine time grid our catalog
has and the 5-row annotation lacks). The clue rows are worth +8.7 [+4.7,
+12.7] of the 21 and are the source the questions were generated from, so
**53.3 is the honest ceiling estimate and 62.0 the leakage-inflated one**.

Reading: ~12 points of Video-Holmes accuracy sit in description quality, not
in retrieval, not in the answer step, and not in the decomposition. Three
untested levers act there: a stronger describer than Qwen3.5-9B, more frames
per clip (currently 4), and question-aware re-description
(`--anchor-repass-top-n`, implemented, never evaluated). Note the human rows
are also *fewer and coarser* than ours (3.4 vs ~61), so the gain is quality
and narrative continuity, not coverage.

**Fragmentation or perception? (2026-09-06, no GPU).** Two explanations
for the human rows' +12 with opposite remedies: the information is in our
61 clips but fragmented (no identity or cause carried across 4-s clips), or
it was never perceived by the 9B describer. `scripts/eval/build_narrative_catalog.py`
windows our own clip text (~45 s per window) and has the 235B write one
paragraph per window *sequentially* with a persistent cast — looking at
nothing new — then the same reader answers over the synthesised rows
(narrative only, and narrative + original clips). A gain says fragmentation
and costs no GPU to deploy; no gain says perception and points GPU at the
describer (stronger model, more frames, question-aware repass). Running on
the fresh 300 (191 videos, ~3–7 windows each). The synthesised rows read like
the human ones — e.g. "the man in the green t-shirt ... shares a moment of
laughter before he suddenly appears to choke ... the woman with long dark
hair wearing a dark zip-up hoodie watches from the doorway with a slight
smile. She then walks to the refrigerator" — identities and causes now carry
across what were 17–18 isolated 4-s clips per window.

**Verdict: perception (2026-09-06).** Re-organising our own clip text into
narratives gains nothing with the same 235B reader (fresh 300, paired):

| catalog (fresh 300, 235B reader) | accuracy | vs ours 41.0 |
|---|---|---|
| narrative synthesised from our clips only (191 videos, 3–7 windows) | 38.7 | −2.3 [−6.3, +1.7] |
| synthesised narrative + our clips | 39.3 | −1.7 [−5.3, +2.0] |
| human narrative rows | 53.3 | +12.3 [+7.0, +17.7] |

Synthesised narrative vs human narrative: −14.7 [−20.0, −9.3]. No type
improves (MHR −3.3, SR −6.8, TA −6.5, CTI +2.2). The continuity the human
rows have (who did what to whom, and why) is not recoverable from our clip
text, so the 9B describer never perceived it. The remaining ~12 points are a
**describer** problem. GPU is now pointed at the describer on the same 191
videos (outputs on scratch, `output/vh_l1_levers/`): 8 frames per clip
(`CLIP_FRAMES=8`) and question-aware re-description of the top-16 anchor
clips (`ANCHOR_REPASS_TOP_N=16`), 4 scavenger shards each; both re-measured
on the fresh 300 against 41.0. The narrative-synthesis path is retired as an
accuracy lever (kept as a cheap catalog-format tool).

Three describer runs are in flight on the same 191 videos (2026-09-06):

| lever | what changes | where it runs | measured on |
|---|---|---|---|
| `frames8` | Qwen3.5-9B, 8 frames per clip instead of 4 | scavenger, 8 shards (~35 min/video at 4 frames, resubmitted on preemption) | derived copies, fresh 300 |
| `repass16` | 9B re-describes the 16 question-nearest clips at 6 frames *with the question in context*; base catalog seeded from the frozen v3 stage files (300 per-question stage dirs, `UNIQUE_VIDEOS=0`) | scavenger, 8 shards | the 300 fresh questions directly (repass rows are question-conditioned, so derived copies cannot carry them) |
| `desc235b` | Qwen3-VL-235B on OpenRouter as the describer, 4 frames, same schema prompt | login node, 4 processes × 12 workers (smoke: 26 clips in ~60 s, 0 errors, ~2.7k tokens per clip, ≈$15 for all 191 videos) | derived copies, fresh 300 |

All three are compared paired against ours 41.0 and the human rows 53.3
with the same 235B reader. Note for `repass16`: 10 of the 300 seeded base
catalogs are the post-repair v3 version (more clips than the derived copies
behind 41.0); the other 290 are identical.

**frames8 result (2026-09-07, fresh 300, paired, 235B reader):** 41.3 vs
ours 41.0, +0.3 [−3.3, +4.0]; no type moves (TCI +5.3, MHR −4.9, both
n.s.); −12.0 vs the human rows. Doubling the frames per 4-s clip changes
nothing: the describer's temporal density was not the gap either. With
desc235b (−1.7) this closes the describer-side levers at the clip level —
what helped was the *unit* (30-s windows) and the *modality* (dialogue),
not frames or model size.

**desc235b result (2026-09-06, fresh 300, paired):** 39.3 vs ours 41.0,
−1.7 [−5.3, +2.0]; vs human rows −14.0 [−19.7, −8.3]. No type moves (MHR
−4.9, PAR +3.3, all n.s.). A 25× larger describer, same 4 frames, same
question-agnostic schema prompt, changes nothing. So the perception gap is
not describer *capacity*. What the human rows have and both describers lack
must come from one of: temporal context (a human row covers ~30–60 s and
reads actions as a sequence; a clip is 4 s and 4 frames), the question
(human rows were written knowing the film's point), or the *format* (our
schema asks for observable facts and marks relations uncertain; the human
rows narrate identities, intent and social reading). `frames8` tests the
first; `repass16` the second; the third needs a narrative-from-pixels
describer (long window, many frames, human-annotator prompt) — the
experiment that sits between "narrative from our text" (−2.3) and "human
narrative" (+12.3).

**Audio was never used (found 2026-09-06).** Video-Holmes ships no subtitle
files, the adapter passes `subtitle_paths=[]`, and every clip in the catalog
has `dialogue_spans: []`. The human rows quote what people say ("the young
woman said it was about to start"). `scripts/eval/transcribe_videos.py`
caches whisper-1 segments per video (PyAV audio extraction, region-pinned
OpenAI host); on the fresh 191 videos whisper also emits sound-effect
captions ("**footsteps**", "**creepy music intensifies**"), which one PAR
question literally asks about. `narr_px` = `build_narrative_catalog.py
--frames-per-window 16 --window-s 30 --asr-dir … --no-clip-text`: the 235B
*looks* at 16 frames per 30-s window, reads the dialogue in it, sees the
previous paragraph and the cast, and writes one human-annotator-style
paragraph (smoke output quotes dialogue, names "Sonya" from the audio, and
carries her across windows). Two catalogs on the fresh 300: narrative rows
only, and narrative rows + our clips (time grid for TA).

**narr_px result (2026-09-06, fresh 300, paired, same 235B reader) — the
first CI-clean accuracy gain on Video-Holmes from any lever:**

| catalog | acc | vs ours 41.0 | vs human 53.3 |
|---|---|---|---|
| narrative-from-pixels + dialogue, rows only (~9 per video) | 42.3 | +1.3 [−3.0, +5.7] | −11.0 [−16.3, −6.0] |
| **narrative-from-pixels + dialogue + our clips** | **47.0** | **+6.0 [+1.3, +10.7]** | −6.3 [−11.3, −1.3] |

By type (hybrid): CTI +13.3, TCI +13.2, SR +6.8, PAR +6.7, IMC +2.0, MHR
+1.6, TA 0.0 (each n.s. alone). The narrative alone is not enough — the
fine clip grid still matters (TA −6.5 without it) — and together they close
half of the human gap. What changed relative to desc235b (same model, −1.7):
the describer sees 30 s at once instead of 4 s, hears the dialogue, keeps
the cast across windows, and is asked to narrate like an annotator instead
of listing observable facts. Next: isolate the dialogue's share (our clips +
dialogue rows, no narrative), and run the hybrid on the full test (1,837)
for a number comparable to 42.5 / Gemini 45.0.

**Dialogue's share (fresh 300, paired):** our clips + whisper dialogue rows
(no narrative, no re-perception) = 44.3, +3.3 [−0.7, +7.3] vs ours; the
narrative on top of that adds +2.7 [−1.3, +6.7] (47.0 vs 44.3). Neither half
is significant alone; the sum is. Dialogue helps IMC/CTI/PAR (+6–7 each),
the narrative adds SR/TCI/CTI. So roughly half of the gain is a modality
the pipeline had simply never ingested, and half is describing 30-s
stretches as a story rather than 4-s clips as facts.

**Full Video-Holmes test with the narr_px_plus catalog (2026-09-06; 1,837
questions, 270 videos, BM25 pointer, 235B reader — identical flags to the
42.5 run): 47.4, paired +4.8 [+2.9, +6.6] over 42.6.** Above Gemini-2.5-Pro's
published 45.0 and every open model on the leaderboard. By type: MHR +4.8*,
SR +5.5*, IMC +5.4*, TCI +7.3*, CTI +6.7*, PAR +6.2 (CI touches 0), **TA
−4.5 [−9.5, +0.5]** — timeline questions are the one type the narrative rows
hurt (30-s paragraphs blur event order; the clips' 4-s grid was what TA
lived on). The configuration was fixed on the fresh 300 before this run; TA
is reported as is. What the number is: open components only (Qwen3.5-9B clip
describer, whisper-1, Qwen3-VL-235B as window describer and text reader),
no question-conditioned perception, no learned retrieval, no decomposition.
Pipeline cost per video ≈ 6 windows × 16 frames + ASR, a few cents.

Post-hoc ordering check (full test, `--catalog-order time`, rows interleaved
by start time instead of narrative-first): 45.3, −2.0 [−3.4, −0.7] vs the
narrative-first 47.4; TA unchanged (−0.5). The TA loss is not the ordering;
narrative-first stays the reported configuration.

**The decomposition on the catalog that has the facts (fresh 300, 2026-09-06).**
The design's premise was "reasoning assembly over an evidence graph that
already contains the facts". With the narr_px_plus catalog, graph2
(probability ranking, no looking, BM25 pointer) scores 44.7 against 44.3 for
direct with the same pointer on the same catalog: **+0.3 [−3.3, +3.7]** —
parity again, exactly where it was on the old catalog (−2.3 n.s.). The
graph does inherit the catalog gain (44.7 vs 38.7 on the old catalog, +6.0
[+1.3, +10.3]); it just adds nothing on top of a single reader. By type MHR
+4.9 [0, +11.5] and SR +9.1 n.s. versus TA −9.7 and CTI −6.7 n.s.: the same
pattern as before (finds clues in multi-hop questions, loses on timelines).
Side finding: on this catalog the **BM25 pointer hurts direct** (44.3 vs
47.0 without it, −2.7 [−5.0, −0.3]) — with narrative rows present, pointing
the reader at four clips pulls it away from the paragraphs. The full-test
47.4 was run *with* the pointer for comparability; the no-pointer full test
(the configuration actually selected on the fresh 300) is running.

**No-pointer full test (2026-09-06): 48.3, +5.8 [+3.9, +7.7] over ours 42.6;
vs the pointer run +1.0 [−0.2, +2.2] (n.s.).** This is the configuration
selected on the fresh 300 *before* any full-test run (whole catalog, no
pointer), so **48.3 is the headline Video-Holmes number**; 47.4 is the
pointer variant kept for flag-for-flag comparability with 42.5. By type
(no pointer vs old catalog): MHR 43.4, SR 59.2, IMC 56.2, TCI 49.1, CTI
47.0, TA 30.0, PAR 49.0.

Two companions (2026-09-06): the **8B reader** (`qwen/qwen3-vl-8b-instruct`)
on the same catalog with pointer scores 40.8 on the full test, +1.7 [−0.1,
+3.4] over its old-catalog 39.1 (IMC +4.7*, TA −4.5) — the small reader
extracts far less from the richer catalog than the 235B does (+4.8/+5.8),
so the catalog gain is reader-limited. Without the pointer the 8B scores 40.4 (−0.3 n.s. vs its pointer run; +1.4 [−0.4, +3.2] vs 39.1): the pointer is neutral for the small reader.

**narr_px v2 — machine-written "key moments" hurt (2026-09-06, fresh 300,
no pointer).** Asking the window describer to add 2–3 annotator-style
key-moment rows per window ("Key moment at 41s: she hides the key — implies:
she expects him"; 18.5 rows per video on top of the narrative) scores
**40.0 vs 47.0, −7.0 [−10.7, −3.7]**; SR −18.2*, CTI −11.1*, PAR −10.0,
TCI −7.9. The human clue rows were worth +8.7 because they are *correct*;
model-written implications are the same interpretive claims that sank the
skill graph's notes (hybrid −8.7, observations-only −13.3). Rule confirmed a
third way: give the reader evidence, never conclusions. The stripped
variant (same v2 narratives, key-moment rows removed) is being measured to
separate the two changes.

Stripped variant: the same v2 narratives without the key-moment rows also
score 40.0 (±0 vs v2), so the −7 came from the *narrative text* changing
under the amended prompt, not from the rows. That is a generation-variance
warning for the +6 itself: a replicate of v1 (identical prompt, fresh
generation) is being measured on the same 300.

**Window grounder (2026-09-06, fresh 300, narr_px_plus catalog, pointer):
null.** `--probe-window-s 30 --frames-per-clip 16 --asr-dir`: graph2 looks
at the ±15-s window around each disputed clip (16 frames + dialogue,
annotator-style row; 250/300 fired, 453 windows, 418 with dialogue) and
re-ranks: 43.7 vs 44.7 no-look, −1.0 [−4.3, +2.3]; on fired questions 41.6
→ 41.2, 10 wrong→right vs 11 right→wrong. Direct with the grounder on the
four pointed windows every time: 41.7, −2.7 [−6.3, +1.0] vs direct+pointer
and **−5.3 [−9.0, −1.7]** vs direct without pointer (47.0). The
observations read well (identities, actions, quoted dialogue) — they are
the same perception the catalog already holds: narr_px_plus was built by
the same describer over the same 30-s windows with the same 16 frames and
dialogue, so question-time re-perception with the same describer returns
what is already there, and the extra rows dilute. A grounder can only add
what L1 lacks — a different capability (speaker attribution, on-screen
text, film form, finer frame rate), not the same look again. Next:
`--continuity`, a second strictly descriptive pass per window (who says
what, on-screen text verbatim, shot scale / POV / repeated shots / cuts,
non-speech sounds) added as rows to the untouched v1 narratives.

**Replicate of narr_px v1 (identical prompt and flags, fresh generation,
2026-09-06): 43.0 on the fresh 300** — −4.0 [−7.7, −0.3] vs the original
generation's 47.0, and only +2.0 [−2.7, +6.7] vs ours 41.0. The describer's
output varies from generation to generation (the 235B MoE is not
deterministic at temperature 0), and the fresh-300 +6 was a favourable
draw. What stands: the full-test 48.3 / 47.4 are measured numbers on 1,837
questions for *that* catalog (+5.8 CI-clean vs 42.6 on the same questions),
and the dialogue share (+3.3) does not depend on generation. What does not
stand yet: the size of the narrative's contribution. Two things follow:
(1) a full-test replicate (all 270 videos regenerated, 1,837 answers) so
the paper reports both generations; (2) a two-view catalog (both
generations' narrative rows + clips) on the fresh 300 — if two independent
descriptions of the same window beat either alone, generation noise is a
lever, not just a hazard.

**Continuity channel (2026-09-06, fresh 300, gen-1 narratives + 1,185
continuity rows + clips, no pointer): 44.7 vs 47.0, −2.3 [−5.3, +0.7].** The
rows are exactly what was asked for (attributed dialogue line by line,
on-screen text verbatim, shot form, sounds — see sample in
`narr_px_cont/continuity/`), and they do not help: SR −6.8, TCI −7.9, PAR
+3.3, all n.s. Fourth added-rows variant in a row that is flat or negative
(grounder rows, key moments, continuity). The reader is not short of
descriptive detail at this point; the catalog's remaining gap to the human
rows is in *what the describer notices as mattering*, which the human rows
have and no extra descriptive channel supplies.

Two-view catalog (gen-1 + gen-2 narrative rows + clips, fresh 300): 43.3 —
+0.3 n.s. vs gen-2 alone, −3.7 [−7.0, −0.3] vs gen-1. Adding the second
generation does not recover the first one's score, so the gap between
generations is not "more evidence in one of them"; it is which things one
draw happened to say. Until the full-test replicate lands, the narrative
lever should be quoted as +2 to +6 on the fresh 300 and +5.8 on the one
full-test generation. (A candidate cause besides sampling: OpenRouter routes
the 235B across providers with different quantisations; pinning a provider
is worth doing for any final run.)

**Full-test replicate (2026-09-06, second generation of all 270 videos'
narratives, same prompt, no pointer): 45.6**, +3.1 [+1.1, +5.1] vs ours
42.6 (CI-clean), −2.7 [−4.4, −1.1] vs the first generation's 48.3. Both
generations clear Gemini-2.5-Pro's 45.0; the honest statement is
**+3 to +6 over the clip catalog depending on the draw, mean of two
generations 47.0 (+4.4)**, and the paper reports both. PAR is the one type
that gains in both generations with CI clear of 0 (+12.9 here, +6.2 before);
TA loses in both (−4.0 / −4.5). OpenRouter serves this model from five
providers at fp8 / bf16 / unknown quantisation; the client now records the
serving provider and can pin one (`OPENROUTER_PROVIDER_ORDER`), which any
final number should do.

**Grounding headroom for a small reader: none (2026-09-06, fresh 300, 8B
reader on narr_px_plus).** Whole catalog 37.3; the gold inference-shot rows
as a pointer 37.3 (**+0.0 [−2.3, +2.3]**); gold rows only (top-8 by overlap)
35.3 (−0.4 n.s.). Pointing the 8B at the exact human-annotated evidence does
nothing, so its 10-point gap to the 235B on the same catalog (47.0) is
inference over evidence it already has, not evidence selection. A learned
grounder for small readers therefore has nothing to win on Video-Holmes;
what a small model would need is *reader* training — and the only
verifiable signals for that are the ones the skill structure provides
(answer, citation hits on annotated evidence, deterministic checks).

**Controller as efficiency (2026-09-06, fresh 300, narr_px_plus, 235B reads
only the top-k BM25 rows of ~66):** top-8 38.7, top-16 36.7, top-24 38.3 —
−8 to −10 [CI-clean] vs the whole catalog 47.0. An untrained selector does
not reach parity at any k, so "same accuracy at 1/4 the reader input" is
not free; it is a target a trained selector would have to hit (from ~38 to
47 at k=16). The old OPD reranker was trained on clip-only catalogs and
scored ≈ BM25 on clue recall; whether it closes this gap on the narrative
catalog is the one measurement that would justify a controller on
efficiency grounds.






 On **CG-Bench** the no-pointer
variant scores 43.9 (vs 44.7 with pointer, −0.8 n.s.; vs 40.5 +3.4 n.s.).



**Generality on CG-Bench (237-question heldout, 67 videos, mean 31 min,
2026-09-06).** Same lever, 60-s windows × 16 frames, the benchmark's own
subtitles as dialogue, narrative rows (~31 per video) prepended to the
original catalog, same reader and pointer as `cg_direct_237`: **44.7 vs
40.5, +4.2 [−1.3, +9.7]** — the same direction and size as Video-Holmes,
under-powered at n=237. Built in ~1 h on the login node.











## Reasoning-grounded accuracy: does the decomposition make *right answers more right*? (2026-09-06)

Plain accuracy pays for ungrounded guesses; process metrics pay for citations
that lead nowhere. The claim the decomposition has to support is the joint
one: **the answer is right AND the evidence it gives is right.** Each
benchmark judges the second half with its own supervision:

- Video-Holmes: the official `evaluate_reasoning.py` judge — grounded = TRAR
  (right answer, right reasoning); a right answer with wrong reasoning (TWAR)
  does not count.
- VRBench: annotated timed reasoning steps — grounded = right answer AND the
  cited clips hit the steps (`scripts/eval/grounded_accuracy.py`: step recall
  ≥ 0.5, or any hit, or citation precision ≥ 0.5).
- CG-Bench: `clue_intervals` — grounded = right answer AND the cited clips hit
  a clue interval.

The control is always the **single-call direct answer with a rationale that
cites clips** (same model, same catalog, same pointer): the decomposition must
beat a reader that is also asked to show its evidence, not a reader that
shows nothing.

| paired comparison (decomposition − cited direct) | accuracy | grounded accuracy |
|---|---|---|
| VH fresh 300, old catalog, graph2 no-look | −1.7 [−6.7, +3.7] | TRAR 34.0 → 35.0, **+1.0 [−4.0, +6.0]** |
| VH fresh 300, old catalog, graph2 + sub-question look | −0.7 [−6.0, +4.7] | TRAR 34.0 → 36.0, **+2.0 [−3.3, +7.3]** |
| VH fresh 300, **narr_px_plus catalog**, graph2 vs direct+rationale (official judge) | −1.0 [−6.3, +4.0] | TRAR 39.0 → 39.3, **+0.3 [−4.3, +5.3]** (TRAR share of right answers 88% vs 85%) |
| VRBench 480 (pilot 60 videos) | −0.8 [−4.6, +2.7] | correct ∧ any step hit 40.8 → 41.0, +0.2 [−4.2, +4.6]; correct ∧ recall ≥ 0.5 31.7 → 29.2, −2.5 [−6.7, +1.9]; correct ∧ precision ≥ 0.5 24.4 → 28.3, **+4.0 [−0.4, +8.1]** |
| CG-Bench 237 (67 videos), graph2 vs direct+rationale, clue_intervals | −1.3 [−6.8, +4.2] | correct ∧ any clue hit 27.4 → 28.3, +0.8 [−4.6, +6.3]; correct ∧ precision ≥ 0.5 14.8 → 19.8, **+5.1 [−0.4, +10.1]** |

Reading so far: the direction is consistent (right answers come with
better-aimed citations: VRBench precision 53.7 vs 50.9 on cited questions,
478 vs 413 questions citing at all; VH TRAR share 91% vs 84%) but no
grounded-accuracy difference is CI-clean yet, and on VRBench it flips sign
between a precision- and a recall-based grounding criterion — the chain cites
fewer, better clips, the rationale cites more, so it covers more steps. Both pending measurements have landed (rows above): on Video-Holmes the
decomposition does not raise judge-grounded accuracy on either catalog; on
VRBench and CG-Bench it raises precision-grounded accuracy.

**Pooled, VRBench 480 + CG-Bench 237 (n = 717 paired, decomposition − cited
direct, 2026-09-06):** accuracy −1.0 [−4.0, +2.1]; correct ∧ any cited gold
hit +0.4 [−3.1, +3.8]; **correct ∧ citation precision ≥ 0.5: 21.2 → 25.5,
+4.3 [+1.0, +7.7]** (CI-clean). So the statement the data supports is:
*the decomposition does not change how often the answer is right, and it
raises by about four points how often a right answer comes with citations
that are mostly on the annotated evidence.* Two caveats travel with it: the
precision criterion was chosen after seeing that the recall criterion is
null (the chain cites fewer, better clips; the rationale cites more, so it
covers as many gold steps), and both sets were used for earlier decisions.
A clean confirmation needs untouched data with the criterion fixed first —
the VRBench held-out 60 videos (61–120 by duration) are the natural set.

**Pre-registration (2026-09-06, before any held-out number exists).** VRBench
videos ranked 61–120 by duration (25.0–38.6 min; the pilot used 1–60),
every question of each video, L1 built with the pilot's exact settings,
same reader/pointer/flags as the pilot. Primary outcome: paired difference,
graph2 (probability ranking, no looking) minus single-call direct with a
clip-citing rationale, on **correct ∧ citation precision ≥ 0.5**. Secondary:
accuracy, correct ∧ any gold step hit, correct ∧ step recall ≥ 0.5. Success =
the primary CI excludes 0 in the positive direction. Chain:
`scratchpad/vrbench_heldout_chain.sh`; outputs
`/fs/nexus-scratch/wuxiyang/vrbench_pilot_v1/heldout60/`.



## Pre-registered: train the small reader with skill-derived verifiable rewards (decided 2026-09-06)

**Why this and nothing else.** Every prompt-only lever is exhausted: the
catalog is where accuracy came from (+3 to +6), decomposition is at parity,
question-time grounding adds nothing, and the 8B reader has zero grounding
headroom (gold pointer +0.0) — its 10-point gap to the 235B on the same
catalog is inference over evidence it already has. The one untested path is
to *train the reader*, and the only dense verifiable signals for that are
the ones the atomic-skill structure provides: answer correctness, citation
hits on annotated evidence (Video-Holmes inference shots; VRBench reasoning
steps; CG clue intervals), and deterministic checks (time-order consistency
of cited clips). This is the RLVR reading of atomic skills: they define the
reward, not the reasoning.

**Setup.**
- Base: an open ~8B instruct model (the deployed reader family). Input:
  question + options + the narr_px_plus-style catalog (clips + dialogue +
  window narratives) as text. Output: a short cited rationale + option.
- Train data: Video-Holmes **train** split only (videos disjoint from test),
  L1 built with the same recipe (9B clips, whisper, 235B window narratives).
  Optionally VRBench pilot-60 questions (never the held-out 60).
- Stage 1 SFT (LoRA): targets = 235B cited rationales on train questions,
  kept only when the answer is right; ~1–2k examples.
- Stage 2 GRPO: on-policy samples (G=8) per train question; reward =
  1.0·correct + 0.5·(citation precision on inference shots, only when
  correct) + 0.1·format − 0.05·length; time-order consistency penalty on
  TA-type questions. Control arm: same GRPO with outcome-only reward.
- Eval (never trained on): Video-Holmes test 1,837 (both catalog
  generations), VRBench held-out 60, CG 237. Metrics: accuracy, grounded
  accuracy (correct ∧ citation precision ≥ 0.5), official judge TRAR share.

**Success criteria fixed now.** Primary: 8B+RLVR ≥ 45.0 on the full VH test
with the paired CI vs the 8B direct run (40.4) excluding 0, and grounded
accuracy above the 8B direct. Secondary: process-reward arm > outcome-only
arm on grounded accuracy (this is the atomic-skills claim); transfer on
VRBench held-out. Budget: 3 weeks; if Stage 2 does not beat Stage 1 on the
fresh 300 within that, report Stage 1 and stop.

Implementation (2026-09-06): `trainer/reader/` — `prompting.py` (exact
evaluation prompt via `direct_messages`, JSON target), `rewards.py`
(correct + citation precision on annotated spans, paid only when correct;
time-order check), `build_sft_data.py` (teacher rationales kept only when
verifiably right), `sft_lora.py` (LoRA, completion-only loss, FA2),
`rft_iterate.py` (sample K from the current policy served by vLLM, keep the
best verified completion per question → next SFT round). Stage 2 runs as
this verifiable-reward iteration first; GRPO proper is swapped in if TRL
can be installed alongside the local FA2 environment without breaking it.
Train L1 for the 233 train videos runs on the gamma partition (32 GB / 4
CPU QoS); the base reader's own baseline (Qwen3.5-9B via vLLM, fresh 300,
direct and cited) is queued so the trained model is compared with itself.
Pipeline smoke on the fresh 300 (not training data): the 235B teacher is
right on 135/300, so verified SFT rows come from ~45% of questions per
teacher sample (sample 4× on the rest); tokenised prompts are p50 10.7k /
p90 15.6k / max 16.2k tokens, so training runs at max_len 24k (30/135 rows
would be dropped at 16k) with LoRA, batch 1, gradient checkpointing.
GRPO proper is available locally after all: the `swift` conda env (which
`.venv-qwen35-serve` resolves to) carries ms-swift 3.10, TRL 0.23 and vLLM
0.8.5, so Stage 2 can run as GRPO with a vLLM rollout server and the
`trainer/reader/rewards.py` reward as the plugin; the RFT loop stays as the
fallback.

Scope decision (2026-09-06): the VRBench pilot-60 questions (480, catalogs
already built) join the training set so VRBench is a trained domain too;
the held-out 60 videos stay evaluation-only. CG-Bench has no trainable
catalogs (the 67-video / 237-q set is the evaluation set) and is reported as
zero-shot transfer. Claims the plan can support: accuracy up on VH and
VRBench (trained), groundedness up (process-reward arm vs outcome-only),
and deployment efficiency (9B dense on one 48 GB GPU vs a 235B-A22B reader)
measured as latency/throughput locally and API cost per 1k questions.

**Full three-benchmark evaluation (decided 2026-09-06).** Video-Holmes is
complete (1,837). CG-Bench will be run on the official mini set (3,000 q over
1,118 videos, mean 28 min; 67 built) and VRBench on a duration-stratified
240-video subset (60 per duration quartile, mean 105 min, disjoint from the
pilot 60 and held-out 60; ~2,000 q), extended to all 957 usable videos if
GPU allows. Cost: CG-mini L1 ≈ 1,051 videos × ~15 min ≈ 260 GPU-h plus ~$600
of window narratives (subtitles serve as dialogue); VRBench subset ≈ 240 ×
~40 min ≈ 160 GPU-h plus ~$150 whisper and ~$150 narratives; answering both
with the 235B reader and the trained 9B ≈ $100. Allowlists and launchers:
`scripts/launch/launch_cg_mini_l1.sh` (16 shards), `launch_vrbench_subset240.sh`
(12 shards); queued behind the train-split L1.

**Base reader's own baseline (2026-09-06, Qwen3.5-9B via local vLLM, text
prompt, narr_px_plus catalog, fresh 300, no pointer): direct 36.3, with cited
rationale 38.0.** The trained model is compared against these; the SFT
watershed is ≥ 41 on the same 300 (rationale format), and the full-test bar
stays 45.0. For reference the 235B reads the same catalog at 47.0 and the
API 8B at 37.3.

Scheduling note (2026-09-06 18:00): `repass16` (question-aware 9B repass,
87/300 done) was paused to free the user's scavenger share for the train-split
L1; its stage caches are kept and the shards can be resubmitted with
`scripts/launch/launch_vh_l1_levers.sh repass16` to fill T3 later.

**Training data built (2026-09-08 00:15).** Train-split L1 (233 videos, 9B
clips), whisper, and window narratives + dialogue + clips for all 1,551
train questions (slim examples). Teacher (235B, cited rationale) in three
option orders: 62.3 / 62.5 / 61.4% right on the train split (vs 45% on the
test fresh 300 — the train questions are easier). Verified SFT rows: 2,886
(right answer; the citation-precision filter did not bind because the train
examples carry no inference-shot spans); prompts p50 13.5k / p90 18.3k /
max 22.2k tokens, 252 supervised tokens per row. First SFT job OOMed on the
full-sequence logits (13k × 248k vocab in fp32 ≈ 13 GB); fixed by
materialising logits for the supervised tail only (`logits_to_keep`), one
epoch to fit the 8-h scavenger limit. Resubmitted.

The resubmitted run trained at ~560 s per optimizer step (8 × 16k-token
sequences), i.e. ~70 s per sequence — ten times the expected cost — because
Qwen3.5's Gated-DeltaNet layers fell back to the torch implementation
(`flash-linear-attention` and `causal-conv1d` were absent from the training
env; transformers logs "fast path is not available"). The epoch-only-saving
run was cancelled (it could not finish in 8 h); an insurance run with step
checkpoints continues on the slow path; the fast-path libraries are being
installed and the full run relaunched afterwards.
Fixed 2026-09-08 02:00: `flash-linear-attention` installed; `causal-conv1d`
wheels and source build both mismatch the environment's C++ ABI, so a
pure-torch `causal_conv1d` shim (depthwise causal conv, numerically
identical) with distribution metadata makes transformers' fast-path gate
pass. Full run `sft_v1c` (2,886 rows, 24k tokens, LoRA r=16, 1 epoch,
checkpoints every 40 steps): **59 s/step vs 487 s/step** on the slow path;
361 steps ≈ 6 h.

**Early read at checkpoint-80 (2026-09-08 05:00): 27.3 vs base 38.0, −10.7
[−17.7, −3.7] — 262/300 answers "A".** Cause found, not the data: the
training prompts were rendered with the tokenizer's default chat template,
which opens a thinking block (`...assistant\n<think>\n`), while the
evaluator serves with thinking disabled (`<think>\n\n</think>\n\n`). The
adapter learned to emit the JSON after an open `<think>` and, given the
closed prefix at evaluation, fell back to reasoning prose and a collapsed
label. Fixed by rendering training prompts with `enable_thinking=False`
(test asserts it); the mis-rendered run was cancelled and two corrected
runs launched: `sft_v2` (all 2,886 rows, lr 1e-4) and `sft_v2b` (1,600
rows, lr 5e-5 — confirmed from the checkpoint's trainer state) — each with step checkpoints and its own evaluation chain. Both run on the
fast path (52–58 s/step); loss 7.0 → 5.2 over the first 40 steps.

**The collapse was vLLM's LoRA serving, not the adapter (2026-09-08 07:20).**
The corrected-prefix checkpoint-40 still collapsed when served with
`--enable-lora` (230/257 "A", 6k-char prose). An HF-only diagnostic
(`trainer/reader/debug_adapter.py`) on training rows shows the adapter is
healthy: completion-only loss equals the full-logits loss at the same
positions (0.865 = 0.865; alignment correct), the adapter lowers the loss
against the base (1.21 → 0.87, 0.82 → 0.54), and greedy generation from the
training prompt yields clean `{"reasoning": …, "label": …}` JSON with clip
citations. vLLM 0.25's LoRA path for Qwen3.5 (served through the
ConditionalGeneration wrapper) mis-applies the adapter. Fix: merge the
adapter into the weights (`trainer/reader/merge_adapter.py`) and serve the
merged model; the evaluation job now does this automatically. The
LoRA-served early reads (27.3 / 28.3) are void.

**Correction (2026-09-08 10:45): the collapse was the served model's *name*,
not vLLM's LoRA path.** The evaluator's client sends
`chat_template_kwargs: {enable_thinking: false}` only when the model name
contains "qwen3"; the adapter was served as `reader`, so vLLM rendered the
thinking-open prefix (`<think>\n`) and the trained reader — trained on the
closed prefix — produced thinking prose and a collapsed label (62/230
outputs even closed a `</think>` block; the base run, served as
`Qwen/Qwen3.5-9B`, had thinking off and 300/300 JSON-first outputs). Merged
serving reproduced the same collapse for the same reason. Fix: serve as
`qwen3.5-9b-reader`. All earlier LoRA/merged reads are void; the two
evaluations are rerunning.

**Where the 9B loses to the 235B on the same catalog (fresh 300, cited
rationale runs; 2026-09-06).** 235B right / 9B wrong: 64; 9B right / 235B
wrong: 41; both right: 73 — the union is 59%, so the small model is not a
strict subset and has room. Three findings that shape the training recipe:
(1) **letter prior**: the 9B picks B 64× and C 61× against gold 47 / 32, and F
only 45× against gold 70 — a positional bias worth several points on its own,
so training data is generated in several option orders (`--shuffle-options`,
permutation recorded and replayed) and the model cannot learn a letter;
(2) **localisation**: in the 64 questions the 235B gets right and the 9B
wrong, the 9B cites none of the 235B's clips in 53% — half of the loss is
looking at the wrong evidence, which cited-rationale SFT targets directly
(only teacher rationales with citation precision ≥ 0.5 are kept where
gold spans exist); the other half is inference over the right evidence;
(3) **not length**: accuracy by catalog-size tercile is 42 / 29 / 43 for the
9B, flat for the 235B — no long-context degradation to fix. Types with the
largest gaps: PAR (30 vs 53), TCI (26 vs 42), TA (36 vs 48); MHR and SR are
equal. Recipe for the watershed: teacher rationales sampled up to 4× per
question in 2 option orders, filtered by answer and citation precision,
mixed with the base model's own correct-and-cited samples (K = 8), LoRA SFT
at 24k tokens; evaluated in the rationale format against 38.0.



## Feasibility probe: are reasoning failures reusable sub-trajectories? (2026-09-06)

`scripts/eval/mine_failure_subtrajectories.py` fits every direct-reader
failure (gold explanation / inference shots / VRBench reasoning_process +
the reader's wrong rationale) to the fixed skill ontology with a failure code
and a per-step "what would executing this need" label. Three failure sets:
VH fresh 300 on the narr_px_plus catalog (163 wrong), VH fresh 300 on the old
catalog (178), VRBench pilot 480 (157).

**Exact sub-trajectories do not recur.** 134 distinct programs among 163 VH
failures, top-5 cover 22 (13%); VRBench 134/157, top-5 cover 16. As
"reusable programs" the hypothesis fails.

**Failure modes do recur, and split three ways:**

| failure code (VH new / VH old / VRBench) | share | what fixes it |
|---|---|---|
| clue in the text but mis-weighted (official judge: RE 87/98) | **60% / 60% / 59%** | nothing tool-shaped: the fact is read and the inference is wrong (graph2 has the gold option at rank 2 in 25/98). This is a *reader training* target, i.e. the RLVR framing, not a skill-library target |
| deterministic structure the model gets wrong: temporal order (VH TA 15/15 of that type) / counting (VRBench 28, and "Counting Problems" is 94/480 questions at **32%** direct accuracy vs 71–93% on every other type) | 10% VH / 18% VRBench | a deterministic skill — the one pattern that has worked (timeline skill) |
| clue not in the text / dialogue needed | 10–14% / 2% | perception at the window+audio level (narr_px, window grounder) |
| film grammar / symbolism | 5–8% | out of reach |

Verdict: the sub-trajectory-bank idea is not feasible as stated (no recurring
programs), and the GPU spend on train-split L1 for mining is not justified.
What is feasible and concrete: (1) VRBench counting — one deterministic
`count_events` skill on a fifth of the benchmark sitting at 32%; (2) temporal
ordering (done, half the TA gap); (3) the window grounder for the
perception tail. The 60% majority is where "atomic skills as an RLVR variant
that regularises reasoning" would have to act, and only training can test it.

**The counting skill, tried (2026-09-06, VRBench pilot, 94 counting
questions, paired vs direct 31.9).** `scripts/eval/count_events_skill.py`:
the model only enumerates instances per 15-row chunk with the running list
in view; merging (explicit links, same identity within 15 s), the count, and
the count→option mapping are code. Deterministic mode **8.5 (−23.4 [−33.0,
−13.8])**; the reader given the enumerated list 21.3 (−10.6). Diagnosis:
the enumeration is wrong in both directions — occurrences are under-counted
(median −2; "meetings between Hong Kim and Hoa Hiep" → 0 found, the names
and the event are not in the clip text), and "major scenes" / "relationships"
/ "activities" are over-counted 5–10× (41 raw instances for a gold of 3),
because the unit is the annotator's story segmentation, not a clip-level
event. Gold counts are 3–5 on 85% of questions and direct is at 32% ≈
chance: VRBench counting is a *semantic* count over a narrative, not a
deterministic one over events. So the deterministic-skill pattern holds
only where the unit is well defined (time order); it does not transfer to
counting here. Retired.


## CG-Bench QA accuracy (first measurement, 2026-09-05)

Only grounding had ever been measured on CG. Direct answering over the whole
catalog with the BM25 pointer and the 235B reader, on the same 237 heldout
questions: **40.5% [34.2, 46.8]** (6–8 options per question; by domain
Instruction & Knowledge 57.1, Art & Culture 53.3, Music/TV 48.0, Life Record
32.4). This is a second accuracy axis the paper can report next to the
grounding table.

## Disk (2026-09-04)

`/gamma/projects` has a 2 TB quota and was 100% full (2.6 GB free). With the
user's go-ahead, nine stale `sft_training/` directories (~140 GB; none
referenced by the paper's eval reports) were **moved** to
`/fs/nexus-scratch/wuxiyang/moved_from_sft_training/` (not backed up) — see
`sft_training/MOVED_TO_SCRATCH.txt` for the list and the scripts whose default
paths pointed there. The VRBench pilot writes to scratch through the
`output/vrbench_pilot_v1` symlink.

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

**Outcome (2026-09-05).** All 28 targeted examples repaired: 1 placeholder clip
left of 2,144 (a JSON-truncated coarse clip in cg_bench:841 that fails
deterministically on retry). The heldout CG pointwise was rebuilt into
`heldout_pointwise_cg_repaired_v1/cg_bench/pointwise.jsonl`: 67 examples,
4,449 rows, **52 placeholder rows (1.2%)** against 43.1% in the frozen set.
The residual is not in the repaired lane: 51 rows sit in 7 examples from the
July lanes (cg_bench:6 26/87, cg_bench:129 15/97, five others ≤3 rows). Five
scoring jobs (SFT, OPD α=0.75, GRPO seeds 42–44; decision_logit) were
submitted on the rebuilt set; the spurious Video-Holmes jobs the submit script
also queued were cancelled. A follow-up repair of cg_bench:6 and :129 would
bring the set under 1% everywhere; the first official numbers come from this
1.2% set and are labelled as such.

**First trustworthy official CG-Bench numbers (repaired v1 set, 67 heldout
questions, decision_logit scoring, ≤5-interval protocol):**

| model | top-1 mIoU / rec.@IoU | top-2 mIoU / rec.@IoU |
|---|---|---|
| published best (Qwen2-VL-72B) | 3.58 / 5.32 | |
| published best 7–8B | 1.63 / 2.89 | |
| BM25 (no learning) | 4.44 / 6.57 | 4.87 / 7.46 |
| SFT (v11) | 5.27 / 7.46 | 4.27 / 5.67 |
| **OPD α=0.75** | **6.13 / 8.96** | 5.19 / 6.87 |
| GRPO seeds 42/43/44 | 6.13 / 8.96 (top-1 identical to OPD on 64–66/67) | 5.19 / 5.19 / 5.07 |

Every learned model and even BM25 clear the published 72B number at top-1;
OPD at top-1 is 1.7× the 72B mIoU. The GRPO adapters have distinct weights
but change OPD's top-1 pick on only 1–3 of 67 questions, so GRPO ≈ OPD on
CG grounding. **k chosen on the cg14 dev split, not on heldout:** OPD dev mIoU / rec.@IoU
by top-k = 9.05/14.29 (k=1), 4.76/5.71 (2), 3.22/1.43 (3), 2.65/1.43 (4),
2.12/1.43 (5); BM25 4.52/5.71 at k=1 and worse beyond. So the reported
system prediction is the **top-1 interval** (the official protocol allows up
to five; more intervals only dilute set-IoU here). Bootstrap CIs on the 67
heldout questions (10k paired draws), top-1:

| model | mIoU [95% CI] | rec.@IoU [95% CI] | vs BM25 (paired) |
|---|---|---|---|
| BM25 | 4.43 [1.49, 8.20] | 6.57 [2.09, 12.24] | — |
| SFT | 5.27 [2.34, 8.85] | 7.46 [2.99, 12.84] | +0.84 [−1.71, +3.21] / +0.90 [−3.28, +4.78] |
| **OPD** | **6.13 [3.11, 9.82]** | **8.96 [4.18, 14.33]** | +1.69 [−0.60, +3.90] / +2.39 [−1.19, +5.97] |
| GRPO s42 | 6.13 [3.01, 9.93] | 8.96 [4.48, 14.63] | +1.69 [−0.63, +3.89] / +2.39 [−1.49, +5.97] |

At top-2 the learned models lose their edge (OPD − BM25: +0.33 / −0.60). So:
the point estimates order BM25 < SFT < OPD ≈ GRPO and OPD's mIoU is 1.7× the
published 72B figure, but with 67 questions **no pairwise difference is
CI-clean** and the comparison to published numbers is on a different question
set. The cheap fix is more heldout *questions* on the same cataloged videos
(CG-Bench has several questions per video with their own clue intervals; no
GPU needed). Done: the 67 heldout videos carry 718 questions in the full
`cgbench.json` and **237 in `cgbench_mini.json`** (the 3,000-question config
the leaderboard evaluations use); the per-question derivation
(`scripts/eval/derive_full_question_examples.py --dataset cg_bench`, adapter
default `use_mini=True`) produced all 237 with the full coarse+fine catalog
(≈87 clips) and clue spans on every row. New set
`heldout_pointwise_cg_questions237_v1`: 237 questions, 15,655 rows, 1.09%
placeholder rows. Five scoring jobs submitted (7456731–7456739); BM25 scored.
This 3.5× larger set is what the paper's CG table should use.

**Result on the 237-question set (top-1, k chosen on dev; 10k paired
bootstrap draws):**

| model | mIoU [95% CI] | rec.@IoU [95% CI] | vs BM25 (paired) |
|---|---|---|---|
| published Qwen2-VL-72B | 3.58 | 5.32 | |
| BM25 (no learning) | 5.05 [3.54, 6.71] | 7.51 [5.15, 10.21] | — |
| SFT v11 | 3.36 [2.12, 4.75] | 4.89 [3.04, 6.92] | −1.70 [−3.47, +0.01] / −2.62 [−5.40, +0.08] |
| OPD α=0.75 | 5.22 [3.74, 6.83] | 7.68 [5.32, 10.21] | **+0.17 [−1.72, +2.04] / +0.17 [−2.87, +3.12]** |
| GRPO 42/43/44 | 5.18 / 5.18 / 5.12 | 7.68 / 7.68 / 7.59 | ≈ OPD |

The 67-question edge (OPD − BM25 +1.69) **did not survive** the larger set:
on 237 questions OPD ≈ BM25 (+0.17, CI centred on 0), SFT is *below* BM25
(significantly so at top-2), and GRPO ≈ OPD. What does hold: every retriever
over the 30-s clip catalog — BM25 included — is above the published 72B
figures on this heldout slice (different question set; indicative only).
Honest CG claim: **the catalog pipeline clears the published grounding
numbers; the learned controller does not beat lexical retrieval on CG-Bench
grounding.**

Final sets (2026-09-05): the two July-lane examples were repaired in place
(cg_bench:6 → 0/97, :129 → 0/107). The 67-question set rebuilt as
`heldout_pointwise_cg_repaired_v3` is at 0.2% placeholder rows. The
per-question derivation writes *copies* of the per-video L1, so the 237-set
had to be re-derived (`cg_heldout_questions_v2`) before its rebuild
(`heldout_pointwise_cg_questions237_v4`) picked up the repaired schemas —
a rebuild from the stale copies (v3) stayed at 1.09%. The re-derived 237-set v4 is at
**0.33% placeholder rows** (51 of 15,655) and the 67-set v3 at 0.2% — under the
1% gate everywhere. **FINAL CG-Bench table (237-question set v4, 0.33% placeholder rows, top-1
chosen on dev, 10k paired bootstrap draws):**

| model | mIoU [95% CI] | rec.@IoU [95% CI] | vs BM25 (paired) |
|---|---|---|---|
| published Qwen2-VL-72B | 3.58 | 5.32 | |
| published best 7–8B | 1.63 | 2.89 | |
| BM25 (no learning) | 4.99 [3.49, 6.66] | 7.43 [5.06, 10.13] | — |
| SFT v11 | 3.36 [2.12, 4.75] | 4.89 [3.04, 6.92] | −1.64 [−3.41, +0.07] / −2.53 [−5.32, +0.17] |
| OPD α=0.75 | 5.22 [3.74, 6.83] | 7.68 [5.32, 10.21] | +0.22 [−1.68, +2.08] / +0.25 [−2.78, +3.21] |
| GRPO 42 / 43 / 44 | 5.18 / 5.18 / 5.12 | 7.68 / 7.68 / 7.59 | ≈ OPD |

The fully repaired 67-question set (v3, 0.2%) reproduces the earlier
reading exactly (OPD 6.13/8.96, BM25 4.43/6.57, +1.69 [−0.60, +3.90]). The
repairs moved nothing beyond noise; the table above is the paper's CG result. The controller's CI-clean win remains Video-Holmes clue recall
(+3.57).

Diagnostics on the 237: OPD and BM25 choose the same top-1 clip on only
20/237 questions yet score alike; the oracle single-clip ceiling is **mIoU
28.94 / rec.@IoU 46.08** (every question has an overlapping clip), so the
headroom is 5× and neither retriever exploits it. Clue recall of an
overlapping clip within top-k: BM25 22.8 / 35.0 / 55.3 / 66.2 (k=1/2/5/10),
OPD 23.2 / 31.6 / 52.7 / 65.8 (R@5 −2.5 [−10.6, +5.9]), SFT 15.6 / 25.7 /
43.5 / 63.3 — so on CG the controller did not learn to rank clue clips higher
at any depth. The cg14 dev split (14 questions) and the 67-question set both
flattered OPD; only the 237 set is large enough to see this. Reciprocal-rank
fusion of OPD and BM25 reads 6.00 / 8.78 on the 237 — but the dev split did
**not** prefer it (7.62 vs OPD 9.05), so it cannot be reported as the system;
noted as an observation only.

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
