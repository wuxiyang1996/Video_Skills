# Paper tables (consolidated 2026-09-06; source: docs/l2_sota_targets.md)

All differences are paired bootstrap 95% CIs on the same questions unless
noted. "Ours" = Qwen3.5-9B clip catalog (4 frames / 4-s clips), whole catalog
in one reader call. Reader = Qwen3-VL-235B (text) unless noted. Fresh 300 =
300 Video-Holmes test questions used for configuration selection; full test =
all 1,837 (never used for selection).

## T1. The evidence ladder (Video-Holmes, same reader)

| catalog given to the reader | fresh 300 | vs ours | full test 1,837 | vs ours |
|---|---|---|---|---|
| ours: 9B clip descriptions | 41.0 | — | 42.6 | — |
| + whisper dialogue rows | 44.3 | +3.3 [−0.7, +7.3] | — | — |
| + 30-s narrative from pixels (16 frames + dialogue), gen 1 | **47.0** | **+6.0 [+1.3, +10.7]** | **48.3** | **+5.8 [+3.9, +7.7]** |
| same, gen 2 (identical prompt, fresh generation) | 43.0 | +2.0 [−2.7, +6.7] | **45.6** | **+3.1 [+1.1, +5.1]** |
| human segment rows only (annotation, no clues) | 53.3 | +12.3 [+7.0, +17.7] | — | — |
| human segment + clue rows (question source; leakage-inflated) | 62.0 | +21.0 [+15.3, +26.7] | — | — |
| published: Qwen2.5-VL-7B / Gemini-2.5-Pro | | | 27.8 / 45.0 | |

Full-test by type (gen 1 / gen 2 vs ours): MHR 43.4/43.1 vs 39.8; SR 59.2/57.2
vs 53.8; IMC 56.2/50.7 vs 49.3; TCI 49.1/41.4 vs 39.9; CTI 47.0/41.5 vs 37.8;
TA 30.0/30.0 vs 34.0; PAR 49.0/53.1 vs 40.2.

## T1b. Training the reader (Qwen3.5-9B, LoRA SFT on verified 235B rationales; rationale format)

| reader | fresh 300 | full test 1,837 | vs 235B teacher (48.3) | vs base 9B |
|---|---|---|---|---|
| base Qwen3.5-9B | 38.0 | 41.0 | −7.3 [−9.7, −4.9] | — |
| sft_v2 (2,886 rows, 1 epoch) | 46.0 | **51.2** (gen-2 catalog: **51.4**, +5.8 [+3.5, +8.2] over the 235B's 45.6) | **+2.8 [+0.5, +5.1]** | **+10.1 [+7.8, +12.5]** (n=1,837) |
| sft_v2 checkpoint-320 | 48.0 | — | | |
| sft_v2b (1,600 rows, lr 5e-5) | 45.7 | — | | |
| sft_v3p (822 citation-precise rows; process arm) | 41.0 (grounded 5.3 vs sft_v2 6.0, n.s.) | — | | +3.0 [−3.3, +9.3] |

By type on the full test (sft_v2 − 235B): TA +20.0*, MHR +3.9, CTI +2.6, TCI +1.8, SR +0.3, PAR −1.6, IMC −4.0.
Zero-shot transfer: VRBench pilot 480 sft_v2 66.9 vs base 67.1 (n.s.); CG 237 (64k) sft_v2 39.0 = base 9B 39.0 (+0.0 [−6.5, +6.5]), 235B 45.5. In-domain VH+VRBench and VH+VRBench+CG readers in training.

## T2. Generalisation of the catalog recipe

| benchmark | ours | + dialogue + narrative + clips | Δ |
|---|---|---|---|
| CG-Bench 237-q heldout (67 videos, 31 min mean; subtitles as dialogue, 60-s windows) | 40.5 | 44.7 | +4.2 [−1.3, +9.7] |
| VRBench pilot 480 (60 videos, 20–25 min) — catalog recipe not yet applied | 72.1 | — | — |
| CG-Bench official clue grounding (237 q): OPD 5.22 mIoU / 7.68 rec@IoU vs BM25 5.05/7.51 vs published 72B 3.58/5.32 | | | |

## T3. Describer levers that did not move accuracy (fresh 300, paired vs the row they modify)

| lever | acc | Δ | reading |
|---|---|---|---|
| 235B as clip describer (4 frames, same schema prompt) | 39.3 | −1.7 [−5.3, +2.0] vs 41.0 | describer capacity is not the gap |
| narrative synthesised from our clip *text* (no new look) | 38.7 | −2.3 [−6.3, +1.7] vs 41.0 | gap is perception, not fragmentation |
| model-written "key moment" rows (implications) | 40.0 | −7.0 [−10.7, −3.7] vs 47.0 | conclusions in the catalog hurt |
| descriptive continuity rows (speakers, on-screen text, film form) | 44.7 | −2.3 [−5.3, +0.7] vs 47.0 | more description does not help |
| two generations' narratives together | 43.3 | −3.7 [−7.0, −0.3] vs 47.0 | generation variance, not coverage |
| question-time window grounder (re-look ±15 s, 16 frames + dialogue) | 41.7 | −5.3 [−9.0, −1.7] vs 47.0 | same perception as L1 adds nothing |
| 8 frames per clip (9B describer) | 41.3 | +0.3 [−3.3, +4.0] vs 41.0 | frame density is not the gap |
| question-aware repass (9B, 16 clips, 6 frames) | paused at 87/300 | | resumable |

## T4. Readers and pointers (full test unless noted)

| reader / pointer | old catalog | narrative catalog |
|---|---|---|
| 235B, whole catalog | 42.6 (pointer) | 48.3 no pointer / 47.4 pointer (gen 1) |
| 8B (Qwen3-VL-8B), whole catalog | 39.1 | 40.8 pointer / 40.4 no pointer |
| 8B given the gold inference-shot rows as pointer (fresh 300) | — | 37.3 vs 37.3 whole: **+0.0 [−2.3, +2.3]** |
| 235B given only top-k BM25 rows, k=8/16/24 (fresh 300) | — | 38.7 / 36.7 / 38.3 (−8 to −10 vs 47.0) |
| oracle retrieval (gold clue clips only), 235B, old catalog | −3.0 to −6.2 vs whole | — |

## T5. Atomic-skill decomposition vs a single reader call (accuracy)

| form of decomposition | evidence | n | Δ vs direct (same clips, same reader) |
|---|---|---|---|
| planner + skills, always-commit | BM25 top-4, full test | 1,823 | −9.4 [−11.9, −6.9] |
| planner + skills, always-commit | whole catalog | 300 | −12.7 [−19.0, −6.3] |
| hybrid (skill notes + vote, one answer call) | whole | 150 | −8.7 [−16.0, −1.3] |
| observations only (no scores) | whole | 150 | −13.3 [−20.7, −6.0] |
| graph2 v1 (one comparative ranking, citations) | whole, control 300 | 300 | −5.3 [−9.0, −2.0] |
| graph2 v2 (probability ranking) | whole, fresh 300 | 300 | −2.3 [−6.7, +2.0] |
| graph2 v2 + sub-question look | whole, fresh 300 | 300 | −1.3 [−5.3, +2.7] |
| graph2 v2 on the narrative catalog | whole+pointer, fresh 300 | 300 | +0.3 [−3.3, +3.7] |
| graph2 + window grounder | narrative catalog | 300 | −0.7 [−4.0, +2.7] |
| graph2, VRBench pilot | whole+pointer | 480 | −5.6 [−8.5, −2.9] (vs cited-rationale direct: −0.2 n.s.) |
| graph2, CG-Bench 237 | whole+pointer | 237 | −1.3 [−6.8, +4.2] vs cited direct |
| timeline skill (deterministic order by cited clip time), 200 TA | | 151 permutation q | fires 35–57; on fired: 26.3 → 35.1 (direct 36.8) |

## T6. What the decomposition does change (process)

| metric | direct (cited rationale) | decomposition | Δ |
|---|---|---|---|
| VH official taxonomy, wrong answers: VOE (omission) share | 17% | 3% | finds the clues |
| VH official taxonomy, wrong answers: RE (reasoning) share | 80% | 91% | infers worse over them |
| VH right answers judged reasoned-right (TRAR) | 84% / 85% | 91% / 88% | (old / new catalog) |
| VH grounded accuracy (right ∧ TRAR), fresh 300, old / new catalog | 34.0 / 39.0 | 35.0 / 39.3 | +1.0 n.s. / +0.3 n.s. |
| VRBench 480: correct ∧ citation precision ≥ 0.5 | 24.4 | 28.3 | +4.0 [−0.4, +8.1] |
| CG 237: correct ∧ citation precision ≥ 0.5 | 14.8 | 19.8 | +5.1 [−0.4, +10.1] |
| **pooled VRBench + CG (n = 717)** | 21.2 | 25.5 | **+4.3 [+1.0, +7.7]** |
| pooled: correct ∧ any gold step hit | 36.4 | 36.8 | +0.4 [−3.1, +3.8] |
| VRBench inference-step recall: OPD vs BM25 (retrieval controller) | 11.00 | 14.57 | +3.57 CI-clean (no accuracy transfer) |
| **VRBench held-out 60 videos (pre-registered; correct ∧ citation precision ≥ 0.5)** | 25.1 | 31.5 | **+6.5 [+2.4, +10.7]**; accuracy −1.8 n.s. |

## T7. Where the failures are (direct reader, fitted to the skill ontology)

| failure mode | VH new / VH old / VRBench | judge type | fixable by |
|---|---|---|---|
| clue in the text, mis-weighted | 60% / 60% / 59% | RE 87/98 | reader training only |
| deterministic structure (time order VH; counting VRBench) | 10% / 11% / 18% | RE | deterministic skill (timeline works; counting does not: units are semantic) |
| clue not in text / dialogue needed | 12% / 15% / 17% | RE/VOE | perception (window + audio) |
| film grammar / symbolism | 5% / 8% / — | RE | — |
| exact sub-trajectory programs recur? | top-5 cover 13% | | no |

## Pending fills
- T3: frames8, repass16 (GPU, scavenger).
- T6: VRBench held-out confirmation of the +4.3 (GPU L1 queued; criterion pre-registered).
- Training line (separate paper section, ARR): Qwen3.5-9B reader baseline → SFT → GRPO with skill-derived rewards; VH test ≥ 45 is the bar.

## T8. Efficiency: trained 9B reader vs the 235B teacher (Video-Holmes full test, 1,837 questions, same catalog)

| reader | params (active) | where | throughput | cost / 1,837 q | accuracy |
|---|---|---|---|---|---|
| Qwen3-VL-235B-A22B (API, OpenRouter/Alibaba, $0.21/M in, $1.9/M out) | 235B (22B) | cloud, 6 workers | ≈24 q/min | ≈$5.7 (p50 prompt 11.8k tok + ~300 out ≈ $0.0031/q; measured CG-length prompts $0.0076/q) | 48.3 |
| trained Qwen3.5-9B (LoRA merged, vLLM bf16, 64k ctx) | 9B | one L40S (48 GB), 8 workers | ≈23 q/min (1,837 q in 80 min after a 5-min load) | ≈$2.5 at a rented $1.8/h L40S ($0.0014/q); $0 on owned hardware | 51.2 |

Same wall-clock throughput on a single 48 GB GPU as the 235B API at 6 concurrent calls, ≈2× cheaper per question at rental prices,
26× fewer parameters, data never leaves the machine, and +2.8 accuracy. Timing from Slurm job 7475561 (base 9B, identical serving path);
API cost from OpenRouter pricing on 2026-09-08 and the measured CG teacher pass ($8.2 / 1,080 questions).
