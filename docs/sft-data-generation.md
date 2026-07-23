# MDP-Style Cold-Start SFT Data

This repository treats L1, L2, and repair as controller processes. The primary
training record is a transition, not a one-shot `video -> full graph` target:

```text
(state_t, action_t, observation_t, state_t_plus_1, reward_proxy_t, done)
```

Chat SFT JSONL is only a behavior-cloning package for the same action. It must
not contain hidden supervision, a post-action graph snapshot, or the final
answer outside the candidate claim being evaluated by a verifier.

## Exported Controllers

| Controller | State | Supervised action | Current cold-start source |
| --- | --- | --- | --- |
| L1 builder | visible clip schemas, partial L1 graph summary, recent tool failures, budget | next atomic create-node/create-edge/anchor/skip action | strict rollout `metadata.graph_compose.execution_trace` plus the grounded L1 graph |
| L1 patch | repair goal, one visible clip schema, partial L1 summary, budget | `apply_l1_evidence_patch` with grounded nodes and edges | `repair_01_plan.json`, `repair_02_clip_schemas.jsonl`, `repair_03_l1_patch.json` |
| L2 / repair | question-visible inputs, compact L1, prior rounds, budget | next L2 or repair tool action | compact verified expert demos |
| Auxiliary verifier | candidate claim, proposed evidence refs and visible evidence text, verifier policy | `supported` or `insufficient` with reason | option verifier records plus verified expert-demo rounds |
| Motif lifecycle | reusable candidate template, support statistics, boundary rules | candidate/shadow/rejected lifecycle action | mined motif bank plus evidence gates |

The deterministic runtime verifier remains authoritative. Learned verifier SFT
is an assistant or calibration model, not a replacement for the hard gate.
Motifs remain non-executable graph priors and must expand into ordinary L1/L2
nodes before verification.

## Reproducible Exports

Run from the `Video_Skills` repository root. These commands use historical
pseudo-gold artifacts from the sibling `video_skills_relaunched` checkout.

```bash
python -m dataset_clip_wrapper.export_l1_builder_sft \
  --rollout-jsonl dataset_clip_wrapper/output/pilot_corrected_v2_20260710/cg_bench/start_0_limit_1/examples.jsonl \
    dataset_clip_wrapper/output/pilot_corrected_v2_20260710/video_holmes/start_0_limit_1/examples.jsonl \
    ../video_skills_relaunched/dataset_clip_wrapper/output/batch3_cg_strict_qwen.jsonl \
    ../video_skills_relaunched/dataset_clip_wrapper/output/batch3_video_holmes_strict_qwen.jsonl \
  --include-dataset cg_bench --include-dataset video_holmes \
  --max-transitions-per-example 256 \
  --transition-output-jsonl dataset_clip_wrapper/output/sft_cold_start/cg_vh_recommended_l1_builder_transitions.jsonl \
  --sft-output-jsonl dataset_clip_wrapper/output/sft_cold_start/cg_vh_recommended_l1_builder_sft.jsonl \
  --quality-report-output dataset_clip_wrapper/output/sft_cold_start/cg_vh_recommended_l1_builder_report.json

python -m dataset_clip_wrapper.export_l1_patch_sft \
  --stage-root ../video_skills_relaunched/dataset_clip_wrapper/output/batch3_latest_trace_repair_api \
  --transition-output-jsonl dataset_clip_wrapper/output/sft_cold_start/batch3_l1_patch_transitions.jsonl \
  --sft-output-jsonl dataset_clip_wrapper/output/sft_cold_start/batch3_l1_patch_sft.jsonl \
  --quality-report-output dataset_clip_wrapper/output/sft_cold_start/batch3_l1_patch_sft_report.json

python -m dataset_clip_wrapper.export_verifier_sft \
  --stage-root ../video_skills_relaunched/dataset_clip_wrapper/output/batch3_p5_audit_guided_repair_stages \
  --expert-demos ../video_skills_relaunched/dataset_clip_wrapper/output/expert_demos/batch3_p5_video_only_expert_demos_compact.jsonl \
  --balance-decisions \
  --transition-output-jsonl dataset_clip_wrapper/output/sft_cold_start/batch3_p5_verifier_gated_balanced_transitions.jsonl \
  --sft-output-jsonl dataset_clip_wrapper/output/sft_cold_start/batch3_p5_verifier_gated_balanced_sft.jsonl \
  --quality-report-output dataset_clip_wrapper/output/sft_cold_start/batch3_p5_verifier_gated_balanced_report.json

python -m dataset_clip_wrapper.export_motif_sft \
  --motif-bank motif/output/batch3_strict_qwen_motif_bank.jsonl \
  --transition-output-jsonl dataset_clip_wrapper/output/sft_cold_start/batch3_motif_gated_transitions.jsonl \
  --sft-output-jsonl dataset_clip_wrapper/output/sft_cold_start/batch3_motif_gated_sft.jsonl \
  --quality-report-output dataset_clip_wrapper/output/sft_cold_start/batch3_motif_gated_report.json

python -m dataset_clip_wrapper.export_l2_retrieval_sft \
  --rollout-jsonl dataset_clip_wrapper/output/pilot_corrected_v2_20260710/cg_bench/start_0_limit_1/examples.jsonl \
  --repair-results dataset_clip_wrapper/output/pilot_corrected_v2_20260710/cg_bench/start_0_limit_1/repair_results.jsonl \
  --transition-output-jsonl dataset_clip_wrapper/output/sft_cold_start/cg_l2_retrieval_transitions.jsonl \
  --sft-output-jsonl dataset_clip_wrapper/output/sft_cold_start/cg_l2_retrieval_sft.jsonl \
  --quality-report-output dataset_clip_wrapper/output/sft_cold_start/cg_l2_retrieval_sft_report.json
```

The existing L2/repair round export remains available through
`python -m dataset_clip_wrapper.export_stepwise_sft`.

## Raw Pilot Generation

For long-video `video_only` data, do not use the sequential-prefix fallback as
the positive teacher trajectory. Build a question-blind coarse L1 summary index,
then expose the question to the L2 controller and record its query-time
retrieval/inspection actions. The local Qwen worker enables this with
`QUERY_TIME_RETRIEVAL=1`.

The pilot additionally enables `LLM_COARSE_SELECTOR=1`: GPT-OSS receives the
question and visible coarse-summary catalog, then emits the atomic
`selected_coarse_indices` action. Lexical retrieval is retained only as a
recorded fallback. This avoids option-token overlap turning unrelated clips
into positive retrieval demonstrations.

Use at least four sampled frames per clip for action, state-change, verifier,
and motif data. The one-frame runner default is suitable only for static-object
smoke tests. The pilot worker therefore defaults to:

```text
CLIP_FRAMES=4
CLIP_MAX_TOKENS=1600
QUERY_TIME_RETRIEVAL=1
LLM_COARSE_SELECTOR=1
```

Hidden clue intervals and gold labels may be used only for offline evaluation,
filtering, and reward labels. They must not appear in `state_t` or the chat
prompt.

## Bias and Training Use

These are cold-start pseudo-gold records, not an unbiased policy dataset.

- L1 builder rows imitate successful teacher actions; failed calls are retained
  only as observations before the next successful action. Add deliberate retry,
  reject, and alternative valid-action data before policy optimization.
- Use a per-example skill-balanced cap for L1 SFT so one dense video cannot
  dominate the controller loss. The exporter also drops duplicate example IDs;
  put corrected/new rollouts before historical files so they take precedence.
- L1 patch rows are positive, repair-triggered actions. Add valid no-op, reject,
  and link-versus-skip transitions before broad repair-policy training.
- The full verifier set should be retained for auditing. Use the deterministic
  `--balance-decisions` subset for initial SFT, then calibrate on a held-out
  validation split without changing the runtime gate.
- `accepted_weak` is an insufficient/repair target, never a positive verifier
  target. Only `accepted_strong` and `resolved_strong` become `supported`.
- A strong/correct L2 retrieval trace with weak option-level answerability is
  exported only when its downstream repair report is `resolved_strong`.
- Motif rows with failed verification, invalid evidence, or hidden-leakage
  markers are exported as `rejected`. Promotion beyond candidate/shadow still
  requires transfer-test outcomes and false-binding examples.
- Split by video or example before SFT. Never split individual transitions from
  the same trajectory across train and validation.
- Do not include VRBench in SFT while its local manifest is evaluation-only.
  Use `--include-dataset`/`--exclude-dataset` on L1 exports and retain the
  dataset filter in the quality report.

Every export report includes `prompt_forbidden_key_hits`; a non-zero value is a
hard failure. Dataset coverage, class balance, duplicate transition ids, empty
evidence catalogs, and source split membership should also be checked before
training.

### Gated pilot package

After collecting a snapshot, build the training-ready pilot package with:

```bash
bash scripts/sft_pilot/prepare_sft_v2.sh
```

The strict builder excludes the evaluation-only `VRBench`, `VideoMME`, and
`OVO-Bench` sources, resolves distinct rows that collide on a transition id,
keeps all currently gated L2 retrieval examples, caps serialized chat length,
and writes SHA-256 hashes to `training_manifest.json`. It fails rather than
writing a silently incomplete training package when a hard gate is violated.

## 2026-07-10 Pilot Audit

Do not train from the original 5-dataset L1 export or the original 27-row
verifier export. The former includes VRBench eval trajectories; the latter
contains weak-positive and unresolved-evidence rows.

Current gated historical artifacts under `dataset_clip_wrapper/output/sft_cold_start/`:

- `cg_vh_train_l1_builder_sft.jsonl`: 1,535 CG-Bench/Video-Holmes atomic L1 actions.
- `cg_vh_recommended_l1_builder_sft.jsonl`: 1,263 deduplicated, per-video
  skill-capped actions; use this version for initial training.
- `batch3_p5_verifier_gated_full_sft.jsonl`: 18 verifier actions (5 supported,
  13 insufficient); 9 unresolved-evidence rows were excluded.
- `batch3_p5_verifier_gated_balanced_sft.jsonl`: 10 actions, balanced 5/5.
- `batch3_motif_gated_sft.jsonl`: 20 lifecycle actions (3 candidate, 3 shadow,
  14 rejected).
- `cg_vh_pilot_recommended_l1_builder_sft.jsonl`: 1,416 atomic L1 actions from
  eight deduplicated CG-Bench/Video-Holmes videos, with corrected/new pilots
  taking precedence over historical duplicates and a 256-action per-video cap.
- `cg_vh_pilot_gated_l2_retrieval_sft.jsonl`: 2 coarse-retrieval actions that
  passed correctness plus direct-strong or downstream-`resolved_strong` gates.

The corrected first-video pilot under `pilot_corrected_v2_20260710/` produced
450 Video-Holmes L1 actions and, for CG-Bench, 73 L1 builder actions, 5 L1 patch
actions, 1 L2 coarse-retrieval action, and 7 option-verifier actions. The CG
retrieval action is labeled positive only after downstream
`resolved_strong` repair. All listed export reports have zero forbidden prompt
keys.

The four-video expansion under `pilot_expand_20260710/` produced high-grade L1
graphs for all four videos. The GPT-OSS coarse selector found the relevant
coarse region in all three audited CG-Bench videos, while the lexical selector
missed it in all three. Raw L2 was not reliable enough to use as pseudo-gold:
three of the four expansion answers were wrong despite structurally strong
traces. Post-answer option repair resolved the one initially correct weak trace
(`cg_bench:17`) and abstained with `needs_more_evidence` on the three incorrect
traces instead of converting them into positives.

Scaling verdict after this pilot:

- **Go** for additional L1 builder, GPT coarse-selection, verifier-negative,
  and abstention data, subject to the same offline correctness and leakage
  gates.
- **No-go** for treating raw L2 `accepted_strong` as positive supervision or
  launching a large unreviewed L2 batch. A direct correctness gate and
  option-level verifier/repair gate are mandatory.
- Before scaling short-video L2 positives, add question-conditioned visual
  reinspection when existing L1 evidence cannot discriminate the options.

## Full Dataset Collection Protocol

The full-data path should keep the same MDP boundary as the pilot. It should
not train a model to map `video -> L1/L2 graph` in one shot. Each exported row
must supervise one controller action from a visible state:

```text
L1:    state_t = partial L1 + visible clip/window schema + budget
       action_t = inspect/select/create-node/create-edge/skip/patch

L2:    state_t = question + cached L1 summary + partial L2 graph + budget
       action_t = select evidence / extract claim / assign role / compose chain

repair: state_t = verifier gap + cached L1 + selected clips + repair budget
        action_t = diagnose / inspect / patch L1 / verify option / abstain

verifier: state_t = candidate claim + proposed refs + visible evidence text
          action_t = supported or insufficient, with failure code

motif: state_t = candidate reusable template + support and gate signals
       action_t = candidate, shadow, or rejected lifecycle decision
```

Use the three local datasets with benchmark-clean split roles:

| Dataset | Training use | Held-out use |
| --- | --- | --- |
| CG-Bench | video-level train/dev split over the 1,219 source videos; clue clips are hidden supervision only | frozen video-level test split |
| Video-Holmes | preserve official train for SFT/GRPO, carve an internal dev split by video | official test is final held-out |
| VRBench | no SFT, GRPO, motif mining, or reward tuning while local data is eval-only | OOD stress test only |

Within the train videos, assign disjoint video-level roles:

- `sft_seed`: first cold-start behavior cloning data.
- `grpo_pool`: rollout prompts for policy optimization, unseen during SFT.
- `dev_tune`: threshold, reward-weight, and early-stopping decisions.

Do not split transitions from the same video or trajectory across roles. For
CG-Bench, the source video is the split key; clue clips and clue intervals are
evaluation/filtering labels and must never appear in `state_t`. For
Video-Holmes, the official split is authoritative. For VRBench, any internal
training split must be labeled as a transductive ablation rather than the
standard benchmark result.

Collection should be two-tier:

1. **Reusable L1 substrate.** Build question-blind clip schemas and L1 evidence
   caches once per unique source video and config hash. Reuse that cache for
   all questions on the same video. This is the only way the full CG-Bench,
   Video-Holmes, and VRBench scale is realistic on four H100/A100-class GPUs.
2. **Question-conditioned controller traces.** Feed `question + cached L1` to
   L2, verifier, repair, and motif curation. Export only gated actions:
   correct and verifier-supported L2 positives, repair-resolved positives,
   verifier negatives, abstentions, and motif lifecycle decisions that preserve
   the non-executable motif boundary.

The current per-question staged pilot is an audit-quality teacher, not the
final full-data engine. It is useful for high-quality L1, GPT coarse-selection,
verifier, repair, and abstention examples. Before running all three datasets at
scale, add a video-level cache and a split-manifest role audit so repeated
questions do not recompute the same L1 graph and train/test roles cannot leak.

For a cold-start SFT mixture, start with a conservative batch:

```text
35% L1 builder / L1 patch
35% L2 retrieval and L2/repair controller
20% verifier, including hard negatives and abstentions
10% motif lifecycle and bounded repair-abstain traces
```

Train the L2/repair actor first against a frozen cached L1 substrate. Defer
joint L1+L2 optimization until after the verifier-gated L2 policy is stable.
The first GRPO run should optimize L2/repair actions with deterministic
verifier rewards, valid-action rewards, evidence-support rewards, abstention
rewards, leakage penalties, and cost/budget penalties. The learned verifier is
only an auxiliary calibration model; the deterministic runtime verifier remains
the hard gate for collection and evaluation.

## Teacher Routing And Cost Control

Use a routed teacher stack instead of replacing the full pipeline with a paid
model. The default bulk teacher is:

```text
clip perception: local Qwen/Qwen3.5-9B
bulk graph and controller teacher: openai/gpt-oss-120b:free
hard-case escalation: small paid teacher such as gpt-5-mini, only after gates
```

The local 9B model handles clip schemas and video perception through the local
OpenAI-compatible server started by `scripts/sft_pilot/run_local_qwen_worker.sh`.
It costs GPU walltime, not API spend. `openai/gpt-oss-120b:free` should handle
bulk L1 composition, coarse selection, L2 retrieval traces, verifier drafts,
and motif curation because it is free and strong enough for broad pseudo-gold
generation when paired with deterministic gates.

Do not use a paid teacher for every example. Escalate only when a cheap/free
trace is useful but uncertain:

- `accepted_weak`, `needs_more_evidence`, malformed JSON, or verifier
  disagreement.
- A correct coarse selection with weak or incorrect option-level answerability.
- A repair case where the gap diagnosis is clear but the option evidence is
  under-supported.
- A small audit sample, for example 5-10 paid reviews per 100 free traces, to
  estimate teacher noise and decide whether routing rules need tightening.

Escalated rows still must pass the same runtime verifier, correctness,
split-role, hidden-leakage, and prompt-forbidden-key checks before entering
SFT. Paid output is evidence for filtering and repair, not a bypass around the
MDP transition format or deterministic gate. In reports, keep the teacher model
and escalation reason in metadata so cost and quality can be audited later.

Current snapshot collection command:

```bash
python -m dataset_clip_wrapper.collect_sft_snapshot \
  --output-dir dataset_clip_wrapper/output/sft_cold_start/collection_20260710_current \
  --pilot-root dataset_clip_wrapper/output/pilot_20260710_free \
  --pilot-root dataset_clip_wrapper/output/pilot_20260710 \
  --pilot-root dataset_clip_wrapper/output/pilot_expand_20260710 \
  --pilot-root dataset_clip_wrapper/output/pilot_corrected_v2_20260710 \
  --extra-rollout-jsonl ../video_skills_relaunched/dataset_clip_wrapper/output/batch3_cg_strict_qwen.jsonl \
  --extra-rollout-jsonl ../video_skills_relaunched/dataset_clip_wrapper/output/batch3_video_holmes_strict_qwen.jsonl \
  --expert-demos ../video_skills_relaunched/dataset_clip_wrapper/output/expert_demos/batch3_p5_video_only_expert_demos_compact.jsonl \
  --motif-bank motif/output/batch3_strict_qwen_motif_bank.jsonl \
  --balance-verifier
```

This command is intentionally snapshot-based. If a pilot job is still running,
rerun the same command after more `examples.jsonl` rows or `repair_stages`
complete. It writes one JSONL and one report per controller plus a top-level
`snapshot_report.json`.
