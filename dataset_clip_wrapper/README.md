# Dataset Clip Wrapper

Wrap the core and streaming video benchmarks into the canonical
`CanonicalVideoExample` schema with clip segmentation for **short**, **long**,
and **streaming** regimes.

Supported datasets:

Core clue/video reasoning benchmarks:

- `video_holmes`
- `cg_bench`
- `vrbench`
- `siv_bench`

Streaming video benchmarks:

- `ovo_bench` — StreamBridge/OVO-Bench-style realtime QA records with `realtime`
  anchors; local coding uses `datasets/streambridge_tiny/tiny_ovo_bench_50videos.json`.
- `videomme` — StreamBridge/VideoMME-style whole-video QA records; local coding
  uses `datasets/streambridge_tiny/tiny_videomme.json`.

The local StreamBridge files are smoke-test data for adapter and pipeline
validation, not accuracy reporting. Full OVO-Bench or VideoMME can reuse the
same adapter format by supplying matching annotation/video layouts.

## Pipeline

```text
dataset adapter
  -> probe duration
  -> clip_policy segmentation (short / long / streaming)
  -> optional perception backbone captions per clip
  -> evidence_candidates + evidence_index clip graph
  -> canonical JSON example
```

## Two-layer graph export

Each canonical example now includes:

- `metadata.clue_memory_graph` — Layer 1 (`ClueMemoryGraph`, question-blind)
- `metadata.reasoning_rollout_shell` — Layer 2 shell linked via `clue_memory_ref`

See [two-layer graph schema](../docs/two-layer-graph-schema.md).

```bash
python dataset_clip_wrapper/smoke_test_two_layer_schema.py
```

## Hyperparameters

### Video regime

| Regime | Default policy |
|--------|----------------|
| `short` | `whole_video` + 4s windows |
| `long` | `hierarchical` 30s coarse index + retrieve-gated 8s fine |
| `streaming` | `fixed_window` with `online=true` and `observation_end_s` |

Override with CLI flags: `--clip-strategy`, `--window-s`, `--overlap-s`,
`--coarse-window-s`, `--fine-window-s`, `--observation-end-s`,
`--index-fine-expansion`, `--retrieval-topk`, `--retrieval-mode`, `--no-retrieval`.

Long-video flow (M3-style):

```text
coarse index (30s windows, all clips in evidence_index)
  -> baseline video_only: question-blind sequential coarse neighborhoods
  -> optional query-time retrieval: visible question + time anchors select coarse neighborhoods
  -> fine windows (8s) only inside retrieved coarse parents
  -> clip-schema backend + graph compose on perception clips only
```

The default `video_only` L1 builder stays question-blind so the perception graph
can be audited independently from answer-time retrieval. For QA experiments,
enable query-time memory with `--query-time-retrieval`; this uses only the
visible question, never hidden clue intervals or official answers. Timestamp
mentions such as `2:04` automatically expand the matching coarse window and its
neighbors unless `--no-time-anchor-expansion` is set.

### Backbone

| `--backbone` | Behavior |
|--------------|----------|
| `annotation_only` | No model calls; dataset annotations + clip spans only |
| `openrouter` | Caption each clip with `--backbone-model` via OpenRouter |

Other backbone inputs:

- `--backbone-model` (default `openai/gpt-5-mini`)
- `--keys-py` (default workspace `keys.py`)
- `--backbone-max-clips`
- `--backbone-request-frames`
- `--run-backbone`

## LLM Two-Stage Pipeline

Stage 1 turns each segmented clip into a structured clip schema. The default
backend is a multimodal OpenRouter model (`qwen/qwen3.5-9b`, closest available
Qwen3.5 ~8B-class VLM on OpenRouter). For offline raw-video smoke tests and
hard `video_only` cases, `--clip-schema-backend video_tools` uses local video
tools instead.

The clip-schema prompt is clue-oriented, not caption-only. It asks for grounded
observations, dialogue, entity mentions, salient objects, place descriptions,
cross-clip cues, searchable phrases, and uncertainty.

Stage 2 uses `openai/gpt-oss-120b` as a neighbor-local VLM/LLM-first L1 graph
composer by default. Instead of asking GPT-OSS to emit one huge graph JSON, the
pipeline gives it one target clip digest plus a small neighbor context. GPT-OSS
emits target clip nodes and sparse semantic edges such as `reappears`,
`same_object`, `state_change`, and `social_cue`. Local code only normalizes IDs,
timestamps, media references, and validates edge endpoints. Heuristic graph
construction is kept as `deterministic` debug/fallback mode, not the primary
quality path.

```text
segment clips (short / long / streaming)
  -> [long] baseline or query-time coarse selection
  -> [long] expand fine windows inside candidates only
  -> clip-schema producer (Qwen or local video_tools)
  -> gpt-oss-120B target-clip + neighbor VLM L1 graph composer
  -> canonical example with evidence_index graph
```

```bash
# Full pipeline (requires keys.py or OPENROUTER_API_KEY)
python -m dataset_clip_wrapper.run_llm_pipeline \
  --dataset video_holmes \
  --regime short \
  --limit 1 \
  --clip-schema-max-clips 2 \
  --output dataset_clip_wrapper/output/video_holmes_llm.jsonl

# Long-video CG-Bench
python -m dataset_clip_wrapper.run_llm_pipeline \
  --dataset cg_bench \
  --regime long \
  --limit 1 \
  --clip-schema-max-clips 2

# Long-video QA/query-memory mode: visible question selects fine neighborhoods
python -m dataset_clip_wrapper.run_llm_pipeline \
  --dataset vrbench \
  --regime long \
  --mode video_only \
  --query-time-retrieval \
  --retrieval-topk 4 \
  --limit 1

# Streaming video benchmark smoke: OVO-Bench-style realtime QA
python -m dataset_clip_wrapper.run_llm_pipeline \
  --dataset ovo_bench \
  --regime streaming \
  --mode video_only \
  --clip-schema-backend video_tools \
  --clip-schema-max-clips 2 \
  --limit 1

# Streaming video benchmark smoke: VideoMME-style whole-video QA
python -m dataset_clip_wrapper.run_llm_pipeline \
  --dataset videomme \
  --regime short \
  --mode video_only \
  --clip-schema-backend video_tools \
  --clip-schema-max-clips 2 \
  --limit 1

# Staged/resumable runner: writes intermediate clip schemas, L1, and L2 files
python -m dataset_clip_wrapper.run_staged_llm_pipeline \
  --dataset vrbench \
  --regime long \
  --mode video_only \
  --query-time-retrieval \
  --retrieval-topk 2 \
  --clip-schema-frames 1 \
  --clip-schema-timeout-s 45 \
  --disable-llm-skills \
  --disable-vlm-skills \
  --stage-dir dataset_clip_wrapper/output/staged_vrbench \
  --output dataset_clip_wrapper/output/vrbench_staged.jsonl

# Deterministic graph compose only (no gpt-oss planner call)
python -m dataset_clip_wrapper.run_llm_pipeline \
  --dataset video_holmes \
  --graph-deterministic \
  --skip-clip-schema \
  --limit 1
```

Hyperparameters:

| Flag | Default | Role |
|------|---------|------|
| `--clip-schema-backend` | `qwen` | `qwen` for OpenRouter VLM, `video_tools` for local raw-video tools |
| `--clip-schema-model` | `qwen/qwen3.5-9b` | multimodal clip-schema producer |
| `--clip-schema-max-clips` | `3` | cap clip-schema calls per example |
| `--graph-model` | `openai/gpt-oss-120b` | graph planner / composer |
| `--graph-composer-mode` | `neighbor_vlm_l1` | target-clip + neighbor local graph JSON; `vlm_l1` is global graph JSON; `skill_plan` is legacy atomic-skill planner; `deterministic` is debug |
| `--graph-timeout-s` | `180` | per OpenRouter graph/L2 request timeout |
| `--graph-neighbor-workers` | `1` | parallel target-clip workers for `neighbor_vlm_l1`; useful for long videos |
| `--no-coarse-summary-index` | off | disable full coarse Qwen summary indexing before long-video retrieval |
| `--graph-deterministic` | off | force deterministic debug composer and skip VLM/LLM graph composition |
| `--keys-py` | workspace `keys.py` | OpenRouter API key source |

Optional local raw-video perception backend:

```bash
python -m dataset_clip_wrapper.run_llm_pipeline \
  --dataset video_holmes \
  --clip-schema-backend video_tools \
  --graph-deterministic \
  --limit 1
```

This backend copies the useful Multi-hop perception-tool pattern into this repo
without depending on the full Multi-hop agent runtime. It samples frames,
computes frame-change signals, tries optional OCR when available, and emits the
same clip-schema fields consumed by the atomic graph composer.

Offline graph-compose smoke test:

```bash
python dataset_clip_wrapper/smoke_test_graph_compose.py
python dataset_clip_wrapper/smoke_test_neighbor_vlm_l1_graph_compose.py
python dataset_clip_wrapper/smoke_test_vlm_l1_graph_compose.py
python dataset_clip_wrapper/smoke_test_video_tools.py
python dataset_clip_wrapper/smoke_test_video_only_takein.py
python dataset_clip_wrapper/smoke_test_coarse_fine_graph_crafting.py
```

`smoke_test_video_only_takein.py` covers all supported current video datasets,
including the streaming video benchmarks (`ovo_bench`, `videomme`). It verifies
that each one can load a real video in `video_only`, sample frames through
`video_tools`, produce clip schemas, craft an `evidence_index` / clue-memory
graph, and avoid hidden-supervision leakage.

`smoke_test_coarse_fine_graph_crafting.py` checks the two-level graph contract.
For long videos, it builds a full-video coarse graph, retrieves a small set of
coarse neighborhoods, expands fine clips only inside those neighborhoods, and
links fine nodes back to parent coarse nodes with `refines` edges. For short
videos, it validates the fine graph directly.

Use this test when checking whether an entire video can be taken in for all
supported datasets. "Entire video" means full temporal coverage at the
appropriate level: short and fixed-window streaming datasets build the fine
graph over the full video, while long datasets build a full-video coarse graph
first and then craft a fine graph only inside retrieved coarse neighborhoods.
Do not interpret the test as full fine-grained VLM processing over every 8s
window of CG-Bench or VRBench.

The pipeline writes this reference layer to `metadata.coarse_fine_graph`:

- `coarse_graph`: full-video clip references for long videos, with
  `media_ref.path`, coarse `time_span` handles, and `coarse_summary` nodes.
- `fine_graph`: full-video fine references for short videos, or retrieved
  fine-neighborhood references for long videos.
- `coarse_to_fine_links`: `refines` edges from fine clips back to parent coarse
  clips.

For long-video `video_only` runs, the staged runner first builds a cached
full-video coarse summary index at `00b_coarse_clip_schemas.jsonl` unless
`--no-coarse-summary-index` is set. These Qwen summaries are visible runtime
perception, not supervision. They are used for query-time coarse retrieval and
are promoted into `evidence_index` / clue memory as `coarse_visual_summary`
context nodes so L2 can cite what neighborhood was searched.

Use the L1 query-memory diagnostic before trusting an L2 answer:

```bash
python dataset_clip_wrapper/evaluate_l1_query_memory.py \
  --topk 5 \
  --output dataset_clip_wrapper/output/l1_query_memory_eval.json \
  dataset_clip_wrapper/output/cg_bench_video_only_qwen_gptoss_entire.jsonl
```

The report separates graph quality from answerability:

- `l1_graph_quality`: semantic node/edge counts, schema coverage, invalid edge
  count, fallback use, and failed compose steps.
- `qa_answerability`: question hit count, option score margin, and an
  `answerable` / `weak` / `insufficient` grade for deciding whether to run L2.

It also surfaces selected coarse indices and a `l2_uses_gold_text_warning`
flag. In valid `video_only` L2 experiments, the planner must not see or copy
`question.answer`; official answers are evaluation-only hidden supervision.

The L2 gate is intentionally conservative. It rejects examples whose long-video
retrieval fell back to `uniform_probe_no_lexical_match`, and it rejects cases
where the top answer options are supported by essentially the same evidence
refs. This keeps "the graph has nodes" separate from "the evidence can
distinguish an answer."

For short videos with explicit timestamps in the visible question, the staged
runner performs a query-time anchor repass by default. It writes
`02b_anchor_clip_schemas.jsonl`, re-sampling only clips near the timestamp with
more frames, then merges those schemas back into L1. This is meant for cases
such as Video-Holmes where a key prop is visible only in a narrow moment. The
repass uses the visible question text but never the answer label or hidden
clues.

Short-video L1 also adds recurrence clue edges for repeated VLM-observed
objects or places, such as a fence/gate appearing earlier and again near a
timestamp. These `short_video_recurrence_linker` nodes/edges store the
cross-time clue explicitly so query memory and L2 can cite it.

The graph also stores an answerability diagnostic subgraph. For each visible
question, the pipeline adds `question_requirement`, `required_modality`, and
when needed `answerability_gap` nodes to `evidence_index` / clue memory. These
nodes are runtime-visible diagnostics, not answer supervision: they record that
the current video-only graph appears to be missing required evidence such as
dialogue, social intent, or causal motivation. Audio/ASR/subtitle recovery is
out of scope for this video-only setting, so dialogue gaps are flagged as
`out_of_scope_modalities` rather than offered as a repair path. This is
important for benchmarks like SIV-Bench, where a visually plausible graph may
still be unable to support questions about confidential information or
hesitation from visual evidence alone. The L1 gate rejects examples with missing
required modalities instead of letting ordinary L2 guess.

These gaps are typed. `answerability_diagnostic` includes `gap_category`,
`l2_repair_policy`, `allowed_repair_l2`, `out_of_scope_modalities`, and
`audio_repair_allowed=false`. For example, a SIV-style question may be marked
`visual_social_common_sense_gap_with_out_of_scope_dialogue` with
`visual_social_l2_may_attempt_weak_repair_no_audio`. That flag means ordinary
L2 answer commit is blocked. A future specialized L2 module may only try
visual social-intent or causal-motive verification; it must not use audio,
ASR, or subtitles as evidence in this scope.

### SIV-Bench under no-audio video-only

SIV-Bench should not be interpreted like the current primary five-dataset
video-only track (`video_holmes`, `videomme`, `ovo_bench`, `cg_bench`,
`vrbench`). It is valuable, but for a different purpose. Many SIV questions ask
why someone is hesitant, hiding information, protecting another person, or
avoiding misunderstanding. Those answers often depend on dialogue or social
context that is outside the current no-audio scope.

The expected behavior is therefore:

- build the L1 visual graph if video is available;
- mark missing `dialogue_or_asr`, `social_intent_or_affect`, or
  `causal_explanation` requirements explicitly;
- create a `commonsense_repair_pack` only as a low-trust bridge hypothesis;
- keep `audio_repair_allowed=false`;
- block final L2 commit unless non-diagnostic video evidence verifies the
  claim.

This makes SIV a stress test for answerability-gap detection and repair
protocols, not a benchmark where video-only L2 should be expected to answer
every item.

## Staged Outputs and Resume

`run_staged_llm_pipeline.py` is the recommended API runner for slow videos. It
creates one directory per example:

```text
00_shell.json
01_perception_spans.json
02_clip_schemas.jsonl
03_l1_inputs.json
04_l1_example.json
05_l2_rollout.json
final_example.json
```

The clip-schema file is appended after each clip, so interrupted Qwen runs can
resume without repeating completed clips. Re-run the same command to reuse
cached stages; pass `--force` to rebuild. If an API run produced malformed JSON,
timeouts, or empty clip schemas, pass `--retry-failed-clip-schemas` to keep the
good cached rows, discard only rows with `model_error`, and recompute those
clips before rebuilding L1/L2.

Clip-schema generation now uses three guards before a clip is allowed into L1:

- OpenRouter `json_schema` response format for the full and compact prompts;
- a compact `json_object` fallback when provider-side schema enforcement fails;
- local normalization / validation that rejects empty or type-broken payloads.

Use the retry flag when comparing graph quality after prompt or format fixes:

```bash
python -m dataset_clip_wrapper.run_staged_llm_pipeline \
  --dataset video_holmes \
  --regime short \
  --mode video_only \
  --query-time-retrieval \
  --clip-schema-frames 1 \
  --clip-schema-timeout-s 45 \
  --disable-llm-skills \
  --disable-vlm-skills \
  --stage-dir dataset_clip_wrapper/output/staged_fix_rerun_video_holmes \
  --output dataset_clip_wrapper/output/video_holmes_retry_failed.jsonl \
  --retry-failed-clip-schemas
```

For long videos, prefer acceleration by reducing work before increasing
parallelism:

- use `--benchmark-profile long_coarse_fine` for CG-Bench/VRBench full-coarse +
  retrieved-fine graph building;
- keep `index_fine_expansion=retrieval_gated`;
- keep the coarse summary index enabled so video-only retrieval has visual text
  to search; without coarse summaries the retriever falls back to uniform probe;
- use visible question + answer options and timestamp anchors to select fine clips;
- keep `--retrieval-topk` small for the first pass;
- use `--clip-schema-frames 1` and lower `--clip-schema-max-tokens`;
- run L1 gate passes with `--graph-deterministic --skip-l2-planner`;
- disable skill-level API calls with `--disable-llm-skills --disable-vlm-skills`;
- retry only failed or relevant clips from the staged cache.

Recommended long-video staged pass:

```bash
python -m dataset_clip_wrapper.run_staged_llm_pipeline \
  --dataset cg_bench \
  --benchmark-profile long_coarse_fine \
  --mode video_only \
  --limit 1 \
  --clip-schema-model qwen/qwen3.5-9b \
  --clip-schema-frames 1 \
  --clip-schema-workers 8 \
  --clip-schema-max-tokens 600 \
  --retry-failed-clip-schemas \
  --graph-model openai/gpt-oss-120b \
  --graph-neighbor-workers 8 \
  --skill-model openai/gpt-oss-120b \
  --llm-skill-scope verifier \
  --stage-dir dataset_clip_wrapper/output/staged_long_coarse_fine_cg \
  --output dataset_clip_wrapper/output/long_coarse_fine_cg.jsonl
```

## L1 Gate Before GPT-OSS L2

For batch experiments, do not run GPT-OSS L2 on every video immediately. First
build a deterministic L1/query-memory candidate:

```bash
python -m dataset_clip_wrapper.run_staged_llm_pipeline \
  --dataset vrbench \
  --regime long \
  --mode video_only \
  --query-time-retrieval \
  --retrieval-topk 2 \
  --graph-deterministic \
  --skip-l2-planner \
  --disable-llm-skills \
  --disable-vlm-skills \
  --stage-dir dataset_clip_wrapper/output/staged_fix_rerun_vr \
  --output dataset_clip_wrapper/output/vrbench_video_only_qwen_l1_gate.jsonl \
  --rebuild-from-stages \
  --no-fill-missing-clip-schemas
```

Then gate the L1 graph without using hidden answers:

```bash
python -m dataset_clip_wrapper.gate_l1_for_l2 \
  --topk 5 \
  --min-option-margin 0.2 \
  --output dataset_clip_wrapper/output/l1_gate_report.json \
  --passed-output dataset_clip_wrapper/output/l1_gate_passed_example_ids.txt \
  dataset_clip_wrapper/output/*_l1_gate.jsonl
```

Only examples that pass this gate should spend GPT-OSS L2 calls. The gate uses
visible-question retrieval, graph size, question-hit score, option-score margin,
and hidden-supervision checks. Gold labels remain evaluation-only fields in the
report and are not used to decide whether L2 runs.

The same locality principle should be used for L2 when needed: do not give
GPT-OSS the full L1 graph as one large prompt. Build a compact evidence pack
from gated L1 refs, option scores, snippets, and uncertainty, then ask L2 to
reason over that pack only. This keeps L2 grounded in retrieved evidence and
avoids a second large-JSON failure mode.

### Long-video retrieval repair

Long-video failures must separate L1 graph structure from L1 target coverage.
A graph can have high node/edge density while still missing the exact clip
needed by the question. CG-Bench sample `cg_bench:14` showed this failure mode:
the local repair pass produced many visual nodes, but those nodes described a
Christmas gift scene and repeatedly said that no animated vehicle was visible.
That is an `l1_target_coverage_failure`, not an L2 reasoning failure.

VRBench can fail differently. The graph may cover the right visual context,
such as ruins, documents, cameras, and group attention, but the answer asks for
motivation or social/causal intent. With audio/subtitle excluded, such examples
should be marked `l1_context_partial_l2_bridge_needed` or
`visual_only_benchmark_limitation` unless the visual evidence strongly anchors
the bridge. When the visible anchors identify an objective situation and a
stable background fact disambiguates the answer, the repair runner may return
`accepted_bridge`. That status is intentionally separate from
`resolved_strong`: it says the answer is supported by visual anchors plus
objective background knowledge, not by direct visual evidence alone.

Use the repair runner for these cases:

```bash
python dataset_clip_wrapper/run_repair_protocol.py \
  --quality-report dataset_clip_wrapper/output/rerun5_quality_report.json \
  --stage-dir dataset_clip_wrapper/output/repair_long_reroute \
  --output dataset_clip_wrapper/output/repair_long_reroute_report.json \
  --datasets cg_bench vrbench \
  --repair-mode reroute \
  --keys-py /fs/gamma-projects/vlm-robot/keys.py \
  --clip-schema-model qwen/qwen3.5-9b \
  --verifier-model openai/gpt-oss-120b
```

`--repair-mode local` expands around previously selected coarse windows.
`--repair-mode reroute` asks GPT-OSS to build a `clue_need_spec` first, then
uses that spec to select coarse windows from the full coarse summary index. The
model-planned spec records:

- visual target and event/action to find;
- visual attributes to resolve;
- positive evidence criteria;
- negative evidence to exclude;
- objective background facts that may be used only by L2 bridge verification;
- bridge evidence criteria for selecting visual context windows;
- forbidden modalities (`audio`, `asr`, `subtitle`, `dialogue`);
- clip inspection instructions for Qwen.

GPT-OSS also acts as the default coarse-window selector in API runs. Lexical
query variants are only a fallback for dry-run/no-api mode or when
`--disable-llm-reroute-selector` is set. This keeps the normal repair path
prompt-driven: the mechanism asks the model what clue must be found and where
to inspect, instead of relying on hand-written retrieval heuristics. The
selector can choose windows in `direct_visual` mode or `bridge_context` mode.
It should abstain only when the coarse summaries contain neither direct visual
evidence nor useful visual anchors for an objective bridge. Because coarse
summaries are lossy, an abstaining selector is retried once in
`exploratory_probe` mode: GPT-OSS must choose a small set of plausible windows
from the full coarse index for Qwen inspection, without using lexical fallback
or hidden labels.

`--repair-mode auto` switches to full reroute when cached repair schemas contain
negative target evidence, such as "no vehicle", "no animation", or "cannot
determine". The output report records `failure_type`,
`negative_coarse_indices`, `selected_coarse_indices`, and
`retrieval_round_count`.

This is inspired by M3-style iterative retrieval/control, but it is not a
multi-agent voting setup. The clue planner and coarse selector only produce
evidence-seeking instructions and candidate evidence packs. The final L2 answer
first tries `verify_claim_support` for `resolved_strong`. If that fails, a
strict objective-background bridge verifier can return `accepted_bridge` only
when real visual refs anchor the situation and stable background facts
disambiguate one option. Commonsense bridge text, negative evidence, and
background facts cannot become L1 evidence nodes by themselves.

The practical status levels are:

- `resolved_strong`: direct visual evidence refs pass `verify_claim_support`.
- `accepted_bridge`: visual anchors plus objective background facts support one
  option; report includes `not_direct_visual_evidence=true`.
- `needs_more_evidence`: neither direct evidence nor bridge evidence is enough.
- `visual_only_benchmark_limitation`: the missing clue appears outside the
  video-only scope, such as audio/subtitle/hidden context.

Repair clip schemas can be parallelized with process workers:

```bash
python dataset_clip_wrapper/run_repair_protocol.py \
  --quality-report dataset_clip_wrapper/output/rerun5_quality_report.json \
  --stage-dir dataset_clip_wrapper/output/repair_long_objective_bridge_api \
  --output dataset_clip_wrapper/output/repair_long_objective_bridge_api_report.json \
  --datasets cg_bench vrbench \
  --repair-mode reroute \
  --repair-clip-schema-workers 4 \
  --keys-py /fs/gamma-projects/vlm-robot/keys.py
```

The process-worker path keeps per-clip OpenRouter total timeouts effective and
checkpoints partial `repair_02_clip_schemas.jsonl` results for resume.

To merge the base five-dataset L1/L2 quality report with long-video repair
reports:

```bash
python dataset_clip_wrapper/report_final_acceptance.py \
  --quality-report dataset_clip_wrapper/output/rerun5_quality_report_strict_qwen.json \
  --repair-report dataset_clip_wrapper/output/repair_long_objective_bridge_api_v7_cg_rebuild_report.json \
  --repair-report dataset_clip_wrapper/output/repair_long_objective_bridge_api_v7_vr_rebuild_report.json \
  --output dataset_clip_wrapper/output/rerun5_final_acceptance_report.json
```

The current strict one-video-per-dataset API check reports:

- `high_l1_all=true`
- `accepted_all=true`
- `strict_vlm_perception_all=true`
- `fallback_clip_schema_total=0`
- `model_error_clip_schema_total=0`
- final L2 status: four `accepted_strong`, one `accepted_bridge`

The strict report also records prompt/output budget and cache statistics for
clip-schema and graph-compose calls (`prompt_chars`, approximate tokens,
`output_chars`, malformed JSON, timeout, compact retry, cache hit/miss counts).

If cached local `video_tools` clip schemas remain, or if Qwen returned
`model_error` rows, rerun the staged pipeline with both retry flags:

```bash
python -m dataset_clip_wrapper.run_staged_llm_pipeline \
  --dataset videomme \
  --benchmark-profile short_multi_hop \
  --mode video_only \
  --rebuild-from-stages \
  --retry-non-backbone-clip-schemas \
  --retry-failed-clip-schemas \
  --clip-schema-backend qwen \
  --clip-schema-workers 8
```

Use the same pattern for OVO/CG/VR cached stages when strict Qwen-only
perception is required. If a failed clip schema shows `finish_reason=length`,
raise `--clip-schema-max-tokens` for the failed-only retry instead of accepting
a fallback schema.

Staged runs also cache neighbor-local GPT-OSS graph composition per clip in
`03_neighbor_vlm_l1_clip_results.jsonl`. If a long short-video/streaming run is
interrupted during graph compose, rerun with `--rebuild-from-stages`; cached
clip-level graph outputs are reused and only missing clips are sent back to
GPT-OSS. This keeps long prompts and long outputs bounded to one local
clip-neighborhood at a time.

Observed strict resume checks:

- VideoMME strict rerun resumed from 76 cached GPT-OSS neighbor results and
  completed the remaining 39 (`neighbor_cache_hits=76`, misses `39`).
- OVO strict rerun resumed from 93 cached neighbor results and completed the
  remaining 22 (`neighbor_cache_hits=93`, misses `22`).
- CG strict rerun reused 36 cached neighbor results and only recomposed 4
  missing clip-neighborhoods.

When rerunning from cached stages:

- `--rebuild-from-stages` ignores `final_example.json` and rebuilds L1/L2 from
  cached intermediate files.
- `--no-fill-missing-clip-schemas` prevents the gate pass from calling Qwen for
  missing clips; useful after interrupted retries.
- `--retry-failed-clip-schemas` should be reserved for perception-quality repair,
  because it will call Qwen for failed clips.
- JSONL outputs are append-only unless `--force` is used or the output file is
  removed. For final strict reports, keep one row per dataset output file to
  avoid mixing stale and fresh examples.

VRBench video-only graph-quality probe:

```bash
python dataset_clip_wrapper/evaluate_vrbench_video_only_graph.py \
  --limit 1 \
  --clip-schema-max-clips 4 \
  --retrieval-topk 4
```

This compares video-only discovered clip-schema / clue-memory nodes against
hidden VRBench `reasoning_process` timestamps for evaluation only. A narrow
sequential budget is expected to miss later long-video targets; increasing
`--retrieval-topk` and `--clip-schema-max-clips` checks whether the graph path
can cover the target once perception reaches the right temporal neighborhood.

## CLI

```bash
cd /fs/gamma-projects/vlm-robot/video_skills_relaunched

# Video-Holmes short regime
python -m dataset_clip_wrapper.cli \
  --dataset video_holmes \
  --regime short \
  --limit 5 \
  --output dataset_clip_wrapper/output/video_holmes_short.jsonl

# CG-Bench long hierarchical
python -m dataset_clip_wrapper.cli \
  --dataset cg_bench \
  --regime long \
  --limit 3 \
  --output dataset_clip_wrapper/output/cg_bench_long.jsonl

# Streaming visibility
python -m dataset_clip_wrapper.cli \
  --dataset video_holmes \
  --regime streaming \
  --observation-end-s 30 \
  --mode video_only \
  --limit 1

# With VLM backbone (requires keys.py or OPENROUTER_API_KEY)
python -m dataset_clip_wrapper.cli \
  --dataset siv_bench \
  --regime short \
  --run-backbone \
  --backbone openrouter \
  --backbone-model openai/gpt-5-mini \
  --backbone-max-clips 2 \
  --limit 1
```

## Smoke Test

```bash
python dataset_clip_wrapper/smoke_test.py
```

## Python API

```python
from dataset_clip_wrapper import WrapperConfig, VideoRegime, BackboneConfig
from dataset_clip_wrapper.pipeline import iter_canonical_examples

config = WrapperConfig(
    dataset_root="/fs/gamma-projects/vlm-robot/datasets",
    dataset="video_holmes",
    regime=VideoRegime.SHORT,
    limit=10,
    backbone=BackboneConfig(name="annotation_only"),
)
for example in iter_canonical_examples(config):
    ...
```

## Output Schema

Each JSONL row matches `schemas/canonical_video_example.schema.json` with:

- `video.derived_clips` — logical clip windows with `source_span`
- `video.segments` — dataset annotations and subtitle spans
- `evidence_candidates` — clips, annotations, optional backbone captions
- `evidence_index` — clip graph nodes/edges + `clip_policy` + `backbone` metadata

For atomic-skill execution, convert a canonical row into the runtime graph shape:

```python
from dataset_clip_wrapper import canonical_example_to_skill_graph

graph = canonical_example_to_skill_graph(example)
```

The bridge preserves clip grounding, source ids, trust/discovery metadata, and
the `expert_demo` / `video_only` hidden-supervision boundary.

See [clip-processing-policy.md](../docs/clip-processing-policy.md) for regime defaults.
