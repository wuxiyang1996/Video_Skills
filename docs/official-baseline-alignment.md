# Official Baseline Alignment

This document separates exact upstream reproduction from adaptations to this
repository's unified schema. A dataset reader, compatible annotation format, or
call into an upstream quick-start API is not by itself an official baseline.

## Source of truth

- Dispider: <https://github.com/Mark12Ding/Dispider>, checked out at
  `/mnt/is_data/xwu/video_skills/code/Dispider`.
- M3-Agent: <https://github.com/ByteDance-Seed/m3-agent>, checked out at
  `/mnt/is_data/xwu/video_skills/code/m3-agent`.
- StreamBridge: <https://github.com/apple/ml-streambridge>, checked out at
  `/mnt/is_data/xwu/video_skills/code/ml-streambridge`.

Record the upstream commit, model identifier, dataset revision, command, and
all deviations in every reported run. Run:

```bash
python3 -m baseline.check_official_baselines
```

before submitting jobs.

## Dispider

### Official settings

The released VideoMME path is
`dispider/eval/model_videomme_long.py`, launched by
`scripts/eval/videomme.sh` with:

- `conv_mode=qwen`
- 16 frames per clip and at most 32 clips
- `temperature=0`, `num_beams=1`
- `max_new_tokens=256`
- the upstream multiple-choice instruction and `"The best answer is:"`
  assistant prefix

The quick-start `inference.py` has a different protocol: up to 100 clips,
`max_new_tokens=1024`, and a question inserted at position zero. It is useful
for a streaming-prefix adaptation, but it is not the released VideoMME
benchmark implementation.

### Current local status

- The official repository, a compatible environment, and the released
  checkpoint are present.
- The checkpoint's `mm_compressor` now resolves to the local
  `stream_compressor` directory.
- The current environment does not contain the upstream FlashAttention
  dependency. A local compatibility change falls back to SDPA, so runs in this
  environment remain an implementation adaptation until the official
  FlashAttention path is restored.
- `baseline/dispider_streaming_eval.py` uses the upstream quick-start wrapper
  over one independently clipped visible prefix per question.
- Therefore its records must be labeled as a **single-turn visible-prefix
  adaptation**, not as an exact official Dispider benchmark reproduction.
- Exact VideoMME numbers should be produced by the upstream evaluation module
  and upstream scorer, without routing generation through the local runner.

For streaming benchmarks, the public repository does not provide an equivalent
OVO-Bench evaluation script. Do not infer official OVO/StreamingBench settings
from the VideoMME script.

Prepare an upstream-format VideoMME manifest and submit exact upstream
generation as follows:

```bash
MANIFEST=/mnt/is_data/xwu/video_skills/outputs/atomic_skills_for_video/manifests/dispider_videomme.json
python3 -m baseline.prepare_dispider_videomme_manifest --output "$MANIFEST"

QA_FILE="$MANIFEST" NUM_SHARDS=4 \
  sbatch --array=0-3 baseline/slurm_dispider_official_videomme.sbatch
```

After all shards finish, run the upstream `dispider/eval/eval_videomme.py`
scorer on the shard-only output directory.

## M3-Agent

### Official settings

The official pipeline has two distinct stages.

Memorization:

- split video into 30-second clips;
- sample 5 fps inside each clip;
- run face detection and speaker processing;
- generate episodic and semantic memories with
  `M3-Agent-Memorization`;
- store entity-centric multimodal graphs.

Relevant released defaults include:

- image/audio matching thresholds: `0.3` / `0.6`;
- maximum image/audio embeddings: `10` / `20`;
- face detection/quality thresholds: `0.85` / `22`;
- generation temperature: `1e-6`.

Control:

- `M3-Agent-Control` through vLLM with tensor parallel size 2;
- at most 5 search/answer rounds;
- retrieval `topk=2` and threshold `0.5` in the control code;
- sampling `temperature=0.6`, `top_p=0.95`, `top_k=20`,
  `max_tokens=1024`;
- GPT-4o (`gpt-4o-2024-11-20`) answer judging.

### Current local status

- The official repository and both official model checkpoints are present.
  The checkpoints currently live under the legacy
  `/mnt/is_data/xwu/video_skills/models` path.
- M3-Bench videos, intermediate outputs, memory graphs, a dedicated compatible
  environment, and evaluator credentials are not installed.
- Current project code only borrows M3-Agent ideas for clip memory and graph
  organization. It does not reproduce M3-Agent memorization or control.
- A FAISS text-memory baseline must not be reported as M3-Agent or
  "M3-Agent-equivalent".

The upstream `configs/api_config.json` example and `control.py` expect different
credential fields. Treat this as an upstream setup issue: provide the Azure
fields expected by `control.py` in a private, ignored config, and never commit
credentials.

### Adapt M3-Agent to the three local benchmarks

OVO-Bench, VideoMME, and StreamingBench can use the official M3-Agent models and
memory/control schemas without using the official M3-Bench dataset. Label this
setting `official_model_adapted_benchmarks`.

The local preparation path is:

```bash
python3 -m baseline.prepare_m3_agent_adapted_benchmarks \
  --datasets ovo_bench videomme streaming_bench \
  --limit-per-dataset 1 \
  --output-dir /mnt/is_data/xwu/video_skills/outputs/atomic_skills_for_video/m3_agent_adapted/manifests/smoke_3bench \
  --artifact-root /mnt/is_data/xwu/video_skills/outputs/atomic_skills_for_video/m3_agent_adapted/artifacts/smoke_3bench
```

This writes:

- `memorization_inputs.jsonl` in the schema consumed by the official
  memorization scripts;
- one official-Control-style annotation JSON per benchmark;
- `clip_plan.jsonl` for causal 30-second media staging;
- a summary and explicit skipped-example records.

VideoMME reuses one full-video graph for questions over the same video.
OVO-Bench and StreamingBench use a question-prefix graph keyed by source video
and observation cutoff. This is more expensive than integer `before_clip`
truncation, but it prevents future leakage when a question arrives inside a
30-second M3 clip. Use this exact-prefix mode for smoke and diagnostic runs,
not for the full three-benchmark run.

Materialize the clips with:

```bash
M3_MANIFEST=/mnt/is_data/xwu/video_skills/outputs/atomic_skills_for_video/m3_agent_adapted/manifests/smoke_3bench
QWEN_ENV=/mnt/is_data/xwu/video_skills/code/vllm_qwen_cu124_venv
FFMPEG=/mnt/is_data/xwu/video_skills/code/dispider_venv/lib/python3.10/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2

"$QWEN_ENV/bin/python" -m baseline.materialize_m3_agent_clips \
  --clip-plan "$M3_MANIFEST/clip_plan.jsonl" \
  --ffmpeg-bin "$FFMPEG"
```

Regular 30-second clips follow the upstream stream-copy command. A final
partial clip ending at a streaming question cutoff is re-encoded with audio to
enforce an exact causal boundary.

For the no-API adapted track, `baseline.run_m3_agent_local` invokes the official
memorization entrypoint while replacing hosted embedding and Whisper calls
with local `Qwen/Qwen3-VL-Embedding-2B` and
`openai/whisper-large-v3-turbo`. It also sorts the upstream clip glob
chronologically. The official Memorization and Control checkpoints remain
unchanged; the local retrieval/speech backends are recorded deviations.

### Full-run cost and optimized causal policy

The initial exact question-prefix plan expands the three benchmarks to 6,899
memory graphs, 81,333 clips, about 650 hours of effective video, and roughly
608 GiB of staged media. This is slow because official Memorization does much
more than compute one embedding. Every 30-second clip is sampled at 5 fps
(about 150 frames) and passes through video/audio decoding, Whisper,
SpeakerLab, InsightFace, the 9B Qwen2.5-Omni episodic/semantic memory
generation, local text embedding, and entity-graph updates. Clips within one
source graph must remain sequential so that face, speaker, and entity
identities remain consistent.

At 16-way GPU parallelism, 81,333 clips leave about 5,083 clips per GPU.
One, two, or four minutes per clip correspond to approximately 3.5, 7, or 14
days of Memorization wall time, respectively. Control runs only after the
graphs exist and adds five iterative Search/Answer rounds per question.

The full run therefore uses `baseline.optimize_m3_agent_manifest` and
`baseline.run_m3_agent_snapshot_memorization`:

- process each source video once instead of rebuilding every question prefix;
- save graph snapshots after complete 30-second clips;
- point each streaming question to the latest snapshot whose clip ends no
  later than the question cutoff;
- never expose a graph state produced from future clips.

This changes the workload to:

- 3,268 source graphs instead of 6,899 question-prefix graphs (53% fewer);
- 56,148 clips instead of 81,333 (31% fewer);
- about 458.5 video hours instead of 650;
- 12,855 StreamingBench clips instead of 38,148 (66% fewer).

The conservative snapshot policy discards the visible but incomplete tail
between the latest 30-second boundary and the question timestamp. Across
StreamingBench this tail averages 14.14 seconds and is at most 29 seconds.
Record the policy as `completed_30s_graph_snapshots`; it is causal and retains
official 30-second clip processing, but it is not identical to either exact
question-prefix Memorization or the original M3-Bench protocol.

Even after deduplication, the run still processes roughly 8.25 million sampled
frames and invokes Qwen2.5-Omni 56,148 times. At 16-way parallelism, expected
Memorization time is roughly 2.5--10 days for one--four minutes per clip, with
the final estimate determined by the smoke run.

The active 12--48 hour fast profile makes additional deviations: it reduces
5 fps to 1 fps, caps Memorization generation at 1024 instead of 4096 tokens,
disables the separate InsightFace, SpeakerLab, and Whisper branches, and allows
up to 32 one-GPU Memorization workers. Qwen2.5-Omni still receives clip video
and audio, and the official Memorization/Control checkpoints, 30-second clip
structure, episodic/semantic graph construction, and five-round Control loop
remain enabled. Report this run as a fast adapted ablation, never as the
official-settings M3 Memorization baseline.

The dedicated local environments now include the face, speaker, audio,
Qwen2.5-Omni, and Control dependencies. The older `steam_video` “M3-Agent”
runner still does not satisfy this definition because it is Qwen3.5 visual
clip retrieval rather than official M3 memorization/control.

## StreamBridge

### Official settings

The released evaluation uses the upstream online model classes and incrementally
feeds one frame at a time.

Common:

- sampled fps: 1;
- deterministic decoding (`do_sample=False`, `num_beams=1`);
- `max_new_tokens=128`;
- model choices: `qwen2vl`, `oryx`, or `llava_ov`.

OVO-Bench:

- preserve all questions from the same source video as an ordered multi-turn
  stream;
- feed only frames between consecutive real-time anchors before each question;
- keep the model memory/cache across turns;
- official script defaults: `MAX_IMG_TOKEN=16384`,
  `POOLING_FACTOR=2`.

VideoMME:

- single-turn, whole-video input;
- official script defaults: `MAX_IMG_TOKEN=65536`,
  `POOLING_FACTOR=4`, `MAX_FRAME_NUM=1024`.

### Current local status

- The official Apple repository and annotations are present.
- No trained StreamBridge checkpoint or dedicated environment is present. The
  upstream README states that checkpoint download is TBD.
- `dataset_clip_wrapper/adapters/streaming_video.py` is only an annotation/video
  resolver. It does not instantiate StreamBridge, perform round-decayed memory
  compression, preserve online cache across OVO turns, or run the activation
  model.
- The available OVO storage exposes per-question `chunked_videos`, while the
  official StreamBridge evaluator expects original source-video paths from its
  annotation. Per-question chunks can support a no-future-leak adaptation, but
  cannot reproduce official multi-turn memory behavior.

Until compatible weights and original video layout are available, report this
component as **StreamBridge-format dataset support**, not a StreamBridge model
baseline.

## Reporting rule

Use one of these labels:

- `official_upstream`: unmodified upstream model path, prompt, sampling,
  streaming state, and scorer;
- `official_model_adapted_protocol`: official weights/code with an explicitly
  documented local protocol change;
- `format_compatible_only`: annotations or schemas are compatible, but the
  official model pipeline is absent;
- `inspired_by`: only architectural ideas are borrowed.

Current classification:

- Dispider local runner: `official_model_adapted_protocol`
- M3-Agent-related local memory code: `inspired_by`
- StreamBridge local adapter: `format_compatible_only`

## Recommended reproduction plan

Maintain two separate experiment tracks:

1. `official_upstream`: preserve the upstream model, prompt, sampling,
   streaming state, dataset protocol, and scorer.
2. `unified_protocol`: evaluate every method under the same local visibility,
   output, failure-accounting, and metric rules.

Never use a unified-protocol result as an official-paper reproduction.

### Priority 1 — Dispider official VideoMME

This is the nearest complete reproduction because the official repository,
checkpoint, VideoMME videos, and a validated 900-video/2700-question manifest
are already available.

Remaining work:

- create a clean upstream environment with Python 3.10, Torch 2.2.0,
  Transformers 4.41.2, and FlashAttention 2.5.9.post1;
- remove the SDPA compatibility fallback by running a clean upstream checkout;
- run `dispider/eval/model_videomme_long.py` directly with full videos, the
  official prompt, 32 clips × 16 frames, greedy decoding, 256 output tokens,
  and the upstream scorer;
- record the upstream Git SHA, checkpoint hash, dataset revision, dependency
  versions, and exact command.

This can produce a credible `official_upstream` VideoMME result. Full
asynchronous active-streaming reproduction remains unavailable because the
public repository does not release the complete streaming inference path.

### Priority 2 — M3-Agent Control over official memory graphs

The official repository and both checkpoints are present. The fastest credible
baseline is to consume the released M3-Bench memory graphs before attempting to
rebuild memorization.

First stage:

- download M3-Bench annotations, intermediate outputs, and official memory
  graphs;
- create the official Control environment with Transformers 4.51.0,
  vLLM 0.8.4, and NumPy 1.26.4;
- run the 32B Control checkpoint with tensor parallel size 2, five
  Search/Answer rounds, retrieval top-2, threshold 0.5, and the official
  decoding parameters;
- configure the GPT-4o answer judge in a private ignored file;
- reproduce M3-Bench robot/web metrics with the official evaluator.

Only after Control is reproduced should full memorization be attempted:
30-second clips, 5 fps, audio, InsightFace, SpeakerLab/ERes2NetV2, and
episodic/semantic/entity graph construction.

### Priority 3 — StreamBridge protocol reproduction

Strict paper-number reproduction is currently blocked: Apple does not publish
the trained Stream-IT/activation checkpoints, and local OVO storage lacks the
original continuous source-video layout expected by the evaluator.

The strongest defensible interim target is
`streambridge_protocol_reimplementation`:

- obtain original OVO source videos and keep each video's questions in one
  ordered session;
- feed frames sequentially at 1 fps;
- preserve visual/text cache and assistant responses across turns;
- implement round-decayed compression;
- use the official prompts, greedy decoding, and 128-token limit;
- report official task-level macro metrics separately from local strict
  metrics.

Without the trained StreamBridge and activation checkpoints, this protocol
implementation must not be reported as the official StreamBridge model.

### Execution order

1. Dispider official VideoMME.
2. M3-Agent official memory graphs + Control.
3. Passive multi-turn StreamBridge protocol.
4. Full M3-Agent memorization.
5. Full StreamBridge only after compatible checkpoints become available.
