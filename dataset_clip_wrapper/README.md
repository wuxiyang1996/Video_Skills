# Dataset Clip Wrapper

Wrap the four core video benchmarks into the canonical
`CanonicalVideoExample` schema with clip segmentation for **short**, **long**,
and **streaming** regimes.

Supported datasets:

- `video_holmes`
- `cg_bench`
- `vrbench`
- `siv_bench`

## Pipeline

```text
dataset adapter
  -> probe duration
  -> clip_policy segmentation (short / long / streaming)
  -> optional perception backbone captions per clip
  -> evidence_candidates + evidence_index clip graph
  -> canonical JSON example
```

## Hyperparameters

### Video regime

| Regime | Default policy |
|--------|----------------|
| `short` | `whole_video` + 4s windows |
| `long` | `hierarchical` 45s coarse + 8s fine |
| `streaming` | `fixed_window` with `online=true` and `observation_end_s` |

Override with CLI flags: `--clip-strategy`, `--window-s`, `--overlap-s`,
`--coarse-window-s`, `--fine-window-s`, `--observation-end-s`.

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

See [clip-processing-policy.md](../docs/clip-processing-policy.md) for regime defaults.
