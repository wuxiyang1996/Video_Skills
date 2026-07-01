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
| `long` | `hierarchical` 30s coarse index + retrieve-gated 8s fine |
| `streaming` | `fixed_window` with `online=true` and `observation_end_s` |

Override with CLI flags: `--clip-strategy`, `--window-s`, `--overlap-s`,
`--coarse-window-s`, `--fine-window-s`, `--observation-end-s`,
`--index-fine-expansion`, `--retrieval-topk`, `--retrieval-mode`, `--no-retrieval`.

Long-video flow (M3-style):

```text
coarse index (30s windows, all clips in evidence_index)
  -> lexical retrieve top-k coarse clips (question + visible segments)
  -> fine windows (8s) only inside retrieved coarse parents
  -> Qwen clip-schema + graph compose on perception clips only
```

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

Stage 1 uses a multimodal OpenRouter model (default `qwen/qwen3.5-9b`, closest
available Qwen3.5 ~8B-class VLM on OpenRouter) to turn each segmented clip into
a structured clip schema.

Stage 2 uses `openai/gpt-oss-120b` to plan and execute Evidence Graph
Construction atomic skills, producing a clue-memory / perception graph.

```text
segment clips (short / long / streaming)
  -> [long] retrieve top-k coarse clips
  -> [long] expand fine windows inside candidates only
  -> Qwen clip-schema producer (perception clips)
  -> gpt-oss-120B graph composer over atomic graph-crafting skills
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
| `--clip-schema-model` | `qwen/qwen3.5-9b` | multimodal clip-schema producer |
| `--clip-schema-max-clips` | `3` | cap Qwen calls per example |
| `--graph-model` | `openai/gpt-oss-120b` | graph planner / composer |
| `--graph-deterministic` | off | apply atomic skills directly from clip schemas |
| `--keys-py` | workspace `keys.py` | OpenRouter API key source |

Offline graph-compose smoke test:

```bash
python dataset_clip_wrapper/smoke_test_graph_compose.py
```

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
