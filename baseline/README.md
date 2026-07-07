# Baseline Video QA Memory

This folder contains the standard baseline path for streaming video QA:

```text
dataset_clip_wrapper canonical schema
  -> normalized video QA examples
  -> clip records
  -> clip embeddings
  -> FAISS retrieval
  -> local Qwen3.5-9B answer generation
```

The baseline is intentionally separate from `dataset_clip_wrapper/`. The wrapper
owns dataset adapters and canonical examples; this folder owns a simple,
comparable retrieval baseline over those canonical examples.

## Local Model Boundary

Current evaluation should use the locally deployed Qwen3.5-9B model:

```text
/mnt/is_data/xwu/video_skills/data/models/qwen35_9b/Qwen3.5-9B
```

This baseline should not call OpenRouter or any hosted API for model inference.
The local A6000-compatible environment is:

```text
/mnt/is_data/xwu/video_skills/code/vllm_qwen_cu124_venv
```

## Standard Records

`schemas.py` defines three JSONL-friendly records:

- `VideoQAExample`: one question over one video, including streaming visibility.
- `VideoClipRecord`: one wrapper-derived video clip span.
- `RetrievedClip`: one retrieval result with score and source clip metadata.

The key idea is that all retrieval, RAG, and Qwen calls refer back to stable
clip IDs and source spans:

```json
{
  "clip_id": "clip:video_id:fine:0007",
  "video_path": "/path/to/video.mp4",
  "start_s": 21.0,
  "end_s": 25.0,
  "visible_until_s": 60.0
}
```

## FAISS Plan

`faiss_store.py` is a small wrapper around FAISS. It stores:

- `index.faiss`: vector index
- `clips.jsonl`: clip metadata aligned to FAISS row IDs, including the clip
  embedding in `embedding`
- `manifest.json`: model/dimension/count metadata

Two embedding backends are currently available:

- `hashing_text`: deterministic text hashing over clip schema text. This is a
  plumbing smoke backend, not a strong semantic retriever.
- `clip`: cross-modal CLIP retrieval. Video clips are embedded by sampling one
  or more frames from the wrapper clip span and averaging CLIP image embeddings.
  Questions are embedded with the matching CLIP text encoder.

The recommended smoke/default setting is currently:

```text
--embedding-backend clip
--clip-model openai/clip-vit-base-patch32
--frames-per-clip 4
```

This keeps the retrieval unit as a wrapper video clip while representing each
clip with multiple sampled frames rather than a single still image.

At query time, the question text is embedded with the same model and searched
against the FAISS index. The query CLI hides full vectors by default to keep
terminal output readable; pass `--include-embeddings` to print the question and
clip embedding arrays.

## Example Commands

Build a clip index from canonical examples:

```bash
/mnt/is_data/xwu/video_skills/code/vllm_qwen_cu124_venv/bin/python \
  -m baseline.build_faiss_index \
  --canonical-jsonl /mnt/is_data/xwu/video_skills/outputs/atomic_skills_for_video/qwen35_streaming_eval/290877/canonical_schemas.jsonl \
  --output-dir /mnt/is_data/xwu/video_skills/outputs/atomic_skills_for_video/baseline_faiss/ovo_videomme_smoke \
  --embedding-backend hashing_text
```

Build a cross-modal CLIP clip index with 4 sampled frames per wrapper clip:

```bash
export HF_HOME=/mnt/is_data/xwu/video_skills/data/models/hf_cache
export HUGGINGFACE_HUB_CACHE=/mnt/is_data/xwu/video_skills/data/models/hf_cache/hub
export TRANSFORMERS_CACHE=/mnt/is_data/xwu/video_skills/data/models/hf_cache/transformers

/mnt/is_data/xwu/video_skills/code/vllm_qwen_cu124_venv/bin/python \
  -m baseline.build_faiss_index \
  --canonical-jsonl /mnt/is_data/xwu/video_skills/outputs/atomic_skills_for_video/qwen35_streaming_eval/290877/canonical_schemas.jsonl \
  --output-dir /mnt/is_data/xwu/video_skills/outputs/atomic_skills_for_video/baseline_faiss/ovo_videomme_clip \
  --embedding-backend clip \
  --clip-model openai/clip-vit-base-patch32 \
  --frames-per-clip 4
```

Query the index:

```bash
/mnt/is_data/xwu/video_skills/code/vllm_qwen_cu124_venv/bin/python \
  -m baseline.query_faiss_index \
  --index-dir /mnt/is_data/xwu/video_skills/outputs/atomic_skills_for_video/baseline_faiss/ovo_videomme_smoke \
  --query "Who did I communicate to when chopping eggplants?" \
  --topk 5
```

FAISS is installed in the current A6000-compatible environment:

```text
/mnt/is_data/xwu/video_skills/code/vllm_qwen_cu124_venv
```

Verified smoke outputs:

```text
/mnt/is_data/xwu/video_skills/outputs/atomic_skills_for_video/baseline_faiss/ovo_videomme_smoke             # hashing_text, 420 clips
/mnt/is_data/xwu/video_skills/outputs/atomic_skills_for_video/baseline_faiss/ovo_videomme_clip_smoke        # CLIP, 10 clips, 1 frame/clip
/mnt/is_data/xwu/video_skills/outputs/atomic_skills_for_video/baseline_faiss/ovo_videomme_clip_5x5          # CLIP, 5+5 examples, 1 frame/clip
/mnt/is_data/xwu/video_skills/outputs/atomic_skills_for_video/baseline_faiss/ovo_videomme_clip_5x5_4frames  # CLIP, 5+5 examples, 4 frames/clip
```

Latest 5+5 smoke:

```text
job_id: 290913
state: COMPLETED
elapsed: 00:07:46
node: cipr-gpu16
canonical input: /mnt/is_data/xwu/video_skills/outputs/atomic_skills_for_video/qwen35_streaming_eval/290877/canonical_schemas.jsonl
output: /mnt/is_data/xwu/video_skills/outputs/atomic_skills_for_video/baseline_faiss/ovo_videomme_clip_5x5_4frames
```

Summary:

```text
examples: 5 OVO-Bench + 5 VideoMME
indexed clips: 420
OVO-Bench clips: 325
VideoMME clips: 95
embedding_backend: clip
embedding_model: openai/clip-vit-base-patch32
embedding_dim: 512
frames_per_clip: 4
```

The output directory contains:

```text
clips.jsonl                     # VideoClipRecord schema + embedding per clip
index.faiss                     # FAISS index over clip embeddings
manifest.json                   # embedding/index metadata
query_clip_4frames_compact.json # query embedding -> top-k clip retrieval smoke
```

Each `clips.jsonl` row stores future-reference metadata:

```text
row_id
example_id
dataset
video_id
clip_id
video_path
start_s
end_s
granularity
visible_until_s
text
embedding
metadata
```

The retrieved clip rows are ready to become local Qwen3.5-9B direct video
inputs:

```python
{
    "type": "video",
    "video": clip["video_path"],
    "video_start": clip["start_s"],
    "video_end": clip["end_s"],
}
```
