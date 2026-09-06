# Moving the pipeline to another cluster (2026-09-06)

Everything below is what the current runs actually depend on. Absolute paths
that must change are listed at the end.

## 1. Code
```
git clone git@github.com:wuxiyang1996/Video_Skills.git
git checkout cleanup/functional-layout-20260724
```
Launchers used for the paper runs are in `scripts/launch/` (copied from the
session scratchpad; each has `sbatch` flags for UMD's `scavenger` partition —
edit partition / account / qos / `--gres` / `--constraint` / `--exclude`).

## 2. Data to copy (sizes)
| what | path here | size | needed for |
|---|---|---|---|
| Video-Holmes videos + QA + annotations | `datasets/Video-Holmes/Benchmark/{videos_cropped,test_Video-Holmes.json,train_Video-Holmes.json,annotations,annotation_training}` | 4.1 GB | everything on VH (L1, whisper, narratives, training, eval) |
| VRBench videos | `datasets/VRBench` | 395 GB | only to *build* new VRBench L1; copy the built catalogs instead (`/fs/nexus-scratch/wuxiyang/vrbench_pilot_v1`, 1.1 GB) |
| CG-Bench videos + subtitles | `datasets/CG-Bench` | 386 GB | only to build new CG L1; the 237-q catalogs are ~2 GB (`l2_paper_cg_vh_20260901/cg_heldout_questions_v1`, `vh_l1_levers/cg_narr_px_plus`) |
| Qwen3.5-9B weights | `Multi-hop-Reasoning-VLM-Agent/.hf_cache/hub/models--Qwen--Qwen3.5-9B` | 24 GB | describer (vLLM) and the trainable reader; or `huggingface-cli download Qwen/Qwen3.5-9B` |
| VH test catalogs (both generations) | `/fs/nexus-scratch/wuxiyang/vh_l1_levers/{narr_px_plus_full,narr_px_plus_full_rep}` | 2 × 5.9 GB | final evaluation of any reader |
| VH frozen per-video L1 + derived per-question copies | `dataset_clip_wrapper/output/{l2_paper_vh_heldout_l1_v3,vh_full_questions_v1}` | 3.8 + 5.9 GB | the "ours" baseline catalog |
| whisper caches | `/fs/nexus-scratch/wuxiyang/{vh_l1_levers/asr/whisper,vh_train_l1/asr/whisper}` | < 50 MB | dialogue rows |
| narrative caches | `.../narr_px*/narratives/*.json` | < 100 MB | rebuild any narrative catalog without API calls |
| VH train L1 (when built) | `/fs/nexus-scratch/wuxiyang/vh_train_l1` | ~5 GB | reader training |

## 3. Environments (three; lockfiles in `envs/`)
1. **vLLM serve** (`.venv-qwen35-vllm`, `envs/vllm-serve.lock.txt`): vllm 0.25 / torch 2.11 cu130 — the 9B describer server inside `scripts/sft_pilot/run_local_qwen_worker.sh` and the reader baseline job. Needs a GPU with compute capability ≥ 8.0 (bf16); the worker sets `HF_HOME`.
2. **eval / catalog venv** (`.venv-qwen35-serve` → resolves to the `swift` conda env, `envs/swift-grpo.lock.txt`): opencv, PyAV, openai, requests, ms-swift 3.10, TRL 0.23, vllm 0.8.5 — everything under `scripts/eval/`, `trainer/reader/`, and GRPO.
3. **PEFT + FlashAttention-2** (`conda/envs/video-skills-grpo`, `envs/peft-fa2.lock.txt`): built by `scripts/grpo/create_conda_env.sh` + `scripts/grpo/install_flash_attn.sh` (torch 2.6 cu124, FA2 2.7.4) — LoRA SFT (`trainer/reader/sft_lora.py`, falls back to sdpa without FA2).
`pip install -r envs/<lock>` inside a fresh venv/conda of the matching Python (3.10 for swift, 3.10/3.11 for the others); FA2 must match the torch/cu build.

## 4. Secrets
`/fs/gamma-projects/vlm-robot/keys.py` with `OPENROUTER_API_KEY` and `OPENAI_API_KEY`
(the OpenAI project key is region-pinned: `transcribe_videos.py --base-url https://us.api.openai.com/v1`).
Copy it out of band, `chmod 600`, and point `--keys-py` (or `OPENROUTER_API_KEY` env) at it. Pin the OpenRouter provider for final runs: `OPENROUTER_PROVIDER_ORDER=Alibaba`.

## 5. Paths that are hard-coded here
- dataset root `/fs/gamma-projects/vlm-robot/datasets` — `scripts/eval/derive_full_question_examples.py --dataset-root`, `scripts/eval/mine_failure_subtrajectories.py`, the adapters' default; pass the flag or edit the default.
- `HF_HOME=/fs/gamma-projects/vlm-robot/Multi-hop-Reasoning-VLM-Agent/.hf_cache` in `scripts/sft_pilot/run_local_qwen_worker.sh` and `scripts/launch/reader_baseline.sbatch`.
- scratch outputs via symlinks `dataset_clip_wrapper/output/{vh_l1_levers,vh_train_l1,vrbench_pilot_v1}` → `/fs/nexus-scratch/wuxiyang/...`; recreate the symlinks or set `PILOT_TAG` to a local directory.
- `keys.py` path in `scripts/launch/*.sh` and `trainer/reader/*.py` defaults (`--keys-py`).
- Slurm: partition / account / qos / GPU gres names in every `scripts/launch/*.sh` and the worker's `--gres`.

## 6. Minimal set for the training line only
VH data (4.1 GB) + Qwen3.5-9B (24 GB) + the two test catalogs (12 GB) + whisper/narrative caches + envs 1–3 + keys. Roughly 45 GB and an afternoon of environment building; the train-split L1 (~60 GPU-h) can be built there directly with `scripts/launch/launch_vh_train_l1.sh` after editing the sbatch flags.
