# Video_Skills：环境搭建与复现手册（2026-09-09）

本文件是把整条流水线（证据目录 → 教师 rationale → 9B reader SFT → 三库评测）搬到任何一台
新机器上所需的全部说明。英文的迁移要点在 `docs/cluster_setup.md`，结果全记录在
`docs/l2_sota_targets.md`，论文表在 `docs/paper_tables.md`。

## 0. 当前结果（同一个 9B adapter `sft_mix`，用来核对复现是否成功）

| held-out 集 | 训练后 9B | 235B 老师（同目录） | 配对差 95% CI | base 9B |
|---|---|---|---|---|
| Video-Holmes 全测 1,837 | 51.3 | 48.3 | +2.9 [+0.7, +5.1] | 41.0 |
| Video-Holmes 全测，gen-2 目录（sft_v2） | 51.4 | 45.6 | +5.8 [+3.5, +8.2] | — |
| VRBench held-out 495（视频 61–120） | 75.4 | 71.7 | +3.6 [+0.2, +7.3] | — |
| CG-Bench 237（64k 上下文，231 可答） | 39.4 | 41.1 | −1.7 [−8.2, +4.8] | 39.0 |

配对 bootstrap 脚本：`scripts/launch/boot_pair.py`（若不在，见 `scratchpad/boot_pair.py` 的副本）。
grounded accuracy（答对 ∧ 引用精度 ≥ 0.5）：VH 用 `trainer/reader/grounded_vh.py`，VRBench/CG 用
`scripts/eval/grounded_accuracy.py` 生成 `*.rollouts.grounded.jsonl` 后按
`scripts/launch/vrbench_heldout_chain.sh` 末尾的内联 python 计算精度判据
（脚本直接打印的 `grounded_accuracy_all` 是召回判据，不是预注册的精度判据）。

## 1. 代码

```bash
git clone git@github.com:wuxiyang1996/Video_Skills.git
cd Video_Skills && git checkout cleanup/functional-layout-20260724
# 或者从 B2 包里的 bundle 恢复（见第 8 节）
```

目录：
- `scripts/eval/`：目录构建（`build_narrative_catalog.py`）、字幕转录（`transcribe_videos.py`）、
  按题派生（`derive_full_question_examples.py`）、答题链与教师（`measure_answer_chain.py`）、
  grounded 指标（`grounded_accuracy.py`）。
- `trainer/reader/`：reader 提示与解析（`prompting.py`）、奖励（`rewards.py`）、SFT 数据
  （`build_sft_data.py`）、LoRA SFT（`sft_lora.py`）、合并（`merge_adapter.py`）、VH grounded（`grounded_vh.py`）。
- `scripts/launch/`：所有 Slurm 启动器和自动链（`reader_sft.sbatch`、`reader_eval.sbatch`、
  `reader_mix_chain.sh`、`cg_train_data_chain.sh`、`vrb_train_data_chain.sh`、`cg_teacher_resume.sh` 等）。
- `dataset_clip_wrapper/`：L1 clip 描述流水线与 OpenRouter 客户端（`perception/openrouter_client.py`）。
- `envs/`：三个环境的 pip lock。`scripts/migrate/`：rsync 与远端安装脚本。

## 2. 三个环境（lock 在 `envs/`）

| 名称 | 用途 | 关键版本 | 建法 |
|---|---|---|---|
| `.venv-qwen35-vllm`（`envs/vllm-serve.lock.txt`） | vLLM 0.25 服务 9B（L1 描述器、reader 评测） | torch 2.11 cu130，vllm 0.25 | `python3.11 -m venv .venv-qwen35-vllm && .venv-qwen35-vllm/bin/pip install -r envs/vllm-serve.lock.txt` |
| `.venv-qwen35-serve` → conda `swift`（`envs/swift-grpo.lock.txt`） | `scripts/eval/*`、`trainer/reader/build_sft_data.py`、教师调用、目录构建 | Python 3.10，opencv、PyAV、openai、requests，ms-swift 3.10，TRL 0.23，vllm 0.8.5 | `conda create -n swift python=3.10 && pip install -r envs/swift-grpo.lock.txt`，然后 `ln -s $(conda info --base)/envs/swift .venv-qwen35-serve` |
| conda `video-skills-grpo`（`envs/peft-fa2.lock.txt`） | LoRA SFT（`sft_lora.py`）、合并 | torch 2.6 cu124，PEFT，FlashAttention-2 2.7.4 | `scripts/grpo/create_conda_env.sh` + `scripts/grpo/install_flash_attn.sh` |

Qwen3.5-9B 训练快路径还需要 `flash-linear-attention` 和 `causal_conv1d`；集群上 `causal_conv1d`
是一个 torch 实现的 shim（安装到 `video-skills-grpo` 的 site-packages，带 dist-info 元数据），没有它会慢 10 倍。
FA2 必须和 torch/CUDA 版本匹配；没有 FA2 时 `sft_lora.py` 自动退回 sdpa。

GPU 要求：bf16（计算能力 ≥ 8.0）。SFT 在 32k 上下文单卡 48 GB 可跑（L40S 约 60 s/step，
A6000 约 1.5 倍慢）；评测 64k 上下文单卡 48 GB。24 GB 卡不行。

## 3. 模型与密钥

- 基座：`Qwen/Qwen3.5-9B`（24 GB，`huggingface-cli download Qwen/Qwen3.5-9B`），
  通过 `HF_HOME` 指向缓存（启动器里写的是 `/fs/gamma-projects/vlm-robot/Multi-hop-Reasoning-VLM-Agent/.hf_cache`）。
- 教师：`qwen/qwen3-vl-235b-a22b-instruct`（OpenRouter，固定 `OPENROUTER_PROVIDER_ORDER=Alibaba`），
  技能/规划模型 `openai/gpt-oss-120b`（OpenRouter），字幕 `whisper-1`（OpenAI，
  `--base-url https://us.api.openai.com/v1`）。
- 密钥文件 `keys.py`（`OPENROUTER_API_KEY`、`OPENAI_API_KEY`），放在仓库外，`chmod 600`，
  用 `--keys-py` 或环境变量指向。**不要进 git、不要进 B2 包。**
- 费用参考：CG 训练集 1,080 题一轮教师约 $8；60 秒窗口叙事目录（161 视频、76 小时、4,582 窗口 × 16 帧）
  是最大的 API 开销，一次约 $150–200。跑之前查余额：`GET https://openrouter.ai/api/v1/credits`。

## 4. 数据（分三档）

**A. 外部数据集（不打包，官方下载）**：Video-Holmes 12 GB（`datasets/Video-Holmes/Benchmark/`）、
CG-Bench 386 GB（`datasets/CG-Bench/{cg_videos,cg_subtitles,cgbench.json,cgbench_mini.json}`）、
VRBench 395 GB（`datasets/VRBench/`，评测用 `VRBench_eval.jsonl`）。只有重新构建 L1 时才需要视频。

**B. 已构建的证据目录与派生题（约 25 GB，scratch）**

| 路径（`/fs/nexus-scratch/wuxiyang/`） | 内容 |
|---|---|
| `vh_l1_levers/narr_px_plus_full`、`narr_px_plus_full_rep` | VH 全测 1,837 题的 gen-1 / gen-2 叙事目录（30 秒窗口 + whisper 对话 + clip） |
| `vh_l1_levers/narr_px_plus`、`asr/whisper`、`all_ids_1837.txt`、`cg_ids_237.txt` | VH fresh-300 目录、字幕缓存、id 列表 |
| `vh_l1_levers/cg_narr_px_plus`、`cg_asr` | CG 237 评测目录（60 秒窗口）与字幕 |
| `vh_train_l1/{narr_px_plus,asr/whisper,train_ids.txt}` | VH train 1,551 题目录 |
| `vrbench_pilot_v1/{derived_pilot60,heldout60,pilot_all_ids.txt}` | VRBench pilot 480 题、held-out 495 题的目录与评测结果 |
| `cg_train_l1/{train_index.json,train_ids.txt,asr,narr_px_plus}` | CG 训练 1,080 题（161 视频，不含 cgbench_mini 和 67 个评测视频） |
| `cg_mini_l1/` | cgbench_mini 的 L1（部分） |

**C. 结果与训练产物（约 25 GB）**

| 路径 | 内容 |
|---|---|
| `dataset_clip_wrapper/output/l2_paper_cg_vh_20260901/{full_vh_mm,cg_qa}` | 所有评测的 `*.rows.jsonl` / `*.rollouts.jsonl`（论文里每个数字的原始行） |
| `/fs/nexus-scratch/wuxiyang/reader_train/` | 教师 rollouts（`teacher_s{0,1,2}`、`vrb_teacher_s*`、`cg_teacher_s*`）、SFT 行（`sft_v1/sft_vrb_p50/sft_cg{,_p50}/sft_mix{,2}.jsonl`）、adapters（`sft_v2/sft_v2b/sft_v3p/sft_mix/sft_mix2/adapter`，各 131 MB） |
| `dataset_clip_wrapper/output/{sft_controller_expansion_20260721_retrieval_paid,sft_auto_20260713_full_retrieval}` | CG 训练题来源的两条干净 L1 lane |
| `dataset_clip_wrapper/output/{vh_full_questions_v1,l2_paper_vh_heldout_l1_v3}` | VH 的冻结 L1 与按题派生 |

可再生、不打包：合并后的全量权重（`dataset_clip_wrapper/output/reader_merged/*`，每个 17 GB，
`merge_adapter.py` 几分钟生成）、三个环境、基座缓存。

## 5. 写死的路径（换机器要改）

- 数据根 `/fs/gamma-projects/vlm-robot/datasets`：`derive_full_question_examples.py --dataset-root`、
  `mine_failure_subtrajectories.py`、各 adapter 默认值。
- `HF_HOME`：`scripts/sft_pilot/run_local_qwen_worker.sh`、`scripts/launch/reader_*.sbatch`。
- scratch 软链接 `dataset_clip_wrapper/output/{vh_l1_levers,vh_train_l1,vrbench_pilot_v1}` → `/fs/nexus-scratch/wuxiyang/...`。
- `keys.py` 路径：`scripts/launch/*.sh`、`trainer/reader/*.py` 的 `--keys-py` 默认值。
- 会话 scratchpad 路径 `/tmp/claude-*/.../scratchpad`：`scripts/launch/*_chain.sh` 里的 `S=`，改成任意可写目录。
- Slurm：每个 `scripts/launch/*.sbatch` 的 partition / account / qos / `--gres` / `--constraint` / `--exclude`。
  gamma 分区示例：`SBATCH_PARTITION=gamma SBATCH_ACCOUNT=gamma SBATCH_QOS=huge-long SBATCH_GRES=gpu:rtxa6000:1`
  （`default` QOS 限每任务 4 核 32 GB，我们的任务要 8 核 96 GB）。环境变量覆盖脚本里的 `#SBATCH`。

## 6. 流水线各步（命令）

```bash
SPY=.venv-qwen35-serve/bin/python; export OPENROUTER_PROVIDER_ORDER=Alibaba
# 6.1 L1 clip 描述（GPU，vLLM 9B，4 秒 clip）：按数据集分 shard
bash scripts/launch/launch_vh_train_l1.sh          # 或 launch_cg_mini_l1.sh / launch_vrbench_subset240.sh
# 6.2 按题派生（slim 每题一个 example）
$SPY scripts/eval/derive_full_question_examples.py --frozen-l1-glob "<L1>/stages/*/04_l1_example.json" \
  --dataset video_holmes --split train --output-root <OUT>/derived --slim      # CG 用 --cg-full
# 6.3 字幕：whisper（VH/VRBench）或 CG 自带 srt 转 json（cg_train_data_chain.sh 里的内联 python）
$SPY scripts/eval/transcribe_videos.py --base-url https://us.api.openai.com/v1 ...
# 6.4 叙事目录（30 秒窗口 VH/VRBench；60 秒窗口 CG；16 帧/窗口，235B）
$SPY scripts/eval/build_narrative_catalog.py --example-index <derived>/example_index.json --example-ids <ids> \
  --output-root <OUT>/narr_px_plus --window-s 30 --frames-per-window 16 --asr-dir <asr> --no-clip-text --keep-clips --slim --workers 8
# 6.5 教师 rationale，三种选项顺序
for seed in 0 1 2; do extra=""; [ $seed -gt 0 ] && extra="--shuffle-options $seed"
  $SPY -m scripts.eval.measure_answer_chain --l1-glob "<catalog>/stages/*/04_l1_example.json" --indices-from all \
    --conditions direct --rationale $extra --example-ids <ids> --workers 6 --timeout-s 300 \
    --skill-model openai/gpt-oss-120b --answer-model qwen/qwen3-vl-235b-a22b-instruct \
    --dump-rollouts teacher_s$seed.rollouts.jsonl --output teacher_s$seed.json; done
# 6.6 校验行（答对 ∧ 引用精度 ≥ 0.5，有金标时）
$SPY -m trainer.reader.build_sft_data --example-index <catalog>/example_index.json \
  --rollouts teacher_s0.rollouts.jsonl teacher_s1.rollouts.jsonl teacher_s2.rollouts.jsonl --output sft.jsonl --min-precision 0.5
# 6.7 LoRA SFT（1 epoch，lr 1e-4，r16，grad-accum 8，completion-only loss，enable_thinking=False；可断点续跑）
DATA=sft.jsonl OUT=<run> MAXLEN=32768 SAVE_STEPS=15 sbatch scripts/launch/reader_sft.sbatch
# 6.8 合并 + vLLM 评测（必须合并；vLLM 直接挂 LoRA 会塌到单一选项；served-model-name 必须含 "qwen3"）
ADAPTER=<run>/adapter KEEP_MERGED=1 MAXLEN=65536 L1GLOB="<catalog>/stages/*/04_l1_example.json" IDS=<ids> OUT=<prefix> \
  sbatch scripts/launch/reader_eval.sbatch        # 可断点续跑：attempt 文件按任务号命名，结束时合并
# 6.9 配对对比
python3 scripts/launch/boot_pair.py <A>.rows.jsonl <B>.rows.jsonl A B
```

一键链：`reader_mix_chain.sh`（等数据 → 合并三库 → SFT → 三个评测 → 配对），`cg_train_data_chain.sh` /
`vrb_train_data_chain.sh`（字幕 → 目录 → 教师 × 3 → 校验行），`cg_teacher_resume.sh`（带余额守卫的教师重跑）。
链用 `setsid nohup bash <chain> >> <log> 2>&1 < /dev/null & disown` 启动，能活过会话重启。

## 7. 复现三个 headline 数字（有目录和 adapter 时，不需要 API）

```bash
R=<reader_train>; P=<vh_l1_levers>; V=<vrbench_pilot_v1>; A=$R/sft_mix/adapter
ADAPTER=$A KEEP_MERGED=1 L1GLOB="$P/narr_px_plus_full/stages/*/04_l1_example.json" IDS=$P/all_ids_1837.txt OUT=<out>/full_vh_mix sbatch scripts/launch/reader_eval.sbatch
ADAPTER=$A KEEP_MERGED=1 L1GLOB="$V/heldout60/derived/vrbench/*/derived/stages/*/04_l1_example.json" IDS=$V/heldout60/all_ids.txt OUT=<out>/vrb_mix sbatch scripts/launch/reader_eval.sbatch
ADAPTER=$A KEEP_MERGED=1 MAXLEN=65536 L1GLOB="$P/cg_narr_px_plus/stages/*/04_l1_example.json" IDS=$P/cg_ids_237.txt OUT=<out>/cg_mix sbatch scripts/launch/reader_eval.sbatch
```
期望：VH 51.3 ± 采样噪声（vLLM 默认解码；同一 adapter 两次全测差 < 1 点）、VRBench 75.4、CG 39.4；
对照行在 `l2_paper_cg_vh_20260901/full_vh_mm/full_narr_px_plus_noptr_1837.rows.jsonl`（235B 48.3）、
`vrbench_pilot_v1/heldout60/measure/rationale.rows.jsonl`（235B 71.7）、`cg_qa/cg_direct_rationale_237.rows.jsonl`（235B 45.5）。

## 8. B2 包（`b2://iclr2027-1/video-skills/<snapshot>/`）

```bash
b2 sync b2://iclr2027-1/video-skills/<snapshot>/ /你的路径/video-skills-snapshot/
cd /你的路径/video-skills-snapshot && sha256sum -c SHA256SUMS
git clone Video_Skills-all-refs.bundle Video_Skills && cd Video_Skills && git checkout cleanup/functional-layout-20260724
tar --zstd -xf reader-train-artifacts.tar.zst -C /你的路径/      # adapters、SFT 行、教师 rollouts
```
`README.zh.md` 列出包内每个归档、其源路径和大小；`MANIFEST.json` 是逐文件清单。
证据目录（B 档）与评测结果（C 档）按需另打 `catalogs-*.tar.zst` / `results-*.tar.zst`，同一前缀下。

## 9. 已知陷阱（都踩过）

- vLLM 直接挂 Qwen3.5 LoRA 输出塌缩 → 先合并；served-model-name 必须含 `qwen3`，客户端才发 `enable_thinking=false`。
- 训练提示要 `enable_thinking=False` 渲染，否则和推理模板前缀不一致。
- 全 logits 会 OOM → `sft_lora.py` 的 completion-only loss（`logits_to_keep`）。
- scavenger 抢占约 20 分钟一次：SFT 用 `--resume` + 小 `SAVE_STEPS`，评测用 attempt 文件续跑。
- `pkill -f <chain>` 会连自己的 shell 一起杀（exit 144）→ 按 pid 杀。
- OpenRouter 余额耗尽返回 402，rows 里只留 `HTTPError`；先查余额。
- 合并权重每个 17 GB，放项目树而不是 200 GB 的 scratch 配额下，评测完删。
