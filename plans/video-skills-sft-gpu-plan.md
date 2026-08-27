# Video Skills：五 LoRA SFT 的 GPU 使用计划

Last updated: 2026-07-25

参考：[`cluster/gamma_umiacs_gpu_usage.md`](../../cluster/gamma_umiacs_gpu_usage.md)、
[`video-skills-posttraining-plan.md`](video-skills-posttraining-plan.md)。

## 设计原则

1. **优先并行**：五张 LoRA 彼此独立（`train_lora_sft.py` 单卡），默认同时开训，不要串行排队。
2. **卡型全开尝试**：`A6000 / L40S / A100 / H100 / H200` 都可提交；按「谁空就用谁」抢吞吐，不教条绑死一种卡。
3. **Gamma 长作业用 `gamma-huge-long`**：多卡并行包、长 wall（至 10d）、更高 CPU/MEM 上限走
   `--partition=gamma --account=gamma --qos=gamma-huge-long`。
4. **可靠 vs 抢占**：`gamma`（A6000/L40S）非抢占优先；`scavenger` 的 A100/H100/H200 可并行填空，但必须频繁 checkpoint + `--requeue`。
5. **小包别独占超大节点**：smoke / Repair / Verifier / Motif 仍优先 1×A6000；大卡留给 L1/L2 或并行 packed 作业。
6. 提交前用 `show_nodes` / `squeue` 复查；下面的 idle 快照会过期。

## QoS（gamma）

| QoS | Max wall | Max / job | 用途 |
|---|---|---|---|
| `default` | 3d | 1 GPU / 32G | 单卡短作业 |
| `medium` | 2d | 2 GPU / 64G | 单卡中等 / 双卡 |
| `high` | 1d | 4 GPU / 128G | 同节点 4 卡并行包 |
| `huge-long` | 10d | 8 GPU / 256G | 多卡并行（≤8） |
| **`gamma-huge-long`** | **10d** | **16 GPU / 512G** | **首选：五 LoRA 并行包、长训** |

单卡短 smoke 仍可用 `default`；**五路并行或长 pilot 优先 `gamma-huge-long`**（多卡 packed 或分作业提交时按账户并发限额选）。

## 当前可见资源（提交前请复检）

| 档位 | 位置 | 显存 | 访问 | 备注 |
|---|---|---|---|---|
| A6000 | `gammagpu[10-17]` | 48GB ×4 | `gamma` 可靠 | smoke / 小包首选 |
| L40S | `gammagpu[18-21]` | 48GB ×4 | `gamma` 可靠 | L1/L2 或 packed |
| L40S | `cml34` / `csd00` 等 | 48GB ×4–8 | `scavenger` | 并行填空（抢占） |
| A100 | `cml32` 等 | 80GB ×4 | `scavenger` | L1/L2 长 context |
| H100-NVL / SXM | `cml31` / `cml33` | 80GB | `scavenger` | 可 resume 长训 |
| H200-SXM | `cml35` / `cml36` | 141GB ×8 | `scavenger` | 空闲时抢；注意 drain/mix |

GRES 字符串：`rtxa6000` · `l40s` · `a100` · `h100-nvl` · `h100-sxm` · `h200-sxm`。

## 任务 → GPU 映射（尽量并行）

| 阶段 | 任务 | 推荐 | Partition / QoS | 并行策略 |
|---|---|---|---|---|
| P0 | split_manifest / v4 rebuild / package gates | CPU | login | 串行硬门 |
| P1 | 五路 tiny-overfit smoke | A6000×1 each 或 packed | `gamma` `default` **或** `gamma-huge-long`×5 | **5 作业同时** / 一作业 5 卡 |
| P1 | base-9B generation baseline | 同 smoke 卡 | 同上 | 可与 smoke 错开或并行 |
| P2 | **L1** full SFT | L40S / A100 / H100 / H200 ×1 | gamma L40S 或 scavenger | 与 L2、三小包并行 |
| P2 | **L2** full SFT（长 context） | L40S / A100 / H100 / H200 ×1 | 同上 | 与 L1 并行 |
| P2 | Repair / Verifier / Motif | A6000×1（或同节点 packed） | `gamma` | **三路同时** |
| P3 | SFT gates | 同训练卡或 A6000 | `gamma` | 五路评估可并行 |

### 并行模式（两种，优先能立刻跑起来的）

**模式 A — 多作业 fan-out（默认）**

```bash
# 五路同时提交；脚本按 specialist 选卡
bash scripts/sft_pilot/submit_five_lora_sft.sh smoke
bash scripts/sft_pilot/submit_five_lora_sft.sh pilot

# gamma fairshare 卡住时，扫 scavenger 空节点并行
NODELIST=csd00 FORCE_PROFILE=l40s_scav bash scripts/sft_pilot/submit_five_lora_sft.sh smoke
FORCE_PROFILE=a100 SPECIALISTS='l1 l2' bash scripts/sft_pilot/submit_five_lora_sft.sh pilot
FORCE_PROFILE=h200 SPECIALISTS='l1' bash scripts/sft_pilot/submit_five_lora_sft.sh pilot
```

**模式 B — `gamma-huge-long` 多卡 packed（同作业并行）**

申请 5–8 张 gamma GPU，作业内用 `CUDA_VISIBLE_DEVICES` 各起一张 LoRA：

```bash
PACK_GPUS=5 QOS=gamma-huge-long bash scripts/sft_pilot/submit_five_lora_sft.sh pack_smoke
PACK_GPUS=5 QOS=gamma-huge-long bash scripts/sft_pilot/submit_five_lora_sft.sh pack_pilot
```

Packed 适合：同类型卡（如 5×A6000 或 4×L40S+溢出）且希望一次占满、避免 fairshare 逐个 Priority。

### 不建议

- 五路串行「训完一张再训下一张」
- smoke 独占整节点 H200
- scavenger 长训却把 `save_steps` 设很大、无 `--requeue`
- 为单卡短作业申请 16 GPU（浪费配额）

## 推荐并行节奏

```text
Day 0 (CPU)
  split_manifest ✅ → specialist_sft_v4 → package gates

Day 0–1  PARALLEL
  5× smoke (+ baselines) on A6000 or gamma-huge-long pack
  若 gamma Priority：立刻改 scavenger L40S/A100/H100/H200 并行填空

Day 1–3  PARALLEL (不要等小包结束再开 L1)
  A6000×3: repair + verifier + motif full SFT
  L40S/A100/H100/H200×2: l1 + l2 full SFT
  或 1× gamma-huge-long packed 5–8 GPU 同跑

Day 3
  五路 evaluate_lora_sft_gates（可并行）
  过门者再进 OPD
```

## 资源请求模板

### A. Gamma A6000 单卡（smoke / 小包）

```bash
#SBATCH --partition=gamma
#SBATCH --account=gamma
#SBATCH --qos=default
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
```

### B. Gamma L40S 单卡（L1/L2）

```bash
#SBATCH --partition=gamma
#SBATCH --account=gamma
#SBATCH --qos=medium
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
```

### C. Gamma `gamma-huge-long` 多卡并行包（首选并行）

```bash
#SBATCH --partition=gamma
#SBATCH --account=gamma
#SBATCH --qos=gamma-huge-long
#SBATCH --gres=gpu:rtxa6000:5    # or l40s:4 / mix via nodelist
#SBATCH --cpus-per-task=20
#SBATCH --mem=160G
#SBATCH --time=2-00:00:00
```

### D. Scavenger A100 / H100 / H200（抢占，需 resume）

```bash
#SBATCH --partition=scavenger
#SBATCH --account=scavenger
#SBATCH --qos=scavenger
#SBATCH --gres=gpu:a100:1        # or h100-nvl:1 / h100-sxm:1 / h200-sxm:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G                # H200 可提到 96G
#SBATCH --time=12:00:00
#SBATCH --requeue
```

## 与五 LoRA 数据量的匹配

| LoRA | train 规模（v3 参考） | 预估 wall（1×48–80GB） | 优先卡型 |
|---|---:|---|---|
| L1 | ~12k | 最长 | L40S / A100 / H100 / H200 |
| L2 | ~0.7k，context 长 | 中等偏长 | L40S / A100 / H100 / H200 |
| Repair | ~0.1k | 短 | A6000 |
| Verifier | ~0.08k | 短 | A6000 |
| Motif | ~0.26k | 短 | A6000 |

## 操作脚本

按计划自动推进（`package_ready → smoke → baselines → pilot → gates → verify → done`；不过 OPD）：

- L1 默认 capped pilot（1536×1 epoch）；verify 后若检测到 capped，自动提交 `L1_FULL=1 EPOCHS=1` substrate 到 `pilot_l1_full/`。
- 单动作族（verifier/motif）跳过 “beat majority” 检查。
- 报告：`gates/lora_sft_gates_report.json`、`verify/sft_pilot_verify_report.json`。

```bash
bash scripts/sft_pilot/advance_five_lora_sft_pipeline.sh
```

状态：`dataset_clip_wrapper/output/sft_training/five_lora_pipeline_*/pipeline_state.json`

手动单阶段：

```bash
# 默认：五路 fan-out 并行
bash scripts/sft_pilot/submit_five_lora_sft.sh smoke
bash scripts/sft_pilot/submit_five_lora_sft.sh pilot

# gamma-huge-long packed
PACK_GPUS=5 QOS=gamma-huge-long bash scripts/sft_pilot/submit_five_lora_sft.sh pack_smoke

# 指定卡型并行
FORCE_PROFILE=h100 SPECIALISTS='l1 l2' bash scripts/sft_pilot/submit_five_lora_sft.sh pilot
FORCE_PROFILE=h200 SPECIALISTS='l1' bash scripts/sft_pilot/submit_five_lora_sft.sh pilot
```

脚本按 specialist 自动选 GRES；提交前仍应 `show_nodes` 确认 idle，并尽量一次提交多路。
