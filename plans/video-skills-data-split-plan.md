# Video Skills：数据划分与 Dataset 角色计划

Last updated: 2026-07-24

相关计划：[posttraining 计划](video-skills-posttraining-plan.md)。  
本地数据根目录：`/fs/gamma-projects/vlm-robot/datasets`。

## 设计结论

按**角色分层**划分数据，而不是按「磁盘上有没有 JSON」硬切。  
**split 键 = source video**：同一视频的所有 QA、L1 cache、trajectory、transition、motif source 只能落在一个角色。

```text
Train substrate（可产生监督）
  Video-Holmes official train
  + CG-Bench 项目侧 video-level train

Dev / tune（阈值、motif、early stop）
  从上述 train 视频再划一刀（仍按 video）

Final report（论文数字）
  Video-Holmes official test
  + CG-Bench frozen video-level test
  + VRBench / Video-MME / OVO / StreamingBench 作 OOD / streaming
```

Video-Holmes **偏难但不丢**：早期不作 L2/RL 主战场，作 curriculum 上层 + social/abstain stress；冷启动主 ramp 用 CG-Bench。

## 1. 工作区可用 benchmark 与角色

| 数据集 | 本地 | 官方 split | Video_Skills 角色 | 备注 |
|---|---|---|---|---|
| **Video-Holmes** | ✅ | train / test | SFT/GRPO 硬课 + 终评 | 适配器真正按 split 读文件；社交/因果多跳，偏难 |
| **CG-Bench** | ✅ 全量 | **无**（单 JSON） | 冷启动 / L2 主粮 | 必须项目侧按 `video_id` 冻 train/dev/test；clue 仅 hidden |
| **VRBench** | ✅ | 仅 eval JSONL | **eval-only / OOD** | 勿进 SFT；adapter 默认 `split=train` 是误导标签 |
| **Video-MME** | ✅ 全量 | 基本 test | **held-out eval** | 适配器现用 `streambridge_tiny`；全量评测需另接路径 |
| **OVO-Bench** | ✅ 全量 | 无 train | **streaming eval** | 同上，勿与 tiny fixture 混报 |
| **StreamingBench** | ✅ | 任务 CSV | streaming eval | 尚无 clip-wrapper 适配器 |
| **SIV-Bench** | ❌ | — | 暂不进协议 | 适配器存在但数据未落地 |
| **M3-Bench** | 仅部分标注 | — | 暂缓 | 需 graph reader |

当前 five_lora SFT 只吃了 **CG + Video-Holmes**；`build_sft_splits.py` 已把 `vrbench` / `ovo_bench` / `videomme` 标为 evaluation-only——保持。

## 2. 三角色 + train 内再切

### 2.1 顶层角色

| 角色 | 用途 | 数据来源 |
|---|---|---|
| `train_substrate` | 一切可产生监督的视频池 | VH official train + CG video-level train |
| `dev_tune` | 阈值、motif promotion、early stop、reward 权重 | 从 train_substrate 再划（按 video） |
| `heldout_test` | 论文终评 | VH official test + CG frozen video test + VRBench/Video-MME/OVO/(StreamingBench) |

### 2.2 Train 视频内角色（给 OPD/RL 留坑）

在 `train_substrate` 视频上再切：

| 角色 | 用途 |
|---|---|
| `sft_seed` | cold-start BC（现有 five_lora / 后续细粒度 SFT） |
| `opd_pool` | student on-policy states + teacher complete-action prior |
| `grpo_pool` | verified RL 采样；**SFT/OPD 未见过的视频** |
| `dev_tune` | 仅调参，不报终局论文数字 |

硬规矩：

- 绝不把同一 trajectory 的 transition 拆进 train 与 dev。
- CG clue interval / gold answer 只做离线过滤与 evaluator，不进 `state_t`。
- VRBench / Video-MME / OVO / StreamingBench **零训练行**。
- 落盘不可变 `split_manifest.json`：`video_id → {dataset, role, ...}`；SFT / OPD / GRPO / 终评共读同一份。

## 3. Video-Holmes 难度与用量

VH 对当前 pipeline **偏难但有用**：

- 任务是社交/因果/时序多跳，证据稀疏，不像 CG「找线索片段」可定位。
- 现有数据：L1 里 VH 多（构图可行），但 L2 core / verifier 正样本几乎是 CG；难例里常有 `video_holmes` abstain。
- 短视频即使 L1 高质，选项仍可能分不开，需要 question-conditioned reinspection。

| 阶段 | Video-Holmes | CG-Bench |
|---|---|---|
| SFT cold-start（尤其 L2/Repair） | 少而精的 gated 正例 + 大量 abstain/负例 | **主粮** |
| OPD / 首轮 GRPO | 小比例 hard pool，或暂缓 | **主优化对象** |
| 终评 / 论文 | 必报（能力上限 + social） | 必报（可定位证据） |

不建议：完全踢掉 VH；或只用 VH 当 L2/RL 主成功门（reward 过稀）。

## 4. 建议比例（可调，先写进 manifest）

比例是 **video 级** 目标，不是 QA 行数硬拷贝。

**Video-Holmes（官方 train ≈ 233 videos / 1551 QA）**

- 内部 `dev_tune`：约 10–15% train videos  
- 其余 train videos：再切 `sft_seed` / `opd_pool` / `grpo_pool`（例如 50% / 25% / 25%，可按规模改）  
- `test`：官方 test 全锁，仅终评  

**CG-Bench（≈ 1219 source videos）**

- 先冻 `heldout_test`：约 15–20% videos（不可变）  
- 剩余：`dev_tune` 约 10–15%，其余进 train 内三角色  
- `cgbench_mini` 只用于 pilot，不进最终 split 声明  

**Eval suite（零训练）**

- VRBench、Video-MME（全量路径）、OVO-Bench、StreamingBench：只报 zero-shot / transfer  
- 与 `streambridge_tiny` fixture 分开记账，禁止混报  

## 5. 落地任务

1. ✅ 新增 [`build_split_manifest.py`](../dataset_clip_wrapper/training/build_split_manifest.py)：
   - 输入：VH train/test、CG 全量 video 列表、固定 salt  
   - 输出：`split_manifest_v1.json` + summary；含 `manifest_hash`  
   - 字段：`dataset`, `video_id`, `role`, `n_questions`, `question_ids`  
2. ✅ 新增 [`evaluate_sft_package_gates.py`](../dataset_clip_wrapper/training/evaluate_sft_package_gates.py) 与 [`evaluate_lora_sft_gates.py`](../dataset_clip_wrapper/training/evaluate_lora_sft_gates.py) 作为 SFT 前/后质量门。  
3. 下一步：用 manifest 过滤/重建 `specialist_sft_v4`；[`build_specialist_sft_v3.py`](../scripts/sft_pilot/build_specialist_sft_v3.py) 与后续 OPD/GRPO collector **强制**读同一 manifest，角色不符则 fail。  
4. 修正 VRBench（及同类）metadata：eval 文件不得 stamp `split=train`。  
5. 为全量 Video-MME / OVO 增加独立 eval 入口，与 tiny adapter 分离。  
6. 在 posttraining 采样器中按数据集难度加权：L2/Repair 早期 **CG ≫ VH**；VH 提高 abstain / hard-negative 权重。

退出门：

- role × video 零重叠  
- eval-only 数据集训练行数 = 0  
- CG heldout video 列表 hash 固定、可复现  
- VH official test 从未出现在任何 SFT/OPD/GRPO 产物  
- 报告中 tiny fixture 与 full benchmark 分列  

## 6. 与 posttraining 计划的接口

[`video-skills-posttraining-plan.md`](video-skills-posttraining-plan.md) §1 的 `sft_seed` / `opd_pool` / `grpo_pool` / `dev_tune` / `heldout_test` **直接消费本计划的 split_manifest**。  
本计划不定义 reward / OPD 算法；只定义「哪些视频能进哪条流水线」。
