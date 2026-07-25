# Video Skills 训练器（SFT 之后）

本目录存放 Motif 门控的 OPD，以及带验证奖励的 GRPO / RLVR 代码。

## 框架选择（verl / ms-swift）

**首轮正式 GRPO 不引入 verl，也不把训练主循环迁到 ms-swift。**

| 选项 | 结论 | 原因 |
|---|---|---|
| 自定义 HF + PEFT | **采用** | 已有 SFT 栈、多 LoRA、字典序 verified reward、Motif dual-loop 都在本仓库；易审计 |
| FlashAttention-2 | **必须** | A6000 上 GRPO / LoRA 默认 `flash_attention_2`；缺包则 fail-closed |
| ms-swift | 不用作 GRPO 主框架 | 当前 venv 只借其 PyTorch；自定义 reward / Motif / 多 LoRA 不适配 |
| verl | 暂不采用 | 适合多机标准 scalar GRPO；我们的 env+字典序 reward+双环 Motif 改造成本过高 |

若以后要多机扩展，再评估把 **采样并行** 交给 verl，reward 仍走本仓库 `trainer.reward`。

## 目录结构

```text
trainer/
  # OPD / 闭环采集
  closed_loop_harness.py
  candidate_action_builder.py
  teacher_action_query.py
  opd_action_distill_adapter.py
  train_opd_kl.py
  collect_opd_*.py

  split_filter.py
  posttraining_manifest.py

  reward/
    milestone_ledger.py
    semantic_judge.py
    verified_reward.py
    bridge.py

  grpo/
    attn_utils.py           # FlashAttention-2 选择 / 校验
    model_runtime.py        # Qwen3.5-9B + PEFT + 序列 logprob
    live_rollout.py         # Motif 门控 live rollout_fn
    types.py
    isolation.py
    advantages.py
    collect_rollouts.py
    train_verified.py       # CPU smoke 或 --gpu FA2 训练

scripts/grpo/
  install_flash_attn.sh
  run_grpo_worker.sh
  submit_grpo_a6000.sh
```

## GRPO 模式

| 模式 | 更新模块 | 前置条件 |
|---|---|---|
| `l2_repair`（默认） | L2 + Repair LoRA | accelerate Motif；L1 冻结 |
| `joint_l1` | L1 + L2 + Repair | `--l2-stable`；L1 更小学习率 |

奖励字典序：

```text
(硬可行性, 终局成功, 可验证原子进度, 证据检查, -成本)
```

## A6000 启动

```bash
# 1) 安装 FlashAttention-2（在 GPU 节点上；submit 脚本默认会装）
sbatch --partition=gamma --account=gamma --gres=gpu:rtxa6000:1 --cpus-per-task=4 --mem=32G \
  --wrap 'bash /fs/gamma-projects/vlm-robot/Video_Skills/scripts/grpo/install_flash_attn.sh'

# 2) smoke：mock 采集 + GPU GRPO（强制 FA2）
bash scripts/grpo/submit_grpo_a6000.sh smoke

# 3) live 采集 + GPU 训练
LIVE=1 LIMIT=16 K=4 bash scripts/grpo/submit_grpo_a6000.sh all

# 4) 仅 GPU 训练（已有 collect 产物时，把 GROUPS 放到 OUTPUT_ROOT/collect）
STAGE=gpu_train bash scripts/grpo/submit_grpo_a6000.sh gpu_train
```

本地单元测试：

```bash
pytest tests/posttraining -q
```

## 正式跑检查清单

1. `.venv-qwen35-serve` 可 `import flash_attn`
2. L2 / Repair adapter 路径存在
3. `split_manifest_v1.json` 的 `grpo_pool` 可过滤到样本
4. live 模式需要 OpenRouter key（`--keys-py`）
5. 不要用 `--allow-sdpa-fallback` 做正式训练
