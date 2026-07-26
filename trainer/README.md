# Video Skills 训练器（SFT 之后）

本目录存放 Motif 门控的 OPD，以及带验证奖励的 GRPO / RLVR 代码。

## 框架 / 环境选择（verl / ms-swift / vllm）

**推荐专用 conda 环境 `video-skills-grpo`（HF/PEFT + FA2）。不装 verl / ms-swift / vllm。**

| 选项 | 结论 | 原因 |
|---|---|---|
| `conda/envs/video-skills-grpo` | **采用** | 干净隔离；torch2.6/cu124 + FA2 + transformers/peft |
| FlashAttention-2 | **必须** | Dao-AILab 预编译 wheel；缺包 fail-closed |
| `.venv-qwen35-serve` | 仅回退 | 旧 serve venv；worker 在新 env 不存在时才用 |
| vllm | spike 候选 | 仅当改成本地 policy 采样时；按 **8×A6000** 预算试 DP |
| ms-swift / verl | 待 spike | 文档有 vLLM GRPO / multi-turn；迁主环前先测吞吐与字典序 reward 编码 |

创建环境：

```bash
bash scripts/grpo/create_conda_env.sh
# 激活
source /fs/gamma-projects/vlm-robot/conda/etc/profile.d/conda.sh
conda activate /fs/gamma-projects/vlm-robot/conda/envs/video-skills-grpo
```

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

## A6000 启动（默认预算 8 卡）

gamma A6000 多为 **4 卡/节点** → 8 卡 ≈ 2 节点。QoS：`default`=1 GPU；`huge-long`≤8 GPU；`gamma-huge-long`≤16 GPU。

| 用途 | 建议占用 | 提交 |
|---|---|---|
| smoke | 1×A6000 | `PROFILE=smoke` 或 `bash ... smoke` |
| 正式 GRPO（现栈） | 8×A6000 | 默认非-smoke，或 `PROFILE=8gpu` |
| fan-out collect | 6×1 卡作业 + 1–2 卡 train | 见 `submit_grpo_a6000.sh` 注释 |
| 本地采样加速 spike | 4 rollout + 2 train + 2 备用 | 需另装 vLLM / ms-swift·verl 后再接 |

```bash
# smoke：1×A6000
bash scripts/grpo/submit_grpo_a6000.sh smoke

# 正式：8×A6000（huge-long，2 nodes）
PROFILE=8gpu LIVE=1 LIMIT=64 K=8 WALLTIME=12:00:00 \
  bash scripts/grpo/submit_grpo_a6000.sh all

# 或显式
NUM_GPUS=8 QOS=huge-long LIVE=1 bash scripts/grpo/submit_grpo_a6000.sh all

# 仅 GPU 训练
STAGE=gpu_train bash scripts/grpo/submit_grpo_a6000.sh gpu_train
```

本地单元测试：

```bash
pytest tests/posttraining -q
```

## 正式跑检查清单

1. `video-skills-grpo` 可 `import flash_attn`
2. L2 / Repair adapter 路径存在
3. `split_manifest_v1.json` 的 `grpo_pool` 可过滤到样本
4. live 模式需要 OpenRouter key（`--keys-py`）
5. 不要用 `--allow-sdpa-fallback` 做正式训练
