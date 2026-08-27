#!/usr/bin/env bash
# GRPO 环境结论与安装入口。
#
# 不需要新建 verl / ms-swift / vllm 专用环境来做首轮 GRPO：
#   - 训练：现有 .venv-qwen35-serve + HF/PEFT + FlashAttention-2
#   - 采集：Motif + OpenRouter planner（或后续本地 transformers serve）
#   - vllm：只在要高吞吐本地采样时再加；不是 verified reward / Motif 必需
#   - ms-swift：当前只借 PyTorch；不要把字典序 reward 迁进去
#   - verl：多机大规模再评估
#
# 本脚本只保证 FA2 可用。
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/fs/gamma-projects/vlm-robot/Video_Skills}"
bash "${REPO_ROOT}/scripts/grpo/install_flash_attn.sh"
echo "GRPO env ready: use existing .venv-qwen35-serve (no verl/ms-swift required)"
