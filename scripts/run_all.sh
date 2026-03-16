#!/usr/bin/env bash
# ======================================================================
#  Co-Evolution: Launch training with dynamic GPU sharing.
#
#  Default mode (MANAGE_VLLM=1):
#    The Python orchestrator manages vLLM instances automatically:
#    - Before each rollout: starts 8 × TP=1 vLLM servers (all GPUs)
#    - Before GRPO training: kills vLLM, runs 8-GPU FSDP training
#    - After training: restarts vLLM with updated adapter weights
#
#  Legacy mode (MANAGE_VLLM=0):
#    Launches a single external vLLM server (GPUs 0-3, TP=4) and
#    trains on GPUs 4-7. Same as the old architecture.
#
#  Prerequisites:
#    conda activate game-ai-agent
#    pip install wandb tensorboard peft   # one-time
#
#  Usage:
#    bash scripts/run_all.sh
#
#    # Override settings via env vars:
#    VLLM_MODEL=Qwen/Qwen3-8B TOTAL_STEPS=50 bash scripts/run_all.sh
#
#    # Train from scratch (gaussian random LoRA init):
#    FROM_SCRATCH=1 bash scripts/run_all.sh
#
#    # Load pre-trained adapters:
#    LOAD_ADAPTERS_FROM=runs/prev_run/lora_adapters bash scripts/run_all.sh
#
#    # Resume a previous run:
#    RUN_DIR=runs/Qwen3-14B_20260315_143022 RESUME=1 bash scripts/run_all.sh
#
#    # Legacy mode (external vLLM):
#    MANAGE_VLLM=0 bash scripts/run_all.sh
# ======================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# ── Headless rendering ────────────────────────────────────────────────
export PYGLET_HEADLESS=1
export SDL_VIDEODRIVER=dummy

# ── HuggingFace cache (avoid re-downloading models) ──────────────────
export HF_HOME="${HF_HOME:-/workspace/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
mkdir -p "${HF_HUB_CACHE}"

# ── PYTHONPATH ────────────────────────────────────────────────────────
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/../GamingAgent:${PROJECT_ROOT}/../AgentEvolver:${PROJECT_ROOT}/../AI_Diplomacy:${PROJECT_ROOT}/../Orak:${PYTHONPATH:-}"

# ── Configurable parameters ──────────────────────────────────────────
MODEL="${VLLM_MODEL:-Qwen/Qwen3-14B}"
PORT="${VLLM_PORT:-8000}"
TP="${VLLM_TP:-4}"
GPU_UTIL="${VLLM_GPU_UTIL:-0.90}"
MANAGE_VLLM="${MANAGE_VLLM:-1}"

TOTAL_STEPS="${TOTAL_STEPS:-100}"
EPISODES="${EPISODES_PER_GAME:-8}"
CKPT_INTERVAL="${CKPT_INTERVAL:-5}"
WANDB_PROJECT="${WANDB_PROJECT:-game-ai-coevolution}"
RUN_DIR="${RUN_DIR:-}"
RESUME="${RESUME:-}"
FROM_SCRATCH="${FROM_SCRATCH:-}"
LOAD_ADAPTERS_FROM="${LOAD_ADAPTERS_FROM:-}"
DEBUG_IO="${DEBUG_IO:-}"
GPU_IDS="${GPU_IDS:-0 1 2 3 4 5 6 7}"

# ── Cleanup on exit (only for legacy mode) ────────────────────────────
VLLM_PID=""
cleanup() {
    echo ""
    echo "[run_all] Shutting down..."
    if [ -n "${VLLM_PID}" ] && kill -0 "${VLLM_PID}" 2>/dev/null; then
        echo "[run_all] Stopping vLLM server (PID ${VLLM_PID})..."
        kill "${VLLM_PID}" 2>/dev/null
        wait "${VLLM_PID}" 2>/dev/null || true
    fi
    echo "[run_all] Done."
}
trap cleanup EXIT INT TERM

# ======================================================================
# Phase 0: Ensure LoRA adapters exist (cold-start init)
# ======================================================================
echo "══════════════════════════════════════════════════════════════"
echo "  Co-Evolution Training"
echo "══════════════════════════════════════════════════════════════"
echo "  Model:         ${MODEL}"
echo "  Total steps:   ${TOTAL_STEPS}"
echo "  Eps/game:      ${EPISODES}"
echo "  Checkpoint:    every ${CKPT_INTERVAL} steps"
if [ "${MANAGE_VLLM}" = "1" ]; then
    echo "  GPU mode:      MANAGED (all GPUs shared, phase-swap)"
    echo "  GPUs:          ${GPU_IDS}"
else
    echo "  GPU mode:      LEGACY (vLLM TP=${TP} + separate GRPO GPUs)"
    echo "  vLLM port:     ${PORT}"
fi
if [ -n "${RUN_DIR}" ]; then
    echo "  Run dir:       ${RUN_DIR}"
fi
if [ -n "${FROM_SCRATCH}" ]; then
    echo "  Start mode:    FROM SCRATCH"
elif [ -n "${RESUME}" ]; then
    echo "  Start mode:    RESUME"
else
    echo "  Start mode:    AUTO"
fi
echo "══════════════════════════════════════════════════════════════"

echo ""
echo "[run_all] Ensuring LoRA adapters exist..."

RESOLVED_RUN_DIR=$(python -c "
import sys, os
os.environ.setdefault('PYGLET_HEADLESS', '1')
os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
from trainer.coevolution.config import CoEvolutionConfig, init_lora_adapters
cfg = CoEvolutionConfig(model_name='${MODEL}')
run_dir_override = '${RUN_DIR}'
if run_dir_override:
    cfg.run_dir = run_dir_override
force = bool('${FROM_SCRATCH}')
if force:
    cfg.start_mode = 'from_scratch'
cfg.resolve_paths()
created = init_lora_adapters(cfg, force=force)
if created:
    print(f'Created {len(created)} adapter(s) (gaussian init): {created}', file=sys.stderr)
else:
    print('All adapters already exist.', file=sys.stderr)
print(cfg.run_dir)
")

RUN_DIR="${RESOLVED_RUN_DIR}"
export ADAPTER_DIR="${RUN_DIR}/lora_adapters"
echo "[run_all] Run dir:     ${RUN_DIR}"
echo "[run_all] Adapter dir: ${ADAPTER_DIR}"

# ======================================================================
# Build training args
# ======================================================================
TRAIN_ARGS=(
    --total-steps "${TOTAL_STEPS}"
    --episodes-per-game "${EPISODES}"
    --checkpoint-interval "${CKPT_INTERVAL}"
    --model "${MODEL}"
    --wandb-project "${WANDB_PROJECT}"
    --run-dir "${RUN_DIR}"
)

if [ -n "${FROM_SCRATCH}" ]; then
    TRAIN_ARGS+=(--from-scratch)
elif [ -n "${RESUME}" ]; then
    TRAIN_ARGS+=(--resume)
fi

if [ -n "${LOAD_ADAPTERS_FROM}" ]; then
    TRAIN_ARGS+=(--load-adapters-from "${LOAD_ADAPTERS_FROM}")
fi

if [ -n "${DEBUG_IO}" ]; then
    TRAIN_ARGS+=(--debug-io)
fi

# ======================================================================
# Launch
# ======================================================================
if [ "${MANAGE_VLLM}" = "1" ]; then
    # ── Managed mode: orchestrator handles vLLM lifecycle ─────────
    # shellcheck disable=SC2086
    TRAIN_ARGS+=(--gpu-ids ${GPU_IDS})
    TRAIN_ARGS+=(--vllm-base-port "${PORT}")
    TRAIN_ARGS+=(--vllm-gpu-util "${GPU_UTIL}")

    echo ""
    echo "[run_all] Starting co-evolution (managed vLLM mode)..."
    python scripts/run_coevolution.py "${TRAIN_ARGS[@]}"
else
    # ── Legacy mode: external vLLM server ─────────────────────────
    TRAIN_ARGS+=(--no-manage-vllm)
    TRAIN_ARGS+=(--vllm-url "http://localhost:${PORT}/v1")

    echo ""
    echo "[run_all] Starting vLLM server (legacy mode)..."
    bash scripts/launch_vllm_coevolution.sh &
    VLLM_PID=$!
    echo "[run_all] vLLM server PID: ${VLLM_PID}"

    # Wait for vLLM health
    echo "[run_all] Waiting for vLLM at http://localhost:${PORT}..."
    MAX_WAIT=300
    WAITED=0
    while [ ${WAITED} -lt ${MAX_WAIT} ]; do
        if curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1 || \
           curl -sf "http://localhost:${PORT}/v1/models" >/dev/null 2>&1; then
            echo "[run_all] vLLM is ready! (waited ${WAITED}s)"
            break
        fi
        if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
            echo "[run_all] ERROR: vLLM server exited unexpectedly."
            exit 1
        fi
        sleep 5
        WAITED=$((WAITED + 5))
        if [ $((WAITED % 30)) -eq 0 ]; then
            echo "[run_all]   ... still waiting (${WAITED}s / ${MAX_WAIT}s)"
        fi
    done

    if [ ${WAITED} -ge ${MAX_WAIT} ]; then
        echo "[run_all] ERROR: vLLM did not start within ${MAX_WAIT}s"
        exit 1
    fi

    echo ""
    echo "[run_all] Starting co-evolution training (legacy mode)..."
    python scripts/run_coevolution.py "${TRAIN_ARGS[@]}"
fi

echo ""
echo "[run_all] Training complete."
