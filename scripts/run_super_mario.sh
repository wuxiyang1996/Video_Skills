#!/usr/bin/env bash
# ======================================================================
#  Super Mario co-evolution training — bootstrapped from a previous
#  checkpoint.
#
#  Loads trained LoRA adapters from a prior co-evolution run (default:
#  step_0034 of the Qwen3-8B_20260321_041333 run) and trains on the
#  Super Mario environment via a subprocess env.
#
#  The training loop runs in the game-ai-agent conda env (NumPy 2.x,
#  vLLM, PEFT, etc.).  Super Mario's NES emulator (gym_super_mario_bros
#  + nes_py) needs NumPy 1.x, so it runs in a separate orak-mario
#  subprocess.  The SubprocessEnv wrapper bridges the two over JSON
#  pipes.
#
#  Usage:
#    bash scripts/run_super_mario.sh
#
#    # Override checkpoint source:
#    CKPT_STEP=step_0029 bash scripts/run_super_mario.sh
#
#    # Also keep training Tetris alongside:
#    EXTRA_GAMES="tetris" bash scripts/run_super_mario.sh
#
#    # Custom step count:
#    TOTAL_STEPS=50 bash scripts/run_super_mario.sh
#
#    # Use a different Python for the Mario subprocess:
#    ORAK_PYTHON=/path/to/python bash scripts/run_super_mario.sh
# ======================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# ── Headless rendering ────────────────────────────────────────────────
export PYGLET_HEADLESS=1
export SDL_VIDEODRIVER=dummy

# Super Mario (NES via stable-retro) needs a framebuffer.
if [ -z "${DISPLAY:-}" ]; then
    if command -v Xvfb &>/dev/null; then
        XVFB_DISPLAY=":99"
        if ! pgrep -f "Xvfb ${XVFB_DISPLAY}" &>/dev/null; then
            echo "[super_mario] Starting Xvfb on ${XVFB_DISPLAY}..."
            Xvfb "${XVFB_DISPLAY}" -screen 0 1024x768x24 &>/dev/null &
            sleep 1
        fi
        export DISPLAY="${XVFB_DISPLAY}"
    else
        echo "[WARN] No DISPLAY set and Xvfb not found — NES rendering may fail."
    fi
fi

# ── Subprocess env (orak-mario) ──────────────────────────────────────
# The Mario NES emulator runs in a separate conda env to avoid the
# NumPy 2.x vs 1.x conflict.  ORAK_PYTHON tells SubprocessEnv which
# interpreter to use for the game subprocess.
export ORAK_PYTHON="${ORAK_PYTHON:-/workspace/miniconda3/envs/orak-mario/bin/python}"

if [ ! -x "${ORAK_PYTHON}" ]; then
    echo "[ERROR] orak-mario Python not found at: ${ORAK_PYTHON}"
    echo "  Create it with:  conda create -n orak-mario python=3.11 && conda run -n orak-mario pip install gym-super-mario-bros nes-py"
    exit 1
fi
echo "[super_mario] Subprocess env Python: ${ORAK_PYTHON}"

# ── HuggingFace cache ────────────────────────────────────────────────
export HF_HOME="${HF_HOME:-/workspace/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
mkdir -p "${HF_HUB_CACHE}"

# ── PYTHONPATH ────────────────────────────────────────────────────────
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/../GamingAgent:${PROJECT_ROOT}/../AgentEvolver:${PROJECT_ROOT}/../AI_Diplomacy:${PROJECT_ROOT}/../Orak:${PYTHONPATH:-}"

# ── Source checkpoint (adapters + skill banks) ────────────────────────
SOURCE_RUN="${SOURCE_RUN:-runs/Qwen3-8B_20260321_041333}"
CKPT_STEP="${CKPT_STEP:-step_0034}"
CKPT_DIR="${SOURCE_RUN}/checkpoints/${CKPT_STEP}"

if [ ! -d "${CKPT_DIR}" ]; then
    echo "[ERROR] Checkpoint not found: ${CKPT_DIR}"
    echo "  Available checkpoints:"
    ls "${SOURCE_RUN}/checkpoints/" 2>/dev/null || echo "  (none)"
    exit 1
fi

# ── Configurable parameters ──────────────────────────────────────────
MODEL="${VLLM_MODEL:-Qwen/Qwen3-8B}"
PORT="${VLLM_PORT:-8000}"
GPU_UTIL="${VLLM_GPU_UTIL:-0.90}"
VLLM_GPUS="${VLLM_GPUS:-0 1 2 3}"
GRPO_GPUS="${GRPO_GPUS:-4 5 6 7}"
SPEC_MODEL="${SPEC_MODEL:-Qwen/Qwen3-0.6B}"
SPEC_TOKENS="${SPEC_TOKENS:-5}"

TOTAL_STEPS="${TOTAL_STEPS:-40}"
EPISODES="${EPISODES_PER_GAME:-6}"
CKPT_INTERVAL="${CKPT_INTERVAL:-5}"
WANDB_PROJECT="${WANDB_PROJECT:-game-ai-coevolution}"
DEBUG_IO="${DEBUG_IO:-}"

# Games: Super Mario, plus any extras the user wants.
# Super Mario uses the Orak environment path.
EXTRA_GAMES="${EXTRA_GAMES:-}"
GAMES="super_mario ${EXTRA_GAMES}"

# ── Banner ────────────────────────────────────────────────────────────
echo "══════════════════════════════════════════════════════════════"
echo "  Super Mario Co-Evolution Training (subprocess env)"
echo "══════════════════════════════════════════════════════════════"
echo "  Source run:    ${SOURCE_RUN}"
echo "  Checkpoint:    ${CKPT_STEP}"
echo "  Model:         ${MODEL}"
echo "  Games:         ${GAMES}"
echo "  Total steps:   ${TOTAL_STEPS}"
echo "  Eps/game:      ${EPISODES}"
echo "  Checkpoint:    every ${CKPT_INTERVAL} steps"
echo "  vLLM GPUs:     ${VLLM_GPUS}"
echo "  GRPO GPUs:     ${GRPO_GPUS}"
echo "  Spec decode:   ${SPEC_MODEL} (${SPEC_TOKENS} tokens)"
echo "  DISPLAY:       ${DISPLAY:-<unset>}"
echo "  ORAK_PYTHON:   ${ORAK_PYTHON}"
echo "══════════════════════════════════════════════════════════════"

# ── Resolve adapter paths from checkpoint ─────────────────────────────
DECISION_ADAPTERS="${CKPT_DIR}/adapters/decision"
SKILLBANK_ADAPTERS="${CKPT_DIR}/adapters/skillbank"
SEED_BANK="${CKPT_DIR}/banks"

echo ""
echo "[super_mario] Loading adapters from ${CKPT_STEP}:"
echo "  Decision:  ${DECISION_ADAPTERS}"
echo "  SkillBank: ${SKILLBANK_ADAPTERS}"
echo "  Seed bank: ${SEED_BANK}"

for adapter_dir in \
    "${DECISION_ADAPTERS}/skill_selection" \
    "${DECISION_ADAPTERS}/action_taking" \
    "${SKILLBANK_ADAPTERS}/segment" \
    "${SKILLBANK_ADAPTERS}/contract" \
    "${SKILLBANK_ADAPTERS}/curator"; do
    if [ ! -f "${adapter_dir}/adapter_config.json" ]; then
        echo "[ERROR] Missing adapter: ${adapter_dir}"
        exit 1
    fi
done
echo "[super_mario] All 5 adapters verified."

# ── Ensure LoRA adapters are copied into the new run ──────────────────
echo ""
echo "[super_mario] Initialising new run..."

RESOLVED_RUN_DIR=$(python -c "
import sys, os
os.environ.setdefault('PYGLET_HEADLESS', '1')
os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
from trainer.coevolution.config import CoEvolutionConfig, init_lora_adapters
cfg = CoEvolutionConfig(
    model_name='${MODEL}',
    pretrained_adapter_paths={
        'skill_selection': '${DECISION_ADAPTERS}/skill_selection',
        'action_taking': '${DECISION_ADAPTERS}/action_taking',
        'segment': '${SKILLBANK_ADAPTERS}/segment',
        'contract': '${SKILLBANK_ADAPTERS}/contract',
        'curator': '${SKILLBANK_ADAPTERS}/curator',
    },
)
cfg.resolve_paths()
created = init_lora_adapters(cfg)
if created:
    print(f'Loaded {len(created)} adapter(s) from checkpoint', file=sys.stderr)
else:
    print('All adapters already exist.', file=sys.stderr)
print(cfg.run_dir)
")

RUN_DIR="${RESOLVED_RUN_DIR}"
echo "[super_mario] New run dir: ${RUN_DIR}"

# ── Build training args ───────────────────────────────────────────────
# shellcheck disable=SC2086
TRAIN_ARGS=(
    --total-steps "${TOTAL_STEPS}"
    --games ${GAMES}
    --curriculum none
    --episodes-per-game "${EPISODES}"
    --checkpoint-interval "${CKPT_INTERVAL}"
    --model "${MODEL}"
    --wandb-project "${WANDB_PROJECT}"
    --wandb-run-name "super-mario-from-${CKPT_STEP}"
    --run-dir "${RUN_DIR}"
    --load-decision-adapters "${DECISION_ADAPTERS}"
    --load-skillbank-adapters "${SKILLBANK_ADAPTERS}"
    --seed-bank-dir "${SEED_BANK}"
    --vllm-gpus ${VLLM_GPUS}
    --grpo-devices ${GRPO_GPUS}
    --vllm-base-port "${PORT}"
    --vllm-gpu-util "${GPU_UTIL}"
    --speculative-model "${SPEC_MODEL}"
    --num-speculative-tokens "${SPEC_TOKENS}"
)

if [ -n "${DEBUG_IO}" ]; then
    TRAIN_ARGS+=(--debug-io)
fi

# ── Launch ────────────────────────────────────────────────────────────
echo ""
echo "[super_mario] Starting co-evolution (Super Mario)..."
echo "[super_mario] Command: python scripts/run_coevolution.py ${TRAIN_ARGS[*]}"
echo ""

python scripts/run_coevolution.py "${TRAIN_ARGS[@]}"

echo ""
echo "[super_mario] Training complete."
