#!/usr/bin/env bash
# Submit per-specialist LoRA SFT with GPU-aware routing (prefer parallel).
# See: plans/video-skills-sft-gpu-plan.md
# Cluster note: cluster/gamma_umiacs_gpu_usage.md
#
# Usage:
#   bash scripts/sft_pilot/submit_five_lora_sft.sh smoke
#   bash scripts/sft_pilot/submit_five_lora_sft.sh pilot
#   SPECIALISTS="repair verifier motif" bash scripts/sft_pilot/submit_five_lora_sft.sh smoke
#
# Parallel pack under gamma-huge-long (one job, N GPUs):
#   PACK_GPUS=5 QOS=gamma-huge-long bash scripts/sft_pilot/submit_five_lora_sft.sh pack_smoke
#   PACK_GPUS=5 QOS=gamma-huge-long bash scripts/sft_pilot/submit_five_lora_sft.sh pack_pilot
#
# Override routing:
#   FORCE_PROFILE=a6000|l40s|a100|h100|h200|l40s_scav bash ...
#   NODELIST=csd00 FORCE_PROFILE=l40s_scav bash ...
#   QOS=gamma-huge-long FORCE_PROFILE=a6000 bash ...   # longer wall / higher limits
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/fs/gamma-projects/vlm-robot/Video_Skills}"
STAGE_ARG="${1:?usage: $0 smoke|pilot|pack_smoke|pack_pilot}"
SPECIALISTS="${SPECIALISTS:-l1 l2 repair verifier motif}"
PACKAGE_ROOT="${PACKAGE_ROOT:-${REPO_ROOT}/dataset_clip_wrapper/output/sft_cold_start/specialist_sft_v4/five_lora}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/dataset_clip_wrapper/output/sft_training/five_lora_${STAGE_ARG}_$(date +%Y%m%d)}"
LOG_ROOT="${LOG_ROOT:-${OUTPUT_ROOT}/slurm_logs}"
RUN_SCRIPT="${REPO_ROOT}/scripts/sft_pilot/run_lora_sft.sh"
FORCE_PROFILE="${FORCE_PROFILE:-}"
NODELIST="${NODELIST:-}"
QOS="${QOS:-}"
PACK_GPUS="${PACK_GPUS:-5}"
PACK_GRES="${PACK_GRES:-rtxa6000}"

mkdir -p "${LOG_ROOT}" "${OUTPUT_ROOT}"

if [[ ! -x "${RUN_SCRIPT}" && ! -f "${RUN_SCRIPT}" ]]; then
  echo "missing ${RUN_SCRIPT}" >&2
  exit 2
fi

# Prefer v4; fall back to v3 only for smoke wiring checks (not for real pilot).
if [[ ! -d "${PACKAGE_ROOT}" ]]; then
  FALLBACK="${REPO_ROOT}/dataset_clip_wrapper/output/sft_cold_start/specialist_sft_v3_20260722/five_lora"
  if [[ "${STAGE_ARG}" == smoke* && -d "${FALLBACK}" ]]; then
    echo "WARN: PACKAGE_ROOT missing; smoke falls back to v3 at ${FALLBACK}" >&2
    PACKAGE_ROOT="${FALLBACK}"
  else
    echo "PACKAGE_ROOT not found: ${PACKAGE_ROOT}" >&2
    echo "Build specialist_sft_v4 (sft_seed+dev_tune only) before pilot." >&2
    exit 2
  fi
fi

case "${STAGE_ARG}" in
  smoke|pilot) MODE=fanout; STAGE="${STAGE_ARG}" ;;
  pack_smoke) MODE=pack; STAGE=smoke ;;
  pack_pilot) MODE=pack; STAGE=pilot ;;
  *)
    echo "usage: $0 smoke|pilot|pack_smoke|pack_pilot" >&2
    exit 2
    ;;
esac

profile_for() {
  local specialist="$1"
  if [[ -n "${FORCE_PROFILE}" ]]; then
    echo "${FORCE_PROFILE}"
    return
  fi
  if [[ "${STAGE}" == "smoke" ]]; then
    echo "a6000"
    return
  fi
  case "${specialist}" in
    l1|l2) echo "l40s" ;;
    repair|verifier|motif) echo "a6000" ;;
    *) echo "a6000" ;;
  esac
}

slurm_flags_for() {
  local profile="$1"
  local qos_override="${2:-}"
  local base=""
  case "${profile}" in
    a6000)
      base="--partition=gamma --account=gamma --qos=default --gres=gpu:rtxa6000:1 --cpus-per-task=4 --mem=32G"
      ;;
    l40s)
      base="--partition=gamma --account=gamma --qos=medium --gres=gpu:l40s:1 --cpus-per-task=8 --mem=64G"
      ;;
    a100)
      base="--partition=scavenger --account=scavenger --qos=scavenger --gres=gpu:a100:1 --cpus-per-task=8 --mem=64G --requeue"
      ;;
    h100)
      base="--partition=scavenger --account=scavenger --qos=scavenger --gres=gpu:h100-nvl:1 --cpus-per-task=8 --mem=96G --requeue"
      ;;
    h200)
      base="--partition=scavenger --account=scavenger --qos=scavenger --gres=gpu:h200-sxm:1 --cpus-per-task=8 --mem=96G --requeue"
      ;;
    l40s_scav)
      base="--partition=scavenger --account=scavenger --qos=scavenger --gres=gpu:l40s:1 --cpus-per-task=4 --mem=32G --requeue"
      ;;
    *)
      echo "unknown profile: ${profile}" >&2
      exit 2
      ;;
  esac
  if [[ -n "${qos_override}" ]]; then
    # Replace existing --qos=... with override (gamma long packs / longer wall).
    base="$(echo "${base}" | sed -E "s/--qos=[^ ]+/--qos=${qos_override}/")"
  fi
  echo "${base}"
}

walltime_for() {
  local specialist="$1"
  if [[ "${STAGE}" == "smoke" ]]; then
    echo "00:45:00"
    return
  fi
  case "${specialist}" in
    l1) echo "1-00:00:00" ;;
    l2) echo "12:00:00" ;;
    *) echo "04:00:00" ;;
  esac
}

pack_walltime() {
  if [[ "${STAGE}" == "smoke" ]]; then
    echo "01:30:00"
  else
    echo "2-00:00:00"
  fi
}

echo "=== five-LoRA SFT submit ==="
echo "MODE=${MODE} STAGE=${STAGE}"
echo "PACKAGE_ROOT=${PACKAGE_ROOT}"
echo "OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "SPECIALISTS=${SPECIALISTS}"
[[ -n "${QOS}" ]] && echo "QOS_OVERRIDE=${QOS}"
echo

submitted=()

if [[ "${MODE}" == "pack" ]]; then
  # One gamma job with PACK_GPUS cards; launch one specialist per GPU in parallel.
  qos="${QOS:-gamma-huge-long}"
  # Map specialists to a list and cap by PACK_GPUS.
  read -r -a specs <<< "${SPECIALISTS}"
  n_specs="${#specs[@]}"
  if (( PACK_GPUS < n_specs )); then
    echo "PACK_GPUS=${PACK_GPUS} < specialists=${n_specs}; raising PACK_GPUS" >&2
    PACK_GPUS="${n_specs}"
  fi
  # CPU/MEM scale roughly with GPUs under gamma-huge-long (max 64CPU / 512G).
  cpus=$(( PACK_GPUS * 4 ))
  (( cpus > 64 )) && cpus=64
  mem=$(( PACK_GPUS * 32 ))
  (( mem > 512 )) && mem=512
  wall="$(pack_walltime)"
  pack_script="${LOG_ROOT}/pack_runner_${STAGE}_$$.sh"
  cat > "${pack_script}" <<EOF
#!/usr/bin/env bash
set -euo pipefail
SPECS=(${specs[*]})
REPO_ROOT="${REPO_ROOT}"
PACKAGE_ROOT="${PACKAGE_ROOT}"
OUTPUT_ROOT="${OUTPUT_ROOT}"
STAGE="${STAGE}"
RUN_SCRIPT="${RUN_SCRIPT}"
pids=()
idx=0
for specialist in "\${SPECS[@]}"; do
  export CUDA_VISIBLE_DEVICES="\${idx}"
  export STAGE SPECIALIST="\${specialist}"
  export DATA_ROOT="\${PACKAGE_ROOT}/\${specialist}"
  export OUTPUT_ROOT="\${OUTPUT_ROOT}/\${specialist}"
  export REPO_ROOT
  mkdir -p "\${OUTPUT_ROOT}"
  echo "[pack] launching \${specialist} on GPU \${idx}"
  bash "\${RUN_SCRIPT}" > "\${OUTPUT_ROOT}/pack_\${specialist}.log" 2>&1 &
  pids+=("\$!")
  idx=\$((idx + 1))
done
fail=0
for i in "\${!pids[@]}"; do
  if ! wait "\${pids[\$i]}"; then
    echo "[pack] FAILED \${SPECS[\$i]} pid=\${pids[\$i]}" >&2
    fail=1
  else
    echo "[pack] OK \${SPECS[\$i]}"
  fi
done
exit "\${fail}"
EOF
  chmod +x "${pack_script}"

  node_args=()
  if [[ -n "${NODELIST}" ]]; then
    node_args=(--nodelist="${NODELIST}")
  fi

  jobid="$(sbatch --parsable \
    --job-name="vs-sft-pack-${STAGE}" \
    --partition=gamma \
    --account=gamma \
    --qos="${qos}" \
    --gres="gpu:${PACK_GRES}:${PACK_GPUS}" \
    --cpus-per-task="${cpus}" \
    --mem="${mem}G" \
    --time="${wall}" \
    "${node_args[@]}" \
    --output="${LOG_ROOT}/pack-${STAGE}-%j.out" \
    --error="${LOG_ROOT}/pack-${STAGE}-%j.err" \
    --export=ALL \
    "${pack_script}")"

  echo "submitted PACK  gres=gpu:${PACK_GRES}:${PACK_GPUS}  qos=${qos}  job=${jobid}  wall=${wall}"
  echo "specialists: ${specs[*]}"
  submitted+=("pack:${jobid}:${PACK_GRES}x${PACK_GPUS}:${qos}")
else
  # Fan-out: one sbatch per specialist (default parallel path).
  for specialist in ${SPECIALISTS}; do
    train="${PACKAGE_ROOT}/${specialist}/train.jsonl"
    dev="${PACKAGE_ROOT}/${specialist}/dev.jsonl"
    if [[ ! -f "${train}" || ! -f "${dev}" ]]; then
      echo "SKIP ${specialist}: missing train/dev under ${PACKAGE_ROOT}/${specialist}" >&2
      continue
    fi
    profile="$(profile_for "${specialist}")"
    flags="$(slurm_flags_for "${profile}" "${QOS}")"
    wall="$(walltime_for "${specialist}")"
    out_dir="${OUTPUT_ROOT}/${specialist}"
    mkdir -p "${out_dir}"

    node_args=()
    if [[ -n "${NODELIST}" ]]; then
      node_args=(--nodelist="${NODELIST}")
    fi

    job_suffix="${JOB_NAME_SUFFIX:-}"
    job_name="vs-sft-${specialist}-${STAGE}${job_suffix}"
    # shellcheck disable=SC2086
    jobid="$(sbatch --parsable \
      --job-name="${job_name}" \
      ${flags} \
      "${node_args[@]}" \
      --time="${wall}" \
      --output="${LOG_ROOT}/${specialist}-${STAGE}-%j.out" \
      --error="${LOG_ROOT}/${specialist}-${STAGE}-%j.err" \
      --export="ALL,STAGE=${STAGE},SPECIALIST=${specialist},DATA_ROOT=${PACKAGE_ROOT}/${specialist},OUTPUT_ROOT=${out_dir},REPO_ROOT=${REPO_ROOT},L1_FULL=${L1_FULL:-0}" \
      "${RUN_SCRIPT}")"

    echo "submitted ${specialist}  profile=${profile}  job=${jobid}  wall=${wall}"
    submitted+=("${specialist}:${jobid}:${profile}")
  done
fi

echo
echo "=== summary ==="
printf '%s\n' "${submitted[@]:-none}"
echo
echo "Monitor: squeue -u \$USER"
echo "Parallel tips:"
echo "  PACK_GPUS=5 QOS=gamma-huge-long $0 pack_smoke"
echo "  FORCE_PROFILE=a100 SPECIALISTS='l1 l2' $0 pilot"
echo "  FORCE_PROFILE=h200 SPECIALISTS='l1' $0 pilot"
echo "  NODELIST=csd00 FORCE_PROFILE=l40s_scav $0 smoke"
