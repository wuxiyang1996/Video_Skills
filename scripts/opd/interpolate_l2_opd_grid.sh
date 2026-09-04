#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/fs/gamma-projects/vlm-robot/Video_Skills}"
BASE_ADAPTER="${BASE_ADAPTER:?set BASE_ADAPTER}"
TUNED_ADAPTER="${TUNED_ADAPTER:?set TUNED_ADAPTER}"
OUTPUT_ROOT="${OUTPUT_ROOT:?set OUTPUT_ROOT}"
# Prefer whitespace here: commas inside sbatch --export are parsed as separators and
# can silently truncate ALPHAS to its first value.
ALPHAS="${ALPHAS:-0.25 0.50 0.75}"
PYTHON="${PYTHON:-/fs/gamma-projects/vlm-robot/conda/envs/video-skills-grpo/bin/python}"

cd "${REPO_ROOT}"
normalized_alphas="${ALPHAS//,/ }"
normalized_alphas="${normalized_alphas//;/ }"
read -r -a alpha_values <<< "${normalized_alphas}"
if [[ "${#alpha_values[@]}" -eq 0 ]]; then
  echo "ERROR: ALPHAS did not contain any values" >&2
  exit 2
fi
for alpha in "${alpha_values[@]}"; do
  suffix="$("${PYTHON}" -c 'import sys; print(f"{float(sys.argv[1]):.2f}".replace(".", ""))' "${alpha}")"
  "${PYTHON}" scripts/posttraining/interpolate_lora_adapters.py \
    --base-adapter "${BASE_ADAPTER}" \
    --tuned-adapter "${TUNED_ADAPTER}" \
    --output-adapter "${OUTPUT_ROOT}/alpha${suffix}/adapter" \
    --alpha-tuned "${alpha}"
  for required in adapter_config.json adapter_model.safetensors interpolation_report.json; do
    if [[ ! -s "${OUTPUT_ROOT}/alpha${suffix}/adapter/${required}" ]]; then
      echo "ERROR: incomplete interpolated adapter alpha=${alpha}: missing ${required}" >&2
      exit 2
    fi
  done
done
