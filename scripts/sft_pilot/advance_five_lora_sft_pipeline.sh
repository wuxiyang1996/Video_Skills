#!/usr/bin/env bash
# Advance Video Skills five-LoRA SFT plan stages in sequence:
#   package_ready -> smoke -> baselines -> pilot -> gates -> verify -> done|blocked
#
# Idempotent: safe to call repeatedly (e.g. every 5m via agent loop).
# After verify, optionally submits L1 full-data substrate (L1_FULL=1 EPOCHS=1)
# when the pilot used capped L1 data. Does NOT auto-start OPD.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/fs/gamma-projects/vlm-robot/Video_Skills}"
PACKAGE_ROOT="${PACKAGE_ROOT:-${REPO_ROOT}/dataset_clip_wrapper/output/sft_cold_start/specialist_sft_v4/five_lora}"
SPLIT_MANIFEST="${SPLIT_MANIFEST:-${REPO_ROOT}/dataset_clip_wrapper/output/sft_cold_start/split_manifest_v1.json}"
TRAIN_ROOT="${REPO_ROOT}/dataset_clip_wrapper/output/sft_training"
SMOKE_ROOT="${SMOKE_ROOT:-${TRAIN_ROOT}/five_lora_smoke_20260725}"
VENV_PY="${VENV_PY:-${REPO_ROOT}/.venv-qwen35-serve/bin/python}"
SPECIALISTS=(l1 l2 repair verifier motif)

# Stable pipeline root: reuse today's dir if present.
if [[ -n "${PIPE_ROOT:-}" ]]; then
  :
elif latest="$(ls -d "${TRAIN_ROOT}/five_lora_pipeline_"* 2>/dev/null | sort | tail -n 1)"; then
  PIPE_ROOT="${latest}"
else
  PIPE_ROOT="${TRAIN_ROOT}/five_lora_pipeline_$(date +%Y%m%d)"
fi
STATE_PATH="${STATE_PATH:-${PIPE_ROOT}/pipeline_state.json}"

mkdir -p "${PIPE_ROOT}/logs" "${PIPE_ROOT}/baselines" "${PIPE_ROOT}/pilot" "${PIPE_ROOT}/gates"
cd "${REPO_ROOT}"

init_state_if_needed() {
  if [[ ! -f "${STATE_PATH}" ]]; then
    cat > "${STATE_PATH}" <<EOF
{
  "schema_version": "video-skills/five-lora-pipeline-v1",
  "stage": "package_ready",
  "pipe_root": "${PIPE_ROOT}",
  "smoke_root": "${SMOKE_ROOT}",
  "package_root": "${PACKAGE_ROOT}",
  "updated_at": null,
  "notes": []
}
EOF
  fi
}

stage_now() {
  "${VENV_PY}" -c "import json; print(json.load(open('${STATE_PATH}'))['stage'])"
}

set_stage() {
  local stage="$1"
  local note="${2:-}"
  STAGE="${stage}" NOTE="${note}" STATE_PATH="${STATE_PATH}" PIPE_ROOT="${PIPE_ROOT}" "${VENV_PY}" - <<'PY'
import json, os, time
path = os.environ["STATE_PATH"]
state = json.load(open(path))
state["stage"] = os.environ["STAGE"]
state["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
state["pipe_root"] = os.environ["PIPE_ROOT"]
note = os.environ.get("NOTE") or ""
if note:
    state.setdefault("notes", []).append(
        {"at": state["updated_at"], "stage": state["stage"], "note": note}
    )
open(path, "w").write(json.dumps(state, indent=2) + "\n")
print(state["stage"])
PY
}

count_active() {
  local pattern="$1"
  squeue -u "${USER}" -h -o '%j %T' 2>/dev/null \
    | awk -v p="${pattern}" '($1 ~ p) && ($2=="RUNNING" || $2=="PENDING") {c++} END{print c+0}'
}

smoke_status() {
  local specialist="$1"
  local report="${SMOKE_ROOT}/${specialist}/smoke/training_report.json"
  # Active job wins over a stale failed report (allows clean resubmits).
  if squeue -u "${USER}" -h -o '%j' 2>/dev/null | grep -qE "vs-sft-${specialist}-smoke"; then
    echo "running"
    return
  fi
  if [[ -f "${report}" ]]; then
    # Smoke = wiring check (train steps + adapter). Full JSON gates are for pilot.
    local adapter_dir="${SMOKE_ROOT}/${specialist}/smoke/adapter"
    if [[ -d "${adapter_dir}" ]]; then
      echo "passed"
      return
    fi
    echo "failed"
    return
  fi
  if ls "${SMOKE_ROOT}/slurm_logs/${specialist}-smoke-"*.err >/dev/null 2>&1; then
    echo "failed"
    return
  fi
  echo "missing"
}

ensure_package_gates() {
  local out="${PIPE_ROOT}/package_gates_report.json"
  "${VENV_PY}" -m dataset_clip_wrapper.training.evaluate_sft_package_gates \
    --package-root "${PACKAGE_ROOT}" \
    --split-manifest "${SPLIT_MANIFEST}" \
    --output "${out}"
  "${VENV_PY}" -c "import json; d=json.load(open('${out}')); raise SystemExit(0 if d['decision']['passed'] else 2)"
}

write_majority_baselines() {
  for specialist in "${SPECIALISTS[@]}"; do
    local dest="${PIPE_ROOT}/baselines/${specialist}"
    mkdir -p "${dest}"
    SPECIALIST="${specialist}" DEST="${dest}" PACKAGE_ROOT="${PACKAGE_ROOT}" "${VENV_PY}" - <<'PY'
from pathlib import Path
import os
from dataset_clip_wrapper.training.evaluate_lora_sft_gates import majority_action_baseline
from dataset_clip_wrapper.training.sft_common import write_json
specialist = os.environ["SPECIALIST"]
dev = Path(os.environ["PACKAGE_ROOT"]) / specialist / "dev.jsonl"
out = Path(os.environ["DEST"]) / "majority_baseline.json"
write_json(out, majority_action_baseline(dev))
print(out)
PY
  done
}

submit_base_baselines() {
  local log_root="${PIPE_ROOT}/logs"
  local part=scavenger account=scavenger qos=scavenger gres=gpu:l40s:1
  if ! sinfo -p scavenger -N -h -o '%N %T %G' 2>/dev/null | rg -q 'idle .*gpu:l40s'; then
    gres=gpu:rtxa6000:1
  fi
  for specialist in "${SPECIALISTS[@]}"; do
    local report="${PIPE_ROOT}/baselines/${specialist}/base_generation_report.json"
    [[ -f "${report}" ]] && { echo "base exists: ${specialist}"; continue; }
    if squeue -u "${USER}" -h -o '%j' 2>/dev/null | grep -qE "vs-base-${specialist}$"; then
      echo "base queued: ${specialist}"
      continue
    fi
    local out_dir="${PIPE_ROOT}/baselines/${specialist}"
    mkdir -p "${out_dir}"
    local jobid
    jobid="$(sbatch --parsable \
      --job-name="vs-base-${specialist}" \
      --partition="${part}" --account="${account}" --qos="${qos}" \
      --gres="${gres}" --cpus-per-task=4 --mem=32G --time=00:45:00 \
      --output="${log_root}/base-${specialist}-%j.out" \
      --error="${log_root}/base-${specialist}-%j.err" \
      --export="ALL,STAGE=base_baseline,SPECIALIST=${specialist},DATA_ROOT=${PACKAGE_ROOT}/${specialist},OUTPUT_ROOT=${out_dir},REPO_ROOT=${REPO_ROOT}" \
      "${REPO_ROOT}/scripts/sft_pilot/run_lora_sft.sh")"
    echo "submitted base ${specialist} -> ${jobid}"
  done
}

submit_pilots() {
  PACKAGE_ROOT="${PACKAGE_ROOT}" \
    OUTPUT_ROOT="${PIPE_ROOT}/pilot" \
    LOG_ROOT="${PIPE_ROOT}/logs" \
    bash "${REPO_ROOT}/scripts/sft_pilot/submit_five_lora_sft.sh" pilot
}

collect_gate_inputs() {
  local gates_root="${PIPE_ROOT}/gates/reports"
  mkdir -p "${gates_root}"
  for specialist in "${SPECIALISTS[@]}"; do
    local dest="${gates_root}/${specialist}"
    mkdir -p "${dest}"
    [[ -f "${PIPE_ROOT}/pilot/${specialist}/pilot/generation_report.json" ]] \
      && cp -f "${PIPE_ROOT}/pilot/${specialist}/pilot/generation_report.json" "${dest}/generation_report.json"
    [[ -f "${PIPE_ROOT}/baselines/${specialist}/base_generation_report.json" ]] \
      && cp -f "${PIPE_ROOT}/baselines/${specialist}/base_generation_report.json" "${dest}/base_generation_report.json"
    [[ -f "${PIPE_ROOT}/pilot/${specialist}/pilot/training_report.json" ]] \
      && cp -f "${PIPE_ROOT}/pilot/${specialist}/pilot/training_report.json" "${dest}/train_metrics.json"
  done
}

run_sft_gates() {
  collect_gate_inputs
  "${VENV_PY}" -m dataset_clip_wrapper.training.evaluate_lora_sft_gates \
    --reports-root "${PIPE_ROOT}/gates/reports" \
    --package-root "${PACKAGE_ROOT}" \
    --output "${PIPE_ROOT}/gates/lora_sft_gates_report.json"
}

run_sft_verify() {
  mkdir -p "${PIPE_ROOT}/verify"
  "${VENV_PY}" -m dataset_clip_wrapper.training.verify_sft_pilot_artifacts \
    --pipe-root "${PIPE_ROOT}" \
    --package-root "${PACKAGE_ROOT}" \
    --output "${PIPE_ROOT}/verify/sft_pilot_verify_report.json"
}

maybe_submit_l1_substrate() {
  local verify_json="${PIPE_ROOT}/verify/sft_pilot_verify_report.json"
  [[ -f "${verify_json}" ]] || return 0
  local capped
  capped="$("${VENV_PY}" -c "import json; print(json.load(open('${verify_json}')).get('l1_capped_pilot', False))")"
  [[ "${capped}" == "True" ]] || return 0
  if [[ -f "${PIPE_ROOT}/pilot_l1_full/l1/pilot/training_report.json" ]]; then
    echo "L1 substrate already present under pilot_l1_full"
    return 0
  fi
  if squeue -u "${USER}" -h -o '%j' 2>/dev/null | grep -qE 'vs-sft-l1-pilot-full'; then
    echo "L1 substrate job already queued/running; skip submit"
    return 0
  fi
  if [[ -f "${PIPE_ROOT}/verify/l1_substrate_submitted.json" ]]; then
    echo "L1 substrate submit marker present; skip"
    return 0
  fi
  echo "Submitting L1 full-data substrate (L1_FULL=1 EPOCHS=1) -> ${PIPE_ROOT}/pilot_l1_full"
  L1_FULL=1 EPOCHS=1 JOB_NAME_SUFFIX='-full' \
    NODELIST="${NODELIST:-csd00}" FORCE_PROFILE="${FORCE_PROFILE:-l40s_scav}" \
    SPECIALISTS='l1' \
    OUTPUT_ROOT="${PIPE_ROOT}/pilot_l1_full" \
    PACKAGE_ROOT="${PACKAGE_ROOT}" \
    bash "${REPO_ROOT}/scripts/sft_pilot/submit_five_lora_sft.sh" pilot \
    | tee "${PIPE_ROOT}/verify/l1_substrate_submit.log"
  "${VENV_PY}" - <<PY
import json, time
from pathlib import Path
Path("${PIPE_ROOT}/verify/l1_substrate_submitted.json").write_text(
    json.dumps({"at": time.strftime("%Y-%m-%dT%H:%M:%S"), "output_root": "${PIPE_ROOT}/pilot_l1_full", "l1_full": 1, "epochs": 1}, indent=2) + "\n"
)
PY
  set_stage done "SFT verify passed; L1 substrate submitted (full data, 1 epoch); stop before OPD"
}

init_state_if_needed
echo "=== advance_five_lora_sft_pipeline ==="
echo "PIPE_ROOT=${PIPE_ROOT}"
STAGE="$(stage_now)"
echo "STAGE=${STAGE}"
echo

# --- package_ready ---
if [[ "${STAGE}" == "package_ready" ]]; then
  if ensure_package_gates; then
    set_stage smoke "package gates passed"
    STAGE=smoke
  else
    set_stage blocked "package gates failed"
    exit 2
  fi
fi

# --- smoke ---
if [[ "${STAGE}" == "smoke" ]]; then
  passed=0; running=0; failed=0; missing=0
  for specialist in "${SPECIALISTS[@]}"; do
    s="$(smoke_status "${specialist}")"
    echo "smoke ${specialist}: ${s}"
    case "$s" in
      passed) passed=$((passed+1)) ;;
      running) running=$((running+1)) ;;
      failed) failed=$((failed+1)) ;;
      *) missing=$((missing+1)) ;;
    esac
  done
  if [[ "${missing}" -eq 5 && "$(count_active 'vs-sft-.*-smoke')" -eq 0 ]]; then
    echo "Submitting smokes..."
    NODELIST="${NODELIST:-csd00}" FORCE_PROFILE=l40s_scav \
      PACKAGE_ROOT="${PACKAGE_ROOT}" OUTPUT_ROOT="${SMOKE_ROOT}" \
      bash "${REPO_ROOT}/scripts/sft_pilot/submit_five_lora_sft.sh" smoke
    set_stage smoke "submitted smokes"
    exit 0
  fi
  if [[ "${passed}" -eq 5 ]]; then
    write_majority_baselines
    set_stage baselines "smoke all passed; majority baselines written"
    STAGE=baselines
  elif [[ "${failed}" -gt 0 && "${running}" -eq 0 ]]; then
    set_stage blocked "smoke failed (${failed} failed, ${passed} passed)"
    exit 2
  else
    set_stage smoke "waiting smoke passed=${passed} running=${running} failed=${failed} missing=${missing}"
    echo "Waiting on smoke (${passed}/5)"
    exit 0
  fi
fi

# --- baselines ---
if [[ "${STAGE}" == "baselines" ]]; then
  [[ -f "${PIPE_ROOT}/baselines/l1/majority_baseline.json" ]] || write_majority_baselines
  submit_base_baselines
  base_done=0
  for specialist in "${SPECIALISTS[@]}"; do
    [[ -f "${PIPE_ROOT}/baselines/${specialist}/base_generation_report.json" ]] && base_done=$((base_done+1))
  done
  active="$(count_active 'vs-base-')"
  echo "base baselines: ${base_done}/5 done, active=${active}"
  if [[ "${base_done}" -eq 5 ]]; then
    set_stage pilot "base baselines complete"
    STAGE=pilot
  elif [[ "${active}" -eq 0 ]]; then
    set_stage blocked "base baselines stuck at ${base_done}/5"
    exit 2
  else
    set_stage baselines "waiting base ${base_done}/5"
    exit 0
  fi
fi

# --- pilot ---
if [[ "${STAGE}" == "pilot" ]]; then
  need_submit=0
  for specialist in "${SPECIALISTS[@]}"; do
    if [[ ! -f "${PIPE_ROOT}/pilot/${specialist}/pilot/training_report.json" ]] \
       && ! squeue -u "${USER}" -h -o '%j' 2>/dev/null | grep -qE "vs-sft-${specialist}-pilot"; then
      need_submit=1
    fi
  done
  if [[ "${need_submit}" -eq 1 ]]; then
    if sinfo -p scavenger -N -h -o '%N %T %G' 2>/dev/null | rg -q '^csd00[[:space:]]+idle[[:space:]].*l40s'; then
      NODELIST=csd00 FORCE_PROFILE=l40s_scav submit_pilots
    else
      # Default routing: small on a6000, l1/l2 on l40s; override L1/L2 to a100 if useful later.
      submit_pilots
    fi
  fi
  pilot_done=0
  pilot_missing_idle=0
  for specialist in "${SPECIALISTS[@]}"; do
    if [[ -f "${PIPE_ROOT}/pilot/${specialist}/pilot/training_report.json" ]]; then
      echo "pilot ${specialist}: done"
      pilot_done=$((pilot_done+1))
    elif squeue -u "${USER}" -h -o '%j' 2>/dev/null | grep -qE "vs-sft-${specialist}-pilot"; then
      echo "pilot ${specialist}: active"
    else
      echo "pilot ${specialist}: missing"
      pilot_missing_idle=$((pilot_missing_idle+1))
    fi
  done
  active="$(count_active 'vs-sft-.*-pilot')"
  if [[ "${pilot_done}" -eq 5 ]]; then
    set_stage gates "all pilots finished"
    STAGE=gates
  elif [[ "${active}" -eq 0 && "${pilot_missing_idle}" -gt 0 ]]; then
    set_stage blocked "pilots stuck at ${pilot_done}/5"
    exit 2
  else
    set_stage pilot "waiting pilots ${pilot_done}/5 active=${active}"
    exit 0
  fi
fi

# --- gates ---
if [[ "${STAGE}" == "gates" ]]; then
  if run_sft_gates; then
    set_stage verify "SFT gates passed; running artifact verify"
    STAGE=verify
  else
    set_stage blocked "SFT gates failed"
    exit 2
  fi
fi

# --- verify ---
if [[ "${STAGE}" == "verify" ]]; then
  if run_sft_verify; then
    set_stage done "SFT verify passed; stop before OPD"
    maybe_submit_l1_substrate || true
    echo "DONE: SFT gates+verify passed. Report: ${PIPE_ROOT}/verify/sft_pilot_verify_report.json"
    echo "Next phase is Motif online wiring + OPD (not auto-started)."
    exit 0
  fi
  set_stage blocked "SFT verify failed"
  exit 2
fi

if [[ "${STAGE}" == "done" ]]; then
  echo "Already done: ${PIPE_ROOT}/gates/lora_sft_gates_report.json"
  [[ -f "${PIPE_ROOT}/verify/sft_pilot_verify_report.json" ]] && \
    echo "Verify: ${PIPE_ROOT}/verify/sft_pilot_verify_report.json"
  exit 0
fi

if [[ "${STAGE}" == "blocked" ]]; then
  echo "Pipeline blocked. See ${STATE_PATH}"
  exit 2
fi

echo "Unknown stage: ${STAGE}"
exit 2
