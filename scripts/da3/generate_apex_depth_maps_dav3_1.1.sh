#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# APEX Depth Map Generation - Depth Anything 3 (DA3) Large v1.1
# For internal R&D / non-commercial use only (CC BY-NC 4.0 models).
#
# Default export: mini_npz (depth + conf)
# Default model:  DA3-LARGE-1.1 (or equivalent alias supported by lux_depth_v3)
#
# Usage:
#   export DA3_NONCOMMERCIAL_ACK=1
#   ./generate_apex_depth_maps_dav3_1.1.sh
#
# Optional overrides:
#   REPO_ROOT=/path/to/Transformation_Portal
#   OUTPUT_DIR=/path/to/output
#   DA3_DEVICE=auto|cuda|mps|cpu
#   DA3_PROCESS_RES=2048
#   DA3_EXPORT_FORMAT=mini_npz|mini_npz-glb|...
#   DA3_MODEL=<depends on CLI flags; script auto-detects flag>
#   PYTHON_BIN=python3
# ==============================================================================

if [[ "${DA3_NONCOMMERCIAL_ACK:-0}" != "1" ]]; then
  echo "❌ DA3 license gate: set DA3_NONCOMMERCIAL_ACK=1 to confirm non-commercial R&D usage." >&2
  echo "   Note: DA3-LARGE-1.1 / DA3NESTED-GIANT-LARGE-1.1 are CC BY-NC 4.0 (non-commercial)." >&2
  exit 2
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  PYTHON_BIN="python3"
fi
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "❌ python not found (tried: python, python3)" >&2
  exit 1
fi

# Resolve repo root robustly (works whether you run from repo root or elsewhere)
if [[ -z "${REPO_ROOT:-}" ]]; then
  if command -v git >/dev/null 2>&1 && git rev-parse --show-toplevel >/dev/null 2>&1; then
    REPO_ROOT="$(git rev-parse --show-toplevel)"
  else
    REPO_ROOT="$(pwd)"
  fi
fi

OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/750Picacho_Depth_Maps_DAV3_1.1_APEX}"
LOG_DIR="${LOG_DIR:-${OUTPUT_DIR}/logs}"
TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"
LOG_FILE="${LOG_DIR}/depth_generation_dav3_1.1_apex_${TIMESTAMP}.log"

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

# Preflight: lux_depth_v3 must be importable
if ! "${PYTHON_BIN}" -c "import lux_depth_v3" >/dev/null 2>&1; then
  echo "❌ lux_depth_v3 is not importable in this Python environment." | tee "${LOG_FILE}"
  echo "   Activate the repo venv and re-run." | tee -a "${LOG_FILE}"
  exit 1
fi

# Preflight: verify the CLI entrypoint exists
CLI_HELP="$("${PYTHON_BIN}" -m lux_depth_v3.cli api-process --help 2>&1 || true)"
if [[ -z "${CLI_HELP}" ]]; then
  echo "❌ Unable to run: ${PYTHON_BIN} -m lux_depth_v3.cli api-process --help" | tee "${LOG_FILE}"
  exit 1
fi

# Detect which model flag is supported by the CLI (future-proofing)
MODEL_FLAG="--model"
MODEL_DEFAULT="large-v1.1"
if grep -q -- "--model-variant" <<<"${CLI_HELP}"; then
  MODEL_FLAG="--model-variant"
  MODEL_DEFAULT="DA3_LARGE_V1_1"
elif grep -q -- "--model-hf-id" <<<"${CLI_HELP}"; then
  MODEL_FLAG="--model-hf-id"
  MODEL_DEFAULT="depth-anything/DA3-LARGE-1.1"
elif grep -q -- "--model" <<<"${CLI_HELP}"; then
  MODEL_FLAG="--model"
  MODEL_DEFAULT="large-v1.1"
fi

DA3_MODEL="${DA3_MODEL:-${MODEL_DEFAULT}}"
DA3_EXPORT_FORMAT="${DA3_EXPORT_FORMAT:-mini_npz}"
DA3_DEVICE="${DA3_DEVICE:-auto}"
DA3_PROCESS_RES="${DA3_PROCESS_RES:-2048}"

# Explicit file map with scene types (absolute paths)
declare -a ITEMS=(
  "interior|${REPO_ROOT}/750Picacho_Source_TIFFs/750Picacho_GreatRoom.tif"
  "interior|${REPO_ROOT}/750Picacho_Source_TIFFs/750Picacho_Kitchen.tif"
  "interior|${REPO_ROOT}/750Picacho_Source_TIFFs/750Picacho_PrimaryBathroom.tif"
  "interior|${REPO_ROOT}/750Picacho_Source_TIFFs/750Picacho_PrimaryBedroom.tif"
  "exterior|${REPO_ROOT}/750Picacho_Source_TIFFs/750Picacho_Aerial.tif"
  "exterior|${REPO_ROOT}/projects/750_picacho_lane/Final_Production_UltraQuality/750Picacho_Pool_UltraQuality.tif"
)

TOTAL="${#ITEMS[@]}"
SUCCESS=0
FAILED=0

{
  echo "========================================"
  echo "APEX Depth Map Generation - DA3 Large v1.1"
  echo "Output:    ${OUTPUT_DIR}"
  echo "Timestamp: ${TIMESTAMP}"
  echo "Python:    ${PYTHON_BIN}"
  echo "Model:     ${DA3_MODEL} (flag: ${MODEL_FLAG})"
  echo "Export:    ${DA3_EXPORT_FORMAT}"
  echo "Device:    ${DA3_DEVICE}"
  echo "Process:   process_res=${DA3_PROCESS_RES}"
  echo "Total:     ${TOTAL}"
  echo "========================================"
  echo
} | tee "${LOG_FILE}"

for idx in "${!ITEMS[@]}"; do
  IFS="|" read -r SCENE_TYPE INPUT_PATH <<< "${ITEMS[$idx]}"
  NUM=$((idx + 1))

  echo "----------------------------------------" | tee -a "${LOG_FILE}"
  echo "[${NUM}/${TOTAL}] $(basename "${INPUT_PATH}")" | tee -a "${LOG_FILE}"
  echo "Scene:  ${SCENE_TYPE}" | tee -a "${LOG_FILE}"
  echo "Input:  ${INPUT_PATH}" | tee -a "${LOG_FILE}"
  echo "----------------------------------------" | tee -a "${LOG_FILE}"

  if [[ ! -f "${INPUT_PATH}" ]]; then
    echo "❌ ERROR: File not found: ${INPUT_PATH}" | tee -a "${LOG_FILE}"
    FAILED=$((FAILED + 1))
    echo "" | tee -a "${LOG_FILE}"
    continue
  fi

  SCENE_OUTPUT="${OUTPUT_DIR}/${SCENE_TYPE}"
  mkdir -p "${SCENE_OUTPUT}"

  STEM="$(basename "${INPUT_PATH}")"
  STEM="${STEM%.tif}"
  STEM="${STEM%.tiff}"
  STEM="${STEM%_UltraQuality}"
  SCENE_LOG="${LOG_DIR}/${STEM}_dav3_${TIMESTAMP}.log"

  START_EPOCH="$(${PYTHON_BIN} - <<'PY'
import time
print(int(time.time()))
PY
)"

  echo "Running DA3 via lux_depth_v3..." | tee -a "${LOG_FILE}"
  echo "  log: ${SCENE_LOG}" | tee -a "${LOG_FILE}"

  set +e
  ${PYTHON_BIN} -m lux_depth_v3.cli api-process \
    "${INPUT_PATH}" \
    --output-dir "${SCENE_OUTPUT}" \
    "${MODEL_FLAG}" "${DA3_MODEL}" \
    --export-format "${DA3_EXPORT_FORMAT}" \
    --device "${DA3_DEVICE}" \
    --process-res "${DA3_PROCESS_RES}" \
    2>&1 | tee "${SCENE_LOG}"
  RC="${PIPESTATUS[0]}"
  set -e

  if [[ "${RC}" -ne 0 ]]; then
    echo "❌ FAILED: $(basename "${INPUT_PATH}") (exit=${RC})" | tee -a "${LOG_FILE}"
    FAILED=$((FAILED + 1))
    echo "" | tee -a "${LOG_FILE}"
    continue
  fi

  NEWEST_NPZ="$(${PYTHON_BIN} - <<PY
import glob, os
out_dir = r"${SCENE_OUTPUT}"
start_epoch = int(r"${START_EPOCH}")
cands = glob.glob(os.path.join(out_dir, "**", "*.npz"), recursive=True)
cands = [p for p in cands if os.path.getmtime(p) >= (start_epoch - 2)]
cands.sort(key=os.path.getmtime, reverse=True)
print(cands[0] if cands else "")
PY
)"

  if [[ -z "${NEWEST_NPZ}" ]]; then
    echo "❌ FAILED: no fresh .npz produced in ${SCENE_OUTPUT}" | tee -a "${LOG_FILE}"
    FAILED=$((FAILED + 1))
    echo "" | tee -a "${LOG_FILE}"
    continue
  fi

  if ! ${PYTHON_BIN} - <<PY >/dev/null 2>&1
import numpy as np
d = np.load(r"${NEWEST_NPZ}")
assert "depth" in d.files
PY
  then
    echo "❌ FAILED: NPZ output invalid (missing depth): ${NEWEST_NPZ}" | tee -a "${LOG_FILE}"
    FAILED=$((FAILED + 1))
    echo "" | tee -a "${LOG_FILE}"
    continue
  fi

  echo "✅ SUCCESS: $(basename "${INPUT_PATH}")" | tee -a "${LOG_FILE}"
  echo "   NPZ: ${NEWEST_NPZ}" | tee -a "${LOG_FILE}"
  SUCCESS=$((SUCCESS + 1))
  echo "" | tee -a "${LOG_FILE}"
done

echo "========================================" | tee -a "${LOG_FILE}"
echo "Batch Processing Complete" | tee -a "${LOG_FILE}"
echo "========================================" | tee -a "${LOG_FILE}"
echo "Total:   ${TOTAL}" | tee -a "${LOG_FILE}"
echo "Success: ${SUCCESS}" | tee -a "${LOG_FILE}"
echo "Failed:  ${FAILED}" | tee -a "${LOG_FILE}"
echo "Output:  ${OUTPUT_DIR}" | tee -a "${LOG_FILE}"
echo "Log:     ${LOG_FILE}" | tee -a "${LOG_FILE}"
echo "========================================" | tee -a "${LOG_FILE}"
echo "" | tee -a "${LOG_FILE}"

echo "Generated outputs:" | tee -a "${LOG_FILE}"
find "${OUTPUT_DIR}" -type f \( -name "*.npz" -o -name "*.png" -o -name "*.npy" \) -exec ls -lh {} \; | tee -a "${LOG_FILE}"

if [[ "${FAILED}" -ne 0 ]]; then
  exit 1
fi
