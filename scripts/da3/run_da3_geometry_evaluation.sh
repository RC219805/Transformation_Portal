#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# DA3 Geometry Evaluation (Picacho R&D)
# - Runs DA3 on a small canonical subset (GreatRoom / Kitchen / Aerial)
# - Produces per-model depth+mesh outputs and a quick metrics summary
#
# IMPORTANT: DA3-LARGE-1.1 / DA3NESTED-GIANT-LARGE-1.1 are CC BY-NC 4.0
#            (non-commercial use only). This script hard-gates execution.
#
# Usage:
#   export DA3_NONCOMMERCIAL_ACK=1
#   ./run_da3_geometry_evaluation.sh
# ==============================================================================

if [[ "${DA3_NONCOMMERCIAL_ACK:-0}" != "1" ]]; then
  echo "❌ DA3 license gate: set DA3_NONCOMMERCIAL_ACK=1 to confirm non-commercial R&D usage." >&2
  echo "   Models in this script are CC BY-NC 4.0 (non-commercial)." >&2
  exit 2
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  PYTHON_BIN="python"
fi
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "❌ python not found (tried: python3, python)" >&2
  exit 1
fi

if [[ -z "${REPO_ROOT:-}" ]]; then
  if command -v git >/dev/null 2>&1 && git rev-parse --show-toplevel >/dev/null 2>&1; then
    REPO_ROOT="$(git rev-parse --show-toplevel)"
  else
    REPO_ROOT="$(pwd)"
  fi
fi

DA3_REPO="${DA3_REPO:-${REPO_ROOT}/external/depth-anything-3}"
INPUT_DIR="${INPUT_DIR:-${REPO_ROOT}/750Picacho_Source_TIFFs}"
WORKDIR="${WORKDIR:-${REPO_ROOT}/workspace/da3_geometry_eval}"

TS="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${WORKDIR}/run_${TS}"

MODEL_A="${MODEL_A:-depth-anything/DA3-LARGE-1.1}"
MODEL_B="${MODEL_B:-depth-anything/DA3NESTED-GIANT-LARGE-1.1}"

BACKEND_HOST="${BACKEND_HOST:-127.0.0.1}"
BACKEND_PORT="${BACKEND_PORT:-8008}"
BACKEND_URL="http://${BACKEND_HOST}:${BACKEND_PORT}"
BACKEND_GALLERY="${OUT_DIR}/gallery"

mkdir -p "${OUT_DIR}" "${BACKEND_GALLERY}"

DEVICE="${DEVICE:-}"
if [[ -z "${DEVICE}" ]]; then
  DEVICE="$(${PYTHON_BIN} - <<'PY'
import torch
if torch.cuda.is_available():
    print("cuda")
elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
    print("mps")
else:
    print("cpu")
PY
)"
fi

echo "============================================================"
echo "DA3 Geometry Evaluation (Picacho R&D)"
echo "Output:   ${OUT_DIR}"
echo "Device:   ${DEVICE}"
echo "Model A:  ${MODEL_A}"
echo "Model B:  ${MODEL_B}"
echo "Backend:  ${BACKEND_URL}"
echo "============================================================"

if ! command -v da3 >/dev/null 2>&1; then
  echo "❌ da3 CLI not found in PATH." >&2
  exit 1
fi

declare -a SCENES=(
  "GreatRoom|${INPUT_DIR}/750Picacho_GreatRoom.tif|depth_gr"
  "Kitchen|${INPUT_DIR}/750Picacho_Kitchen.tif|depth_kt"
  "Aerial|${INPUT_DIR}/750Picacho_Aerial.tif|depth_ar"
)

convert_to_png() {
  local in_tif="$1"
  local out_png="$2"
  ${PYTHON_BIN} - <<PY
from PIL import Image
im = Image.open(r"${in_tif}")
if im.mode not in ("RGB", "RGBA"):
    im = im.convert("RGB")
elif im.mode == "RGBA":
    bg = Image.new("RGB", im.size, (0,0,0))
    bg.paste(im, mask=im.split()[-1])
    im = bg
im.save(r"${out_png}", format="PNG", optimize=True)
PY
}

BACKEND_PID=""

stop_backend() {
  if [[ -n "${BACKEND_PID}" ]] && kill -0 "${BACKEND_PID}" >/dev/null 2>&1; then
    echo "Stopping backend (pid=${BACKEND_PID})..."
    kill "${BACKEND_PID}" >/dev/null 2>&1 || true
    sleep 2 || true
    kill -9 "${BACKEND_PID}" >/dev/null 2>&1 || true
  fi
  BACKEND_PID=""
}

wait_backend_ready() {
  local timeout_s="${1:-60}"
  local start
  start="$(${PYTHON_BIN} - <<'PY'
import time
print(time.time())
PY
)"
  while true; do
    if command -v curl >/dev/null 2>&1; then
      if curl -sf "${BACKEND_URL}/status" >/dev/null 2>&1; then
        return 0
      fi
    fi
    local now
    now="$(${PYTHON_BIN} - <<'PY'
import time
print(time.time())
PY
)"
    if ${PYTHON_BIN} - <<PY >/dev/null 2>&1
start=float("${start}")
now=float("${now}")
timeout=float("${timeout_s}")
import sys
sys.exit(0 if (now-start) < timeout else 1)
PY
    then
      sleep 1
    else
      return 1
    fi
  done
}

AUTO_HELP="$(da3 auto --help 2>&1 || true)"
USE_BACKEND_ARGS=(--use-backend)
if echo "${AUTO_HELP}" | grep -Eq -- "--use-backend[[:space:]]+(URL|url|HTTP|http|BACKEND|backend)"; then
  USE_BACKEND_ARGS=(--use-backend "${BACKEND_URL}")
fi

run_model() {
  local model="$1"
  local tag="$2"

  echo ""
  echo "============================================================"
  echo "Model: ${model} (${tag})"
  echo "============================================================"

  stop_backend
  (cd "${DA3_REPO}" && da3 backend \
      --model-dir "${model}" \
      --device "${DEVICE}" \
      --host "${BACKEND_HOST}" \
      --port "${BACKEND_PORT}" \
      --gallery-dir "${BACKEND_GALLERY}" \
      > "${OUT_DIR}/backend_${tag}.log" 2>&1) &
  BACKEND_PID="$!"

  if ! wait_backend_ready 60; then
    echo "❌ Backend failed to start; tail:" >&2
    tail -n 80 "${OUT_DIR}/backend_${tag}.log" >&2 || true
    stop_backend
    return 1
  fi
  echo "✅ Backend ready"

  local model_fail=0

  for entry in "${SCENES[@]}"; do
    IFS="|" read -r name tif scene_tag <<<"${entry}"

    echo ""
    echo "----------------------------"
    echo "Scene: ${name}"
    echo "Input: ${tif}"
    echo "----------------------------"

    if [[ ! -f "${tif}" ]]; then
      echo "❌ Missing input: ${tif}" >&2
      model_fail=1
      continue
    fi

    STAGE_DIR="${OUT_DIR}/stage_${scene_tag}"
    EXPORT_DIR="${OUT_DIR}/export_${scene_tag}_${tag}"
    mkdir -p "${STAGE_DIR}" "${EXPORT_DIR}"

    PNG_PATH="${STAGE_DIR}/${scene_tag}.png"
    convert_to_png "${tif}" "${PNG_PATH}"

    (cd "${DA3_REPO}" && da3 auto "${STAGE_DIR}" \
        --export-dir "${EXPORT_DIR}" \
        --export-format "mini_npz-glb" \
        "${USE_BACKEND_ARGS[@]}" \
        > "${EXPORT_DIR}/da3_run_${tag}.log" 2>&1) || {
          echo "❌ da3 auto failed; tail:" >&2
          tail -n 80 "${EXPORT_DIR}/da3_run_${tag}.log" >&2 || true
          model_fail=1
          continue
        }

    NPZ_PATH="$(${PYTHON_BIN} - <<PY
import glob, os
c = glob.glob(r"${EXPORT_DIR}/**/*.npz", recursive=True)
c.sort(key=os.path.getmtime, reverse=True)
print(c[0] if c else "")
PY
)"
    GLB_PATH="$(${PYTHON_BIN} - <<PY
import glob, os
c = glob.glob(r"${EXPORT_DIR}/**/*.glb", recursive=True)
c.sort(key=os.path.getmtime, reverse=True)
print(c[0] if c else "")
PY
)"

    if [[ -z "${NPZ_PATH}" || -z "${GLB_PATH}" ]]; then
      echo "❌ Missing outputs (.npz/.glb)" >&2
      model_fail=1
      continue
    fi

    cp -f "${NPZ_PATH}" "${OUT_DIR}/depth_${scene_tag}_${tag}.npz"
    cp -f "${GLB_PATH}" "${OUT_DIR}/mesh_${scene_tag}_${tag}.glb"

    ${PYTHON_BIN} - <<PY
import json, numpy as np
from PIL import Image
from pathlib import Path

npz = np.load(r"${OUT_DIR}/depth_${scene_tag}_${tag}.npz")
depth = npz["depth"].astype(np.float32)
conf = npz["conf"].astype(np.float32) if "conf" in npz.files else None

lo, hi = np.nanpercentile(depth, 1), np.nanpercentile(depth, 99)
if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
    lo, hi = float(np.nanmin(depth)), float(np.nanmax(depth))
vis = np.clip((depth - lo) / (hi - lo + 1e-6), 0, 1)
vis_u8 = (vis * 255).astype(np.uint8)

vis_dir = Path(r"${OUT_DIR}") / "vis"
vis_dir.mkdir(parents=True, exist_ok=True)
Image.fromarray(vis_u8).save(vis_dir / f"depth_{r'${scene_tag}'}_{r'${tag}'}.png")

m = {
  "depth_shape": list(depth.shape),
  "depth_min": float(np.nanmin(depth)),
  "depth_max": float(np.nanmax(depth)),
  "depth_p05": float(np.nanpercentile(depth, 5)),
  "depth_p50": float(np.nanpercentile(depth, 50)),
  "depth_p95": float(np.nanpercentile(depth, 95)),
  "depth_nan_ratio": float(np.isnan(depth).mean()),
}
if conf is not None:
  m.update({
    "conf_shape": list(conf.shape),
    "conf_min": float(np.nanmin(conf)),
    "conf_max": float(np.nanmax(conf)),
    "conf_p50": float(np.nanpercentile(conf, 50)),
  })

Path(r"${OUT_DIR}").joinpath(f"metrics_{r'${scene_tag}'}_{r'${tag}'}.json").write_text(json.dumps(m, indent=2))
PY

    echo "✅ Scene complete: ${name} (${tag})"
  done

  stop_backend
  return "${model_fail}"
}

FAIL=0
run_model "${MODEL_A}" "A" || FAIL=1
run_model "${MODEL_B}" "B" || FAIL=1

REPORT="${OUT_DIR}/report.md"
{
  echo "# DA3 Geometry Evaluation (Picacho R&D)"
  echo ""
  echo "- Timestamp: ${TS}"
  echo "- Device: ${DEVICE}"
  echo "- Backend: ${BACKEND_URL}"
  echo "- Model A: ${MODEL_A}"
  echo "- Model B: ${MODEL_B}"
  echo ""
  echo "Outputs are in: ${OUT_DIR}"
} > "${REPORT}"

echo ""
echo "============================================================"
echo "Done. Report: ${REPORT}"
echo "============================================================"

if [[ "${FAIL}" -ne 0 ]]; then
  exit 1
fi
