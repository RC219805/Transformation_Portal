#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MANIFEST_PATH="$REPO_ROOT/config/fastvlm_runtime_manifest.json"
RUNTIME_REQUIREMENTS_FILE="$REPO_ROOT/config/fastvlm_runtime_requirements.txt"
RUNTIME_ROOT="$REPO_ROOT/.runtime/fastvlm"
MODELS="smoke,default"
ALL_MODELS=0
DRY_RUN=0
VERIFY_ONLY=0
SKIP_MODEL_DOWNLOAD=0
SKIP_VERIFY=0

usage() {
  cat <<'EOF'
Usage: scripts/setup/install_fastvlm_runtime.sh [options]

Options:
  --models ROLE[,ROLE...]     Model roles to install (default: smoke,default)
  --all-models                Install all manifest model roles
  --verify-only               Verify the existing runtime and exit
  --skip-model-download       Install runtime sources/venv but do not download models
  --skip-verify               Skip final runtime verification
  --dry-run                   Print planned actions without writing files or using network
  --help                      Show this help
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --models)
      MODELS="${2:-}"
      shift 2
      ;;
    --models=*)
      MODELS="${1#*=}"
      shift
      ;;
    --all-models)
      ALL_MODELS=1
      shift
      ;;
    --verify-only)
      VERIFY_ONLY=1
      shift
      ;;
    --skip-model-download)
      SKIP_MODEL_DOWNLOAD=1
      shift
      ;;
    --skip-verify)
      SKIP_VERIFY=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [ "$ALL_MODELS" -eq 1 ]; then
  MODEL_ARGS=(--all-models)
else
  MODEL_ARGS=(--models "$MODELS")
fi

REPO_PY="$("$REPO_ROOT/scripts/setup/resolve_python_311.sh")"

manifest_source_value() {
  "$REPO_PY" -c 'import json, sys
from pathlib import Path
manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(manifest["runtime_sources"][sys.argv[2]][sys.argv[3]])' "$MANIFEST_PATH" "$1" "$2"
}

FASTVLM_REF="$(manifest_source_value ml_fastvlm revision)"
MLX_VLM_REF="$(manifest_source_value mlx_vlm revision)"
FASTVLM_REPO="$(manifest_source_value ml_fastvlm repo_url)"
MLX_VLM_REPO="$(manifest_source_value mlx_vlm repo_url)"
FASTVLM_DIR="$RUNTIME_ROOT/ml-fastvlm"
MLX_VLM_DIR="$RUNTIME_ROOT/mlx-vlm"
VENV_DIR="$RUNTIME_ROOT/.venv-fastvlm"
VENV_PY="$VENV_DIR/bin/python"

run() {
  if [ "$DRY_RUN" -eq 1 ]; then
    printf '[dry-run]'
    for arg in "$@"; do
      printf ' %q' "$arg"
    done
    printf '\n'
  else
    "$@"
  fi
}

verify_runtime() {
  run "$REPO_PY" "$REPO_ROOT/scripts/validation/validate_fastvlm_runtime.py" \
    --manifest "$MANIFEST_PATH" \
    --runtime-root "$RUNTIME_ROOT" \
    "${MODEL_ARGS[@]}" \
    --verify-only
}

ensure_clone() {
  repo_url="$1"
  revision="$2"
  target_dir="$3"
  if [ "$DRY_RUN" -eq 1 ]; then
    if [ -d "$target_dir/.git" ]; then
      run git -C "$target_dir" fetch origin "$revision"
      run git -C "$target_dir" checkout --detach "$revision"
    else
      run git clone "$repo_url" "$target_dir"
      run git -C "$target_dir" checkout --detach "$revision"
    fi
    return
  fi

  mkdir -p "$(dirname "$target_dir")"
  if [ -d "$target_dir/.git" ]; then
    git -C "$target_dir" fetch origin "$revision"
  else
    git clone "$repo_url" "$target_dir"
  fi
  git -C "$target_dir" checkout --detach "$revision"
}

if [ "$VERIFY_ONLY" -eq 1 ]; then
  verify_runtime
  exit 0
fi

if [ "$DRY_RUN" -eq 1 ]; then
  echo "[dry-run] runtime_root=$RUNTIME_ROOT"
else
  mkdir -p "$RUNTIME_ROOT"
fi

ensure_clone "$FASTVLM_REPO" "$FASTVLM_REF" "$FASTVLM_DIR"
ensure_clone "$MLX_VLM_REPO" "$MLX_VLM_REF" "$MLX_VLM_DIR"

if [ ! -x "$VENV_PY" ]; then
  run "$REPO_PY" -m venv "$VENV_DIR"
fi
run "$VENV_PY" -m pip install --requirement "$RUNTIME_REQUIREMENTS_FILE"
run "$VENV_PY" -m pip install --no-deps -e "$MLX_VLM_DIR"
if [ "$DRY_RUN" -eq 1 ]; then
  echo "[dry-run] skip pip freeze capture"
else
  "$VENV_PY" -m pip freeze > "$RUNTIME_ROOT/fastvlm-pip-freeze.txt"
fi

if [ "$SKIP_MODEL_DOWNLOAD" -eq 0 ]; then
  run "$VENV_PY" "$REPO_ROOT/scripts/setup/download_fastvlm_models.py" \
    --manifest "$MANIFEST_PATH" \
    --runtime-root "$RUNTIME_ROOT" \
    "${MODEL_ARGS[@]}"
fi

if [ "$SKIP_VERIFY" -eq 0 ]; then
  verify_runtime
fi
