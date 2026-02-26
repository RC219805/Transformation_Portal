#!/usr/bin/env bash
# Full-chain determinism trial on real files:
#   Phase 4C: extract_capture_metadata
#   Phase 4D: build_metadata_manifest
#   Phase 4E: build_provenance_manifest
#   Phase 4E: build_provenance_merkle
#
# Runs N times, compares byte-level artifact hashes across runs, then optionally
# repeats from /tmp to validate relocatability/CWD-independence.
#
# Exit codes:
#   0  success (all deterministic)
#   2  usage / missing dependency
#   3  preflight failure (paths/tools missing)
#   4  pipeline run failure
#   5  determinism mismatch detected

set -euo pipefail
IFS=$'\n\t'
umask 022

RUNS=2
STRICT=1
DO_TMP=1
CLEAN=0
INPUT_DIR=""
OUT_DIR=""
PYTHON_BIN="${PYTHON_BIN:-python3}"

die() { echo "ERROR: $*" >&2; exit "${2:-1}"; }
log() { echo "[$(date -u +"%Y-%m-%dT%H:%M:%SZ")] $*" >&2; }
have() { command -v "$1" >/dev/null 2>&1; }

sha256_file() {
  local file_path="$1"
  if have sha256sum; then
    sha256sum "$file_path" | awk '{print $1}'
  elif have shasum; then
    shasum -a 256 "$file_path" | awk '{print $1}'
  else
    die "Need sha256sum or shasum installed." 2
  fi
}

resolve_repo_root() {
  local here d
  here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  d="$here"
  while [[ "$d" != "/" ]]; do
    if [[ -d "$d/.git" ]]; then
      echo "$d"
      return 0
    fi
    if [[ -f "$d/pyproject.toml" && -d "$d/.github/workflows" ]]; then
      echo "$d"
      return 0
    fi
    d="$(dirname "$d")"
  done
  return 1
}

usage() {
  cat >&2 <<'USAGE'
Usage:
  scripts/diagnostics/full_chain_determinism_trial.sh --input <raw_dir> [options]

Required:
  --input <dir>         Directory containing trial RAW files (CR2/DNG/NEF/ARW/TIF/etc)

Options:
  --out <dir>           Output directory for trial artifacts (default: <repo>/trial_runs/<timestamp>)
  --runs <N>            Number of runs to compare (default: 2)
  --no-strict           Disable --strict for extraction (default: strict enabled)
  --no-tmp              Skip /tmp relocatability run (default: enabled)
  --clean               Remove output directory before running
  --python <path>       Python executable (default: $PYTHON_BIN)
  -h, --help            Show help

Examples:
  scripts/diagnostics/full_chain_determinism_trial.sh --input ./trial_dataset/input_raw
  scripts/diagnostics/full_chain_determinism_trial.sh --input /Volumes/RAW_TRIAL --runs 3 --no-tmp
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --input)
      INPUT_DIR="${2:-}"
      shift 2
      ;;
    --out)
      OUT_DIR="${2:-}"
      shift 2
      ;;
    --runs)
      RUNS="${2:-}"
      shift 2
      ;;
    --no-strict)
      STRICT=0
      shift 1
      ;;
    --no-tmp)
      DO_TMP=0
      shift 1
      ;;
    --clean)
      CLEAN=1
      shift 1
      ;;
    --python)
      PYTHON_BIN="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "Unknown argument: $1 (use --help)" 2
      ;;
  esac
done

[[ -n "$INPUT_DIR" ]] || { usage; exit 2; }
[[ -d "$INPUT_DIR" ]] || die "--input dir not found: $INPUT_DIR" 3
[[ "$RUNS" =~ ^[0-9]+$ ]] || die "--runs must be an integer" 2
(( RUNS >= 2 )) || die "--runs must be >= 2 to compare determinism" 2
have "$PYTHON_BIN" || die "Python not found: $PYTHON_BIN" 2

# Determinism-friendly process environment.
export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"
export LC_ALL="${LC_ALL:-C}"
export LANG="${LANG:-C}"
export TZ="${TZ:-UTC}"

REPO_ROOT="$(resolve_repo_root)" || die "Could not resolve repo root from script location." 3

TOOL_4C="$REPO_ROOT/tools/extract_capture_metadata.py"
TOOL_4D="$REPO_ROOT/tools/build_metadata_manifest.py"
TOOL_4E_P="$REPO_ROOT/tools/build_provenance_manifest.py"
TOOL_4E_M="$REPO_ROOT/tools/build_provenance_merkle.py"

[[ -f "$TOOL_4C" ]] || die "Missing tool: $TOOL_4C" 3
[[ -f "$TOOL_4D" ]] || die "Missing tool: $TOOL_4D" 3
[[ -f "$TOOL_4E_P" ]] || die "Missing tool: $TOOL_4E_P" 3
[[ -f "$TOOL_4E_M" ]] || die "Missing tool: $TOOL_4E_M" 3

STAMP="$(date -u +"%Y%m%dT%H%M%SZ")"
if [[ -z "$OUT_DIR" ]]; then
  OUT_DIR="$REPO_ROOT/trial_runs/full_chain_determinism_$STAMP"
fi

if [[ "$CLEAN" == "1" && -e "$OUT_DIR" ]]; then
  log "--clean set; removing existing OUT_DIR: $OUT_DIR"
  rm -rf "$OUT_DIR"
fi

mkdir -p "$OUT_DIR"

{
  echo "trial_utc=$STAMP"
  echo "repo_root=$REPO_ROOT"
  echo "input_dir=$(cd "$INPUT_DIR" && pwd)"
  echo "out_dir=$OUT_DIR"
  echo "runs=$RUNS"
  echo "strict=$STRICT"
  echo "do_tmp=$DO_TMP"
  echo "python=$($PYTHON_BIN --version 2>&1 | tr -d '\r')"
  echo "pythonhashseed=$PYTHONHASHSEED"
  echo "lc_all=$LC_ALL"
  echo "tz=$TZ"
} > "$OUT_DIR/trial_meta.txt"

log "Repo root: $REPO_ROOT"
log "Input dir:  $INPUT_DIR"
log "Out dir:    $OUT_DIR"
log "Runs:       $RUNS"
log "Strict:     $STRICT"
log "Tmp run:    $DO_TMP"

run_pipeline_once() {
  local run_label="$1"
  local work_cwd="$2"
  local out_run_dir="$OUT_DIR/$run_label"

  mkdir -p "$out_run_dir/artifacts"

  local capture="$out_run_dir/artifacts/capture_metadata.tp.meta.capture.v1.json"
  local manifest="$out_run_dir/artifacts/metadata_manifest.tp.meta.capture_manifest.v1.json"
  local provenance="$out_run_dir/artifacts/provenance_manifest.tp.meta.provenance.v1.json"
  local merkle="$out_run_dir/artifacts/provenance_merkle.tp.meta.provenance_merkle.v1.json"

  log "=== ${run_label}: executing from CWD=$work_cwd ==="

  (
    cd "$work_cwd"

    if [[ "$STRICT" == "1" ]]; then
      "$PYTHON_BIN" "$TOOL_4C" --input "$INPUT_DIR" --output "$capture" --strict
    else
      "$PYTHON_BIN" "$TOOL_4C" --input "$INPUT_DIR" --output "$capture"
    fi

    "$PYTHON_BIN" "$TOOL_4D" --capture "$capture" --output "$manifest" --enforce-fingerprint
    "$PYTHON_BIN" "$TOOL_4E_P" --capture "$capture" --manifest "$manifest" --output "$provenance"
    "$PYTHON_BIN" "$TOOL_4E_M" --provenance "$provenance" --output "$merkle"
  ) || return 1

  local ledger="$out_run_dir/artifacts.sha256"
  : > "$ledger"
  local artifact
  for artifact in "$capture" "$manifest" "$provenance" "$merkle"; do
    [[ -f "$artifact" ]] || die "Expected artifact missing after ${run_label}: $artifact" 4
    printf "%s  %s\n" "$(sha256_file "$artifact")" "$(basename "$artifact")" >> "$ledger"
  done

  local sizes="$out_run_dir/artifacts.sizes"
  : > "$sizes"
  for artifact in "$capture" "$manifest" "$provenance" "$merkle"; do
    if stat -c %s "$artifact" >/dev/null 2>&1; then
      printf "%s  %s\n" "$(stat -c %s "$artifact")" "$(basename "$artifact")" >> "$sizes"
    else
      printf "%s  %s\n" "$(stat -f %z "$artifact")" "$(basename "$artifact")" >> "$sizes"
    fi
  done

  log "=== ${run_label}: completed ==="
}

compare_ledgers() {
  local first="$1"
  shift
  local ok=1
  local other
  for other in "$@"; do
    if ! diff -u "$first" "$other" >/dev/null 2>&1; then
      log "DETERMINISM MISMATCH: $first vs $other"
      diff -u "$first" "$other" >&2 || true
      ok=0
    fi
  done
  [[ "$ok" == "1" ]]
}

log "Starting primary runs from repo root..."
for i in $(seq 1 "$RUNS"); do
  run_label="$(printf "run_%02d" "$i")"
  if ! run_pipeline_once "$run_label" "$REPO_ROOT"; then
    die "Pipeline failed during ${run_label}" 4
  fi
done

PRIMARY_FIRST="$OUT_DIR/run_01/artifacts.sha256"
PRIMARY_OTHERS=()
for i in $(seq 2 "$RUNS"); do
  PRIMARY_OTHERS+=("$OUT_DIR/$(printf "run_%02d" "$i")/artifacts.sha256")
done

log "Comparing primary run ledgers..."
if ! compare_ledgers "$PRIMARY_FIRST" "${PRIMARY_OTHERS[@]}"; then
  die "Primary determinism failure detected. See diffs above and $OUT_DIR." 5
fi
log "Primary determinism: PASS"

if [[ "$DO_TMP" == "1" ]]; then
  TMPDIR_BASE="${TMPDIR:-/tmp}"
  TMP_CWD="$TMPDIR_BASE/tp_full_chain_trial_${STAMP}_$$"
  mkdir -p "$TMP_CWD"
  trap 'rm -rf "$TMP_CWD" >/dev/null 2>&1 || true' EXIT

  log "Starting /tmp runs from: $TMP_CWD"
  for i in $(seq 1 "$RUNS"); do
    run_label="$(printf "tmp_run_%02d" "$i")"
    if ! run_pipeline_once "$run_label" "$TMP_CWD"; then
      die "Pipeline failed during ${run_label} (/tmp)" 4
    fi
  done

  TMP_FIRST="$OUT_DIR/tmp_run_01/artifacts.sha256"
  TMP_OTHERS=()
  for i in $(seq 2 "$RUNS"); do
    TMP_OTHERS+=("$OUT_DIR/$(printf "tmp_run_%02d" "$i")/artifacts.sha256")
  done

  log "Comparing /tmp run ledgers..."
  if ! compare_ledgers "$TMP_FIRST" "${TMP_OTHERS[@]}"; then
    die "/tmp determinism failure detected. See diffs above and $OUT_DIR." 5
  fi
  log "/tmp determinism: PASS"

  log "Comparing primary vs /tmp ledgers (CWD-independence)..."
  if ! diff -u "$PRIMARY_FIRST" "$TMP_FIRST" >/dev/null 2>&1; then
    log "PRIMARY vs /tmp mismatch (CWD-independence failure)"
    diff -u "$PRIMARY_FIRST" "$TMP_FIRST" >&2 || true
    die "Relocatability/CWD-independence failure detected. See $OUT_DIR." 5
  fi
  log "Primary vs /tmp: PASS"
fi

log "ALL CHECKS PASSED"
log "Artifacts + ledgers captured under: $OUT_DIR"
exit 0
