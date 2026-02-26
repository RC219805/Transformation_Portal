#!/usr/bin/env bash
# Full-chain determinism trial:
#   Phase 4C: extract_capture_metadata
#   Phase 4D: build_metadata_manifest
#   Phase 4E: build_provenance_manifest
#   Phase 4E: build_provenance_merkle
#
# Modes:
#   RAW mode:      --input-root <dir> (runs 4C -> 4D -> 4E -> merkle)
#   Artifact mode: --capture-metadata <json> (skips 4C extraction, seeds capture artifact)
#
# Exit codes:
#   0  success (all configured comparisons passed)
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
VERBOSE=0
HASH_INPUT=1
COMPARE_MODE="raw"
STRIP_JSON_KEYS=""
INPUT_ROOT=""
CAPTURE_METADATA_IN=""
OUT_DIR=""
PYTHON_BIN="${PYTHON_BIN:-python3}"
TMP_CWD=""
INPUT_MODE=""
INPUT_COUNT="0"
TOOL_4C=""
TOOL_4D=""
TOOL_4E_P=""
TOOL_4E_M=""


die() { echo "ERROR: $*" >&2; exit "${2:-1}"; }
log() { echo "[$(date -u +"%Y-%m-%dT%H:%M:%SZ")] $*" >&2; }
have() { command -v "$1" >/dev/null 2>&1; }

cleanup() {
  if [[ -n "${TMP_CWD:-}" && -d "${TMP_CWD:-}" ]]; then
    rm -rf "$TMP_CWD" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

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

filesize() {
  local file_path="$1"
  if stat -c %s "$file_path" >/dev/null 2>&1; then
    stat -c %s "$file_path"
  else
    stat -f %z "$file_path"
  fi
}

canonical_json_sha256_file() {
  local file_path="$1"
  local strip_csv="${STRIP_JSON_KEYS}"
  "$PYTHON_BIN" - "$file_path" "$strip_csv" <<'PY'
import hashlib
import json
import sys

path = sys.argv[1]
strip_csv = sys.argv[2] if len(sys.argv) > 2 else ""
strip = {k.strip() for k in strip_csv.split(",") if k.strip()}

def cleanse(value):
    if isinstance(value, dict):
        return {k: cleanse(v) for k, v in value.items() if k not in strip}
    if isinstance(value, list):
        return [cleanse(v) for v in value]
    return value

with open(path, "rb") as fh:
    data = json.load(fh)

canonical = json.dumps(
    cleanse(data),
    sort_keys=True,
    separators=(",", ":"),
    ensure_ascii=False,
)
print(hashlib.sha256(canonical.encode("utf-8")).hexdigest())
PY
}

resolve_repo_root() {
  local here d
  here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
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

maybe_require_tool() {
  local tool="$1"
  local env_gate="${2:-}"
  if have "$tool"; then
    return 0
  fi
  if [[ -n "$env_gate" && "${!env_gate:-0}" == "1" ]]; then
    die "Missing required runtime tool: $tool (set by $env_gate=1)" 2
  fi
  log "WARN: optional runtime tool not found: $tool (set ${env_gate:-<no gate>}=1 to require)"
}

run_with_log() {
  local log_path="$1"
  shift
  if [[ "$VERBOSE" == "1" ]]; then
    "$@" 2>&1 | tee "$log_path"
  else
    "$@" >"$log_path" 2>&1
  fi
}

compare_ledgers() {
  local first="$1"
  shift
  local ok=1
  local other
  for other in "$@"; do
    if ! diff -u "$first" "$other" >/dev/null 2>&1; then
      log "LEDGER MISMATCH: $first vs $other"
      diff -u "$first" "$other" >&2 || true
      ok=0
    fi
  done
  [[ "$ok" == "1" ]]
}

comparison_passes() {
  local raw_ok="$1"
  local canonical_ok="$2"
  case "$COMPARE_MODE" in
    raw)
      [[ "$raw_ok" == "1" ]]
      ;;
    canonical)
      [[ "$canonical_ok" == "1" ]]
      ;;
    both)
      [[ "$raw_ok" == "1" && "$canonical_ok" == "1" ]]
      ;;
    *)
      return 1
      ;;
  esac
}

log_classification() {
  local scope="$1"
  local raw_ok="$2"
  local canonical_ok="$3"
  if [[ "$raw_ok" != "1" && "$canonical_ok" == "1" ]]; then
    log "$scope classification: RAW mismatch with canonical match (likely serialization drift)."
  elif [[ "$canonical_ok" != "1" ]]; then
    log "$scope classification: canonical mismatch (likely semantic drift)."
  fi
}

usage() {
  cat >&2 <<'USAGE'
Usage:
  scripts/diagnostics/full_chain_determinism_trial.sh (--input-root <raw_dir> | --capture-metadata <json>) [options]

Required (choose one):
  --input-root <dir>        RAW mode: directory containing trial files for 4C extraction
  --capture-metadata <file> Artifact mode: existing Phase 4C artifact JSON (skip 4C extraction)

Options:
  --out <dir>               Output directory (default: <repo>/trial_runs/full_chain_determinism_<timestamp>)
  --runs <N>                Number of runs (default: 2). If N=1, skips comparison.
  --no-strict               Disable --strict for 4C extraction (RAW mode only)
  --no-tmp                  Skip /tmp relocatability run (default: enabled)
  --clean                   Remove output directory before running
  --verbose                 Stream per-step command output while also writing logs
  --no-input-hash           Do not record input SHA-256 ledger
  --hash-input              Record input SHA-256 ledger (default: enabled)
  --compare <mode>          Comparison gate mode: raw | canonical | both (default: raw)
  --strip-json-keys <csv>   CSV keys removed before canonical JSON hash (default: none)
  --tool-4c <path>          Override path to extract_capture_metadata.py
  --tool-4d <path>          Override path to build_metadata_manifest.py
  --tool-4e-prov <path>     Override path to build_provenance_manifest.py
  --tool-4e-merkle <path>   Override path to build_provenance_merkle.py
  --python <path>           Python executable (default: python3; override via $PYTHON_BIN env)
  -h, --help                Show help

Compatibility alias:
  --input <dir>             Alias for --input-root

Examples:
  scripts/diagnostics/full_chain_determinism_trial.sh --input-root ./trial_dataset/input_raw --runs 3
  scripts/diagnostics/full_chain_determinism_trial.sh --capture-metadata tests/golden/phase4/expected_capture_metadata.tp.meta.capture.v1.json --runs 2 --no-tmp --compare both
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --input-root|--input)
      INPUT_ROOT="${2:-}"
      shift 2
      ;;
    --capture-metadata)
      CAPTURE_METADATA_IN="${2:-}"
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
    --verbose)
      VERBOSE=1
      shift 1
      ;;
    --no-input-hash)
      HASH_INPUT=0
      shift 1
      ;;
    --hash-input)
      HASH_INPUT=1
      shift 1
      ;;
    --compare)
      COMPARE_MODE="${2:-}"
      shift 2
      ;;
    --strip-json-keys)
      STRIP_JSON_KEYS="${2:-}"
      shift 2
      ;;
    --tool-4c)
      TOOL_4C="${2:-}"
      shift 2
      ;;
    --tool-4d)
      TOOL_4D="${2:-}"
      shift 2
      ;;
    --tool-4e-prov)
      TOOL_4E_P="${2:-}"
      shift 2
      ;;
    --tool-4e-merkle)
      TOOL_4E_M="${2:-}"
      shift 2
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

if [[ -n "$INPUT_ROOT" && -n "$CAPTURE_METADATA_IN" ]]; then
  die "Provide exactly one of --input-root or --capture-metadata." 2
fi
if [[ -z "$INPUT_ROOT" && -z "$CAPTURE_METADATA_IN" ]]; then
  usage
  die "Missing required input. Provide --input-root or --capture-metadata." 2
fi

[[ "$RUNS" =~ ^[0-9]+$ ]] || die "--runs must be an integer" 2
(( RUNS >= 1 )) || die "--runs must be >= 1" 2

case "$COMPARE_MODE" in
  raw|canonical|both) ;;
  *)
    die "--compare must be one of: raw, canonical, both" 2
    ;;
esac

have "$PYTHON_BIN" || die "Python not found: $PYTHON_BIN" 2

if [[ -n "$INPUT_ROOT" ]]; then
  [[ -d "$INPUT_ROOT" ]] || die "--input-root dir not found: $INPUT_ROOT" 3
  INPUT_ROOT="$(cd "$INPUT_ROOT" && pwd -P)"
  INPUT_MODE="raw"
else
  [[ -f "$CAPTURE_METADATA_IN" ]] || die "--capture-metadata file not found: $CAPTURE_METADATA_IN" 3
  CAPTURE_METADATA_IN="$(cd "$(dirname "$CAPTURE_METADATA_IN")" && pwd -P)/$(basename "$CAPTURE_METADATA_IN")"
  INPUT_MODE="artifact"
fi

# Determinism-friendly process environment.
export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"
export LC_ALL="${LC_ALL:-C}"
export LANG="${LANG:-C}"
export TZ="${TZ:-UTC}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

REPO_ROOT="$(resolve_repo_root)" || die "Could not resolve repo root from script location." 3

: "${TOOL_4C:=$REPO_ROOT/tools/extract_capture_metadata.py}"
: "${TOOL_4D:=$REPO_ROOT/tools/build_metadata_manifest.py}"
: "${TOOL_4E_P:=$REPO_ROOT/tools/build_provenance_manifest.py}"
: "${TOOL_4E_M:=$REPO_ROOT/tools/build_provenance_merkle.py}"

[[ -f "$TOOL_4C" ]] || die "Missing tool: $TOOL_4C" 3
[[ -f "$TOOL_4D" ]] || die "Missing tool: $TOOL_4D" 3
[[ -f "$TOOL_4E_P" ]] || die "Missing tool: $TOOL_4E_P" 3
[[ -f "$TOOL_4E_M" ]] || die "Missing tool: $TOOL_4E_M" 3

if [[ "$INPUT_MODE" == "raw" ]]; then
  maybe_require_tool "exiftool" "TP_TRIAL_REQUIRE_EXIFTOOL"
fi

STAMP="$(date -u +"%Y%m%dT%H%M%SZ")"
if [[ -z "$OUT_DIR" ]]; then
  OUT_DIR="$REPO_ROOT/trial_runs/full_chain_determinism_$STAMP"
elif [[ "$OUT_DIR" != /* ]]; then
  OUT_DIR="$REPO_ROOT/$OUT_DIR"
fi

OUT_BASE="$(basename "$OUT_DIR")"
if [[ "$OUT_BASE" == "." || "$OUT_BASE" == ".." ]]; then
  die "Refusing unsafe OUT_DIR leaf: $OUT_BASE (from $OUT_DIR)" 3
fi
OUT_PARENT="$(dirname "$OUT_DIR")"
[[ -d "$OUT_PARENT" ]] || mkdir -p "$OUT_PARENT"
OUT_PARENT="$(cd "$OUT_PARENT" && pwd -P)"
OUT_DIR="$OUT_PARENT/$OUT_BASE"

if [[ "$CLEAN" == "1" ]]; then
  CLEAN_ROOT="$REPO_ROOT/trial_runs"
  [[ -d "$CLEAN_ROOT" ]] || mkdir -p "$CLEAN_ROOT"
  CLEAN_ROOT="$(cd "$CLEAN_ROOT" && pwd -P)"
  case "$OUT_DIR" in
    "/"|"$REPO_ROOT"|"$CLEAN_ROOT")
      die "--clean refuses dangerous OUT_DIR: $OUT_DIR" 3
      ;;
  esac
  if [[ "$OUT_DIR" != "$CLEAN_ROOT/"* ]]; then
    die "--clean requires OUT_DIR under $CLEAN_ROOT: $OUT_DIR" 3
  fi
fi

if [[ "$CLEAN" == "1" && -e "$OUT_DIR" ]]; then
  log "--clean set; removing existing OUT_DIR: $OUT_DIR"
  rm -rf "$OUT_DIR"
fi

mkdir -p "$OUT_DIR"

{
  echo "trial_utc=$STAMP"
  echo "repo_root=$REPO_ROOT"
  echo "input_mode=$INPUT_MODE"
  echo "input_root=${INPUT_ROOT:-}"
  echo "capture_metadata_seed=${CAPTURE_METADATA_IN:-}"
  echo "out_dir=$OUT_DIR"
  echo "runs=$RUNS"
  echo "strict=$STRICT"
  echo "do_tmp=$DO_TMP"
  echo "verbose=$VERBOSE"
  echo "hash_input=$HASH_INPUT"
  echo "compare_mode=$COMPARE_MODE"
  echo "strip_json_keys=$STRIP_JSON_KEYS"
  echo "python=$($PYTHON_BIN --version 2>&1 | tr -d '\r')"
  echo "pythonhashseed=$PYTHONHASHSEED"
  echo "lc_all=$LC_ALL"
  echo "tz=$TZ"
  echo "omp_num_threads=$OMP_NUM_THREADS"
  echo "mkl_num_threads=$MKL_NUM_THREADS"
  echo "openblas_num_threads=$OPENBLAS_NUM_THREADS"
  echo "numexpr_num_threads=$NUMEXPR_NUM_THREADS"
  if have git; then
    echo "git_head=$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || echo NA)"
    echo "git_dirty_count=$(git -C "$REPO_ROOT" status --porcelain 2>/dev/null | wc -l | tr -d ' ')"
  fi
  if have exiftool; then
    echo "exiftool_version=$(exiftool -ver 2>/dev/null || echo NA)"
  else
    echo "exiftool_version=NOT_FOUND"
  fi
  echo "tool_4c_path=$TOOL_4C"
  echo "tool_4d_path=$TOOL_4D"
  echo "tool_4e_prov_path=$TOOL_4E_P"
  echo "tool_4e_merkle_path=$TOOL_4E_M"
  echo "tool_4c_sha256=$(sha256_file "$TOOL_4C")"
  echo "tool_4d_sha256=$(sha256_file "$TOOL_4D")"
  echo "tool_4e_prov_sha256=$(sha256_file "$TOOL_4E_P")"
  echo "tool_4e_merkle_sha256=$(sha256_file "$TOOL_4E_M")"
} > "$OUT_DIR/trial_meta.txt"

log "Repo root:    $REPO_ROOT"
log "Input mode:   $INPUT_MODE"
log "Input root:   ${INPUT_ROOT:-<n/a>}"
log "Capture seed: ${CAPTURE_METADATA_IN:-<n/a>}"
log "Out dir:      $OUT_DIR"
log "Runs:         $RUNS"
log "Strict:       $STRICT"
log "Tmp run:      $DO_TMP"
log "Verbose:      $VERBOSE"
log "Hash input:   $HASH_INPUT"
log "Compare mode: $COMPARE_MODE"

capture_input_manifests() {
  local files_out="$OUT_DIR/input.files.txt"
  local sizes_out="$OUT_DIR/input.sizes.txt"
  local hashes_out="$OUT_DIR/input.sha256.txt"
  local tmp_sorted="$OUT_DIR/.input_files_abs.txt"
  local file_path rel_path

  : > "$files_out"
  : > "$sizes_out"
  if [[ "$HASH_INPUT" == "1" ]]; then
    : > "$hashes_out"
  fi

  if [[ "$INPUT_MODE" == "raw" ]]; then
    find "$INPUT_ROOT" -type f | LC_ALL=C sort > "$tmp_sorted"
    while IFS= read -r file_path; do
      rel_path="${file_path#"$INPUT_ROOT"/}"
      printf "%s\n" "$rel_path" >> "$files_out"
      printf "%s  %s\n" "$(filesize "$file_path")" "$rel_path" >> "$sizes_out"
      if [[ "$HASH_INPUT" == "1" ]]; then
        printf "%s  %s\n" "$(sha256_file "$file_path")" "$rel_path" >> "$hashes_out"
      fi
    done < "$tmp_sorted"
    rm -f "$tmp_sorted"
  else
    rel_path="$(basename "$CAPTURE_METADATA_IN")"
    printf "%s\n" "$rel_path" >> "$files_out"
    printf "%s  %s\n" "$(filesize "$CAPTURE_METADATA_IN")" "$rel_path" >> "$sizes_out"
    if [[ "$HASH_INPUT" == "1" ]]; then
      printf "%s  %s\n" "$(sha256_file "$CAPTURE_METADATA_IN")" "$rel_path" >> "$hashes_out"
    fi
  fi

  INPUT_COUNT="$(wc -l < "$files_out" | tr -d ' ')"
  [[ "$INPUT_COUNT" != "0" ]] || die "No input files discovered for mode=$INPUT_MODE." 3
}

capture_input_manifests
echo "input_count=$INPUT_COUNT" >> "$OUT_DIR/trial_meta.txt"
echo "input_manifest=$OUT_DIR/input.files.txt" >> "$OUT_DIR/trial_meta.txt"
echo "input_sizes_manifest=$OUT_DIR/input.sizes.txt" >> "$OUT_DIR/trial_meta.txt"
if [[ "$HASH_INPUT" == "1" ]]; then
  echo "input_hash_manifest=$OUT_DIR/input.sha256.txt" >> "$OUT_DIR/trial_meta.txt"
fi

log "Input files:  $INPUT_COUNT"
log "Strip keys:   ${STRIP_JSON_KEYS:-<none>}"

run_pipeline_once() {
  local run_label="$1"
  local work_cwd="$2"
  local out_run_dir="$OUT_DIR/$run_label"

  mkdir -p "$out_run_dir/artifacts"

  local capture="$out_run_dir/artifacts/capture_metadata.tp.meta.capture.v1.json"
  local manifest="$out_run_dir/artifacts/metadata_manifest.tp.meta.capture_manifest.v1.json"
  local provenance="$out_run_dir/artifacts/provenance_manifest.tp.meta.provenance.v1.json"
  local merkle="$out_run_dir/artifacts/provenance_merkle.tp.meta.provenance_merkle.v1.json"
  local logs_dir="$out_run_dir/logs"
  local log_4c="$logs_dir/4C.log"
  local log_4d="$logs_dir/4D.log"
  local log_4e_prov="$logs_dir/4E_prov.log"
  local log_4e_merkle="$logs_dir/4E_merkle.log"
  local commandline_file="$out_run_dir/commandline.txt"
  local env_file="$out_run_dir/env.txt"

  local -a cmd_4c=()
  local -a cmd_4d=("$PYTHON_BIN" "$TOOL_4D" --input "$capture" --out "$manifest" --require-fingerprint-match)
  local -a cmd_4e_provenance=("$PYTHON_BIN" "$TOOL_4E_P" --capture-metadata "$capture" --metadata-manifest "$manifest" --out "$provenance")
  local -a cmd_4e_merkle=("$PYTHON_BIN" "$TOOL_4E_M" --input "$provenance" --out "$merkle")

  if [[ "$INPUT_MODE" == "raw" ]]; then
    cmd_4c=("$PYTHON_BIN" "$TOOL_4C" --input-root "$INPUT_ROOT" --out "$capture")
    if [[ "$STRICT" == "1" ]]; then
      cmd_4c+=(--strict)
    fi
  else
    cmd_4c=(cp "$CAPTURE_METADATA_IN" "$capture")
  fi

  log "=== ${run_label}: executing from CWD=$work_cwd ==="
  mkdir -p "$logs_dir"

  {
    printf "mode=%s\n" "$INPUT_MODE"
    printf "compare_mode=%s\n" "$COMPARE_MODE"
    printf "cwd=%s\n" "$work_cwd"
    printf "cmd_4c="
    printf '%q ' "${cmd_4c[@]}"
    printf "\n"
    printf "cmd_4d="
    printf '%q ' "${cmd_4d[@]}"
    printf "\n"
    printf "cmd_4e_provenance="
    printf '%q ' "${cmd_4e_provenance[@]}"
    printf "\n"
    printf "cmd_4e_merkle="
    printf '%q ' "${cmd_4e_merkle[@]}"
    printf "\n"
  } > "$commandline_file"

  {
    printf "PYTHONHASHSEED=%s\n" "$PYTHONHASHSEED"
    printf "LC_ALL=%s\n" "$LC_ALL"
    printf "LANG=%s\n" "$LANG"
    printf "TZ=%s\n" "$TZ"
    printf "OMP_NUM_THREADS=%s\n" "$OMP_NUM_THREADS"
    printf "MKL_NUM_THREADS=%s\n" "$MKL_NUM_THREADS"
    printf "OPENBLAS_NUM_THREADS=%s\n" "$OPENBLAS_NUM_THREADS"
    printf "NUMEXPR_NUM_THREADS=%s\n" "$NUMEXPR_NUM_THREADS"
  } > "$env_file"

  (
    cd "$work_cwd"
    run_with_log "$log_4c" "${cmd_4c[@]}" || exit 1
    run_with_log "$log_4d" "${cmd_4d[@]}" || exit 1
    run_with_log "$log_4e_prov" "${cmd_4e_provenance[@]}" || exit 1
    run_with_log "$log_4e_merkle" "${cmd_4e_merkle[@]}" || exit 1
  ) || return 1

  local ledger="$out_run_dir/artifacts.sha256"
  local canon_ledger="$out_run_dir/artifacts.canon.sha256"
  local sizes="$out_run_dir/artifacts.sizes"
  local artifact canonical_hash

  : > "$ledger"
  : > "$canon_ledger"
  : > "$sizes"

  for artifact in "$capture" "$manifest" "$provenance" "$merkle"; do
    [[ -f "$artifact" ]] || die "Expected artifact missing after ${run_label}: $artifact" 4
    printf "%s  %s\n" "$(sha256_file "$artifact")" "$(basename "$artifact")" >> "$ledger"
    canonical_hash="$(canonical_json_sha256_file "$artifact")" || die "Failed canonical JSON hash for $artifact" 4
    printf "%s  %s\n" "$canonical_hash" "$(basename "$artifact")" >> "$canon_ledger"
    if stat -c %s "$artifact" >/dev/null 2>&1; then
      printf "%s  %s\n" "$(stat -c %s "$artifact")" "$(basename "$artifact")" >> "$sizes"
    else
      printf "%s  %s\n" "$(stat -f %z "$artifact")" "$(basename "$artifact")" >> "$sizes"
    fi
  done

  log "=== ${run_label}: completed ==="
}

log "Starting primary runs from repo root..."
for i in $(seq 1 "$RUNS"); do
  run_label="$(printf "run_%02d" "$i")"
  if ! run_pipeline_once "$run_label" "$REPO_ROOT"; then
    log "Pipeline failed during ${run_label}. Tail of step logs:"
    tail -200 "$OUT_DIR/$run_label/logs/"*.log 2>/dev/null || true
    die "Pipeline failed during ${run_label}" 4
  fi
done

if (( RUNS < 2 )); then
  log "RUNS=1, skipping determinism comparisons by request."
  log "Artifacts + ledgers captured under: $OUT_DIR"
  exit 0
fi

PRIMARY_RAW_FIRST="$OUT_DIR/run_01/artifacts.sha256"
PRIMARY_RAW_OTHERS=()
PRIMARY_CANON_FIRST="$OUT_DIR/run_01/artifacts.canon.sha256"
PRIMARY_CANON_OTHERS=()
PRIMARY_SIZES_FIRST="$OUT_DIR/run_01/artifacts.sizes"
PRIMARY_SIZES_OTHERS=()
for i in $(seq 2 "$RUNS"); do
  PRIMARY_RAW_OTHERS+=("$OUT_DIR/$(printf "run_%02d" "$i")/artifacts.sha256")
  PRIMARY_CANON_OTHERS+=("$OUT_DIR/$(printf "run_%02d" "$i")/artifacts.canon.sha256")
  PRIMARY_SIZES_OTHERS+=("$OUT_DIR/$(printf "run_%02d" "$i")/artifacts.sizes")
done

log "Comparing primary raw ledgers..."
primary_raw_ok=1
if ! compare_ledgers "$PRIMARY_RAW_FIRST" "${PRIMARY_RAW_OTHERS[@]}"; then
  primary_raw_ok=0
fi
log "Comparing primary canonical ledgers..."
primary_canonical_ok=1
if ! compare_ledgers "$PRIMARY_CANON_FIRST" "${PRIMARY_CANON_OTHERS[@]}"; then
  primary_canonical_ok=0
fi
if ! comparison_passes "$primary_raw_ok" "$primary_canonical_ok"; then
  log_classification "Primary" "$primary_raw_ok" "$primary_canonical_ok"
  die "Primary determinism failure detected (compare=$COMPARE_MODE). See $OUT_DIR." 5
fi
if [[ "$COMPARE_MODE" == "canonical" && "$primary_raw_ok" != "1" && "$primary_canonical_ok" == "1" ]]; then
  log "WARN: Primary raw mismatch tolerated because --compare=canonical."
fi
log "Primary comparison gate: PASS"

log "Comparing primary size ledgers..."
if ! compare_ledgers "$PRIMARY_SIZES_FIRST" "${PRIMARY_SIZES_OTHERS[@]}"; then
  die "Primary artifact-size mismatch detected. See $OUT_DIR." 5
fi
log "Primary artifact sizes: PASS"

if [[ "$DO_TMP" == "1" ]]; then
  TMPDIR_BASE="${TMPDIR:-/tmp}"
  TMP_CWD="$TMPDIR_BASE/tp_full_chain_trial_${STAMP}_$$"
  mkdir -p "$TMP_CWD"

  log "Starting /tmp runs from: $TMP_CWD"
  for i in $(seq 1 "$RUNS"); do
    run_label="$(printf "tmp_run_%02d" "$i")"
    if ! run_pipeline_once "$run_label" "$TMP_CWD"; then
      log "Pipeline failed during ${run_label} (/tmp). Tail of step logs:"
      tail -200 "$OUT_DIR/$run_label/logs/"*.log 2>/dev/null || true
      die "Pipeline failed during ${run_label} (/tmp)" 4
    fi
  done

  TMP_RAW_FIRST="$OUT_DIR/tmp_run_01/artifacts.sha256"
  TMP_RAW_OTHERS=()
  TMP_CANON_FIRST="$OUT_DIR/tmp_run_01/artifacts.canon.sha256"
  TMP_CANON_OTHERS=()
  TMP_SIZES_FIRST="$OUT_DIR/tmp_run_01/artifacts.sizes"
  TMP_SIZES_OTHERS=()
  for i in $(seq 2 "$RUNS"); do
    TMP_RAW_OTHERS+=("$OUT_DIR/$(printf "tmp_run_%02d" "$i")/artifacts.sha256")
    TMP_CANON_OTHERS+=("$OUT_DIR/$(printf "tmp_run_%02d" "$i")/artifacts.canon.sha256")
    TMP_SIZES_OTHERS+=("$OUT_DIR/$(printf "tmp_run_%02d" "$i")/artifacts.sizes")
  done

  log "Comparing /tmp raw ledgers..."
  tmp_raw_ok=1
  if ! compare_ledgers "$TMP_RAW_FIRST" "${TMP_RAW_OTHERS[@]}"; then
    tmp_raw_ok=0
  fi
  log "Comparing /tmp canonical ledgers..."
  tmp_canonical_ok=1
  if ! compare_ledgers "$TMP_CANON_FIRST" "${TMP_CANON_OTHERS[@]}"; then
    tmp_canonical_ok=0
  fi
  if ! comparison_passes "$tmp_raw_ok" "$tmp_canonical_ok"; then
    log_classification "/tmp" "$tmp_raw_ok" "$tmp_canonical_ok"
    die "/tmp determinism failure detected (compare=$COMPARE_MODE). See $OUT_DIR." 5
  fi
  if [[ "$COMPARE_MODE" == "canonical" && "$tmp_raw_ok" != "1" && "$tmp_canonical_ok" == "1" ]]; then
    log "WARN: /tmp raw mismatch tolerated because --compare=canonical."
  fi
  log "/tmp comparison gate: PASS"

  log "Comparing /tmp size ledgers..."
  if ! compare_ledgers "$TMP_SIZES_FIRST" "${TMP_SIZES_OTHERS[@]}"; then
    die "/tmp artifact-size mismatch detected. See $OUT_DIR." 5
  fi
  log "/tmp artifact sizes: PASS"

  log "Comparing primary vs /tmp raw/canonical ledgers..."
  primary_vs_tmp_raw_ok=1
  if ! compare_ledgers "$PRIMARY_RAW_FIRST" "$TMP_RAW_FIRST"; then
    primary_vs_tmp_raw_ok=0
  fi
  primary_vs_tmp_canonical_ok=1
  if ! compare_ledgers "$PRIMARY_CANON_FIRST" "$TMP_CANON_FIRST"; then
    primary_vs_tmp_canonical_ok=0
  fi

  if ! comparison_passes "$primary_vs_tmp_raw_ok" "$primary_vs_tmp_canonical_ok"; then
    log_classification "Primary vs /tmp" "$primary_vs_tmp_raw_ok" "$primary_vs_tmp_canonical_ok"
    die "Relocatability/CWD-independence mismatch detected (compare=$COMPARE_MODE). See $OUT_DIR." 5
  fi
  if [[ "$COMPARE_MODE" == "canonical" && "$primary_vs_tmp_raw_ok" != "1" && "$primary_vs_tmp_canonical_ok" == "1" ]]; then
    log "WARN: Primary vs /tmp raw mismatch tolerated because --compare=canonical."
  fi
  log "Primary vs /tmp comparison gate: PASS"

  log "Comparing primary vs /tmp size ledgers..."
  if ! compare_ledgers "$PRIMARY_SIZES_FIRST" "$TMP_SIZES_FIRST"; then
    die "Relocatability/CWD-independence size mismatch detected. See $OUT_DIR." 5
  fi
  log "Primary vs /tmp sizes: PASS"
fi

log "ALL CHECKS PASSED"
log "Artifacts + ledgers captured under: $OUT_DIR"
exit 0
