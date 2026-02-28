#!/usr/bin/env bash
set -euo pipefail

ARCHIVE_INDEX=""
ARCHIVE_ROOT=""
OUT_ROOT="archive_reports/fixity"
WORKERS="1"
STRICT="false"
STRICT_IDENTITY="false"
VALIDATE_SCHEMAS="true"
PYTHON_BIN="${PYTHON_BIN:-python3}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --archive-index)
      ARCHIVE_INDEX="$2"
      shift 2
      ;;
    --archive-root)
      ARCHIVE_ROOT="$2"
      shift 2
      ;;
    --out-root)
      OUT_ROOT="$2"
      shift 2
      ;;
    --workers)
      WORKERS="$2"
      shift 2
      ;;
    --strict)
      STRICT="true"
      shift
      ;;
    --strict-identity)
      STRICT_IDENTITY="true"
      shift
      ;;
    --no-validate-schemas)
      VALIDATE_SCHEMAS="false"
      shift
      ;;
    -h|--help)
      cat <<'EOF'
Usage: run_fixity_cycle.sh --archive-index <path> --archive-root <path> [options]

Options:
  --out-root <path>           Output root (default: archive_reports/fixity)
  --workers <n>               Hash/verify worker count (default: 1)
  --strict                    Fail hash scan on any non-ok rows
  --strict-identity           Fail hash scan on duplicate identity keys
  --no-validate-schemas       Disable archive hash schema validation
EOF
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if [[ -z "$ARCHIVE_INDEX" || -z "$ARCHIVE_ROOT" ]]; then
  echo "--archive-index and --archive-root are required" >&2
  exit 2
fi

TODAY_UTC="$(date -u +%F)"
RUN_ID="$(date -u +%H%M%S)"
RUN_DIR="${OUT_ROOT}/${TODAY_UTC}/${RUN_ID}"
SCAN_DIR="${RUN_DIR}/scan"
VERIFY_DIR="${RUN_DIR}/verify"
REPORT_PATH="${VERIFY_DIR}/verification_report.json"

mkdir -p "$SCAN_DIR" "$VERIFY_DIR"

HASH_CMD=("$PYTHON_BIN" tools/archive_hash_manifest.py
  --archive-index "$ARCHIVE_INDEX"
  --archive-root "$ARCHIVE_ROOT"
  --out-dir "$SCAN_DIR"
  --workers "$WORKERS"
)
if [[ "$STRICT" == "true" ]]; then
  HASH_CMD+=(--strict)
fi
if [[ "$STRICT_IDENTITY" == "true" ]]; then
  HASH_CMD+=(--strict-identity)
fi
if [[ "$VALIDATE_SCHEMAS" == "true" ]]; then
  HASH_CMD+=(--validate-schemas)
fi

"${HASH_CMD[@]}"

"$PYTHON_BIN" tools/verify_hash_manifest.py \
  --hash-manifest "$SCAN_DIR/hash_manifest.csv.gz" \
  --archive-root "$ARCHIVE_ROOT" \
  --report-path "$REPORT_PATH" \
  --workers "$WORKERS" \
  --verify-all

"$PYTHON_BIN" - <<'PY' "$RUN_DIR" "$SCAN_DIR" "$REPORT_PATH"
from __future__ import annotations

import json
from pathlib import Path
import sys

run_dir = Path(sys.argv[1])
scan_dir = Path(sys.argv[2])
report_path = Path(sys.argv[3])

summary = {
    "schema_version": "tp.archive.fixity_cycle.summary.v1",
    "run_dir": str(run_dir),
    "hash_manifest": str(scan_dir / "hash_manifest.csv.gz"),
    "hash_summary": str(scan_dir / "hash_summary.json"),
    "merkle_roots": str(scan_dir / "merkle_roots.json"),
    "verification_report": str(report_path),
}
(run_dir / "fixity_cycle_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

echo "Fixity cycle completed"
echo "Run directory: $RUN_DIR"
