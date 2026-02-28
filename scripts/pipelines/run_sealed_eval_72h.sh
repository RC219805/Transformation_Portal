#!/usr/bin/env bash
set -euo pipefail

ARCHIVE_INDEX=""
ARCHIVE_ROOT=""
OUT_ROOT="archive_reports/sealed_eval"
EVAL_COMMAND=""
WORKERS="1"
PYTHON_BIN="${PYTHON_BIN:-python3}"
SUBSET_ROOT=""
ALLOW_WRITABLE_SUBSET="false"
VALIDATE_SCHEMAS="true"

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
    --subset-root)
      SUBSET_ROOT="$2"
      shift 2
      ;;
    --eval-command)
      EVAL_COMMAND="$2"
      shift 2
      ;;
    --workers)
      WORKERS="$2"
      shift 2
      ;;
    --no-validate-schemas)
      VALIDATE_SCHEMAS="false"
      shift
      ;;
    --allow-writable-subset)
      ALLOW_WRITABLE_SUBSET="true"
      shift
      ;;
    -h|--help)
      cat <<'EOF'
Usage: run_sealed_eval_72h.sh --archive-index <path> --archive-root <path> [options]

Options:
  --out-root <path>         Output root (default: archive_reports/sealed_eval)
  --subset-root <path>      Read-only subset mount path (default: --archive-root)
  --eval-command <string>   Optional evaluation command executed between pre/post fixity checks
  --workers <n>             Worker count for hashing and verification (default: 1)
  --no-validate-schemas     Disable archive hash schema validation
  --allow-writable-subset   Bypass read-only subset enforcement (for local dev only)
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

if [[ -z "$SUBSET_ROOT" ]]; then
  SUBSET_ROOT="$ARCHIVE_ROOT"
fi
if [[ ! -d "$SUBSET_ROOT" ]]; then
  echo "Subset root does not exist or is not a directory: $SUBSET_ROOT" >&2
  exit 2
fi

TODAY_UTC="$(date -u +%F)"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_DIR="${OUT_ROOT}/${TODAY_UTC}/${RUN_ID}"
PRE_DIR="${RUN_DIR}/pre"
POST_DIR="${RUN_DIR}/post"
PRE_REPORT="${PRE_DIR}/verification_report.json"
POST_REPORT="${POST_DIR}/verification_report.json"
MOUNT_CONTRACT="${RUN_DIR}/subset_mount_contract.json"
AUDIT_DIR="${RUN_DIR}/audit_package"
mkdir -p "$PRE_DIR" "$POST_DIR" "$AUDIT_DIR"

READ_ONLY_OBSERVED="true"
if [[ -w "$SUBSET_ROOT" ]]; then
  READ_ONLY_OBSERVED="false"
fi
if [[ "$READ_ONLY_OBSERVED" != "true" && "$ALLOW_WRITABLE_SUBSET" != "true" ]]; then
  echo "Subset mount is writable: $SUBSET_ROOT" >&2
  echo "Provide a read-only mount or pass --allow-writable-subset for local development." >&2
  exit 6
fi

"$PYTHON_BIN" - <<'PY' "$MOUNT_CONTRACT" "$SUBSET_ROOT" "$READ_ONLY_OBSERVED" "$ALLOW_WRITABLE_SUBSET"
from __future__ import annotations

import json
from pathlib import Path
import sys

contract_path = Path(sys.argv[1])
subset_root = sys.argv[2]
read_only_observed = sys.argv[3] == "true"
allow_writable_subset = sys.argv[4] == "true"

payload = {
    "schema_version": "tp.archive.sealed_eval.mount_contract.v1",
    "subset_root": subset_root,
    "read_only_required": not allow_writable_subset,
    "read_only_observed": read_only_observed,
    "allow_writable_subset": allow_writable_subset,
}
contract_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

PRE_HASH_CMD=("$PYTHON_BIN" tools/archive_hash_manifest.py
  --archive-index "$ARCHIVE_INDEX"
  --archive-root "$SUBSET_ROOT"
  --out-dir "$PRE_DIR"
  --workers "$WORKERS"
)
if [[ "$VALIDATE_SCHEMAS" == "true" ]]; then
  PRE_HASH_CMD+=(--validate-schemas)
fi
"${PRE_HASH_CMD[@]}"

"$PYTHON_BIN" tools/verify_hash_manifest.py \
  --hash-manifest "$PRE_DIR/hash_manifest.csv.gz" \
  --archive-root "$SUBSET_ROOT" \
  --report-path "$PRE_REPORT" \
  --workers "$WORKERS" \
  --verify-all

EVAL_EXIT=0
if [[ -n "$EVAL_COMMAND" ]]; then
  set +e
  (
    cd "$SUBSET_ROOT"
    bash -lc "$EVAL_COMMAND"
  )
  EVAL_EXIT=$?
  set -e
fi

POST_HASH_CMD=("$PYTHON_BIN" tools/archive_hash_manifest.py
  --archive-index "$ARCHIVE_INDEX"
  --archive-root "$SUBSET_ROOT"
  --out-dir "$POST_DIR"
  --workers "$WORKERS"
)
if [[ "$VALIDATE_SCHEMAS" == "true" ]]; then
  POST_HASH_CMD+=(--validate-schemas)
fi
"${POST_HASH_CMD[@]}"

"$PYTHON_BIN" tools/verify_hash_manifest.py \
  --hash-manifest "$POST_DIR/hash_manifest.csv.gz" \
  --archive-root "$SUBSET_ROOT" \
  --report-path "$POST_REPORT" \
  --workers "$WORKERS" \
  --verify-all

"$PYTHON_BIN" - <<'PY' "$PRE_DIR/hash_manifest.csv.gz" "$POST_DIR/hash_manifest.csv.gz" "$RUN_DIR" "$PRE_REPORT" "$POST_REPORT" "$EVAL_COMMAND" "$EVAL_EXIT" "$SUBSET_ROOT" "$MOUNT_CONTRACT"
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
import sys

pre_manifest = Path(sys.argv[1])
post_manifest = Path(sys.argv[2])
run_dir = Path(sys.argv[3])
pre_report = Path(sys.argv[4])
post_report = Path(sys.argv[5])
eval_command = sys.argv[6]
eval_exit = int(sys.argv[7])
subset_root = sys.argv[8]
mount_contract = Path(sys.argv[9])


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def verification_passed(path: Path) -> bool:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return bool(payload.get("rows_mismatched") == 0)

pre_digest = digest(pre_manifest)
post_digest = digest(post_manifest)
manifest_match = pre_digest == post_digest
pre_ok = verification_passed(pre_report)
post_ok = verification_passed(post_report)

summary = {
    "schema_version": "tp.archive.sealed_eval.summary.v1",
    "run_dir": str(run_dir),
    "subset_root": subset_root,
    "subset_mount_contract": str(mount_contract),
    "pre": {
        "hash_manifest": str(pre_manifest),
        "verification_report": str(pre_report),
        "sha256": pre_digest,
        "verification_passed": pre_ok,
    },
    "post": {
        "hash_manifest": str(post_manifest),
        "verification_report": str(post_report),
        "sha256": post_digest,
        "verification_passed": post_ok,
    },
    "evaluation_command": eval_command,
    "evaluation_exit_code": eval_exit,
    "hash_manifest_match": manifest_match,
    "sealed_integrity_passed": bool(manifest_match and pre_ok and post_ok and eval_exit == 0),
}

(run_dir / "sealed_eval_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

if [[ ! -f "$RUN_DIR/sealed_eval_summary.json" ]]; then
  echo "sealed_eval_summary.json missing" >&2
  exit 4
fi

"$PYTHON_BIN" - <<'PY' "$AUDIT_DIR" "$RUN_DIR"
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import sys

audit_dir = Path(sys.argv[1])
run_dir = Path(sys.argv[2])

copy_map = {
    "subset_mount_contract.json": run_dir / "subset_mount_contract.json",
    "pre_hash_manifest.csv.gz": run_dir / "pre" / "hash_manifest.csv.gz",
    "pre_verification_report.json": run_dir / "pre" / "verification_report.json",
    "post_hash_manifest.csv.gz": run_dir / "post" / "hash_manifest.csv.gz",
    "post_verification_report.json": run_dir / "post" / "verification_report.json",
    "sealed_eval_summary.json": run_dir / "sealed_eval_summary.json",
}

for destination_name, source in sorted(copy_map.items()):
    destination = audit_dir / destination_name
    shutil.copy2(source, destination)

entries = []
for path in sorted(audit_dir.iterdir()):
    if not path.is_file() or path.name == "audit_manifest.json":
        continue
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    entries.append(
        {
            "filename": path.name,
            "sha256": digest,
            "size_bytes": int(path.stat().st_size),
        }
    )

manifest_payload = {
    "schema_version": "tp.archive.sealed_eval.audit_package.v1",
    "file_count": len(entries),
    "files": entries,
}
(audit_dir / "audit_manifest.json").write_text(json.dumps(manifest_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

INTEGRITY_PASSED="$("$PYTHON_BIN" - <<'PY' "$RUN_DIR/sealed_eval_summary.json"
from __future__ import annotations
import json
from pathlib import Path
import sys
payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print("true" if payload.get("sealed_integrity_passed") else "false")
PY
)"

if [[ "$INTEGRITY_PASSED" != "true" ]]; then
  echo "Sealed evaluation integrity check failed" >&2
  echo "Run directory: $RUN_DIR" >&2
  exit 10
fi

echo "Sealed evaluation completed"
echo "Run directory: $RUN_DIR"
echo "Audit package: $AUDIT_DIR"
