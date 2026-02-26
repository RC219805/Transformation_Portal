#!/usr/bin/env python3
"""Verify Phase 4D metadata hash + manifest byte parity across Python runtimes."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

EXIT_RUNTIME_FAILURE = 31
EXIT_PARITY_MISMATCH = 32

DEFAULT_CAPTURE_JSON_FILENAME = "expected_capture_metadata.tp.meta.capture.v1.json"
DEFAULT_CAPTURE_JSON = Path(__file__).resolve().parents[1] / "tests" / "golden" / "phase4" / DEFAULT_CAPTURE_JSON_FILENAME
DEFAULT_EXPECTED_MANIFEST = (
    Path(__file__).resolve().parents[1]
    / "tests"
    / "golden"
    / "phase4"
    / "expected_metadata_manifest.tp.meta.capture_manifest.v1.json"
)


def _run_checked(command: list[str], *, cwd: Path) -> str:
    result = subprocess.run(
        command,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"command failed (exit {result.returncode}): {' '.join(command)}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
    return result.stdout.strip()


def _compute_phase4d_outputs(
    *,
    project_root: Path,
    capture_json: Path,
) -> dict[str, Any]:
    for import_root in (project_root, project_root / "src"):
        import_root_str = str(import_root)
        if import_root_str not in sys.path:
            sys.path.insert(0, import_root_str)

    from tp.phase4.hash_capture_metadata import (  # pylint: disable=import-outside-toplevel
        METADATA_CONTRACT_VERSION,
        METADATA_MANIFEST_CONTRACT_VERSION,
        compute_metadata_sha256,
        serialize_metadata_manifest,
    )

    records = json.loads(capture_json.read_text(encoding="utf-8"))
    sorted_records = sorted(records, key=lambda record: record["relative_path"])
    entries = [
        {
            "relative_path": record["relative_path"],
            "file_sha256": record["file_sha256"],
            "metadata_sha256": compute_metadata_sha256(record),
        }
        for record in sorted_records
    ]
    payload = {
        "metadata_manifest_contract_version": METADATA_MANIFEST_CONTRACT_VERSION,
        "metadata_contract_version": METADATA_CONTRACT_VERSION,
        "entries": entries,
    }
    manifest_bytes = serialize_metadata_manifest(payload)
    return {
        "manifest_bytes_b64": base64.b64encode(manifest_bytes).decode("ascii"),
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "metadata_sha256_list": [entry["metadata_sha256"] for entry in entries],
    }


def _run_probe(
    *,
    python_executable: str,
    script_path: Path,
    project_root: Path,
    capture_json: Path,
) -> dict[str, Any]:
    stdout = _run_checked(
        [
            python_executable,
            str(script_path),
            "--probe",
            "--project-root",
            str(project_root),
            "--capture-json",
            str(capture_json),
        ],
        cwd=project_root,
    )
    return json.loads(stdout)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python-a", help="First Python interpreter path.")
    parser.add_argument("--python-b", help="Second Python interpreter path.")
    parser.add_argument(
        "--project-root",
        default=str(Path(__file__).resolve().parents[1]),
        help="Project root containing tp/ and tests/.",
    )
    parser.add_argument(
        "--capture-json",
        default=str(DEFAULT_CAPTURE_JSON),
        help="Capture metadata JSON used to derive Phase 4D hashes.",
    )
    parser.add_argument(
        "--expected-manifest",
        default=str(DEFAULT_EXPECTED_MANIFEST),
        help="Golden expected metadata manifest bytes.",
    )
    parser.add_argument(
        "--probe",
        action="store_true",
        help="Internal mode: emit runtime-specific Phase 4D outputs as JSON.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    project_root = Path(args.project_root).resolve()
    capture_json = Path(args.capture_json).resolve()

    if args.probe:
        payload = _compute_phase4d_outputs(project_root=project_root, capture_json=capture_json)
        print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
        return 0

    if not args.python_a or not args.python_b:
        print("Cross-runtime parity check requires --python-a and --python-b")
        return EXIT_RUNTIME_FAILURE

    expected_manifest = Path(args.expected_manifest).resolve()
    expected_manifest_bytes = expected_manifest.read_bytes()
    expected_manifest_sha256 = hashlib.sha256(expected_manifest_bytes).hexdigest()
    expected_manifest_b64 = base64.b64encode(expected_manifest_bytes).decode("ascii")
    expected_manifest_payload = json.loads(expected_manifest_bytes.decode("utf-8"))
    expected_hashes = [entry["metadata_sha256"] for entry in expected_manifest_payload["entries"]]

    script_path = Path(__file__).resolve()
    try:
        output_a = _run_probe(
            python_executable=args.python_a,
            script_path=script_path,
            project_root=project_root,
            capture_json=capture_json,
        )
        output_b = _run_probe(
            python_executable=args.python_b,
            script_path=script_path,
            project_root=project_root,
            capture_json=capture_json,
        )
    except RuntimeError as exc:
        print(f"Phase 4D cross-runtime parity check failed: {exc}")
        return EXIT_RUNTIME_FAILURE

    if output_a != output_b:
        print("Phase 4D cross-runtime parity check failed: runtime outputs diverged")
        print(f"python_a output: {json.dumps(output_a, sort_keys=True)}")
        print(f"python_b output: {json.dumps(output_b, sort_keys=True)}")
        return EXIT_PARITY_MISMATCH

    manifest_b64 = output_a["manifest_bytes_b64"]
    manifest_sha256 = output_a["manifest_sha256"]
    metadata_hashes = output_a["metadata_sha256_list"]

    if manifest_b64 != expected_manifest_b64:
        print("Phase 4D cross-runtime parity check failed: manifest bytes do not match golden")
        return EXIT_PARITY_MISMATCH
    if manifest_sha256 != expected_manifest_sha256:
        print("Phase 4D cross-runtime parity check failed: manifest SHA256 does not match golden")
        return EXIT_PARITY_MISMATCH
    if metadata_hashes != expected_hashes:
        print("Phase 4D cross-runtime parity check failed: metadata_sha256 list does not match golden")
        return EXIT_PARITY_MISMATCH

    print(f"manifest_sha256: {manifest_sha256}")
    print("Phase 4D cross-runtime parity check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
