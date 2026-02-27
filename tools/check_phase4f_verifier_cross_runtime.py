#!/usr/bin/env python3
"""Verify Phase 4F verifier parity across Python runtimes."""

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

PROJECT_ROOT_DEFAULT = Path(__file__).resolve().parents[1]
DEFAULT_CAPTURE_JSON = PROJECT_ROOT_DEFAULT / "tests" / "golden" / "phase4" / "expected_capture_metadata.tp.meta.capture.v1.json"
DEFAULT_METADATA_MANIFEST_JSON = (
    PROJECT_ROOT_DEFAULT / "tests" / "golden" / "phase4" / "expected_metadata_manifest.tp.meta.capture_manifest.v1.json"
)
DEFAULT_PROVENANCE_MANIFEST_JSON = (
    PROJECT_ROOT_DEFAULT / "tests" / "golden" / "phase4" / "expected_provenance_manifest.tp.meta.provenance.v1.json"
)
DEFAULT_PROVENANCE_MERKLE_JSON = (
    PROJECT_ROOT_DEFAULT / "tests" / "golden" / "phase4" / "expected_provenance_merkle.tp.meta.provenance_merkle.v1.json"
)
DEFAULT_EXPECTED_REPORT_JSON = (
    PROJECT_ROOT_DEFAULT
    / "tests"
    / "golden"
    / "phase4"
    / "expected_verification_report.tp.meta.verification_report.v1.json"
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


def _compute_phase4f_outputs(
    *,
    project_root: Path,
    capture_metadata_path: Path,
    metadata_manifest_path: Path,
    provenance_manifest_path: Path,
    provenance_merkle_path: Path,
) -> dict[str, Any]:
    for import_root in (project_root, project_root / "src"):
        import_root_str = str(import_root)
        if import_root_str not in sys.path:
            sys.path.insert(0, import_root_str)

    from tp.phase4.verify_phase4_chain import (  # pylint: disable=import-outside-toplevel
        build_verification_report_payload,
        serialize_verification_report_payload,
        verify_phase4_chain_from_paths,
    )

    verification = verify_phase4_chain_from_paths(
        capture_metadata_path=capture_metadata_path,
        metadata_manifest_path=metadata_manifest_path,
        provenance_manifest_path=provenance_manifest_path,
        provenance_merkle_path=provenance_merkle_path,
        metadata_schema_path=project_root / "schemas" / "phase4" / "metadata.schema.json",
        metadata_manifest_schema_path=project_root / "schemas" / "phase4" / "metadata_manifest.schema.json",
        provenance_manifest_schema_path=project_root / "schemas" / "phase4" / "provenance_manifest.schema.json",
        provenance_merkle_schema_path=project_root / "schemas" / "phase4" / "provenance_merkle.schema.json",
        strict_input_order=True,
    )
    report_payload = build_verification_report_payload(
        inputs=verification["inputs"],
        computed=verification["computed"],
        passed=True,
    )
    report_bytes = serialize_verification_report_payload(report_payload)
    return {
        "report_bytes_b64": base64.b64encode(report_bytes).decode("ascii"),
        "report_sha256": hashlib.sha256(report_bytes).hexdigest(),
        "inputs": verification["inputs"],
        "computed": verification["computed"],
    }


def _run_probe(
    *,
    python_executable: str,
    script_path: Path,
    project_root: Path,
    capture_metadata_path: Path,
    metadata_manifest_path: Path,
    provenance_manifest_path: Path,
    provenance_merkle_path: Path,
) -> dict[str, Any]:
    stdout = _run_checked(
        [
            python_executable,
            str(script_path),
            "--probe",
            "--project-root",
            str(project_root),
            "--capture-metadata",
            str(capture_metadata_path),
            "--metadata-manifest",
            str(metadata_manifest_path),
            "--provenance-manifest",
            str(provenance_manifest_path),
            "--provenance-merkle",
            str(provenance_merkle_path),
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
        default=str(PROJECT_ROOT_DEFAULT),
        help="Project root containing src/, schemas/, and tests/.",
    )
    parser.add_argument(
        "--capture-metadata",
        default=str(DEFAULT_CAPTURE_JSON),
        help="Capture metadata artifact path.",
    )
    parser.add_argument(
        "--metadata-manifest",
        default=str(DEFAULT_METADATA_MANIFEST_JSON),
        help="Metadata manifest artifact path.",
    )
    parser.add_argument(
        "--provenance-manifest",
        default=str(DEFAULT_PROVENANCE_MANIFEST_JSON),
        help="Provenance manifest artifact path.",
    )
    parser.add_argument(
        "--provenance-merkle",
        default=str(DEFAULT_PROVENANCE_MERKLE_JSON),
        help="Provenance merkle artifact path.",
    )
    parser.add_argument(
        "--expected-report",
        default=str(DEFAULT_EXPECTED_REPORT_JSON),
        help="Expected deterministic verification report fixture bytes.",
    )
    parser.add_argument(
        "--probe",
        action="store_true",
        help="Internal mode: emit runtime-specific Phase 4F outputs as JSON.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    project_root = Path(args.project_root).resolve()
    capture_metadata_path = Path(args.capture_metadata).resolve()
    metadata_manifest_path = Path(args.metadata_manifest).resolve()
    provenance_manifest_path = Path(args.provenance_manifest).resolve()
    provenance_merkle_path = Path(args.provenance_merkle).resolve()

    if args.probe:
        payload = _compute_phase4f_outputs(
            project_root=project_root,
            capture_metadata_path=capture_metadata_path,
            metadata_manifest_path=metadata_manifest_path,
            provenance_manifest_path=provenance_manifest_path,
            provenance_merkle_path=provenance_merkle_path,
        )
        print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
        return 0

    if not args.python_a or not args.python_b:
        print("Phase 4F cross-runtime parity check requires --python-a and --python-b")
        return EXIT_RUNTIME_FAILURE

    expected_report_path = Path(args.expected_report).resolve()
    try:
        expected_report_bytes = expected_report_path.read_bytes()
    except OSError as exc:
        print(f"Phase 4F cross-runtime parity check failed: unable to read expected report fixture: {exc}")
        return EXIT_RUNTIME_FAILURE
    expected_report_sha256 = hashlib.sha256(expected_report_bytes).hexdigest()
    expected_report_b64 = base64.b64encode(expected_report_bytes).decode("ascii")

    script_path = Path(__file__).resolve()
    try:
        output_a = _run_probe(
            python_executable=args.python_a,
            script_path=script_path,
            project_root=project_root,
            capture_metadata_path=capture_metadata_path,
            metadata_manifest_path=metadata_manifest_path,
            provenance_manifest_path=provenance_manifest_path,
            provenance_merkle_path=provenance_merkle_path,
        )
        output_b = _run_probe(
            python_executable=args.python_b,
            script_path=script_path,
            project_root=project_root,
            capture_metadata_path=capture_metadata_path,
            metadata_manifest_path=metadata_manifest_path,
            provenance_manifest_path=provenance_manifest_path,
            provenance_merkle_path=provenance_merkle_path,
        )
    except RuntimeError as exc:
        print(f"Phase 4F cross-runtime parity check failed: {exc}")
        return EXIT_RUNTIME_FAILURE

    if output_a != output_b:
        print("Phase 4F cross-runtime parity check failed: runtime outputs diverged")
        print(f"python_a output: {json.dumps(output_a, sort_keys=True)}")
        print(f"python_b output: {json.dumps(output_b, sort_keys=True)}")
        return EXIT_PARITY_MISMATCH

    report_b64 = output_a["report_bytes_b64"]
    report_sha256 = output_a["report_sha256"]
    if report_b64 != expected_report_b64:
        print("Phase 4F cross-runtime parity check failed: report bytes do not match golden")
        return EXIT_PARITY_MISMATCH
    if report_sha256 != expected_report_sha256:
        print("Phase 4F cross-runtime parity check failed: report SHA256 does not match golden")
        return EXIT_PARITY_MISMATCH

    print(f"report_sha256: {report_sha256}")
    print("Phase 4F cross-runtime parity check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
