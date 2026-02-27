#!/usr/bin/env python3
"""Phase 4F standalone verifier for the Phase 4C/4D/4E capture provenance chain."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence
from uuid import uuid4

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tp.phase4.verify_phase4_chain import (  # noqa: E402
    FAILURE_LABEL_ALIGNMENT_FAILURE,
    FAILURE_LABEL_MALFORMED_INPUT,
    FAILURE_LABEL_MERKLE_MISMATCH,
    FAILURE_LABEL_METADATA_HASH_MISMATCH,
    FAILURE_LABEL_PROVENANCE_ENTRY_HASH_MISMATCH,
    FAILURE_LABEL_SCHEMA_VALIDATION_FAILURE,
    Phase4AlignmentError,
    Phase4MerkleMismatchError,
    Phase4MetadataHashMismatchError,
    Phase4ProvenanceEntryHashMismatchError,
    Phase4SchemaValidationError,
    Phase4VerificationInputError,
    build_verification_report_payload,
    collect_report_inputs_from_paths,
    default_failure_computed_block,
    serialize_verification_report_payload,
    validate_verification_report_payload,
    verify_phase4_chain_from_paths,
)

EXIT_SUCCESS = 0
EXIT_MALFORMED_INPUT = 31
EXIT_SCHEMA_VALIDATION_FAILURE = 32
EXIT_ALIGNMENT_FAILURE = 33
EXIT_METADATA_HASH_MISMATCH = 34
EXIT_PROVENANCE_ENTRY_HASH_MISMATCH = 35
EXIT_MERKLE_MISMATCH = 36
EXIT_REPORT_WRITE_FAILURE = 37

DEFAULT_METADATA_SCHEMA_PATH = PROJECT_ROOT / "schemas" / "phase4" / "metadata.schema.json"
DEFAULT_METADATA_MANIFEST_SCHEMA_PATH = PROJECT_ROOT / "schemas" / "phase4" / "metadata_manifest.schema.json"
DEFAULT_PROVENANCE_MANIFEST_SCHEMA_PATH = PROJECT_ROOT / "schemas" / "phase4" / "provenance_manifest.schema.json"
DEFAULT_PROVENANCE_MERKLE_SCHEMA_PATH = PROJECT_ROOT / "schemas" / "phase4" / "provenance_merkle.schema.json"
DEFAULT_VERIFICATION_REPORT_SCHEMA_PATH = PROJECT_ROOT / "schemas" / "phase4" / "verification_report.schema.json"


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        tmp_path.write_bytes(data)
        tmp_path.replace(path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--capture-metadata",
        required=True,
        help="Input Phase 4C artifact path (capture_metadata.tp.meta.capture.v1.json).",
    )
    parser.add_argument(
        "--metadata-manifest",
        required=True,
        help="Input Phase 4D artifact path (metadata_manifest.tp.meta.capture_manifest.v1.json).",
    )
    parser.add_argument(
        "--provenance-manifest",
        required=True,
        help="Input Phase 4E artifact path (provenance_manifest.tp.meta.provenance.v1.json).",
    )
    parser.add_argument(
        "--provenance-merkle",
        required=True,
        help="Input Phase 4E artifact path (provenance_merkle.tp.meta.provenance_merkle.v1.json).",
    )
    parser.add_argument(
        "--strict-input-order",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require inputs to already be sorted by relative_path where ordering is defined (default: true).",
    )
    parser.add_argument(
        "--out-report",
        default=None,
        help="Optional output path for tp.meta.verification_report.v1 artifact.",
    )
    parser.add_argument(
        "--write-report-on-failure",
        action="store_true",
        help="When --out-report is set, emit a deterministic failure report if verification fails.",
    )
    parser.add_argument(
        "--verification-report-schema",
        default=str(DEFAULT_VERIFICATION_REPORT_SCHEMA_PATH),
        help=f"Verification report schema path (default: {DEFAULT_VERIFICATION_REPORT_SCHEMA_PATH}).",
    )
    return parser.parse_args(argv)


def _build_report_for_success(
    *,
    verification_result: dict[str, object],
) -> dict[str, object]:
    return build_verification_report_payload(
        inputs=dict(verification_result["inputs"]),
        computed=dict(verification_result["computed"]),
        passed=True,
    )


def _build_report_for_failure(
    *,
    capture_metadata_path: Path,
    metadata_manifest_path: Path,
    provenance_manifest_path: Path,
    provenance_merkle_path: Path,
    failure_label: str,
    failure_message: str,
) -> dict[str, object]:
    return build_verification_report_payload(
        inputs=collect_report_inputs_from_paths(
            capture_metadata_path=capture_metadata_path,
            metadata_manifest_path=metadata_manifest_path,
            provenance_manifest_path=provenance_manifest_path,
            provenance_merkle_path=provenance_merkle_path,
        ),
        computed=default_failure_computed_block(),
        passed=False,
        failure_code_label=failure_label,
        failure_message=failure_message,
    )


def _write_report_or_exit(
    *,
    report_path: Path,
    report_payload: dict[str, object],
    report_schema_path: Path,
) -> int:
    try:
        report_schema_payload = json.loads(report_schema_path.read_text(encoding="utf-8"))
        if not isinstance(report_schema_payload, dict):
            raise Phase4SchemaValidationError("verification report schema must be a JSON object")
        validate_verification_report_payload(report_payload, report_schema=report_schema_payload)
        report_bytes = serialize_verification_report_payload(report_payload)
    except Phase4SchemaValidationError as exc:
        print(f"Schema validation failure: {exc}", file=sys.stderr)
        return EXIT_SCHEMA_VALIDATION_FAILURE
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        print(f"Schema validation failure: unable to prepare verification report: {exc}", file=sys.stderr)
        return EXIT_SCHEMA_VALIDATION_FAILURE

    try:
        _atomic_write_bytes(report_path, report_bytes)
    except OSError as exc:
        print(f"Report write failure: {exc}", file=sys.stderr)
        return EXIT_REPORT_WRITE_FAILURE
    return EXIT_SUCCESS


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = _parse_args(argv)
    except SystemExit as exc:
        code = int(exc.code)
        if code == 0:
            return EXIT_SUCCESS
        return EXIT_MALFORMED_INPUT

    capture_metadata_path = Path(args.capture_metadata)
    metadata_manifest_path = Path(args.metadata_manifest)
    provenance_manifest_path = Path(args.provenance_manifest)
    provenance_merkle_path = Path(args.provenance_merkle)
    out_report_path = Path(args.out_report) if args.out_report else None
    report_schema_path = Path(args.verification_report_schema)

    if args.write_report_on_failure and out_report_path is None:
        print("Malformed input: --write-report-on-failure requires --out-report", file=sys.stderr)
        return EXIT_MALFORMED_INPUT

    try:
        verification_result = verify_phase4_chain_from_paths(
            capture_metadata_path=capture_metadata_path,
            metadata_manifest_path=metadata_manifest_path,
            provenance_manifest_path=provenance_manifest_path,
            provenance_merkle_path=provenance_merkle_path,
            metadata_schema_path=DEFAULT_METADATA_SCHEMA_PATH,
            metadata_manifest_schema_path=DEFAULT_METADATA_MANIFEST_SCHEMA_PATH,
            provenance_manifest_schema_path=DEFAULT_PROVENANCE_MANIFEST_SCHEMA_PATH,
            provenance_merkle_schema_path=DEFAULT_PROVENANCE_MERKLE_SCHEMA_PATH,
            strict_input_order=bool(args.strict_input_order),
        )
    except Phase4VerificationInputError as exc:
        print(f"Malformed input: {exc}", file=sys.stderr)
        if out_report_path is not None and args.write_report_on_failure:
            report_payload = _build_report_for_failure(
                capture_metadata_path=capture_metadata_path,
                metadata_manifest_path=metadata_manifest_path,
                provenance_manifest_path=provenance_manifest_path,
                provenance_merkle_path=provenance_merkle_path,
                failure_label=FAILURE_LABEL_MALFORMED_INPUT,
                failure_message=str(exc),
            )
            report_exit_code = _write_report_or_exit(
                report_path=out_report_path,
                report_payload=report_payload,
                report_schema_path=report_schema_path,
            )
            if report_exit_code != EXIT_SUCCESS:
                return report_exit_code
        return EXIT_MALFORMED_INPUT
    except Phase4SchemaValidationError as exc:
        print(f"Schema validation failure: {exc}", file=sys.stderr)
        if out_report_path is not None and args.write_report_on_failure:
            report_payload = _build_report_for_failure(
                capture_metadata_path=capture_metadata_path,
                metadata_manifest_path=metadata_manifest_path,
                provenance_manifest_path=provenance_manifest_path,
                provenance_merkle_path=provenance_merkle_path,
                failure_label=FAILURE_LABEL_SCHEMA_VALIDATION_FAILURE,
                failure_message=str(exc),
            )
            report_exit_code = _write_report_or_exit(
                report_path=out_report_path,
                report_payload=report_payload,
                report_schema_path=report_schema_path,
            )
            if report_exit_code != EXIT_SUCCESS:
                return report_exit_code
        return EXIT_SCHEMA_VALIDATION_FAILURE
    except Phase4AlignmentError as exc:
        print(f"Alignment failure: {exc}", file=sys.stderr)
        if out_report_path is not None and args.write_report_on_failure:
            report_payload = _build_report_for_failure(
                capture_metadata_path=capture_metadata_path,
                metadata_manifest_path=metadata_manifest_path,
                provenance_manifest_path=provenance_manifest_path,
                provenance_merkle_path=provenance_merkle_path,
                failure_label=FAILURE_LABEL_ALIGNMENT_FAILURE,
                failure_message=str(exc),
            )
            report_exit_code = _write_report_or_exit(
                report_path=out_report_path,
                report_payload=report_payload,
                report_schema_path=report_schema_path,
            )
            if report_exit_code != EXIT_SUCCESS:
                return report_exit_code
        return EXIT_ALIGNMENT_FAILURE
    except Phase4MetadataHashMismatchError as exc:
        print(f"Metadata hash mismatch: {exc}", file=sys.stderr)
        if out_report_path is not None and args.write_report_on_failure:
            report_payload = _build_report_for_failure(
                capture_metadata_path=capture_metadata_path,
                metadata_manifest_path=metadata_manifest_path,
                provenance_manifest_path=provenance_manifest_path,
                provenance_merkle_path=provenance_merkle_path,
                failure_label=FAILURE_LABEL_METADATA_HASH_MISMATCH,
                failure_message=str(exc),
            )
            report_exit_code = _write_report_or_exit(
                report_path=out_report_path,
                report_payload=report_payload,
                report_schema_path=report_schema_path,
            )
            if report_exit_code != EXIT_SUCCESS:
                return report_exit_code
        return EXIT_METADATA_HASH_MISMATCH
    except Phase4ProvenanceEntryHashMismatchError as exc:
        print(f"Provenance entry hash mismatch: {exc}", file=sys.stderr)
        if out_report_path is not None and args.write_report_on_failure:
            report_payload = _build_report_for_failure(
                capture_metadata_path=capture_metadata_path,
                metadata_manifest_path=metadata_manifest_path,
                provenance_manifest_path=provenance_manifest_path,
                provenance_merkle_path=provenance_merkle_path,
                failure_label=FAILURE_LABEL_PROVENANCE_ENTRY_HASH_MISMATCH,
                failure_message=str(exc),
            )
            report_exit_code = _write_report_or_exit(
                report_path=out_report_path,
                report_payload=report_payload,
                report_schema_path=report_schema_path,
            )
            if report_exit_code != EXIT_SUCCESS:
                return report_exit_code
        return EXIT_PROVENANCE_ENTRY_HASH_MISMATCH
    except Phase4MerkleMismatchError as exc:
        print(f"Merkle mismatch: {exc}", file=sys.stderr)
        if out_report_path is not None and args.write_report_on_failure:
            report_payload = _build_report_for_failure(
                capture_metadata_path=capture_metadata_path,
                metadata_manifest_path=metadata_manifest_path,
                provenance_manifest_path=provenance_manifest_path,
                provenance_merkle_path=provenance_merkle_path,
                failure_label=FAILURE_LABEL_MERKLE_MISMATCH,
                failure_message=str(exc),
            )
            report_exit_code = _write_report_or_exit(
                report_path=out_report_path,
                report_payload=report_payload,
                report_schema_path=report_schema_path,
            )
            if report_exit_code != EXIT_SUCCESS:
                return report_exit_code
        return EXIT_MERKLE_MISMATCH

    if out_report_path is not None:
        report_payload = _build_report_for_success(verification_result=verification_result)
        report_exit_code = _write_report_or_exit(
            report_path=out_report_path,
            report_payload=report_payload,
            report_schema_path=report_schema_path,
        )
        if report_exit_code != EXIT_SUCCESS:
            return report_exit_code

    return EXIT_SUCCESS


if __name__ == "__main__":
    raise SystemExit(main())
