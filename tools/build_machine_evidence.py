#!/usr/bin/env python3
"""Build a tp.meta.evidence.v1 artifact from tp.meta.machine.v1 JSON."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

from transformation_portal.ingest.evidence import build_evidence_payload, canonical_evidence_bytes, load_projection_profile

EXIT_SUCCESS = 0
EXIT_INPUT_ERROR = 2
EXIT_OUTPUT_ERROR = 3
EXIT_BUILD_ERROR = 4


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="input_path", default="-", help="Input machine JSON path (default: stdin).")
    parser.add_argument("--out", dest="output_path", default="-", help="Output evidence JSON path (default: stdout).")
    parser.add_argument(
        "--profile",
        default=None,
        help="Optional projection profile JSON path (default: tp.projection.machine_to_evidence.v1).",
    )
    parser.add_argument(
        "--emit-sha256",
        action="store_true",
        help="Emit evidence_sha256 to stderr.",
    )
    return parser.parse_args()


def _read_input(path_arg: str) -> bytes:
    if path_arg == "-":
        raw = sys.stdin.buffer.read()
        if not raw.strip():
            raise ValueError("No JSON input provided on stdin")
        return raw

    path = Path(path_arg)
    try:
        return path.read_bytes()
    except OSError as exc:
        raise ValueError(f"Unable to read input file {path}: {exc}") from exc


def _write_output(path_arg: str, payload: bytes) -> None:
    if path_arg == "-":
        sys.stdout.buffer.write(payload)
        return

    path = Path(path_arg)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.write_bytes(payload)
    except OSError as exc:
        raise ValueError(f"Unable to write output file {path}: {exc}") from exc


def main() -> int:
    args = _parse_args()

    try:
        raw_input = _read_input(args.input_path)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_INPUT_ERROR

    try:
        machine_payload = json.loads(raw_input.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        print(f"Input JSON parse failed: {exc}", file=sys.stderr)
        return EXIT_INPUT_ERROR

    if not isinstance(machine_payload, dict):
        print("Input JSON must be an object", file=sys.stderr)
        return EXIT_INPUT_ERROR

    try:
        profile = load_projection_profile(Path(args.profile) if args.profile else None)
        evidence_payload = build_evidence_payload(machine_payload, projection_profile=profile)
        evidence_bytes = canonical_evidence_bytes(evidence_payload)
    except (TypeError, ValueError) as exc:
        print(f"Evidence build failed: {exc}", file=sys.stderr)
        return EXIT_BUILD_ERROR
    except Exception:  # noqa: BLE001 - deterministic exit code with traceback for debugging unexpected failures.
        print("Evidence build failed with unexpected error:", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)
        return EXIT_BUILD_ERROR

    try:
        _write_output(args.output_path, evidence_bytes)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_OUTPUT_ERROR

    if args.emit_sha256:
        print(evidence_payload["evidence_sha256"], file=sys.stderr)

    return EXIT_SUCCESS


if __name__ == "__main__":
    raise SystemExit(main())
