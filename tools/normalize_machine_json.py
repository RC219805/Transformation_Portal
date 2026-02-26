#!/usr/bin/env python3
"""Normalize ingest machine/provenance JSON under the governed ingest_v1 profile."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

from transformation_portal.ingest.normalize_machine_json import DEFAULT_NORMALIZATION_PROFILE, normalize_machine_json_bytes

EXIT_SUCCESS = 0
EXIT_INPUT_ERROR = 2
EXIT_OUTPUT_ERROR = 3
EXIT_NORMALIZATION_ERROR = 4


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="input_path", default="-", help="Input JSON path (default: stdin).")
    parser.add_argument("--out", dest="output_path", default="-", help="Output JSON path (default: stdout).")
    parser.add_argument(
        "--profile",
        default=DEFAULT_NORMALIZATION_PROFILE,
        help=f"Normalization profile (default: {DEFAULT_NORMALIZATION_PROFILE}).",
    )
    parser.add_argument(
        "--emit-sha256",
        action="store_true",
        help="Emit SHA256 over normalized canonical bytes to stderr.",
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
        normalized_bytes = normalize_machine_json_bytes(raw_input, profile=args.profile)
    except Exception as exc:  # noqa: BLE001 - CLI returns deterministic exit code on normalization failures.
        print(f"Normalization failed: {exc}", file=sys.stderr)
        return EXIT_NORMALIZATION_ERROR

    try:
        _write_output(args.output_path, normalized_bytes)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_OUTPUT_ERROR

    if args.emit_sha256:
        print(hashlib.sha256(normalized_bytes).hexdigest(), file=sys.stderr)

    return EXIT_SUCCESS


if __name__ == "__main__":
    raise SystemExit(main())
