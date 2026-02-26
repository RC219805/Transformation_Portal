#!/usr/bin/env python3
"""Run deterministic mixed-media ingest batch normalization and manifest generation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from transformation_portal.ingest.batch import BATCH_MANIFEST_FILENAME, run_ingest_batch
from transformation_portal.ingest.normalize_machine_json import DEFAULT_NORMALIZATION_PROFILE

EXIT_SUCCESS = 0
EXIT_INPUT_ERROR = 2
EXIT_BATCH_FAILURE = 3


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, help="Input directory containing mixed-media files.")
    parser.add_argument("--output-dir", required=True, help="Output directory for normalized JSON and manifest.")
    parser.add_argument(
        "--profile",
        default=DEFAULT_NORMALIZATION_PROFILE,
        help=f"Normalization profile (default: {DEFAULT_NORMALIZATION_PROFILE}).",
    )
    parser.add_argument(
        "--manifest-filename",
        default=BATCH_MANIFEST_FILENAME,
        help=f"Manifest filename under --output-dir (default: {BATCH_MANIFEST_FILENAME}).",
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Disable recursive input discovery.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists() or not input_dir.is_dir():
        print(f"Input directory not found: {input_dir}", file=sys.stderr)
        return EXIT_INPUT_ERROR

    try:
        manifest = run_ingest_batch(
            input_dir=input_dir,
            output_dir=output_dir,
            profile=args.profile,
            recursive=not args.no_recursive,
            manifest_filename=args.manifest_filename,
        )
    except Exception as exc:  # noqa: BLE001 - CLI maps all batch failures to deterministic exit code.
        print(f"Batch ingest failed: {exc}", file=sys.stderr)
        return EXIT_BATCH_FAILURE

    manifest_path = output_dir / args.manifest_filename
    print(
        f"Batch ingest complete: items={manifest['item_count']} profile={manifest['normalization_profile']} "
        f"manifest={manifest_path}"
    )
    return EXIT_SUCCESS


if __name__ == "__main__":
    raise SystemExit(main())
