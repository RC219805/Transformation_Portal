#!/usr/bin/env python3
"""Phase 4C deterministic capture metadata extractor."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tp.phase4.canonicalize_capture_metadata import (
    ConfigValidationError,
    ExtractionFailure,
    PathNormalizationError,
    SchemaValidationError,
    StrictWarningsError,
    extract_capture_metadata_records,
    load_capture_metadata_config,
    write_capture_metadata_artifact,
)

EXIT_SUCCESS = 0
EXIT_CONFIG_INVALID = 2
EXIT_PATH_NORMALIZATION_FAILURE = 3
EXIT_EXTRACTION_FAILURE = 4
EXIT_SCHEMA_VALIDATION_FAILURE = 5
EXIT_STRICT_WARNING_FAILURE = 6

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "capture_metadata_config.json"
DEFAULT_SCHEMA_PATH = PROJECT_ROOT / "schemas" / "phase4" / "metadata.schema.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True, help="Root directory containing capture files.")
    parser.add_argument(
        "--out",
        required=True,
        help="Output artifact path (for example artifacts/capture_metadata.tp.meta.capture.v1.json).",
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help=f"Canonicalization config path (default: {DEFAULT_CONFIG_PATH}).",
    )
    parser.add_argument("--strict", action="store_true", help="Fail when extraction_warnings are present.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_root = Path(args.input_root)
    out_path = Path(args.out)
    config_path = Path(args.config)

    try:
        config = load_capture_metadata_config(config_path)
    except ConfigValidationError as exc:
        print(f"Config invalid: {exc}", file=sys.stderr)
        return EXIT_CONFIG_INVALID

    try:
        records = extract_capture_metadata_records(
            input_root=input_root,
            config=config,
            strict=args.strict,
            schema_path=DEFAULT_SCHEMA_PATH,
            extractor_name="extract_capture_metadata.py",
            extractor_version="phase4c-v1",
        )
        write_capture_metadata_artifact(records, out_path=out_path)
        return EXIT_SUCCESS
    except PathNormalizationError as exc:
        print(f"Path normalization failure: {exc}", file=sys.stderr)
        return EXIT_PATH_NORMALIZATION_FAILURE
    except ExtractionFailure as exc:
        print(f"Extraction failure: {exc}", file=sys.stderr)
        return EXIT_EXTRACTION_FAILURE
    except SchemaValidationError as exc:
        print(f"Schema validation failure: {exc}", file=sys.stderr)
        return EXIT_SCHEMA_VALIDATION_FAILURE
    except StrictWarningsError as exc:
        print(f"Strict-mode warning failure: {exc}", file=sys.stderr)
        return EXIT_STRICT_WARNING_FAILURE
    except OSError as exc:
        print(f"Extraction failure: {exc}", file=sys.stderr)
        return EXIT_EXTRACTION_FAILURE


if __name__ == "__main__":
    raise SystemExit(main())
