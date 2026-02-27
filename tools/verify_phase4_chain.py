#!/usr/bin/env python3
"""Phase 4F standalone verifier for the Phase 4C/4D/4E capture provenance chain."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tp.phase4.verify_phase4_chain import (  # noqa: E402
    Phase4AlignmentError,
    Phase4MerkleMismatchError,
    Phase4MetadataHashMismatchError,
    Phase4ProvenanceEntryHashMismatchError,
    Phase4SchemaValidationError,
    Phase4VerificationInputError,
    verify_phase4_chain_from_paths,
)

# ADR-041 freezes verifier routing to the dedicated 31-37 range.
EXIT_SUCCESS = 0
EXIT_MALFORMED_INPUT = 31
EXIT_SCHEMA_VALIDATION_FAILURE = 32
EXIT_ALIGNMENT_FAILURE = 33
EXIT_METADATA_HASH_MISMATCH = 34
EXIT_PROVENANCE_ENTRY_HASH_MISMATCH = 35
EXIT_MERKLE_MISMATCH = 36

DEFAULT_METADATA_SCHEMA_PATH = PROJECT_ROOT / "schemas" / "phase4" / "metadata.schema.json"
DEFAULT_METADATA_MANIFEST_SCHEMA_PATH = PROJECT_ROOT / "schemas" / "phase4" / "metadata_manifest.schema.json"
DEFAULT_PROVENANCE_MANIFEST_SCHEMA_PATH = PROJECT_ROOT / "schemas" / "phase4" / "provenance_manifest.schema.json"
DEFAULT_PROVENANCE_MERKLE_SCHEMA_PATH = PROJECT_ROOT / "schemas" / "phase4" / "provenance_merkle.schema.json"


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
    return parser.parse_args(argv)


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

    try:
        verify_phase4_chain_from_paths(
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
        return EXIT_MALFORMED_INPUT
    except Phase4SchemaValidationError as exc:
        print(f"Schema validation failure: {exc}", file=sys.stderr)
        return EXIT_SCHEMA_VALIDATION_FAILURE
    except Phase4AlignmentError as exc:
        print(f"Alignment failure: {exc}", file=sys.stderr)
        return EXIT_ALIGNMENT_FAILURE
    except Phase4MetadataHashMismatchError as exc:
        print(f"Metadata hash mismatch: {exc}", file=sys.stderr)
        return EXIT_METADATA_HASH_MISMATCH
    except Phase4ProvenanceEntryHashMismatchError as exc:
        print(f"Provenance entry hash mismatch: {exc}", file=sys.stderr)
        return EXIT_PROVENANCE_ENTRY_HASH_MISMATCH
    except Phase4MerkleMismatchError as exc:
        print(f"Merkle mismatch: {exc}", file=sys.stderr)
        return EXIT_MERKLE_MISMATCH

    return EXIT_SUCCESS


if __name__ == "__main__":
    raise SystemExit(main())
