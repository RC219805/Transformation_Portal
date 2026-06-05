#!/usr/bin/env python3
"""Verify run-card integrity invariants."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running from source checkout without pip install.
_SRC = Path(__file__).resolve().parents[2] / "src"
if _SRC.is_dir():
    sys.path.insert(0, str(_SRC))

from transformation_portal.lux_depth_v3.artifact_manager import compute_artifact_merkle_root as _compute_artifact_merkle_root
from transformation_portal.lux_depth_v3.validators.run_card_integrity import (
    DEFAULT_SCHEMA_V1_PATH,
    verify_run_card_integrity,
)

compute_artifact_merkle_root = _compute_artifact_merkle_root


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify run-card integrity invariants.")
    parser.add_argument("run_cards", nargs="+", help="Run card JSON file path(s)")
    parser.add_argument(
        "--schema-path",
        type=Path,
        default=None,
        help=f"Optional schema override (default autodetects v1/v2; v1 path: {DEFAULT_SCHEMA_V1_PATH})",
    )
    parser.add_argument(
        "--check-canonical-json",
        action="store_true",
        help="Fail if file text is not canonical JSON serialization (sort_keys=True, indent=2).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    exit_code = 0

    for run_card_arg in args.run_cards:
        run_card_path = Path(run_card_arg)
        errors = verify_run_card_integrity(
            run_card_path,
            schema_path=args.schema_path,
            check_canonical_json=args.check_canonical_json,
        )
        if errors:
            exit_code = 1
            print(f"❌ Run card integrity verification failed: {run_card_path}")
            for error in errors:
                print(f"  - {error}")
        else:
            print(f"✅ Run card integrity verified: {run_card_path}")

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
