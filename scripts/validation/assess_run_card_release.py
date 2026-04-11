#!/usr/bin/env python3
"""Assess Lux run-card release readiness under policy-driven trust gates."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running from source checkout without pip install
_SRC = Path(__file__).resolve().parents[2] / "src"
if _SRC.is_dir():
    sys.path.insert(0, str(_SRC))

from transformation_portal.lux_depth_v3.validators.release_assessment import assess_run_card_release


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_card", help="Path to run_card_*.json")
    parser.add_argument("--require-v2", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require-native-attestation", action="store_true")
    parser.add_argument("--require-dsse-attestation", action="store_true")
    parser.add_argument("--require-sigstore-bundle", action="store_true")
    parser.add_argument("--require-rekor-inclusion", action="store_true")
    parser.add_argument("--gpg", action="store_true", help="Verify native/DSSE GPG signatures when present.")
    parser.add_argument("--sigstore-key", default=None, help="Optional cosign key path for Sigstore bundle verification.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    assessment = assess_run_card_release(
        run_card_path=Path(args.run_card),
        require_v2=bool(args.require_v2),
        require_native_attestation=bool(args.require_native_attestation),
        require_dsse_attestation=bool(args.require_dsse_attestation),
        require_sigstore_bundle=bool(args.require_sigstore_bundle),
        require_rekor_inclusion=bool(args.require_rekor_inclusion),
        verify_gpg=bool(args.gpg),
        cosign_key_path=(Path(args.sigstore_key) if args.sigstore_key else None),
    )
    print(json.dumps(assessment, indent=2, sort_keys=True))
    return 0 if assessment["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
