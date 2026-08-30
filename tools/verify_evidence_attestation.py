#!/usr/bin/env python3
"""Verify that a detached attestation binds to an evidence payload."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

from transformation_portal.attestation.detached import canonical_attestation_preimage_bytes
from transformation_portal.attestation.verify import (
    bind_attestation_to_evidence,
    validate_detached_attestation_surface,
    verify_attestation_self_hash,
)

EXIT_SUCCESS = 0
EXIT_INPUT_ERROR = 2
EXIT_VERIFY_FAILED = 5
_GPG_SIGNATURE_ALGORITHM = "openpgp-clearsign"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", required=True, help="Path to tp.meta.evidence.v1 JSON.")
    parser.add_argument("--attestation", required=True, help="Path to tp.attestation.detached.v1 JSON.")
    parser.add_argument("--gpg", action="store_true", help="Verify with gpg clearsign backend.")
    parser.add_argument(
        "--allow-missing-attestation-sha",
        action="store_true",
        help="Allow missing/null attestation_sha256 for migration compatibility.",
    )
    return parser.parse_args()


def _read_json_object(path: Path, *, name: str) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Unable to read {name} JSON: {exc}") from exc

    if not isinstance(payload, dict):
        raise ValueError(f"{name} JSON must be an object")
    return payload


def main() -> int:
    args = _parse_args()

    try:
        evidence = _read_json_object(Path(args.evidence), name="evidence")
        attestation = _read_json_object(Path(args.attestation), name="attestation")
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_INPUT_ERROR

    try:
        validate_detached_attestation_surface(attestation)
        bind_attestation_to_evidence(attestation, evidence)
        verify_attestation_self_hash(attestation, require_digest=(not args.allow_missing_attestation_sha))
    except (TypeError, ValueError) as exc:
        print(f"Attestation validation failed: {exc}", file=sys.stderr)
        return EXIT_VERIFY_FAILED
    except Exception:  # noqa: BLE001 - deterministic exit code with traceback for debugging unexpected failures.
        print("Attestation validation failed with unexpected error:", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)
        return EXIT_VERIFY_FAILED

    if args.gpg:
        try:
            signature_algorithm = attestation["signature"]["algorithm"]
            if signature_algorithm != _GPG_SIGNATURE_ALGORITHM:
                raise ValueError(
                    "signature.algorithm must be "
                    f"{_GPG_SIGNATURE_ALGORITHM!r} when --gpg is enabled, got {signature_algorithm!r}"
                )

            from transformation_portal.attestation.gpg import gpg_verify_clearsign

            signature_text = str(attestation["signature"]["signature"])
            gpg_verify_clearsign(
                signature_text,
                expected_payload=canonical_attestation_preimage_bytes(evidence),
                key_id=str(attestation["signature"]["key_id"]),
            )
        except (TypeError, ValueError) as exc:
            print(f"Signature verification failed: {exc}", file=sys.stderr)
            return EXIT_VERIFY_FAILED
        except Exception:  # noqa: BLE001 - deterministic exit code with traceback for debugging unexpected failures.
            print("Signature verification failed with unexpected error:", file=sys.stderr)
            print(traceback.format_exc(), file=sys.stderr)
            return EXIT_VERIFY_FAILED

    return EXIT_SUCCESS


if __name__ == "__main__":
    raise SystemExit(main())
