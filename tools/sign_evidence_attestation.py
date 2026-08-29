#!/usr/bin/env python3
"""Build and sign a tp.attestation.detached.v1 payload for evidence JSON."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from uuid import uuid4

from transformation_portal.attestation.detached import (
    build_detached_attestation_payload,
    canonical_attestation_bytes,
    canonical_attestation_preimage_bytes,
)
from transformation_portal.attestation.gpg import gpg_clearsign_bytes, gpg_verify_clearsign

EXIT_SUCCESS = 0
EXIT_INPUT_ERROR = 2
EXIT_OUTPUT_ERROR = 3
EXIT_BUILD_ERROR = 4


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        tmp_path.write_bytes(data)
        tmp_path.replace(path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", required=True, help="Path to tp.meta.evidence.v1 JSON.")
    parser.add_argument("--out", required=True, help="Path to write tp.attestation.detached.v1 JSON.")
    parser.add_argument("--gpg", action="store_true", help="Use gpg clearsign backend.")
    parser.add_argument("--gpg-key-id", default=None, help="Optional gpg key id for signing.")
    parser.add_argument(
        "--key-id",
        required=True,
        help="GPG-resolvable primary key ID or fingerprint to record in attestation.signature.key_id.",
    )
    parser.add_argument("--algorithm", default="openpgp-clearsign", help="Signature algorithm label.")
    parser.add_argument("--signed-at", default=None, help="RFC3339 UTC timestamp, e.g. 2026-02-26T22:33:17Z.")
    parser.add_argument(
        "--no-recompute-check",
        action="store_true",
        help=(
            "DEPRECATED: disable the evidence_sha256 recompute guard for forensic migration only; "
            "default verification still rejects inconsistent evidence."
        ),
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
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_INPUT_ERROR

    try:
        preimage_bytes = canonical_attestation_preimage_bytes(evidence)

        if not args.gpg:
            raise ValueError("No signing backend selected (use --gpg)")
        if args.algorithm != "openpgp-clearsign":
            raise ValueError("--gpg requires --algorithm 'openpgp-clearsign'")
        if args.no_recompute_check:
            print(
                "Warning: --no-recompute-check is deprecated and does not relax verifier binding checks.",
                file=sys.stderr,
            )

        signature_text = gpg_clearsign_bytes(preimage_bytes, key_id=args.gpg_key_id)
        gpg_verify_clearsign(
            signature_text,
            expected_payload=preimage_bytes,
            key_id=args.key_id,
        )
        signature_block = {
            "algorithm": args.algorithm,
            "key_id": args.key_id,
            "signature": signature_text,
        }

        attestation_payload = build_detached_attestation_payload(
            evidence,
            signature=signature_block,
            signed_at=args.signed_at,
            enforce_recompute_match=(not args.no_recompute_check),
        )
        output_bytes = canonical_attestation_bytes(attestation_payload)
    except (TypeError, ValueError) as exc:
        print(f"Attestation build failed: {exc}", file=sys.stderr)
        return EXIT_BUILD_ERROR
    except Exception:  # noqa: BLE001 - deterministic exit code with traceback for debugging unexpected failures.
        print("Attestation build failed with unexpected error:", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)
        return EXIT_BUILD_ERROR

    try:
        output_path = Path(args.out)
        _atomic_write_bytes(output_path, output_bytes)
    except OSError as exc:
        print(f"Unable to write output: {exc}", file=sys.stderr)
        return EXIT_OUTPUT_ERROR

    return EXIT_SUCCESS


if __name__ == "__main__":
    raise SystemExit(main())
