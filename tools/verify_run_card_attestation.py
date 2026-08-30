#!/usr/bin/env python3
"""Verify detached run-card attestations and optional Sigstore bundles."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.is_dir():
    sys.path.insert(0, str(_SRC))

# pylint: disable=wrong-import-position
from transformation_portal.attestation.dsse import (
    DSSE_IN_TOTO_JSON_PAYLOAD_TYPE,
    decode_dsse_payload,
    decode_dsse_signature_bytes,
    pre_auth_encode,
)
from transformation_portal.attestation.gpg import gpg_verify_clearsign, gpg_verify_detached_signature_bytes
from transformation_portal.attestation.run_card_detached import (
    bind_run_card_detached_attestation,
    canonical_run_card_attestation_preimage_bytes,
    validate_run_card_detached_attestation_surface,
    verify_run_card_attestation_self_hash,
)
from transformation_portal.attestation.run_card_intoto import (
    decode_run_card_statement_from_envelope,
    validate_run_card_statement_binding,
)
from transformation_portal.attestation.sigstore import cosign_verify_blob
from transformation_portal.lux_depth_v3.validators import verify_run_card_integrity

EXIT_SUCCESS = 0
EXIT_INPUT_ERROR = 2
EXIT_VERIFY_FAILED = 5


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-card", required=True, help="Path to run_card_*.json.")
    parser.add_argument("--native-attestation", default=None, help="Optional native detached attestation path.")
    parser.add_argument("--dsse-attestation", default=None, help="Optional DSSE attestation path.")
    parser.add_argument("--sigstore-bundle", default=None, help="Optional Sigstore bundle path for the DSSE file.")
    parser.add_argument("--gpg", action="store_true", help="Verify GPG signatures when present.")
    parser.add_argument("--sigstore-key", default=None, help="Optional cosign key path for bundle verification.")
    parser.add_argument("--require-native", action="store_true", help="Fail if the native detached attestation is missing.")
    parser.add_argument("--require-dsse", action="store_true", help="Fail if the DSSE attestation is missing.")
    parser.add_argument(
        "--require-sigstore-bundle",
        action="store_true",
        help="Fail if the Sigstore bundle is missing. Requires a DSSE attestation.",
    )
    parser.add_argument(
        "--allow-missing-attestation-sha",
        action="store_true",
        help="Allow missing/null native attestation_sha256 for migration compatibility.",
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


def _read_bytes(path: Path, *, name: str) -> bytes:
    try:
        return path.read_bytes()
    except OSError as exc:
        raise ValueError(f"Unable to read {name} bytes: {exc}") from exc


def _default_sidecar_path(run_card_path: Path, suffix: str) -> Path:
    return run_card_path.with_suffix(suffix)


def main() -> int:
    args = _parse_args()
    run_card_path = Path(args.run_card)

    try:
        integrity_errors = verify_run_card_integrity(run_card_path, check_canonical_json=True)
        if integrity_errors:
            raise ValueError(
                "run card integrity verification failed before attestation verification: " + "; ".join(integrity_errors)
            )
        run_card_payload = _read_json_object(run_card_path, name="run card")
        run_card_bytes = _read_bytes(run_card_path, name="run card")
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_INPUT_ERROR

    native_path = (
        Path(args.native_attestation)
        if args.native_attestation
        else _default_sidecar_path(
            run_card_path,
            ".attestation.native.json",
        )
    )
    dsse_path = (
        Path(args.dsse_attestation)
        if args.dsse_attestation
        else _default_sidecar_path(
            run_card_path,
            ".attestation.dsse.json",
        )
    )
    bundle_path = (
        Path(args.sigstore_bundle)
        if args.sigstore_bundle
        else _default_sidecar_path(
            run_card_path,
            ".attestation.dsse.sigstore.bundle.json",
        )
    )
    native_requested = args.native_attestation is not None
    dsse_requested = args.dsse_attestation is not None
    sigstore_bundle_requested = args.sigstore_bundle is not None or args.require_sigstore_bundle

    try:
        if (args.require_native or native_requested) and not native_path.exists():
            raise ValueError(f"native detached attestation not found: {native_path}")
        if native_path.exists():
            native_attestation = _read_json_object(native_path, name="native attestation")
            validate_run_card_detached_attestation_surface(native_attestation)
            bind_run_card_detached_attestation(
                native_attestation,
                run_card_payload,
                run_card_bytes=run_card_bytes,
            )
            verify_run_card_attestation_self_hash(
                native_attestation,
                require_digest=(not args.allow_missing_attestation_sha),
            )
            if args.gpg:
                signature_algorithm = native_attestation["signature"]["algorithm"]
                if signature_algorithm != "openpgp-clearsign":
                    raise ValueError(
                        "native signature.algorithm must be 'openpgp-clearsign' when --gpg is enabled, "
                        f"got {signature_algorithm!r}"
                    )
                gpg_verify_clearsign(
                    str(native_attestation["signature"]["signature"]),
                    expected_payload=canonical_run_card_attestation_preimage_bytes(
                        run_card_payload,
                        run_card_bytes=run_card_bytes,
                    ),
                    key_id=str(native_attestation["signature"]["key_id"]),
                )

        if sigstore_bundle_requested and not dsse_path.exists():
            raise ValueError("cannot verify a Sigstore bundle when the DSSE attestation is missing")
        if (args.require_dsse or dsse_requested) and not dsse_path.exists():
            raise ValueError(f"DSSE attestation not found: {dsse_path}")
        if dsse_path.exists():
            dsse_envelope = _read_json_object(dsse_path, name="DSSE attestation")
            statement = decode_run_card_statement_from_envelope(dsse_envelope)
            validate_run_card_statement_binding(
                statement,
                run_card_path=run_card_path,
                run_card_payload=run_card_payload,
                run_card_bytes=run_card_bytes,
            )
            if args.gpg:
                gpg_verify_detached_signature_bytes(
                    decode_dsse_signature_bytes(dsse_envelope),
                    pre_auth_encode(DSSE_IN_TOTO_JSON_PAYLOAD_TYPE, decode_dsse_payload(dsse_envelope)),
                )
            if args.require_sigstore_bundle and not bundle_path.exists():
                raise ValueError(f"required Sigstore bundle not found: {bundle_path}")
            if sigstore_bundle_requested and not bundle_path.exists():
                raise ValueError(f"Sigstore bundle not found: {bundle_path}")
            if sigstore_bundle_requested and bundle_path.exists():
                cosign_verify_blob(
                    blob_path=dsse_path,
                    bundle_path=bundle_path,
                    key_path=(Path(args.sigstore_key) if args.sigstore_key else None),
                )
        elif args.require_sigstore_bundle:
            raise ValueError("cannot require a Sigstore bundle when the DSSE attestation is missing")
    except (TypeError, ValueError) as exc:
        print(f"Attestation verification failed: {exc}", file=sys.stderr)
        return EXIT_VERIFY_FAILED
    except Exception:  # noqa: BLE001 - deterministic exit code with traceback for debugging unexpected failures.
        print("Attestation verification failed with unexpected error:", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)
        return EXIT_VERIFY_FAILED

    return EXIT_SUCCESS


if __name__ == "__main__":
    raise SystemExit(main())
