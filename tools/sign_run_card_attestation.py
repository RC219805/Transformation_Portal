#!/usr/bin/env python3
"""Build detached attestations for a Lux run card."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from uuid import uuid4

_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.is_dir():
    sys.path.insert(0, str(_SRC))

# pylint: disable=wrong-import-position
from transformation_portal.attestation.dsse import DSSE_IN_TOTO_JSON_PAYLOAD_TYPE, pre_auth_encode
from transformation_portal.attestation.gpg import (
    gpg_clearsign_bytes,
    gpg_detached_sign_bytes,
    gpg_verify_clearsign,
)
from transformation_portal.attestation.run_card_detached import (
    build_run_card_detached_attestation_payload,
    canonical_run_card_attestation_bytes,
    canonical_run_card_attestation_preimage_bytes,
)
from transformation_portal.attestation.run_card_intoto import (
    build_run_card_dsse_envelope,
    canonical_run_card_statement_bytes,
)
from transformation_portal.attestation.sigstore import cosign_sign_blob
from transformation_portal.lux_depth_v3.validators import verify_run_card_integrity

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
    parser.add_argument("--run-card", required=True, help="Path to run_card_*.json.")
    parser.add_argument(
        "--format",
        choices=("native", "dsse", "both"),
        default="both",
        help="Which attestation format(s) to emit.",
    )
    parser.add_argument("--native-out", default=None, help="Optional path for native detached attestation JSON.")
    parser.add_argument("--dsse-out", default=None, help="Optional path for DSSE in-toto attestation JSON.")
    parser.add_argument(
        "--sigstore-bundle-out",
        default=None,
        help="Optional Sigstore bundle path for the DSSE envelope (signs the DSSE file bytes).",
    )
    parser.add_argument("--release-assessment", default=None, help="Optional JSON assessment payload to bind into predicate.")
    parser.add_argument("--gpg", action="store_true", help="Use GPG backends for native and DSSE signatures.")
    parser.add_argument("--gpg-key-id", default=None, help="Optional gpg key id for signing.")
    parser.add_argument(
        "--key-id",
        required=True,
        help="Key identifier to record; native signing requires it to resolve to the signing primary GPG key.",
    )
    parser.add_argument("--signed-at", default=None, help="RFC3339 UTC timestamp, e.g. 2026-04-10T20:15:00Z.")
    parser.add_argument("--sigstore-key", default=None, help="Optional cosign key path for bundle signing.")
    parser.add_argument(
        "--no-sigstore-tlog-upload",
        action="store_true",
        help="Disable transparency-log upload when creating the optional Sigstore bundle.",
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
    if args.sigstore_bundle_out and args.format == "native":
        print("--sigstore-bundle-out requires --format dsse or --format both", file=sys.stderr)
        return EXIT_INPUT_ERROR

    try:
        integrity_errors = verify_run_card_integrity(run_card_path, check_canonical_json=True)
        if integrity_errors:
            raise ValueError("run card integrity verification failed before signing: " + "; ".join(integrity_errors))
        run_card_payload = _read_json_object(run_card_path, name="run card")
        run_card_bytes = _read_bytes(run_card_path, name="run card")
        release_assessment = (
            _read_json_object(Path(args.release_assessment), name="release assessment")
            if args.release_assessment is not None
            else None
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_INPUT_ERROR

    if not args.gpg:
        print("Attestation build failed: No signing backend selected (use --gpg)", file=sys.stderr)
        return EXIT_BUILD_ERROR

    try:
        native_out = (
            Path(args.native_out)
            if args.native_out
            else _default_sidecar_path(
                run_card_path,
                ".attestation.native.json",
            )
        )
        dsse_out = (
            Path(args.dsse_out)
            if args.dsse_out
            else _default_sidecar_path(
                run_card_path,
                ".attestation.dsse.json",
            )
        )

        if args.format in {"native", "both"}:
            preimage_bytes = canonical_run_card_attestation_preimage_bytes(
                run_card_payload,
                run_card_bytes=run_card_bytes,
            )
            signature_text = gpg_clearsign_bytes(preimage_bytes, key_id=args.gpg_key_id)
            gpg_verify_clearsign(
                signature_text,
                expected_payload=preimage_bytes,
                key_id=args.key_id,
            )
            native_payload = build_run_card_detached_attestation_payload(
                run_card_payload,
                run_card_bytes=run_card_bytes,
                signature={
                    "algorithm": "openpgp-clearsign",
                    "key_id": args.key_id,
                    "signature": signature_text,
                },
                signed_at=args.signed_at,
                toolchain={"signer": "tools/sign_run_card_attestation.py"},
            )
            _atomic_write_bytes(native_out, canonical_run_card_attestation_bytes(native_payload))

        if args.format in {"dsse", "both"}:
            statement_bytes = canonical_run_card_statement_bytes(
                run_card_path=run_card_path,
                run_card_payload=run_card_payload,
                run_card_bytes=run_card_bytes,
                release_assessment=release_assessment,
            )
            dsse_signature = gpg_detached_sign_bytes(
                pre_auth_encode(DSSE_IN_TOTO_JSON_PAYLOAD_TYPE, statement_bytes),
                key_id=args.gpg_key_id,
            )
            dsse_payload = build_run_card_dsse_envelope(
                run_card_path=run_card_path,
                run_card_payload=run_card_payload,
                run_card_bytes=run_card_bytes,
                key_id=args.key_id,
                signature_bytes=dsse_signature,
                release_assessment=release_assessment,
            )
            _atomic_write_bytes(dsse_out, json.dumps(dsse_payload, indent=2, sort_keys=True).encode("utf-8"))

            if args.sigstore_bundle_out:
                bundle_path = Path(args.sigstore_bundle_out)
                cosign_sign_blob(
                    blob_path=dsse_out,
                    bundle_path=bundle_path,
                    key_path=(Path(args.sigstore_key) if args.sigstore_key else None),
                    tlog_upload=(not args.no_sigstore_tlog_upload),
                )
    except (TypeError, ValueError) as exc:
        print(f"Attestation build failed: {exc}", file=sys.stderr)
        return EXIT_BUILD_ERROR
    except OSError as exc:
        print(f"Unable to write output: {exc}", file=sys.stderr)
        return EXIT_OUTPUT_ERROR
    except Exception:  # noqa: BLE001 - deterministic exit code with traceback for debugging unexpected failures.
        print("Attestation build failed with unexpected error:", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)
        return EXIT_BUILD_ERROR

    return EXIT_SUCCESS


if __name__ == "__main__":
    raise SystemExit(main())
