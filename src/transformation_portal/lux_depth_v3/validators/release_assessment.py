"""Policy-driven release assessment for Lux run cards and attestations."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

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
from transformation_portal.lux_depth_v3.run_card_contract import infer_run_card_version
from transformation_portal.lux_depth_v3.validators.run_card_integrity import verify_run_card_integrity


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"unable to load {label} {path}: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"{label} must be a JSON object")
    return parsed


def _default_sidecar_path(run_card_path: Path, suffix: str) -> Path:
    return run_card_path.with_suffix(suffix)


def _bundle_has_rekor_inclusion(bundle_payload: Mapping[str, Any]) -> bool:
    verification_material = bundle_payload.get("verificationMaterial")
    if isinstance(verification_material, Mapping):
        tlog_entries = verification_material.get("tlogEntries")
        if isinstance(tlog_entries, list) and tlog_entries:
            return True
    return False


def assess_run_card_release(
    *,
    run_card_path: Path,
    require_v2: bool = True,
    require_native_attestation: bool = False,
    require_dsse_attestation: bool = False,
    require_sigstore_bundle: bool = False,
    require_rekor_inclusion: bool = False,
    allow_missing_attestation_sha: bool = False,
    verify_gpg: bool = False,
    cosign_key_path: Path | None = None,
) -> dict[str, Any]:
    """Assess a run card against release-policy requirements."""
    checks: list[dict[str, Any]] = []

    integrity_errors = verify_run_card_integrity(run_card_path, check_canonical_json=True)
    checks.append(
        {
            "name": "run_card_integrity",
            "status": "PASS" if not integrity_errors else "FAIL",
            "details": [] if not integrity_errors else integrity_errors,
        }
    )
    if integrity_errors:
        return {
            "status": "FAIL",
            "run_card_path": str(run_card_path),
            "checks": checks,
        }

    run_card_payload = _load_json_object(run_card_path, label="run card")
    run_card_bytes = run_card_path.read_bytes()
    run_card_version = infer_run_card_version(run_card_payload)
    checks.append(
        {
            "name": "run_card_version",
            "status": "PASS" if (run_card_version == "v2" or not require_v2) else "FAIL",
            "details": {"detected": run_card_version, "required": "v2" if require_v2 else "any"},
        }
    )

    native_path = _default_sidecar_path(run_card_path, ".attestation.native.json")
    dsse_path = _default_sidecar_path(run_card_path, ".attestation.dsse.json")
    bundle_path = _default_sidecar_path(run_card_path, ".attestation.dsse.sigstore.bundle.json")

    if require_native_attestation or native_path.exists():
        native_errors: list[str] = []
        if not native_path.exists():
            native_errors.append(f"missing native detached attestation: {native_path}")
        else:
            try:
                native_attestation = _load_json_object(native_path, label="native attestation")
                validate_run_card_detached_attestation_surface(native_attestation)
                bind_run_card_detached_attestation(
                    native_attestation,
                    run_card_payload,
                    run_card_bytes=run_card_bytes,
                )
                verify_run_card_attestation_self_hash(
                    native_attestation,
                    require_digest=(not allow_missing_attestation_sha),
                )
                if verify_gpg:
                    signature_algorithm = native_attestation["signature"]["algorithm"]
                    if signature_algorithm != "openpgp-clearsign":
                        raise ValueError(
                            "native signature.algorithm must be 'openpgp-clearsign' for GPG verification, "
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
            except Exception as exc:  # noqa: BLE001 - normalized assessment surface
                native_errors.append(str(exc))
        checks.append(
            {
                "name": "native_attestation",
                "status": "PASS" if not native_errors else "FAIL",
                "details": {"path": str(native_path), "errors": native_errors},
            }
        )

    if require_dsse_attestation or dsse_path.exists():
        dsse_errors: list[str] = []
        if not dsse_path.exists():
            dsse_errors.append(f"missing DSSE attestation: {dsse_path}")
        else:
            try:
                dsse_envelope = _load_json_object(dsse_path, label="DSSE attestation")
                statement = decode_run_card_statement_from_envelope(dsse_envelope)
                validate_run_card_statement_binding(
                    statement,
                    run_card_path=run_card_path,
                    run_card_payload=run_card_payload,
                    run_card_bytes=run_card_bytes,
                )
                if verify_gpg:
                    gpg_verify_detached_signature_bytes(
                        decode_dsse_signature_bytes(dsse_envelope),
                        pre_auth_encode(DSSE_IN_TOTO_JSON_PAYLOAD_TYPE, decode_dsse_payload(dsse_envelope)),
                    )
            except Exception as exc:  # noqa: BLE001 - normalized assessment surface
                dsse_errors.append(str(exc))
        checks.append(
            {
                "name": "dsse_attestation",
                "status": "PASS" if not dsse_errors else "FAIL",
                "details": {"path": str(dsse_path), "errors": dsse_errors},
            }
        )

    if require_sigstore_bundle or require_rekor_inclusion:
        bundle_errors: list[str] = []
        rekor_inclusion = False
        if not dsse_path.exists():
            bundle_errors.append("cannot verify Sigstore bundle without DSSE attestation")
        elif not bundle_path.exists():
            bundle_errors.append(f"missing Sigstore bundle: {bundle_path}")
        else:
            try:
                bundle_payload = _load_json_object(bundle_path, label="Sigstore bundle")
                rekor_inclusion = _bundle_has_rekor_inclusion(bundle_payload)
                cosign_verify_blob(
                    blob_path=dsse_path,
                    bundle_path=bundle_path,
                    key_path=cosign_key_path,
                )
            except Exception as exc:  # noqa: BLE001 - normalized assessment surface
                bundle_errors.append(str(exc))
        if require_rekor_inclusion and not rekor_inclusion:
            bundle_errors.append("Sigstore bundle does not record Rekor inclusion evidence")
        checks.append(
            {
                "name": "sigstore_bundle",
                "status": "PASS" if not bundle_errors else "FAIL",
                "details": {
                    "path": str(bundle_path),
                    "rekor_inclusion": rekor_inclusion,
                    "errors": bundle_errors,
                },
            }
        )

    overall_status = "PASS" if all(check["status"] == "PASS" for check in checks) else "FAIL"
    return {
        "status": overall_status,
        "run_card_path": str(run_card_path),
        "run_card_version": run_card_version,
        "checks": checks,
    }
