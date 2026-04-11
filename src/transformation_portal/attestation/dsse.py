"""DSSE helpers for deterministic in-toto payload envelopes."""

from __future__ import annotations

import base64
from collections.abc import Mapping
from typing import Any

DSSE_IN_TOTO_JSON_PAYLOAD_TYPE = "application/vnd.in-toto+json"


def pre_auth_encode(payload_type: str, payload: bytes) -> bytes:
    """Produce the DSSE v1 pre-authentication encoding."""
    if not isinstance(payload_type, str) or not payload_type:
        raise ValueError("payload_type must be a non-empty string")
    return (
        b"DSSEv1 "
        + str(len(payload_type)).encode("ascii")
        + b" "
        + payload_type.encode("utf-8")
        + b" "
        + str(len(payload)).encode("ascii")
        + b" "
        + payload
    )


def build_dsse_envelope(
    *,
    payload: bytes,
    payload_type: str,
    key_id: str,
    signature_bytes: bytes,
) -> dict[str, Any]:
    """Build a DSSE envelope with a single signature entry."""
    if not isinstance(key_id, str) or not key_id:
        raise ValueError("key_id must be a non-empty string")
    return {
        "payload": base64.b64encode(payload).decode("ascii"),
        "payloadType": payload_type,
        "signatures": [
            {
                "keyid": key_id,
                "sig": base64.b64encode(signature_bytes).decode("ascii"),
            }
        ],
    }


def decode_dsse_payload(envelope: Mapping[str, Any]) -> bytes:
    payload = envelope.get("payload")
    if not isinstance(payload, str) or not payload:
        raise ValueError("DSSE envelope payload must be a non-empty base64 string")
    try:
        return base64.b64decode(payload.encode("ascii"), validate=True)
    except Exception as exc:  # noqa: BLE001 - normalized validation error surface
        raise ValueError("DSSE envelope payload is not valid base64") from exc


def decode_dsse_signature_bytes(envelope: Mapping[str, Any], *, signature_index: int = 0) -> bytes:
    signatures = envelope.get("signatures")
    if not isinstance(signatures, list) or not signatures:
        raise ValueError("DSSE envelope signatures must be a non-empty list")
    try:
        signature = signatures[signature_index]
    except IndexError as exc:
        raise ValueError("DSSE envelope signature_index out of range") from exc
    if not isinstance(signature, Mapping):
        raise ValueError("DSSE envelope signature entry must be an object")
    sig = signature.get("sig")
    if not isinstance(sig, str) or not sig:
        raise ValueError("DSSE envelope signature.sig must be a non-empty base64 string")
    try:
        return base64.b64decode(sig.encode("ascii"), validate=True)
    except Exception as exc:  # noqa: BLE001 - normalized validation error surface
        raise ValueError("DSSE envelope signature.sig is not valid base64") from exc


def validate_dsse_envelope_surface(envelope: Mapping[str, Any]) -> None:
    """Validate the structural surface of a DSSE envelope."""
    decode_dsse_payload(envelope)
    payload_type = envelope.get("payloadType")
    if not isinstance(payload_type, str) or not payload_type:
        raise ValueError("DSSE envelope payloadType must be a non-empty string")
    signatures = envelope.get("signatures")
    if not isinstance(signatures, list) or not signatures:
        raise ValueError("DSSE envelope signatures must be a non-empty list")
    for index, signature in enumerate(signatures):
        if not isinstance(signature, Mapping):
            raise ValueError(f"DSSE envelope signatures[{index}] must be an object")
        key_id = signature.get("keyid")
        if not isinstance(key_id, str) or not key_id:
            raise ValueError(f"DSSE envelope signatures[{index}].keyid must be a non-empty string")
        decode_dsse_signature_bytes(envelope, signature_index=index)
