#!/usr/bin/env python3
"""
Phase 3.2 Detached RFC 3161 timestamp anchoring for Merkle artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import secrets
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlparse
from uuid import uuid4

EXIT_TIMESTAMP_FAILURE = 8
EXIT_INVALID_TIMESTAMP_RESPONSE = 9
EXPECTED_ROOTS_FILENAME = "merkle_roots.json"
EXPECTED_SIGNATURE_FILENAME = "merkle_roots.sig.json"
SHA256_OID = "2.16.840.1.101.3.4.2.1"
TIMESTAMP_QUERY_CONTENT_TYPE = "application/timestamp-query"
TIMESTAMP_REPLY_CONTENT_TYPE = "application/timestamp-reply"


class DerParseError(ValueError):
    """Raised when DER parsing fails for TSA responses."""


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        tmp.write_bytes(data)
        tmp.replace(path)
    finally:
        if tmp.exists():
            tmp.unlink()


def _encode_length(length: int) -> bytes:
    if length < 0:
        raise ValueError("length must be non-negative")
    if length < 0x80:
        return bytes([length])
    octets = length.to_bytes((length.bit_length() + 7) // 8, "big")
    return bytes([0x80 | len(octets)]) + octets


def _encode_tlv(tag: int, value: bytes) -> bytes:
    return bytes([tag]) + _encode_length(len(value)) + value


def _encode_positive_integer(value: int) -> bytes:
    if value < 0:
        raise ValueError("INTEGER value must be non-negative")
    raw = value.to_bytes(max(1, (value.bit_length() + 7) // 8), "big")
    if raw[0] & 0x80:
        raw = b"\x00" + raw
    return _encode_tlv(0x02, raw)


def _encode_base128(value: int) -> bytes:
    if value < 0:
        raise ValueError("OID component must be non-negative")
    parts: list[int] = [value & 0x7F]
    value >>= 7
    while value:
        parts.append(0x80 | (value & 0x7F))
        value >>= 7
    return bytes(reversed(parts))


def _encode_oid(oid: str) -> bytes:
    components = [int(part) for part in oid.split(".")]
    if len(components) < 2:
        raise ValueError("OID must have at least 2 components")
    if components[0] not in (0, 1, 2):
        raise ValueError("OID first component must be 0, 1, or 2")
    if components[0] < 2 and components[1] > 39:
        raise ValueError("OID second component must be <= 39 when first component is 0 or 1")

    first_subidentifier = (40 * components[0]) + components[1]
    encoded = _encode_base128(first_subidentifier)
    for component in components[2:]:
        encoded += _encode_base128(component)
    return _encode_tlv(0x06, encoded)


def _encode_sequence(*elements: bytes) -> bytes:
    return _encode_tlv(0x30, b"".join(elements))


def _encode_octet_string(data: bytes) -> bytes:
    return _encode_tlv(0x04, data)


def _encode_boolean(value: bool) -> bytes:
    return _encode_tlv(0x01, b"\xFF" if value else b"\x00")


def _read_length(data: bytes, offset: int) -> tuple[int, int]:
    if offset >= len(data):
        raise DerParseError("truncated DER length")
    first = data[offset]
    offset += 1
    if first < 0x80:
        return first, offset
    count = first & 0x7F
    if count == 0:
        raise DerParseError("indefinite DER lengths are not supported")
    if offset + count > len(data):
        raise DerParseError("truncated DER long-form length")
    length = int.from_bytes(data[offset : offset + count], "big")
    return length, offset + count


def _read_tlv(data: bytes, offset: int) -> tuple[int, bytes, int]:
    if offset >= len(data):
        raise DerParseError("unexpected end of DER input")
    tag = data[offset]
    offset += 1
    length, offset = _read_length(data, offset)
    end = offset + length
    if end > len(data):
        raise DerParseError("truncated DER value")
    return tag, data[offset:end], end


def _decode_positive_integer(value: bytes) -> int:
    if not value:
        raise DerParseError("INTEGER must not be empty")
    if value[0] & 0x80:
        raise DerParseError("negative INTEGER is not supported")
    return int.from_bytes(value, "big")


def _build_timestamp_request(message_digest: bytes, nonce: int, cert_req: bool) -> bytes:
    if len(message_digest) != 32:
        raise ValueError("sha256 message digest must be 32 bytes")

    algorithm_identifier = _encode_sequence(
        _encode_oid(SHA256_OID),
        _encode_tlv(0x05, b""),
    )
    message_imprint = _encode_sequence(
        algorithm_identifier,
        _encode_octet_string(message_digest),
    )
    elements = [
        _encode_positive_integer(1),  # TimeStampReq.version v1
        message_imprint,
        _encode_positive_integer(nonce),
    ]
    if cert_req:
        elements.append(_encode_boolean(True))
    return _encode_sequence(*elements)


def _parse_timestamp_status(response_bytes: bytes) -> tuple[int, bool]:
    top_tag, top_value, top_next = _read_tlv(response_bytes, 0)
    if top_tag != 0x30:
        raise DerParseError("response is not a DER SEQUENCE")
    if top_next != len(response_bytes):
        raise DerParseError("trailing bytes after DER SEQUENCE")

    status_info_tag, status_info_value, offset = _read_tlv(top_value, 0)
    if status_info_tag != 0x30:
        raise DerParseError("status info is not a DER SEQUENCE")

    status_tag, status_value, _ = _read_tlv(status_info_value, 0)
    if status_tag != 0x02:
        raise DerParseError("pkiStatus is not a DER INTEGER")
    status_code = _decode_positive_integer(status_value)

    has_token = offset < len(top_value)
    if has_token:
        token_tag, _, token_end = _read_tlv(top_value, offset)
        if token_tag != 0x30:
            raise DerParseError("timeStampToken is not a DER SEQUENCE")
        if token_end != len(top_value):
            raise DerParseError("unexpected trailing fields in TimeStampResp")

    return status_code, has_token


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    target_group = parser.add_mutually_exclusive_group(required=True)
    target_group.add_argument("--roots", help="Path to merkle_roots.json to timestamp")
    target_group.add_argument("--signature", help="Path to merkle_roots.sig.json to timestamp")
    parser.add_argument("--tsa-url", required=True, help="RFC 3161 TSA endpoint URL")
    parser.add_argument("--out", required=True, help="Output path for detached .tsr response")
    parser.add_argument(
        "--nonce",
        type=int,
        default=None,
        help="Optional positive nonce. Defaults to a random 128-bit nonce.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=30.0,
        help="HTTP timeout in seconds (default: 30)",
    )
    parser.add_argument(
        "--cert-req",
        action="store_true",
        help="Set RFC 3161 certReq=true in the request",
    )
    args = parser.parse_args()

    target_path: Path
    target_kind: str
    expected_name: str
    if args.roots:
        target_path = Path(args.roots)
        target_kind = "roots"
        expected_name = EXPECTED_ROOTS_FILENAME
    else:
        target_path = Path(args.signature)
        target_kind = "signature"
        expected_name = EXPECTED_SIGNATURE_FILENAME

    parsed_tsa_url = urlparse(args.tsa_url)
    if parsed_tsa_url.scheme not in {"http", "https"} or not parsed_tsa_url.netloc:
        print("Timestamp request failed: --tsa-url must be an absolute http(s) URL")
        return EXIT_TIMESTAMP_FAILURE

    if args.timeout_seconds <= 0:
        print("Timestamp request failed: --timeout-seconds must be > 0")
        return EXIT_TIMESTAMP_FAILURE

    nonce = args.nonce if args.nonce is not None else secrets.randbits(128)
    if nonce <= 0:
        print("Timestamp request failed: --nonce must be a positive integer")
        return EXIT_TIMESTAMP_FAILURE

    try:
        if target_path.name != expected_name:
            print(f"Timestamp request failed: --{target_kind} must reference {expected_name}")
            return EXIT_TIMESTAMP_FAILURE

        target_bytes = target_path.read_bytes()
        digest_bytes = hashlib.sha256(target_bytes).digest()
        digest_hex = digest_bytes.hex()
        timestamp_request = _build_timestamp_request(digest_bytes, nonce, cert_req=args.cert_req)

        http_request = urllib.request.Request(
            args.tsa_url,
            data=timestamp_request,
            method="POST",
            headers={
                "Content-Type": TIMESTAMP_QUERY_CONTENT_TYPE,
                "Accept": TIMESTAMP_REPLY_CONTENT_TYPE,
                "User-Agent": "transformation-portal-phase3.2-timestamp-cli/1",
            },
        )
        with urllib.request.urlopen(http_request, timeout=args.timeout_seconds) as response:
            response_bytes = response.read()
            content_type = response.headers.get("Content-Type", "")
            media_type = content_type.split(";", maxsplit=1)[0].strip().lower()

        if media_type and media_type != TIMESTAMP_REPLY_CONTENT_TYPE:
            print(f"Invalid timestamp response: unexpected Content-Type {content_type!r}")
            return EXIT_INVALID_TIMESTAMP_RESPONSE

        status_code, has_token = _parse_timestamp_status(response_bytes)
        if status_code not in (0, 1):
            print(f"Timestamp rejected by TSA: pkiStatus={status_code}")
            return EXIT_INVALID_TIMESTAMP_RESPONSE
        if not has_token:
            print("Invalid timestamp response: granted status without timeStampToken")
            return EXIT_INVALID_TIMESTAMP_RESPONSE

        atomic_write(Path(args.out), response_bytes)
        print(f"Timestamp response written to {args.out} (target={target_kind}, sha256={digest_hex}, nonce={nonce})")
        return 0
    except urllib.error.HTTPError as exc:
        print(f"Timestamp request failed: HTTP {exc.code}")
        return EXIT_TIMESTAMP_FAILURE
    except urllib.error.URLError as exc:
        print(f"Timestamp request failed: {exc.reason}")
        return EXIT_TIMESTAMP_FAILURE
    except OSError as exc:
        print(f"Timestamp request failed: {exc}")
        return EXIT_TIMESTAMP_FAILURE
    except DerParseError as exc:
        print(f"Invalid timestamp response: {exc}")
        return EXIT_INVALID_TIMESTAMP_RESPONSE
    except ValueError as exc:
        print(f"Timestamp request failed: {exc}")
        return EXIT_TIMESTAMP_FAILURE


if __name__ == "__main__":
    raise SystemExit(main())
