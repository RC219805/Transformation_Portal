"""Tests for the Phase 3.2 detached RFC 3161 timestamp CLI."""

from __future__ import annotations

import contextlib
import hashlib
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TIMESTAMP_TOOL = PROJECT_ROOT / "tools" / "timestamp_merkle_signature.py"
SHA256_OID = "2.16.840.1.101.3.4.2.1"

pytestmark = [pytest.mark.regression]


def _run_cli(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )


def _timestamp(
    *,
    target_flag: str,
    target_path: Path,
    tsa_url: str,
    out_path: Path,
    nonce: int | None = None,
) -> subprocess.CompletedProcess[str]:
    command = [
        sys.executable,
        str(TIMESTAMP_TOOL),
        target_flag,
        str(target_path),
        "--tsa-url",
        tsa_url,
        "--out",
        str(out_path),
    ]
    if nonce is not None:
        command.extend(["--nonce", str(nonce)])
    return _run_cli(command)


def _write_roots(path: Path) -> None:
    path.write_text(
        "{\n" '  "hash_algorithm": "sha256",\n' '  "tree_method_version": "v1",\n' '  "global_root": "6f32b71a"\n' "}\n",
        encoding="utf-8",
    )


def _write_signature(path: Path) -> None:
    path.write_text(
        "{\n"
        '  "envelope_version": "1",\n'
        '  "signature_algorithm": "ed25519",\n'
        '  "signed_artifact": "merkle_roots.json",\n'
        '  "signature_base64": "c2ln"\n'
        "}\n",
        encoding="utf-8",
    )


def _der_encode_length(length: int) -> bytes:
    if length < 0x80:
        return bytes([length])
    as_bytes = length.to_bytes((length.bit_length() + 7) // 8, "big")
    return bytes([0x80 | len(as_bytes)]) + as_bytes


def _der_tlv(tag: int, value: bytes) -> bytes:
    return bytes([tag]) + _der_encode_length(len(value)) + value


def _der_integer(value: int) -> bytes:
    raw = value.to_bytes(max(1, (value.bit_length() + 7) // 8), "big")
    if raw[0] & 0x80:
        raw = b"\x00" + raw
    return _der_tlv(0x02, raw)


def _der_base128(value: int) -> bytes:
    parts = [value & 0x7F]
    value >>= 7
    while value:
        parts.append(0x80 | (value & 0x7F))
        value >>= 7
    return bytes(reversed(parts))


def _der_oid(oid: str) -> bytes:
    components = [int(part) for part in oid.split(".")]
    first = (40 * components[0]) + components[1]
    value = _der_base128(first)
    for component in components[2:]:
        value += _der_base128(component)
    return _der_tlv(0x06, value)


def _der_sequence(*elements: bytes) -> bytes:
    return _der_tlv(0x30, b"".join(elements))


def _der_octet_string(value: bytes) -> bytes:
    return _der_tlv(0x04, value)


def _der_context0(value: bytes) -> bytes:
    return _der_tlv(0xA0, value)


def _build_timestamp_response(status: int, *, include_token: bool) -> bytes:
    status_info = _der_sequence(_der_integer(status))
    elements: list[bytes] = [status_info]
    if include_token:
        token = _der_sequence(
            _der_oid("1.2.840.113549.1.7.2"),
            _der_context0(_der_sequence(_der_integer(1))),
        )
        elements.append(token)
    return _der_sequence(*elements)


def _der_read_length(data: bytes, offset: int) -> tuple[int, int]:
    first = data[offset]
    offset += 1
    if first < 0x80:
        return first, offset
    count = first & 0x7F
    length = int.from_bytes(data[offset : offset + count], "big")
    return length, offset + count


def _der_read_tlv(data: bytes, offset: int) -> tuple[int, bytes, int]:
    tag = data[offset]
    offset += 1
    length, offset = _der_read_length(data, offset)
    end = offset + length
    return tag, data[offset:end], end


def _der_decode_integer(value: bytes) -> int:
    if value and value[0] == 0x00:
        value = value[1:]
    return int.from_bytes(value or b"\x00", "big")


def _der_decode_base128(data: bytes, offset: int) -> tuple[int, int]:
    value = 0
    while True:
        octet = data[offset]
        offset += 1
        value = (value << 7) | (octet & 0x7F)
        if (octet & 0x80) == 0:
            return value, offset


def _der_decode_oid(value: bytes) -> str:
    first_value, offset = _der_decode_base128(value, 0)
    if first_value < 40:
        components = [0, first_value]
    elif first_value < 80:
        components = [1, first_value - 40]
    else:
        components = [2, first_value - 80]
    while offset < len(value):
        component, offset = _der_decode_base128(value, offset)
        components.append(component)
    return ".".join(str(component) for component in components)


def _decode_timestamp_query(query_bytes: bytes) -> tuple[str, bytes, int]:
    req_tag, req_value, req_end = _der_read_tlv(query_bytes, 0)
    assert req_tag == 0x30
    assert req_end == len(query_bytes)

    version_tag, version_value, offset = _der_read_tlv(req_value, 0)
    assert version_tag == 0x02
    assert _der_decode_integer(version_value) == 1

    imprint_tag, imprint_value, offset = _der_read_tlv(req_value, offset)
    assert imprint_tag == 0x30

    algo_tag, algo_value, imprint_offset = _der_read_tlv(imprint_value, 0)
    assert algo_tag == 0x30
    algo_offset = 0
    oid_tag, oid_value, algo_offset = _der_read_tlv(algo_value, algo_offset)
    assert oid_tag == 0x06
    assert _der_decode_oid(oid_value) == SHA256_OID
    null_tag, null_value, algo_offset = _der_read_tlv(algo_value, algo_offset)
    assert null_tag == 0x05
    assert null_value == b""
    assert algo_offset == len(algo_value)

    digest_tag, digest_value, digest_end = _der_read_tlv(imprint_value, imprint_offset)
    assert digest_tag == 0x04
    assert digest_end == len(imprint_value)

    nonce_tag, nonce_value, _ = _der_read_tlv(req_value, offset)
    assert nonce_tag == 0x02
    nonce = _der_decode_integer(nonce_value)

    return SHA256_OID, digest_value, nonce


@contextlib.contextmanager
def _tsa_server(
    *,
    response_body: bytes,
    response_status: int = 200,
    response_content_type: str = "application/timestamp-reply",
):
    state: dict[str, str | bytes] = {
        "request_body": b"",
        "request_content_type": "",
        "request_accept": "",
    }

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802
            content_length = int(self.headers.get("Content-Length", "0"))
            state["request_body"] = self.rfile.read(content_length)
            state["request_content_type"] = self.headers.get("Content-Type", "")
            state["request_accept"] = self.headers.get("Accept", "")

            self.send_response(response_status)
            if response_content_type:
                self.send_header("Content-Type", response_content_type)
            self.send_header("Content-Length", str(len(response_body)))
            self.end_headers()
            if response_body:
                self.wfile.write(response_body)

        def log_message(self, format: str, *args: object) -> None:  # noqa: A003
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}/tsa"
    try:
        yield url, state
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


def test_timestamp_roots_success_writes_detached_tsr(tmp_path: Path) -> None:
    roots_path = tmp_path / "merkle_roots.json"
    tsr_path = tmp_path / "merkle_roots.tsr"
    _write_roots(roots_path)

    response = _build_timestamp_response(0, include_token=True)
    with _tsa_server(response_body=response) as (tsa_url, state):
        result = _timestamp(
            target_flag="--roots",
            target_path=roots_path,
            tsa_url=tsa_url,
            out_path=tsr_path,
            nonce=42,
        )

    assert result.returncode == 0, result.stderr
    assert tsr_path.read_bytes() == response
    assert "Timestamp response written" in result.stdout
    assert state["request_content_type"] == "application/timestamp-query"
    assert state["request_accept"] == "application/timestamp-reply"

    oid, digest, nonce = _decode_timestamp_query(state["request_body"])  # type: ignore[arg-type]
    assert oid == SHA256_OID
    assert digest == hashlib.sha256(roots_path.read_bytes()).digest()
    assert nonce == 42


def test_timestamp_signature_success_hashes_signature_bytes(tmp_path: Path) -> None:
    signature_path = tmp_path / "merkle_roots.sig.json"
    tsr_path = tmp_path / "nested" / "timestamps" / "merkle_roots.sig.tsr"
    _write_signature(signature_path)

    response = _build_timestamp_response(0, include_token=True)
    with _tsa_server(response_body=response) as (tsa_url, state):
        result = _timestamp(
            target_flag="--signature",
            target_path=signature_path,
            tsa_url=tsa_url,
            out_path=tsr_path,
            nonce=7,
        )

    assert result.returncode == 0, result.stderr
    assert tsr_path.exists()

    _, digest, nonce = _decode_timestamp_query(state["request_body"])  # type: ignore[arg-type]
    assert digest == hashlib.sha256(signature_path.read_bytes()).digest()
    assert nonce == 7


def test_timestamp_fails_when_target_filename_is_invalid(tmp_path: Path) -> None:
    invalid_roots_path = tmp_path / "other_roots.json"
    invalid_roots_path.write_text("{}\n", encoding="utf-8")
    tsr_path = tmp_path / "out.tsr"

    result = _timestamp(
        target_flag="--roots",
        target_path=invalid_roots_path,
        tsa_url="https://tsa.example.invalid/ts",
        out_path=tsr_path,
    )

    assert result.returncode == 8
    assert "--roots must reference merkle_roots.json" in result.stdout


def test_timestamp_fails_with_missing_target_file(tmp_path: Path) -> None:
    missing_path = tmp_path / "merkle_roots.json"
    tsr_path = tmp_path / "out.tsr"

    result = _timestamp(
        target_flag="--roots",
        target_path=missing_path,
        tsa_url="https://tsa.example.invalid/ts",
        out_path=tsr_path,
    )

    assert result.returncode == 8
    assert "Timestamp request failed" in result.stdout


def test_timestamp_fails_on_http_error(tmp_path: Path) -> None:
    roots_path = tmp_path / "merkle_roots.json"
    tsr_path = tmp_path / "out.tsr"
    _write_roots(roots_path)

    with _tsa_server(response_body=b"failure", response_status=500) as (tsa_url, _):
        result = _timestamp(
            target_flag="--roots",
            target_path=roots_path,
            tsa_url=tsa_url,
            out_path=tsr_path,
            nonce=1,
        )

    assert result.returncode == 8
    assert "HTTP 500" in result.stdout


def test_timestamp_fails_on_malformed_tsa_response(tmp_path: Path) -> None:
    roots_path = tmp_path / "merkle_roots.json"
    tsr_path = tmp_path / "out.tsr"
    _write_roots(roots_path)

    with _tsa_server(response_body=b"not-der") as (tsa_url, _):
        result = _timestamp(
            target_flag="--roots",
            target_path=roots_path,
            tsa_url=tsa_url,
            out_path=tsr_path,
            nonce=1,
        )

    assert result.returncode == 9
    assert "Invalid timestamp response" in result.stdout


def test_timestamp_fails_on_rejected_tsa_status(tmp_path: Path) -> None:
    roots_path = tmp_path / "merkle_roots.json"
    tsr_path = tmp_path / "out.tsr"
    _write_roots(roots_path)

    response = _build_timestamp_response(2, include_token=False)
    with _tsa_server(response_body=response) as (tsa_url, _):
        result = _timestamp(
            target_flag="--roots",
            target_path=roots_path,
            tsa_url=tsa_url,
            out_path=tsr_path,
            nonce=1,
        )

    assert result.returncode == 9
    assert "Timestamp rejected by TSA" in result.stdout


def test_timestamp_fails_when_granted_response_omits_token(tmp_path: Path) -> None:
    roots_path = tmp_path / "merkle_roots.json"
    tsr_path = tmp_path / "out.tsr"
    _write_roots(roots_path)

    response = _build_timestamp_response(0, include_token=False)
    with _tsa_server(response_body=response) as (tsa_url, _):
        result = _timestamp(
            target_flag="--roots",
            target_path=roots_path,
            tsa_url=tsa_url,
            out_path=tsr_path,
            nonce=1,
        )

    assert result.returncode == 9
    assert "granted status without timeStampToken" in result.stdout


def test_timestamp_fails_on_unexpected_content_type(tmp_path: Path) -> None:
    roots_path = tmp_path / "merkle_roots.json"
    tsr_path = tmp_path / "out.tsr"
    _write_roots(roots_path)

    response = _build_timestamp_response(0, include_token=True)
    with _tsa_server(response_body=response, response_content_type="application/json") as (tsa_url, _):
        result = _timestamp(
            target_flag="--roots",
            target_path=roots_path,
            tsa_url=tsa_url,
            out_path=tsr_path,
            nonce=1,
        )

    assert result.returncode == 9
    assert "unexpected Content-Type" in result.stdout
