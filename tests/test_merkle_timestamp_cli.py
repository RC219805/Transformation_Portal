"""Tests for the Phase 3.2 detached RFC 3161 timestamp CLI."""

from __future__ import annotations

import contextlib
import errno
import hashlib
import os
import shutil
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


def _run_cli(command: list[str], *, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )


def _timestamp(
    *,
    target_flag: str,
    target_path: Path,
    tsa_url: str,
    out_path: Path,
    tsa_ca_file: Path | None = None,
    tsa_ca_path: Path | None = None,
    allow_insecure_http: bool = False,
    nonce: int | None = None,
    cert_req: bool = True,
    env: dict[str, str] | None = None,
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
    if tsa_ca_file is not None:
        command.extend(["--tsa-ca-file", str(tsa_ca_file)])
    if tsa_ca_path is not None:
        command.extend(["--tsa-ca-path", str(tsa_ca_path)])
    if allow_insecure_http:
        command.append("--allow-insecure-http")
    if nonce is not None:
        command.extend(["--nonce", str(nonce)])
    if cert_req:
        command.append("--cert-req")
    else:
        command.append("--no-cert-req")
    return _run_cli(command, env=env)


def _write_roots(path: Path) -> None:
    path.write_text(
        "{\n"
        + '  "hash_algorithm": "sha256",\n'
        + '  "tree_method_version": "v1",\n'
        + '  "global_root": "6f32b71a"\n'
        + "}\n",
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


def _decode_timestamp_query(query_bytes: bytes) -> tuple[str, bytes, int, bool]:
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

    nonce_tag, nonce_value, offset = _der_read_tlv(req_value, offset)
    assert nonce_tag == 0x02
    nonce = _der_decode_integer(nonce_value)

    cert_req = False
    if offset < len(req_value):
        cert_req_tag, cert_req_value, offset = _der_read_tlv(req_value, offset)
        assert cert_req_tag == 0x01
        assert cert_req_value in (b"\x00", b"\xff")
        cert_req = cert_req_value == b"\xff"
    assert offset == len(req_value)

    return SHA256_OID, digest_value, nonce, cert_req


class _LocalTsaSigner:
    """Minimal local RFC3161 TSA signer powered by openssl."""

    def __init__(self, workspace: Path):
        self.workspace = workspace
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.ca_cert = self.workspace / "ca.crt"
        self.openssl_config = self.workspace / "openssl_tsa.cnf"
        self._counter = 0
        self._init_signing_material()

    @staticmethod
    def _run(cmd: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
        return subprocess.run(cmd, cwd=str(cwd), check=True, capture_output=True, text=True)

    def _init_signing_material(self) -> None:
        self._run(["openssl", "genrsa", "-out", "ca.key", "2048"], cwd=self.workspace)
        self._run(
            [
                "openssl",
                "req",
                "-x509",
                "-new",
                "-key",
                "ca.key",
                "-sha256",
                "-days",
                "3650",
                "-subj",
                "/CN=TP Test CA",
                "-out",
                "ca.crt",
            ],
            cwd=self.workspace,
        )
        self._run(["openssl", "genrsa", "-out", "tsa.key", "2048"], cwd=self.workspace)
        (self.workspace / "tsa_ext.cnf").write_text(
            "\n".join(
                [
                    "basicConstraints=CA:FALSE",
                    "keyUsage = digitalSignature, nonRepudiation",
                    "extendedKeyUsage = critical,timeStamping",
                    "subjectKeyIdentifier=hash",
                    "authorityKeyIdentifier=keyid,issuer",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        self._run(
            ["openssl", "req", "-new", "-key", "tsa.key", "-subj", "/CN=TP Test TSA", "-out", "tsa.csr"],
            cwd=self.workspace,
        )
        self._run(
            [
                "openssl",
                "x509",
                "-req",
                "-in",
                "tsa.csr",
                "-CA",
                "ca.crt",
                "-CAkey",
                "ca.key",
                "-CAcreateserial",
                "-out",
                "tsa.crt",
                "-days",
                "3650",
                "-sha256",
                "-extfile",
                "tsa_ext.cnf",
            ],
            cwd=self.workspace,
        )

        tsa_dir = self.workspace / "tsa"
        tsa_dir.mkdir(parents=True, exist_ok=True)
        (tsa_dir / "tsaserial").write_text("01\n", encoding="utf-8")
        for filename in ("tsa.crt", "ca.crt", "tsa.key"):
            (tsa_dir / filename).write_bytes((self.workspace / filename).read_bytes())

        self.openssl_config.write_text(
            "\n".join(
                [
                    "[ tsa ]",
                    "default_tsa = tsa_config1",
                    "",
                    "[ tsa_config1 ]",
                    "dir = ./tsa",
                    "serial = $dir/tsaserial",
                    "crypto_device = builtin",
                    "signer_cert = $dir/tsa.crt",
                    "certs = $dir/ca.crt",
                    "signer_key = $dir/tsa.key",
                    "signer_digest = sha256",
                    "default_policy = 1.2.3.4.1",
                    "other_policies = 1.2.3.4.5.6",
                    "digests = sha256",
                    "accuracy = secs:1, millisecs:500, microsecs:100",
                    "clock_precision_digits = 0",
                    "ordering = yes",
                    "tsa_name = yes",
                    "ess_cert_id_chain = no",
                    "",
                ]
            ),
            encoding="utf-8",
        )

    def sign_query(self, query_bytes: bytes) -> bytes:
        self._counter += 1
        query_path = self.workspace / f"query_{self._counter}.tsq"
        response_path = self.workspace / f"response_{self._counter}.tsr"
        query_path.write_bytes(query_bytes)
        self._run(
            [
                "openssl",
                "ts",
                "-reply",
                "-queryfile",
                str(query_path),
                "-config",
                str(self.openssl_config),
                "-out",
                str(response_path),
            ],
            cwd=self.workspace,
        )
        return response_path.read_bytes()


@pytest.fixture(scope="module", name="local_tsa_signer")
def fixture_local_tsa_signer(tmp_path_factory: pytest.TempPathFactory) -> _LocalTsaSigner:
    if shutil.which("openssl") is None:
        pytest.skip("OpenSSL not available in test environment")
    return _LocalTsaSigner(tmp_path_factory.mktemp("local_tsa"))


@contextlib.contextmanager
def _tsa_server(
    *,
    response_body: bytes,
    response_status: int = 200,
    response_content_type: str = "application/timestamp-reply",
    response_factory=None,
):
    state: dict[str, str | bytes] = {
        "request_body": b"",
        "request_content_type": "",
        "request_accept": "",
        "response_body": b"",
    }

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802
            content_length = int(self.headers.get("Content-Length", "0"))
            state["request_body"] = self.rfile.read(content_length)
            state["request_content_type"] = self.headers.get("Content-Type", "")
            state["request_accept"] = self.headers.get("Accept", "")

            status = response_status
            content_type = response_content_type
            body = response_body
            if response_factory is not None:
                status, content_type, body = response_factory(state["request_body"])  # type: ignore[arg-type]
            state["response_body"] = body

            self.send_response(status)
            if content_type:
                self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            if body:
                self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:  # pylint: disable=redefined-builtin
            return

    try:
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    except PermissionError as exc:
        pytest.skip(f"Local HTTP server bind not permitted in this environment: {exc}")
    except OSError as exc:
        if exc.errno in {errno.EPERM, errno.EACCES}:
            pytest.skip(f"Local HTTP server bind not permitted in this environment: {exc}")
        raise
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}/tsa"
    try:
        yield url, state
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


@contextlib.contextmanager
def _tsa_redirect_server(*, final_response_factory):
    state: dict[str, str | bytes] = {
        "initial_request_body": b"",
        "followup_request_body": b"",
    }

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802
            content_length = int(self.headers.get("Content-Length", "0"))
            request_body = self.rfile.read(content_length)

            if self.path == "/tsa":
                state["initial_request_body"] = request_body
                self.send_response(302)
                self.send_header("Location", "/final")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return

            if self.path == "/final":
                state["followup_request_body"] = request_body
                body = final_response_factory(request_body)
                self.send_response(200)
                self.send_header("Content-Type", "application/timestamp-reply")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def log_message(self, format: str, *args: object) -> None:  # pylint: disable=redefined-builtin
            return

    try:
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    except PermissionError as exc:
        pytest.skip(f"Local HTTP server bind not permitted in this environment: {exc}")
    except OSError as exc:
        if exc.errno in {errno.EPERM, errno.EACCES}:
            pytest.skip(f"Local HTTP server bind not permitted in this environment: {exc}")
        raise
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}/tsa"
    try:
        yield url, state
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


def test_timestamp_roots_success_writes_detached_tsr(tmp_path: Path, local_tsa_signer: _LocalTsaSigner) -> None:
    roots_path = tmp_path / "merkle_roots.json"
    tsr_path = tmp_path / "merkle_roots.tsr"
    _write_roots(roots_path)

    with _tsa_server(
        response_body=b"",
        response_factory=lambda query: (200, "application/timestamp-reply", local_tsa_signer.sign_query(query)),
    ) as (tsa_url, state):
        result = _timestamp(
            target_flag="--roots",
            target_path=roots_path,
            tsa_url=tsa_url,
            out_path=tsr_path,
            tsa_ca_file=local_tsa_signer.ca_cert,
            allow_insecure_http=True,
            nonce=42,
        )

    assert result.returncode == 0, result.stderr
    assert tsr_path.exists()
    assert tsr_path.read_bytes() == state["response_body"]
    assert "Timestamp response written" in result.stdout
    assert state["request_content_type"] == "application/timestamp-query"
    assert state["request_accept"] == "application/timestamp-reply"

    oid, digest, nonce, cert_req = _decode_timestamp_query(state["request_body"])  # type: ignore[arg-type]
    assert oid == SHA256_OID
    assert digest == hashlib.sha256(roots_path.read_bytes()).digest()
    assert nonce == 42
    assert cert_req is True


def test_timestamp_signature_success_hashes_signature_bytes(tmp_path: Path, local_tsa_signer: _LocalTsaSigner) -> None:
    signature_path = tmp_path / "merkle_roots.sig.json"
    tsr_path = tmp_path / "nested" / "timestamps" / "merkle_roots.sig.tsr"
    _write_signature(signature_path)

    with _tsa_server(
        response_body=b"",
        response_factory=lambda query: (200, "application/timestamp-reply", local_tsa_signer.sign_query(query)),
    ) as (tsa_url, state):
        result = _timestamp(
            target_flag="--signature",
            target_path=signature_path,
            tsa_url=tsa_url,
            out_path=tsr_path,
            tsa_ca_file=local_tsa_signer.ca_cert,
            allow_insecure_http=True,
            nonce=7,
        )

    assert result.returncode == 0, result.stderr
    assert tsr_path.exists()
    assert tsr_path.read_bytes() == state["response_body"]

    _, digest, nonce, cert_req = _decode_timestamp_query(state["request_body"])  # type: ignore[arg-type]
    assert digest == hashlib.sha256(signature_path.read_bytes()).digest()
    assert nonce == 7
    assert cert_req is True


def test_timestamp_includes_cert_req_when_flag_set(tmp_path: Path, local_tsa_signer: _LocalTsaSigner) -> None:
    roots_path = tmp_path / "merkle_roots.json"
    tsr_path = tmp_path / "merkle_roots.tsr"
    _write_roots(roots_path)

    with _tsa_server(
        response_body=b"",
        response_factory=lambda query: (200, "application/timestamp-reply", local_tsa_signer.sign_query(query)),
    ) as (tsa_url, state):
        result = _timestamp(
            target_flag="--roots",
            target_path=roots_path,
            tsa_url=tsa_url,
            out_path=tsr_path,
            tsa_ca_file=local_tsa_signer.ca_cert,
            allow_insecure_http=True,
            nonce=314159,
            cert_req=True,
        )

    assert result.returncode == 0, result.stderr
    _, digest, nonce, cert_req = _decode_timestamp_query(state["request_body"])  # type: ignore[arg-type]
    assert digest == hashlib.sha256(roots_path.read_bytes()).digest()
    assert nonce == 314159
    assert cert_req is True


def test_timestamp_omits_cert_req_with_no_cert_req_flag(tmp_path: Path, local_tsa_signer: _LocalTsaSigner) -> None:
    roots_path = tmp_path / "merkle_roots.json"
    tsr_path = tmp_path / "merkle_roots.tsr"
    _write_roots(roots_path)

    with _tsa_server(
        response_body=b"",
        response_factory=lambda query: (200, "application/timestamp-reply", local_tsa_signer.sign_query(query)),
    ) as (tsa_url, state):
        _timestamp(
            target_flag="--roots",
            target_path=roots_path,
            tsa_url=tsa_url,
            out_path=tsr_path,
            tsa_ca_file=local_tsa_signer.ca_cert,
            allow_insecure_http=True,
            nonce=2718,
            cert_req=False,
        )

    _, digest, nonce, cert_req = _decode_timestamp_query(state["request_body"])  # type: ignore[arg-type]
    assert digest == hashlib.sha256(roots_path.read_bytes()).digest()
    assert nonce == 2718
    assert cert_req is False


def test_timestamp_no_cert_req_fails_strict_verification_without_embedded_signer_cert(
    tmp_path: Path, local_tsa_signer: _LocalTsaSigner
) -> None:
    roots_path = tmp_path / "merkle_roots.json"
    tsr_path = tmp_path / "merkle_roots.tsr"
    _write_roots(roots_path)

    with _tsa_server(
        response_body=b"",
        response_factory=lambda query: (200, "application/timestamp-reply", local_tsa_signer.sign_query(query)),
    ) as (tsa_url, _):
        result = _timestamp(
            target_flag="--roots",
            target_path=roots_path,
            tsa_url=tsa_url,
            out_path=tsr_path,
            tsa_ca_file=local_tsa_signer.ca_cert,
            allow_insecure_http=True,
            nonce=2718,
            cert_req=False,
        )

    assert result.returncode == 9
    assert "signer certificate not found" in result.stdout


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


def test_timestamp_fails_fast_when_openssl_unavailable(tmp_path: Path) -> None:
    roots_path = tmp_path / "merkle_roots.json"
    tsr_path = tmp_path / "out.tsr"
    _write_roots(roots_path)

    env = os.environ.copy()
    env["PATH"] = ""

    result = _timestamp(
        target_flag="--roots",
        target_path=roots_path,
        tsa_url="https://tsa.example.invalid/ts",
        out_path=tsr_path,
        nonce=1,
        env=env,
    )

    assert result.returncode == 8
    assert "OpenSSL executable is required" in result.stdout


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
            allow_insecure_http=True,
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
            allow_insecure_http=True,
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
            allow_insecure_http=True,
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
            allow_insecure_http=True,
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
            allow_insecure_http=True,
            nonce=1,
        )

    assert result.returncode == 9
    assert "unexpected Content-Type" in result.stdout


def test_timestamp_rejects_http_tsa_url_without_insecure_flag(tmp_path: Path) -> None:
    roots_path = tmp_path / "merkle_roots.json"
    tsr_path = tmp_path / "out.tsr"
    _write_roots(roots_path)

    with _tsa_server(response_body=b"unused") as (tsa_url, _):
        result = _timestamp(
            target_flag="--roots",
            target_path=roots_path,
            tsa_url=tsa_url,
            out_path=tsr_path,
            nonce=1,
        )

    assert result.returncode == 8
    assert "must use https unless --allow-insecure-http is set" in result.stdout


def test_timestamp_rejects_non_absolute_tsa_url(tmp_path: Path) -> None:
    roots_path = tmp_path / "merkle_roots.json"
    tsr_path = tmp_path / "out.tsr"
    _write_roots(roots_path)

    result = _timestamp(
        target_flag="--roots",
        target_path=roots_path,
        tsa_url="/tsa",
        out_path=tsr_path,
        nonce=1,
    )

    assert result.returncode == 8
    assert "--tsa-url must be an absolute http(s) URL" in result.stdout


def test_timestamp_rejects_redirect_response(tmp_path: Path, local_tsa_signer: _LocalTsaSigner) -> None:
    roots_path = tmp_path / "merkle_roots.json"
    tsr_path = tmp_path / "out.tsr"
    _write_roots(roots_path)

    with _tsa_redirect_server(final_response_factory=local_tsa_signer.sign_query) as (tsa_url, state):
        result = _timestamp(
            target_flag="--roots",
            target_path=roots_path,
            tsa_url=tsa_url,
            out_path=tsr_path,
            tsa_ca_file=local_tsa_signer.ca_cert,
            allow_insecure_http=True,
            nonce=1,
        )

    assert result.returncode == 8
    assert "HTTP 302" in result.stdout
    assert state["initial_request_body"]
    assert state["followup_request_body"] == b""
    assert not tsr_path.exists()


def test_timestamp_fails_when_cryptographic_verification_fails(tmp_path: Path, local_tsa_signer: _LocalTsaSigner) -> None:
    roots_path = tmp_path / "merkle_roots.json"
    tsr_path = tmp_path / "out.tsr"
    _write_roots(roots_path)

    # Structurally valid enough for parser, but not a valid signed RFC3161 token.
    response = _build_timestamp_response(0, include_token=True)
    with _tsa_server(response_body=response) as (tsa_url, _):
        result = _timestamp(
            target_flag="--roots",
            target_path=roots_path,
            tsa_url=tsa_url,
            out_path=tsr_path,
            tsa_ca_file=local_tsa_signer.ca_cert,
            allow_insecure_http=True,
            nonce=1,
        )

    assert result.returncode == 9
    assert "cryptographic timestamp verification failed" in result.stdout
