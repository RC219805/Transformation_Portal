from __future__ import annotations

import builtins
import contextlib
import json
import os
import socket
import socketserver
import ssl
import stat
import threading
from pathlib import Path

import pytest

from transformation_portal.core.security import signing, tls
from transformation_portal.core.security.signing import (
    CertificateSigner,
    CertificateVerifier,
    SigningError,
    generate_ed25519_keypair,
    load_private_key,
    save_keypair,
)
from transformation_portal.core.security.tls import (
    ThreadedTLSServer,
    TLSError,
    TLSTCPServer,
    create_client_ssl_context,
    create_server_ssl_context,
    create_tls_connection,
    generate_self_signed_cert,
)

pytestmark = [pytest.mark.unit, pytest.mark.security]


def test_certificate_signer_round_trip() -> None:
    private_key, _ = generate_ed25519_keypair()
    manifest_json = json.dumps(
        {
            "root_hash": "sha256:1234",
            "artifacts": [],
        },
        sort_keys=True,
    )

    signer = CertificateSigner(private_key, issuer="test-suite")
    cert = signer.sign_manifest(manifest_json, metadata={"stage": "depth"})

    assert CertificateVerifier.verify(manifest_json, cert) is True
    assert CertificateVerifier.verify_with_public_key(manifest_json, cert, signer.public_key_bytes) is True


@pytest.mark.skipif(os.name != "posix", reason="POSIX file modes")
@pytest.mark.parametrize("existing_mode", [None, 0o644], ids=["new-file", "existing-world-readable-file"])
def test_save_keypair_private_key_is_readable_by_its_owner_only(tmp_path: Path, existing_mode: int | None) -> None:
    """The umask must not decide who can read the signing key."""
    private_key, public_key = generate_ed25519_keypair()
    private_path = tmp_path / "signing.key"
    public_path = tmp_path / "signing.pub"
    if existing_mode is not None:
        private_path.write_bytes(b"a previous, longer key file that has to be replaced completely")
        private_path.chmod(existing_mode)

    previous_umask = os.umask(0o022)
    try:
        save_keypair(private_key, public_key, private_path, public_path)
    finally:
        os.umask(previous_umask)

    assert stat.S_IMODE(private_path.stat().st_mode) == 0o600
    assert private_path.read_bytes() == private_key
    assert load_private_key(private_path) == private_key
    assert public_path.read_bytes() == public_key


def test_generate_ed25519_keypair_requires_crypto(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(signing, "_CRYPTO_AVAILABLE", False)

    with pytest.raises(SigningError, match="cryptography library required"):
        generate_ed25519_keypair()


def test_tls_context_helpers_support_generated_certificates(tmp_path) -> None:
    cert_path, key_path = generate_self_signed_cert("localhost", tmp_path)

    server_ctx = create_server_ssl_context(cert_path, key_path, cert_path, verify_client=False)
    client_ctx = create_client_ssl_context(cert_path, key_path, cert_path, verify_hostname=False)

    assert isinstance(server_ctx, ssl.SSLContext)
    assert isinstance(client_ctx, ssl.SSLContext)
    assert server_ctx.verify_mode == ssl.CERT_OPTIONAL
    assert client_ctx.verify_mode == ssl.CERT_REQUIRED


def test_generate_self_signed_cert_requires_cryptography(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    real_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):
        if name.startswith("cryptography"):
            raise ImportError("missing cryptography")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)

    with pytest.raises(TLSError, match="cryptography library required"):
        tls.generate_self_signed_cert("localhost", tmp_path)


class _GreetingHandler(socketserver.BaseRequestHandler):
    def handle(self) -> None:
        self.server.handled.append(self.client_address)  # type: ignore[attr-defined]
        self.request.sendall(b"ok")


@contextlib.contextmanager
def _serving(server_class: type[TLSTCPServer], tmp_path: Path, *, handshake_timeout: float | None = None):
    """Serve ``_GreetingHandler`` over mTLS; yield ``(server, port, client_context)``."""
    cert_path, key_path = generate_self_signed_cert("localhost", tmp_path)
    server = server_class(
        ("127.0.0.1", 0),
        _GreetingHandler,
        create_server_ssl_context(cert_path, key_path, cert_path),
    )
    server.handled = []  # type: ignore[attr-defined]
    if handshake_timeout is not None:
        server.handshake_timeout = handshake_timeout
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
    thread.start()
    try:
        yield server, server.server_address[1], create_client_ssl_context(cert_path, key_path, cert_path)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_threaded_tls_server_serves_clients_while_a_peer_stalls_the_handshake(tmp_path: Path) -> None:
    """One peer that connects and stays silent must not block the accept loop."""
    with _serving(ThreadedTLSServer, tmp_path) as (_, port, client_context):
        # Opened first, never sends a ClientHello; closed before the server shuts down.
        with socket.create_connection(("127.0.0.1", port)):
            with create_tls_connection("localhost", port, client_context, timeout=2.0) as connection:
                assert connection.recv(2) == b"ok"


@pytest.mark.parametrize("server_class", [TLSTCPServer, ThreadedTLSServer])
def test_tls_server_drops_a_peer_that_never_completes_the_handshake(tmp_path: Path, server_class: type[TLSTCPServer]) -> None:
    with _serving(server_class, tmp_path, handshake_timeout=0.2) as (server, port, client_context):
        with socket.create_connection(("127.0.0.1", port), timeout=2.0) as stalled:
            assert stalled.recv(1) == b""  # closed by the server once the handshake timed out

            with create_tls_connection("localhost", port, client_context, timeout=2.0) as connection:
                assert connection.recv(2) == b"ok"

        assert len(server.handled) == 1  # the stalled peer never reached the handler


def test_tls_server_does_not_run_the_handler_after_a_failed_handshake(tmp_path: Path) -> None:
    with _serving(ThreadedTLSServer, tmp_path) as (server, port, client_context):
        with socket.create_connection(("127.0.0.1", port), timeout=2.0) as plaintext:
            plaintext.sendall(b"GET / HTTP/1.1\r\n\r\n")
            # The server answers with a TLS alert at most, then closes or resets.
            with contextlib.suppress(ConnectionResetError):
                while plaintext.recv(4096):
                    pass

        with create_tls_connection("localhost", port, client_context, timeout=2.0) as connection:
            assert connection.recv(2) == b"ok"

        assert len(server.handled) == 1


def test_create_tls_connection_wraps_socket_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise_connection(*args, **kwargs):
        raise OSError("connection refused")

    monkeypatch.setattr(socket, "create_connection", _raise_connection)

    with pytest.raises(TLSError, match="Failed to create TLS connection"):
        create_tls_connection("127.0.0.1", 443, ssl.create_default_context(), timeout=0.01)


def test_create_tls_connection_enforces_tls12_floor(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeSocket:
        pass

    class FakeSSLContext:
        def __init__(self) -> None:
            self.minimum_version = ssl.TLSVersion.TLSv1
            self.options = 0
            self.wrap_calls: list[tuple[object, str]] = []

        def wrap_socket(self, sock, *, server_hostname):
            self.wrap_calls.append((sock, server_hostname))
            return "wrapped-socket"

    monkeypatch.setattr(socket, "create_connection", lambda *args, **kwargs: FakeSocket())

    context = FakeSSLContext()
    wrapped = create_tls_connection("127.0.0.1", 443, context, timeout=0.01)

    assert wrapped == "wrapped-socket"
    assert context.minimum_version == ssl.TLSVersion.TLSv1_2
    tlsv1_disable = getattr(ssl, "OP_NO_TLSv1", 0)
    tlsv11_disable = getattr(ssl, "OP_NO_TLSv1_1", 0)
    if tlsv1_disable:
        assert context.options & tlsv1_disable
    if tlsv11_disable:
        assert context.options & tlsv11_disable
    assert context.wrap_calls and context.wrap_calls[0][1] == "127.0.0.1"
