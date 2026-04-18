from __future__ import annotations

import builtins
import json
import socket
import ssl

import pytest

from transformation_portal.core.security import signing, tls
from transformation_portal.core.security.signing import (
    CertificateSigner,
    CertificateVerifier,
    SigningError,
    generate_ed25519_keypair,
)
from transformation_portal.core.security.tls import (
    TLSError,
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
            self.wrap_calls: list[tuple[object, str]] = []

        def wrap_socket(self, sock, *, server_hostname):
            self.wrap_calls.append((sock, server_hostname))
            return "wrapped-socket"

    monkeypatch.setattr(socket, "create_connection", lambda *args, **kwargs: FakeSocket())

    context = FakeSSLContext()
    wrapped = create_tls_connection("127.0.0.1", 443, context, timeout=0.01)

    assert wrapped == "wrapped-socket"
    assert context.minimum_version == ssl.TLSVersion.TLSv1_2
    assert context.wrap_calls and context.wrap_calls[0][1] == "127.0.0.1"
