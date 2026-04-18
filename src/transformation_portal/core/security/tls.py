"""TLS utilities for secure node communication.

This module provides mTLS (mutual TLS) support for all network
traffic between nodes. Features:
- Server and client SSL contexts
- Certificate verification
- CA trust bundle management

Example:
    >>> # Server side
    >>> ssl_ctx = create_server_ssl_context(
    ...     certfile=Path("server.crt"),
    ...     keyfile=Path("server.key"),
    ...     cafile=Path("ca.crt"),
    ... )
    >>>
    >>> # Client side
    >>> ssl_ctx = create_client_ssl_context(
    ...     certfile=Path("client.crt"),
    ...     keyfile=Path("client.key"),
    ...     cafile=Path("ca.crt"),
    ... )
"""

from __future__ import annotations

import logging
import socket
import socketserver
import ssl
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)
_MINIMUM_TLS_VERSION = ssl.TLSVersion.TLSv1_2


class TLSError(RuntimeError):
    """Raised for TLS configuration or connection errors."""


def _enforce_minimum_tls_version(
    ctx: ssl.SSLContext,
    *,
    min_version: ssl.TLSVersion = _MINIMUM_TLS_VERSION,
) -> ssl.SSLContext:
    """Ensure the SSL context enforces a minimum TLS protocol version."""
    current_min_version = getattr(ctx, "minimum_version", None)
    if current_min_version is None or current_min_version < min_version:
        ctx.minimum_version = min_version
    return ctx


def create_server_ssl_context(
    certfile: Path,
    keyfile: Path,
    cafile: Path,
    *,
    verify_client: bool = True,
    min_version: ssl.TLSVersion = ssl.TLSVersion.TLSv1_2,
) -> ssl.SSLContext:
    """Create SSL context for server with client verification.

    Args:
        certfile: Path to server certificate (PEM)
        keyfile: Path to server private key (PEM)
        cafile: Path to CA certificate bundle (PEM)
        verify_client: If True, require client certificate
        min_version: Minimum TLS version

    Returns:
        Configured SSLContext

    Raises:
        TLSError: If certificate loading fails
    """
    try:
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        _enforce_minimum_tls_version(ctx, min_version=min_version)

        # Load server certificate and key
        ctx.load_cert_chain(
            certfile=str(certfile),
            keyfile=str(keyfile),
        )

        # Load CA for client verification
        ctx.load_verify_locations(cafile=str(cafile))

        if verify_client:
            ctx.verify_mode = ssl.CERT_REQUIRED
        else:
            ctx.verify_mode = ssl.CERT_OPTIONAL

        # Security hardening
        ctx.set_ciphers("ECDHE+AESGCM:DHE+AESGCM:ECDHE+CHACHA20:DHE+CHACHA20")
        ctx.options |= ssl.OP_NO_SSLv2 | ssl.OP_NO_SSLv3

        logger.info("Created server SSL context: cert=%s, verify_client=%s", certfile, verify_client)
        return ctx

    except Exception as e:
        raise TLSError(f"Failed to create server SSL context: {e}")


def create_client_ssl_context(
    certfile: Path,
    keyfile: Path,
    cafile: Path,
    *,
    verify_hostname: bool = True,
    min_version: ssl.TLSVersion = ssl.TLSVersion.TLSv1_2,
) -> ssl.SSLContext:
    """Create SSL context for client with server verification.

    Args:
        certfile: Path to client certificate (PEM)
        keyfile: Path to client private key (PEM)
        cafile: Path to CA certificate bundle (PEM)
        verify_hostname: If True, verify server hostname
        min_version: Minimum TLS version

    Returns:
        Configured SSLContext

    Raises:
        TLSError: If certificate loading fails
    """
    try:
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        _enforce_minimum_tls_version(ctx, min_version=min_version)

        # Load client certificate and key
        ctx.load_cert_chain(
            certfile=str(certfile),
            keyfile=str(keyfile),
        )

        # Load CA for server verification
        ctx.load_verify_locations(cafile=str(cafile))
        ctx.check_hostname = verify_hostname
        ctx.verify_mode = ssl.CERT_REQUIRED

        # Security hardening
        ctx.set_ciphers("ECDHE+AESGCM:DHE+AESGCM:ECDHE+CHACHA20:DHE+CHACHA20")

        logger.info("Created client SSL context: cert=%s", certfile)
        return ctx

    except Exception as e:
        raise TLSError(f"Failed to create client SSL context: {e}")


class TLSTCPServer(socketserver.TCPServer):
    """TCP server with TLS support.

    Wraps all connections with SSL/TLS using the provided context.

    Example:
        >>> ssl_ctx = create_server_ssl_context(...)
        >>> server = TLSTCPServer(("0.0.0.0", 5000), MyHandler, ssl_ctx)
        >>> server.serve_forever()
    """

    allow_reuse_address = True

    def __init__(
        self,
        server_address: Tuple[str, int],
        handler_class,
        ssl_context: ssl.SSLContext,
    ) -> None:
        """Initialize TLS server.

        Args:
            server_address: (host, port) tuple
            handler_class: Request handler class
            ssl_context: SSL context for wrapping connections
        """
        self.ssl_context = ssl_context
        super().__init__(server_address, handler_class)

    def get_request(self) -> Tuple[socket.socket, Tuple[str, int]]:
        """Accept connection and wrap with TLS."""
        newsocket, fromaddr = super().get_request()
        try:
            connstream = self.ssl_context.wrap_socket(
                newsocket,
                server_side=True,
            )
            return connstream, fromaddr
        except ssl.SSLError as e:
            logger.warning("TLS handshake failed from %s: %s", fromaddr, e)
            newsocket.close()
            raise


class ThreadedTLSServer(socketserver.ThreadingMixIn, TLSTCPServer):
    """Threaded TCP server with TLS support."""

    daemon_threads = True


def create_tls_connection(
    host: str,
    port: int,
    ssl_context: ssl.SSLContext,
    *,
    timeout: float = 30.0,
    server_hostname: Optional[str] = None,
) -> ssl.SSLSocket:
    """Create a TLS-wrapped connection.

    Args:
        host: Target hostname
        port: Target port
        ssl_context: Client SSL context
        timeout: Connection timeout
        server_hostname: Override server hostname for verification

    Returns:
        TLS-wrapped socket

    Raises:
        TLSError: If connection fails
    """
    try:
        _enforce_minimum_tls_version(ssl_context)
        sock = socket.create_connection((host, port), timeout=timeout)
        return ssl_context.wrap_socket(
            sock,
            server_hostname=server_hostname or host,
        )
    except Exception as e:
        raise TLSError(f"Failed to create TLS connection to {host}:{port}: {e}")


def generate_self_signed_cert(
    common_name: str,
    output_dir: Path,
    *,
    days_valid: int = 365,
) -> Tuple[Path, Path]:
    """Generate a self-signed certificate for testing.

    Note: Requires the cryptography library.

    Args:
        common_name: Certificate CN
        output_dir: Directory for output files
        days_valid: Certificate validity in days

    Returns:
        Tuple of (cert_path, key_path)
    """
    try:
        from datetime import datetime, timedelta, timezone

        from cryptography import x509
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import rsa
        from cryptography.x509.oid import NameOID

        # Generate key
        key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
        )

        # Generate certificate
        subject = issuer = x509.Name(
            [
                x509.NameAttribute(NameOID.COMMON_NAME, common_name),
            ]
        )

        cert = (
            x509.CertificateBuilder()
            .subject_name(subject)
            .issuer_name(issuer)
            .public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(datetime.now(timezone.utc))
            .not_valid_after(datetime.now(timezone.utc) + timedelta(days=days_valid))
            .add_extension(
                x509.SubjectAlternativeName(
                    [
                        x509.DNSName(common_name),
                        x509.DNSName("localhost"),
                    ]
                ),
                critical=False,
            )
            .sign(key, hashes.SHA256())
        )

        # Write files
        output_dir.mkdir(parents=True, exist_ok=True)

        cert_path = output_dir / f"{common_name}.crt"
        key_path = output_dir / f"{common_name}.key"

        cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
        key_path.write_bytes(
            key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.TraditionalOpenSSL,
                encryption_algorithm=serialization.NoEncryption(),
            )
        )

        logger.info("Generated self-signed certificate: %s", cert_path)
        return cert_path, key_path

    except ImportError:
        raise TLSError("cryptography library required for certificate generation")
