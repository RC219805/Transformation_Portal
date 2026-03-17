"""Cryptographic signing for reproducibility certificates.

This module provides Ed25519-based digital signatures for execution
manifests, enabling:
- Verifiable execution records
- Supply-chain attestation
- Tamper-evident certificates
- External verification

Example:
    >>> priv, pub = generate_ed25519_keypair()
    >>> signer = CertificateSigner(priv)
    >>>
    >>> cert = signer.sign_manifest(manifest_json)
    >>> cert.save(Path("signed_cert.json"))
    >>>
    >>> # Later: verify
    >>> assert CertificateVerifier.verify(manifest_json, cert)
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# Check for cryptography library
_CRYPTO_AVAILABLE = False
try:
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (
        Ed25519PrivateKey,
        Ed25519PublicKey,
    )

    _CRYPTO_AVAILABLE = True
except ImportError:
    logger.warning("cryptography library not available. " "Install with: pip install cryptography")


class SigningError(RuntimeError):
    """Raised for signing/verification errors."""


def _require_crypto() -> None:
    """Raise if cryptography is not available."""
    if not _CRYPTO_AVAILABLE:
        raise SigningError("cryptography library required for signing. " "Install with: pip install cryptography")


def _canonical_json(data: str | Dict[str, Any]) -> str:
    """Convert to canonical JSON for deterministic hashing.

    Args:
        data: JSON string or dictionary

    Returns:
        Canonical JSON string (sorted keys, no whitespace)
    """
    if isinstance(data, str):
        obj = json.loads(data)
    else:
        obj = data
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def _b64_encode(b: bytes) -> str:
    """Base64 encode bytes to string."""
    return base64.b64encode(b).decode("ascii")


def _b64_decode(s: str) -> bytes:
    """Base64 decode string to bytes."""
    return base64.b64decode(s.encode("ascii"))


def generate_ed25519_keypair() -> Tuple[bytes, bytes]:
    """Generate a new Ed25519 keypair.

    Returns:
        Tuple of (private_key_bytes, public_key_bytes)

    Raises:
        SigningError: If cryptography library not available
    """
    _require_crypto()

    priv = Ed25519PrivateKey.generate()
    pub = priv.public_key()

    priv_bytes = priv.private_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PrivateFormat.Raw,
        encryption_algorithm=serialization.NoEncryption(),
    )
    pub_bytes = pub.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )

    logger.debug("Generated new Ed25519 keypair")
    return priv_bytes, pub_bytes


def load_private_key(path: Path) -> bytes:
    """Load private key from file.

    Args:
        path: Path to key file (raw bytes or PEM)

    Returns:
        Private key bytes
    """
    content = path.read_bytes()

    # Try raw bytes first (32 bytes)
    if len(content) == 32:
        return content

    # Try PEM format
    _require_crypto()
    try:
        key = serialization.load_pem_private_key(content, password=None)
        return key.private_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PrivateFormat.Raw,
            encryption_algorithm=serialization.NoEncryption(),
        )
    except Exception as e:
        raise SigningError(f"Failed to load private key: {e}")


def save_keypair(
    priv_bytes: bytes,
    pub_bytes: bytes,
    priv_path: Path,
    pub_path: Path,
) -> None:
    """Save keypair to files.

    Args:
        priv_bytes: Private key bytes
        pub_bytes: Public key bytes
        priv_path: Path for private key
        pub_path: Path for public key
    """
    priv_path.write_bytes(priv_bytes)
    pub_path.write_bytes(pub_bytes)
    logger.info("Saved keypair to %s and %s", priv_path, pub_path)


@dataclass(frozen=True)
class SignedCertificate:
    """Cryptographically signed certificate.

    Attributes:
        certificate_id: Unique certificate identifier
        manifest_hash: SHA-256 hash of canonical manifest JSON
        root_hash: Merkle root hash from manifest
        public_key_b64: Base64-encoded public key
        signature_b64: Base64-encoded Ed25519 signature
        signed_at: ISO timestamp of signing
        issuer: Certificate issuer identifier
        metadata: Additional certificate metadata
    """

    certificate_id: str
    manifest_hash: str
    root_hash: str
    public_key_b64: str
    signature_b64: str
    signed_at: str
    issuer: str = "transformation-portal"
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        # Handle frozen dataclass with mutable default
        if self.metadata is None:
            object.__setattr__(self, "metadata", {})

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "certificate_id": self.certificate_id,
            "manifest_hash": self.manifest_hash,
            "root_hash": self.root_hash,
            "public_key_b64": self.public_key_b64,
            "signature_b64": self.signature_b64,
            "signed_at": self.signed_at,
            "issuer": self.issuer,
            "metadata": self.metadata or {},
        }

    def to_json(self, *, pretty: bool = True) -> str:
        """Convert to JSON string.

        Args:
            pretty: If True, format with indentation

        Returns:
            JSON string
        """
        indent = 2 if pretty else None
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    def save(self, path: Path) -> None:
        """Save certificate to file.

        Args:
            path: Output file path
        """
        path.write_text(self.to_json())
        logger.info("Saved signed certificate to %s", path)

    @classmethod
    def load(cls, path: Path) -> "SignedCertificate":
        """Load certificate from file.

        Args:
            path: Input file path

        Returns:
            SignedCertificate
        """
        data = json.loads(path.read_text())
        return cls(
            certificate_id=data["certificate_id"],
            manifest_hash=data["manifest_hash"],
            root_hash=data["root_hash"],
            public_key_b64=data["public_key_b64"],
            signature_b64=data["signature_b64"],
            signed_at=data["signed_at"],
            issuer=data.get("issuer", "unknown"),
            metadata=data.get("metadata", {}),
        )


class CertificateSigner:
    """Signs execution manifests with Ed25519.

    Example:
        >>> priv, pub = generate_ed25519_keypair()
        >>> signer = CertificateSigner(priv, issuer="my-system")
        >>>
        >>> cert = signer.sign_manifest(manifest_json)
        >>> print(f"Certificate ID: {cert.certificate_id}")
    """

    def __init__(
        self,
        private_key_bytes: bytes,
        *,
        issuer: str = "transformation-portal",
    ) -> None:
        """Initialize signer.

        Args:
            private_key_bytes: 32-byte Ed25519 private key
            issuer: Certificate issuer identifier
        """
        _require_crypto()

        self._priv = Ed25519PrivateKey.from_private_bytes(private_key_bytes)
        self._pub = self._priv.public_key()
        self._issuer = issuer
        self._sign_count = 0

    def sign_manifest(
        self,
        manifest_json: str,
        *,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SignedCertificate:
        """Sign an execution manifest.

        Args:
            manifest_json: JSON string of the manifest
            metadata: Additional metadata to include

        Returns:
            SignedCertificate
        """
        # Canonicalize and hash manifest
        canon = _canonical_json(manifest_json).encode("utf-8")
        manifest_hash = hashlib.sha256(canon).hexdigest()

        # Extract root hash from manifest
        data = json.loads(manifest_json)
        root_hash = data.get("root_hash", "")

        # Create payload: manifest_hash || root_hash
        payload = (manifest_hash + root_hash).encode("utf-8")

        # Sign with Ed25519
        signature = self._priv.sign(payload)

        # Get public key bytes
        pub_bytes = self._pub.public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )

        # Generate certificate ID
        self._sign_count += 1
        cert_id = f"cert_{manifest_hash[:8]}_{self._sign_count:04d}"

        cert = SignedCertificate(
            certificate_id=cert_id,
            manifest_hash=manifest_hash,
            root_hash=root_hash,
            public_key_b64=_b64_encode(pub_bytes),
            signature_b64=_b64_encode(signature),
            signed_at=datetime.now(timezone.utc).isoformat(),
            issuer=self._issuer,
            metadata=metadata or {},
        )

        logger.info(
            "Signed manifest: cert_id=%s, manifest_hash=%s",
            cert_id,
            manifest_hash[:8],
        )

        return cert

    @property
    def public_key_bytes(self) -> bytes:
        """Get public key bytes."""
        return self._pub.public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )

    @property
    def public_key_b64(self) -> str:
        """Get public key as base64."""
        return _b64_encode(self.public_key_bytes)


class CertificateVerifier:
    """Verifies signed certificates.

    Example:
        >>> cert = SignedCertificate.load(Path("cert.json"))
        >>> manifest_json = Path("manifest.json").read_text()
        >>>
        >>> if CertificateVerifier.verify(manifest_json, cert):
        ...     print("Certificate valid!")
    """

    @staticmethod
    def verify(
        manifest_json: str,
        cert: SignedCertificate,
    ) -> bool:
        """Verify a signed certificate.

        Args:
            manifest_json: Original manifest JSON
            cert: Certificate to verify

        Returns:
            True if certificate is valid
        """
        _require_crypto()

        try:
            # Verify manifest hash
            canon = _canonical_json(manifest_json).encode("utf-8")
            manifest_hash = hashlib.sha256(canon).hexdigest()

            if manifest_hash != cert.manifest_hash:
                logger.warning("Manifest hash mismatch")
                return False

            # Verify root hash from manifest
            data = json.loads(manifest_json)
            if data.get("root_hash", "") != cert.root_hash:
                logger.warning("Root hash mismatch")
                return False

            # Verify signature
            payload = (cert.manifest_hash + cert.root_hash).encode("utf-8")
            pub_bytes = _b64_decode(cert.public_key_b64)
            pub = Ed25519PublicKey.from_public_bytes(pub_bytes)

            sig = _b64_decode(cert.signature_b64)
            pub.verify(sig, payload)

            logger.debug("Certificate verified: %s", cert.certificate_id)
            return True

        except Exception as e:
            logger.warning("Certificate verification failed: %s", e)
            return False

    @staticmethod
    def verify_with_public_key(
        manifest_json: str,
        cert: SignedCertificate,
        public_key_bytes: bytes,
    ) -> bool:
        """Verify certificate with a specific public key.

        Args:
            manifest_json: Original manifest JSON
            cert: Certificate to verify
            public_key_bytes: Expected public key

        Returns:
            True if valid and signed by the specified key
        """
        # First check public key matches
        cert_pub = _b64_decode(cert.public_key_b64)
        if cert_pub != public_key_bytes:
            logger.warning("Public key mismatch")
            return False

        return CertificateVerifier.verify(manifest_json, cert)
