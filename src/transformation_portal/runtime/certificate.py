"""Reproducibility Certificate for verifiable execution records.

This module provides cryptographic certificates that prove:
- Run identity (manifest hash)
- Execution integrity (root hash)
- Verifiable signature

Certificates can be:
- Exported and shared
- Verified independently
- Used for audit trails
- Compared across runs

Example:
    >>> # Build certificate from manifest
    >>> builder = CertificateBuilder()
    >>> cert = builder.build(manifest)
    >>>
    >>> # Export certificate
    >>> cert.save(Path("certificate.json"))
    >>>
    >>> # Later: verify certificate
    >>> loaded = ReproducibilityCertificate.load(Path("certificate.json"))
    >>> assert loaded.verify(manifest)
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import secrets
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def _hash_str(s: str) -> str:
    """Compute SHA-256 hash of a string."""
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _hmac_sign(data: str, key: bytes) -> str:
    """Create HMAC-SHA256 signature."""
    return hmac.new(key, data.encode("utf-8"), hashlib.sha256).hexdigest()


@dataclass(frozen=True)
class CertificateMetadata:
    """Metadata for the certificate.

    Attributes:
        issuer: Certificate issuer identifier
        issued_at: ISO timestamp of issuance
        expires_at: Optional expiration timestamp
        version: Certificate format version
    """

    issuer: str
    issued_at: str
    expires_at: Optional[str] = None
    version: str = "1.0"


@dataclass
class ReproducibilityCertificate:
    """Cryptographic certificate for execution reproducibility.

    Provides verifiable proof of a pipeline execution including:
    - Manifest hash (run identity)
    - Root hash (Merkle root of all nodes)
    - Signature (integrity verification)

    Attributes:
        certificate_id: Unique certificate identifier
        manifest_hash: SHA-256 hash of the manifest JSON
        root_hash: Merkle root hash of all node hashes
        signature: HMAC signature for verification
        run_id: Original run identifier
        node_count: Number of nodes in the run
        metadata: Certificate metadata
        claims: Additional verifiable claims
    """

    certificate_id: str
    manifest_hash: str
    root_hash: str
    signature: str
    run_id: str
    node_count: int
    metadata: CertificateMetadata
    claims: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert certificate to dictionary."""
        return {
            "certificate_id": self.certificate_id,
            "manifest_hash": self.manifest_hash,
            "root_hash": self.root_hash,
            "signature": self.signature,
            "run_id": self.run_id,
            "node_count": self.node_count,
            "metadata": {
                "issuer": self.metadata.issuer,
                "issued_at": self.metadata.issued_at,
                "expires_at": self.metadata.expires_at,
                "version": self.metadata.version,
            },
            "claims": self.claims,
        }

    def to_json(self, *, pretty: bool = True) -> str:
        """Convert certificate to JSON string.

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
        logger.info("Saved reproducibility certificate to %s", path)

    @classmethod
    def load(cls, path: Path) -> "ReproducibilityCertificate":
        """Load certificate from file.

        Args:
            path: Input file path

        Returns:
            Loaded ReproducibilityCertificate
        """
        data = json.loads(path.read_text())

        meta_data = data.get("metadata", {})
        metadata = CertificateMetadata(
            issuer=meta_data.get("issuer", ""),
            issued_at=meta_data.get("issued_at", ""),
            expires_at=meta_data.get("expires_at"),
            version=meta_data.get("version", "1.0"),
        )

        cert = cls(
            certificate_id=data["certificate_id"],
            manifest_hash=data["manifest_hash"],
            root_hash=data["root_hash"],
            signature=data["signature"],
            run_id=data["run_id"],
            node_count=data["node_count"],
            metadata=metadata,
            claims=data.get("claims", {}),
        )

        logger.info("Loaded reproducibility certificate from %s", path)
        return cert

    def verify_manifest(self, manifest_json: str) -> bool:
        """Verify certificate against a manifest.

        Args:
            manifest_json: JSON string of the manifest

        Returns:
            True if manifest hash matches
        """
        computed_hash = _hash_str(manifest_json)
        return computed_hash == self.manifest_hash

    def verify_signature(self, key: bytes) -> bool:
        """Verify certificate signature.

        Args:
            key: HMAC key used for signing

        Returns:
            True if signature is valid
        """
        payload = f"{self.manifest_hash}:{self.root_hash}"
        expected = _hmac_sign(payload, key)
        return hmac.compare_digest(expected, self.signature)

    def is_expired(self) -> bool:
        """Check if certificate is expired.

        Returns:
            True if expired (or no expiration set)
        """
        if not self.metadata.expires_at:
            return False

        expires = datetime.fromisoformat(self.metadata.expires_at)
        return datetime.now(timezone.utc) > expires


class CertificateBuilder:
    """Builder for reproducibility certificates.

    Creates signed certificates from execution manifests.

    Example:
        >>> builder = CertificateBuilder(
        ...     issuer="transformation-portal",
        ...     signing_key=os.urandom(32),
        ... )
        >>>
        >>> cert = builder.build(manifest)
        >>> cert.save(Path("cert.json"))
    """

    def __init__(
        self,
        *,
        issuer: str = "transformation-portal",
        signing_key: Optional[bytes] = None,
    ) -> None:
        """Initialize certificate builder.

        Args:
            issuer: Certificate issuer identifier
            signing_key: HMAC key for signing (generated if not provided)
        """
        self.issuer = issuer
        self._signing_key = signing_key or secrets.token_bytes(32)

    def build(
        self,
        manifest: "ExecutionManifest",
        *,
        claims: Optional[Dict[str, Any]] = None,
        expires_in_days: Optional[int] = None,
    ) -> ReproducibilityCertificate:
        """Build certificate from manifest.

        Args:
            manifest: Execution manifest to certify
            claims: Additional claims to include
            expires_in_days: Certificate expiration in days

        Returns:
            ReproducibilityCertificate
        """
        # Generate certificate ID
        cert_id = f"cert_{secrets.token_hex(8)}"

        # Hash manifest
        manifest_json = manifest.to_json(pretty=False)
        manifest_hash = _hash_str(manifest_json)

        # Create signature
        payload = f"{manifest_hash}:{manifest.root_hash}"
        signature = _hmac_sign(payload, self._signing_key)

        # Compute expiration
        expires_at = None
        if expires_in_days:
            from datetime import timedelta

            expires = datetime.now(timezone.utc) + timedelta(days=expires_in_days)
            expires_at = expires.isoformat()

        # Build metadata
        metadata = CertificateMetadata(
            issuer=self.issuer,
            issued_at=datetime.now(timezone.utc).isoformat(),
            expires_at=expires_at,
        )

        cert = ReproducibilityCertificate(
            certificate_id=cert_id,
            manifest_hash=manifest_hash,
            root_hash=manifest.root_hash,
            signature=signature,
            run_id=manifest.run_id,
            node_count=len(manifest.node_hashes),
            metadata=metadata,
            claims=claims or {},
        )

        logger.info(
            "Built reproducibility certificate: id=%s, run=%s, nodes=%d",
            cert_id,
            manifest.run_id,
            len(manifest.node_hashes),
        )

        return cert

    def build_from_manifest_json(
        self,
        manifest_json: str,
        *,
        claims: Optional[Dict[str, Any]] = None,
    ) -> ReproducibilityCertificate:
        """Build certificate from manifest JSON.

        Args:
            manifest_json: JSON string of the manifest
            claims: Additional claims

        Returns:
            ReproducibilityCertificate
        """
        data = json.loads(manifest_json)

        # Extract required fields
        manifest_hash = _hash_str(manifest_json)
        root_hash = data["root_hash"]
        run_id = data["run_id"]
        node_count = len(data.get("node_hashes", []))

        # Generate certificate
        cert_id = f"cert_{secrets.token_hex(8)}"
        payload = f"{manifest_hash}:{root_hash}"
        signature = _hmac_sign(payload, self._signing_key)

        metadata = CertificateMetadata(
            issuer=self.issuer,
            issued_at=datetime.now(timezone.utc).isoformat(),
        )

        return ReproducibilityCertificate(
            certificate_id=cert_id,
            manifest_hash=manifest_hash,
            root_hash=root_hash,
            signature=signature,
            run_id=run_id,
            node_count=node_count,
            metadata=metadata,
            claims=claims or {},
        )

    @property
    def signing_key(self) -> bytes:
        """Get signing key (for verification)."""
        return self._signing_key


# Import for type hints
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from transformation_portal.runtime.execution_manifest import ExecutionManifest
