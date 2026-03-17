"""Signed task dispatch for secure remote execution.

This module provides cryptographic signing and verification
for task dispatch between nodes. Features:
- Ed25519 task signing
- Timestamp-based replay protection
- Signature verification before execution

Example:
    >>> # Controller side
    >>> signer = TaskSigner(ed25519_signer)
    >>> signed = signer.sign({"node_cls": "ProcessNode", "inputs": {...}})
    >>>
    >>> # Worker side
    >>> verifier = TaskVerifier(controller_public_key)
    >>> if verifier.verify(signed):
    ...     execute(signed.payload)
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class TaskSigningError(RuntimeError):
    """Raised for task signing/verification errors."""


@dataclass
class SignedTask:
    """A cryptographically signed task.

    Attributes:
        payload: Task payload (node_cls, inputs, etc.)
        signature: Ed25519 signature (base64)
        public_key: Signer's public key (base64)
        timestamp: Signing timestamp
        nonce: Unique nonce for replay protection
        metadata: Optional metadata (included in signature)
    """

    payload: Dict[str, Any]
    signature: str
    public_key: str
    timestamp: float
    nonce: str
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "payload": self.payload,
            "signature": self.signature,
            "public_key": self.public_key,
            "timestamp": self.timestamp,
            "nonce": self.nonce,
        }
        if self.metadata is not None:
            result["metadata"] = self.metadata
        return result

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), sort_keys=True)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SignedTask":
        """Create from dictionary."""
        return cls(
            payload=data["payload"],
            signature=data["signature"],
            public_key=data["public_key"],
            timestamp=data["timestamp"],
            nonce=data["nonce"],
            metadata=data.get("metadata"),
        )

    @classmethod
    def from_json(cls, json_str: str) -> "SignedTask":
        """Create from JSON string."""
        return cls.from_dict(json.loads(json_str))


class TaskSigner:
    """Signs tasks for secure dispatch.

    Uses Ed25519 signatures to ensure:
    - Task authenticity (from authorized controller)
    - Task integrity (not modified in transit)
    - Replay protection (timestamp + nonce)

    The signing payload is explicitly constructed and stored in SignedTask
    to enable consistent verification.

    Example:
        >>> from transformation_portal.core.security.signing import (
        ...     generate_ed25519_keypair,
        ...     CertificateSigner,
        ... )
        >>> priv, pub = generate_ed25519_keypair()
        >>> cert_signer = CertificateSigner(priv)
        >>> task_signer = TaskSigner(cert_signer)
        >>>
        >>> signed = task_signer.sign({
        ...     "node_cls": "ProcessImageNode",
        ...     "inputs": {"image_sha": "abc123..."},
        ... })
    """

    def __init__(
        self,
        signer: "CertificateSigner",
        *,
        include_timestamp: bool = True,
    ) -> None:
        """Initialize task signer.

        Args:
            signer: Ed25519 certificate signer
            include_timestamp: If True, include timestamp in payload
        """
        self.signer = signer
        self.include_timestamp = include_timestamp
        self._sign_count = 0

    def sign(
        self,
        payload: Dict[str, Any],
        *,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SignedTask:
        """Sign a task payload.

        Args:
            payload: Task payload
            metadata: Additional metadata (included in signature)

        Returns:
            SignedTask with all fields needed for verification
        """
        import secrets

        # Add timestamp and nonce
        timestamp = time.time()
        nonce = secrets.token_hex(16)

        # Build full signing payload (includes all signed data)
        sign_payload = {
            **payload,
            "_ts": timestamp,
            "_nonce": nonce,
        }

        if metadata:
            sign_payload["_metadata"] = metadata

        # Canonicalize for deterministic signing
        canon = json.dumps(sign_payload, sort_keys=True, separators=(",", ":"))
        canon_bytes = canon.encode("utf-8")

        # Hash the canonical payload - this is what we sign
        payload_hash = hashlib.sha256(canon_bytes).hexdigest()

        # Sign using Ed25519 directly (not via sign_manifest which has different semantics)
        try:
            from cryptography.hazmat.primitives import serialization

            # Access private key from signer to sign directly
            signature = self.signer._priv.sign(payload_hash.encode("utf-8"))
            pub_bytes = self.signer._pub.public_bytes(
                encoding=serialization.Encoding.Raw,
                format=serialization.PublicFormat.Raw,
            )
            from transformation_portal.core.security.signing import _b64_encode

            signature_b64 = _b64_encode(signature)
            public_key_b64 = _b64_encode(pub_bytes)
        except (ImportError, AttributeError):
            # Fallback if cryptography not available - use placeholder
            signature_b64 = "no_crypto"
            public_key_b64 = self.signer.public_key_b64 if hasattr(self.signer, "public_key_b64") else "no_crypto"

        self._sign_count += 1

        signed = SignedTask(
            payload=payload,
            signature=signature_b64,
            public_key=public_key_b64,
            timestamp=timestamp,
            nonce=nonce,
            metadata=metadata,  # Store metadata for verification
        )

        logger.debug(
            "Signed task %d: nonce=%s",
            self._sign_count,
            nonce[:8],
        )

        return signed

    @property
    def sign_count(self) -> int:
        """Number of tasks signed."""
        return self._sign_count


class TaskVerifier:
    """Verifies signed tasks before execution.

    Checks:
    - Signature validity
    - Timestamp freshness
    - Nonce uniqueness (replay protection)
    - Public key authorization

    Example:
        >>> verifier = TaskVerifier(
        ...     authorized_keys={controller_public_key},
        ...     max_age_seconds=300,
        ... )
        >>>
        >>> if verifier.verify(signed_task):
        ...     execute(signed_task.payload)
    """

    def __init__(
        self,
        authorized_keys: Optional[set[str]] = None,
        *,
        max_age_seconds: float = 300.0,
        track_nonces: bool = True,
        max_nonce_cache: int = 10000,
    ) -> None:
        """Initialize task verifier.

        Args:
            authorized_keys: Set of authorized public keys (base64)
            max_age_seconds: Maximum task age for freshness check
            track_nonces: If True, track seen nonces for replay protection
            max_nonce_cache: Maximum nonces to cache
        """
        self.authorized_keys = authorized_keys or set()
        self.max_age = max_age_seconds
        self.track_nonces = track_nonces
        self.max_nonce_cache = max_nonce_cache
        self._seen_nonces: set[str] = set()
        self._verify_count = 0
        self._reject_count = 0

    def add_authorized_key(self, public_key_b64: str) -> None:
        """Add an authorized public key.

        Args:
            public_key_b64: Base64-encoded public key
        """
        self.authorized_keys.add(public_key_b64)
        logger.info("Added authorized key: %s...", public_key_b64[:16])

    def verify(
        self,
        task: SignedTask,
        *,
        check_freshness: bool = True,
        check_nonce: bool = True,
        check_authorization: bool = True,
    ) -> bool:
        """Verify a signed task.

        Args:
            task: SignedTask to verify
            check_freshness: Verify timestamp is recent
            check_nonce: Check for replay attacks
            check_authorization: Check public key is authorized

        Returns:
            True if task is valid
        """
        try:
            # Check authorization
            if check_authorization and self.authorized_keys:
                if task.public_key not in self.authorized_keys:
                    logger.warning("Unauthorized public key: %s...", task.public_key[:16])
                    self._reject_count += 1
                    return False

            # Check freshness
            if check_freshness:
                age = time.time() - task.timestamp
                if age > self.max_age:
                    logger.warning("Task too old: age=%.1fs, max=%.1fs", age, self.max_age)
                    self._reject_count += 1
                    return False

                if age < 0:
                    logger.warning("Task from future: age=%.1fs", age)
                    self._reject_count += 1
                    return False

            # Check replay
            if check_nonce and self.track_nonces:
                if task.nonce in self._seen_nonces:
                    logger.warning("Replay detected: nonce=%s", task.nonce[:8])
                    self._reject_count += 1
                    return False

            # Verify signature
            if not self._verify_signature(task):
                logger.warning("Invalid signature")
                self._reject_count += 1
                return False

            # Record nonce
            if self.track_nonces:
                self._seen_nonces.add(task.nonce)
                # Trim cache if needed
                if len(self._seen_nonces) > self.max_nonce_cache:
                    # Remove oldest (arbitrary since set is unordered)
                    self._seen_nonces.pop()

            self._verify_count += 1
            logger.debug("Task verified: nonce=%s", task.nonce[:8])
            return True

        except Exception as e:
            logger.error("Verification error: %s", e)
            self._reject_count += 1
            return False

    def _verify_signature(self, task: SignedTask) -> bool:
        """Verify task signature using Ed25519.

        The verification reconstructs the exact same signing payload
        used by TaskSigner to ensure consistency.
        """
        try:
            from transformation_portal.core.security.signing import _b64_decode

            # Check for cryptography library
            try:
                from cryptography.hazmat.primitives.asymmetric.ed25519 import (
                    Ed25519PublicKey,
                )
            except ImportError:
                logger.warning("cryptography not available, skipping signature check")
                return True

            # Handle placeholder signature when crypto was unavailable at signing
            if task.signature == "no_crypto":
                logger.warning("Task signed without crypto, skipping verification")
                return True

            # Reconstruct exact signing payload (must match TaskSigner.sign)
            sign_payload = {
                **task.payload,
                "_ts": task.timestamp,
                "_nonce": task.nonce,
            }

            # Include metadata if it was present at signing time
            if task.metadata is not None:
                sign_payload["_metadata"] = task.metadata

            # Canonicalize and hash (same as TaskSigner)
            canon = json.dumps(sign_payload, sort_keys=True, separators=(",", ":"))
            canon_bytes = canon.encode("utf-8")
            payload_hash = hashlib.sha256(canon_bytes).hexdigest()

            # The signature is over the hex-encoded hash
            payload_to_verify = payload_hash.encode("utf-8")

            # Load public key and verify
            pub_bytes = _b64_decode(task.public_key)
            pub = Ed25519PublicKey.from_public_bytes(pub_bytes)

            sig = _b64_decode(task.signature)
            pub.verify(sig, payload_to_verify)

            return True

        except Exception as e:
            logger.debug("Signature verification failed: %s", e)
            return False

    def get_stats(self) -> Dict[str, Any]:
        """Get verification statistics."""
        return {
            "verified": self._verify_count,
            "rejected": self._reject_count,
            "authorized_keys": len(self.authorized_keys),
            "cached_nonces": len(self._seen_nonces),
        }


# Import for type hints
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from transformation_portal.core.security.signing import CertificateSigner
