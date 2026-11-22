"""Immutable Execution Ledger with hash-chaining and signing.

This module provides a tamper-evident, append-only ledger for
recording pipeline executions. Features:
- Append-only structure
- Hash-chaining (each entry links to previous)
- Optional signing per entry
- Verification of integrity

Example:
    >>> ledger = ImmutableLedger(Path("ledger.log"), signer=signer)
    >>>
    >>> # Append execution record
    >>> entry = ledger.append(manifest_json, certificate)
    >>>
    >>> # Verify ledger integrity
    >>> assert ledger.verify()
    >>>
    >>> # Get history
    >>> for entry in ledger.iter_entries():
    ...     print(f"Run: {entry.manifest['run_id']}")
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

try:  # POSIX advisory locking; append fails closed without it.
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX platforms
    fcntl = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

# Fields that make a stored certificate a SignedCertificate bound to its
# manifest; such certificates are checked against the manifest in verify().
_SIGNED_CERTIFICATE_FIELDS = ("manifest_hash", "root_hash", "public_key_b64", "signature_b64")


def _hash_dict(obj: Dict[str, Any]) -> str:
    """Compute SHA-256 hash of a dictionary.

    Uses canonical JSON for determinism.
    """
    canon = json.dumps(obj, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()


class LedgerError(RuntimeError):
    """Raised for ledger errors."""


@dataclass(frozen=True)
class LedgerEntry:
    """Entry in the immutable ledger.

    Attributes:
        entry_id: Sequential entry identifier
        prev_hash: Hash of previous entry (or "GENESIS")
        entry_hash: Hash of this entry's content
        timestamp: ISO timestamp of entry creation
        manifest: Execution manifest data
        certificate: Signed certificate data
        metadata: Additional entry metadata
    """

    entry_id: int
    prev_hash: str
    entry_hash: str
    timestamp: str
    manifest: Dict[str, Any]
    certificate: Dict[str, Any]
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "entry_id": self.entry_id,
            "prev_hash": self.prev_hash,
            "entry_hash": self.entry_hash,
            "timestamp": self.timestamp,
            "manifest": self.manifest,
            "certificate": self.certificate,
            "metadata": self.metadata,
        }

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), sort_keys=True)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LedgerEntry":
        """Create from dictionary."""
        return cls(
            entry_id=data["entry_id"],
            prev_hash=data["prev_hash"],
            entry_hash=data["entry_hash"],
            timestamp=data["timestamp"],
            manifest=data["manifest"],
            certificate=data["certificate"],
            metadata=data.get("metadata", {}),
        )


class ImmutableLedger:
    """Append-only, hash-chained execution ledger.

    Provides tamper-evident storage of execution records.
    Each entry is linked to the previous via hash-chaining,
    making any modification detectable.

    Example:
        >>> ledger = ImmutableLedger(Path("executions.ledger"))
        >>>
        >>> # Add entry
        >>> entry = ledger.append(manifest_json, cert)
        >>> print(f"Entry {entry.entry_id}: {entry.entry_hash[:8]}")
        >>>
        >>> # Verify integrity
        >>> if not ledger.verify():
        ...     raise RuntimeError("Ledger tampered!")
        >>>
        >>> # Query history
        >>> for entry in ledger.iter_entries():
        ...     print(entry.manifest["run_id"])
    """

    GENESIS_HASH = "GENESIS"

    def __init__(
        self,
        path: Path,
        *,
        signer: Optional["CertificateSigner"] = None,
        create: bool = True,
    ) -> None:
        """Initialize ledger.

        Args:
            path: Path to ledger file
            signer: Optional signer, kept on the handle; the ledger does not
                yet sign entry hashes itself (embedded certificates are
                checked by verify())
            create: If True, create file if it doesn't exist
        """
        self.path = path
        self.signer = signer
        self._entry_count = 0
        self._last_hash = self.GENESIS_HASH

        # Ensure parent directory exists
        path.parent.mkdir(parents=True, exist_ok=True)

        if create and not path.exists():
            # Append mode creates the file without truncating one that a
            # concurrent handle created (and appended to) in the meantime.
            with path.open("a", encoding="utf-8"):
                pass
            logger.info("Created new ledger: %s", path)
        if path.exists():
            # Load existing state
            self._load_state()

    def _load_state(self) -> None:
        """Load state from existing ledger."""
        content = self.path.read_text(encoding="utf-8")
        lines = [line for line in content.splitlines() if line.strip()]

        if not lines:
            self._entry_count = 0
            self._last_hash = self.GENESIS_HASH
            return

        # Get last entry
        last = json.loads(lines[-1])
        self._entry_count = last["entry_id"]
        self._last_hash = last["entry_hash"]

        logger.debug(
            "Loaded ledger: %d entries, last_hash=%s",
            self._entry_count,
            self._last_hash[:8],
        )

    def _compute_entry_hash(
        self,
        prev_hash: str,
        manifest: Dict[str, Any],
        certificate: Dict[str, Any],
        metadata: Dict[str, Any],
        *,
        entry_id: Optional[int] = None,
        timestamp: Optional[str] = None,
    ) -> str:
        """Compute hash for entry content.

        Entries appended by this version also bind ``entry_id`` and
        ``timestamp`` into the hashed body. Without them the legacy body
        (written by older versions) is hashed; ``verify`` still accepts it
        for entries that were written that way.
        """
        body: Dict[str, Any] = {
            "prev_hash": prev_hash,
            "manifest": manifest,
            "certificate": certificate,
            "metadata": metadata,
        }
        if entry_id is not None or timestamp is not None:
            body["entry_id"] = entry_id
            body["timestamp"] = timestamp
        return _hash_dict(body)

    @staticmethod
    def _read_tail_entry(handle: Any) -> Optional[Dict[str, Any]]:
        """Return the last non-empty line of an open binary ledger as a dict.

        Reads backwards from the end so an append does not re-read the whole
        ledger. Raises LedgerError when the tail is not a complete, parseable
        entry (for example a torn write), so nothing is chained onto it.
        """
        handle.seek(0, os.SEEK_END)
        pos = handle.tell()
        buf = b""
        while pos > 0:
            step = min(8192, pos)
            pos -= step
            handle.seek(pos)
            buf = handle.read(step) + buf
            stripped = buf.rstrip()
            if not stripped:
                continue
            newline = stripped.rfind(b"\n")
            if newline < 0 and pos > 0:
                continue
            if not buf.endswith(b"\n"):
                raise LedgerError(f"Ledger tail is not newline-terminated (torn write?): {handle.name}")
            try:
                data = json.loads(stripped[newline + 1 :].decode("utf-8"))
                if not isinstance(data, dict):
                    raise TypeError("entry is not a JSON object")
                int(data["entry_id"])
                str(data["entry_hash"])
            except (ValueError, KeyError, TypeError) as exc:
                raise LedgerError(f"Ledger tail entry is corrupt: {exc}") from exc
            return data
        return None

    def append(
        self,
        manifest_json: str,
        certificate: "SignedCertificate",
        *,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> LedgerEntry:
        """Append a new entry to the ledger.

        Args:
            manifest_json: Execution manifest as JSON
            certificate: Signed certificate
            metadata: Additional metadata

        Returns:
            Created LedgerEntry
        """
        # Serialise everything first: a value that cannot be stored fails here,
        # before any state changes. Hash the JSON round-tripped values, i.e.
        # exactly what is written and later re-read by verify().
        manifest = json.loads(manifest_json)
        cert_dict = json.loads(certificate.to_json())
        entry_metadata = json.loads(json.dumps(metadata or {}))

        if fcntl is None:  # pragma: no cover - non-POSIX platforms
            raise LedgerError("ImmutableLedger.append requires POSIX file locking (fcntl)")

        with self.path.open("a+b") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                # Re-read the head under the lock: another handle or process
                # may have appended since this handle cached its state.
                tail = self._read_tail_entry(handle)
                if tail is None:
                    prev_id, prev_hash = 0, self.GENESIS_HASH
                else:
                    prev_id, prev_hash = int(tail["entry_id"]), str(tail["entry_hash"])

                entry_id = prev_id + 1
                timestamp = datetime.now(timezone.utc).isoformat()
                entry_hash = self._compute_entry_hash(
                    prev_hash,
                    manifest,
                    cert_dict,
                    entry_metadata,
                    entry_id=entry_id,
                    timestamp=timestamp,
                )
                entry = LedgerEntry(
                    entry_id=entry_id,
                    prev_hash=prev_hash,
                    entry_hash=entry_hash,
                    timestamp=timestamp,
                    manifest=manifest,
                    certificate=cert_dict,
                    metadata=entry_metadata,
                )
                line = (entry.to_json() + "\n").encode("utf-8")

                handle.seek(0, os.SEEK_END)
                start = handle.tell()
                try:
                    handle.write(line)
                    handle.flush()
                    os.fsync(handle.fileno())
                except BaseException:
                    # Roll back a partial line so later appends chain cleanly.
                    try:
                        handle.truncate(start)
                        handle.flush()
                    except OSError:
                        logger.exception("Could not roll back a failed ledger append: %s", self.path)
                    raise

                # Update cached state only once the entry is durable.
                self._entry_count = entry_id
                self._last_hash = entry_hash
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

        logger.info(
            "Appended ledger entry %d: hash=%s, prev=%s",
            entry_id,
            entry_hash[:8],
            prev_hash[:8] if prev_hash != self.GENESIS_HASH else "GENESIS",
        )

        return entry

    def verify(self) -> bool:
        """Verify ledger integrity.

        Checks:
        - Hash chain is unbroken
        - Entry hashes are correct (including entry_id and timestamp for
          entries written by this version)
        - No missing entries, and the head this handle last appended or
          loaded is still present (truncation and emptying are detected)
        - SignedCertificate-shaped certificates still match their manifest

        The chain itself is unkeyed: a writer who recomputes every hash can
        still rewrite history; only the embedded certificates are signed.
        A malformed ledger file verifies as False rather than raising.

        Returns:
            True if ledger is valid
        """
        try:
            return self._verify_entries()
        except (ValueError, KeyError, TypeError, AttributeError) as exc:
            logger.error("Ledger is malformed: %s", exc)
            return False

    def _verify_entries(self) -> bool:
        """Body of verify(); malformed content may raise."""
        prev_hash = self.GENESIS_HASH
        expected_id = 0
        head_seen = self._entry_count == 0

        for entry in self.iter_entries():
            expected_id += 1

            # Check entry ID sequence
            if entry.entry_id != expected_id:
                logger.error(
                    "Entry ID mismatch: expected %d, got %d",
                    expected_id,
                    entry.entry_id,
                )
                return False

            # Check prev_hash linkage
            if entry.prev_hash != prev_hash:
                logger.error(
                    "Chain broken at entry %d: expected prev=%s, got %s",
                    entry.entry_id,
                    prev_hash[:8],
                    entry.prev_hash[:8],
                )
                return False

            # Verify entry hash: the current body first, then the legacy body
            # for entries written before entry_id/timestamp were hashed.
            computed = self._compute_entry_hash(
                entry.prev_hash,
                entry.manifest,
                entry.certificate,
                entry.metadata,
                entry_id=entry.entry_id,
                timestamp=entry.timestamp,
            )
            if computed != entry.entry_hash:
                legacy = self._compute_entry_hash(
                    entry.prev_hash,
                    entry.manifest,
                    entry.certificate,
                    entry.metadata,
                )
                if legacy != entry.entry_hash:
                    logger.error(
                        "Hash mismatch at entry %d: expected %s, got %s",
                        entry.entry_id,
                        entry.entry_hash[:8],
                        computed[:8],
                    )
                    return False

            if not self._certificate_matches_manifest(entry):
                logger.error("Certificate of entry %d does not match its manifest", entry.entry_id)
                return False

            if entry.entry_id == self._entry_count:
                if entry.entry_hash != self._last_hash:
                    logger.error("Entry %d differs from the head this handle appended or loaded", entry.entry_id)
                    return False
                head_seen = True

            prev_hash = entry.entry_hash

        if not head_seen:
            logger.error(
                "Ledger truncated: %d entries on disk, this handle saw %d",
                expected_id,
                self._entry_count,
            )
            return False

        logger.info("Ledger verified: %d entries valid", expected_id)
        return True

    @staticmethod
    def _certificate_matches_manifest(entry: LedgerEntry) -> bool:
        """Check a SignedCertificate-shaped certificate against its manifest.

        Certificates without the signed-certificate fields (opaque caller
        data) are not interpreted. Missing crypto support fails closed.
        """
        cert = entry.certificate
        if not isinstance(cert, dict) or not any(field in cert for field in _SIGNED_CERTIFICATE_FIELDS):
            return True
        from transformation_portal.core.security.signing import CertificateVerifier, SignedCertificate, SigningError

        try:
            signed = SignedCertificate(
                certificate_id=cert["certificate_id"],
                manifest_hash=cert["manifest_hash"],
                root_hash=cert["root_hash"],
                public_key_b64=cert["public_key_b64"],
                signature_b64=cert["signature_b64"],
                signed_at=cert["signed_at"],
                issuer=cert.get("issuer", "unknown"),
                metadata=cert.get("metadata", {}),
            )
            return CertificateVerifier.verify(json.dumps(entry.manifest), signed)
        except (KeyError, TypeError, SigningError) as exc:
            logger.error("Certificate of entry %d is unusable: %s", entry.entry_id, exc)
            return False

    def iter_entries(self) -> Iterator[LedgerEntry]:
        """Iterate over all entries in order.

        Yields:
            LedgerEntry objects
        """
        content = self.path.read_text(encoding="utf-8")

        for line in content.splitlines():
            if not line.strip():
                continue

            data = json.loads(line)
            yield LedgerEntry.from_dict(data)

    def get_entries(self) -> List[LedgerEntry]:
        """Get all entries as a list.

        Returns:
            List of LedgerEntry objects
        """
        return list(self.iter_entries())

    def get_entry(self, entry_id: int) -> Optional[LedgerEntry]:
        """Get a specific entry by ID.

        Args:
            entry_id: Entry ID to find

        Returns:
            LedgerEntry or None if not found
        """
        for entry in self.iter_entries():
            if entry.entry_id == entry_id:
                return entry
        return None

    def get_by_run_id(self, run_id: str) -> Optional[LedgerEntry]:
        """Find entry by run ID.

        Args:
            run_id: Run ID to search for

        Returns:
            LedgerEntry or None if not found
        """
        for entry in self.iter_entries():
            if entry.manifest.get("run_id") == run_id:
                return entry
        return None

    @property
    def entry_count(self) -> int:
        """Number of entries in ledger."""
        return self._entry_count

    @property
    def last_hash(self) -> str:
        """Hash of the last entry."""
        return self._last_hash

    def get_summary(self) -> Dict[str, Any]:
        """Get ledger summary.

        Returns:
            Dictionary with ledger statistics
        """
        return {
            "path": str(self.path),
            "entry_count": self._entry_count,
            "last_hash": self._last_hash,
            "is_valid": self.verify() if self._entry_count > 0 else True,
        }


# Import for type hints
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from transformation_portal.core.security.signing import (
        CertificateSigner,
        SignedCertificate,
    )
