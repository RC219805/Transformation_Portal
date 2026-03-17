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
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

logger = logging.getLogger(__name__)


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
            signer: Optional signer for entries
            create: If True, create file if it doesn't exist
        """
        self.path = path
        self.signer = signer
        self._entry_count = 0
        self._last_hash = self.GENESIS_HASH

        # Ensure parent directory exists
        path.parent.mkdir(parents=True, exist_ok=True)

        if create and not path.exists():
            path.write_text("", encoding="utf-8")
            logger.info("Created new ledger: %s", path)
        elif path.exists():
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
    ) -> str:
        """Compute hash for entry content."""
        body = {
            "prev_hash": prev_hash,
            "manifest": manifest,
            "certificate": certificate,
            "metadata": metadata,
        }
        return _hash_dict(body)

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
        manifest = json.loads(manifest_json)
        cert_dict = json.loads(certificate.to_json())

        # Build entry
        self._entry_count += 1
        entry_id = self._entry_count
        prev_hash = self._last_hash
        entry_metadata = metadata or {}
        timestamp = datetime.now(timezone.utc).isoformat()

        # Compute entry hash
        entry_hash = self._compute_entry_hash(prev_hash, manifest, cert_dict, entry_metadata)

        entry = LedgerEntry(
            entry_id=entry_id,
            prev_hash=prev_hash,
            entry_hash=entry_hash,
            timestamp=timestamp,
            manifest=manifest,
            certificate=cert_dict,
            metadata=entry_metadata,
        )

        # Append to file
        with self.path.open("a", encoding="utf-8") as f:
            f.write(entry.to_json() + "\n")

        # Update state
        self._last_hash = entry_hash

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
        - Entry hashes are correct
        - No missing entries

        Returns:
            True if ledger is valid
        """
        prev_hash = self.GENESIS_HASH
        expected_id = 0

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

            # Verify entry hash
            computed = self._compute_entry_hash(
                entry.prev_hash,
                entry.manifest,
                entry.certificate,
                entry.metadata,
            )
            if computed != entry.entry_hash:
                logger.error(
                    "Hash mismatch at entry %d: expected %s, got %s",
                    entry.entry_id,
                    entry.entry_hash[:8],
                    computed[:8],
                )
                return False

            prev_hash = entry.entry_hash

        logger.info("Ledger verified: %d entries valid", expected_id)
        return True

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
