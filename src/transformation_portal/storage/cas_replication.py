"""Remote CAS Replication for distributed artifact storage.

This module provides CAS replication capabilities:
- Push/pull artifacts by SHA256
- Batch transfers to minimize round-trips
- Resume-safe (idempotent: skip if exists)
- Optional compression

Example:
    >>> replicator = CASReplicator(local_cas)
    >>> peer = CASPeer(host="worker-1", port=6000)
    >>>
    >>> # Push local objects to remote
    >>> replicator.push_many(peer, ["abc123...", "def456..."])
    >>>
    >>> # Pull missing objects from remote
    >>> replicator.pull_many(peer, ["xyz789..."])
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import socket
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from transformation_portal.core.security.path_safety import (
    PathSafetyError,
    validate_sha256,
)
from transformation_portal.storage.cas_store import ArtifactStore, CASError

logger = logging.getLogger(__name__)


def _b64_encode(b: bytes) -> str:
    """Base64 encode bytes."""
    return base64.b64encode(b).decode("ascii")


def _b64_decode(s: str) -> bytes:
    """Base64 decode string."""
    return base64.b64decode(s.encode("ascii"))


class ReplicationError(RuntimeError):
    """Raised for replication errors."""


@dataclass(frozen=True)
class CASPeer:
    """Remote CAS peer endpoint.

    Attributes:
        host: Peer hostname or IP
        port: Peer port (default: 6000)
        name: Optional human-readable name
    """

    host: str
    port: int = 6000
    name: Optional[str] = None

    @property
    def address(self) -> str:
        """Get address string."""
        return f"{self.host}:{self.port}"


@dataclass
class ReplicationStats:
    """Statistics for replication operations.

    Attributes:
        objects_pushed: Number of objects pushed
        objects_pulled: Number of objects pulled
        bytes_transferred: Total bytes transferred
        errors: Number of errors
    """

    objects_pushed: int = 0
    objects_pulled: int = 0
    bytes_transferred: int = 0
    errors: int = 0


class CASReplicator:
    """Client-side CAS replication logic.

    Provides methods to replicate CAS objects between local
    and remote stores.

    Protocol:
        Request: { "op": "has_many"|"get_many"|"put_many", ... }
        Response: { "present": [...] | "items": [...] | "ok": true }

    Example:
        >>> replicator = CASReplicator(local_cas)
        >>> peer = CASPeer("worker-1", 6000)
        >>>
        >>> # Check what peer has
        >>> present = replicator.has_many(peer, ["sha1", "sha2"])
        >>>
        >>> # Push missing to peer
        >>> replicator.push_many(peer, ["sha1", "sha2"])
    """

    def __init__(
        self,
        local_cas: ArtifactStore,
        *,
        timeout: float = 30.0,
        buffer_size: int = 10 * 1024 * 1024,
        compress: bool = True,
    ) -> None:
        """Initialize replicator.

        Args:
            local_cas: Local CAS store
            timeout: Socket timeout in seconds
            buffer_size: Receive buffer size
            compress: Whether to compress transfers
        """
        self.cas = local_cas
        self.timeout = timeout
        self.buffer_size = buffer_size
        self.compress = compress
        self.stats = ReplicationStats()

    def _send_request(self, peer: CASPeer, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Send request to peer and get response.

        Args:
            peer: CAS peer to contact
            payload: Request payload

        Returns:
            Response payload

        Raises:
            ReplicationError: If communication fails
        """
        try:
            with socket.create_connection((peer.host, peer.port), timeout=self.timeout) as sock:
                # Send request
                request_data = json.dumps(payload).encode("utf-8")
                sock.sendall(request_data)
                sock.shutdown(socket.SHUT_WR)

                # Receive response
                chunks = []
                while True:
                    chunk = sock.recv(self.buffer_size)
                    if not chunk:
                        break
                    chunks.append(chunk)

                response_data = b"".join(chunks).decode("utf-8")
                return json.loads(response_data)

        except socket.timeout:
            self.stats.errors += 1
            raise ReplicationError(f"Timeout connecting to {peer.address}")

        except ConnectionRefusedError:
            self.stats.errors += 1
            raise ReplicationError(f"Connection refused: {peer.address}")

        except Exception as e:
            self.stats.errors += 1
            raise ReplicationError(f"Replication error: {e}")

    def has_many(self, peer: CASPeer, shas: List[str]) -> List[bool]:
        """Check which objects peer has.

        Args:
            peer: CAS peer to query
            shas: List of SHA256 hashes to check

        Returns:
            List of booleans (True if present)
        """
        # Validate all SHAs first
        for sha in shas:
            validate_sha256(sha)

        response = self._send_request(peer, {"op": "has_many", "shas": shas})

        if "error" in response:
            raise ReplicationError(response["error"])

        return response.get("present", [])

    def push_many(
        self,
        peer: CASPeer,
        shas: List[str],
        *,
        skip_existing: bool = True,
    ) -> int:
        """Push local objects to peer.

        Args:
            peer: CAS peer to push to
            shas: SHA256 hashes to push
            skip_existing: If True, skip objects peer already has

        Returns:
            Number of objects pushed
        """
        # Validate and filter
        valid_shas = []
        for sha in shas:
            try:
                validate_sha256(sha)
                valid_shas.append(sha)
            except PathSafetyError:
                logger.warning("Invalid SHA, skipping: %s", sha)

        if not valid_shas:
            return 0

        # Check what peer needs
        need = valid_shas
        if skip_existing:
            present = self.has_many(peer, valid_shas)
            need = [sha for sha, has in zip(valid_shas, present) if not has]

        if not need:
            logger.debug("Peer %s already has all %d objects", peer.address, len(valid_shas))
            return 0

        # Build items to push
        items = []
        total_bytes = 0

        for sha in need:
            path = self.cas._object_path(sha)
            if not path.exists():
                logger.warning("Local object missing, skipping: %s", sha[:8])
                continue

            data = path.read_bytes()
            total_bytes += len(data)

            # Optionally compress
            if self.compress:
                compressed = zlib.compress(data)
                if len(compressed) < len(data):
                    items.append(
                        {
                            "sha": sha,
                            "size": len(data),
                            "compressed": True,
                            "data_b64": _b64_encode(compressed),
                        }
                    )
                    continue

            items.append(
                {
                    "sha": sha,
                    "size": len(data),
                    "compressed": False,
                    "data_b64": _b64_encode(data),
                }
            )

        if not items:
            return 0

        # Send to peer
        response = self._send_request(peer, {"op": "put_many", "items": items})

        if "error" in response:
            raise ReplicationError(response["error"])

        pushed = len(items)
        self.stats.objects_pushed += pushed
        self.stats.bytes_transferred += total_bytes

        logger.info(
            "Pushed %d objects (%d bytes) to %s",
            pushed,
            total_bytes,
            peer.address,
        )

        return pushed

    def pull_many(
        self,
        peer: CASPeer,
        shas: List[str],
        *,
        skip_existing: bool = True,
    ) -> int:
        """Pull objects from peer.

        Args:
            peer: CAS peer to pull from
            shas: SHA256 hashes to pull
            skip_existing: If True, skip objects we already have

        Returns:
            Number of objects pulled
        """
        # Validate and filter
        need = []
        for sha in shas:
            try:
                validate_sha256(sha)
                if skip_existing and self.cas._object_path(sha).exists():
                    continue
                need.append(sha)
            except PathSafetyError:
                logger.warning("Invalid SHA, skipping: %s", sha)

        if not need:
            logger.debug("Already have all %d objects", len(shas))
            return 0

        # Request from peer
        response = self._send_request(peer, {"op": "get_many", "shas": need})

        if "error" in response:
            raise ReplicationError(response["error"])

        # Store received items
        items = response.get("items", [])
        pulled = 0
        total_bytes = 0

        for item in items:
            sha = item["sha"]
            try:
                validate_sha256(sha)
            except PathSafetyError:
                logger.warning("Invalid SHA in response, skipping: %s", sha)
                continue

            data = _b64_decode(item["data_b64"])

            # Decompress if needed
            if item.get("compressed", False):
                data = zlib.decompress(data)

            # Verify hash
            actual_sha = hashlib.sha256(data).hexdigest()
            if actual_sha != sha:
                logger.warning("SHA mismatch, skipping: expected %s, got %s", sha[:8], actual_sha[:8])
                continue

            # Store in CAS
            path = self.cas._object_path(sha)
            path.parent.mkdir(parents=True, exist_ok=True)

            if not path.exists():
                path.write_bytes(data)
                pulled += 1
                total_bytes += len(data)

        self.stats.objects_pulled += pulled
        self.stats.bytes_transferred += total_bytes

        logger.info(
            "Pulled %d objects (%d bytes) from %s",
            pulled,
            total_bytes,
            peer.address,
        )

        return pulled

    def sync(
        self,
        peer: CASPeer,
        shas: List[str],
    ) -> Tuple[int, int]:
        """Bidirectional sync with peer.

        Ensures both sides have all specified objects.

        Args:
            peer: CAS peer to sync with
            shas: SHA256 hashes to sync

        Returns:
            Tuple of (pushed, pulled) counts
        """
        pushed = self.push_many(peer, shas)
        pulled = self.pull_many(peer, shas)
        return pushed, pulled

    def get_stats(self) -> Dict[str, Any]:
        """Get replication statistics."""
        return {
            "objects_pushed": self.stats.objects_pushed,
            "objects_pulled": self.stats.objects_pulled,
            "bytes_transferred": self.stats.bytes_transferred,
            "errors": self.stats.errors,
        }
