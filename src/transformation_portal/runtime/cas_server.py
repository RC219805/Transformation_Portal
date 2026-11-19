"""CAS Replication Server for serving artifacts to remote peers.

This module provides a TCP server that handles CAS replication
requests from remote peers.

Protocol:
    Request: { "op": "has_many"|"get_many"|"put_many", ... }
    Response: { "present": [...] | "items": [...] | "ok": true }

Example:
    >>> cas = ArtifactStore(Path("/data/cas"))
    >>> start_cas_server(cas, host="0.0.0.0", port=6000)
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import socketserver
import zlib
from pathlib import Path
from typing import Any, Dict, Optional

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


class CASHandler(socketserver.BaseRequestHandler):
    """Handler for CAS replication requests.

    Handles operations:
    - has_many: Check which objects exist
    - get_many: Retrieve objects
    - put_many: Store objects
    """

    # Injected by server
    cas: Optional[ArtifactStore] = None
    compress: bool = True
    buffer_size: int = 10 * 1024 * 1024

    def _recv(self) -> Dict[str, Any]:
        """Receive and parse request."""
        chunks = []
        while True:
            chunk = self.request.recv(self.buffer_size)
            if not chunk:
                break
            chunks.append(chunk)
            # Check for complete JSON
            try:
                data = b"".join(chunks).decode("utf-8")
                return json.loads(data)
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue

        data = b"".join(chunks).decode("utf-8")
        return json.loads(data)

    def _send(self, payload: Dict[str, Any]) -> None:
        """Send response."""
        response = json.dumps(payload).encode("utf-8")
        self.request.sendall(response)

    def handle(self) -> None:
        """Handle incoming request."""
        try:
            req = self._recv()
            op = req.get("op")

            if op == "has_many":
                self._handle_has_many(req)
            elif op == "get_many":
                self._handle_get_many(req)
            elif op == "put_many":
                self._handle_put_many(req)
            else:
                self._send({"error": f"Unknown operation: {op}"})

        except Exception as e:
            logger.error("CAS handler error: %s", e)
            self._send({"error": str(e)})

    def _has_valid_object(self, sha: str) -> bool:
        """Return True only if the object exists and its content matches ``sha``.

        A torn or corrupt object must not be reported as present, otherwise
        peers never push the good copy and the corruption is permanent.
        """
        if not self.cas._object_path(sha).exists():
            return False
        try:
            return self.cas.verify_object(sha)
        except (CASError, OSError):
            return False

    def _handle_has_many(self, req: Dict[str, Any]) -> None:
        """Handle has_many operation."""
        shas = req.get("shas", [])
        present = []

        for sha in shas:
            try:
                validate_sha256(sha)
                present.append(self._has_valid_object(sha))
            except PathSafetyError:
                present.append(False)

        logger.debug("has_many: %d/%d present", sum(present), len(shas))
        self._send({"present": present})

    def _handle_get_many(self, req: Dict[str, Any]) -> None:
        """Handle get_many operation."""
        shas = req.get("shas", [])
        items = []

        for sha in shas:
            try:
                validate_sha256(sha)
                path = self.cas._object_path(sha)

                if not path.exists():
                    continue

                # Serve only bytes whose digest was verified; a corrupt object
                # is treated as missing.
                try:
                    with self.cas.open_verified(sha) as handle:
                        data = handle.read()
                except CASError as exc:
                    logger.warning("Not serving unverifiable CAS object %s: %s", sha[:8], exc)
                    continue

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

            except PathSafetyError:
                logger.warning("Invalid SHA in get_many: %s", sha)
            except Exception as e:
                logger.error("Error reading %s: %s", sha[:8], e)

        logger.debug("get_many: returning %d items", len(items))
        self._send({"items": items})

    def _handle_put_many(self, req: Dict[str, Any]) -> None:
        """Handle put_many operation."""
        items = req.get("items", [])
        stored = 0

        for item in items:
            sha = item.get("sha")
            try:
                validate_sha256(sha)

                # Skip only objects that exist AND verify; a torn or corrupt
                # object is repaired by the push below.
                if self._has_valid_object(sha):
                    continue

                data = _b64_decode(item["data_b64"])

                # Decompress if needed
                if item.get("compressed", False):
                    data = zlib.decompress(data)

                # Verify hash
                actual = hashlib.sha256(data).hexdigest()
                if actual != sha:
                    logger.warning("SHA mismatch in put_many: %s vs %s", sha[:8], actual[:8])
                    continue

                # Store through the CAS: temp file + fsync + digest check +
                # os.replace, so a failed write never publishes a partial object
                # and an existing corrupt object is replaced.
                self.cas.add_bytes(data, verify=True)
                stored += 1

            except PathSafetyError:
                logger.warning("Invalid SHA in put_many: %s", sha)
            except Exception as e:
                logger.error("Error storing %s: %s", sha[:8] if sha else "?", e)

        logger.debug("put_many: stored %d items", stored)
        self._send({"ok": True, "stored": stored})


class ThreadedCASServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    """Threaded TCP server for CAS replication."""

    allow_reuse_address = True


def start_cas_server(
    cas: ArtifactStore,
    *,
    host: str = "0.0.0.0",  # nosec B104 - CAS server intentionally binds to all interfaces
    port: int = 6000,
    compress: bool = True,
    threaded: bool = True,
) -> None:
    """Start the CAS replication server.

    Args:
        cas: Local CAS store to serve
        host: Bind address
        port: Bind port
        compress: Whether to compress transfers
        threaded: Whether to handle requests in threads
    """
    CASHandler.cas = cas
    CASHandler.compress = compress

    server_class = ThreadedCASServer if threaded else socketserver.TCPServer

    with server_class((host, port), CASHandler) as server:
        logger.info("CAS server listening on %s:%d", host, port)
        print(f"[CAS] Serving on {host}:{port}")
        server.serve_forever()


def _run_cas_server(cas: ArtifactStore, host: str, port: int) -> None:
    """Process target for :class:`CASServerProcess` (module-level so it pickles under spawn)."""
    start_cas_server(cas, host=host, port=port)


class CASServerProcess:
    """Wrapper to run CAS server in a subprocess.

    Example:
        >>> server = CASServerProcess(cas, port=6000)
        >>> server.start()
        >>> # ... do work ...
        >>> server.stop()
    """

    def __init__(
        self,
        cas: ArtifactStore,
        *,
        host: str = "0.0.0.0",  # nosec B104 - CAS server intentionally binds to all interfaces
        port: int = 6000,
        start_method: str = "spawn",
    ) -> None:
        """Initialize server wrapper.

        Args:
            cas: CAS store to serve
            host: Bind address
            port: Bind port
            start_method: Multiprocessing start method (spawn, fork, forkserver)
        """
        self.cas = cas
        self.host = host
        self.port = port
        self.start_method = start_method
        self._process = None

    def start(self) -> None:
        """Start server in background process."""
        import multiprocessing

        ctx = multiprocessing.get_context(self.start_method)
        self._process = ctx.Process(
            target=_run_cas_server,
            args=(self.cas, self.host, self.port),
            daemon=True,
        )
        self._process.start()
        logger.info("Started CAS server process: pid=%d", self._process.pid)

    def stop(self) -> None:
        """Stop the server process."""
        if self._process and self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=5)
            logger.info("Stopped CAS server process")

    @property
    def is_running(self) -> bool:
        """Check if server is running."""
        return self._process is not None and self._process.is_alive()
