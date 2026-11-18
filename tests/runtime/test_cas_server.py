"""Tests for the CAS replication server (T-10)."""

from __future__ import annotations

import base64
import errno
import hashlib
import json
import os
import socket
import time
from pathlib import Path
from typing import Any, Dict
from unittest.mock import patch

import pytest

from transformation_portal.runtime.cas_server import CASHandler, CASServerProcess
from transformation_portal.storage.cas_store import ArtifactStore

pytestmark = pytest.mark.unit


class _FakeRequest:
    """Minimal socket stand-in: serves one request payload and records the reply."""

    def __init__(self, payload: Dict[str, Any]) -> None:
        self._pending = json.dumps(payload).encode("utf-8")
        self.sent = b""

    def recv(self, size: int) -> bytes:
        chunk, self._pending = self._pending[:size], self._pending[size:]
        return chunk

    def sendall(self, data: bytes) -> None:
        self.sent += data


@pytest.fixture
def cas(tmp_path: Path):
    store = ArtifactStore(tmp_path / "cas")
    with patch.object(CASHandler, "cas", store), patch.object(CASHandler, "compress", False):
        yield store


def _rpc(payload: Dict[str, Any]) -> Dict[str, Any]:
    request = _FakeRequest(payload)
    CASHandler(request, ("127.0.0.1", 0), None)
    return json.loads(request.sent)


def _item(data: bytes) -> Dict[str, Any]:
    return {"sha": hashlib.sha256(data).hexdigest(), "data_b64": base64.b64encode(data).decode(), "compressed": False}


def _torn_object(cas: ArtifactStore, blob: bytes) -> Path:
    """Simulate an earlier interrupted replication write left at the final path."""
    path = cas._object_path(hashlib.sha256(blob).hexdigest())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(blob[:1000])
    return path


def test_torn_object_is_not_present_not_served_and_repaired(cas: ArtifactStore) -> None:
    blob = os.urandom(300_000)
    sha = hashlib.sha256(blob).hexdigest()
    path = _torn_object(cas, blob)

    assert _rpc({"op": "has_many", "shas": [sha]}) == {"present": [False]}
    assert _rpc({"op": "get_many", "shas": [sha]}) == {"items": []}

    assert _rpc({"op": "put_many", "items": [_item(blob)]}) == {"ok": True, "stored": 1}
    assert path.read_bytes() == blob
    assert cas.verify_object(sha) is True
    assert _rpc({"op": "has_many", "shas": [sha]}) == {"present": [True]}
    served = _rpc({"op": "get_many", "shas": [sha]})["items"]
    assert [base64.b64decode(i["data_b64"]) for i in served] == [blob]
    # A second push of a valid object is a no-op.
    assert _rpc({"op": "put_many", "items": [_item(blob)]}) == {"ok": True, "stored": 0}


def test_failed_write_publishes_nothing(cas: ArtifactStore) -> None:
    blob = os.urandom(300_000)
    sha = hashlib.sha256(blob).hexdigest()
    real_write = os.write

    def enospc_after_1000_bytes(fd: int, data: bytes) -> int:
        real_write(fd, bytes(data[:1000]))
        raise OSError(errno.ENOSPC, "No space left on device")

    with patch("transformation_portal.storage.cas_store.os.write", side_effect=enospc_after_1000_bytes):
        reply = _rpc({"op": "put_many", "items": [_item(blob)]})

    assert reply == {"ok": True, "stored": 0}
    assert not cas._object_path(sha).exists()
    assert list(cas._object_path(sha).parent.iterdir()) == []  # temp file cleaned up
    assert _rpc({"op": "has_many", "shas": [sha]}) == {"present": [False]}


def test_put_many_rejects_mismatched_digest(cas: ArtifactStore) -> None:
    item = _item(b"payload")
    item["data_b64"] = base64.b64encode(b"other").decode()
    assert _rpc({"op": "put_many", "items": [item]}) == {"ok": True, "stored": 0}
    assert not cas._object_path(item["sha"]).exists()


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def test_server_process_starts_under_spawn(tmp_path: Path) -> None:
    cas = ArtifactStore(tmp_path / "cas")
    blob = b"replicated over the wire"
    cas.add_bytes(blob)
    sha = hashlib.sha256(blob).hexdigest()
    port = _free_port()

    server = CASServerProcess(cas, host="127.0.0.1", port=port, start_method="spawn")
    server.start()
    try:
        deadline = time.monotonic() + 60
        reply = None
        while reply is None:
            assert server.is_running, "CAS server process exited during startup"
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=5) as sock:
                    sock.sendall(json.dumps({"op": "has_many", "shas": [sha]}).encode())
                    sock.shutdown(socket.SHUT_WR)
                    reply = json.loads(sock.makefile("rb").read())
            except ConnectionRefusedError:
                assert time.monotonic() < deadline, "CAS server never accepted a connection"
                time.sleep(0.05)
        assert reply == {"present": [True]}
    finally:
        server.stop()
    assert not server.is_running
