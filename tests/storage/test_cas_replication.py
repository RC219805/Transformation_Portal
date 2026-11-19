"""Replication must never weaken the local CAS: atomic, verified, fail closed.

The peer transport is replaced by scripted responses, so nothing here opens a
socket. Payloads are small byte strings under a ``tmp_path`` CAS root.
"""

from __future__ import annotations

import base64
import errno
import hashlib
import os
import zlib
from pathlib import Path
from typing import Any

import pytest

from transformation_portal.storage.cas_replication import CASPeer, CASReplicator, ReplicationError
from transformation_portal.storage.cas_store import ArtifactStore, CASError

pytestmark = pytest.mark.unit

_PEER = CASPeer("peer.invalid")


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _item(payload: bytes, *, compressed: bool = False, sha: str | None = None) -> dict[str, Any]:
    body = zlib.compress(payload) if compressed else payload
    return {
        "sha": sha or _sha(payload),
        "size": len(payload),
        "compressed": compressed,
        "data_b64": base64.b64encode(body).decode("ascii"),
    }


def _serve(monkeypatch: pytest.MonkeyPatch, replicator: CASReplicator, *responses: dict[str, Any]) -> list[dict[str, Any]]:
    """Answer the replicator's requests in order and record what it asked."""
    requests: list[dict[str, Any]] = []
    scripted = list(responses)

    def send(_peer: CASPeer, request: dict[str, Any]) -> dict[str, Any]:
        requests.append(request)
        return scripted.pop(0)

    monkeypatch.setattr(replicator, "_send_request", send)
    return requests


def _fill_the_disk_half_way_through(patch: pytest.MonkeyPatch, payload: bytes) -> None:
    """Fail with ENOSPC after half of ``payload``, whichever primitive writes it."""
    real_write = os.write

    def torn_os_write(fd: int, data: bytes) -> int:
        if bytes(data) == payload:
            real_write(fd, payload[: len(payload) // 2])
            raise OSError(errno.ENOSPC, "No space left on device")
        return real_write(fd, data)

    def torn_write_bytes(self: Path, data: bytes) -> int:
        with open(self, "wb") as handle:
            handle.write(data[: len(data) // 2])
        raise OSError(errno.ENOSPC, "No space left on device")

    patch.setattr(os, "write", torn_os_write)
    patch.setattr(Path, "write_bytes", torn_write_bytes)


# --------------------------------------------------------------------------- #
# pull_many
# --------------------------------------------------------------------------- #


def test_pull_stores_verified_objects_and_counts_them(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    local = ArtifactStore(tmp_path / "local")
    plain, packed = b"plain-object" * 64, b"compressible-object" * 64
    replicator = CASReplicator(local)
    requests = _serve(monkeypatch, replicator, {"items": [_item(plain), _item(packed, compressed=True)]})

    assert replicator.pull_many(_PEER, [_sha(plain), _sha(packed)]) == 2

    assert requests == [{"op": "get_many", "shas": [_sha(plain), _sha(packed)]}]
    assert local.verify_object(_sha(plain)) and local.verify_object(_sha(packed))
    assert local.get_object(_sha(packed)).path.read_bytes() == packed
    assert replicator.get_stats()["objects_pulled"] == 2
    assert replicator.get_stats()["bytes_transferred"] == len(plain) + len(packed)


def test_interrupted_pull_cannot_leave_a_torn_object_at_its_content_address(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    local = ArtifactStore(tmp_path / "local")
    payload = b"model-weights" * 4096
    sha = _sha(payload)
    replicator = CASReplicator(local, compress=False)
    _serve(monkeypatch, replicator, {"items": [_item(payload)]}, {"items": [_item(payload)]})

    with monkeypatch.context() as full_disk:
        _fill_the_disk_half_way_through(full_disk, payload)
        with pytest.raises((OSError, CASError)):
            replicator.pull_many(_PEER, [sha])

    # A reader must never find half an object where the whole one belongs.
    assert not local.has_object(sha)
    assert [path.name for path in local.objects_dir.rglob("*") if path.is_file()] == []

    assert replicator.pull_many(_PEER, [sha]) == 1
    assert local.verify_object(sha)


def test_pull_repairs_a_corrupt_object_instead_of_skipping_it(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    local = ArtifactStore(tmp_path / "local")
    payload = b"model-weights" * 4096
    sha = _sha(payload)
    damaged = local._object_path(sha)  # noqa: SLF001 - plant what an earlier torn write left behind
    damaged.parent.mkdir(parents=True)
    damaged.write_bytes(payload[: len(payload) // 2])
    replicator = CASReplicator(local)
    requests = _serve(monkeypatch, replicator, {"items": [_item(payload)]})

    # ``skip_existing`` means "skip what I hold", and half an object is not held.
    assert replicator.pull_many(_PEER, [sha]) == 1

    assert [request["op"] for request in requests] == ["get_many"]
    assert local.verify_object(sha)
    assert damaged.read_bytes() == payload


def test_pull_skips_an_intact_object_without_asking_the_peer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    local = ArtifactStore(tmp_path / "local")
    held = local.add_bytes(b"already-held")
    replicator = CASReplicator(local)
    requests = _serve(monkeypatch, replicator)

    assert replicator.pull_many(_PEER, [held.sha256]) == 0
    assert requests == []


def test_pull_rejects_bytes_that_do_not_match_their_address(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    local = ArtifactStore(tmp_path / "local")
    wanted = _sha(b"wanted")
    replicator = CASReplicator(local)
    _serve(monkeypatch, replicator, {"items": [_item(b"something else", sha=wanted)]})

    assert replicator.pull_many(_PEER, [wanted]) == 0
    assert not local.has_object(wanted)
    assert not local.has_object(_sha(b"something else"))


# --------------------------------------------------------------------------- #
# has_many / push_many
# --------------------------------------------------------------------------- #


def _three_local_objects(tmp_path: Path) -> tuple[ArtifactStore, list[str]]:
    source = ArtifactStore(tmp_path / "source")
    return source, [source.add_bytes(b"object-%d" % index).sha256 for index in range(3)]


@pytest.mark.parametrize(
    "answer",
    [
        {},
        {"present": None},
        {"present": []},
        {"present": [False, False]},
        {"present": [False, False, False, False]},
        {"present": "false"},
        {"present": ["false", "false", "false"]},
        {"present": [0, 0, 0]},
    ],
    ids=["missing", "null", "empty", "short", "long", "not-a-list", "strings", "integers"],
)
def test_push_fails_closed_when_the_peer_does_not_answer_for_every_object(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, answer: dict[str, Any]
) -> None:
    source, shas = _three_local_objects(tmp_path)
    replicator = CASReplicator(source)
    requests = _serve(monkeypatch, replicator, answer, {"ok": True})

    # Reading an incomplete answer as "the peer has everything" would report
    # success while replicating nothing.
    with pytest.raises(ReplicationError, match="Malformed has_many response"):
        replicator.push_many(_PEER, shas)

    assert [request["op"] for request in requests] == ["has_many"]
    assert replicator.get_stats()["objects_pushed"] == 0


def test_push_sends_exactly_the_objects_the_peer_lacks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source, shas = _three_local_objects(tmp_path)
    replicator = CASReplicator(source, compress=False)
    requests = _serve(monkeypatch, replicator, {"present": [True, False, True]}, {"ok": True})

    assert replicator.push_many(_PEER, shas) == 1

    assert requests[0] == {"op": "has_many", "shas": shas}
    assert requests[1]["op"] == "put_many"
    assert [item["sha"] for item in requests[1]["items"]] == [shas[1]]
    assert base64.b64decode(requests[1]["items"][0]["data_b64"]) == b"object-1"


def test_has_many_returns_one_flag_per_requested_object(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source, shas = _three_local_objects(tmp_path)
    replicator = CASReplicator(source)
    _serve(monkeypatch, replicator, {"present": [True, False, True]}, {"error": "peer is read-only"})

    assert replicator.has_many(_PEER, shas) == [True, False, True]
    with pytest.raises(ReplicationError, match="peer is read-only"):
        replicator.has_many(_PEER, shas)
