"""Phase 4.A - shared contract tests for every ``ArtifactStore`` backend.

Local backend is always included. The S3 backend also runs by default
against a ``moto`` mock so the contract is exercised end-to-end without
a live AWS / MinIO endpoint; setting ``TP_TEST_S3_URL`` +
``TP_TEST_S3_BUCKET`` switches the same suite to a real S3-compatible
endpoint, mirroring the Phase 1.B Postgres and Phase 2.B Redis
patterns. The fixture builds a per-test ``prefix`` so parallel runs
(pytest-xdist) and shared-tenant buckets coexist without colliding on
keys.

Coverage:

- ``write_bytes`` round-trip + ``head`` returns the same fingerprint.
- ``open_bytes`` streams the body in chunks; reassembly equals the
  written bytes.
- ``list_for_job`` returns sorted entries with metadata.
- Path-traversal validation rejects ``..``, absolute paths, empty
  job_ids — preserving the pre-Phase-4 guarantees.
- ``delete`` (single + entire-job) and the resulting absence.
- Fingerprint contract: ``ok`` for small files,
  ``skipped_size`` for files above the bounded limit.
- The Phase 4 Merkle helper (``compute_artifact_merkle_root``)
  produces the same root for both backends given the same artifacts.
"""

from __future__ import annotations

import asyncio
import os
import threading
import time
import uuid
from typing import AsyncIterator

import pytest
import pytest_asyncio

from transformation_portal.lux_depth_v3.artifact_manager import compute_artifact_merkle_root
from transformation_portal.orchestrator.artifact_store import (
    ArtifactNotFoundError,
    ArtifactPathValidationError,
    ArtifactStore,
    ArtifactStoreError,
)
from transformation_portal.orchestrator.artifact_store.local import LocalArtifactStore

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]

_S3_URL_ENV = "TP_TEST_S3_URL"
_S3_BUCKET_ENV = "TP_TEST_S3_BUCKET"
_BAD_JOB_IDS = [
    "../escape",
    "/absolute",
    "nested/job",
    "nested\\job",
    "\x00bad",
    ".",
    "..",
]


def _have_moto() -> bool:
    try:
        import boto3  # noqa: F401
        import moto  # noqa: F401
    except ImportError:
        return False
    return True


def _have_live_s3() -> bool:
    return bool(os.getenv(_S3_URL_ENV, "").strip() and os.getenv(_S3_BUCKET_ENV, "").strip())


def _available_backends() -> list[str]:
    backends = ["local"]
    if _have_live_s3() or _have_moto():
        backends.append("s3")
    return backends


@pytest.fixture(params=_available_backends())
def backend(request: pytest.FixtureRequest) -> str:
    return request.param


@pytest_asyncio.fixture
async def store(backend: str, tmp_path) -> AsyncIterator[ArtifactStore]:
    """Yield a freshly-reset ``ArtifactStore`` for the parameterized backend.

    The S3 branch prefers a live endpoint when ``TP_TEST_S3_URL`` is set;
    otherwise it spins up a ``moto`` mock so the contract is exercised in
    every CI lane. Per-test ``key_prefix`` isolates parallel cases and
    shared-tenant buckets.
    """
    if backend == "local":
        instance: ArtifactStore = LocalArtifactStore(root_dir=tmp_path / "artifacts")
        try:
            yield instance
        finally:
            await instance.reset()
            await instance.close()
        return

    if backend != "s3":
        raise RuntimeError(f"unknown backend {backend!r}")

    try:
        from transformation_portal.orchestrator.artifact_store.s3 import S3ArtifactStore
    except ImportError:
        pytest.skip("S3 backend not importable (boto3 likely missing)")

    prefix = f"tp/test/{uuid.uuid4().hex[:12]}"
    if _have_live_s3():
        instance = S3ArtifactStore(
            bucket=os.environ[_S3_BUCKET_ENV],
            prefix=prefix,
            endpoint_url=os.environ[_S3_URL_ENV],
        )
        await instance.reset()
        try:
            yield instance
        finally:
            await instance.reset()
            await instance.close()
        return

    # Mocked S3 via moto. ``mock_aws`` patches boto3 globally inside the
    # context, so the store's lazy ``boto3.client("s3")`` returns the mock.
    try:
        import boto3
        from moto import mock_aws
    except ImportError:
        pytest.skip("moto not available; install dev extras to exercise the S3 branch")

    previous_env = {key: os.environ.get(key) for key in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_DEFAULT_REGION")}
    os.environ.update(
        AWS_ACCESS_KEY_ID="test",
        AWS_SECRET_ACCESS_KEY="test",
        AWS_DEFAULT_REGION="us-east-1",
    )
    with mock_aws():
        bucket = f"tp-test-{uuid.uuid4().hex[:8]}"
        boto3.client("s3", region_name="us-east-1").create_bucket(Bucket=bucket)
        instance = S3ArtifactStore(
            bucket=bucket,
            prefix=prefix,
            region_name="us-east-1",
        )
        try:
            yield instance
        finally:
            await instance.reset()
            await instance.close()
    for key, value in previous_env.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


# ---------------------------------------------------------------------------
# write_bytes / head / open_bytes
# ---------------------------------------------------------------------------


async def test_write_then_head_returns_matching_metadata(store: ArtifactStore) -> None:
    meta = await store.write_bytes("job-a", "outputs/result.json", b'{"ok": true}')
    assert meta.relative_path == "outputs/result.json"
    assert meta.size_bytes == 12
    assert meta.content_type == "application/json"
    assert meta.fingerprint_status == "ok"
    assert meta.sha256_hex is not None and len(meta.sha256_hex) == 64

    head = await store.head("job-a", "outputs/result.json")
    assert head.relative_path == meta.relative_path
    assert head.size_bytes == meta.size_bytes
    assert head.content_type == meta.content_type
    assert head.sha256_hex == meta.sha256_hex
    assert head.fingerprint_status == "ok"


async def test_explicit_content_type_round_trips_through_head_and_list(store: ArtifactStore) -> None:
    content_type = "application/vnd.transformation-portal.test+json"
    meta = await store.write_bytes(
        "job-content-type",
        "outputs/payload.bin",
        b'{"ok": true}',
        content_type=content_type,
    )
    assert meta.content_type == content_type

    head = await store.head("job-content-type", "outputs/payload.bin")
    assert head.content_type == content_type

    listed = await store.list_for_job("job-content-type")
    assert [item.content_type for item in listed] == [content_type]


async def test_open_bytes_round_trip(store: ArtifactStore) -> None:
    body = b"deterministic-bytes-" * 200
    await store.write_bytes("job-b", "outputs/log.txt", body)
    stream = await store.open_bytes("job-b", "outputs/log.txt")
    chunks: list[bytes] = []
    async for chunk in stream:
        chunks.append(chunk)
    assert b"".join(chunks) == body


async def test_open_bytes_raises_on_missing(store: ArtifactStore) -> None:
    with pytest.raises(ArtifactNotFoundError):
        gen = await store.open_bytes("job-missing", "nope.txt")
        async for _chunk in gen:
            pass


async def test_head_raises_on_missing(store: ArtifactStore) -> None:
    with pytest.raises(ArtifactNotFoundError):
        await store.head("job-missing", "nope.txt")


# ---------------------------------------------------------------------------
# Path-traversal validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_path",
    [
        "../escape.txt",
        "outputs/../../etc/passwd",
        "/absolute/path",
        "",
        "\x00null-byte",
        "back\\slash.txt",
    ],
)
async def test_path_traversal_rejected(store: ArtifactStore, bad_path: str) -> None:
    with pytest.raises(ArtifactPathValidationError):
        await store.write_bytes("job-trav", bad_path, b"x")


async def test_empty_job_id_rejected(store: ArtifactStore) -> None:
    with pytest.raises(ArtifactPathValidationError):
        await store.write_bytes("", "outputs/x.txt", b"x")


@pytest.mark.parametrize("bad_job_id", _BAD_JOB_IDS)
@pytest.mark.parametrize(
    "operation",
    [
        "write_bytes",
        "head",
        "open_bytes",
        "list_for_job",
        "delete_single",
        "delete_job",
    ],
)
async def test_bad_job_ids_rejected_for_every_public_method(
    store: ArtifactStore,
    bad_job_id: str,
    operation: str,
) -> None:
    with pytest.raises(ArtifactPathValidationError):
        if operation == "write_bytes":
            await store.write_bytes(bad_job_id, "outputs/x.txt", b"x")
        elif operation == "head":
            await store.head(bad_job_id, "outputs/x.txt")
        elif operation == "open_bytes":
            stream = await store.open_bytes(bad_job_id, "outputs/x.txt")
            async for _chunk in stream:
                pass
        elif operation == "list_for_job":
            await store.list_for_job(bad_job_id)
        elif operation == "delete_single":
            await store.delete(bad_job_id, "outputs/x.txt")
        elif operation == "delete_job":
            await store.delete(bad_job_id)
        else:  # pragma: no cover - parametrization guard
            raise AssertionError(f"unexpected operation {operation!r}")


async def test_local_symlinked_job_directory_escape_rejected(tmp_path) -> None:
    root = tmp_path / "artifacts"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    try:
        os.symlink(outside, root / "job-link")
    except (AttributeError, NotImplementedError, OSError) as exc:
        pytest.skip(f"symlink unavailable on this platform: {exc}")

    store = LocalArtifactStore(root_dir=root)
    with pytest.raises(ArtifactPathValidationError):
        await store.write_bytes("job-link", "outputs/x.txt", b"x")


async def test_local_list_and_bulk_delete_skip_symlink_escape(tmp_path) -> None:
    root = tmp_path / "artifacts"
    outside = tmp_path / "outside"
    job_root = root / "job-link-file"
    outside.mkdir()
    job_root.mkdir(parents=True)
    outside_file = outside / "secret.txt"
    outside_file.write_text("secret", encoding="utf-8")
    try:
        os.symlink(outside_file, job_root / "escape.txt")
    except (AttributeError, NotImplementedError, OSError) as exc:
        pytest.skip(f"symlink unavailable on this platform: {exc}")

    store = LocalArtifactStore(root_dir=root)
    assert await store.list_for_job("job-link-file") == []
    assert await store.delete("job-link-file") == 0
    assert outside_file.read_text(encoding="utf-8") == "secret"


# ---------------------------------------------------------------------------
# list_for_job
# ---------------------------------------------------------------------------


async def test_list_for_job_sorts_by_relative_path(store: ArtifactStore) -> None:
    await store.write_bytes("job-l", "z/last.txt", b"z")
    await store.write_bytes("job-l", "a/first.txt", b"a")
    await store.write_bytes("job-l", "m/mid.txt", b"m")

    listed = await store.list_for_job("job-l")
    assert [item.relative_path for item in listed] == [
        "a/first.txt",
        "m/mid.txt",
        "z/last.txt",
    ]


async def test_list_for_job_empty_when_no_writes(store: ArtifactStore) -> None:
    assert await store.list_for_job("ghost-job") == []


# ---------------------------------------------------------------------------
# delete
# ---------------------------------------------------------------------------


async def test_delete_single_artifact(store: ArtifactStore) -> None:
    await store.write_bytes("job-d", "a.txt", b"a")
    await store.write_bytes("job-d", "b.txt", b"b")
    deleted = await store.delete("job-d", "a.txt")
    assert deleted == 1
    listed = await store.list_for_job("job-d")
    assert [item.relative_path for item in listed] == ["b.txt"]


async def test_delete_single_missing_returns_zero(store: ArtifactStore) -> None:
    assert await store.delete("job-d", "missing.txt") == 0


async def test_delete_entire_job(store: ArtifactStore) -> None:
    await store.write_bytes("job-bulk", "a.txt", b"a")
    await store.write_bytes("job-bulk", "subdir/b.txt", b"b")
    await store.write_bytes("job-bulk", "subdir/nested/c.txt", b"c")

    deleted = await store.delete("job-bulk")
    assert deleted == 3
    assert await store.list_for_job("job-bulk") == []


async def test_s3_bulk_delete_raises_on_per_key_errors() -> None:
    from transformation_portal.orchestrator.artifact_store.s3 import S3ArtifactStore

    class ErroringBulkDeleteClient:
        def list_objects_v2(self, **_kwargs):
            return {
                "Contents": [{"Key": "tp/test/jobs/job-bulk/a.txt"}],
                "IsTruncated": False,
            }

        def delete_objects(self, **_kwargs):
            return {
                "Errors": [{"Key": "tp/test/jobs/job-bulk/a.txt", "Code": "AccessDenied"}],
                "Deleted": [],
            }

    store = S3ArtifactStore(bucket="bucket", prefix="tp/test", client=ErroringBulkDeleteClient())
    with pytest.raises(ArtifactStoreError, match="S3 delete_objects failed"):
        await store.delete("job-bulk")


async def test_s3_bulk_delete_counts_confirmed_deletions() -> None:
    from transformation_portal.orchestrator.artifact_store.s3 import S3ArtifactStore

    class PartialBulkDeleteClient:
        def list_objects_v2(self, **_kwargs):
            return {
                "Contents": [
                    {"Key": "tp/test/jobs/job-bulk/a.txt"},
                    {"Key": "tp/test/jobs/job-bulk/b.txt"},
                ],
                "IsTruncated": False,
            }

        def delete_objects(self, **_kwargs):
            return {
                "Errors": [],
                "Deleted": [{"Key": "tp/test/jobs/job-bulk/a.txt"}],
            }

    store = S3ArtifactStore(bucket="bucket", prefix="tp/test", client=PartialBulkDeleteClient())
    assert await store.delete("job-bulk") == 1


async def test_s3_list_for_job_skips_invalid_external_keys() -> None:
    from transformation_portal.orchestrator.artifact_store.s3 import S3ArtifactStore

    class InvalidKeyListClient:
        def __init__(self) -> None:
            self.head_keys: list[str] = []

        def list_objects_v2(self, **_kwargs):
            return {
                "Contents": [
                    {"Key": "tp/test/jobs/job-external/valid.txt"},
                    {"Key": "tp/test/jobs/job-external/../escape.txt"},
                    {"Key": "tp/test/jobs/job-external/back\\slash.txt"},
                    {"Key": "tp/test/jobs/job-external/sub/../../escape.txt"},
                ],
                "IsTruncated": False,
            }

        def head_object(self, **kwargs):
            key = kwargs["Key"]
            self.head_keys.append(key)
            if key != "tp/test/jobs/job-external/valid.txt":
                raise AssertionError(f"invalid key should not be headed: {key}")
            return {
                "ContentLength": 999_999_999,
                "ContentType": "text/plain",
            }

    client = InvalidKeyListClient()
    store = S3ArtifactStore(bucket="bucket", prefix="tp/test", client=client)

    listed = await store.list_for_job("job-external")

    assert [item.relative_path for item in listed] == ["valid.txt"]
    assert client.head_keys == ["tp/test/jobs/job-external/valid.txt"]


async def test_local_content_type_sidecar_mutations_are_serialized(tmp_path, monkeypatch) -> None:
    store = LocalArtifactStore(root_dir=tmp_path / "artifacts")
    original_load = store._load_content_types  # noqa: SLF001 - targeted concurrency regression
    active_lock = threading.Lock()
    active_loads = 0
    max_active_loads = 0

    def slow_load(normalized_job_id: str) -> dict[str, str]:
        nonlocal active_loads, max_active_loads
        with active_lock:
            active_loads += 1
            max_active_loads = max(max_active_loads, active_loads)
        try:
            time.sleep(0.05)
            return original_load(normalized_job_id)
        finally:
            with active_lock:
                active_loads -= 1

    monkeypatch.setattr(store, "_load_content_types", slow_load)

    await asyncio.gather(
        *(
            store.write_bytes(
                "job-sidecar-race",
                f"outputs/{index}.bin",
                b"x",
                content_type=f"application/vnd.tp.race-{index}",
            )
            for index in range(4)
        )
    )

    assert max_active_loads == 1
    listed = await store.list_for_job("job-sidecar-race")
    assert [item.content_type for item in listed] == [
        "application/vnd.tp.race-0",
        "application/vnd.tp.race-1",
        "application/vnd.tp.race-2",
        "application/vnd.tp.race-3",
    ]


# ---------------------------------------------------------------------------
# Merkle parity with the Phase 4 helper
# ---------------------------------------------------------------------------


async def test_merkle_root_matches_phase4_helper(store: ArtifactStore) -> None:
    """``compute_artifact_merkle_root`` works directly on the store's
    metadata so backends produce a deterministic Merkle root from the
    same input bytes."""
    await store.write_bytes("job-merkle", "a.txt", b"alpha")
    await store.write_bytes("job-merkle", "b.txt", b"beta")
    await store.write_bytes("job-merkle", "c.txt", b"gamma")

    listed = await store.list_for_job("job-merkle")
    artifact_index = [
        {"relative_path": meta.relative_path, "sha256": meta.sha256_hex} for meta in listed if meta.sha256_hex is not None
    ]
    assert len(artifact_index) == 3
    root = compute_artifact_merkle_root(artifact_index)
    assert isinstance(root, str)
    assert len(root) == 64
    # Deterministic across backends because the helper sorts by
    # ``relative_path`` before concatenating sha256 bytes.
    # Re-list-and-recompute must produce the same root.
    again = await store.list_for_job("job-merkle")
    again_index = [
        {"relative_path": meta.relative_path, "sha256": meta.sha256_hex} for meta in again if meta.sha256_hex is not None
    ]
    assert compute_artifact_merkle_root(again_index) == root


# ---------------------------------------------------------------------------
# Fingerprint contract: skipped_size for large files
# ---------------------------------------------------------------------------


async def test_fingerprint_skipped_when_oversized(store: ArtifactStore, monkeypatch) -> None:
    """Files larger than ``ARTIFACT_FINGERPRINT_MAX_BYTES`` must report
    ``fingerprint_status='skipped_size'`` without reading the bytes."""
    from transformation_portal.portal import job_artifacts as job_artifacts_module

    # Lower the cap so the test body stays small and fast.
    monkeypatch.setattr(job_artifacts_module, "ARTIFACT_FINGERPRINT_MAX_BYTES", 16)
    # Both backends import the cap directly; patch both shims so the
    # patched value is observed regardless of which backend is parametrized.
    from transformation_portal.orchestrator.artifact_store import local as local_module
    from transformation_portal.orchestrator.artifact_store import s3 as s3_module

    monkeypatch.setattr(local_module, "ARTIFACT_FINGERPRINT_MAX_BYTES", 16)
    monkeypatch.setattr(s3_module, "ARTIFACT_FINGERPRINT_MAX_BYTES", 16)

    body = b"x" * 32  # twice the (patched) cap
    meta = await store.write_bytes("job-big", "big.bin", body)
    assert meta.size_bytes == 32
    assert meta.fingerprint_status == "skipped_size"
    assert meta.sha256_hex is None

    head = await store.head("job-big", "big.bin")
    assert head.fingerprint_status == "skipped_size"
    assert head.sha256_hex is None
