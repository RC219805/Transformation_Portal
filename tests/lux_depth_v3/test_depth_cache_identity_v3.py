"""Verified identity-v3 depth-cache entry contracts."""

from __future__ import annotations

import hashlib
import io
import json
import os
import stat
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from tests.core.test_execution_identity_v3 import _materialized as _core_materialized
from tests.core.test_execution_plan import _refingerprint, _valid_payload, _with_backend_shape
from transformation_portal.core.execution_identity_v3 import (
    BackendRuntimeIdentity,
    ExecutionIdentityV3,
    MaterializedExecutionIdentityV3,
)
from transformation_portal.core.execution_plan import CanonicalExecutionPlan
from transformation_portal.ingest.canonical_json import canonicalize_json
from transformation_portal.lux_depth_v3 import depth_cache as depth_cache_module
from transformation_portal.lux_depth_v3.depth_cache import (
    DEPTH_CACHE_POINTER_SCHEMA,
    DEPTH_CACHE_SCHEMA,
    DepthCache,
)

pytestmark = pytest.mark.unit


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _plan(*, revision: str, config: str) -> CanonicalExecutionPlan:
    payload = _with_backend_shape(_valid_payload(), ["da3"])
    payload["config_fingerprint_sha256"] = config
    payload["resolved_model"]["revision"] = revision
    payload["backend_candidates"][0]["model_contracts"][0]["model"]["revision"] = revision
    payload["nodes"][1]["configuration"]["resolved_model_revision"] = revision
    return CanonicalExecutionPlan.from_payload(_refingerprint(payload))


def _identity(
    *,
    input_label: str = "input",
    revision: str = "a" * 40,
    weights: str = _sha("weights"),
    dependency: str = _sha("dependency"),
    config: str = _sha("config"),
) -> MaterializedExecutionIdentityV3:
    plan = _plan(revision=revision, config=config)
    seed = ExecutionIdentityV3.from_plan(
        plan,
        stage_node_id="lux.depth",
        candidate_id="da3",
        input_id="input-000001",
    )
    runtime = BackendRuntimeIdentity.from_seed(
        seed,
        materialized_weights_sha256=weights,
        dependency_lock_sha256=dependency,
        interpreter_identity_sha256=_sha("interpreter"),
        platform_identity_sha256=_sha("platform"),
        accelerator_identity_sha256=_sha("accelerator"),
        source_identity_sha256=_sha("source"),
    )
    return MaterializedExecutionIdentityV3.from_plan(
        plan,
        stage_node_id="lux.depth",
        candidate_id="da3",
        input_id="input-000001",
        executed_backend="da3",
        input_content_sha256=_sha(input_label),
        backend_runtime_identities=(runtime,),
        dependency_lock_sha256=runtime.dependency_lock_sha256,
        interpreter_identity_sha256=runtime.interpreter_identity_sha256,
        platform_identity_sha256=runtime.platform_identity_sha256,
        accelerator_identity_sha256=runtime.accelerator_identity_sha256,
        source_identity_sha256=runtime.source_identity_sha256,
    )


def _entry_paths(cache: DepthCache, identity: MaterializedExecutionIdentityV3):
    key = identity.cache_key(DEPTH_CACHE_SCHEMA)
    pointer_path = cache._entry_path(key)
    pointer = json.loads(pointer_path.read_bytes())
    return pointer_path, cache._object_path(pointer["npy_sha256"]), pointer


def _rewrite_pointer(pointer_path, pointer) -> None:
    pointer_path.write_bytes(canonicalize_json(pointer))


def test_store_publishes_closed_canonical_pointer_last_and_exact_npy_digest(tmp_path) -> None:
    cache = DepthCache(tmp_path)
    identity = _identity()
    depth = np.arange(24, dtype=np.float32).reshape(4, 6)

    assert cache.store(identity, depth)
    pointer_path, object_path, pointer = _entry_paths(cache, identity)

    assert pointer_path.read_bytes() == canonicalize_json(pointer)
    assert set(pointer) == {
        "schema",
        "cache_schema",
        "cache_key",
        "execution_identity_sha256",
        "config_fingerprint_sha256",
        "input_content_sha256",
        "model_constituents",
        "materialized_weights_sha256",
        "dependency_lock_sha256",
        "npy_sha256",
        "byte_length",
        "shape",
        "dtype",
    }
    assert pointer["schema"] == DEPTH_CACHE_POINTER_SCHEMA
    assert pointer["cache_schema"] == DEPTH_CACHE_SCHEMA
    assert pointer["cache_key"] == identity.cache_key(DEPTH_CACHE_SCHEMA)
    assert pointer["execution_identity_sha256"] == identity.execution_identity_sha256
    assert pointer["npy_sha256"] == hashlib.sha256(object_path.read_bytes()).hexdigest()
    assert pointer["byte_length"] == object_path.stat().st_size
    assert pointer["shape"] == [4, 6]
    assert pointer["dtype"] == depth.dtype.str
    assert not any("path" in key for key in pointer)


def test_same_complete_identity_is_a_verified_hit(tmp_path) -> None:
    cache = DepthCache(tmp_path)
    identity = _identity()
    depth = np.linspace(0, 1, 20, dtype=np.float32).reshape(4, 5)

    assert cache.store(identity=identity, depth=depth)
    cached = cache.get(identity=identity)

    assert cached is not None
    np.testing.assert_array_equal(cached, depth)


@pytest.mark.parametrize(
    "depth",
    [
        np.arange(4, dtype=np.float32),
        np.arange(16, dtype=np.float32).reshape(4, 4).T,
        np.empty((0, 2), dtype=np.float32),
        np.ones((2, 2), dtype=np.complex64),
        np.array([[object()]], dtype=object),
    ],
    ids=["one-dimensional", "non-contiguous", "empty", "complex", "object"],
)
def test_store_rejects_noncanonical_depth_arrays(tmp_path, depth) -> None:
    cache = DepthCache(tmp_path)

    assert not cache.store(_identity(), depth)
    assert cache.stats()["entry_count"] == 0


@pytest.mark.parametrize(
    "changed_identity",
    [
        _identity(config=_sha("different-config")),
        _identity(revision="b" * 40),
        _identity(weights=_sha("different-weights")),
        _identity(dependency=_sha("different-dependency")),
    ],
    ids=["config", "model-revision", "weights", "dependency-lock"],
)
def test_identity_change_is_a_miss(tmp_path, changed_identity) -> None:
    cache = DepthCache(tmp_path)
    assert cache.store(_identity(), np.ones((3, 3), dtype=np.float32))
    assert cache.get(changed_identity) is None


def test_incomplete_seed_and_legacy_string_key_cannot_access_cache(tmp_path) -> None:
    cache = DepthCache(tmp_path)
    seed = ExecutionIdentityV3.from_plan(
        _plan(revision="a" * 40, config=_sha("config")),
        stage_node_id="lux.depth",
        candidate_id="da3",
        input_id="input-000001",
    )

    with (
        patch.object(cache, "_validate_namespace_roots") as validate_namespace,
        patch.object(cache, "_entry_path") as derive_entry_path,
        patch.object(cache, "_serialize_depth") as serialize_depth,
    ):
        assert cache.get(seed) is None  # type: ignore[arg-type]
        assert not cache.store(seed, np.ones((2, 2), dtype=np.float32))  # type: ignore[arg-type]
        assert cache.get("image_sha", "config") is None
        assert cache.get(image_sha256="image_sha", config_fingerprint="config") is None
        assert not cache.store("image_sha", "config", np.ones((2, 2), dtype=np.float32))
        assert not cache.store(
            image_sha256="image_sha",
            config_fingerprint="config",
            depth=np.ones((2, 2), dtype=np.float32),
        )
        validate_namespace.assert_not_called()
        derive_entry_path.assert_not_called()
        serialize_depth.assert_not_called()


def test_legacy_npy_without_pointer_is_a_miss_and_not_counted(tmp_path) -> None:
    cache = DepthCache(tmp_path)
    legacy = cache.cache_dir / "legacy_config.npy"
    np.save(legacy, np.ones((2, 2), dtype=np.float32), allow_pickle=False)

    assert cache.get(_identity()) is None
    assert cache.stats()["entry_count"] == 0


def test_missing_object_with_pointer_is_a_miss_and_pointer_is_cleaned(tmp_path) -> None:
    cache = DepthCache(tmp_path)
    identity = _identity()
    assert cache.store(identity, np.ones((2, 2), dtype=np.float32))
    pointer_path, object_path, _ = _entry_paths(cache, identity)
    object_path.unlink()

    assert cache.get(identity) is None
    assert not pointer_path.exists()


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        ("schema", "tp.lux.depth-cache.pointer.v999"),
        ("cache_schema", "tp.lux.depth-cache.v999"),
        ("shape", [99, 99]),
        ("dtype", "<f8"),
        ("byte_length", 1),
    ],
)
def test_pointer_schema_and_array_metadata_mismatch_are_misses(tmp_path, field, invalid) -> None:
    cache = DepthCache(tmp_path)
    identity = _identity()
    assert cache.store(identity, np.ones((2, 2), dtype=np.float32))
    pointer_path, _, pointer = _entry_paths(cache, identity)
    pointer[field] = invalid
    _rewrite_pointer(pointer_path, pointer)

    assert cache.get(identity) is None
    assert not pointer_path.exists()


@pytest.mark.parametrize("mutation", ["missing", "unknown"], ids=["missing-key", "unknown-key"])
def test_pointer_closed_schema_rejects_missing_and_unknown_keys(tmp_path, mutation) -> None:
    cache = DepthCache(tmp_path)
    identity = _identity()
    assert cache.store(identity, np.ones((2, 2), dtype=np.float32))
    pointer_path, _, pointer = _entry_paths(cache, identity)
    if mutation == "missing":
        pointer.pop("shape")
    else:
        pointer["unexpected"] = True
    _rewrite_pointer(pointer_path, pointer)

    assert cache.get(identity) is None
    assert not pointer_path.exists()


def test_pointer_rejects_duplicate_json_keys(tmp_path) -> None:
    cache = DepthCache(tmp_path)
    identity = _identity()
    assert cache.store(identity, np.ones((2, 2), dtype=np.float32))
    pointer_path, _, pointer = _entry_paths(cache, identity)
    raw = canonicalize_json(pointer)
    pointer_path.write_bytes(raw[:-1] + b',"schema":"tp.lux.depth-cache.pointer.v1"}')

    assert cache.get(identity) is None
    assert not pointer_path.exists()


def test_pointer_rejects_noncanonical_json(tmp_path) -> None:
    cache = DepthCache(tmp_path)
    identity = _identity()
    assert cache.store(identity, np.ones((2, 2), dtype=np.float32))
    pointer_path, _, pointer = _entry_paths(cache, identity)
    pointer_path.write_text(json.dumps(pointer, indent=2), encoding="utf-8")

    assert cache.get(identity) is None


def test_pointer_rejects_oversize_payload(tmp_path) -> None:
    cache = DepthCache(tmp_path)
    identity = _identity()
    assert cache.store(identity, np.ones((2, 2), dtype=np.float32))
    pointer_path, _, _ = _entry_paths(cache, identity)
    pointer_path.write_bytes(b" " * (64 * 1024 + 1))

    assert cache.get(identity) is None


def test_pointer_rejects_placeholder_digest(tmp_path) -> None:
    cache = DepthCache(tmp_path)
    identity = _identity()
    assert cache.store(identity, np.ones((2, 2), dtype=np.float32))
    pointer_path, _, pointer = _entry_paths(cache, identity)
    pointer["npy_sha256"] = "0" * 64
    _rewrite_pointer(pointer_path, pointer)

    assert cache.get(identity) is None


@pytest.mark.parametrize(
    ("projection", "invalid"),
    [
        ("config_fingerprint_sha256", _sha("wrong-config")),
        ("dependency_lock_sha256", _sha("wrong-dependency")),
    ],
)
def test_pointer_identity_projection_mismatch_is_a_miss(tmp_path, projection, invalid) -> None:
    cache = DepthCache(tmp_path)
    identity = _identity()
    assert cache.store(identity, np.ones((2, 2), dtype=np.float32))
    pointer_path, _, pointer = _entry_paths(cache, identity)
    pointer[projection] = invalid
    _rewrite_pointer(pointer_path, pointer)

    assert cache.get(identity) is None


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        ("model_lock_revision", "b" * 40),
        ("materialized_weights_sha256", _sha("wrong-weights")),
    ],
)
def test_pointer_model_projection_mismatch_is_a_miss(tmp_path, field, invalid) -> None:
    cache = DepthCache(tmp_path)
    identity = _identity()
    assert cache.store(identity, np.ones((2, 2), dtype=np.float32))
    pointer_path, _, pointer = _entry_paths(cache, identity)
    pointer["model_constituents"][0][field] = invalid
    _rewrite_pointer(pointer_path, pointer)

    assert cache.get(identity) is None


def test_corrupt_npy_checksum_is_a_miss(tmp_path) -> None:
    cache = DepthCache(tmp_path)
    identity = _identity()
    assert cache.store(identity, np.ones((2, 2), dtype=np.float32))
    pointer_path, object_path, _ = _entry_paths(cache, identity)
    with object_path.open("r+b") as handle:
        handle.seek(-1, os.SEEK_END)
        handle.write(b"X")

    assert cache.get(identity) is None
    assert not pointer_path.exists()


def test_object_array_is_rejected_with_allow_pickle_false(tmp_path, monkeypatch) -> None:
    cache = DepthCache(tmp_path)
    identity = _identity()
    assert cache.store(identity, np.ones((2, 2), dtype=np.float32))
    pointer_path, object_path, pointer = _entry_paths(cache, identity)

    buffer = io.BytesIO()
    np.save(buffer, np.array([[{"never": "execute"}]], dtype=object), allow_pickle=True)
    object_bytes = buffer.getvalue()
    object_path.write_bytes(object_bytes)
    pointer["npy_sha256"] = hashlib.sha256(object_bytes).hexdigest()
    pointer["byte_length"] = len(object_bytes)
    pointer["shape"] = [1, 1]
    pointer["dtype"] = "<f4"
    new_object_path = cache._object_path(pointer["npy_sha256"])
    new_object_path.parent.mkdir(exist_ok=True)
    object_path.replace(new_object_path)
    _rewrite_pointer(pointer_path, pointer)

    def forbidden_load(*_args, **_kwargs):
        pytest.fail("np.load must not run for an object-dtype NumPy header")

    monkeypatch.setattr(depth_cache_module.np, "load", forbidden_load)
    assert cache.get(identity) is None


def test_forged_huge_shape_is_rejected_before_numpy_load(tmp_path, monkeypatch) -> None:
    cache = DepthCache(tmp_path)
    identity = _identity()
    assert cache.store(identity, np.ones((2, 2), dtype=np.float32))
    pointer_path, object_path, pointer = _entry_paths(cache, identity)

    buffer = io.BytesIO()
    np.lib.format.write_array_header_1_0(
        buffer,
        {"descr": "<f4", "fortran_order": False, "shape": (100_000, 100_000)},
    )
    forged_bytes = buffer.getvalue()  # Header only: no 40GB payload.
    pointer["npy_sha256"] = hashlib.sha256(forged_bytes).hexdigest()
    pointer["byte_length"] = len(forged_bytes)
    pointer["shape"] = [100_000, 100_000]
    pointer["dtype"] = "<f4"
    forged_path = cache._object_path(pointer["npy_sha256"])
    object_path.unlink()
    forged_path.parent.mkdir(exist_ok=True)
    forged_path.write_bytes(forged_bytes)
    _rewrite_pointer(pointer_path, pointer)

    def forbidden_load(*_args, **_kwargs):
        pytest.fail("np.load must not run for an inconsistent huge-shape pointer")

    monkeypatch.setattr(depth_cache_module.np, "load", forbidden_load)
    assert cache.get(identity) is None


def test_actual_huge_shape_with_benign_sidecar_is_rejected_before_numpy_load(tmp_path, monkeypatch) -> None:
    cache = DepthCache(tmp_path)
    identity = _identity()
    assert cache.store(identity, np.ones((1, 1), dtype=np.float32))
    pointer_path, object_path, pointer = _entry_paths(cache, identity)

    buffer = io.BytesIO()
    np.lib.format.write_array_header_1_0(
        buffer,
        {"descr": "<f4", "fortran_order": False, "shape": (100_000, 100_000)},
    )
    forged_bytes = buffer.getvalue() + b"\0" * np.dtype("<f4").itemsize
    pointer["npy_sha256"] = hashlib.sha256(forged_bytes).hexdigest()
    pointer["byte_length"] = len(forged_bytes)
    pointer["shape"] = [1, 1]
    pointer["dtype"] = "<f4"
    forged_path = cache._object_path(pointer["npy_sha256"])
    object_path.unlink()
    forged_path.parent.mkdir(exist_ok=True)
    forged_path.write_bytes(forged_bytes)
    _rewrite_pointer(pointer_path, pointer)

    def forbidden_load(*_args, **_kwargs):
        pytest.fail("np.load must not run for a malicious actual NumPy header")

    monkeypatch.setattr(depth_cache_module.np, "load", forbidden_load)
    assert cache.get(identity) is None


@pytest.mark.parametrize("variant", ["fortran", "unsupported-version", "trailing-payload"])
def test_actual_numpy_header_contract_is_bound_before_numpy_load(tmp_path, monkeypatch, variant) -> None:
    cache = DepthCache(tmp_path)
    identity = _identity()
    assert cache.store(identity, np.ones((1, 1), dtype=np.float32))
    pointer_path, object_path, pointer = _entry_paths(cache, identity)

    if variant == "unsupported-version":
        forged_bytes = b"\x93NUMPY\x03\x00" + b"\0" * 8 + b"\0" * 4
    else:
        buffer = io.BytesIO()
        np.lib.format.write_array_header_1_0(
            buffer,
            {"descr": "<f4", "fortran_order": variant == "fortran", "shape": (1, 1)},
        )
        payload_items = 2 if variant == "trailing-payload" else 1
        forged_bytes = buffer.getvalue() + b"\0" * (np.dtype("<f4").itemsize * payload_items)

    pointer["npy_sha256"] = hashlib.sha256(forged_bytes).hexdigest()
    pointer["byte_length"] = len(forged_bytes)
    pointer["shape"] = [1, 1]
    pointer["dtype"] = "<f4"
    forged_path = cache._object_path(pointer["npy_sha256"])
    object_path.unlink()
    forged_path.parent.mkdir(exist_ok=True)
    forged_path.write_bytes(forged_bytes)
    _rewrite_pointer(pointer_path, pointer)

    def forbidden_load(*_args, **_kwargs):
        pytest.fail(f"np.load must not run for {variant} NumPy metadata")

    monkeypatch.setattr(depth_cache_module.np, "load", forbidden_load)
    assert cache.get(identity) is None


def test_path_replacement_during_np_load_is_rejected_and_reconciled(tmp_path, monkeypatch) -> None:
    cache = DepthCache(tmp_path)
    identity = _identity()
    depth = np.arange(16, dtype=np.float32).reshape(4, 4)
    assert cache.store(identity, depth)
    _, object_path, _ = _entry_paths(cache, identity)

    replacement_path = object_path.with_name("replacement.npy")
    with replacement_path.open("wb") as handle:
        np.save(handle, np.full((4, 4), 999, dtype=np.float32), allow_pickle=False)
    original_load = np.load

    def replacing_load(handle, *args, **kwargs):
        replacement_path.replace(object_path)
        return original_load(handle, *args, **kwargs)

    monkeypatch.setattr(depth_cache_module.np, "load", replacing_load)
    cached = cache.get(identity)

    assert cached is None
    assert not object_path.exists()


def test_shared_object_replacement_during_validation_cannot_commit_stale_ledger(
    tmp_path,
    monkeypatch,
) -> None:
    cache = DepthCache(tmp_path)
    depth = np.arange(16, dtype=np.float32).reshape(4, 4)
    first = _identity(input_label="shared-race-first")
    second = _identity(input_label="shared-race-second")
    assert cache.store(first, depth)
    _, object_path, _ = _entry_paths(cache, first)
    replacement_path = object_path.with_name("replacement.npy")
    replacement_path.write_bytes(b"corrupt-replacement" * 256)
    original_load = np.load
    replaced = False

    def replacing_load(handle, *args, **kwargs):
        nonlocal replaced
        if not replaced:
            replacement_path.replace(object_path)
            replaced = True
        return original_load(handle, *args, **kwargs)

    monkeypatch.setattr(depth_cache_module.np, "load", replacing_load)

    assert cache.store(second, depth)
    assert replaced
    first_cached = cache.get(first)
    second_cached = cache.get(second)
    assert first_cached is None
    assert second_cached is not None
    np.testing.assert_array_equal(second_cached, depth)
    with cache._locked_shards(range(depth_cache_module._LOCK_SHARD_COUNT)):
        state = cache._read_quota_state_locked()
    assert state is not None
    assert state.phase == "clean"
    assert state.physical_size_bytes == cache._physical_size_bytes()


def test_object_publication_without_pointer_is_never_a_hit(tmp_path, monkeypatch) -> None:
    cache = DepthCache(tmp_path)
    identity = _identity()
    real_atomic_write = cache._atomic_write_namespace

    def fail_pointer(path, root, data):
        if root == cache._entries_dir:
            raise OSError("simulated pointer publication crash")
        return real_atomic_write(path, root, data)

    monkeypatch.setattr(cache, "_atomic_write_namespace", fail_pointer)
    assert not cache.store(identity, np.ones((3, 3), dtype=np.float32))
    assert cache.get(identity) is None
    assert list((cache.cache_dir / "v1" / "entries").glob("*/*.json")) == []
    assert len(list((cache.cache_dir / "v1" / "objects").glob("*/*.npy"))) == 1

    assert cache.stats()["entry_count"] == 0
    assert list((cache.cache_dir / "v1" / "objects").glob("*/*.npy")) == []


def test_two_identities_share_one_immutable_object_and_clear_removes_all_pairs(tmp_path) -> None:
    cache = DepthCache(tmp_path)
    depth = np.arange(12, dtype=np.float32).reshape(3, 4)
    assert cache.store(_identity(input_label="one"), depth)
    assert cache.store(_identity(input_label="two"), depth)

    assert cache.stats()["entry_count"] == 2
    assert len(list((cache.cache_dir / "v1" / "entries").glob("*/*.json"))) == 2
    assert len(list((cache.cache_dir / "v1" / "objects").glob("*/*.npy"))) == 1

    cache.clear()
    assert cache.stats()["entry_count"] == 0
    assert list((cache.cache_dir / "v1" / "entries").glob("*/*.json")) == []
    assert list((cache.cache_dir / "v1" / "objects").glob("*/*.npy")) == []


def test_restore_after_missing_object_reconciles_clean_quota_ledger(tmp_path) -> None:
    cache = DepthCache(tmp_path)
    identity = _identity(input_label="restore-missing-object")
    depth = np.arange(12, dtype=np.float32).reshape(3, 4)
    assert cache.store(identity, depth)
    _pointer_path, object_path, _pointer = _entry_paths(cache, identity)
    object_path.unlink()

    assert cache.store(identity, depth)

    with cache._locked_shards(range(depth_cache_module._LOCK_SHARD_COUNT)):
        state = cache._read_quota_state_locked()
    physical_bytes = sum(path.stat().st_size for path in cache._entries_dir.glob("*/*.json")) + sum(
        path.stat().st_size for path in cache._objects_dir.glob("*/*.npy")
    )
    assert state is not None
    assert state.phase == "clean"
    assert state.physical_size_bytes == physical_bytes


def test_positive_limit_eviction_leaves_only_complete_pointer_object_pairs(tmp_path) -> None:
    max_bytes = 3_500
    cache = DepthCache(tmp_path, max_size_gb=max_bytes / (1024**3))
    for index in range(6):
        depth = np.full((16, 16), index, dtype=np.float32)
        assert cache.store(_identity(input_label=f"eviction-{index}"), depth)

    pointer_paths = list((cache.cache_dir / "v1" / "entries").glob("*/*.json"))
    object_paths = list((cache.cache_dir / "v1" / "objects").glob("*/*.npy"))
    referenced_digests = {json.loads(path.read_bytes())["npy_sha256"] for path in pointer_paths}
    object_digests = {path.stem for path in object_paths}
    physical_bytes = sum(path.stat().st_size for path in pointer_paths + object_paths)

    assert 0 < len(pointer_paths) < 6
    assert referenced_digests == object_digests
    assert physical_bytes <= max_bytes


def test_quota_eviction_preserves_shared_object_until_last_pointer(tmp_path) -> None:
    cache = DepthCache(tmp_path, max_size_gb=1.0)
    depth = np.arange(256, dtype=np.float32).reshape(16, 16)
    first = _identity(input_label="shared-first")
    second = _identity(input_label="shared-second")
    third = _identity(input_label="shared-third")
    assert cache.store(first, depth)
    assert cache.store(second, depth)

    pointer_path, object_path, _ = _entry_paths(cache, first)
    target_bytes = object_path.stat().st_size + pointer_path.stat().st_size
    cache.max_size_gb = target_bytes / (1024**3)

    assert cache.store(third, depth)
    assert cache.get(third) is not None
    assert cache.stats()["entry_count"] == 1
    assert object_path.exists()

    cache.max_size_gb = 0
    cache._enforce_size_limit()
    assert cache.stats()["entry_count"] == 0
    assert not object_path.exists()


def test_infeasible_shared_object_pointer_preserves_existing_verified_hit(tmp_path) -> None:
    cache = DepthCache(tmp_path, max_size_gb=1.0)
    depth = np.arange(256, dtype=np.float32).reshape(16, 16)
    existing_identity = _identity(input_label="shared-existing")
    assert cache.store(existing_identity, depth)
    pointer_path, object_path, _ = _entry_paths(cache, existing_identity)
    exact_existing_bytes = pointer_path.stat().st_size + object_path.stat().st_size
    cache.max_size_gb = exact_existing_bytes / (1024**3)

    larger_pointer_identity = _core_materialized(
        CanonicalExecutionPlan.from_payload(_with_backend_shape(_valid_payload(), ["ensemble"])),
        input_content_sha256=_sha("shared-larger-pointer"),
    )
    assert not cache.store(larger_pointer_identity, depth)

    cached = cache.get(existing_identity)
    assert cached is not None
    np.testing.assert_array_equal(cached, depth)
    assert pointer_path.exists()
    assert object_path.exists()


def test_zero_size_limit_refuses_object_without_leaving_artifacts(tmp_path) -> None:
    cache = DepthCache(tmp_path, max_size_gb=0)
    assert not cache.store(_identity(), np.ones((8, 8), dtype=np.float32))

    assert cache.get(_identity()) is None
    assert cache.stats()["entry_count"] == 0
    assert list((cache.cache_dir / "v1" / "entries").glob("*/*.json")) == []
    assert list((cache.cache_dir / "v1" / "objects").glob("*/*.npy")) == []


def test_serialized_object_limit_includes_npy_header_bytes(tmp_path, monkeypatch) -> None:
    cache = DepthCache(tmp_path)
    depth = np.ones((3, 3), dtype=np.float64)
    monkeypatch.setattr(depth_cache_module, "_ABSOLUTE_OBJECT_MAX_BYTES", depth.nbytes + 1)

    assert not cache.store(_identity(), depth)
    assert list(cache._entries_dir.glob("*/*.json")) == []
    assert list(cache._objects_dir.glob("*/*.npy")) == []


def test_stats_counts_only_complete_verified_entries_and_unique_physical_bytes(tmp_path) -> None:
    cache = DepthCache(tmp_path)
    depth = np.arange(8, dtype=np.float32).reshape(2, 4)
    assert cache.store(_identity(input_label="one"), depth)
    assert cache.store(_identity(input_label="two"), depth)
    malformed = cache.cache_dir / "v1" / "entries" / "aa" / f"{'a' * 64}.json"
    malformed.parent.mkdir(exist_ok=True)
    malformed.write_text("{}", encoding="utf-8")

    stats = cache.stats()
    pointer_bytes = sum(path.stat().st_size for path in (cache.cache_dir / "v1" / "entries").glob("*/*.json"))
    object_bytes = sum(path.stat().st_size for path in (cache.cache_dir / "v1" / "objects").glob("*/*.npy"))

    assert stats["entry_count"] == 2
    assert stats["size_gb"] == pytest.approx((pointer_bytes + object_bytes) / (1024**3))
    assert not malformed.exists()


def test_stats_validates_shared_cas_object_once_but_retains_all_pointer_rows(tmp_path, monkeypatch) -> None:
    cache = DepthCache(tmp_path)
    depth = np.arange(16, dtype=np.float32).reshape(4, 4)
    for index in range(32):
        assert cache.store(_identity(input_label=f"shared-{index}"), depth)

    validation_count = 0
    real_inspect_verified_object = cache._inspect_verified_object

    def counting_inspect_verified_object(path):
        nonlocal validation_count
        validation_count += 1
        return real_inspect_verified_object(path)

    monkeypatch.setattr(cache, "_inspect_verified_object", counting_inspect_verified_object)
    stats = cache.stats()

    assert validation_count == 1
    assert stats["entry_count"] == 32
    assert len(list(cache._objects_dir.glob("*/*.npy"))) == 1


def test_normal_shared_object_store_fully_validates_cas_at_most_once(tmp_path, monkeypatch) -> None:
    cache = DepthCache(tmp_path)
    depth = np.arange(16, dtype=np.float32).reshape(4, 4)
    assert cache.store(_identity(input_label="shared-first"), depth)

    validation_count = 0
    real_inspect_verified_object = cache._inspect_verified_object

    def counting_inspect_verified_object(path):
        nonlocal validation_count
        validation_count += 1
        return real_inspect_verified_object(path)

    monkeypatch.setattr(cache, "_inspect_verified_object", counting_inspect_verified_object)

    assert cache.store(_identity(input_label="shared-second"), depth)
    assert validation_count == 1
    assert cache.stats()["entry_count"] == 2


@pytest.mark.parametrize("operation", ["get", "stats"])
def test_deeply_nested_pointer_is_a_cleanable_miss(tmp_path, operation) -> None:
    cache = DepthCache(tmp_path)
    identity = _identity()
    assert cache.store(identity, np.ones((2, 2), dtype=np.float32))
    pointer_path, _, pointer = _entry_paths(cache, identity)
    needle = f'"dtype":"{pointer["dtype"]}"'.encode("utf-8")
    nested = b'"dtype":' + (b"[" * 1_500) + json.dumps(pointer["dtype"]).encode("utf-8") + (b"]" * 1_500)
    poisoned = pointer_path.read_bytes().replace(needle, nested)
    assert poisoned != pointer_path.read_bytes()
    assert len(poisoned) < 64 * 1024
    pointer_path.write_bytes(poisoned)

    if operation == "get":
        assert cache.get(identity) is None
    else:
        assert cache.stats()["entry_count"] == 0

    assert not pointer_path.exists()


@pytest.mark.parametrize(
    ("root_attribute", "destination_suffix"),
    [("_entries_dir", ".json"), ("_objects_dir", ".npy")],
)
def test_housekeeping_removes_only_exact_governed_temp_names(
    tmp_path,
    root_attribute,
    destination_suffix,
) -> None:
    cache = DepthCache(tmp_path)
    root = getattr(cache, root_attribute)
    digest = _sha(f"{root_attribute}-temp")
    shard_dir = root / digest[:2]
    shard_dir.mkdir(exist_ok=True)
    governed = shard_dir / f".{digest}{destination_suffix}.tmp-{'a' * 32}"
    governed.write_bytes(b"governed-temp")
    lookalikes = [
        shard_dir / f".{digest}{destination_suffix}.tmp-{'g' * 32}",
        shard_dir / f".{digest}{destination_suffix}.tmp-{'b' * 31}",
        shard_dir / f".{digest}{destination_suffix}.tmp-{'c' * 32}.extra",
        shard_dir / f".{digest}.bin.tmp-{'d' * 32}",
    ]
    for path in lookalikes:
        path.write_bytes(b"user-sentinel")

    cache = DepthCache(tmp_path)

    # A clean durable quota state is the constructor fast path. Periodic or
    # explicit housekeeping owns orphan cleanup rather than every process
    # initialization rehashing the complete namespace.
    assert governed.read_bytes() == b"governed-temp"
    assert all(path.read_bytes() == b"user-sentinel" for path in lookalikes)
    cache.stats()

    assert not governed.exists()
    assert all(path.read_bytes() == b"user-sentinel" for path in lookalikes)

    governed = shard_dir / f".{digest}{destination_suffix}.tmp-{'f' * 32}"
    governed.write_bytes(b"governed-temp")
    cache._enforce_size_limit()

    assert not governed.exists()
    assert all(path.read_bytes() == b"user-sentinel" for path in lookalikes)

    governed = shard_dir / f".{digest}{destination_suffix}.tmp-{'0' * 32}"
    governed.write_bytes(b"governed-temp")
    cache.clear()

    assert not governed.exists()
    assert all(path.read_bytes() == b"user-sentinel" for path in lookalikes)


def test_undeletable_governed_temp_remains_in_physical_size_accounting(tmp_path, monkeypatch) -> None:
    cache = DepthCache(tmp_path)
    digest = _sha("undeletable-temp")
    shard_dir = cache._objects_dir / digest[:2]
    shard_dir.mkdir(exist_ok=True)
    temp_path = shard_dir / f".{digest}.npy.tmp-{'f' * 32}"
    payload = b"stale-but-still-physical"
    temp_path.write_bytes(payload)
    real_unlink = cache._unlink_namespace

    def fail_governed_temp_unlink(path, root, **kwargs):
        if path == temp_path:
            raise OSError("simulated undeletable temp")
        return real_unlink(path, root, **kwargs)

    monkeypatch.setattr(cache, "_unlink_namespace", fail_governed_temp_unlink)

    stats = cache.stats()

    assert temp_path.read_bytes() == payload
    assert stats["entry_count"] == 0
    assert stats["size_gb"] == pytest.approx(len(payload) / (1024**3))


@pytest.mark.parametrize("substituted_root", ["base", ".depth_cache", "v1", "entries", "objects", "locks"])
def test_constructor_rejects_namespace_root_symlinks_without_outside_writes(tmp_path, substituted_root) -> None:
    base = tmp_path / "base"
    outside = tmp_path / "outside"
    outside.mkdir()

    if substituted_root == "base":
        base.symlink_to(outside, target_is_directory=True)
    else:
        base.mkdir()
        depth_root = base / ".depth_cache"
        if substituted_root == ".depth_cache":
            depth_root.symlink_to(outside, target_is_directory=True)
        else:
            depth_root.mkdir()
            v1_root = depth_root / "v1"
            if substituted_root == "v1":
                v1_root.symlink_to(outside, target_is_directory=True)
            else:
                v1_root.mkdir()
                for name in ("entries", "objects", "locks"):
                    target = v1_root / name
                    if name == substituted_root:
                        target.symlink_to(outside, target_is_directory=True)
                        break
                    target.mkdir()

    with pytest.raises(OSError, match="namespace root"):
        DepthCache(base)
    assert list(outside.iterdir()) == []


def test_constructor_swap_during_hierarchy_creation_never_writes_outside(tmp_path, monkeypatch) -> None:
    base = tmp_path / "base"
    base.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    original = tmp_path / "base-original"
    real_mkdir = os.mkdir
    swapped = False

    def swapping_mkdir(path, mode=0o777, *, dir_fd=None):
        nonlocal swapped
        if not swapped and os.fspath(path) == ".depth_cache" and dir_fd is not None:
            base.rename(original)
            base.symlink_to(outside, target_is_directory=True)
            swapped = True
        return real_mkdir(path, mode, dir_fd=dir_fd)

    monkeypatch.setattr(depth_cache_module.os, "mkdir", swapping_mkdir)

    with pytest.raises(OSError, match="securely initialized"):
        DepthCache(base)

    assert swapped
    assert list(outside.iterdir()) == []
    assert list(original.iterdir()) == []


def test_constructor_fsyncs_each_created_directory_through_parent(tmp_path, monkeypatch) -> None:
    real_mkdir = os.mkdir
    real_fsync = os.fsync
    events: list[tuple[str, object]] = []

    def recording_mkdir(path, mode=0o777, *, dir_fd=None):
        result = real_mkdir(path, mode, dir_fd=dir_fd)
        events.append(("mkdir", (os.fspath(path), dir_fd)))
        return result

    def recording_fsync(descriptor):
        events.append(("fsync", descriptor))
        return real_fsync(descriptor)

    monkeypatch.setattr(depth_cache_module.os, "mkdir", recording_mkdir)
    monkeypatch.setattr(depth_cache_module.os, "fsync", recording_fsync)

    DepthCache(tmp_path / "nested" / "base")

    mkdir_indexes = [index for index, event in enumerate(events) if event[0] == "mkdir"]
    assert len(mkdir_indexes) == 7
    for index in mkdir_indexes:
        _, (_, parent_descriptor) = events[index]
        assert events[index + 1] == ("fsync", parent_descriptor)


def test_constructor_fsync_failure_rolls_back_created_directory(tmp_path, monkeypatch) -> None:
    base = tmp_path / "base"
    base.mkdir()
    base_identity = (base.stat().st_dev, base.stat().st_ino)
    real_fsync = os.fsync
    failed = False

    def fail_created_directory_fsync(descriptor):
        nonlocal failed
        descriptor_stat = os.fstat(descriptor)
        if (
            not failed
            and (descriptor_stat.st_dev, descriptor_stat.st_ino) == base_identity
            and (base / ".depth_cache").is_dir()
        ):
            failed = True
            raise OSError("simulated parent directory fsync failure")
        return real_fsync(descriptor)

    monkeypatch.setattr(depth_cache_module.os, "fsync", fail_created_directory_fsync)

    with pytest.raises(OSError, match="securely initialized"):
        DepthCache(base)

    assert failed
    assert list(base.iterdir()) == []


def test_constructor_retry_fences_preexisting_cache_owned_parent(tmp_path, monkeypatch) -> None:
    base = tmp_path / "base"
    cache_root = base / ".depth_cache"
    cache_root.mkdir(parents=True)
    base_identity = (base.stat().st_dev, base.stat().st_ino)
    real_mkdir = os.mkdir
    real_fsync = os.fsync
    events: list[tuple[str, object]] = []

    def recording_mkdir(path, mode=0o777, *, dir_fd=None):
        events.append(("mkdir", os.fspath(path)))
        return real_mkdir(path, mode, dir_fd=dir_fd)

    def recording_fsync(descriptor):
        descriptor_stat = os.fstat(descriptor)
        events.append(("fsync", (descriptor_stat.st_dev, descriptor_stat.st_ino)))
        return real_fsync(descriptor)

    monkeypatch.setattr(depth_cache_module.os, "mkdir", recording_mkdir)
    monkeypatch.setattr(depth_cache_module.os, "fsync", recording_fsync)

    DepthCache._initialize_namespace(base)

    cache_parent_fence = events.index(("fsync", base_identity))
    first_descendant_creation = events.index(("mkdir", "v1"))
    assert cache_parent_fence < first_descendant_creation


def test_constructor_retry_fences_preexisting_intermediate_caller_path(tmp_path, monkeypatch) -> None:
    leftover = tmp_path / "leftover-intermediate"
    leftover.mkdir()
    publishing_parent_identity = (tmp_path.stat().st_dev, tmp_path.stat().st_ino)
    real_fsync = os.fsync
    fenced_identities: list[tuple[int, int]] = []

    def recording_fsync(descriptor):
        descriptor_stat = os.fstat(descriptor)
        fenced_identities.append((descriptor_stat.st_dev, descriptor_stat.st_ino))
        return real_fsync(descriptor)

    monkeypatch.setattr(depth_cache_module.os, "fsync", recording_fsync)

    DepthCache(leftover / "base")

    assert publishing_parent_identity in fenced_identities


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS standard aliases only")
def test_constructor_accepts_standard_macos_tmp_and_var_aliases(tmp_path) -> None:
    with tempfile.TemporaryDirectory(prefix="tp-depth-cache-", dir="/tmp") as temporary:
        tmp_base = Path(temporary) / "cache"
        tmp_cache = DepthCache(tmp_base)
        assert tmp_cache.cache_dir == Path("/private/tmp").joinpath(*tmp_base.parts[2:], ".depth_cache")

    private_var = Path("/private/var")
    try:
        relative_var_path = tmp_path.relative_to(private_var)
    except ValueError:
        pytest.skip("pytest temporary directory is not rooted below /private/var")
    var_base = Path("/var") / relative_var_path / "cache"
    var_cache = DepthCache(var_base)

    assert var_cache.cache_dir == tmp_path / "cache" / ".depth_cache"


@pytest.mark.parametrize("root_attribute", ["cache_dir", "_v1_dir", "_entries_dir", "_objects_dir", "_locks_dir"])
def test_namespace_root_replacement_after_initialization_fails_closed(tmp_path, root_attribute) -> None:
    cache = DepthCache(tmp_path / "base")
    outside = tmp_path / "outside"
    outside.mkdir()
    namespace_root = getattr(cache, root_attribute)
    namespace_root.rename(namespace_root.with_name(f"{namespace_root.name}-original"))
    namespace_root.symlink_to(outside, target_is_directory=True)

    assert not cache.store(_identity(), np.ones((2, 2), dtype=np.float32))
    assert cache.get(_identity()) is None
    assert list(outside.iterdir()) == []


@pytest.mark.parametrize(
    ("root_attribute", "published_suffix"),
    [("_objects_dir", ".npy"), ("_entries_dir", ".json")],
)
def test_child_shard_swap_during_publication_cannot_write_outside(
    tmp_path,
    monkeypatch,
    root_attribute,
    published_suffix,
) -> None:
    cache = DepthCache(tmp_path / "base")
    outside = tmp_path / "outside"
    outside.mkdir()
    swapped: list[tuple[Path, str]] = []
    real_replace = os.replace

    def swapping_replace(source, destination, *, src_dir_fd=None, dst_dir_fd=None):
        destination_name = os.fspath(destination)
        if (
            not swapped
            and destination_name != depth_cache_module._QUOTA_STATE_NAME
            and destination_name.endswith(published_suffix)
        ):
            namespace_root = getattr(cache, root_attribute)
            shard_path = namespace_root / destination_name[:2]
            backup_path = shard_path.with_name(f"{shard_path.name}-original")
            shard_path.rename(backup_path)
            shard_path.symlink_to(outside, target_is_directory=True)
            swapped.append((backup_path, destination_name))
        return real_replace(
            source,
            destination,
            src_dir_fd=src_dir_fd,
            dst_dir_fd=dst_dir_fd,
        )

    monkeypatch.setattr(depth_cache_module.os, "replace", swapping_replace)
    assert not cache.store(_identity(), np.ones((2, 2), dtype=np.float32))

    assert len(swapped) == 1
    backup_path, destination_name = swapped[0]
    assert list(outside.iterdir()) == []
    assert not (backup_path / destination_name).exists()


def test_objects_root_replacement_during_publication_rolls_back_moved_destination(
    tmp_path,
    monkeypatch,
) -> None:
    cache = DepthCache(tmp_path / "base")
    moved_root = tmp_path / "objects-moved-outside"
    current_root = cache._objects_dir
    published_names: list[str] = []
    real_replace = os.replace

    def replacing_objects_root(source, destination, *, src_dir_fd=None, dst_dir_fd=None):
        destination_name = os.fspath(destination)
        if not published_names and destination_name.endswith(".npy"):
            current_root.rename(moved_root)
            current_root.mkdir()
            published_names.append(destination_name)
        return real_replace(
            source,
            destination,
            src_dir_fd=src_dir_fd,
            dst_dir_fd=dst_dir_fd,
        )

    monkeypatch.setattr(depth_cache_module.os, "replace", replacing_objects_root)

    assert not cache.store(_identity(), np.ones((2, 2), dtype=np.float32))
    assert len(published_names) == 1
    assert list(moved_root.glob("*/*.npy")) == []
    assert list(moved_root.glob("*/*.tmp-*")) == []
    assert list(current_root.glob("*/*.npy")) == []


def test_child_shard_swap_during_unlink_cannot_remove_outside_file(tmp_path, monkeypatch) -> None:
    cache = DepthCache(tmp_path / "base")
    identity = _identity()
    assert cache.store(identity, np.ones((2, 2), dtype=np.float32))
    pointer_path, _, _ = _entry_paths(cache, identity)
    outside = tmp_path / "outside"
    outside.mkdir()
    outside_pointer = outside / pointer_path.name
    outside_pointer.write_bytes(b"outside-sentinel")
    backup_path = pointer_path.parent.with_name(f"{pointer_path.parent.name}-original")
    real_rename = os.rename
    swapped = False

    def swapping_rename(source, destination, *, src_dir_fd=None, dst_dir_fd=None):
        nonlocal swapped
        if not swapped and os.fspath(source) == pointer_path.name and src_dir_fd is not None:
            pointer_path.parent.rename(backup_path)
            pointer_path.parent.symlink_to(outside, target_is_directory=True)
            swapped = True
        return real_rename(
            source,
            destination,
            src_dir_fd=src_dir_fd,
            dst_dir_fd=dst_dir_fd,
        )

    monkeypatch.setattr(depth_cache_module.os, "rename", swapping_rename)
    cache.clear()

    assert swapped
    assert outside_pointer.read_bytes() == b"outside-sentinel"
    assert not (backup_path / pointer_path.name).exists()


def test_lock_root_swap_during_acquisition_cannot_redirect_lock_or_publication(tmp_path, monkeypatch) -> None:
    cache = DepthCache(tmp_path / "base")
    outside = tmp_path / "outside"
    outside.mkdir()
    backup_root = cache._locks_dir.with_name("locks-original")
    real_acquire = depth_cache_module._acquire_platform_file_lock
    swapped = False

    def swapping_acquire(descriptor):
        nonlocal swapped
        cache._locks_dir.rename(backup_root)
        cache._locks_dir.symlink_to(outside, target_is_directory=True)
        swapped = True
        return real_acquire(descriptor)

    monkeypatch.setattr(depth_cache_module, "_acquire_platform_file_lock", swapping_acquire)
    assert not cache.store(_identity(), np.ones((2, 2), dtype=np.float32))

    assert swapped
    assert list(outside.iterdir()) == []


def test_get_returns_miss_when_namespace_root_changes_during_lock_acquisition(tmp_path, monkeypatch) -> None:
    cache = DepthCache(tmp_path / "base")
    identity = _identity()
    assert cache.store(identity, np.ones((2, 2), dtype=np.float32))
    outside = tmp_path / "outside"
    outside.mkdir()
    backup_root = cache._locks_dir.with_name("locks-original")
    real_acquire = depth_cache_module._acquire_platform_file_lock
    swapped = False

    def swapping_acquire(descriptor):
        nonlocal swapped
        cache._locks_dir.rename(backup_root)
        cache._locks_dir.symlink_to(outside, target_is_directory=True)
        swapped = True
        return real_acquire(descriptor)

    monkeypatch.setattr(depth_cache_module, "_acquire_platform_file_lock", swapping_acquire)

    assert cache.get(identity) is None
    assert swapped
    assert list(outside.iterdir()) == []


def test_quota_state_publication_rolls_back_when_locks_root_is_replaced(
    tmp_path,
    monkeypatch,
) -> None:
    cache = DepthCache(tmp_path / "base")
    moved_root = tmp_path / "locks-moved-outside"
    current_root = cache._locks_dir
    real_replace = os.replace
    swapped = False

    def replacing_locks_root(source, destination, *, src_dir_fd=None, dst_dir_fd=None):
        nonlocal swapped
        if not swapped and os.fspath(destination) == depth_cache_module._QUOTA_STATE_NAME:
            current_root.rename(moved_root)
            current_root.mkdir()
            swapped = True
        return real_replace(
            source,
            destination,
            src_dir_fd=src_dir_fd,
            dst_dir_fd=dst_dir_fd,
        )

    monkeypatch.setattr(depth_cache_module.os, "replace", replacing_locks_root)

    assert not cache.store(_identity(), np.ones((2, 2), dtype=np.float32))
    assert swapped
    assert not (moved_root / depth_cache_module._QUOTA_STATE_NAME).exists()
    assert list(moved_root.glob(".*.tmp-*")) == []
    assert list(current_root.iterdir()) == []


def test_quota_directory_fsync_failure_retains_prepared_maximum_authority(
    tmp_path,
    monkeypatch,
) -> None:
    max_bytes = 3_000
    cache = DepthCache(tmp_path, max_size_gb=max_bytes / (1024**3))
    locks_identity = (cache._locks_dir.stat().st_dev, cache._locks_dir.stat().st_ino)
    real_fsync = os.fsync
    failed = False

    def fail_locks_directory_fsync(descriptor):
        nonlocal failed
        descriptor_stat = os.fstat(descriptor)
        if not failed and (descriptor_stat.st_dev, descriptor_stat.st_ino) == locks_identity:
            failed = True
            raise OSError("simulated quota directory fsync failure")
        return real_fsync(descriptor)

    monkeypatch.setattr(depth_cache_module.os, "fsync", fail_locks_directory_fsync)

    assert not cache.store(_identity(), np.ones((2, 2), dtype=np.float32))
    assert failed
    state = cache._read_quota_state_locked()
    assert state is not None
    assert state.phase == "prepared"
    assert state.max_size_bytes == max_bytes

    with pytest.raises(ValueError, match="already configured with a different maximum"):
        DepthCache(tmp_path, max_size_gb=10_000 / (1024**3))


def test_quota_state_replacement_during_read_fails_closed(tmp_path, monkeypatch) -> None:
    cache = DepthCache(tmp_path)
    quota_path = cache._locks_dir / depth_cache_module._QUOTA_STATE_NAME
    replacement = cache._locks_dir / "replacement-quota.json"
    replacement.write_bytes(quota_path.read_bytes())
    real_read = os.read
    replaced = False

    def replacing_read(descriptor, count):
        nonlocal replaced
        raw = real_read(descriptor, count)
        if not replaced:
            os.replace(replacement, quota_path)
            replaced = True
        return raw

    monkeypatch.setattr(depth_cache_module.os, "read", replacing_read)

    with cache._locked_shards(range(depth_cache_module._LOCK_SHARD_COUNT)):
        with pytest.raises(OSError, match="changed or was replaced"):
            cache._read_quota_state_locked()

    assert replaced


@pytest.mark.parametrize("failure_stage", ["fchmod", "write", "fsync"])
def test_object_write_failure_removes_exact_owned_temporary_file(
    tmp_path,
    monkeypatch,
    failure_stage,
) -> None:
    cache = DepthCache(tmp_path)
    digest = _sha(f"failed-{failure_stage}")
    object_path = cache._object_path(digest)
    real_fchmod = os.fchmod
    real_fsync = os.fsync
    real_write_all = cache._write_all

    def fail_fchmod(descriptor, mode):
        if failure_stage == "fchmod":
            raise OSError("simulated object chmod failure")
        return real_fchmod(descriptor, mode)

    def fail_write(descriptor, data):
        if failure_stage == "write":
            raise OSError("simulated object write failure")
        return real_write_all(descriptor, data)

    def fail_fsync(descriptor):
        if failure_stage == "fsync" and stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise OSError("simulated object fsync failure")
        return real_fsync(descriptor)

    monkeypatch.setattr(depth_cache_module.os, "fchmod", fail_fchmod)
    monkeypatch.setattr(depth_cache_module.os, "fsync", fail_fsync)
    monkeypatch.setattr(cache, "_write_all", fail_write)

    with pytest.raises(OSError, match="simulated object"):
        cache._atomic_write_namespace(object_path, cache._objects_dir, b"object-bytes")

    assert not object_path.exists()
    assert list(object_path.parent.glob(f".{object_path.name}.tmp-*")) == []


def test_object_write_failure_after_bytes_removes_exact_owned_temporary_file(tmp_path, monkeypatch) -> None:
    cache = DepthCache(tmp_path)
    real_write_all = cache._write_all

    def fail_after_object_write(descriptor, data):
        real_write_all(descriptor, data)
        if data.startswith(b"\x93NUMPY"):
            raise OSError("simulated object write failure")

    monkeypatch.setattr(cache, "_write_all", fail_after_object_write)

    assert not cache.store(_identity(), np.ones((3, 3), dtype=np.float32))
    assert list(cache._objects_dir.glob("*/*.tmp-*")) == []
    assert list(cache._objects_dir.glob("*/*.npy")) == []


def test_atomic_cleanup_closes_rollback_descriptor_when_directory_fsync_fails(
    tmp_path,
    monkeypatch,
) -> None:
    cache = DepthCache(tmp_path)
    first_digest = "aa" + _sha("first")[2:]
    second_digest = "aa" + _sha("second")[2:]
    cache._atomic_write_namespace(cache._object_path(first_digest), cache._objects_dir, b"first")
    duplicated: list[int] = []
    closed: list[int] = []
    real_close = os.close
    real_dup = os.dup
    real_fsync = os.fsync

    def recording_dup(descriptor):
        duplicated_descriptor = real_dup(descriptor)
        duplicated.append(duplicated_descriptor)
        return duplicated_descriptor

    def recording_close(descriptor):
        closed.append(descriptor)
        return real_close(descriptor)

    def fail_write(_descriptor, _data):
        raise OSError("simulated publication write failure")

    def fail_cleanup_fsync(descriptor):
        if duplicated and descriptor == duplicated[-1]:
            raise OSError("simulated cleanup directory fsync failure")
        return real_fsync(descriptor)

    monkeypatch.setattr(depth_cache_module.os, "dup", recording_dup)
    monkeypatch.setattr(depth_cache_module.os, "close", recording_close)
    monkeypatch.setattr(depth_cache_module.os, "fsync", fail_cleanup_fsync)
    monkeypatch.setattr(cache, "_write_all", fail_write)

    with pytest.raises(OSError, match="cleanup directory fsync"):
        cache._atomic_write_namespace(cache._object_path(second_digest), cache._objects_dir, b"second")

    assert len(duplicated) == 1
    assert duplicated[0] in closed


def test_prepared_quota_state_is_reconciled_to_exact_clean_state_on_restart(tmp_path) -> None:
    max_size_gb = 1.0
    cache = DepthCache(tmp_path, max_size_gb=max_size_gb)
    identity = _identity()
    depth = np.arange(16, dtype=np.float32).reshape(4, 4)
    assert cache.store(identity, depth)

    with cache._locked_shards(range(depth_cache_module._LOCK_SHARD_COUNT)):
        state = cache._load_quota_state_locked()
        cache._write_quota_state_locked(
            phase="prepared",
            max_size_bytes=state.max_size_bytes,
            physical_size_bytes=0,
            store_count=state.store_count,
            reserved_add_bytes=123,
            planned_remove_bytes=456,
        )

    restarted = DepthCache(tmp_path, max_size_gb=max_size_gb)
    cached = restarted.get(identity)
    assert cached is not None
    np.testing.assert_array_equal(cached, depth)

    with restarted._locked_shards(range(depth_cache_module._LOCK_SHARD_COUNT)):
        recovered = restarted._read_quota_state_locked()
    pointer_bytes = sum(path.stat().st_size for path in restarted._entries_dir.glob("*/*.json"))
    object_bytes = sum(path.stat().st_size for path in restarted._objects_dir.glob("*/*.npy"))

    assert recovered is not None
    assert recovered.phase == "clean"
    assert recovered.physical_size_bytes == pointer_bytes + object_bytes
    assert recovered.reserved_add_bytes == 0
    assert recovered.planned_remove_bytes == 0


def test_clean_quota_state_is_constructor_fast_path(tmp_path, monkeypatch) -> None:
    cache = DepthCache(tmp_path, max_size_gb=1.0)
    abandoned_quota_temp = cache._locks_dir / (f"{depth_cache_module._QUOTA_STATE_NAME}.tmp-{'a' * 32}")
    abandoned_quota_temp.write_bytes(b"abandoned")

    def fail_scan(_self):
        raise AssertionError("clean constructor must not scan the cache namespace")

    monkeypatch.setattr(DepthCache, "_scan_cache_locked", fail_scan)

    restarted = DepthCache(tmp_path, max_size_gb=1.0)

    assert restarted._configured_max_size_bytes == 1024**3
    assert not abandoned_quota_temp.exists()


def test_namespace_maximum_conflicts_and_explicit_resize_use_compare_and_swap(tmp_path) -> None:
    small_bytes = 3_000
    large_bytes = 10_000
    small_gb = small_bytes / (1024**3)
    large_gb = large_bytes / (1024**3)
    owner = DepthCache(tmp_path, max_size_gb=small_gb)
    stale_peer = DepthCache(tmp_path, max_size_gb=small_gb)

    with pytest.raises(ValueError, match="already configured with a different maximum"):
        DepthCache(tmp_path, max_size_gb=large_gb)

    owner.max_size_gb = large_gb
    owner._enforce_size_limit()

    with owner._locked_shards(range(depth_cache_module._LOCK_SHARD_COUNT)):
        state = owner._read_quota_state_locked()
    assert state is not None
    assert state.max_size_bytes == large_bytes
    assert not stale_peer.store(_identity(input_label="stale-peer"), np.ones((2, 2), dtype=np.float32))
    stale_peer.max_size_gb = 20_000 / (1024**3)
    with pytest.raises(ValueError, match="changed before the requested resize"):
        stale_peer._enforce_size_limit()
    with owner._locked_shards(range(depth_cache_module._LOCK_SHARD_COUNT)):
        unchanged_state = owner._read_quota_state_locked()
    assert unchanged_state is not None
    assert unchanged_state.max_size_bytes == large_bytes
    assert DepthCache(tmp_path, max_size_gb=large_gb)._configured_max_size_bytes == large_bytes


def test_legacy_cleanup_uses_stable_root_descriptor_during_root_swap(tmp_path, monkeypatch) -> None:
    cache = DepthCache(tmp_path / "base")
    legacy_path = cache.cache_dir / "legacy.npy"
    np.save(legacy_path, np.ones((2, 2), dtype=np.float32), allow_pickle=False)
    outside = tmp_path / "outside"
    outside.mkdir()
    outside_legacy = outside / legacy_path.name
    outside_legacy.write_bytes(b"outside-sentinel")
    backup_root = cache.cache_dir.with_name(f"{cache.cache_dir.name}-original")
    real_rename = os.rename
    swapped = False

    def swapping_rename(source, destination, *, src_dir_fd=None, dst_dir_fd=None):
        nonlocal swapped
        if not swapped and os.fspath(source) == legacy_path.name and src_dir_fd is not None:
            cache.cache_dir.rename(backup_root)
            cache.cache_dir.symlink_to(outside, target_is_directory=True)
            swapped = True
        return real_rename(
            source,
            destination,
            src_dir_fd=src_dir_fd,
            dst_dir_fd=dst_dir_fd,
        )

    monkeypatch.setattr(depth_cache_module.os, "rename", swapping_rename)
    cache.clear()

    assert swapped
    assert outside_legacy.read_bytes() == b"outside-sentinel"
    assert not (backup_root / legacy_path.name).exists()


@pytest.mark.parametrize("root_attribute", ["_entries_dir", "_objects_dir", "_locks_dir"])
def test_clean_quota_state_is_bound_to_every_replaceable_child_root(
    tmp_path,
    monkeypatch,
    root_attribute,
) -> None:
    max_bytes = 3_000
    cache = DepthCache(tmp_path, max_size_gb=max_bytes / (1024**3))
    root = getattr(cache, root_attribute)
    moved_root = root.with_name(f"{root.name}-old")
    root.rename(moved_root)
    root.mkdir()

    oversized_artifact = None
    if root_attribute in {"_entries_dir", "_objects_dir"}:
        digest = _sha(root_attribute)
        shard = root / digest[:2]
        shard.mkdir()
        suffix = ".json" if root_attribute == "_entries_dir" else ".npy"
        oversized_artifact = shard / f"{digest}{suffix}"
        oversized_artifact.write_bytes(b"x" * 5_000)

    scan_count = 0
    real_scan = DepthCache._scan_cache_locked

    def counting_scan(self):
        nonlocal scan_count
        scan_count += 1
        return real_scan(self)

    monkeypatch.setattr(DepthCache, "_scan_cache_locked", counting_scan)
    restarted = DepthCache(tmp_path, max_size_gb=max_bytes / (1024**3))

    assert scan_count >= 1
    with restarted._locked_shards(range(depth_cache_module._LOCK_SHARD_COUNT)):
        state = restarted._read_quota_state_locked()
    assert state is not None
    assert state.phase == "clean"
    assert state.physical_size_bytes == restarted._physical_size_bytes()
    assert state.physical_size_bytes <= max_bytes
    if oversized_artifact is not None:
        assert not oversized_artifact.exists()


def test_clean_quota_state_is_bound_to_cache_root_when_v1_is_moved_intact(
    tmp_path,
    monkeypatch,
) -> None:
    max_bytes = 3_000
    cache = DepthCache(tmp_path, max_size_gb=max_bytes / (1024**3))
    moved_cache_root = tmp_path / ".depth_cache-old"
    cache.cache_dir.rename(moved_cache_root)
    cache.cache_dir.mkdir()
    (moved_cache_root / "v1").rename(cache.cache_dir / "v1")
    oversized_legacy = cache.cache_dir / "legacy.npy"
    oversized_legacy.write_bytes(b"x" * 5_000)
    scan_count = 0
    real_scan = DepthCache._scan_cache_locked

    def counting_scan(self):
        nonlocal scan_count
        scan_count += 1
        return real_scan(self)

    monkeypatch.setattr(DepthCache, "_scan_cache_locked", counting_scan)
    restarted = DepthCache(tmp_path, max_size_gb=max_bytes / (1024**3))

    assert scan_count >= 1
    assert not oversized_legacy.exists()
    with restarted._locked_shards(range(depth_cache_module._LOCK_SHARD_COUNT)):
        state = restarted._read_quota_state_locked()
    assert state is not None
    assert state.physical_size_bytes == restarted._physical_size_bytes() == 0


def test_quota_publication_is_invalidated_when_objects_root_changes_after_replace(
    tmp_path,
    monkeypatch,
) -> None:
    cache = DepthCache(tmp_path, max_size_gb=1.0)
    moved_objects = tmp_path / "objects-old"
    real_replace = os.replace
    swapped = False

    def replacing_objects_after_quota(source, destination, *, src_dir_fd=None, dst_dir_fd=None):
        nonlocal swapped
        result = real_replace(
            source,
            destination,
            src_dir_fd=src_dir_fd,
            dst_dir_fd=dst_dir_fd,
        )
        if not swapped and os.fspath(destination) == depth_cache_module._QUOTA_STATE_NAME:
            cache._objects_dir.rename(moved_objects)
            cache._objects_dir.mkdir()
            swapped = True
        return result

    monkeypatch.setattr(depth_cache_module.os, "replace", replacing_objects_after_quota)

    assert not cache.store(_identity(input_label="root-swap"), np.ones((2, 2), dtype=np.float32))
    assert swapped
    assert not (cache._locks_dir / depth_cache_module._QUOTA_STATE_NAME).exists()

    monkeypatch.setattr(depth_cache_module.os, "replace", real_replace)
    restarted = DepthCache(tmp_path, max_size_gb=1.0)
    with restarted._locked_shards(range(depth_cache_module._LOCK_SHARD_COUNT)):
        state = restarted._read_quota_state_locked()
    assert state is not None
    assert state.physical_size_bytes == restarted._physical_size_bytes() == 0


@pytest.mark.parametrize("writer", ["object", "quota"])
def test_close_error_cannot_skip_identity_bound_temporary_cleanup(tmp_path, monkeypatch, writer) -> None:
    cache = DepthCache(tmp_path)
    real_close = os.close
    raised = False

    def fail_first_regular_close_after_closing(descriptor):
        nonlocal raised
        descriptor_stat = os.fstat(descriptor)
        result = real_close(descriptor)
        if not raised and stat.S_ISREG(descriptor_stat.st_mode):
            raised = True
            raise OSError("simulated temporary descriptor close failure")
        return result

    def fail_write(_descriptor, _data):
        raise OSError("simulated publication write failure")

    monkeypatch.setattr(depth_cache_module.os, "close", fail_first_regular_close_after_closing)
    monkeypatch.setattr(cache, "_write_all", fail_write)

    with pytest.raises(OSError, match="descriptor close failure"):
        if writer == "object":
            digest = _sha("close-cleanup")
            cache._atomic_write_namespace(cache._object_path(digest), cache._objects_dir, b"payload")
        else:
            with cache._locked_shards(range(depth_cache_module._LOCK_SHARD_COUNT)):
                cache._write_quota_state_locked(
                    phase="prepared",
                    max_size_bytes=cache._configured_max_size_bytes,
                    physical_size_bytes=0,
                    store_count=0,
                )

    assert raised
    assert list(cache._objects_dir.glob("*/*.tmp-*")) == []
    assert list(cache._objects_dir.glob("*/.remove-*")) == []
    assert list(cache._locks_dir.glob(f"{depth_cache_module._QUOTA_STATE_NAME}.tmp-*")) == []
    assert list(cache._locks_dir.glob(".remove-*")) == []


def test_maximum_length_legacy_basename_uses_bounded_removal_quarantine(tmp_path) -> None:
    cache = DepthCache(tmp_path)
    legacy_name = f"{'x' * 246}.npy"
    assert len(legacy_name.encode("utf-8")) == 250
    legacy_path = cache.cache_dir / legacy_name
    legacy_path.write_bytes(b"legacy")

    cache.clear()

    assert not legacy_path.exists()
    assert list(cache.cache_dir.glob(".remove-*")) == []


@pytest.mark.parametrize("failure_stage", ["stat", "unlink"])
def test_failed_post_rename_cleanup_remains_accounted_and_recovers_on_restart(
    tmp_path,
    monkeypatch,
    failure_stage,
) -> None:
    cache = DepthCache(tmp_path)
    digest = _sha(f"quarantine-{failure_stage}")
    shard = cache._objects_dir / digest[:2]
    shard.mkdir()
    temp_path = shard / f".{digest}.npy.tmp-{'a' * 32}"
    payload = b"still-physical-after-failed-cleanup"
    temp_path.write_bytes(payload)
    real_stat = os.stat
    real_unlink = os.unlink
    failed = False

    def fail_quarantine_stat(path, *args, **kwargs):
        nonlocal failed
        if not failed and os.fspath(path).startswith(depth_cache_module._REMOVAL_QUARANTINE_PREFIX):
            failed = True
            raise OSError("simulated quarantine stat failure")
        return real_stat(path, *args, **kwargs)

    def fail_quarantine_unlink(path, *args, **kwargs):
        nonlocal failed
        if not failed and os.fspath(path).startswith(depth_cache_module._REMOVAL_QUARANTINE_PREFIX):
            failed = True
            raise OSError("simulated quarantine unlink failure")
        return real_unlink(path, *args, **kwargs)

    if failure_stage == "stat":
        monkeypatch.setattr(depth_cache_module.os, "stat", fail_quarantine_stat)
    else:
        monkeypatch.setattr(depth_cache_module.os, "unlink", fail_quarantine_unlink)

    stats = cache.stats()

    assert failed
    quarantines = list(shard.glob(".remove-*"))
    assert len(quarantines) == 1
    assert quarantines[0].read_bytes() == payload
    assert stats["size_gb"] == pytest.approx(len(payload) / (1024**3))

    monkeypatch.setattr(depth_cache_module.os, "stat", real_stat)
    monkeypatch.setattr(depth_cache_module.os, "unlink", real_unlink)
    restarted = DepthCache(tmp_path)
    assert list(shard.glob(".remove-*")) == []
    with restarted._locked_shards(range(depth_cache_module._LOCK_SHARD_COUNT)):
        state = restarted._read_quota_state_locked()
    assert state is not None
    assert state.phase == "clean"
    assert state.physical_size_bytes == 0


def test_identity_mismatched_quarantine_is_counted_but_not_deleted(tmp_path) -> None:
    cache = DepthCache(tmp_path)
    digest = _sha("unknown-quarantine")
    shard = cache._objects_dir / digest[:2]
    shard.mkdir()
    quarantine = shard / f"{depth_cache_module._REMOVAL_QUARANTINE_PREFIX}{'0' * 64}-{'a' * 32}"
    payload = b"identity-mismatched"
    quarantine.write_bytes(payload)

    stats = cache.stats()

    assert quarantine.read_bytes() == payload
    assert stats["entry_count"] == 0
    assert stats["size_gb"] == pytest.approx(len(payload) / (1024**3))


def test_repopulated_fixed_name_keeps_quarantine_and_prepared_accounting(tmp_path, monkeypatch) -> None:
    cache = DepthCache(tmp_path)
    identity = _identity(input_label="repopulated-name")
    assert cache.store(identity, np.ones((2, 2), dtype=np.float32))
    pointer_path, _, _ = _entry_paths(cache, identity)
    sentinel = b"replacement-pointer"
    real_stat = os.stat
    repopulated = False

    def repopulate_after_quarantine_stat(path, *args, **kwargs):
        nonlocal repopulated
        result = real_stat(path, *args, **kwargs)
        if not repopulated and os.fspath(path).startswith(depth_cache_module._REMOVAL_QUARANTINE_PREFIX):
            pointer_path.write_bytes(sentinel)
            repopulated = True
        return result

    monkeypatch.setattr(depth_cache_module.os, "stat", repopulate_after_quarantine_stat)
    cache.clear()

    assert repopulated
    assert pointer_path.read_bytes() == sentinel
    quarantines = list(pointer_path.parent.glob(".remove-*"))
    assert len(quarantines) == 1
    with cache._locked_shards(range(depth_cache_module._LOCK_SHARD_COUNT)):
        state = cache._read_quota_state_locked()
    assert state is not None
    assert state.phase == "prepared"

    monkeypatch.setattr(depth_cache_module.os, "stat", real_stat)
    restarted = DepthCache(tmp_path)
    assert restarted.stats()["entry_count"] == 0
    assert not pointer_path.exists()
    assert list(pointer_path.parent.glob(".remove-*")) == []


def test_post_validation_hardlink_prevents_false_successful_unlink(tmp_path, monkeypatch) -> None:
    directory = tmp_path / "unlink"
    directory.mkdir()
    victim = directory / "victim"
    victim.write_bytes(b"verified-bytes")
    expected = victim.stat()
    descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    real_unlink = os.unlink
    linked = False

    def link_back_before_unlink(path, *args, **kwargs):
        nonlocal linked
        if not linked and os.fspath(path).startswith(depth_cache_module._REMOVAL_QUARANTINE_PREFIX):
            os.link(
                path,
                victim.name,
                src_dir_fd=descriptor,
                dst_dir_fd=descriptor,
                follow_symlinks=False,
            )
            linked = True
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(depth_cache_module.os, "unlink", link_back_before_unlink)
    try:
        removed = DepthCache._unlink_matching_inode(descriptor, victim.name, expected)
    finally:
        os.close(descriptor)

    assert linked
    assert not removed
    assert victim.read_bytes() == b"verified-bytes"


def test_projection_corrupt_pointer_restore_has_exact_clean_quota_ledger(tmp_path) -> None:
    cache = DepthCache(tmp_path)
    identity = _identity(input_label="projection-restore")
    depth = np.arange(12, dtype=np.float32).reshape(3, 4)
    assert cache.store(identity, depth)
    pointer_path, _, pointer = _entry_paths(cache, identity)
    pointer["config_fingerprint_sha256"] = _sha("wrong-config")
    _rewrite_pointer(pointer_path, pointer)

    assert cache.store(identity, depth)
    restored = cache.get(identity)
    assert restored is not None
    np.testing.assert_array_equal(restored, depth)

    with cache._locked_shards(range(depth_cache_module._LOCK_SHARD_COUNT)):
        state = cache._read_quota_state_locked()
    assert state is not None
    assert state.phase == "clean"
    assert state.physical_size_bytes == cache._physical_size_bytes()


def test_get_removes_projection_mismatch_and_commits_exact_ledger(tmp_path) -> None:
    cache = DepthCache(tmp_path)
    identity = _identity(input_label="projection-reject")
    assert cache.store(identity, np.ones((2, 2), dtype=np.float32))
    pointer_path, object_path, pointer = _entry_paths(cache, identity)
    pointer["dependency_lock_sha256"] = _sha("wrong-lock")
    _rewrite_pointer(pointer_path, pointer)

    assert cache.get(identity) is None
    assert not pointer_path.exists()
    assert not object_path.exists()
    with cache._locked_shards(range(depth_cache_module._LOCK_SHARD_COUNT)):
        state = cache._read_quota_state_locked()
    assert state is not None
    assert state.phase == "clean"
    assert state.physical_size_bytes == cache._physical_size_bytes() == 0


def test_rejected_pointer_replacement_before_scan_is_preserved(tmp_path, monkeypatch) -> None:
    cache = DepthCache(tmp_path)
    identity = _identity(input_label="rejected-pointer-race")
    assert cache.store(identity, np.ones((2, 2), dtype=np.float32))
    pointer_path, _, pointer = _entry_paths(cache, identity)
    pointer["config_fingerprint_sha256"] = _sha("wrong-config")
    _rewrite_pointer(pointer_path, pointer)
    replacement = pointer_path.with_name("replacement.json")
    replacement.write_bytes(b"{}")
    real_scan = cache._scan_cache_locked
    swapped = False

    def replace_before_scan():
        nonlocal swapped
        if not swapped:
            os.replace(replacement, pointer_path)
            swapped = True
        return real_scan()

    monkeypatch.setattr(cache, "_scan_cache_locked", replace_before_scan)

    assert cache.get(identity) is None
    assert swapped
    assert pointer_path.read_bytes() == b"{}"


def test_long_lived_instance_cleans_abandoned_quota_temp_on_next_load(tmp_path) -> None:
    cache = DepthCache(tmp_path)
    quota_temp = cache._locks_dir / f"{depth_cache_module._QUOTA_STATE_NAME}.tmp-{'b' * 32}"
    quota_temp.write_bytes(b"abandoned")

    assert cache.store(_identity(input_label="quota-temp-peer"), np.ones((2, 2), dtype=np.float32))
    assert not quota_temp.exists()
