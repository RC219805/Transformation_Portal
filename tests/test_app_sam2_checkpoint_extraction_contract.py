#!/usr/bin/env python3
"""Phase 2C extraction-readiness contract for app SAM2 checkpoint helpers."""

from __future__ import annotations

import hashlib
import importlib
import os
from pathlib import Path

import pytest

from transformation_portal.portal import sam2_checkpoint_security

pytestmark = [pytest.mark.unit, pytest.mark.security]

orchestrator_app = importlib.import_module("app")


_PHASE_2C_LEGACY_NAMES = (
    "ManagedSam2CheckpointValidationResult",
    "_ManagedSam2ChecksumCacheEntry",
    "_Sam2CacheKey",
    "_ManagedSam2BoundedChecksumCache",
    "_managed_sam2_reason_message",
    "_managed_sam2_checksum_cache_key",
    "_clear_managed_sam2_checksum_cache",
    "_hash_file_sha256",
    "_cached_managed_sam2_checksum_result",
    "_resolve_managed_sam2_checkpoint_validation",
    "_validate_managed_sam2_checkpoint_path",
)


@pytest.fixture(autouse=True)
def _clear_managed_sam2_checksum_cache() -> None:
    orchestrator_app._clear_managed_sam2_checksum_cache()
    yield
    orchestrator_app._clear_managed_sam2_checksum_cache()


def _realpath(path: Path) -> Path:
    return Path(os.path.realpath(path))


def test_phase_2c_legacy_sam2_checkpoint_helpers_remain_available_from_app() -> None:
    for helper_name in _PHASE_2C_LEGACY_NAMES:
        assert getattr(orchestrator_app, helper_name) is not None


def test_phase_2c_app_models_are_extracted_module_models() -> None:
    assert orchestrator_app.ManagedSam2CheckpointValidationResult is (
        sam2_checkpoint_security.ManagedSam2CheckpointValidationResult
    )
    assert orchestrator_app._ManagedSam2ChecksumCacheEntry is sam2_checkpoint_security._ManagedSam2ChecksumCacheEntry
    assert orchestrator_app._Sam2CacheKey is sam2_checkpoint_security._Sam2CacheKey
    assert orchestrator_app._ManagedSam2BoundedChecksumCache is (sam2_checkpoint_security._ManagedSam2BoundedChecksumCache)


def test_phase_2c_managed_sam2_reason_messages_preserve_codes() -> None:
    assert (
        orchestrator_app._managed_sam2_reason_message("checkpoint_file_too_large")
        == "SAM2 checkpoint path exceeds checksum verification size limit"
    )
    assert orchestrator_app._managed_sam2_reason_message("invalid_path_value") == "Invalid path value"
    assert orchestrator_app._managed_sam2_reason_message("path_outside_allowed_roots") == "Path outside allowed roots"
    assert orchestrator_app._managed_sam2_reason_message("untrusted_checkpoint_path") == "SAM2 checkpoint path is not trusted"
    assert orchestrator_app._managed_sam2_reason_message("unknown_reason") == "Invalid path value"


def test_phase_2c_repo_controlled_missing_checkpoint_path_remains_accepted() -> None:
    validation = orchestrator_app._resolve_managed_sam2_checkpoint_validation("./models/sam2/sam2.1_hiera_large.pt")

    assert validation.reason is None
    assert validation.normalized_path is not None
    assert validation.normalized_path.endswith("models/sam2/sam2.1_hiera_large.pt")


def test_phase_2c_invalid_path_reason_is_preserved(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(orchestrator_app, "ALLOWED_INPUT_ROOTS", [_realpath(tmp_path)])

    validation = orchestrator_app._resolve_managed_sam2_checkpoint_validation("bad\x00path")

    assert validation.normalized_path is None
    assert validation.reason == "invalid_path_value"
    with pytest.raises(orchestrator_app._PortalValidationReasonError) as exc_info:
        orchestrator_app._validate_managed_sam2_checkpoint_path("bad\x00path")
    assert exc_info.value.reason == "invalid_path_value"


def test_phase_2c_outside_root_reason_is_preserved(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    allowed_root = _realpath(tmp_path / "allowed")
    outside_root = _realpath(tmp_path / "outside")
    allowed_root.mkdir()
    outside_root.mkdir()
    checkpoint_path = outside_root / "sam2-outside.pt"
    checkpoint_path.write_bytes(b"outside")
    monkeypatch.setattr(orchestrator_app, "ALLOWED_INPUT_ROOTS", [allowed_root])

    validation = orchestrator_app._resolve_managed_sam2_checkpoint_validation(str(checkpoint_path))

    assert validation.normalized_path is None
    assert validation.reason == "path_outside_allowed_roots"


def test_phase_2c_external_checkpoint_requires_trusted_digest(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    input_root = _realpath(tmp_path)
    checkpoint_path = input_root / "sam2-governed.pt"
    checkpoint_bytes = b"trusted checkpoint bytes"
    checkpoint_path.write_bytes(checkpoint_bytes)
    digest = hashlib.sha256(checkpoint_bytes).hexdigest()
    monkeypatch.setattr(orchestrator_app, "ALLOWED_INPUT_ROOTS", [input_root])
    monkeypatch.setattr(orchestrator_app, "MANAGED_SAM2_TRUSTED_SHA256", {digest})

    trusted = orchestrator_app._resolve_managed_sam2_checkpoint_validation(str(checkpoint_path))

    assert trusted.reason is None
    assert trusted.normalized_path == str(checkpoint_path)

    orchestrator_app._clear_managed_sam2_checksum_cache()
    monkeypatch.setattr(orchestrator_app, "MANAGED_SAM2_TRUSTED_SHA256", set())

    untrusted = orchestrator_app._resolve_managed_sam2_checkpoint_validation(str(checkpoint_path))

    assert untrusted.normalized_path is None
    assert untrusted.reason == "untrusted_checkpoint_path"


def test_phase_2c_default_managed_allowlist_trusts_canonical_sam21_large_digest(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    input_root = _realpath(tmp_path)
    checkpoint_path = input_root / "sam2-canonical.pt"
    checkpoint_path.write_bytes(b"canonical placeholder bytes")
    canonical_digest = "2647878d5dfa5098f2f8649825738a9345572bae2d4350a2468587ece47dd318"
    monkeypatch.setattr(orchestrator_app, "ALLOWED_INPUT_ROOTS", [input_root])
    monkeypatch.setattr(orchestrator_app, "_hash_file_sha256", lambda path, chunk_size=1024 * 1024: canonical_digest)

    validation = orchestrator_app._resolve_managed_sam2_checkpoint_validation(str(checkpoint_path))

    assert canonical_digest in orchestrator_app.MANAGED_SAM2_TRUSTED_SHA256
    assert validation.reason is None
    assert validation.normalized_path == str(checkpoint_path)


def test_phase_2c_oversized_checkpoint_reason_is_preserved(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    input_root = _realpath(tmp_path)
    checkpoint_path = input_root / "sam2-oversized.pt"
    checkpoint_path.write_bytes(b"oversized")
    monkeypatch.setattr(orchestrator_app, "ALLOWED_INPUT_ROOTS", [input_root])
    monkeypatch.setattr(orchestrator_app, "MANAGED_SAM2_CHECKSUM_MAX_BYTES", 1)

    validation = orchestrator_app._resolve_managed_sam2_checkpoint_validation(str(checkpoint_path))

    assert validation.normalized_path is None
    assert validation.reason == "checkpoint_file_too_large"
    with pytest.raises(orchestrator_app._PortalValidationReasonError) as exc_info:
        orchestrator_app._validate_managed_sam2_checkpoint_path(str(checkpoint_path))
    assert exc_info.value.reason == "checkpoint_file_too_large"


def test_phase_2c_app_cached_checksum_wrapper_translates_module_error(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.pt"

    with pytest.raises(orchestrator_app._PortalValidationReasonError) as exc_info:
        orchestrator_app._cached_managed_sam2_checksum_result(missing_path)

    assert exc_info.value.reason == "invalid_path_value"


def test_phase_2c_checksum_cache_key_shape_tracks_file_identity(tmp_path: Path) -> None:
    checkpoint_path = _realpath(tmp_path / "sam2-keyed.pt")
    checkpoint_path.write_bytes(b"checkpoint")

    cache_key = orchestrator_app._managed_sam2_checksum_cache_key(checkpoint_path)

    assert isinstance(cache_key, orchestrator_app._Sam2CacheKey)
    assert cache_key.path == str(checkpoint_path)
    assert cache_key.size == len(b"checkpoint")
    assert isinstance(cache_key.mtime_ns, int)
    assert isinstance(cache_key.dev, int)
    assert isinstance(cache_key.ino, int)
    assert isinstance(cache_key.ctime_ns, int)


def test_phase_2c_checksum_cache_reuses_hash_results_and_clear_resets_state(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    input_root = _realpath(tmp_path)
    checkpoint_path = input_root / "sam2-cached.pt"
    checkpoint_bytes = b"cached checkpoint bytes"
    checkpoint_path.write_bytes(checkpoint_bytes)
    digest = hashlib.sha256(checkpoint_bytes).hexdigest()
    monkeypatch.setattr(orchestrator_app, "ALLOWED_INPUT_ROOTS", [input_root])
    monkeypatch.setattr(orchestrator_app, "MANAGED_SAM2_TRUSTED_SHA256", {digest})

    hash_calls: list[Path] = []
    original_hash = orchestrator_app._hash_file_sha256

    def _counting_hash(path: Path, chunk_size: int = 1024 * 1024) -> str:
        hash_calls.append(path)
        return original_hash(path, chunk_size)

    monkeypatch.setattr(orchestrator_app, "_hash_file_sha256", _counting_hash)

    first = orchestrator_app._validate_managed_sam2_checkpoint_path(str(checkpoint_path))
    second = orchestrator_app._validate_managed_sam2_checkpoint_path(str(checkpoint_path))

    assert first == str(checkpoint_path)
    assert second == str(checkpoint_path)
    assert hash_calls == [checkpoint_path]
    assert len(orchestrator_app._MANAGED_SAM2_CHECKSUM_CACHE) == 1

    orchestrator_app._clear_managed_sam2_checksum_cache()

    assert len(orchestrator_app._MANAGED_SAM2_CHECKSUM_CACHE) == 0


def test_phase_2c_bounded_checksum_cache_preserves_fifo_eviction() -> None:
    cache = orchestrator_app._ManagedSam2BoundedChecksumCache(max_entries=2)
    entry = orchestrator_app._ManagedSam2ChecksumCacheEntry(digest="abc", reason=None)
    key1 = orchestrator_app._Sam2CacheKey("/path/a.pt", 100, 1000, 1, 1001, 2000)
    key2 = orchestrator_app._Sam2CacheKey("/path/b.pt", 200, 1000, 1, 1002, 2000)
    key3 = orchestrator_app._Sam2CacheKey("/path/c.pt", 300, 1000, 1, 1003, 2000)

    cache[key1] = entry
    cache[key2] = entry
    cache[key3] = entry

    assert key1 not in cache
    assert key2 in cache
    assert key3 in cache
    assert len(cache) == 2

    cache[key2] = orchestrator_app._ManagedSam2ChecksumCacheEntry(digest="updated", reason=None)
    assert len(cache) == 2
    assert cache[key2].digest == "updated"

    cache.clear()

    assert len(cache) == 0
    assert len(cache._insertion_order) == 0
