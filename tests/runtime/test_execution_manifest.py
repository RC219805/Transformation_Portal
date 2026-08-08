"""Behavioral coverage for run-level execution manifests."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

from transformation_portal.runtime import execution_manifest as manifest_module
from transformation_portal.runtime.execution_manifest import (
    EnvironmentInfo,
    ExecutionManifest,
    ManifestBuilder,
    _hash_dict,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def environment() -> EnvironmentInfo:
    """Return stable environment metadata for serialization tests."""
    return EnvironmentInfo(
        python_version="3.12.test",
        platform="test-platform",
        hostname="test-host",
        user="test-user",
        cwd="/workspace",
    )


def _manifest(environment: EnvironmentInfo) -> ExecutionManifest:
    node_hashes = ["node-b", "node-a"]
    return ExecutionManifest(
        run_id="run-test",
        node_hashes=node_hashes,
        root_hash=_hash_dict({"nodes": sorted(node_hashes)}),
        created_at="2026-08-08T12:00:00+00:00",
        duration_seconds=2.5,
        environment=environment,
        metadata={"preset": "audit"},
    )


def test_hash_dict_is_canonical_and_content_sensitive() -> None:
    """Dictionary insertion order does not affect the manifest digest."""
    first = _hash_dict({"outer": {"b": 2, "a": 1}, "items": [3, 2, 1]})
    reordered = _hash_dict({"items": [3, 2, 1], "outer": {"a": 1, "b": 2}})
    changed = _hash_dict({"outer": {"b": 2, "a": 1}, "items": [1, 2, 3]})

    assert first == "0df0f9efb0df20f199431be3cd3b7f366e4454c6993d504e57bd157aac938935"
    assert first == reordered
    assert first != changed
    assert len(first) == 64


@pytest.mark.parametrize(
    ("user", "username", "expected_user"),
    [
        ("primary-user", "fallback-user", "primary-user"),
        (None, "windows-user", "windows-user"),
        (None, None, "unknown"),
    ],
)
def test_environment_capture_records_platform_and_user_fallbacks(
    monkeypatch: pytest.MonkeyPatch,
    user: str | None,
    username: str | None,
    expected_user: str,
) -> None:
    """Environment capture is explicit and supports Unix and Windows user names."""
    monkeypatch.setattr(manifest_module.sys, "version", "3.12.test")
    monkeypatch.setattr(manifest_module.platform, "platform", lambda: "test-platform")
    monkeypatch.setattr(manifest_module.platform, "node", lambda: "test-host")
    monkeypatch.setattr(manifest_module.os, "getcwd", lambda: "/test/workspace")

    for name, value in (("USER", user), ("USERNAME", username)):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)

    assert EnvironmentInfo.capture() == EnvironmentInfo(
        python_version="3.12.test",
        platform="test-platform",
        hostname="test-host",
        user=expected_user,
        cwd="/test/workspace",
    )


def test_manifest_serialization_supports_pretty_and_compact_json(environment: EnvironmentInfo) -> None:
    """The portable manifest payload remains stable across JSON formatting modes."""
    manifest = _manifest(environment)

    payload = manifest.to_dict()
    pretty = manifest.to_json()
    compact = manifest.to_json(pretty=False)

    assert payload == {
        "version": "1.0",
        "run_id": "run-test",
        "node_hashes": ["node-b", "node-a"],
        "root_hash": manifest.root_hash,
        "created_at": "2026-08-08T12:00:00+00:00",
        "duration_seconds": 2.5,
        "environment": {
            "python_version": "3.12.test",
            "platform": "test-platform",
            "hostname": "test-host",
            "user": "test-user",
            "cwd": "/workspace",
        },
        "metadata": {"preset": "audit"},
    }
    assert "\n" in pretty
    assert "\n" not in compact
    assert json.loads(pretty) == payload
    assert json.loads(compact) == payload


def test_manifest_save_load_and_root_hash_tamper_detection(
    tmp_path: Path,
    environment: EnvironmentInfo,
) -> None:
    """A saved manifest round-trips and detects altered node provenance."""
    manifest = _manifest(environment)
    path = tmp_path / "execution-manifest.json"

    manifest.save(path)
    loaded = ExecutionManifest.load(path)

    assert path.read_text() == manifest.to_json()
    assert loaded == manifest
    assert loaded.verify_root_hash() is True

    loaded.node_hashes.append("tampered-node")
    assert loaded.verify_root_hash() is False


def test_manifest_load_supplies_backward_compatible_optional_defaults(
    tmp_path: Path,
    environment: EnvironmentInfo,
) -> None:
    """Older payloads without optional metadata still load deterministically."""
    payload = _manifest(environment).to_dict()
    payload.pop("duration_seconds")
    payload.pop("metadata")
    payload["environment"] = {"python_version": "legacy-python"}
    path = tmp_path / "legacy-manifest.json"
    path.write_text(json.dumps(payload))

    loaded = ExecutionManifest.load(path)

    assert loaded.duration_seconds == 0
    assert loaded.metadata == {}
    assert loaded.environment == EnvironmentInfo(
        python_version="legacy-python",
        platform="",
        hostname="",
        user="",
        cwd="",
    )


def test_manifest_builder_tracks_run_duration_and_metadata(
    monkeypatch: pytest.MonkeyPatch,
    environment: EnvironmentInfo,
) -> None:
    """A started builder uses the tracked run ID and elapsed duration."""
    timestamps = iter((100.0, 103.25))
    monkeypatch.setattr(manifest_module.time, "time", lambda: next(timestamps))
    monkeypatch.setattr(manifest_module.EnvironmentInfo, "capture", lambda: environment)
    builder = ManifestBuilder()

    builder.start("tracked-run")
    manifest = builder.build(["node-z", "node-a"], metadata={"quality": "golden"})

    assert manifest.run_id == "tracked-run"
    assert manifest.node_hashes == ["node-z", "node-a"]
    assert manifest.duration_seconds == pytest.approx(3.25)
    assert manifest.environment == environment
    assert manifest.metadata == {"quality": "golden"}
    assert datetime.fromisoformat(manifest.created_at).tzinfo is not None
    assert manifest.verify_root_hash() is True


def test_manifest_builder_supports_explicit_and_generated_run_ids(
    monkeypatch: pytest.MonkeyPatch,
    environment: EnvironmentInfo,
) -> None:
    """Callers may override the run ID or rely on the timestamp fallback."""
    monkeypatch.setattr(manifest_module.time, "time", lambda: 1234.9)
    monkeypatch.setattr(manifest_module.EnvironmentInfo, "capture", lambda: environment)
    builder = ManifestBuilder()
    builder.start("tracked-run")

    explicit = builder.build([], run_id="explicit-run", metadata={})
    generated = ManifestBuilder().build([])

    assert explicit.run_id == "explicit-run"
    assert explicit.duration_seconds == 0.0
    assert explicit.metadata == {}
    assert generated.run_id == "run_1234"
    assert generated.duration_seconds == 0.0
    assert generated.metadata == {}
    assert explicit.root_hash == generated.root_hash


def test_manifest_builder_builds_from_dag_and_rejects_missing_dag(
    monkeypatch: pytest.MonkeyPatch,
    environment: EnvironmentInfo,
) -> None:
    """DAG manifests use every node key and fail clearly without a DAG."""

    class FakeDAG:
        nodes = {"node-b": object(), "node-a": object()}

    monkeypatch.setattr(manifest_module.EnvironmentInfo, "capture", lambda: environment)
    manifest = ManifestBuilder(FakeDAG()).build_from_dag(
        run_id="dag-run",
        metadata={"source": "dag"},
    )

    assert manifest.run_id == "dag-run"
    assert manifest.node_hashes == ["node-b", "node-a"]
    assert manifest.metadata == {"source": "dag"}
    assert manifest.verify_root_hash() is True

    with pytest.raises(ValueError, match="No DAG provided to ManifestBuilder"):
        ManifestBuilder().build_from_dag()
