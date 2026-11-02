"""Closed local-MinIO resource identity without network or credential discovery."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

import pytest

from transformation_portal.orchestrator import artifact_store
from transformation_portal.orchestrator.artifact_store import s3 as s3_module
from transformation_portal.orchestrator.artifact_store.s3 import S3ArtifactStore

pytestmark = pytest.mark.unit


@pytest.fixture(name="client")
def fixture_client(monkeypatch: pytest.MonkeyPatch) -> Iterator[Any]:
    """Create an actual client using fake explicit credentials; never send I/O."""
    boto3 = pytest.importorskip("boto3")
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", "true")
    instance = boto3.client(
        "s3",
        endpoint_url="http://127.0.0.1:9",
        region_name="us-east-1",
        aws_access_key_id="private-test-access-key",
        aws_secret_access_key="private-test-secret-key",
    )
    yield instance
    instance.close()


def _store(root: Path, client: Any, *, endpoint: str = "http://127.0.0.1:9") -> S3ArtifactStore:
    return S3ArtifactStore(
        bucket="private-pilot",
        endpoint_url=endpoint,
        region_name="us-east-1",
        client=client,
        photographic_publication_root=root,
    )


def test_initialized_default_client_has_detached_secret_free_identity(tmp_path: Path, client: Any) -> None:
    store = _store(tmp_path, client)
    identity = store.native_resource_identity()
    assert identity is not None
    assert identity["root"] == str(tmp_path.resolve())
    assert identity["client"]["client_id"] == id(client)
    assert identity["client"]["total_max_attempts"] == 5
    assert identity["client"]["max_pool_connections"] == 10
    assert store.native_publication_root == tmp_path.resolve()
    assert store.native_resource_identity() == identity
    serialized = json.dumps(identity)
    assert "private-test-access-key" not in serialized
    assert "private-test-secret-key" not in serialized
    identity["client"]["max_pool_connections"] = 999
    assert store.native_resource_identity()["client"]["max_pool_connections"] == 10


def test_uninitialized_or_unmapped_store_never_creates_client_or_directory(
    tmp_path: Path, client: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden() -> None:
        pytest.fail("resource inspection must not load or initialize boto3")

    monkeypatch.setattr(s3_module, "_load_boto3", forbidden)
    store = _store(tmp_path, None)
    assert store.native_resource_identity() is None
    assert S3ArtifactStore(bucket="ordinary", client=client).native_resource_identity() is None
    missing = tmp_path / "missing"
    assert _store(missing, client).native_resource_identity() is None
    assert not missing.exists()
    assert _store(Path("relative"), client).native_resource_identity() is None
    assert _store(tmp_path, object()).native_resource_identity() is None


@pytest.mark.parametrize(
    "endpoint",
    [
        "https://s3.amazonaws.com:443",
        "http://192.168.1.2:9000",
        "http://localhost:9",
        "http://127.0.0.1",
        "http://user:secret@127.0.0.1:9",
        "http://127.0.0.1:9/bucket",
        "http://127.0.0.1:9?token=secret",
        "http://127.0.0.1:9#fragment",
        "http://127.0.0.1:10",
        "http://127.0.0.1:99999",
        "http://127.0.0.1:0",
        "http://127.0.0.1:9\n",
        "file://127.0.0.1:9",
    ],
)
def test_unsupported_or_mismatched_endpoint_falls_back(tmp_path: Path, client: Any, endpoint: str) -> None:
    assert _store(tmp_path, client, endpoint=endpoint).native_resource_identity() is None


def test_actual_endpoint_drift_falls_back(tmp_path: Path, client: Any) -> None:
    store = _store(tmp_path, client)
    client._endpoint.host = "https://remote.example:443"
    assert store.native_resource_identity() is None


def test_directory_identity_and_symlink_are_not_rebound(tmp_path: Path, client: Any) -> None:
    root = tmp_path / "host-data"
    root.mkdir()
    store = _store(root, client)
    assert store.native_resource_identity() is not None
    root.rename(tmp_path / "retained-host-data")
    root.mkdir()
    assert store.native_publication_root is None
    assert store.native_resource_identity() is None
    alias = tmp_path / "alias"
    alias.symlink_to(root, target_is_directory=True)
    assert _store(alias, client).native_resource_identity() is None


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("connect_timeout", float("nan")),
        ("read_timeout", float("inf")),
        ("connect_timeout", True),
        ("read_timeout", 0),
        ("read_timeout", 61),
        ("max_pool_connections", 11),
        ("max_pool_connections", True),
        ("retries", {"mode": "adaptive"}),
        ("retries", {"mode": "legacy", "total_max_attempts": 11}),
        ("retries", {"mode": "legacy", "total_max_attempts": True}),
        ("retries", {"mode": "legacy", "max_attempts": 2}),
        ("retries", {"mode": "standard", "extra": []}),
        ("s3", {"addressing_style": "virtual"}),
        ("s3", {"use_accelerate_endpoint": True}),
        ("s3", {"payload_signing_enabled": "false"}),
        ("signature_version", "s3"),
        ("parameter_validation", False),
        ("request_checksum_calculation", "unknown"),
        ("response_checksum_validation", "unknown"),
        ("client_context_params", {"arbitrary": "value"}),
        ("endpoint_discovery_enabled", True),
        ("tcp_keepalive", 1),
        ("user_agent", "x" * 1025),
        ("user_agent_extra", "x" * 1025),
        ("user_agent_appid", "x" * 1025),
        ("region_name", "x" * 65),
        ("request_min_compression_size_bytes", 2**32),
    ],
)
def test_unmeasured_sdk_configuration_falls_back(tmp_path: Path, client: Any, name: str, value: Any) -> None:
    setattr(client.meta.config, name, value)
    assert _store(tmp_path, client).native_resource_identity() is None


def test_pinned_config_and_client_replacements_change_identity(tmp_path: Path, client: Any) -> None:
    store = _store(tmp_path, client)
    original = store.native_resource_identity()
    client.meta.config.user_agent_extra = "bounded-extra"
    assert store.native_resource_identity() != original
    client.meta.config.retries = {"mode": "standard", "total_max_attempts": 2}
    changed = store.native_resource_identity()
    assert changed is not None and changed != original
    client.meta.config.s3 = {"addressing_style": "path", "payload_signing_enabled": True}
    settings = store.native_resource_identity()
    assert settings is not None
    settings["client"]["s3"]["addressing_style"] = "virtual"
    assert store.native_resource_identity()["client"]["s3"]["addressing_style"] == "path"
    boto3 = pytest.importorskip("boto3")
    replacement = boto3.client(
        "s3",
        endpoint_url="http://127.0.0.1:9",
        region_name="us-east-1",
        aws_access_key_id="different-private-key",
        aws_secret_access_key="different-private-secret",
    )
    try:
        store._client = replacement
        assert store.native_resource_identity() != original
    finally:
        replacement.close()


def test_actual_proxy_crt_and_session_config_are_unqualified(
    tmp_path: Path, client: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _store(tmp_path, client)
    compat = pytest.importorskip("botocore.compat")

    monkeypatch.setattr(compat, "HAS_CRT", True)
    assert store.native_resource_identity() is None
    monkeypatch.setattr(compat, "HAS_CRT", False)
    session = client._endpoint.http_session
    monkeypatch.setattr(session._proxy_config, "proxy_url_for", lambda url: "http://user:secret@proxy.invalid")
    assert store.native_resource_identity() is None
    monkeypatch.setattr(session._proxy_config, "proxy_url_for", lambda url: None)
    session._max_pool_connections = 50
    assert store.native_resource_identity() is None


def test_unrecognized_or_unbounded_supplied_sdk_options_fall_back(tmp_path: Path, client: Any) -> None:
    store = _store(tmp_path, client)
    client.meta.config._user_provided_options["unrecognized_option"] = "bounded"
    assert store.native_resource_identity() is None
    del client.meta.config._user_provided_options["unrecognized_option"]
    client.meta.config._user_provided_options["user_agent_extra"] = "x" * 1025
    assert store.native_resource_identity() is None


def test_factory_wires_optional_host_capacity_root_only_for_s3(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TP_ARTIFACT_STORE", "s3")
    monkeypatch.setenv("TP_ARTIFACT_BUCKET", "private-pilot")
    monkeypatch.setenv("TP_ARTIFACT_ENDPOINT_URL", "http://127.0.0.1:9")
    monkeypatch.setenv("TP_PHOTO_REVIEW_S3_PUBLICATION_ROOT", str(tmp_path))
    artifact_store.reset_singleton()
    try:
        store = artifact_store.get_artifact_store()
        assert type(store) is S3ArtifactStore
        assert store.native_publication_root == tmp_path.resolve()
        assert store.native_resource_identity() is None
        assert store._client is None
        artifact_store.reset_singleton()
        missing = tmp_path / "not-created"
        monkeypatch.setenv("TP_PHOTO_REVIEW_S3_PUBLICATION_ROOT", str(missing))
        assert artifact_store.get_artifact_store().native_publication_root is None
        assert not missing.exists()
        artifact_store.reset_singleton()
        monkeypatch.setenv("TP_ARTIFACT_STORE", "local")
        monkeypatch.setenv("TP_ARTIFACT_LOCAL_ROOT", str(tmp_path / "local"))
        local = artifact_store.get_artifact_store()
        assert local.backend == "local"
        assert local.root_dir == (tmp_path / "local").resolve()
    finally:
        artifact_store.reset_singleton()
