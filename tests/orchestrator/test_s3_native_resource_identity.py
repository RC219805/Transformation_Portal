"""Closed local-MinIO resource identity without network or credential discovery."""

from __future__ import annotations

import copy
import gc
import json
import weakref
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


@pytest.mark.parametrize("verify", [True, "external-ca"], ids=["default-ca", "external-ca"])
def test_https_client_keeps_ordinary_store_but_uses_conservative_profile(
    tmp_path: Path, client: Any, verify: bool | str
) -> None:
    boto3 = pytest.importorskip("boto3")
    assert _store(tmp_path, client).native_resource_identity() is not None
    endpoint = "https://127.0.0.1:9"
    tls_verify = str(tmp_path / "unqualified-ca-bundle.pem") if verify == "external-ca" else verify
    https_client = boto3.client(
        "s3",
        endpoint_url=endpoint,
        region_name="us-east-1",
        aws_access_key_id="private-test-access-key",
        aws_secret_access_key="private-test-secret-key",
        verify=tls_verify,
    )
    try:
        store = _store(tmp_path, https_client, endpoint=endpoint)
        assert store.native_resource_identity() is None
        assert store._client is https_client
        assert https_client.meta.endpoint_url == endpoint
        assert https_client._endpoint.http_session._verify == tls_verify
    finally:
        https_client.close()


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


def test_packaged_model_identity_is_recomputed_and_detects_in_place_shape_edits(tmp_path: Path, client: Any) -> None:
    store = _store(tmp_path, client)
    before = store.native_resource_identity()
    assert before is not None and set(before["client"]["model"]) == {"description_sha256", "operations_sha256"}
    assert store.native_resource_identity() == before
    body = client.meta.service_model._service_description["shapes"]["Body"]
    original = body["type"]
    try:
        body["type"] = "string"
        assert store.native_resource_identity() is None
    finally:
        body["type"] = original  # boto3's default session shares its loader cache.


@pytest.mark.parametrize("payload_name", ["ContentLength", "missing"])
def test_cached_effective_shape_edit_cannot_borrow_unchanged_backing_model(
    tmp_path: Path, client: Any, payload_name: str
) -> None:
    store = _store(tmp_path, client)
    assert store.native_resource_identity() is not None
    operation = client.meta.service_model.operation_model("GetObject")
    operation.output_shape.serialization["payload"] = payload_name
    assert store.native_resource_identity() is None


def test_actual_custom_nonstreaming_s3_model_is_unqualified_offline(tmp_path: Path, client: Any) -> None:
    boto3 = pytest.importorskip("boto3")
    payload = copy.deepcopy(client.meta.service_model._service_description)
    payload["shapes"]["GetObjectOutput"].pop("payload")
    payload["shapes"]["GetObjectOutput"]["members"]["Body"].pop("streaming")
    models = tmp_path / "custom-models"
    folder = models / "s3" / "2006-03-01"
    folder.mkdir(parents=True)
    (folder / "service-2.json").write_text(json.dumps(payload), encoding="utf-8")
    session = boto3.Session()
    loader = session._session.get_component("data_loader")
    loader.search_paths.insert(0, str(models))
    custom = session.client(
        "s3",
        endpoint_url="http://127.0.0.1:9",
        region_name="us-east-1",
        aws_access_key_id="private-offline-key",
        aws_secret_access_key="private-offline-secret",
    )
    try:
        assert type(custom).__module__ == "botocore.client"
        assert custom.meta.service_model.operation_model("GetObject").has_streaming_output is False
        assert _store(tmp_path, custom).native_resource_identity() is None
        # Removing the observable override after construction cannot hide the
        # actual already-loaded altered model behind normal search paths.
        loader.search_paths.remove(str(models))
        assert _store(tmp_path, custom).native_resource_identity() is None
    finally:
        custom.close()


def test_unbounded_actual_model_is_rejected_before_serialization(
    tmp_path: Path, client: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Unexpected:
        def to_dict(self):
            pytest.fail("Untrusted model protocols must not run")

    payload = client.meta.service_model._service_description
    payload["untrusted"] = Unexpected()
    try:
        assert _store(tmp_path, client).native_resource_identity() is None
    finally:
        del payload["untrusted"]
    assert s3_module._native_model_digest([0] * (s3_module._NATIVE_MODEL_NODES + 1)) is None
    cycle = {}
    cycle["self"] = cycle
    assert s3_module._native_model_digest(cycle) is None
    assert s3_module._native_model_digest({"large": "x" * (2 * 1024**2 + 1)}) is None
    monkeypatch.setattr(s3_module, "_NATIVE_MODEL_BYTES", 1)
    assert s3_module._native_model_digest({"small": "value"}) is None


def test_bounded_packaged_loader_refuses_oversized_model_before_json_decode(
    tmp_path: Path, client: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(s3_module, "_NATIVE_MODEL_BYTES", 16)
    # Keep the real graph check out of this one test: the packaged gzip reader
    # must independently refuse expansion beyond its finite byte bound.
    monkeypatch.setattr(s3_module, "_native_model_digest", lambda value: "a" * 64)
    assert _store(tmp_path, client).native_resource_identity() is None


def test_packaged_validation_models_do_not_wait_for_cyclic_gc(
    tmp_path: Path, client: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = s3_module._native_operation_models
    references = []

    def observed(model):
        if model is not client.meta.service_model:
            references.append(weakref.ref(model))
        return original(model)

    monkeypatch.setattr(s3_module, "_native_operation_models", observed)
    enabled = gc.isenabled()
    gc.disable()
    try:
        for _ in range(3):
            assert _store(tmp_path, client).native_resource_identity() is not None
        assert len(references) == 3 and all(reference() is None for reference in references)
    finally:
        if enabled:
            gc.enable()


def test_packaged_model_decoder_failure_retains_conservative_fallback(
    tmp_path: Path, client: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    def malformed(*args, **kwargs):
        raise RecursionError("Invalid bounded SDK model")

    monkeypatch.setattr(s3_module.json, "loads", malformed)
    assert _store(tmp_path, client).native_resource_identity() is None


def test_cached_raw_shape_member_limit_precedes_lazy_materialization(
    tmp_path: Path, client: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _store(tmp_path, client)
    assert store.native_resource_identity() is not None
    operation = client.meta.service_model.operation_model("GetObject")
    shape = operation.output_shape
    shape._shape_model = {"type": "structure", "members": {f"field-{index}": {"shape": "Body"} for index in range(257)}}
    shape.__dict__.pop("members", None)
    operation.__dict__.pop("has_streaming_output", None)

    def forbidden(*args, **kwargs):
        pytest.fail("Unbounded cached shape definitions must not materialize")

    monkeypatch.setattr(shape._shape_resolver, "resolve_shape_ref", forbidden)
    assert store.native_resource_identity() is None
