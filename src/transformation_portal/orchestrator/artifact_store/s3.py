"""S3-backed ``ArtifactStore``.

Activated by ``TP_ARTIFACT_STORE=s3`` plus the standard S3 env vars:

- ``TP_ARTIFACT_BUCKET`` (required)
- ``TP_ARTIFACT_PREFIX`` (default ``tp/artifacts``; the per-job
  layout is ``{prefix}/jobs/{job_id}/{relative_path}``)
- ``TP_ARTIFACT_ENDPOINT_URL`` (optional; lets the store target
  MinIO / LocalStack / R2 / any S3-compatible endpoint without
  changing client code)
- ``AWS_*`` for credentials (standard boto3 resolution chain)

Implementation notes:

- ``boto3`` is imported lazily so single-instance local deployments
  do not pay the dependency cost at import time. Construction
  validates the bucket name only; the first network call materializes
  the boto3 client.
- All boto3 calls are wrapped in ``asyncio.to_thread`` because
  ``boto3`` is fundamentally synchronous. The thread-pool model is
  acceptable for artifact I/O which is dominated by network latency
  rather than CPU.
- Fingerprinting checks ``HeadObject`` metadata, then computes SHA-256
  while streaming the body. Both GET metadata and observed bytes enforce
  ``ARTIFACT_FINGERPRINT_MAX_BYTES``; a larger replacement object cannot
  bypass the budget. Size disagreement reports an unavailable fingerprint.
- Object metadata sets ``Content-Type`` and presigned GET URLs can
  carry response-header overrides so signed delivery preserves the
  orchestrator's artifact response semantics.
- Readiness uses ``HeadBucket`` so ``/ready`` verifies bucket/access
  instead of only proving that the client can sign a URL.

The runtime dependency is part of the base package metadata and
lockfiles, but the module imports nothing eagerly, so local
deployments do not pay S3 client construction cost unless they set
``TP_ARTIFACT_STORE=s3``.
"""

from __future__ import annotations

import asyncio
import hashlib
import ipaddress
import json
import math
from collections import OrderedDict
from pathlib import Path, PurePosixPath
from typing import Any, AsyncIterator, List, Optional
from urllib.parse import urlsplit

from transformation_portal.ingest.canonical_json import dumps_json
from transformation_portal.orchestrator.artifact_store.base import (
    ArtifactNotFoundError,
    ArtifactObjectMetadata,
    ArtifactPathValidationError,
    ArtifactStore,
    ArtifactStoreError,
)
from transformation_portal.orchestrator.artifact_store.local import _normalize_job_id, _normalize_relative_path
from transformation_portal.portal.job_artifacts import ARTIFACT_FINGERPRINT_MAX_BYTES, _artifact_content_type

_CHUNK_BYTES = 1024 * 1024
_DEFAULT_PREFIX = "tp/artifacts"
_NATIVE_MODEL_BYTES = 4 * 1024**2
_NATIVE_MODEL_NODES = 65536


def _normalize_prefix(prefix: str) -> str:
    raw = str(prefix or _DEFAULT_PREFIX).strip().strip("/")
    if not raw or "\\" in raw or "\x00" in raw:
        raise ArtifactStoreError("invalid S3 artifact prefix")
    candidate = PurePosixPath(raw)
    if candidate.is_absolute() or any(part in {"", ".", ".."} for part in candidate.parts):
        raise ArtifactStoreError("invalid S3 artifact prefix")
    return candidate.as_posix()


def _content_type_for(relative_path: str) -> str:
    return _artifact_content_type(Path(relative_path))


def _load_boto3() -> Any:
    """Lazy boto3 import with an actionable error when the dep is missing."""
    try:
        import boto3  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - covered by env where boto3 is uninstalled
        raise ArtifactStoreError(
            "TP_ARTIFACT_STORE=s3 requires the 'boto3' package. "
            "Install it (added to base.in for the S3 deployment lane) or "
            "fall back to TP_ARTIFACT_STORE=local."
        ) from exc
    return boto3


def _native_loopback_endpoint(value: Any) -> str | None:
    """Recognize the closed local-MinIO pilot endpoint without DNS/network I/O."""
    if type(value) is not str or len(value) > 512 or any(ord(char) <= 32 or ord(char) == 127 for char in value):
        return None
    try:
        parsed = urlsplit(value)
        if (
            parsed.scheme != "http"
            or parsed.username is not None
            or parsed.password is not None
            or parsed.path not in {"", "/"}
            or parsed.query
            or parsed.fragment
            or parsed.hostname is None
            or parsed.port is None
            or parsed.port == 0
        ):
            return None
        if not ipaddress.ip_address(parsed.hostname).is_loopback:
            return None
        return value.rstrip("/")
    except ValueError:
        return None


def _native_config_digest(config: Any) -> str | None:
    """Bind finite SDK options, including in-place mutation of unused defaults."""
    from botocore.config import Config

    def scalar(value: Any) -> bool:
        if value is None or type(value) is bool:
            return True
        if type(value) is str:
            return len(value) <= 1024
        return type(value) in {int, float} and math.isfinite(value) and 0 <= value <= 2**31

    def option(value: Any) -> bool:
        return scalar(value) or (
            type(value) is dict
            and len(value) <= 16
            and all(type(key) is str and len(key) <= 64 and scalar(item) for key, item in value.items())
        )

    if type(config) is not Config or len(Config.OPTION_DEFAULTS) > 64:
        return None
    provided = config._user_provided_options
    if type(provided) is not dict or set(provided) - set(Config.OPTION_DEFAULTS):
        return None
    values = {name: getattr(config, name) for name in Config.OPTION_DEFAULTS}
    if not all(option(value) for value in (*values.values(), *provided.values())):
        return None
    encoded = dumps_json({"effective": values, "provided": provided}, sort_keys=True, allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _native_model_digest(payload: Any) -> str | None:
    """Reject unbounded/mutable-object graphs before model serialization."""
    nodes, text = 0, 0
    active: set[int] = set()

    def bounded(value: Any, depth: int = 0) -> bool:
        nonlocal nodes, text
        nodes += 1
        if nodes > _NATIVE_MODEL_NODES or depth > 24:
            return False
        if type(value) is str:
            text += len(value)
            return len(value) <= 262144 and text <= 2 * 1024**2
        if value is None or type(value) in {bool, int}:
            return type(value) is not int or value.bit_length() <= 64
        if type(value) is float:
            return math.isfinite(value)
        if type(value) not in {dict, OrderedDict, list} or len(value) > _NATIVE_MODEL_NODES or id(value) in active:
            return False
        active.add(id(value))
        try:
            if type(value) is list:
                return all(bounded(item, depth + 1) for item in value)
            return all(
                type(key) is str and bounded(key, depth + 1) and bounded(item, depth + 1) for key, item in value.items()
            )
        finally:
            active.remove(id(value))

    if not bounded(payload):
        return None
    encoded = dumps_json(payload, sort_keys=True, allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest() if len(encoded) <= _NATIVE_MODEL_BYTES else None


def _native_http_model(actual: Any, packaged: Any) -> dict[str, Any]:
    """Accept only the pinned S3 handler's exact cold or warm HTTP mapping."""
    if (
        type(actual) not in {dict, OrderedDict}
        or type(packaged) not in {dict, OrderedDict}
        or _native_model_digest(actual) is None
        or _native_model_digest(packaged) is None
    ):
        raise ValueError("Native S3 HTTP model exceeds its finite graph bound")
    cold = dict(packaged)
    warm = dict(cold)
    uri = cold.get("requestUri")
    if type(uri) is not str:
        raise ValueError("Native S3 packaged HTTP URI is invalid")
    bucket_path = "/{Bucket}"
    if uri.startswith(bucket_path):
        # botocore.handlers.remove_bucket_from_url_paths_from_model mutates
        # these two fields on first use. Normalize private copies only.
        warm["requestUri"] = uri[len(bucket_path) :]
        auth_path = uri.split("?")[0]
        warm["authPath"] = f"{auth_path}/" if auth_path == bucket_path else auth_path
    if actual != cold and actual != warm:
        raise ValueError("Native S3 HTTP model is not an exact packaged cold or warm state")
    return cold


def _native_operation_definition(actual: Any, packaged: Any) -> dict[str, Any]:
    if (
        type(actual) not in {dict, OrderedDict}
        or type(packaged) not in {dict, OrderedDict}
        or _native_model_digest(actual) is None
    ):
        raise ValueError("Native S3 operation definition exceeds its finite graph bound")
    return {**actual, "http": _native_http_model(actual.get("http"), packaged.get("http"))}


def _native_model_description(actual: Any, packaged: Any) -> dict[str, Any]:
    """Canonicalize only validated HTTP state, preserving all other authority."""
    if _native_model_digest(actual) is None or _native_model_digest(packaged) is None:
        raise ValueError("Native S3 service description exceeds its finite graph bound")
    operations, expected = actual.get("operations"), packaged.get("operations")
    if (
        type(operations) not in {dict, OrderedDict}
        or type(expected) not in {dict, OrderedDict}
        or len(operations) > 256
        or set(operations) != set(expected)
    ):
        raise ValueError("Native S3 service operation set differs from its packaged authority")
    return {
        **actual,
        "operations": {
            name: _native_operation_definition(definition, expected[name]) for name, definition in operations.items()
        },
    }


def _native_operation_models(model: Any, packaged_operations: Any) -> dict[str, Any]:
    """Include effective cached shape serialization, not only backing JSON."""
    from botocore.model import ListShape, MapShape, OperationModel, Shape, StringShape, StructureShape

    count = 0

    def shape_payload(shape: Any, depth: int = 0) -> Any:
        nonlocal count
        count += 1
        if count > _NATIVE_MODEL_NODES or depth > 16:
            raise ValueError("Native S3 operation shape graph exceeds its bound")
        if shape is None:
            return None
        if type(shape) not in {Shape, StringShape, StructureShape, ListShape, MapShape}:
            raise ValueError("Unsupported native S3 operation shape")
        definition = shape._shape_model
        if type(definition) not in {dict, OrderedDict}:
            raise ValueError("Unsupported native S3 shape definition")
        if type(shape) is StructureShape:
            members = definition.get("members", {})
            if type(members) not in {dict, OrderedDict} or len(members) > 256:
                raise ValueError("Native S3 shape definition has unbounded members")
        definition_identity = _native_model_digest(definition)
        if definition_identity is None:
            raise ValueError("Native S3 shape definition exceeds its finite graph bound")
        result = {
            "name": shape.name,
            "type": shape.type_name,
            "definition_sha256": definition_identity,
            "serialization": shape.serialization,
            "metadata": shape.metadata,
        }
        if type(shape) is StructureShape:
            if (
                type(shape.members) not in {dict, OrderedDict}
                or len(shape.members) > 256
                or any(type(name) is not str or len(name) > 1024 for name in shape.members)
            ):
                raise ValueError("Native S3 operation has unbounded members")
            result["members"] = {name: shape_payload(value, depth + 1) for name, value in shape.members.items()}
        elif type(shape) is ListShape:
            result["member"] = shape_payload(shape.member, depth + 1)
        elif type(shape) is MapShape:
            result.update(key=shape_payload(shape.key, depth + 1), value=shape_payload(shape.value, depth + 1))
        return result

    result = {}
    for name in ("GetObject", "HeadObject", "PutObject", "DeleteObject", "ListObjectsV2"):
        operation = model.operation_model(name)
        if type(operation) is not OperationModel:
            raise ValueError("Unsupported native S3 operation model")
        expected = packaged_operations[name]
        operation_identity = _native_model_digest(_native_operation_definition(operation._operation_model, expected))
        if operation_identity is None:
            raise ValueError("Native S3 operation definition exceeds its finite graph bound")
        input_shape = shape_payload(operation.input_shape)
        output_shape = shape_payload(operation.output_shape)
        result[name] = {
            "definition_sha256": operation_identity,
            "http": _native_http_model(operation.http, expected["http"]),
            "streaming_input": operation.has_streaming_input,
            "streaming_output": operation.has_streaming_output,
            "input": input_shape,
            "output": output_shape,
        }
    return result


def _native_service_model_identity(client: Any) -> dict[str, str] | None:
    """Bind actual S3 models to the verified packaged model and SDK extras."""
    import boto3
    from botocore.loaders import JSONFileLoader, Loader
    from botocore.model import ServiceModel

    loader, model = client._loader, client.meta.service_model
    if (
        type(loader) is not Loader
        or type(loader.file_loader) is not JSONFileLoader
        or loader._extras_types != Loader.BUILTIN_EXTRAS_TYPES
        or type(model) is not ServiceModel
        or model.api_version != "2006-03-01"
        or len(loader.search_paths) > 3
    ):
        return None
    builtin = Path(Loader.BUILTIN_DATA_PATH).resolve()
    packaged_paths = {builtin, (Path(boto3.__file__).resolve().parent / "data").resolve()}
    customer = Path(Loader.CUSTOMER_DATA_PATH).resolve()
    for value in loader.search_paths:
        if type(value) is not str or len(value) > 4096:
            return None
        path = Path(value).resolve()
        if path not in packaged_paths and not (path == customer and not path.exists()):
            return None
    if _native_model_digest(model._service_description) is None:
        return None

    class BoundedFileLoader(JSONFileLoader):
        def _load_file(self, full_path: str, open_method: Any) -> Any:
            path = Path(full_path)
            if not path.is_file():
                return None
            if path.resolve() != path or not path.is_relative_to(builtin):
                raise ValueError("Packaged S3 model path changed")
            with open_method(full_path, "rb") as stream:
                raw = stream.read(_NATIVE_MODEL_BYTES + 1)
            if len(raw) > _NATIVE_MODEL_BYTES:
                raise ValueError("Packaged S3 model exceeds its byte bound")
            return json.loads(raw.decode("utf-8"), object_pairs_hook=OrderedDict)

    expected_loader = Loader(
        extra_search_paths=[str(builtin)], file_loader=BoundedFileLoader(), include_default_search_paths=False
    )
    expected = expected_loader.load_service_model("s3", "service-2", api_version="2006-03-01")
    actual = _native_model_digest(_native_model_description(model._service_description, expected))
    if actual is None or actual != _native_model_digest(expected):
        return None
    effective = _native_model_digest(_native_operation_models(model, expected["operations"]))
    reference = ServiceModel(expected, service_name="s3")
    try:
        packaged_effective = _native_model_digest(_native_operation_models(reference, expected["operations"]))
    finally:
        # The fresh validation model caches operations that refer back to it.
        # Break only that private cycle; never alter the live client's caches.
        reference._instance_cache.clear()
    if effective is None or effective != packaged_effective:
        return None
    return {"description_sha256": actual, "operations_sha256": effective}


def _native_client_settings(client: Any) -> dict[str, Any] | None:
    """Read only the finite SDK path measured by the local-MinIO pilot.

    This is eligibility, not S3 configuration. Other clients retain their
    ordinary behavior and use the conservative photographic resource profile.
    """
    from botocore.client import BaseClient
    from botocore.compat import HAS_CRT
    from botocore.exceptions import BotoCoreError
    from botocore.httpsession import URLLib3Session

    if (
        not isinstance(client, BaseClient)
        or type(client).__module__ != "botocore.client"
        or type(client).__name__ != "S3"
        or client.meta.service_model.service_name != "s3"
        or HAS_CRT
    ):
        return None
    config, endpoint = client.meta.config, client._endpoint
    config_digest = _native_config_digest(config)
    session = endpoint.http_session
    url = _native_loopback_endpoint(client.meta.endpoint_url)
    if (
        url is None
        or config_digest is None
        or _native_loopback_endpoint(endpoint.host) != url
        or type(session) is not URLLib3Session
        or type(config.region_name) is not str
        or not 1 <= len(config.region_name) <= 64
    ):
        return None
    if session._proxy_config.proxy_url_for(url) is not None or config.proxies or config.proxies_config:
        return None
    timeouts = (config.connect_timeout, config.read_timeout)
    if any(type(value) not in {int, float} or not math.isfinite(value) or not 0 < value <= 60 for value in timeouts):
        return None
    pool = config.max_pool_connections
    if type(pool) is not int or not 1 <= pool <= 10 or session._max_pool_connections != pool:
        return None
    if session._timeout.connect_timeout != timeouts[0] or session._timeout.read_timeout != timeouts[1]:
        return None
    retries = config.retries
    if type(retries) is not dict or set(retries) - {"mode", "total_max_attempts"}:
        return None
    mode = retries.get("mode")
    if mode not in {"legacy", "standard"}:
        return None
    # The pinned SDK's legacy S3 model inherits five attempts; standard uses
    # three. Explicit Config limits have already been normalized by botocore.
    attempts = retries.get("total_max_attempts", 5 if mode == "legacy" else 3)
    if type(attempts) is not int or not 1 <= attempts <= 10:
        return None
    s3 = config.s3
    if s3 is not None and (
        type(s3) is not dict
        or set(s3) - {"addressing_style", "payload_signing_enabled"}
        or s3.get("addressing_style", "auto") not in {"auto", "path"}
        or type(s3.get("payload_signing_enabled", False)) is not bool
    ):
        return None
    if (
        config.signature_version != "s3v4"
        or config.parameter_validation is not True
        or config.client_cert is not None
        or config.use_dualstack_endpoint not in {None, False}
        or config.use_fips_endpoint not in {None, False}
        or config.endpoint_discovery_enabled not in {None, False}
        or config.client_context_params
        or config.sigv4a_signing_region_set is not None
        or config.auth_scheme_preference is not None
        or config.request_checksum_calculation not in {"when_supported", "when_required"}
        or config.response_checksum_validation not in {"when_supported", "when_required"}
        or (config.tcp_keepalive is not None and type(config.tcp_keepalive) is not bool)
    ):
        return None
    try:
        model_identity = _native_service_model_identity(client)
    except (BotoCoreError, KeyError, RecursionError):
        return None
    if model_identity is None:
        return None
    return {
        "endpoint": url,
        "region": config.region_name,
        "client_id": id(client),
        "config_id": id(config),
        "config_sha256": config_digest,
        "model": model_identity,
        "session_id": id(session),
        "pool_manager_id": id(session._manager),
        "connect_timeout": timeouts[0],
        "read_timeout": timeouts[1],
        "max_pool_connections": pool,
        "retry_mode": mode,
        "total_max_attempts": attempts,
        "s3": dict(s3) if s3 is not None else None,
        "request_checksum_calculation": config.request_checksum_calculation,
        "response_checksum_validation": config.response_checksum_validation,
        "tcp_keepalive": config.tcp_keepalive,
    }


class S3ArtifactStore(ArtifactStore):
    """S3 implementation of ``ArtifactStore``.

    A single ``S3ArtifactStore`` instance is intended for the lifetime
    of the process. ``boto3.client("s3", ...)`` instances are
    thread-safe; the same client is reused across calls.
    """

    def __init__(
        self,
        *,
        bucket: str,
        prefix: str = _DEFAULT_PREFIX,
        endpoint_url: Optional[str] = None,
        region_name: Optional[str] = None,
        client: Optional[Any] = None,
        photographic_publication_root: Optional[Path] = None,
    ) -> None:
        if not bucket:
            raise ArtifactStoreError("S3ArtifactStore requires a non-empty bucket (TP_ARTIFACT_BUCKET).")
        self._bucket = bucket
        self._prefix = _normalize_prefix(prefix)
        self._endpoint_url = endpoint_url or None
        self._region_name = region_name or None
        self._client = client
        self._lock = asyncio.Lock()
        self._native_publication_root: Path | None = None
        self._native_publication_stat: tuple[int, int] | None = None
        if isinstance(photographic_publication_root, Path) and photographic_publication_root.is_absolute():
            try:
                if not photographic_publication_root.is_symlink():
                    root = photographic_publication_root.resolve(strict=True)
                    if root.is_dir():
                        info = root.stat()
                        self._native_publication_root = root
                        self._native_publication_stat = info.st_dev, info.st_ino
            except (OSError, ValueError):
                pass

    @property
    def backend(self) -> str:
        return "s3"

    @property
    def bucket(self) -> str:
        return self._bucket

    @property
    def key_prefix(self) -> str:
        return self._prefix

    @property
    def native_publication_root(self) -> Path | None:
        """Operator-mapped MinIO host capacity root, never a remote S3 claim.

        The private qualification campaign must independently verify the actual
        container bind/endpoint mapping. Merely configuring this path does not
        establish that physical mapping, and no directory is created here.
        """
        root = self._native_publication_root
        if root is None:
            return None
        try:
            info = root.stat()
            if root.resolve() == root and root.is_dir() and (info.st_dev, info.st_ino) == self._native_publication_stat:
                return root
        except (OSError, ValueError):
            pass
        return None

    def native_resource_identity(self) -> dict[str, Any] | None:
        """Return a private closed identity only for the bounded local pilot.

        No client is initialized and no network/credential discovery executes.
        Unsupported configuration simply retains conservative admission.
        """
        root = self.native_publication_root
        root_identity = self._native_publication_stat
        if root is None or root_identity is None or self._client is None:
            return None
        if (
            type(self._bucket) is not str
            or not 1 <= len(self._bucket) <= 255
            or len(self._prefix) > 1024
            or len(str(root)) > 4096
        ):
            return None
        try:
            settings = _native_client_settings(self._client)
            configured = _native_loopback_endpoint(self._endpoint_url)
            if settings is None or configured is None or configured != settings["endpoint"]:
                return None
            if self._region_name is not None and self._region_name != settings["region"]:
                return None
            return {
                "schema": "tp.photo.s3_resource_identity.v1",
                "bucket": self._bucket,
                "prefix": self._prefix,
                "root": str(root),
                "device": root_identity[0],
                "inode": root_identity[1],
                "client": settings,
            }
        except (AttributeError, ImportError, OSError, TypeError, ValueError):
            return None

    async def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        async with self._lock:
            if self._client is not None:
                return self._client
            boto3 = _load_boto3()
            self._client = await asyncio.to_thread(
                boto3.client,
                "s3",
                endpoint_url=self._endpoint_url,
                region_name=self._region_name,
            )
            return self._client

    def _job_prefix(self, job_id: str) -> str:
        normalized_job_id = _normalize_job_id(job_id)
        return f"{self._prefix}/jobs/{normalized_job_id}"

    def _object_key(self, job_id: str, relative_path: str) -> str:
        normalized = _normalize_relative_path(relative_path)
        return f"{self._job_prefix(job_id)}/{normalized}"

    async def write_immutable_file(
        self,
        job_id: str,
        relative_path: str,
        source_path: Path,
        *,
        content_type: Optional[str] = None,
    ) -> None:
        key = self._object_key(job_id, relative_path)
        client = await self._get_client()

        def create() -> None:
            try:
                with source_path.open("rb") as source:
                    client.put_object(
                        Bucket=self._bucket,
                        Key=key,
                        Body=source,
                        ContentLength=source_path.stat().st_size,
                        ContentType=content_type or _content_type_for(relative_path),
                        IfNoneMatch="*",
                    )
            except Exception as exc:
                raise ArtifactStoreError("immutable S3 generation write failed") from exc

        await asyncio.to_thread(create)

    async def head(self, job_id: str, relative_path: str) -> ArtifactObjectMetadata:
        key = self._object_key(job_id, relative_path)
        client = await self._get_client()
        try:
            head = await asyncio.to_thread(client.head_object, Bucket=self._bucket, Key=key)
        except Exception as exc:  # noqa: BLE001 - boto3 ClientError surfaces here
            if _is_not_found(exc):
                raise ArtifactNotFoundError(f"artifact not found: {job_id}/{relative_path}") from exc
            raise ArtifactStoreError(f"S3 head_object failed for {key}") from exc

        size_bytes = int(head.get("ContentLength", 0)) if "ContentLength" in head else None
        content_type = head.get("ContentType") or _content_type_for(relative_path)

        if size_bytes is None:
            sha256_hex: Optional[str] = None
            status = "unavailable"
        elif size_bytes > ARTIFACT_FINGERPRINT_MAX_BYTES:
            sha256_hex = None
            status = "skipped_size"
        else:
            sha256_hex, status = await self._stream_sha256(client, key, expected_size_bytes=size_bytes)

        return ArtifactObjectMetadata(
            relative_path=_normalize_relative_path(relative_path),
            size_bytes=size_bytes,
            content_type=content_type,
            sha256_hex=sha256_hex,
            fingerprint_status=status,
        )

    async def open_bytes(
        self,
        job_id: str,
        relative_path: str,
    ) -> AsyncIterator[bytes]:
        key = self._object_key(job_id, relative_path)
        client = await self._get_client()
        try:
            response = await asyncio.to_thread(client.get_object, Bucket=self._bucket, Key=key)
        except Exception as exc:  # noqa: BLE001 - boto3 ClientError
            if _is_not_found(exc):
                raise ArtifactNotFoundError(f"artifact not found: {job_id}/{relative_path}") from exc
            raise ArtifactStoreError(f"S3 get_object failed for {key}") from exc

        body = response["Body"]

        async def _stream() -> AsyncIterator[bytes]:
            try:
                while True:
                    chunk = await asyncio.to_thread(body.read, _CHUNK_BYTES)
                    if not chunk:
                        return
                    yield chunk
            finally:
                close = getattr(body, "close", None)
                if close is not None:
                    await asyncio.to_thread(close)

        return _stream()

    async def list_for_job(self, job_id: str) -> List[ArtifactObjectMetadata]:
        prefix = f"{self._job_prefix(job_id)}/"
        client = await self._get_client()
        results: List[ArtifactObjectMetadata] = []
        continuation: Optional[str] = None
        while True:
            kwargs = {"Bucket": self._bucket, "Prefix": prefix}
            if continuation:
                kwargs["ContinuationToken"] = continuation
            try:
                response = await asyncio.to_thread(client.list_objects_v2, **kwargs)
            except Exception as exc:  # noqa: BLE001 - boto3 ClientError
                raise ArtifactStoreError(f"S3 list_objects_v2 failed for {prefix}") from exc

            for entry in response.get("Contents", []) or []:
                key = entry["Key"]
                if not key.startswith(prefix):
                    continue
                relative = key[len(prefix) :]
                if not relative or relative.endswith("/"):
                    continue
                try:
                    normalized_relative = _normalize_relative_path(relative)
                except ArtifactPathValidationError:
                    continue
                try:
                    head = await asyncio.to_thread(client.head_object, Bucket=self._bucket, Key=key)
                except Exception as exc:  # noqa: BLE001 - boto3 ClientError
                    if _is_not_found(exc):
                        continue
                    raise ArtifactStoreError(f"S3 head_object failed for {key}") from exc
                size_bytes = int(head.get("ContentLength", 0)) if "ContentLength" in head else None
                content_type = head.get("ContentType") or _content_type_for(normalized_relative)
                if size_bytes is None:
                    sha256_hex: Optional[str] = None
                    status = "unavailable"
                elif size_bytes > ARTIFACT_FINGERPRINT_MAX_BYTES:
                    sha256_hex = None
                    status = "skipped_size"
                else:
                    sha256_hex, status = await self._stream_sha256(client, key, expected_size_bytes=size_bytes)
                results.append(
                    ArtifactObjectMetadata(
                        relative_path=normalized_relative,
                        size_bytes=size_bytes,
                        content_type=content_type,
                        sha256_hex=sha256_hex,
                        fingerprint_status=status,
                    )
                )

            if not response.get("IsTruncated"):
                break
            continuation = response.get("NextContinuationToken")
            if not continuation:
                break

        results.sort(key=lambda meta: meta.relative_path)
        return results

    async def write_bytes(
        self,
        job_id: str,
        relative_path: str,
        body: bytes,
        *,
        content_type: Optional[str] = None,
    ) -> ArtifactObjectMetadata:
        key = self._object_key(job_id, relative_path)
        client = await self._get_client()
        explicit_content_type = str(content_type).strip() if content_type is not None else ""
        resolved_content_type = explicit_content_type or _content_type_for(relative_path)
        try:
            await asyncio.to_thread(
                client.put_object,
                Bucket=self._bucket,
                Key=key,
                Body=body,
                ContentType=resolved_content_type,
            )
        except Exception as exc:  # noqa: BLE001 - boto3 ClientError
            raise ArtifactStoreError(f"S3 put_object failed for {key}") from exc

        size_bytes: int = len(body)
        if size_bytes > ARTIFACT_FINGERPRINT_MAX_BYTES:
            sha256_hex: Optional[str] = None
            status = "skipped_size"
        else:
            sha256_hex = hashlib.sha256(body).hexdigest()
            status = "ok"
        return ArtifactObjectMetadata(
            relative_path=_normalize_relative_path(relative_path),
            size_bytes=size_bytes,
            content_type=resolved_content_type,
            sha256_hex=sha256_hex,
            fingerprint_status=status,
        )

    async def write_file(
        self,
        job_id: str,
        relative_path: str,
        source_path: Path,
        *,
        content_type: Optional[str] = None,
    ) -> ArtifactObjectMetadata:
        key = self._object_key(job_id, relative_path)
        source = Path(source_path)
        if not source.is_file():
            raise ArtifactNotFoundError(f"source artifact not found: {source}")
        try:
            size_bytes: Optional[int] = source.stat().st_size
        except OSError:
            size_bytes = None

        explicit_content_type = str(content_type).strip() if content_type is not None else ""
        resolved_content_type = explicit_content_type or _content_type_for(relative_path)
        client = await self._get_client()

        def _upload() -> None:
            with source.open("rb") as handle:
                client.upload_fileobj(
                    handle,
                    self._bucket,
                    key,
                    ExtraArgs={"ContentType": resolved_content_type},
                )

        try:
            await asyncio.to_thread(_upload)
        except Exception as exc:  # noqa: BLE001 - boto3 ClientError
            raise ArtifactStoreError(f"S3 upload_fileobj failed for {key}") from exc

        sha256_hex, status = self._file_sha256(source, size_bytes)
        return ArtifactObjectMetadata(
            relative_path=_normalize_relative_path(relative_path),
            size_bytes=size_bytes,
            content_type=resolved_content_type,
            sha256_hex=sha256_hex,
            fingerprint_status=status,
        )

    async def presign_get(
        self,
        job_id: str,
        relative_path: str,
        *,
        expires_seconds: int,
        content_type: Optional[str] = None,
        content_disposition: Optional[str] = None,
        cache_control: Optional[str] = None,
    ) -> Optional[str]:
        key = self._object_key(job_id, relative_path)
        client = await self._get_client()
        params = {"Bucket": self._bucket, "Key": key}
        if content_type:
            params["ResponseContentType"] = str(content_type)
        if content_disposition:
            params["ResponseContentDisposition"] = str(content_disposition)
        if cache_control:
            params["ResponseCacheControl"] = str(cache_control)
        try:
            return await asyncio.to_thread(
                client.generate_presigned_url,
                "get_object",
                Params=params,
                ExpiresIn=int(expires_seconds),
            )
        except Exception as exc:  # noqa: BLE001 - boto3 ClientError
            raise ArtifactStoreError(f"S3 presign failed for {key}") from exc

    async def check_ready(self) -> None:
        client = await self._get_client()
        try:
            await asyncio.to_thread(client.head_bucket, Bucket=self._bucket)
        except Exception as exc:  # noqa: BLE001 - boto3 ClientError
            raise ArtifactStoreError("S3 artifact store readiness check failed") from exc

    async def delete(self, job_id: str, relative_path: Optional[str] = None) -> int:
        if relative_path is None:
            # Bulk-delete every object under the job prefix.
            prefix = f"{self._job_prefix(job_id)}/"
            client = await self._get_client()
            deleted = 0
            continuation: Optional[str] = None
            while True:
                kwargs = {"Bucket": self._bucket, "Prefix": prefix}
                if continuation:
                    kwargs["ContinuationToken"] = continuation
                response = await asyncio.to_thread(client.list_objects_v2, **kwargs)
                keys = [{"Key": entry["Key"]} for entry in (response.get("Contents", []) or [])]
                if keys:
                    # ``delete_objects`` accepts up to 1000 keys per call.
                    for batch_start in range(0, len(keys), 1000):
                        batch = keys[batch_start : batch_start + 1000]
                        deleted += await self._delete_object_batch(client, batch)
                if not response.get("IsTruncated"):
                    break
                continuation = response.get("NextContinuationToken")
                if not continuation:
                    break
            return deleted

        key = self._object_key(job_id, relative_path)
        client = await self._get_client()
        try:
            await asyncio.to_thread(client.head_object, Bucket=self._bucket, Key=key)
        except Exception as exc:  # noqa: BLE001 - boto3 ClientError
            if _is_not_found(exc):
                return 0
            raise ArtifactStoreError(f"S3 head_object failed for {key}") from exc
        try:
            await asyncio.to_thread(client.delete_object, Bucket=self._bucket, Key=key)
        except Exception as exc:  # noqa: BLE001 - boto3 ClientError
            raise ArtifactStoreError(f"S3 delete_object failed for {key}") from exc
        return 1

    async def reset(self) -> None:
        """Test-only: delete every object under the configured prefix.

        Production deployments must not invoke ``reset`` — it is
        equivalent to ``aws s3 rm s3://{bucket}/{prefix}/jobs/ --recursive``
        and is destructive. Implemented to satisfy the contract test
        suite that uses a per-test key prefix so parallel cases
        (``pytest-xdist``) and shared-tenant buckets stay isolated.
        """
        client = await self._get_client()
        prefix = f"{self._prefix}/jobs/"
        continuation: Optional[str] = None
        while True:
            kwargs = {"Bucket": self._bucket, "Prefix": prefix}
            if continuation:
                kwargs["ContinuationToken"] = continuation
            response = await asyncio.to_thread(client.list_objects_v2, **kwargs)
            keys = [{"Key": entry["Key"]} for entry in (response.get("Contents", []) or [])]
            if keys:
                for batch_start in range(0, len(keys), 1000):
                    batch = keys[batch_start : batch_start + 1000]
                    await self._delete_object_batch(client, batch)
            if not response.get("IsTruncated"):
                break
            continuation = response.get("NextContinuationToken")
            if not continuation:
                break

    async def _delete_object_batch(self, client: Any, batch: List[dict[str, str]]) -> int:
        response = await asyncio.to_thread(
            client.delete_objects,
            Bucket=self._bucket,
            Delete={"Objects": batch, "Quiet": False},
        )
        errors = response.get("Errors", []) or []
        if errors:
            sample = ", ".join(f"{error.get('Key', '<unknown>')}:{error.get('Code', '<unknown>')}" for error in errors[:3])
            raise ArtifactStoreError(f"S3 delete_objects failed for {len(errors)} keys under {self._prefix}: {sample}")
        deleted = response.get("Deleted", []) or []
        return len(deleted)

    async def _stream_sha256(self, client: Any, key: str, *, expected_size_bytes: int) -> tuple[Optional[str], str]:
        try:
            response = await asyncio.to_thread(client.get_object, Bucket=self._bucket, Key=key)
        except Exception as exc:  # noqa: BLE001 - boto3 ClientError
            if _is_not_found(exc):
                return None, "unavailable"
            return None, "unavailable"
        body = response["Body"]
        digest = hashlib.sha256()
        try:
            content_length = response.get("ContentLength")
            if content_length is not None:
                if not isinstance(content_length, int) or isinstance(content_length, bool) or content_length < 0:
                    return None, "unavailable"
                if content_length > ARTIFACT_FINGERPRINT_MAX_BYTES:
                    return None, "skipped_size"
                if content_length != expected_size_bytes:
                    return None, "unavailable"
            bytes_read = 0
            while True:
                # Read at most one byte beyond the budget to distinguish an
                # exact-cap object from an oversized body with missing or
                # stale ContentLength. Never hash bytes beyond the cap.
                read_size = min(_CHUNK_BYTES, ARTIFACT_FINGERPRINT_MAX_BYTES - bytes_read + 1)
                chunk = await asyncio.to_thread(body.read, read_size)
                if not chunk:
                    break
                bytes_read += len(chunk)
                if bytes_read > ARTIFACT_FINGERPRINT_MAX_BYTES:
                    return None, "skipped_size"
                digest.update(chunk)
            if bytes_read != expected_size_bytes:
                return None, "unavailable"
        except Exception:  # noqa: BLE001 - any read failure → "unavailable"
            return None, "unavailable"
        finally:
            close = getattr(body, "close", None)
            if close is not None:
                try:
                    await asyncio.to_thread(close)
                except Exception:  # noqa: BLE001 - best-effort
                    pass
        return digest.hexdigest(), "ok"

    def _file_sha256(self, path: Path, size_bytes: Optional[int]) -> tuple[Optional[str], str]:
        if size_bytes is None:
            return None, "unavailable"
        if size_bytes > ARTIFACT_FINGERPRINT_MAX_BYTES:
            return None, "skipped_size"
        digest = hashlib.sha256()
        try:
            with path.open("rb") as handle:
                while True:
                    chunk = handle.read(_CHUNK_BYTES)
                    if not chunk:
                        break
                    digest.update(chunk)
        except OSError:
            return None, "unavailable"
        return digest.hexdigest(), "ok"


def _is_not_found(exc: Exception) -> bool:
    """Return ``True`` for boto3 ``ClientError`` 404 / NoSuchKey responses."""
    response = getattr(exc, "response", None)
    if not isinstance(response, dict):
        return False
    error = response.get("Error", {}) or {}
    code = str(error.get("Code", "")).strip()
    if code in {"404", "NoSuchKey", "NotFound"}:
        return True
    status = (
        response.get("ResponseMetadata", {}).get("HTTPStatusCode")
        if isinstance(response.get("ResponseMetadata"), dict)
        else None
    )
    return status == 404


__all__ = ["S3ArtifactStore"]
