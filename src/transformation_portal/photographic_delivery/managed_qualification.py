"""Private, supervised managed CPU evidence; never dispatch or photographic approval."""

from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import hmac
import importlib
import importlib.metadata
import importlib.util
import ipaddress
import json
import math
import os
import resource
import secrets
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import urlsplit

from transformation_portal.ingest.canonical_json import dumps_json
from transformation_portal.photographic_delivery import qualification as offline

SCHEMA = "tp.photo.managed_materials_qualification.v1"
MANIFEST_SCHEMA = "tp.photo.managed_materials_manifest.v1"
MODULE = "transformation_portal.photographic_delivery.managed_qualification"
MAX_REPORT_BYTES = 4 * offline.MIB
SCENARIOS = frozenset({"success", "cancel_helper", "cancel_publication", "cancel_transport", "publication_failure"})
PROVIDERS = frozenset({"retained-local", "canonical-s3"})
_SERVICE_ENV = (
    "TP_MANAGED_QUALIFICATION_REDIS_URL",
    "TP_MANAGED_QUALIFICATION_S3_ENDPOINT",
    "TP_MANAGED_QUALIFICATION_S3_BUCKET",
    "TP_MANAGED_QUALIFICATION_S3_ACCESS_KEY_ID",
    "TP_MANAGED_QUALIFICATION_S3_SECRET_ACCESS_KEY",
)
_MANAGED_MODULES = (
    MODULE,
    "app",
    "transformation_portal.api.routes.photographic_review",
    "transformation_portal.api.v1.photographic_review",
    "transformation_portal.api.v1.envelopes",
    "transformation_portal.orchestrator.photographic_review_service",
    "transformation_portal.orchestrator.photographic_admission",
    "transformation_portal.orchestrator.photographic_review_access",
    "transformation_portal.orchestrator.photographic_review_store",
    "transformation_portal.portal.photographic_review_access",
    "transformation_portal.orchestrator.artifact_store.local",
    "transformation_portal.orchestrator.artifact_store.generation",
    "transformation_portal.orchestrator.storage.postgres",
    "transformation_portal.orchestrator.storage.operational",
    "transformation_portal.orchestrator.artifact_store.s3",
    "transformation_portal.orchestrator.queue.locator",
    "transformation_portal.orchestrator.queue.redis",
    "transformation_portal.orchestrator.execution_dispatch",
    "transformation_portal.orchestrator.lux_depth_adapter",
    "transformation_portal.orchestrator.photography_adapter",
    "transformation_portal.orchestrator.photography_v6_adapter",
    "transformation_portal.core.execution_plan_v4",
    "transformation_portal.core.execution_plan_v5",
    "transformation_portal.orchestrator.models",
    "transformation_portal.core.security.frontdoor_identity",
    "transformation_portal.core.security.tenant",
    "transformation_portal.photographic_delivery.native_resource_inputs",
    "transformation_portal.photographic_delivery.native_resources",
)


def _profile_identity() -> str | None:
    from transformation_portal.photographic_delivery.native_resources import runtime_profile_identity

    return runtime_profile_identity()


def runtime_identity() -> dict[str, Any]:
    """Observe the complete harness path without importing the global app in its parent."""
    identity = offline._runtime_identity()
    for name in _MANAGED_MODULES:
        location = importlib.util.find_spec(name)
        if location is None or location.origin is None:
            raise RuntimeError("Managed qualification requires file-backed implementation modules")
        identity["module_sha256"][name] = hashlib.sha256(Path(location.origin).read_bytes()).hexdigest()
    for name in (
        "fastapi",
        "starlette",
        "pydantic",
        "SQLAlchemy",
        "asyncpg",
        "httpx",
        "boto3",
        "botocore",
        "s3transfer",
        "redis",
        "jsonschema",
        "jsonschema-specifications",
        "referencing",
        "rpds-py",
        "attrs",
        "jmespath",
        "urllib3",
    ):
        identity["packages"][name] = importlib.metadata.version(name)
    identity["managed_profile_identity"] = _profile_identity()
    return identity


def _database_url(environment: dict[str, str] | None = None) -> str:
    from sqlalchemy.engine import make_url

    raw = (os.environ if environment is None else environment).get("TP_MANAGED_QUALIFICATION_DATABASE_URL", "")
    try:
        parsed = make_url(raw)
    except Exception:
        raise ValueError("A dedicated Postgres *_test database is required") from None
    if parsed.get_backend_name() != "postgresql" or not (parsed.database or "").endswith("_test"):
        raise ValueError("A dedicated Postgres *_test database is required")
    return raw


def _loopback_url(raw: str, *, redis: bool = False) -> str:
    """Deny cloud endpoints, credentials and DNS indirection in private campaigns."""
    try:
        parsed = urlsplit(raw)
        address = ipaddress.ip_address(parsed.hostname or "")
        valid_path = parsed.path in {"", "/"} if not redis else parsed.path.lstrip("/").isdigit()
        if (
            parsed.scheme != ("redis" if redis else "http")
            or not address.is_loopback
            or parsed.port is None
            or not valid_path
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError
        if redis and not 0 <= int(parsed.path.lstrip("/")) <= 15:
            raise ValueError
    except ValueError:
        raise ValueError("Canonical qualification requires explicit credential-free loopback service endpoints") from None
    return raw


def _canonical_services(environment: dict[str, str] | None = None) -> dict[str, str]:
    source = os.environ if environment is None else environment
    values = {key: source.get(key, "") for key in _SERVICE_ENV}
    _loopback_url(values[_SERVICE_ENV[0]], redis=True)
    _loopback_url(values[_SERVICE_ENV[1]])
    bucket = values[_SERVICE_ENV[2]]
    if not 3 <= len(bucket) <= 63 or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789.-" for character in bucket):
        raise ValueError("Canonical qualification requires an explicit private test bucket")
    if not values[_SERVICE_ENV[3]] or not values[_SERVICE_ENV[4]]:
        raise ValueError("Canonical qualification requires local test-service credentials in its private environment")
    return values


def _minio_receipt(container: str | None, publication_root: Path | None, endpoint: str) -> dict[str, Any]:
    if (
        container is None
        or not container.startswith("tp-native-canonical-")
        or len(container) > 128
        or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for character in container)
        or publication_root is None
        or not publication_root.is_absolute()
        or not publication_root.is_dir()
        or publication_root != publication_root.resolve()
    ):
        raise ValueError("Canonical qualification requires its owned MinIO container and canonical bind directory")
    projection = (
        '{"mounts":{{json .Mounts}},"ports":{{json .NetworkSettings.Ports}},'
        '"image":{{json .Config.Image}},"image_id":{{json .Image}},"running":{{json .State.Running}},'
        '"entrypoint":{{json .Path}},"arguments":{{json .Args}}}'
    )
    inspected = subprocess.run(
        ["docker", "inspect", "--format", projection, container],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        timeout=15,
        check=True,
    )
    if len(inspected.stdout) > 256 * 1024:
        raise ValueError("Owned MinIO inspection exceeded its metadata budget")
    data = json.loads(inspected.stdout)
    mounts = [row for row in data["mounts"] if row.get("Destination") == "/data"]
    ports = data["ports"].get("9000/tcp", [])
    target = urlsplit(endpoint)
    if (
        data.get("running") is not True
        or data.get("entrypoint") != "/usr/bin/docker-entrypoint.sh"
        or data.get("arguments") != ["server", "/data", "--console-address", ":9001"]
        or len(mounts) != 1
        or mounts[0].get("Type") != "bind"
        or mounts[0].get("RW") is not True
        or Path(mounts[0]["Source"]).resolve() != publication_root
        or len(ports) != 1
        or ports[0].get("HostIp") != target.hostname
        or ports[0].get("HostPort") != str(target.port)
        or not isinstance(data.get("image_id"), str)
        or not data["image_id"].startswith("sha256:")
    ):
        raise ValueError("Owned MinIO container does not prove the requested endpoint and physical publication mapping")
    version = subprocess.run(
        ["docker", "exec", container, "minio", "--version"],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        timeout=15,
        check=True,
    )
    if len(version.stdout) > 4096 or not version.stdout.startswith(b"minio version "):
        raise ValueError("Owned MinIO container version is unavailable")
    return {
        "container": container,
        "endpoint": endpoint,
        "publication_root": str(publication_root),
        "image": data["image"],
        "image_id": data["image_id"],
        "entrypoint": data["entrypoint"],
        "arguments": data["arguments"],
        "version": version.stdout.decode("utf-8").strip(),
        "bind_destination": "/data",
        "read_write": True,
    }


def _bound_file(path: Path, expected_sha256: str, expected_size: int) -> dict[str, Any]:
    if type(expected_size) is not int or not 0 < expected_size <= offline.MAX_SOURCE_BYTES:
        raise ValueError("Canonical input requires a bounded recorded byte length")
    with offline._open_regular(path, offline.MAX_SOURCE_BYTES) as stream:
        before = os.fstat(stream.fileno())
        if before.st_size != expected_size:
            raise ValueError("Canonical original source differs from its immutable plan")
        hasher = hashlib.sha256()
        count = 0
        while chunk := stream.read(min(offline.MIB, expected_size - count + 1)):
            count += len(chunk)
            if count > expected_size:
                raise ValueError("Canonical original source grew beyond its immutable byte bound")
            hasher.update(chunk)
        digest = hasher.hexdigest()
        after = os.fstat(stream.fileno())
    if offline._identity(before) != offline._identity(after) or count != expected_size or digest != expected_sha256:
        raise ValueError("Canonical original source differs from its immutable plan")
    return {"sha256": digest, "size_bytes": before.st_size, "file_identity": offline._identity(before)}


def _canonical_authority(plan_path: Path | None, bindings_path: Path | None) -> dict[str, Any]:
    """Validate real Unified process authority without executing its model plan."""
    from transformation_portal.orchestrator.lux_depth_adapter import UnifiedBindings, validate_unified_dispatch

    if plan_path is None or bindings_path is None:
        raise ValueError("Canonical S3 qualification requires explicit Unified plan and bindings files")
    with offline._open_regular(plan_path, 512 * 1024) as stream:
        plan = stream.read(512 * 1024 + 1)
    with offline._open_regular(bindings_path, 64 * 1024) as stream:
        bindings = stream.read(64 * 1024 + 1)
    envelope = UnifiedBindings(bindings)
    if envelope.workflow != "process":
        raise ValueError("Canonical S3 qualification currently covers Unified process authority only")
    parsed = validate_unified_dispatch(plan, bindings).to_payload()
    original_root = Path(envelope.to_payload()["photography_bindings"]["input_root"])
    originals = []
    for row in parsed["inference"]["inputs"]:
        path = original_root / row["path"]
        if not path.is_relative_to(original_root):
            raise ValueError("Canonical original escaped its bound input root")
        originals.append(
            {"path": str(path), "relative_path": row["path"], "bound": _bound_file(path, row["sha256"], row["size_bytes"])}
        )
    return {
        "plan": plan.decode("utf-8"),
        "bindings": bindings.decode("utf-8"),
        "plan_sha256": hashlib.sha256(plan).hexdigest(),
        "bindings_sha256": hashlib.sha256(bindings).hexdigest(),
        "originals": originals,
        "scope": "retained_fixture_import_under_validated_unified_process_authority_no_inference",
    }


def _manifest(path: Path) -> list[dict[str, Any]]:
    from transformation_portal.photographic_delivery.materials import MaterialsRecipe

    document = offline._read_json(path, 256 * 1024)
    if not isinstance(document, dict) or set(document) != {"schema", "sources"} or document["schema"] != MANIFEST_SCHEMA:
        raise ValueError("Managed qualification requires its closed retained-source manifest")
    sources = document["sources"]
    if not isinstance(sources, list) or not 1 <= len(sources) <= 8:
        raise ValueError("Managed qualification requires one to eight explicit retained sources")
    result, seen = [], set()
    for row in sources:
        if not isinstance(row, dict) or set(row) != {"path", "sha256", "recipe"} or not isinstance(row["path"], str):
            raise ValueError("Each retained source requires exactly path, sha256 and recipe")
        source = Path(row["path"])
        if not source.is_absolute() or source in seen:
            raise ValueError("Retained sources must have distinct absolute paths")
        seen.add(source)
        bound = offline._source(source, row["sha256"])
        recipe = MaterialsRecipe.from_payload(row["recipe"])
        result.append({"path": str(source), "bound": bound, "recipe": recipe.to_payload()})
    return result


def _canonical_scratch_plan(sources: list[dict[str, Any]], repeats: int, authority: dict[str, Any]) -> dict[str, Any]:
    """Bound complete retained data plus one live operation, not all trial staging."""
    from transformation_portal.lux_depth_v5.preview import MAX_PREVIEW_BYTES
    from transformation_portal.photographic_delivery.materials import (
        MAX_MATERIALS_DESCRIPTOR_BYTES,
        MAX_MATERIALS_OVERLAY_BYTES,
    )

    products = [
        row["bound"]["width"] * row["bound"]["height"] * 8
        + 65536
        + MAX_PREVIEW_BYTES
        + MAX_MATERIALS_OVERLAY_BYTES
        + MAX_MATERIALS_DESCRIPTOR_BYTES
        for row in sources
    ]
    terms = {
        "base_metadata_bytes": 256 * offline.MIB,
        "retained_source_and_generation_snapshot_bytes": 3 * sum(row["bound"]["size_bytes"] for row in sources),
        "retained_original_copy_bytes": sum(row["bound"]["size_bytes"] for row in authority["originals"]),
        "all_s3_products_and_independent_verifier_copies_bytes": 2 * repeats * sum(products),
        "one_live_operation_transient_bytes": max(
            row["bound"]["size_bytes"] + 4 * product for row, product in zip(sources, products, strict=True)
        ),
    }
    # Each POST drains its source/private products/publication staging before
    # the next trial. The verifier begins only after that worker has exited;
    # all S3 products and every independent verification copy remain retained.
    # The transient allowance includes the current publication and MinIO write.
    return {
        "schema": "tp.photo.private_canonical_campaign_scratch.v1",
        "scope": "complete_private_campaign_lifetime_bound_not_operation_admission_limits",
        "terms": terms,
        "total_bytes": sum(terms.values()),
        "product_upper_bound_bytes": products,
        "repeat_count": repeats,
    }


def _headroom(
    sources: list[dict[str, Any]],
    repeats: int,
    output_parent: Path,
    authority: dict[str, Any] | None = None,
    publication_root: Path | None = None,
) -> tuple[int, list[str]]:
    import psutil

    from transformation_portal.lux_depth_v5.preview import MAX_PREVIEW_BYTES
    from transformation_portal.orchestrator.photographic_admission import ReviewAdmissionConfig, effective_interactive_reserve

    config = ReviewAdmissionConfig(enabled=True)
    if (authority is None) != (publication_root is None):
        raise ValueError("Canonical campaign scratch requires both original authority and its publication filesystem")
    if authority is not None:
        scratch = _canonical_scratch_plan(sources, repeats, authority)["total_bytes"]
    else:
        # Preserve the retained-local campaign's existing complete-copy bound.
        scratch = 256 * offline.MIB
        for row in sources:
            bound = row["bound"]
            product = bound["width"] * bound["height"] * 8 + 65536 + 2 * MAX_PREVIEW_BYTES + 256 * 1024
            scratch += 2 * bound["size_bytes"] + 3 * product * repeats
    reasons = []
    if scratch > config.scratch_bytes - config.interactive_scratch_bytes:
        reasons.append("scratch_operation_limit")
    for filesystem in dict.fromkeys([output_parent, *([publication_root] if publication_root is not None else [])]):
        if shutil.disk_usage(filesystem).free < scratch + config.interactive_scratch_bytes + config.scratch_floor_bytes:
            reasons.append("physical_scratch_headroom")
            break
    if psutil.virtual_memory().available < config.memory_bytes + effective_interactive_reserve(config) + 2 * offline.GIB:
        reasons.append("physical_memory_headroom")
    return scratch, reasons


def _configure_worker(root: Path) -> None:
    """Only the new child receives these flags, keys and paths."""
    tenant = "qual_" + uuid.uuid4().hex
    environment = {
        "TP_ORCHESTRATOR_STATE_BACKEND": "postgres",
        "TP_ORCHESTRATOR_QUEUE_BACKEND": "memory",
        "TP_DATABASE_URL": _database_url(),
        "TP_ARTIFACT_STORE": "local",
        "TP_ARTIFACT_LOCAL_ROOT": str(root / "artifacts"),
        "TP_PILOT_CONTROL_PLANE_ENABLED": "1",
        "TP_PILOT_TENANT_WORKSPACE_ROOT": str(root / "workspaces"),
        "TP_PILOT_TENANT_CAS_ROOT": str(root / "cas"),
        "TP_PILOT_ACTOR_TENANTS_JSON": dumps_json({"qualification@example.invalid": tenant}),
        "TP_FRONTDOOR_IDENTITY_SECRET": secrets.token_hex(32),
        "TP_API_KEY": secrets.token_hex(32),
        "TP_PHOTO_REVIEW_ENABLED": "1",
        "TP_PHOTO_REVIEW_NATIVE_MATERIALS_ENABLED": "1",
        "TP_ORCHESTRATOR_IN_PROCESS_WORKERS_ENABLED": "0",
        "TP_ALLOWED_INPUT_ROOTS": str(root),
        "TP_ALLOWED_OUTPUT_ROOTS": str(root),
    }
    spec = offline._read_json(root / "specification.json", MAX_REPORT_BYTES)
    if spec.get("provider", "retained-local") == "canonical-s3":
        services = _canonical_services()
        environment.update(
            TP_ORCHESTRATOR_QUEUE_BACKEND="redis",
            TP_REDIS_URL=services[_SERVICE_ENV[0]],
            TP_REDIS_KEY_PREFIX="tp:managed-qualification:" + spec["service_namespace"],
            TP_ORCHESTRATOR_EXECUTION_ROOT=str(root / "execution"),
            TP_ARTIFACT_STORE="s3",
            TP_ARTIFACT_ENDPOINT_URL=services[_SERVICE_ENV[1]],
            TP_ARTIFACT_BUCKET=services[_SERVICE_ENV[2]],
            TP_ARTIFACT_PREFIX="tp/managed-qualification/" + spec["service_namespace"],
            TP_ARTIFACT_REGION="us-east-1",
            TP_PHOTO_REVIEW_S3_PUBLICATION_ROOT=spec["s3_publication_root"],
            AWS_ACCESS_KEY_ID=services[_SERVICE_ENV[3]],
            AWS_SECRET_ACCESS_KEY=services[_SERVICE_ENV[4]],
            AWS_DEFAULT_REGION="us-east-1",
            AWS_EC2_METADATA_DISABLED="true",
        )
    os.environ.update(environment)
    os.environ["TP_MANAGED_QUALIFICATION_TENANT"] = tenant


def _headers(method: str, target: str) -> dict[str, str]:
    payload = {
        "v": 1,
        "iat": int(time.time()),
        "method": method,
        "target": target,
        "actor": {"username": "qualification", "accessEmail": "qualification@example.invalid", "role": "admin"},
    }
    encoded = base64.urlsafe_b64encode(offline._canonical(payload)).decode("ascii").rstrip("=")
    signature = hmac.new(
        os.environ["TP_FRONTDOOR_IDENTITY_SECRET"].encode(), b"tp.frontdoor.actor.v1\n" + encoded.encode(), hashlib.sha256
    ).hexdigest()
    return {"x-api-key": os.environ["TP_API_KEY"], "x-tp-actor-assertion": encoded + "." + signature}


async def _wait_for_barrier(operation: asyncio.Task[Any], event: Any, description: str, timeout: float = 30) -> None:
    """Surface request failures immediately; never strand an owned waiter thread."""

    async def observe() -> None:
        while not event.is_set():
            await asyncio.sleep(0.005)

    waiter = asyncio.create_task(observe())
    try:
        completed, _ = await asyncio.wait({operation, waiter}, timeout=timeout, return_when=asyncio.FIRST_COMPLETED)
        if operation in completed:
            operation.result()  # Preserve its actual HTTP/authority failure.
            raise RuntimeError(f"Managed request completed before its {description} barrier")
        if waiter not in completed:
            raise TimeoutError(f"Managed request did not reach its {description} barrier")
    finally:
        waiter.cancel()
        await asyncio.gather(waiter, return_exceptions=True)


class _Observer:
    """Metadata only; never retain decoded arrays, responses or result buffers."""

    def __init__(self, root: Path, service: Any, scenario: str = "success", expected_profile_identity: str | None = None):
        self.root, self.service = root, service
        self.records: list[dict[str, Any]] = []
        self.lock = threading.Lock()
        self.helper_entered, self.helper_release = threading.Event(), threading.Event()
        self.pause_helper = False
        self.scenario = scenario
        self.publication_entered = asyncio.Event()
        self.publication_release = asyncio.Event()
        self.publication_count = 0
        self.pause_transport = False
        self.transport_entered = asyncio.Event()
        self.transport_release = asyncio.Event()
        self.transport_cancelled = asyncio.Event()
        self.transport_cancel_release = asyncio.Event()
        self.inner_response_returned = asyncio.Event()
        self.expected_profile_identity = expected_profile_identity
        self.started = time.monotonic()
        self.publication_store_type: Any = None
        self.publication_root: Path | None = None
        self.object_root: Path | None = None

    def capture(self, phase: str, *, inspection_crop: tuple[int, int, int, int] | None = None) -> None:
        import psutil

        from transformation_portal.orchestrator.photographic_admission import current_review_lease

        multiplier = 1 if sys.platform == "darwin" else 1024
        lease = current_review_lease.get()
        qualified = getattr(lease.cost, "native_resource_qualification", None) if lease is not None else None
        record = {
            "phase": phase,
            "inspection_crop": list(inspection_crop) if inspection_crop is not None else None,
            "elapsed_seconds": time.monotonic() - self.started,
            "rss_cumulative_high_water_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * multiplier),
            "current_rss_bytes": psutil.Process().memory_info().rss,
            "admission": self.service.admission.snapshot(),
            "scratch": offline._scratch_usage(self.root),
            "physical_free_bytes": shutil.disk_usage(self.root).free,
            "filesystem_device": self.root.stat().st_dev,
            "native_resource_qualification": qualified.to_payload() if qualified is not None else None,
            "lease": (
                None
                if lease is None
                else {
                    "memory_bytes": lease.cost.memory_bytes,
                    "scratch_bytes": lease.cost.scratch_bytes,
                    "decode_slots": lease.cost.decode_slots,
                    "io_slots": lease.cost.io_slots,
                    "interactive": lease.cost.interactive,
                    "native_response_owned": getattr(lease, "native_response_owned", False),
                }
            ),
        }
        if self.publication_root is not None:
            record["publication_filesystem"] = {
                "root": str(self.publication_root),
                "device": self.publication_root.stat().st_dev,
                "physical_free_bytes": shutil.disk_usage(self.publication_root).free,
                "owned_prefix_usage": offline._scratch_usage(self.object_root) if self.object_root is not None else None,
            }
        with self.lock:
            self.records.append(record)


@contextmanager
def _observe_operations(observer: _Observer) -> Iterator[None]:
    from transformation_portal.orchestrator import photographic_review_service as service
    from transformation_portal.orchestrator.artifact_store.local import LocalArtifactStore
    from transformation_portal.photographic_delivery import materials
    from transformation_portal.portal import photographic_review_access as access

    load, create, snapshot, verify = (
        service.load_image,
        materials.create_materials_revision,
        access._publication_snapshot,
        access._verify_staged,
    )
    mutation = observer.service.mutate
    write_class = observer.publication_store_type or LocalArtifactStore
    write = write_class.write_immutable_file
    response_call = service._OwnedNativeJSONResponse.__call__

    async def response_returned(*args: Any, **kwargs: Any) -> None:
        try:
            await response_call(*args, **kwargs)
        finally:
            observer.capture("native_inner_response_returned")
            observer.inner_response_returned.set()

    async def loaded(*args: Any, **kwargs: Any) -> Any:
        result = await load(*args, **kwargs)
        observer.capture("source_loaded")
        return result

    def created(*args: Any, **kwargs: Any) -> Any:
        from transformation_portal.orchestrator.photographic_admission import current_review_lease
        from transformation_portal.photographic_delivery.native_resources import PROFILE_ID

        lease = current_review_lease.get()
        qualified = getattr(lease.cost, "native_resource_qualification", None) if lease is not None else None
        if (
            qualified is None
            or qualified.profile_id != PROFILE_ID
            or qualified.profile_identity != observer.expected_profile_identity
            or qualified.source_sha256 != args[0].source_sha256
            or lease is None
            or lease.cost.memory_bytes != 6 * offline.GIB
            or not getattr(lease, "native_response_owned", False)
        ):
            raise ValueError("Managed qualification requires its exact governed response-owned native lease")
        observer.capture("native_operation_start")
        result = create(*args, **kwargs)
        observer.capture("native_helper_return")
        if observer.pause_helper:
            observer.helper_entered.set()
            if not observer.helper_release.wait(30):
                raise TimeoutError("Concurrent inspection did not release its native barrier")
        return result

    def staged(*args: Any, **kwargs: Any) -> Any:
        result = snapshot(*args, **kwargs)
        observer.capture("publication_staged")
        return result

    async def verified(*args: Any, **kwargs: Any) -> None:
        await verify(*args, **kwargs)
        observer.capture("publication_verified")
        observer.publication_count += 1
        if observer.scenario == "publication_failure" and observer.publication_count == 2:
            raise OSError("Controlled private qualification publication failure")
        if observer.scenario == "cancel_publication" and observer.publication_count == 2:
            observer.publication_entered.set()
            await observer.publication_release.wait()

    async def written(*args: Any, **kwargs: Any) -> None:
        await write(*args, **kwargs)
        observer.capture("publication_written")

    async def mutate(*args: Any, **kwargs: Any) -> Any:
        observer.capture("before_cas_after_product_drop")
        result = await mutation(*args, **kwargs)
        observer.capture("after_cas")
        return result

    service.load_image, materials.create_materials_revision = loaded, created
    access._publication_snapshot, access._verify_staged = staged, verified
    setattr(write_class, "write_immutable_file", written)
    setattr(service._OwnedNativeJSONResponse, "__call__", response_returned)
    observer.service.mutate = mutate
    try:
        yield
    finally:
        observer.helper_release.set()
        observer.publication_release.set()
        observer.transport_release.set()
        observer.transport_cancel_release.set()
        service.load_image, materials.create_materials_revision = load, create
        access._publication_snapshot, access._verify_staged = snapshot, verify
        setattr(write_class, "write_immutable_file", write)
        setattr(service._OwnedNativeJSONResponse, "__call__", response_call)
        observer.service.mutate = mutation


class _ObservedTransport:
    """Block outside the deployed middleware; keep only ASGI metadata/events."""

    def __init__(self, app: Any, observer: _Observer) -> None:
        self.app, self.observer = app, observer

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        observer = self.observer
        native = scope["type"] == "http" and scope.get("method") == "POST" and scope.get("path", "").endswith("/revisions")
        blocked = False

        async def observed_send(message: Any) -> None:
            nonlocal blocked
            if native and observer.pause_transport and not blocked and message["type"] == "http.response.body":
                blocked = True
                observer.capture("native_outer_transport_blocked")
                observer.transport_entered.set()
                try:
                    await observer.transport_release.wait()
                except asyncio.CancelledError:
                    if observer.scenario == "cancel_transport":
                        from transformation_portal.orchestrator.photographic_admission import drain_owned

                        observer.transport_cancelled.set()
                        await drain_owned(observer.transport_cancel_release.wait())
                    raise
            await send(message)

        try:
            await self.app(scope, receive, observed_send)
        finally:
            if native:
                observer.capture("native_outer_transport_quiescent")


def _require_transport_reservation(observer: _Observer, operation: asyncio.Task[Any]) -> None:
    snapshot = observer.service.admission.snapshot()
    if operation.done() or snapshot["memory_reserved_bytes"] != 6 * offline.GIB or snapshot["outstanding"] != 1:
        raise RuntimeError("Deployed transport released native memory/ticket before outer send became quiescent")


async def _wait_for_owned_transport(observer: _Observer, operation: asyncio.Task[Any]) -> None:
    await _wait_for_barrier(operation, observer.transport_entered, "outer transport send")
    await _wait_for_barrier(operation, observer.inner_response_returned, "inner response return")
    _require_transport_reservation(observer, operation)
    observer.capture("native_outer_transport_holds_reservation_after_inner_return")


def _deployed_application(service: Any) -> Any:
    """Capture the real router's service while importing its complete deployed app."""
    from transformation_portal.api.routes import photographic_review as routes

    if "app" in sys.modules:
        raise RuntimeError("Managed qualification requires a fresh deployed application import")
    constructor = routes.PhotographicReviewService
    setattr(routes, "PhotographicReviewService", lambda: service)
    try:
        application = importlib.import_module("app")
    finally:
        setattr(routes, "PhotographicReviewService", constructor)
    if not isinstance(application.app, application._PhotographicReviewApplication):
        raise RuntimeError("Managed qualification requires the deployed outer photographic lifecycle")
    return application


async def _seed_canonical(
    app: Any, root: Path, spec: dict[str, Any], tenant: str, job_id: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Import retained fixtures through real locator/fence/generation authorities."""
    from transformation_portal.ingest.canonical_json import canonicalize_json
    from transformation_portal.orchestrator.artifact_store.generation import GenerationPublisher
    from transformation_portal.orchestrator.artifact_store.s3 import S3ArtifactStore
    from transformation_portal.orchestrator.lux_depth_adapter import UnifiedBindings, validate_unified_dispatch
    from transformation_portal.orchestrator.photography_adapter import PhotographyBindings
    from transformation_portal.orchestrator.queue.locator import RedisLocatorQueueBroker
    from transformation_portal.orchestrator.storage.base import JobRecord

    authority = spec["canonical_authority"]
    original_root = root / "workspaces" / tenant / "inputs"
    original_root.mkdir(parents=True)
    for row in authority["originals"]:
        target = original_root / row["relative_path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(offline._copy_source, Path(row["path"]), target, row["bound"])
    envelope = UnifiedBindings(authority["bindings"].encode("utf-8")).to_payload()
    rebound = PhotographyBindings.from_payload({**envelope["photography_bindings"], "input_root": str(original_root)})
    bindings = UnifiedBindings(canonicalize_json({**envelope, "photography_bindings": rebound.to_payload()})).canonical_bytes
    plan = authority["plan"].encode("utf-8")
    validate_unified_dispatch(plan, bindings)
    records, broker, store = app.get_operational_record_store(), app.get_queue_broker(), app._artifact_store()
    if type(broker) is not RedisLocatorQueueBroker or type(store) is not S3ArtifactStore or not app._uses_dispatch_authority():
        raise ValueError("Canonical qualification requires actual Redis and S3 provider factories")
    attempt_root = root / "workspaces" / tenant / "outputs" / job_id
    request = {"pipeline": "lux-depth", "args": {"workflow": "process", "output_dir": str(attempt_root)}}
    locator = await records.admit(
        JobRecord(id=job_id, created_at=time.time(), request=request, effective_request={**request, "tenant_id": tenant}),
        plan,
        tenant_id=tenant,
        output_root=str(attempt_root),
        requested_output_root=str(attempt_root),
        global_limit=4,
        tenant_limit=2,
        execution_bindings=bindings,
    )
    worker_id = "qualification-fixture-import"
    await broker.enqueue(locator)
    lease = await broker.acquire_lease(worker_id, lease_seconds=300)
    if lease is None or lease.request != locator:
        raise ValueError("Canonical qualification did not acquire its exact Redis locator")
    try:
        fence = await records.claim_dispatch(locator, worker_id, lease_seconds=300)
        files, expected = {}, {}
        for index, row in enumerate(spec["sources"]):
            relative = f"v6/input-{index:04d}/delivery.tif"
            path = Path(fence.output_root) / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            await asyncio.to_thread(offline._copy_source, Path(row["path"]), path, row["bound"])
            files[relative] = path
            expected[relative] = {key: row["bound"][key] for key in ("sha256", "size_bytes")}
        await GenerationPublisher(artifact_store=store, record_store=records).publish(
            fence,
            files,
            state="succeeded",
            exit_code=0,
            artifacts={},
            run_summary={"qualification_fixture_import": True, "model_inference_performed": False},
            expected_file_integrity=expected,
        )
        for outbox_id, own_job_id, _kind, _payload in await records.pending_outbox():
            if own_job_id == job_id:
                await records.acknowledge_outbox(outbox_id)
    finally:
        await broker.release_lease(worker_id, job_id)
        await broker.close()
    manifest = await records.committed_manifest(job_id, tenant)
    if manifest is None or len(manifest["files"]) != len(spec["sources"]):
        raise ValueError("Canonical fixture publication did not commit its exact source inventory")
    return manifest["files"], {
        "schema": "tp.photo.private_canonical_fixture_import.v1",
        "redis_locator_acquired": True,
        "plan_sha256": hashlib.sha256(plan).hexdigest(),
        "bindings_sha256": hashlib.sha256(bindings).hexdigest(),
        "generation_manifest_sha256": hashlib.sha256(canonicalize_json(manifest)).hexdigest(),
        "generation_id": manifest["generation_id"],
        "originals_copied": len(authority["originals"]),
        "bindings": bindings.decode("utf-8"),
        "model_inference_performed": False,
    }


async def _managed_run(root: Path, spec: dict[str, Any]) -> dict[str, Any]:
    import httpx

    from transformation_portal.orchestrator.artifact_store.local import LocalArtifactStore
    from transformation_portal.orchestrator.photographic_review_service import PhotographicReviewService
    from transformation_portal.orchestrator.photographic_review_store import get_review_store
    from transformation_portal.orchestrator.storage.base import JobRecord
    from transformation_portal.orchestrator.storage.postgres import _SharedEngine

    service = PhotographicReviewService()
    app = _deployed_application(service)

    # Request cleanup is unrelated to qualification and must not sweep existing
    # rows in the dedicated database. The actual helper takes its normal cheap exit.
    app._LAST_REQUEST_JOB_CLEANUP_AT = time.time() + 3600
    tenant, job_id = os.environ["TP_MANAGED_QUALIFICATION_TENANT"], "qual_" + uuid.uuid4().hex
    job_root = root / "workspaces" / tenant / "outputs" / job_id
    items: list[dict[str, Any]] = []
    lookup: dict[str, Path] = {}
    store = app._artifact_store()
    canonical = spec.get("provider", "retained-local") == "canonical-s3"
    canonical_import = None
    if canonical:
        items, canonical_import = await _seed_canonical(app, root, spec, tenant, job_id)
    elif not isinstance(store, LocalArtifactStore):
        raise ValueError("Managed resource qualification requires the real local artifact store")
    for index, row in enumerate([] if canonical else spec["sources"]):
        relative = f"v6/input-{index:04d}/delivery.tif"
        path = job_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(offline._copy_source, Path(row["path"]), path, row["bound"])
        await store.write_immutable_file(tenant + "__" + job_id, relative, path, content_type="image/tiff")
        items.append({"path": relative, "sha256": row["bound"]["sha256"], "size_bytes": row["bound"]["size_bytes"]})
        lookup[relative] = path
    now = time.time()
    repository = app._job_repository()
    if not canonical:
        await repository.create(
            JobRecord(
                id=job_id,
                created_at=now,
                state="succeeded",
                finished_at=now,
                exit_code=0,
                request={"pipeline": "lux-depth-v3", "args": {"output_dir": str(job_root)}},
                effective_request={"tenant_id": tenant, "pipeline": "lux-depth-v3", "args": {"output_dir": str(job_root)}},
                artifacts={"items": items, "lifecycle": {"mirror_status": "mirrored", "artifact_store_backend": "local"}},
                artifact_lookup=lookup,
            )
        )
    observer = _Observer(root, service, spec["scenario"], spec["runtime"]["managed_profile_identity"])
    observer.publication_store_type = type(store)
    if canonical:
        observer.publication_root = Path(spec["s3_publication_root"])
        observer.object_root = observer.publication_root / store.bucket / store.key_prefix
    api = app.app
    base = f"/v1/jobs/{job_id}/photographic-review"
    results, version = [], 0
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=_ObservedTransport(api, observer)), base_url="http://localhost"
    ) as client:

        async def request(method: str, target: str, *, expected_status: int = 200, **kwargs: Any) -> Any:
            response = await client.request(method, target, headers=_headers(method, target), **kwargs)
            if response.status_code != expected_status:
                raise RuntimeError(f"Managed request rejected: {response.status_code} {response.text[:500]}")
            return response

        async def inspect_while_held(phase: str, *, shifted: bool = False) -> None:
            from urllib.parse import urlencode

            selected = items[-1]
            geometry = spec["sources"][-1]["bound"]
            width, height = min(64, geometry["width"]), min(64, geometry["height"])
            # A distinct crop avoids treating the earlier read's cached result
            # as fresh inspection work while the outer transport is blocked.
            x = int(shifted and geometry["width"] > width)
            y = int(shifted and not x and geometry["height"] > height)
            target = (
                base
                + "/native-crop?"
                + urlencode(
                    {
                        "artifact_path": selected["path"],
                        "sha256": selected["sha256"],
                        "source_identity_sha256": selected["sha256"],
                        "x": x,
                        "y": y,
                        "width": width,
                        "height": height,
                    }
                )
            )
            await asyncio.wait_for(request("GET", target), timeout=10)
            observer.capture(phase, inspection_crop=(x, y, width, height))

        try:
            unauthorized = await client.get(base)
            if unauthorized.status_code != 401:
                raise RuntimeError("Managed fixture did not reject an unauthenticated request")
            forged = await client.get(base, headers={"x-api-key": os.environ["TP_API_KEY"], "x-tp-actor": "qualification"})
            if forged.status_code != 401:
                raise RuntimeError("Unsigned actor headers unexpectedly authorized managed review")
            wrong_tenant = await client.get(base, headers={**_headers("GET", base), "x-tp-tenant-id": "foreign_qualification"})
            if wrong_tenant.status_code != 403:
                raise RuntimeError("A conflicting tenant selector unexpectedly authorized managed review")
            if any(service.admission.snapshot()[key] for key in ("active", "queued", "outstanding")):
                raise RuntimeError("Failed authentication entered photographic scheduling")
            with _observe_operations(observer):
                catalogue = await request("GET", base)
                del catalogue
                from transformation_portal.orchestrator.photographic_review_service import PhotographicTransportMiddleware

                if not isinstance(api.middleware_stack, PhotographicTransportMiddleware):
                    raise RuntimeError("Managed qualification bypassed the deployed outer photographic middleware")
                observer.capture("catalogue_complete")
                for repeat in range(spec["repeats"]):
                    for index, row in enumerate(spec["sources"]):
                        entry = items[index]
                        payload = {
                            "artifact_path": entry["path"],
                            "sha256": entry["sha256"],
                            "source_identity_sha256": entry["sha256"],
                            "scene_id": f"input-{index:04d}",
                            "expected_version": version,
                            "recipe": row["recipe"],
                            "preview_only": False,
                        }
                        concurrent = spec["scenario"] == "success" and spec["concurrent_inspection"] and repeat == index == 0
                        observer.pause_helper = concurrent or spec["scenario"] == "cancel_helper"
                        observer.pause_transport = repeat == index == 0 and spec["scenario"] in {
                            "success",
                            "cancel_transport",
                        }
                        operation = asyncio.create_task(
                            request(
                                "POST",
                                base + "/revisions",
                                json=payload,
                                expected_status=503 if spec["scenario"] == "publication_failure" else 200,
                            )
                        )
                        try:
                            if spec["scenario"] in {"cancel_helper", "cancel_publication"}:
                                if spec["scenario"] == "cancel_helper":
                                    await _wait_for_barrier(operation, observer.helper_entered, "native cancellation")
                                else:
                                    await _wait_for_barrier(
                                        operation, observer.publication_entered, "publication cancellation"
                                    )
                                held = service.admission.snapshot()
                                operation.cancel()
                                await asyncio.sleep(0)
                                operation.cancel()
                                await asyncio.sleep(0)
                                current = service.admission.snapshot()
                                if (
                                    operation.done()
                                    or held["memory_reserved_bytes"] != current["memory_reserved_bytes"]
                                    or current["memory_reserved_bytes"] == 0
                                ):
                                    raise RuntimeError("Repeated cancellation abandoned owned native work")
                                observer.capture("repeated_cancellation_holds_owned_memory")
                                observer.helper_release.set()
                                observer.publication_release.set()
                                try:
                                    await operation
                                except asyncio.CancelledError:
                                    pass
                                else:
                                    raise RuntimeError("Cancelled native operation unexpectedly completed")
                                del operation
                                continue
                            if concurrent:
                                await _wait_for_barrier(operation, observer.helper_entered, "concurrent inspection")
                                await inspect_while_held("same_tenant_inspection_completed_while_native_held")
                            observer.helper_release.set()
                            if observer.pause_transport:
                                await _wait_for_owned_transport(observer, operation)
                                if concurrent:
                                    await inspect_while_held(
                                        "same_tenant_inspection_completed_while_outer_transport_blocked", shifted=True
                                    )
                                    _require_transport_reservation(observer, operation)
                                if spec["scenario"] == "cancel_transport":
                                    operation.cancel()
                                    await _wait_for_barrier(operation, observer.transport_cancelled, "transport cancellation")
                                    operation.cancel()
                                    await asyncio.sleep(0)
                                    _require_transport_reservation(observer, operation)
                                    observer.capture("repeated_transport_cancellation_holds_owned_memory")
                                    observer.transport_cancel_release.set()
                                    try:
                                        await operation
                                    except asyncio.CancelledError:
                                        pass
                                    else:
                                        raise RuntimeError("Cancelled native transport unexpectedly completed")
                                    committed = await get_review_store().get(job_id, tenant)
                                    if len(committed["revisions"]) != version + 1:
                                        raise RuntimeError("Transport cancellation lost or duplicated its committed revision")
                                    results.append(
                                        {
                                            "source_index": index,
                                            "repeat": repeat,
                                            "revision_id": committed["revisions"][-1]["id"],
                                        }
                                    )
                                    version += 1
                                    observer.capture("cancelled_transport_committed_revision_recovered")
                                    del committed, operation
                                    continue
                                observer.transport_release.set()
                            response = await operation
                        except BaseException:
                            operation.cancel()
                            observer.helper_release.set()
                            observer.publication_release.set()
                            observer.transport_release.set()
                            observer.transport_cancel_release.set()
                            await asyncio.gather(operation, return_exceptions=True)
                            raise
                        finally:
                            observer.helper_release.set()
                        observer.capture("response_sent")
                        if spec["scenario"] == "publication_failure":
                            del response, operation
                            continue
                        revision = response.json()["data"]["revision"]
                        results.append({"source_index": index, "repeat": repeat, "revision_id": revision["id"]})
                        del response, revision, operation
                        version += 1
            state = await get_review_store().get(job_id, tenant)
            if state["decisions"] or state["deliveries"] or len(state["revisions"]) != version:
                raise RuntimeError("Qualification unexpectedly changed approvals/deliveries or lost a revision")
            if spec["scenario"] not in {"success", "cancel_transport"}:
                published = await store.list_for_job(job_id if canonical else tenant + "__" + job_id)
                if any(entry.relative_path.startswith("photographic-review/") for entry in published):
                    raise RuntimeError("Failed native operation left an uncommitted publication")
            for result, revision in zip(results, state["revisions"], strict=True):
                products = _committed_products(
                    root,
                    {"tenant_id": tenant, "job_id": job_id, "provider": spec.get("provider", "retained-local")},
                    revision,
                    spec["sources"][result["source_index"]],
                    result["source_index"],
                )
                receipt_path = root / ("receipt-" + result["revision_id"] + ".json")
                offline._write_json(receipt_path, revision["receipt"])
                result.update(products=products, receipt_path=str(receipt_path))
            snapshot = service.admission.snapshot()
            if any(
                snapshot[key] for key in ("active", "queued", "outstanding", "memory_reserved_bytes", "scratch_reserved_bytes")
            ):
                raise RuntimeError("Managed qualification left owned reservations active")
            observer.capture("quiescent")
            transport_verified = spec["scenario"] in {"success", "cancel_transport"}
            if transport_verified and not any(
                record["phase"] == "native_outer_transport_holds_reservation_after_inner_return" for record in observer.records
            ):
                raise RuntimeError("Managed qualification did not prove deployed transport ownership")
            return {
                "job_id": job_id,
                "tenant_id": tenant,
                "results": results,
                "scenario": spec["scenario"],
                "phases": observer.records,
                "final_admission": snapshot,
                "transport_ownership_verified": transport_verified,
                "provider": spec.get("provider", "retained-local"),
                "canonical_fixture_import": canonical_import,
            }
        finally:
            observer.helper_release.set()
            observer.publication_release.set()
            observer.transport_release.set()
            observer.transport_cancel_release.set()
            await service.close()
            if canonical:
                await store.close()
            await _SharedEngine.dispose(os.environ["TP_DATABASE_URL"])


def _worker(specification: Path) -> None:
    import psutil

    signal.pthread_sigmask(signal.SIG_UNBLOCK, {signal.SIGINT, signal.SIGTERM})
    multiplier = 1 if sys.platform == "darwin" else 1024
    initial_high_water = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * multiplier)
    baseline = psutil.Process().memory_info().rss
    started = time.monotonic()
    spec = offline._read_json(specification, MAX_REPORT_BYTES)
    root = specification.parent
    _configure_worker(root)
    if runtime_identity() != spec["runtime"]:
        raise ValueError("Managed qualification runtime changed before execution")
    result = asyncio.run(_managed_run(root, spec))
    if runtime_identity() != spec["runtime"]:
        raise ValueError("Managed qualification runtime changed during execution")
    peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * multiplier)
    offline._write_json(
        root / "managed-result.json",
        {
            **result,
            "runtime": spec["runtime"],
            "baseline_rss_bytes": baseline,
            "initial_process_high_water_bytes": initial_high_water,
            "peak_rss_bytes": peak,
            "incremental_peak_rss_bytes": max(0, peak - baseline),
            "checkpoint_semantics": "cumulative_process_high_water_not_phase_exclusive",
            "baseline_semantics": "actual_rss_after_module_imports_before_managed_setup",
            "worker_seconds": time.monotonic() - started,
            "source_inventory_authority": (
                "postgres_canonical_committed_generation_unified_process"
                if spec.get("provider") == "canonical-s3"
                else "postgres_job_record_retained_inventory"
            ),
            "transport": "deployed_app_security_middleware_outer_asgi_finalizer_and_transport_send",
        },
    )


def _committed_products(
    root: Path, result: dict[str, Any], committed: dict[str, Any], source: dict[str, Any], source_index: int
) -> list[dict[str, Any]]:
    """Reconstruct storage authority from fresh Postgres state, never worker paths."""
    from transformation_portal.orchestrator.artifact_store.local import _normalize_job_id, _normalize_relative_path

    identity = _normalize_job_id(committed["id"])
    scope = _normalize_job_id(result["tenant_id"] + "__" + result["job_id"])
    canonical = result.get("provider") == "canonical-s3"
    store_root = root / "verified-products" / result["job_id"] if canonical else root / "artifacts" / scope
    variant = committed["variant"]
    original = committed.get("native_source", {})
    if (
        variant.get("kind") != "review_asset"
        or variant.get("storage_path") != f"photographic-review/{identity}/master.tif"
        or variant.get("artifact_path") != f"revisions/{identity}/master.tif"
        or variant.get("sha256") != committed["receipt"]["revision_sha256"]
        or variant.get("scene_id") != f"input-{source_index:04d}"
        or original.get("artifact_path") != f"v6/input-{source_index:04d}/delivery.tif"
        or original.get("sha256") != source["bound"]["sha256"]
        or original.get("source_identity_sha256") != source["bound"]["sha256"]
    ):
        raise ValueError("Durable managed variant/source authority changed")
    expected = {
        "materials-overlay.png": "image/png",
        "materials-preview.png": "image/png",
        "materials-diagnostics.json": "application/json",
    }
    products = committed.get("native_products")
    if (
        not isinstance(products, list)
        or len(products) != len(expected)
        or {row.get("filename") for row in products} != set(expected)
    ):
        raise ValueError("Durable managed diagnostic closure changed")
    rows = [{**variant, "content_type": "image/tiff"}]
    for row in products:
        if (
            row.get("kind") != "review_asset"
            or row.get("content_type") != expected[row["filename"]]
            or row.get("storage_path") != f"photographic-review/{identity}/{row['filename']}"
        ):
            raise ValueError("Durable managed diagnostic storage/content type changed")
        rows.append(row)
    if any(type(row.get("size_bytes")) is not int or not 0 < row["size_bytes"] <= offline.MAX_SOURCE_BYTES for row in rows):
        raise ValueError("Durable managed product byte bounds changed")
    return [
        {
            "path": str(store_root / _normalize_relative_path(row["storage_path"])),
            "sha256": row["sha256"],
            "size_bytes": row["size_bytes"],
            "content_type": row["content_type"],
            **({"storage_path": row["storage_path"], "storage_job_id": result["job_id"]} if canonical else {}),
        }
        for row in rows
    ]


def _verify_master_content(before: Any, after: Any, receipt: dict[str, Any]) -> None:
    if (
        before.content_hash() != receipt["input_master_content_sha256"]
        or after.content_hash() != receipt["output_master_content_sha256"]
    ):
        raise ValueError("Actual managed master content differs from its committed receipt")


def _s3_store(spec: dict[str, Any]) -> Any:
    from transformation_portal.orchestrator.artifact_store.s3 import S3ArtifactStore

    services = _canonical_services()
    os.environ.update(
        AWS_ACCESS_KEY_ID=services[_SERVICE_ENV[3]],
        AWS_SECRET_ACCESS_KEY=services[_SERVICE_ENV[4]],
        AWS_DEFAULT_REGION="us-east-1",
        AWS_EC2_METADATA_DISABLED="true",
    )
    return S3ArtifactStore(
        bucket=services[_SERVICE_ENV[2]],
        prefix="tp/managed-qualification/" + spec["service_namespace"],
        endpoint_url=services[_SERVICE_ENV[1]],
        region_name="us-east-1",
    )


async def _verify_canonical_inventory(spec: dict[str, Any], result: dict[str, Any]) -> None:
    from transformation_portal.ingest.canonical_json import canonicalize_json
    from transformation_portal.orchestrator.storage.operational import PostgresOperationalRecordStore
    from transformation_portal.orchestrator.storage.postgres import _SharedEngine

    url = _database_url()
    store = _s3_store(spec)
    try:
        records = PostgresOperationalRecordStore(database_url=url)
        locator = await records.get_locator(result["job_id"])
        if locator is None or locator.tenant_id != result["tenant_id"]:
            raise ValueError("Canonical source locator changed after managed worker exit")
        plan, bindings = await records.fetch_plan(locator), await records.fetch_execution_bindings(locator)
        manifest = await records.committed_manifest(result["job_id"], result["tenant_id"])
        imported = result["canonical_fixture_import"]
        if (
            hashlib.sha256(plan).hexdigest() != imported["plan_sha256"]
            or hashlib.sha256(plan).hexdigest() != spec["canonical_authority"]["plan_sha256"]
            or bindings is None
            or hashlib.sha256(bindings).hexdigest() != imported["bindings_sha256"]
            or manifest is None
            or hashlib.sha256(canonicalize_json(manifest)).hexdigest() != imported["generation_manifest_sha256"]
            or manifest["generation_id"] != imported["generation_id"]
            or len(manifest["files"]) != len(spec["sources"])
        ):
            raise ValueError("Canonical source generation or execution authority changed after managed worker exit")
        for index, (entry, source) in enumerate(zip(manifest["files"], spec["sources"], strict=True)):
            if (
                entry["path"] != f"v6/input-{index:04d}/delivery.tif"
                or entry["storage_path"] != f"generations/{manifest['generation_id']}/{entry['path']}"
                or any(entry[key] != source["bound"][key] for key in ("sha256", "size_bytes"))
            ):
                raise ValueError("Canonical committed source inventory differs from the retained fixture manifest")
            chunks = await store.open_bytes(result["job_id"], entry["storage_path"])
            count, digest = 0, hashlib.sha256()
            try:
                async for chunk in chunks:
                    count += len(chunk)
                    if count > entry["size_bytes"]:
                        raise ValueError("Canonical source object exceeds its committed length")
                    digest.update(chunk)
            finally:
                await chunks.aclose()
            if count != entry["size_bytes"] or digest.hexdigest() != entry["sha256"]:
                raise ValueError("Canonical source object differs from its committed bytes")
    finally:
        await store.close()
        await _SharedEngine.dispose(url)


async def _materialize_s3_products(root: Path, spec: dict[str, Any], products: list[dict[str, Any]]) -> None:
    """Verify fresh committed object bytes into the independent verifier's namespace."""
    from transformation_portal.portal.photographic_review_access import _stream_snapshot

    store = _s3_store(spec)
    try:
        for product in products:
            path = Path(product["path"])
            if not path.is_relative_to(root / "verified-products") or path.exists() or path.is_symlink():
                raise ValueError("Independent S3 verification requires a fresh owned product path")
            path.parent.mkdir(parents=True, exist_ok=True)
            chunks = await store.open_bytes(product["storage_job_id"], product["storage_path"])
            await _stream_snapshot(chunks, path, offline.MAX_SOURCE_BYTES, product["size_bytes"], product["sha256"])
    finally:
        await store.close()


def _verify(specification: Path) -> None:
    """Independent decoded verification after the managed worker has exited."""
    import numpy as np
    from PIL import Image

    from transformation_portal.lux_depth_v4.photography import linear_to_srgb, output_srgb_icc
    from transformation_portal.photographic_delivery.materials import validate_materials_revision_receipt
    from transformation_portal.photographic_delivery.models import DeliveryImage

    spec = offline._read_json(specification, MAX_REPORT_BYTES)
    root = specification.parent
    if runtime_identity() != spec["runtime"]:
        raise ValueError("Independent verification runtime changed before execution")
    result = offline._read_json(root / "managed-result.json", MAX_REPORT_BYTES)
    canonical = spec.get("provider") == "canonical-s3"
    if canonical:
        asyncio.run(_verify_canonical_inventory(spec, result))

    async def persisted() -> dict[str, Any]:
        from sqlalchemy import select

        from transformation_portal.orchestrator.models import PhotographicReviewEventModel
        from transformation_portal.orchestrator.photographic_review_store import PostgresReviewStore
        from transformation_portal.orchestrator.storage.postgres import _SharedEngine

        url = _database_url()
        try:
            state = await PostgresReviewStore(url).get(result["job_id"], result["tenant_id"])
            _, factory = await _SharedEngine.get(url)
            async with factory() as session:
                rows = (
                    await session.scalars(
                        select(PhotographicReviewEventModel)
                        .where(PhotographicReviewEventModel.job_id == result["job_id"])
                        .order_by(PhotographicReviewEventModel.version)
                    )
                ).all()
                if len(rows) != len(result["results"]):
                    raise ValueError("Durable managed event inventory differs after worker exit")
                for version, row in enumerate(rows, 1):
                    payload = json.loads(row.canonical_bytes)
                    if (
                        row.tenant_id != result["tenant_id"]
                        or row.version != version
                        or payload["version"] != version
                        or hashlib.sha256(row.canonical_bytes).hexdigest() != row.sha256
                        or offline._canonical(payload) != row.canonical_bytes
                    ):
                        raise ValueError("Durable managed event authority differs after worker exit")
                if (
                    rows
                    and json.loads(rows[-1].canonical_bytes)["projection_sha256"]
                    != hashlib.sha256(offline._canonical(state)).hexdigest()
                ):
                    raise ValueError("Durable managed projection differs from its committed event")
            return state
        finally:
            await _SharedEngine.dispose(url)

    state = asyncio.run(persisted())
    if state["decisions"] or state["deliveries"] or state["state_version"] != len(result["results"]):
        raise ValueError("Durable managed state differs after worker exit")
    verified = []
    for row, committed in zip(result["results"], state["revisions"], strict=True):
        source = spec["sources"][row["source_index"]]
        receipt_path = Path(row["receipt_path"])
        if not receipt_path.is_relative_to(root):
            raise ValueError("Managed receipt escaped its owned namespace")
        receipt = offline._read_json(receipt_path, 256 * 1024)
        receipt_bytes = offline._canonical(receipt)
        if receipt_path.read_bytes() != receipt_bytes:
            raise ValueError("Managed receipt canonical bytes changed")
        if committed["id"] != row["revision_id"] or committed["receipt"] != receipt:
            raise ValueError("Managed receipt differs from the durable committed revision")
        products = _committed_products(root, result, committed, source, row["source_index"])
        if row["products"] != products:
            raise ValueError("Managed worker inventory differs from fresh committed Postgres metadata")
        if canonical:
            asyncio.run(_materialize_s3_products(root, spec, products))
        for product in products:
            path = Path(product["path"])
            if not path.is_relative_to(root) or path.is_symlink():
                raise ValueError("Managed product escaped its owned namespace")
            with offline._open_regular(path, offline.MAX_SOURCE_BYTES) as stream:
                digest = hashlib.sha256()
                count = 0
                while block := stream.read(offline.MIB):
                    count += len(block)
                    if count > product["size_bytes"]:
                        raise ValueError("Committed managed product exceeded its byte identity")
                    digest.update(block)
                if (
                    os.fstat(stream.fileno()).st_size != product["size_bytes"]
                    or count != product["size_bytes"]
                    or digest.hexdigest() != product["sha256"]
                ):
                    raise ValueError("Committed managed product bytes changed")
        master = products[0]
        validate_materials_revision_receipt(receipt, master["sha256"])
        diagnostics = {Path(product["path"]).name: product for product in products[1:]}
        if set(diagnostics) != {entry["filename"] for entry in receipt["diagnostics"]}:
            raise ValueError("Managed diagnostic inventory differs from its receipt")
        for entry in receipt["diagnostics"]:
            if any(diagnostics[entry["filename"]][key] != entry[key] for key in ("sha256", "size_bytes")):
                raise ValueError("Managed diagnostic authority differs from its receipt")
        for name in ("materials-preview.png", "materials-overlay.png"):
            with Image.open(diagnostics[name]["path"]) as image:
                if (
                    image.format != "PNG"
                    or image.size != tuple(reversed(receipt["preview"]["shape"]))
                    or image.mode != ("RGBA" if receipt["alpha_mode"] else "RGB")
                    or image.info.get("icc_profile") != output_srgb_icc()
                ):
                    raise ValueError("Managed PNG diagnostic lost its shape, alpha or profile")
                image.load()
        descriptor = offline._read_json(Path(diagnostics["materials-diagnostics.json"]["path"]), 256 * 1024)
        if any(
            descriptor[key] != receipt[key]
            for key in (
                "revision_sha256",
                "input_master_content_sha256",
                "execution_output_master_content_sha256",
                "output_master_content_sha256",
                "evidence_sha256",
                "operations_sha256",
                "execution",
            )
        ):
            raise ValueError("Managed diagnostic descriptor differs from its committed execution")
        if receipt["source_master_sha256"] != source["bound"]["sha256"] or receipt["recipe"] != source["recipe"]:
            raise ValueError("Committed managed receipt changed source or recipe")
        before = DeliveryImage(
            "source.tif", source["bound"]["sha256"], "qualification", source_path=Path(source["path"])
        ).decode()
        after = DeliveryImage("master.tif", master["sha256"], "qualification", source_path=Path(master["path"])).decode()
        if before.shape != after.shape or (before.alpha is None) != (after.alpha is None) or after.source_bit_depth != 16:
            raise ValueError("Managed output lost native geometry, alpha or precision")
        _verify_master_content(before, after, receipt)
        difference = int(
            np.max(
                np.abs(
                    np.rint(linear_to_srgb(before.pixels) * 65535).astype(np.int32)
                    - np.rint(linear_to_srgb(after.pixels) * 65535).astype(np.int32)
                )
            )
        )
        zero = all(region["strength"] == 0 for region in source["recipe"]["regions"])
        if zero and difference > 1:
            raise ValueError("Managed zero-effect output exceeds one 16-bit code")
        if before.alpha is not None and float(np.max(np.abs(before.alpha - after.alpha))) > 1 / 65535:
            raise ValueError("Managed output changed straight alpha")
        verified.append(
            {
                "revision_id": row["revision_id"],
                "products_verified": len(products) + 1,
                "zero_effect": zero,
                "maximum_16bit_code_delta": difference,
                "source_index": row["source_index"],
                "product_identity_sha256": hashlib.sha256(
                    offline._canonical(
                        {
                            "products": sorted(
                                (Path(product["path"]).name, product["sha256"], product["size_bytes"]) for product in products
                            ),
                            "receipt_sha256": hashlib.sha256(receipt_bytes).hexdigest(),
                        }
                    )
                ).hexdigest(),
            }
        )
        del before, after
    if runtime_identity() != spec["runtime"]:
        raise ValueError("Independent verification runtime changed during execution")
    offline._write_json(root / "verification.json", {"scope": "independent_after_managed_worker_exit", "results": verified})


def _run_process(root: Path, mode: str, *, memory_limit: int, scratch_limit: int, timeout: int = 300) -> dict[str, int]:
    import psutil

    log = root / (mode + ".log")
    environment = {
        key: value
        for key, value in os.environ.items()
        if key in {"PATH", "HOME", "LANG", "SYSTEMROOT", "TP_MANAGED_QUALIFICATION_DATABASE_URL", *_SERVICE_ENV}
    }
    scratch = root / "scratch"
    scratch.mkdir(exist_ok=True)
    environment.update(
        PYTHONPATH=os.pathsep.join([str(Path(__file__).resolve().parents[2]), str(Path(__file__).resolve().parents[3])]),
        TMPDIR=str(scratch),
        TMP=str(scratch),
        TEMP=str(scratch),
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        NUMEXPR_NUM_THREADS="1",
        VECLIB_MAXIMUM_THREADS="1",
    )
    spec = offline._read_json(root / "specification.json", MAX_REPORT_BYTES)
    publication_root = Path(spec["s3_publication_root"]) if spec.get("provider") == "canonical-s3" else None
    object_root = (
        publication_root / _canonical_services()[_SERVICE_ENV[2]] / "tp/managed-qualification" / spec["service_namespace"]
        if publication_root is not None
        else None
    )
    started, peak, scratch_peak = time.monotonic(), 0, 0
    with log.open("xb") as output:
        previous = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGINT, signal.SIGTERM})
        try:
            process = subprocess.Popen(
                [sys.executable, "-m", MODULE, mode, str(root / "specification.json")],
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=output,
                stderr=output,
                start_new_session=True,
            )
        except BaseException:
            signal.pthread_sigmask(signal.SIG_SETMASK, previous)
            raise
        try:
            signal.pthread_sigmask(signal.SIG_SETMASK, previous)
            owned = psutil.Process(process.pid)
            while process.poll() is None:
                try:
                    peak = max(peak, owned.memory_info().rss)
                    if owned.children(recursive=True):
                        raise RuntimeError("Managed qualification worker unexpectedly launched another process")
                except psutil.NoSuchProcess:
                    pass
                current_scratch = offline._scratch_usage(root)["accounted_bytes"]
                if object_root is not None and object_root.exists():
                    current_scratch += offline._scratch_usage(object_root)["accounted_bytes"]
                scratch_peak = max(scratch_peak, current_scratch)
                if peak > memory_limit or psutil.virtual_memory().available < 8 * offline.GIB:
                    raise RuntimeError("Managed qualification memory bound exceeded")
                if scratch_peak > scratch_limit or shutil.disk_usage(root).free < 4 * offline.GIB + max(
                    0, scratch_limit - scratch_peak
                ):
                    raise RuntimeError("Managed qualification scratch/headroom bound exceeded")
                if publication_root is not None and shutil.disk_usage(publication_root).free < 4 * offline.GIB + max(
                    0, scratch_limit - scratch_peak
                ):
                    raise RuntimeError("Managed qualification provider scratch/headroom bound exceeded")
                if log.stat().st_size > offline.MIB or time.monotonic() - started > timeout:
                    raise RuntimeError("Managed qualification time/log bound exceeded")
                time.sleep(0.05)
            if process.returncode:
                raise RuntimeError(f"Managed qualification {mode} failed; inspect its private log")
        finally:
            # Shield repeated cancellation while draining only this fresh group.
            mask = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGINT, signal.SIGTERM})
            try:
                for event in (signal.SIGTERM, signal.SIGKILL):
                    try:
                        os.killpg(process.pid, event)
                    except ProcessLookupError:
                        pass
                    try:
                        process.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        pass
                process.wait()
                deadline = time.monotonic() + 3
                while True:
                    try:
                        os.killpg(process.pid, 0)
                    except ProcessLookupError:
                        break
                    if time.monotonic() >= deadline:
                        raise RuntimeError("Managed qualification process group survived its owned drain")
                    time.sleep(0.01)
            finally:
                signal.pthread_sigmask(signal.SIG_SETMASK, mask)
    return {"sampled_peak_rss_bytes": peak, "sampled_peak_accounted_bytes": scratch_peak}


def qualify(
    manifest: Path,
    output: Path,
    expected_profile_identity: str,
    *,
    repeats: int = 3,
    concurrent_inspection: bool = True,
    preflight_only: bool = False,
    scenario: str = "success",
    provider: str = "retained-local",
    canonical_plan: Path | None = None,
    canonical_bindings: Path | None = None,
    s3_publication_root: Path | None = None,
    s3_container: str | None = None,
    process_timeout_seconds: int = 300,
) -> dict[str, Any]:
    """Run repeated cohorts in one process, then verify in a separate process."""
    _database_url()
    if type(process_timeout_seconds) is not int or not 1 <= process_timeout_seconds <= 900:
        raise ValueError("Managed qualification process timeout requires an integer from 1 to 900 seconds")
    if (
        type(repeats) is not int
        or not 1 <= repeats <= 3
        or type(concurrent_inspection) is not bool
        or type(preflight_only) is not bool
        or scenario not in SCENARIOS
        or provider not in PROVIDERS
        or (scenario != "success" and repeats != 1)
    ):
        raise ValueError("Managed qualification requires one to three repeats and explicit boolean options")
    from transformation_portal.core.image_artifact import validate_source_sha256
    from transformation_portal.orchestrator.photographic_admission import ReviewAdmissionConfig

    validate_source_sha256(expected_profile_identity)
    sources = _manifest(manifest)
    authority, minio = None, None
    if provider == "canonical-s3":
        services = _canonical_services()
        authority = _canonical_authority(canonical_plan, canonical_bindings)
        if len(authority["originals"]) != len(sources):
            raise ValueError("Canonical qualification requires one plan input per retained fixture source")
        minio = _minio_receipt(s3_container, s3_publication_root, services[_SERVICE_ENV[1]])
    elif any(value is not None for value in (canonical_plan, canonical_bindings, s3_publication_root, s3_container)):
        raise ValueError("Canonical authority and service options require the canonical-s3 provider")
    preservation_only = all(region["strength"] == 0 for source in sources for region in source["recipe"]["regions"])
    if scenario != "success" and len(sources) != 1:
        raise ValueError("Failure qualification requires one explicit retained source and one repeat")
    output = output.expanduser().absolute()
    if output.exists() or output.is_symlink() or output.parent != output.parent.resolve(strict=True):
        raise ValueError("Managed qualification output requires a fresh non-symlink directory")
    if any(output.is_relative_to(Path(row["path"]).parent) for row in sources):
        raise ValueError("Managed qualification output must remain separate from retained sources")
    runtime = runtime_identity()
    scratch, reasons = (
        _headroom(sources, repeats, output.parent, authority, s3_publication_root)
        if authority is not None
        else _headroom(sources, repeats, output.parent)
    )
    if runtime["managed_profile_identity"] != expected_profile_identity:
        reasons.append("governed_runtime_profile_mismatch")
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "blocked" if reasons else "planned",
        "blocking_reasons": reasons,
        "scope": (
            "private_managed_cpu_canonical_unified_redis_loopback_s3_postgres_not_frontdoor_or_photographic_acceptance"
            if provider == "canonical-s3"
            else "private_managed_cpu_local_store_postgres_not_frontdoor_or_photographic_acceptance"
        ),
        "provider": provider,
        "minio_service_receipt": minio,
        "runtime": runtime,
        "source_sha256": [row["bound"]["sha256"] for row in sources],
        "repeats": repeats,
        "scenario": scenario,
        "process_timeout_seconds": process_timeout_seconds,
        "canonical_authority": (
            {key: authority[key] for key in ("plan_sha256", "bindings_sha256", "scope")} if authority is not None else None
        ),
        "s3_publication_root": str(s3_publication_root) if s3_publication_root is not None else None,
        "config": asdict(ReviewAdmissionConfig(enabled=True)),
        "scratch_reservation_bytes": scratch,
        "canonical_scratch_plan": _canonical_scratch_plan(sources, repeats, authority) if authority is not None else None,
        "production_flags_changed": False,
        "model_inference_performed": False,
        "aws_gpu_validation_performed": False,
        "photographic_acceptance": "not_established",
        "database": {"backend": "postgres", "dedicated_test_database": True},
        "encoded_products_repeatable": None,
        "managed_profile_qualification": "not_established",
        "native_pixel_validation": (
            "zero_effect_preservation"
            if preservation_only
            else "nonzero_observations_without_independent_protected_delta_acceptance"
        ),
    }
    output.mkdir(mode=0o700)
    offline._write_json(output / "qualification.json", report)
    if reasons or preflight_only:
        return report
    specification = {
        "runtime": runtime,
        "sources": sources,
        "repeats": repeats,
        "concurrent_inspection": concurrent_inspection,
        "scenario": scenario,
        "process_timeout_seconds": process_timeout_seconds,
        "provider": provider,
        "canonical_authority": authority,
        "s3_publication_root": str(s3_publication_root) if s3_publication_root is not None else None,
        "service_namespace": uuid.uuid4().hex,
    }
    offline._write_json(output / "specification.json", specification)
    try:
        with offline._supervisor_signals():
            observed = _run_process(
                output, "_worker", memory_limit=12 * offline.GIB, scratch_limit=scratch, timeout=process_timeout_seconds
            )
            result = offline._read_json(output / "managed-result.json", MAX_REPORT_BYTES)
            if result["runtime"] != runtime:
                raise ValueError("Managed worker evidence runtime differs")
            observed_peak = max(result["peak_rss_bytes"], observed.get("sampled_peak_rss_bytes", 0))
            incremental_peak = max(0, observed_peak - result["baseline_rss_bytes"])
            margin = math.ceil(incremental_peak * 1.5)
            _run_process(
                output, "_verify", memory_limit=12 * offline.GIB, scratch_limit=scratch, timeout=process_timeout_seconds
            )
            verification = offline._read_json(output / "verification.json", MAX_REPORT_BYTES)
            for row in sources:
                if offline._source(Path(row["path"]), row["bound"]["sha256"]) != row["bound"]:
                    raise ValueError("Retained source changed during managed qualification")
            if authority is not None:
                for row in authority["originals"]:
                    if _bound_file(Path(row["path"]), row["bound"]["sha256"], row["bound"]["size_bytes"]) != row["bound"]:
                        raise ValueError("Canonical original source changed during managed qualification")
            repeatable = None
            if repeats >= 2 and scenario == "success":
                repeated = verification["results"]
                repeatable = all(
                    len({row["product_identity_sha256"] for row in repeated if row["source_index"] == index}) == 1
                    and sum(row["source_index"] == index for row in repeated) == repeats
                    for index in range(len(sources))
                )
            measured_reasons = []
            if scenario in {"success", "cancel_transport"} and result.get("transport_ownership_verified") is not True:
                measured_reasons.append("deployed_transport_ownership_unverified")
            if margin > 6 * offline.GIB:
                measured_reasons.append("measured_margin_exceeds_native_profile")
            if repeatable is False:
                measured_reasons.append("encoded_products_not_repeatable")
            report.update(
                status="not_qualified" if measured_reasons else "measured",
                blocking_reasons=measured_reasons,
                measured_incremental_peak_plus_50_percent_bytes=margin,
                complete_managed_result=result,
                independent_verification=verification,
                sampled=observed,
                conservative_complete_peak_rss_bytes=observed_peak,
                conservative_incremental_peak_rss_bytes=incremental_peak,
                source_unchanged=True,
                encoded_products_repeatable=repeatable,
                managed_profile_qualification=(
                    "passed_for_declared_cohort"
                    if not measured_reasons
                    and margin <= 6 * offline.GIB
                    and repeatable is True
                    and concurrent_inspection
                    and preservation_only
                    else "not_established"
                ),
            )
    except BaseException:
        report.update(status="failed", source_unchanged=None)
        offline._write_json(output / "qualification.json", report, replace_owned_report=True)
        raise
    offline._write_json(output / "qualification.json", report, replace_owned_report=True)
    return report


def main() -> int:
    if len(sys.argv) == 3 and sys.argv[1] in {"_worker", "_verify"}:
        (_worker if sys.argv[1] == "_worker" else _verify)(Path(sys.argv[2]))
        return 0
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-profile-identity", required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--no-concurrent-inspection", action="store_true")
    parser.add_argument("--scenario", choices=sorted(SCENARIOS), default="success")
    parser.add_argument("--provider", choices=sorted(PROVIDERS), default="retained-local")
    parser.add_argument("--canonical-plan", type=Path)
    parser.add_argument("--canonical-bindings", type=Path)
    parser.add_argument("--s3-publication-root", type=Path)
    parser.add_argument("--s3-container")
    parser.add_argument("--process-timeout-seconds", type=int, default=300)
    arguments = parser.parse_args()
    report = qualify(
        arguments.manifest,
        arguments.output,
        arguments.expected_profile_identity,
        repeats=arguments.repeats,
        concurrent_inspection=not arguments.no_concurrent_inspection,
        preflight_only=arguments.preflight_only,
        scenario=arguments.scenario,
        provider=arguments.provider,
        canonical_plan=arguments.canonical_plan,
        canonical_bindings=arguments.canonical_bindings,
        s3_publication_root=arguments.s3_publication_root,
        s3_container=arguments.s3_container,
        process_timeout_seconds=arguments.process_timeout_seconds,
    )
    print(dumps_json({"schema": SCHEMA, "status": report["status"], "blocking_reasons": report["blocking_reasons"]}))
    return 0 if report["status"] in {"planned", "measured"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
