"""App adapter for authenticated, hash-bound photographic review IO.

The browser never supplies physical paths or storage keys. Existing artifact
retrieval remains authoritative for indexed outputs; original references come
only from immutable photographic dispatch plans plus their physical bindings.
Review derivatives are readable only while referenced by the saved review state.
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import os
import re
import stat
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any, AsyncIterable, AsyncIterator, Awaitable, Callable, Mapping, TypeVar

from fastapi import Request
from starlette.responses import FileResponse, Response, StreamingResponse

from transformation_portal.core.execution_plan import decode_bounded_json_object
from transformation_portal.ingest.canonical_json import canonicalize_json
from transformation_portal.lux_depth_v3.execution_evidence import (
    _open_confined_artifact,
    _pin_output_root,
    _validate_confined_entry_identity,
)
from transformation_portal.lux_depth_v4.io import directory_path, snapshot
from transformation_portal.orchestrator.artifact_store.generation import _verify_staged
from transformation_portal.orchestrator.photographic_admission import (
    ReviewOverload,
    current_review_lease,
    owned_review_directory,
    run_review_thread,
)
from transformation_portal.orchestrator.photographic_review_access import (
    ReviewAccessError,
    ReviewContext,
    ReviewRuntime,
)
from transformation_portal.orchestrator.photographic_review_store import get_review_store

_SHA = re.compile(r"^[a-f0-9]{64}$")
_SCENE = re.compile(r"(?:^|/)(input-[0-9]{4,})(?:/|$)")
_PHOTOGRAPHS = frozenset({".tif", ".tiff", ".jpg", ".jpeg", ".png"})
MAX_REVIEW_IMAGE_BYTES = 512 * 1024**2
MAX_REVIEW_ZIP_BYTES = 16 * 1024**3
MAX_NATIVE_DESCRIPTOR_BYTES = 2 * 1024**2
MAX_NATIVE_ICC_BYTES = 16 * 1024**2
_Result = TypeVar("_Result")


async def _owned_operation(operation: Awaitable[_Result], *, completed: Callable[[], None] | None = None) -> _Result:
    """Drain owned IO before cancellation releases snapshots or publication keys.

    The completion hook records a successful immutable write even when caller
    cancellation arrived during the operation, so its exact new key can be
    rolled back after the write has stopped. Failed writes never acquire it.
    """
    worker = asyncio.ensure_future(operation)
    try:
        result = await asyncio.shield(worker)
    except asyncio.CancelledError:
        while not worker.done():
            try:
                await asyncio.shield(worker)
            except asyncio.CancelledError:
                continue
            except Exception:
                break
        if not worker.cancelled() and worker.exception() is None and completed is not None:
            completed()
        raise
    if completed is not None:
        completed()
    return result


def _icc_bytes(data: bytes) -> bytes:
    """Read only the bounded uint8 NPY carrier emitted by photographic writers."""
    import numpy as np

    stream = io.BytesIO(data)
    version = np.lib.format.read_magic(stream)
    if version == (1, 0):
        shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream)
    elif version == (2, 0):
        shape, fortran, dtype = np.lib.format.read_array_header_2_0(stream)
    else:
        raise ReviewAccessError("Native ICC carrier version is unsupported")
    if dtype != np.dtype("uint8") or fortran or len(shape) != 1 or not 0 < shape[0] <= MAX_NATIVE_ICC_BYTES:
        raise ReviewAccessError("Native ICC carrier must contain bounded raw profile bytes")
    profile = stream.read(MAX_NATIVE_ICC_BYTES + 1)
    if len(profile) != shape[0]:
        raise ReviewAccessError("Native ICC carrier has an invalid byte length")
    return profile


def _publication_snapshot(source: Path, target: Path, maximum_bytes: int) -> tuple[int, str]:
    """Copy a pinned, stable regular source into an owned private staging path."""
    root_path = directory_path(source.parent)
    with _pin_output_root(root_path) as root:
        descriptor, relative = _open_confined_artifact(root, root_path / source.name)
        try:
            before = os.fstat(descriptor)
            if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or not 0 < before.st_size <= maximum_bytes:
                raise ReviewAccessError("Review publication source must be a bounded regular file")
            digest, size = hashlib.sha256(), 0
            with target.open("xb") as output:
                while chunk := os.read(descriptor, 1024 * 1024):
                    size += len(chunk)
                    if size > min(before.st_size, maximum_bytes):
                        raise ReviewAccessError("Review publication source exceeded its byte budget")
                    digest.update(chunk)
                    output.write(chunk)
                output.flush()
                os.fsync(output.fileno())
            after = os.fstat(descriptor)
            if size != before.st_size or (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
            ):
                raise ReviewAccessError("Review publication source changed during staging")
            _validate_confined_entry_identity(root, relative, before, context="review publication source")
            return size, digest.hexdigest()
        finally:
            os.close(descriptor)


def _safe_relative(value: Any) -> str:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise ReviewAccessError("Invalid review artifact path", status=400, code="INVALID_ARGUMENT")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in value.split("/")):
        raise ReviewAccessError("Invalid review artifact path", status=400, code="INVALID_ARGUMENT")
    return value


def _identity(reference: Mapping[str, Any], maximum_bytes: int) -> tuple[str, int]:
    sha, size = reference.get("sha256"), reference.get("size_bytes")
    if not isinstance(sha, str) or not _SHA.fullmatch(sha) or type(size) is not int or size <= 0:
        raise ReviewAccessError("Review source lacks verified content identity")
    if type(maximum_bytes) is not int or maximum_bytes <= 0 or size > maximum_bytes:
        raise ReviewAccessError("Review source exceeds the admitted byte budget", status=413, code="RESOURCE_EXHAUSTED")
    return sha, size


def _photo(reference: Mapping[str, Any]) -> bool:
    path = PurePosixPath(str(reference.get("path") or reference.get("relative_path") or ""))
    name = path.name.lower()
    if path.suffix.lower() not in _PHOTOGRAPHS or any(word in name for word in ("depth", "mask", "preview", "proxy")):
        return False
    return name == "delivery.tif" or "enhanced" in name or name in {"final.tif", "final.tiff", "final.jpg", "final.png"}


def _saved_assets(state: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract only saved server-owned derivative records, never browser refs."""
    assets: list[dict[str, Any]] = []
    candidates = [row.get("variant") for row in state.get("revisions", []) if isinstance(row, dict)]
    candidates.extend(item for row in state.get("deliveries", []) if isinstance(row, dict) for item in row.get("files", []))
    for value in candidates:
        if not isinstance(value, dict):
            continue
        storage = value.get("storage_path")
        if value.get("kind") == "review_asset" and isinstance(storage, str) and storage.startswith("photographic-review/"):
            _safe_relative(storage)
            assets.append(dict(value))
    return assets


class AppReviewAccess:
    """Reuse app auth/path/store helpers without importing its module globally."""

    def __init__(self, app_module: Any):
        self.app = app_module

    def runtime(self) -> ReviewRuntime:
        return ReviewRuntime(
            authorize=self.authorize,
            error_response=self.app._error_response,
            authorize_history=self.authorize_history,
            authorize_admission=self.authorize_admission,
        )

    def _storage_job_id(self, job: Any) -> str:
        # Match the parent's ordinary retention/deletion namespace. Canonical
        # generations use the global job ID; legacy tenant stores namespace it.
        return job.id if self.app._uses_dispatch_authority() else self.app._artifact_storage_job_id(job)

    async def _job(self, request: Request, job_id: str) -> tuple[Any, Any] | Response:
        app = self.app
        if not app.API_KEY_SECRET:
            return app._error_response(
                503,
                code="AUTH_CONFIGURATION_ERROR",
                message="photographic review requires configured authentication",
                details={"job_id": job_id},
            )
        if not app._has_valid_api_key(request):
            return app._error_response(
                401, code="UNAUTHORIZED", message="invalid or missing API key", details={"job_id": job_id}
            )
        tenant, error = await app._pilot_tenant_from_request(request, action="photographic_review")
        if error is not None:
            return error
        job = await app._load_job_view(job_id)
        if job is None:
            return app._error_response(404, code="NOT_FOUND", message="job not found", details={"job_id": job_id})
        error = await app._pilot_enforce_job_tenant(tenant, job, action="photographic_review", request=request)
        if error is not None:
            return error
        if app._artifacts_deleted(job):
            return app._error_response(410, code="ARTIFACT_DELETED", message="artifacts deleted", details={"job_id": job_id})
        if job.state != "succeeded":
            return app._error_response(
                409, code="CONFLICT", message="photographic review requires a succeeded job", details={"job_id": job_id}
            )
        return job, tenant

    async def _originals(self, job: Any, tenant: Any) -> tuple[list[dict[str, Any]], str | None]:
        app = self.app
        if not app._uses_dispatch_authority():
            return [], "immutable_original_binding_unavailable"
        records = app.get_operational_record_store()
        locator = await records.get_locator(job.id)
        if locator is None:
            return [], "immutable_original_binding_unavailable"
        if tenant is not None and locator.tenant_id != tenant.tenant_id:
            raise ReviewAccessError("Original tenant binding mismatch", status=403, code="FORBIDDEN")
        raw = await records.fetch_plan(locator)
        bindings = await records.fetch_execution_bindings(locator)
        if bindings is None:
            return [], "immutable_original_binding_unavailable"
        from transformation_portal.orchestrator.lux_depth_adapter import (
            UnifiedBindings,
            is_unified_bindings,
            validate_unified_dispatch,
        )
        from transformation_portal.orchestrator.photography_adapter import PhotographyBindings, validate_photography_dispatch
        from transformation_portal.orchestrator.photography_v6_adapter import validate_v6_dispatch

        plan = decode_bounded_json_object(raw)
        if is_unified_bindings(bindings):
            validate_unified_dispatch(raw, bindings)
            bindings = UnifiedBindings(bindings).photography_bindings_bytes
        elif plan.get("schema") == "tp.execution.plan.v5":
            validate_v6_dispatch(raw, bindings)
        elif plan.get("schema") == "tp.execution.plan.v4":
            validate_photography_dispatch(raw, bindings)
        else:
            return [], "immutable_photographic_plan_unavailable"
        # V6 freezes finishing in the outer V5 plan; its validated V4 inference
        # child remains the sole input and source-color authority.
        if plan["schema"] == "tp.execution.plan.v5":
            plan = plan["inference"]
        bound = PhotographyBindings(bindings).to_payload()
        policy = app._execution_policy()
        root = Path(bound["input_root"])
        if policy.resolve_path(str(root), policy.input_roots) != root:
            raise ReviewAccessError("Original source is no longer authorized", status=403, code="FORBIDDEN")
        if tenant is not None:
            policy.tenant_guard(tenant).enforce_path(root)
        result = []
        for item in plan["inputs"]:
            relative = _safe_relative(item["path"])
            if PurePosixPath(relative).suffix.lower() not in _PHOTOGRAPHS:
                continue
            scene_id = item["id"]
            result.append(
                {
                    "artifact_path": f"original/{scene_id}{PurePosixPath(relative).suffix.lower()}",
                    "sha256": item["sha256"],
                    "source_identity_sha256": item["sha256"],
                    "size_bytes": item["size_bytes"],
                    "scene_id": scene_id,
                    "role": "original",
                    "source_name": PurePosixPath(relative).name,
                    "label": PurePosixPath(relative).stem,
                    "input_color": plan["configuration"]["input_color"],
                    "kind": "original",
                    "private_original_root": str(root),
                    "private_original_path": relative,
                }
            )
        return result, None if result else "original_format_not_supported_for_review"

    async def _references(self, request: Request, job: Any, tenant: Any) -> list[dict[str, Any]]:
        app = self.app
        if app._uses_dispatch_authority():
            records = app.get_operational_record_store()
            locator = await records.get_locator(job.id)
            if locator is None:
                return []
            if tenant is not None and locator.tenant_id != tenant.tenant_id:
                raise ReviewAccessError("Artifact tenant binding mismatch", status=403, code="FORBIDDEN")
            manifest = await records.committed_manifest(job.id, locator.tenant_id)
            items = (manifest or {}).get("files", [])
        else:
            items = job.artifacts.get("items", [])
        inventory = list(items)
        items = [item for item in inventory if _photo(item)]
        # Unified process retains its V5 predecessor as reconstruction evidence,
        # never as an independently reviewed revision. Its final contract is V6.
        v6 = [item for item in items if str(item.get("path") or item.get("relative_path") or "").startswith("v6/")]
        items = v6 or [
            item
            for item in items
            if "source-v5" not in PurePosixPath(str(item.get("path") or item.get("relative_path"))).parts
        ]
        originals, unavailable = await self._originals(job, tenant)
        names = {item["scene_id"]: item["label"] for item in originals}
        references = list(originals)
        final_scenes: set[str] = set()
        for item in items:
            path = _safe_relative(item.get("path") or item.get("relative_path"))
            match = _SCENE.search(path)
            scene_id = match.group(1) if match else PurePosixPath(path).stem
            if scene_id in final_scenes:
                raise ReviewAccessError("Photographic final selection is ambiguous", status=409, code="CONFLICT")
            final_scenes.add(scene_id)
            reference = {
                "artifact_path": path,
                "sha256": item.get("sha256"),
                "size_bytes": item.get("size_bytes"),
                "scene_id": scene_id,
                "role": "final",
                "source_name": PurePosixPath(path).name,
                "label": names.get(scene_id, scene_id),
                "input_color": "auto",
                "kind": "indexed",
                "identity_status": "retained_artifact_identity",
            }
            if reference["sha256"] is None and not app._uses_dispatch_authority():
                identity = await self._fresh_indexed_identity(request, job, reference)
                reference.update(identity)
                reference["identity_status"] = "fresh_snapshot_not_native_acceptance"
            _identity(reference, 1024**4)
            reference["source_identity_sha256"] = reference["sha256"]
            if unavailable:
                reference["original_unavailable_reason"] = unavailable
            references.append(reference)
            if app._uses_dispatch_authority():
                native = await self._native_reference(request, job, reference, inventory)
                if native is not None:
                    references.append(native)
        return references

    async def _read_indexed(self, request: Request, job: Any, ref: dict, maximum_bytes: int) -> bytes:
        expected_sha, expected_size = _identity(ref, maximum_bytes)
        chunks = await self._indexed_chunks(request, job, ref["artifact_path"], maximum_bytes)
        with owned_review_directory(prefix="tp-native-review-") as directory:
            target = Path(directory) / "snapshot"
            await _owned_operation(_stream_snapshot(chunks, target, maximum_bytes, expected_size, expected_sha))
            return await _owned_operation(run_review_thread(target.read_bytes))

    async def _native_reference(self, request: Request, job: Any, final: dict, inventory: list[dict]) -> dict | None:
        """Select retained native components exclusively from committed inventory.

        The fixed V5/V6 carrier names are producer contracts, never filesystem
        discovery. No inferred sibling path grants read authority or substitutes
        for a missing descriptor/alpha inventory entry.
        """
        parent = PurePosixPath(final["artifact_path"]).parent
        records: dict[str, dict] = {}
        for item in inventory:
            raw_path = item.get("path") or item.get("relative_path")
            if not isinstance(raw_path, str):
                continue
            path = PurePosixPath(_safe_relative(raw_path))
            if path.parent != parent or path.name not in {"master.npy", "photograph.json", "alpha.npy", "source-icc.npy"}:
                continue
            if path.name in records:
                raise ReviewAccessError("Native master inventory is ambiguous")
            records[path.name] = {
                "artifact_path": raw_path,
                "sha256": item.get("sha256"),
                "size_bytes": item.get("size_bytes"),
                "kind": "indexed",
                "source_name": path.name,
                "scene_id": final["scene_id"],
                "role": "native_component",
                "input_color": "linear_srgb",
                "identity_status": "committed_inventory_identity",
            }
        if "photograph.json" not in records:
            final["working_master_unavailable_reason"] = "native_descriptor_not_in_committed_inventory"
            return None
        raw = await self._read_indexed(request, job, records["photograph.json"], MAX_NATIVE_DESCRIPTOR_BYTES)
        descriptor = decode_bounded_json_object(raw)
        if descriptor.get("schema") not in {
            "tp.lux.photograph.v2",
            "tp.lux.photograph.v3",
            "tp.lux.graded_photograph.v1",
            "tp.lux.graded_photograph.v2",
        }:
            raise ReviewAccessError("Native photographic descriptor schema is unsupported")
        if descriptor.get("input_id") != final["scene_id"]:
            raise ReviewAccessError("Native descriptor belongs to a different scene")
        from transformation_portal.photographic_delivery.presentation import descriptor_policy

        policy = descriptor_policy(
            descriptor,
            descriptor_sha256=records["photograph.json"]["sha256"],
            artifact_sha256=final["sha256"],
            source_identity_sha256=final["source_identity_sha256"],
            delivery_path=str(PurePosixPath(final["artifact_path"]).relative_to(parent.parent)),
        )
        if policy is not None:
            final.update(presentation_policy=policy.to_payload(), presentation_policy_sha256=policy.sha256)
        if "master.npy" not in records:
            final["working_master_unavailable_reason"] = "native_master_not_in_committed_inventory"
            return None
        master = descriptor.get("master")
        if (
            not isinstance(master, dict)
            or master.get("schema") != "tp.image.master.v1"
            or master.get("color_space") != "linear_srgb"
        ):
            raise ReviewAccessError("Native descriptor lacks explicit linear-sRGB master authority")
        alpha = records.get("alpha.npy")
        if master.get("alpha_mode") == "straight":
            if alpha is None:
                final["working_master_unavailable_reason"] = "native_alpha_not_in_committed_inventory"
                return None
            _identity(alpha, MAX_REVIEW_IMAGE_BYTES)
        elif master.get("alpha_mode") is not None or alpha is not None:
            raise ReviewAccessError("Native alpha inventory disagrees with the descriptor")
        profile = None
        if "source-icc.npy" in records:
            carrier = records["source-icc.npy"]
            carrier_bytes = await self._read_indexed(request, job, carrier, MAX_NATIVE_ICC_BYTES + 65536)
            profile_bytes = _icc_bytes(carrier_bytes)
            profile_sha = hashlib.sha256(profile_bytes).hexdigest()
            if master.get("source_icc_sha256") != profile_sha:
                raise ReviewAccessError("Native ICC carrier disagrees with its retained descriptor")
            profile = {
                **carrier,
                "artifact_path": str(parent / "source-icc.raw"),
                "source_name": "source.icc",
                "sha256": profile_sha,
                "size_bytes": len(profile_bytes),
                "kind": "native_icc",
                "carrier_ref": carrier,
            }
        main = records["master.npy"]
        _identity(main, MAX_REVIEW_IMAGE_BYTES)
        identity = {
            "schema": "tp.photo.source_identity.v1",
            "master_sha256": main["sha256"],
            "descriptor_sha256": records["photograph.json"]["sha256"],
            "alpha_sha256": alpha["sha256"] if alpha is not None else None,
            "source_icc_sha256": profile["sha256"] if profile is not None else None,
        }
        reference = {
            **main,
            "role": "working_master",
            "label": final["label"],
            "version": "native",
            "source_identity_sha256": hashlib.sha256(canonicalize_json(identity)).hexdigest(),
            "native": {"descriptor": records["photograph.json"], "alpha": alpha, "source_icc": profile},
        }
        policy = descriptor_policy(
            descriptor,
            descriptor_sha256=records["photograph.json"]["sha256"],
            artifact_sha256=main["sha256"],
            source_identity_sha256=reference["source_identity_sha256"],
        )
        if policy is not None:
            reference.update(presentation_policy=policy.to_payload(), presentation_policy_sha256=policy.sha256)
        return reference

    async def authorize(self, request: Request, job_id: str) -> ReviewContext | Response:
        return await self._authorize(request, job_id, include_references=True)

    async def authorize_history(self, request: Request, job_id: str) -> ReviewContext | Response:
        """Authenticate retained history without discovering or reading photographs."""
        return await self._authorize(request, job_id, include_references=False)

    async def authorize_admission(self, request: Request, job_id: str) -> ReviewContext | Response:
        return await self._authorize(request, job_id, include_references=False, admission=True)

    async def _binding_identity(self, job: Any, tenant_id: str) -> str:
        if self.app._uses_dispatch_authority():
            inventory = await self.app.get_operational_record_store().committed_manifest(job.id, tenant_id)
        else:
            inventory = job.artifacts
        # The inventory binds descriptor/receipt bytes, including their recorded
        # presentation, without reading those bytes before resource admission.
        return hashlib.sha256(canonicalize_json(inventory or {})).hexdigest()

    async def _authorize(
        self, request: Request, job_id: str, *, include_references: bool, admission: bool = False
    ) -> ReviewContext | Response:
        app = self.app
        try:
            await app._cleanup_expired_jobs(app._now(), force=False)
            authorized = await self._job(request, job_id)
            if isinstance(authorized, Response):
                return authorized
            job, tenant = authorized
            tenant_id = tenant.tenant_id if tenant is not None else app._job_tenant_id(job) or "local"
            if app.PILOT_CONTROL_PLANE_ENABLED:
                actor = app._verified_frontdoor_actor(request)
                actor = {**actor, "assurance": "verified_frontdoor_actor"}
            else:
                # API middleware authenticates non-pilot callers. Unsigned actor
                # headers cannot manufacture a human identity for approvals.
                actor = {"username": "api_key_operator", "role": "operator", "assurance": "backend_api_key"}
            references = await self._references(request, job, tenant) if include_references else []
            binding_identity = await self._binding_identity(job, tenant_id) if admission else None

            async def revalidate() -> None:
                await self._revalidate(request, job, tenant_id)
                if binding_identity is not None:
                    current = await self._job(request, job_id)
                    if (
                        isinstance(current, Response)
                        or await self._binding_identity(current[0], tenant_id) != binding_identity
                    ):
                        raise ReviewAccessError("Photographic source bindings changed; reload before review")

            issued: set[str] = set()

            async def publish(storage: str, source: Path, mime: str) -> None:
                await self._publish(request, job, storage, source, mime)
                issued.add(storage)

            return ReviewContext(
                job_id=job_id,
                tenant_id=tenant_id,
                actor=actor,
                references=references,
                materialize=lambda ref, dest, limit: self._materialize(request, job, tenant_id, ref, dest, limit),
                publish=publish,
                discard=lambda paths: self._discard(job, tenant_id, issued, paths),
                revalidate=revalidate,
                discover_references=lambda: self._references(request, job, tenant),
                binding_identity=binding_identity,
                publication_root=getattr(self.app._artifact_store(), "root_dir", None) if admission else None,
            )
        except ReviewAccessError as exc:
            return app._error_response(exc.status, code=exc.code, message=str(exc), details={"job_id": job_id})
        except Exception:
            app.LOGGER.exception("photographic review access unavailable for job %s", job_id)
            return app._error_response(
                503,
                code="REVIEW_UNAVAILABLE",
                message="photographic review unavailable",
                details={"job_id": job_id},
                retriable=True,
            )

    async def _materialize(
        self, request: Request, job: Any, tenant_id: str, ref: dict, destination: Path, maximum_bytes: int
    ) -> dict:
        materialized = False

        def completed() -> None:
            nonlocal materialized
            materialized = True

        try:
            identity = await _owned_operation(
                self._materialize_verified(request, job, tenant_id, ref, destination, maximum_bytes), completed=completed
            )
            current = await self._job(request, job.id)
            if isinstance(current, Response):
                destination.unlink(missing_ok=True)
                raise ReviewAccessError(
                    "Parent job became unavailable during inspection", status=current.status_code, code="ARTIFACT_UNAVAILABLE"
                )
            return identity
        except BaseException as exc:
            if materialized:
                destination.unlink(missing_ok=True)
            if isinstance(exc, ReviewAccessError) or not isinstance(exc, Exception):
                raise
            self.app.LOGGER.exception("photographic review artifact read unavailable for job %s", job.id)
            raise ReviewAccessError("Review artifact is unavailable", status=503, code="ARTIFACT_UNAVAILABLE") from exc

    async def _materialize_verified(
        self, request: Request, job: Any, tenant_id: str, ref: dict, destination: Path, maximum_bytes: int
    ) -> dict:
        expected_sha, expected_size = _identity(ref, maximum_bytes)
        current = await self._job(request, job.id)
        if isinstance(current, Response):
            raise ReviewAccessError("Parent job is unavailable", status=current.status_code, code="ARTIFACT_UNAVAILABLE")
        if ref.get("kind") == "original":
            # Re-derive immutable refs under the current tenant/path policy.
            originals, _ = await self._originals(*current)
            if ref not in originals:
                raise ReviewAccessError("Original binding changed")
            data, _ = await _owned_operation(
                run_review_thread(
                    snapshot,
                    Path(ref["private_original_root"]),
                    Path(ref["private_original_root"]) / ref["private_original_path"],
                    maximum_bytes=maximum_bytes,
                )
            )
            chunks = _single_chunk(data)
        elif ref.get("kind") == "review_asset":
            state = await get_review_store().get(job.id, tenant_id)
            # Presentation is derived from the committed receipt after selecting
            # the saved variant. It never grants storage access or replaces any
            # persisted path, byte identity, tenant, or source-reference field.
            selected = {
                key: value for key, value in ref.items() if key not in {"presentation_policy", "presentation_policy_sha256"}
            }
            if selected not in _saved_assets(state):
                raise ReviewAccessError("Derivative is not in the saved review state", status=404, code="NOT_FOUND")
            if "presentation_policy" in ref or "presentation_policy_sha256" in ref:
                from transformation_portal.photographic_delivery.presentation import revision_policy

                entries = [entry for entry in state.get("revisions", []) if entry.get("variant") == selected]
                policy = revision_policy(entries[0].get("receipt"), selected["sha256"]) if len(entries) == 1 else None
                if (
                    policy is None
                    or ref.get("presentation_policy") != policy.to_payload()
                    or ref.get("presentation_policy_sha256") != policy.sha256
                ):
                    raise ReviewAccessError("Saved revision presentation no longer matches its committed receipt")
            storage = _safe_relative(ref["storage_path"])
            chunks = await self.app._artifact_store().open_bytes(self._storage_job_id(job), storage)
        elif ref.get("kind") == "native_icc":
            carrier = ref.get("carrier_ref")
            if not isinstance(carrier, dict) or carrier.get("kind") != "indexed":
                raise ReviewAccessError("Native ICC carrier reference is unavailable")
            data = await self._read_indexed(request, current[0], carrier, MAX_NATIVE_ICC_BYTES + 65536)
            chunks = _single_chunk(_icc_bytes(data))
        elif ref.get("kind") == "indexed":
            chunks = await self._indexed_chunks(request, current[0], ref["artifact_path"], maximum_bytes)
        else:
            raise ReviewAccessError("Unknown review source", status=400, code="INVALID_ARGUMENT")
        return await _stream_snapshot(chunks, destination, maximum_bytes, expected_size, expected_sha)

    async def _indexed_chunks(
        self, request: Request, job: Any, artifact_path: str, maximum_bytes: int
    ) -> AsyncIterator[bytes]:
        headers = [(key, value) for key, value in request.scope.get("headers", []) if key.lower() != b"x-tp-artifact-delivery"]
        streamed_request = Request({**request.scope, "headers": [*headers, (b"x-tp-artifact-delivery", b"stream")]})
        response = await self.app._get_job_artifact(job.id, _safe_relative(artifact_path), request=streamed_request)
        if response.status_code != 200:
            raise ReviewAccessError(
                "Indexed artifact is unavailable", status=response.status_code, code="ARTIFACT_UNAVAILABLE"
            )
        if isinstance(response, StreamingResponse):
            return _byte_stream(response.body_iterator)
        if isinstance(response, FileResponse):
            _, path, _ = self.app._validate_resolved_job_artifact_path(job, Path(response.path))
            data, _ = await _owned_operation(
                run_review_thread(snapshot, self.app._job_output_dir(job), path, maximum_bytes=maximum_bytes)
            )
            return _single_chunk(data)
        raise ReviewAccessError("Unexpected artifact response", status=503, code="ARTIFACT_UNAVAILABLE")

    async def _fresh_indexed_identity(self, request: Request, job: Any, reference: dict) -> dict:
        """Hash one authorized legacy stream, without asserting native acceptance.

        Missing legacy SHA commonly means only that its small fingerprint budget
        was exceeded. Retain the indexed byte length and enforce it on the full
        authenticated stream. Every later request repeats this observation, so a
        changed file cannot inherit a decision bound to the previous hash.
        """
        size = reference.get("size_bytes")
        if type(size) is not int or size <= 0:
            raise ReviewAccessError("Legacy artifact lacks a recorded byte length")
        if size > MAX_REVIEW_IMAGE_BYTES:
            raise ReviewAccessError("Review source exceeds the admitted byte budget", status=413, code="RESOURCE_EXHAUSTED")
        chunks = await self._indexed_chunks(request, job, reference["artifact_path"], MAX_REVIEW_IMAGE_BYTES)
        with owned_review_directory(prefix="tp-review-identity-") as directory:
            identity = await _owned_operation(
                _stream_snapshot(chunks, Path(directory) / "snapshot", MAX_REVIEW_IMAGE_BYTES, size, None)
            )
        current = await self._job(request, job.id)
        if isinstance(current, Response):
            raise ReviewAccessError(
                "Parent job became unavailable during inspection", status=current.status_code, code="ARTIFACT_UNAVAILABLE"
            )
        return identity

    async def _publish(self, request: Request, job: Any, storage_path: str, source: Path, content_type: str) -> None:
        storage_path = _safe_relative(storage_path)
        if not storage_path.startswith("photographic-review/"):
            raise ReviewAccessError("Invalid review publication namespace", status=400, code="INVALID_ARGUMENT")
        current = await self._job(request, job.id)
        if isinstance(current, Response):
            raise ReviewAccessError("Parent job is unavailable", status=current.status_code, code="ARTIFACT_UNAVAILABLE")
        is_zip = (
            source.suffix.lower() == ".zip"
            and content_type == "application/zip"
            and PurePosixPath(storage_path).suffix.lower() == ".zip"
        )
        maximum_bytes = MAX_REVIEW_ZIP_BYTES if is_zip else MAX_REVIEW_IMAGE_BYTES
        created = False

        def completed() -> None:
            nonlocal created
            created = True

        store = None
        storage_job_id = self._storage_job_id(job)
        try:
            with owned_review_directory(prefix="tp-review-publish-") as directory:
                staged = Path(directory).resolve() / "artifact"
                size, digest = await _owned_operation(run_review_thread(_publication_snapshot, source, staged, maximum_bytes))
                store = self.app._artifact_store()
                lease = current_review_lease.get()
                if lease is not None:
                    local_root = getattr(store, "root_dir", None)
                    if local_root is not None:
                        parent = (local_root / storage_job_id / PurePosixPath(storage_path).parent).resolve()
                        if not parent.is_relative_to(local_root):
                            raise ReviewAccessError("Invalid review publication namespace")
                        if not parent.exists():
                            # A new server-created publication namespace is
                            # private to this operation; track its in-progress
                            # staging blocks too, without crediting old data.
                            lease.track_path(parent)
                    lease.check_scratch(0)
                await _owned_operation(
                    store.write_immutable_file(storage_job_id, storage_path, staged, content_type=content_type),
                    completed=completed,
                )
                if lease is not None and getattr(store, "root_dir", None) is not None:
                    # Only this successful immutable write is owned; existing
                    # retained directories never reduce remaining commitments.
                    lease.track_path(store.root_dir / storage_job_id / storage_path)
                # Reuse GenerationPublisher's complete streamed digest/length
                # verification. Store metadata alone cannot grant visibility.
                await _owned_operation(_verify_staged(store, storage_job_id, storage_path, size, digest))
                current = await self._job(request, job.id)
                if isinstance(current, Response):
                    raise ReviewAccessError(
                        "Parent job became unavailable during publication",
                        status=current.status_code,
                        code="ARTIFACT_UNAVAILABLE",
                    )
        except BaseException as exc:
            if created and store is not None:
                # The immutable write succeeded in this invocation; an existing
                # key collision never sets created and must never be deleted.
                try:
                    await _owned_operation(store.delete(storage_job_id, storage_path))
                except Exception:
                    self.app.LOGGER.exception("private review publication cleanup deferred for job %s", job.id)
            if isinstance(exc, (ReviewAccessError, ReviewOverload)) or not isinstance(exc, Exception):
                raise
            self.app.LOGGER.exception("photographic review artifact publication unavailable for job %s", job.id)
            raise ReviewAccessError(
                "Review artifact publication unavailable", status=503, code="ARTIFACT_UNAVAILABLE"
            ) from exc

    async def _revalidate(self, request: Request, job: Any, tenant_id: str) -> None:
        current = await self._job(request, job.id)
        if isinstance(current, Response):
            if current.status_code == 404 and await self.app._load_job_view(job.id) is None:
                forget = getattr(get_review_store(), "forget", None)
                if forget is not None:
                    await forget(job.id, tenant_id)
            raise ReviewAccessError(
                "Parent job is no longer available for photographic review",
                status=current.status_code,
                code="ARTIFACT_UNAVAILABLE",
            )

    async def _discard(self, job: Any, tenant_id: str, issued: set[str], paths: list[str]) -> None:
        """Reclaim only this request's known, unreferenced successful writes.

        An uncertain state read fails closed: parent retention remains the final
        reclamation path. A committed projection always wins over rollback.
        """
        if not isinstance(paths, list) or len(paths) > 128 or any(path not in issued for path in paths):
            raise ReviewAccessError("Review rollback rejected an unowned artifact", status=400, code="INVALID_ARGUMENT")
        state = await get_review_store().get(job.id, tenant_id)
        retained = {ref["storage_path"] for ref in _saved_assets(state)}
        store = self.app._artifact_store()
        for path in dict.fromkeys(paths):
            if path in retained:
                continue
            await _owned_operation(store.delete(self._storage_job_id(job), path))
            issued.discard(path)


async def forget_review_for_job(app_module: Any, job: Any) -> None:
    """Release process-local review metadata after parent deletion succeeds."""
    forget = getattr(get_review_store(), "forget", None)
    if forget is not None:
        await forget(job.id, app_module._job_tenant_id(job) or "local")


async def _stream_snapshot(
    chunks: AsyncIterator[bytes], destination: Path, maximum_bytes: int, expected_size: int, expected_sha: str | None
) -> dict:
    digest, size, created = hashlib.sha256(), 0, False
    try:
        with destination.open("xb") as handle:
            created = True
            async for chunk in chunks:
                if not isinstance(chunk, bytes):
                    raise ReviewAccessError("Invalid artifact stream")
                size += len(chunk)
                if size > min(maximum_bytes, expected_size):
                    raise ReviewAccessError("Artifact exceeds its recorded byte budget")
                digest.update(chunk)
                handle.write(chunk)
        observed_sha = digest.hexdigest()
        if size != expected_size or (expected_sha is not None and observed_sha != expected_sha):
            raise ReviewAccessError("Artifact bytes changed since verification")
        return {"sha256": observed_sha, "size_bytes": size}
    except BaseException:
        if created:
            destination.unlink(missing_ok=True)
        raise
    finally:
        close = getattr(chunks, "aclose", None)
        if close is not None:
            await close()


async def _single_chunk(data: bytes) -> AsyncIterator[bytes]:
    yield data


async def _byte_stream(chunks: AsyncIterable[object]) -> AsyncIterator[bytes]:
    """Keep Starlette's broader stream type out of binary artifact snapshots."""
    try:
        async for chunk in chunks:
            if not isinstance(chunk, bytes):
                raise ReviewAccessError("Invalid artifact stream")
            yield chunk
    finally:
        close = getattr(chunks, "aclose", None)
        if close is not None:
            await close()
