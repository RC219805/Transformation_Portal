"""Hash-bound native inspection, explicit human decisions, revisions and delivery.

Rendering produces an 8-bit sRGB display derivative from native coordinates;
the master stays untouched. Numeric diagnostics never grant human approval.
All IO is injected by the existing tenant/artifact authorization boundary.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import inspect
import io
import logging
import math
import shutil
import tempfile
import time
import uuid
from collections import OrderedDict
from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import asynccontextmanager
from contextvars import Token
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import CodeType, FrameType
from typing import TYPE_CHECKING, Any, ParamSpec, TypeVar
from urllib.parse import quote

from starlette.responses import FileResponse, JSONResponse, Response
from starlette.types import Receive, Scope, Send

from transformation_portal.api.v1.envelopes import ApiEnvelope
from transformation_portal.api.v1.photographic_review import (
    DeliveryRequest,
    LightingGroupsRequest,
    ReviewDecisionRequest,
    RevisionRequest,
)
from transformation_portal.ingest.canonical_json import canonicalize_json
from transformation_portal.orchestrator.photographic_admission import (
    PhotographicAdmissionController,
    ReviewAdmissionConfig,
    ReviewCost,
    ReviewLease,
    ReviewOverload,
    ReviewRequestTicket,
    current_review_lease,
    drain_owned,
    operation_cost,
    owned_review_directory,
    run_review_thread,
)
from transformation_portal.orchestrator.photographic_review_access import ReviewAccessError, ReviewContext, ReviewInputError
from transformation_portal.orchestrator.photographic_review_store import ReviewConflict, get_review_store

if TYPE_CHECKING:
    from transformation_portal.photographic_delivery import DeliveryImage, ExportPreset, PresentationPolicy

JsonObject = dict[str, Any]
ReviewChange = Callable[[JsonObject], JsonObject]
NativeCrop = tuple[int, int, int, int]
# Fit one tenant's complete current catalogue: 32 scenes with original/native/
# final sources plus the 128-revision limit, without retaining pixel buffers.
_INSPECTION_CACHE_LIMIT = 1024
_TENANT_INSPECTION_CACHE_LIMIT = 256
InspectionKey = tuple[str, str, str, str, str, str]
_Params = ParamSpec("_Params")
_Result = TypeVar("_Result")

MAX_IMAGE_BYTES = 512 * 1024**2
MAX_BUNDLE_BYTES = 16 * 1024**3
LOGGER = logging.getLogger(__name__)
HEADERS = {"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"}


def _clear_failed_frames(
    error: BaseException, completed_frames: Sequence[FrameType] = (), completed_codes: Sequence[CodeType] = ()
) -> None:
    """Drop worker locals and explicitly closed owned coroutine frames."""
    pending = [error]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        trace = current.__traceback__
        while trace is not None:
            frame = trace.tb_frame
            # A shared exception may also point into another subscriber's
            # suspended retry. Only its owner can prove an async frame closed;
            # frame.clear() alone does not protect suspended coroutines.
            suspended_code = frame.f_code.co_flags & (inspect.CO_COROUTINE | inspect.CO_ASYNC_GENERATOR | inspect.CO_GENERATOR)
            if not suspended_code or frame in completed_frames or frame.f_code in completed_codes:
                try:
                    frame.clear()
                except RuntimeError:
                    pass  # The current executing frame cannot be cleared.
            trace = trace.tb_next
        pending.extend(item for item in (current.__cause__, current.__context__) if item is not None)
        if isinstance(current, BaseExceptionGroup):
            pending.extend(current.exceptions)


async def _owned_thread(operation: Callable[_Params, _Result], *args: _Params.args, **kwargs: _Params.kwargs) -> _Result:
    """Keep caller-owned paths and semaphore capacity until blocking work stops.

    Cancelling asyncio.to_thread does not stop its running thread. Shield the
    worker and defer cancellation until it is quiescent, including repeated
    cancellation, so enclosing scratch and resource guards retain ownership.
    """
    worker = asyncio.create_task(run_review_thread(operation, *args, **kwargs))
    # These single-await wrappers cannot resume after propagating an error.
    # Once this worker has stopped, its source-bound arguments are quiescent.
    worker_codes = (run_review_thread.__code__, PhotographicAdmissionController.run_blocking.__code__)
    cancelled = False
    try:
        return await asyncio.shield(worker)
    except asyncio.CancelledError:
        cancelled = True
        while not worker.done():
            try:
                await asyncio.shield(worker)
            except asyncio.CancelledError:
                continue
            except Exception:
                break
        if not worker.cancelled():
            failure = worker.exception()  # Preserve cancellation while draining a separate worker failure.
            if failure is not None:
                _clear_failed_frames(failure, completed_codes=worker_codes)
        raise
    except BaseException as error:
        _clear_failed_frames(error, completed_codes=worker_codes)
        raise
    finally:
        # A propagated traceback retains this frame, even after the worker
        # stopped. Do not keep its source snapshot/closure/future alive there.
        del operation, args, kwargs, worker
        if cancelled:
            # The wakeup callback can still hold a completed worker's discarded
            # result until this event-loop turn ends. Keep the reservation
            # through that handoff, including repeated cancellation.
            await drain_owned(asyncio.sleep(0))


class _OwnedDownloadResponse(FileResponse):
    """Release verified download scratch after every response execution exit.

    FileResponse background tasks do not run on range-error early returns or
    failed/cancelled sends. Keep its normal range/header behavior while owning
    the private directory in a response-level finally block.
    """

    def __init__(self, path: Path, *, directory: Path, media_type: str, filename: str, headers: dict[str, str]) -> None:
        super().__init__(path, media_type=media_type, filename=filename, headers=headers)
        self._directory = directory
        self.admission_lease: ReviewLease | None = None
        self.admission_ticket: ReviewRequestTicket | None = None

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            try:
                await _owned_thread(shutil.rmtree, self._directory)
            finally:
                try:
                    if self.admission_lease is not None:
                        await self.admission_lease.release()
                finally:
                    if self.admission_ticket is not None:
                        await self.admission_ticket.release()


@asynccontextmanager
async def publication(context: ReviewContext) -> AsyncIterator[list[str]]:
    """Rollback only this request's uncommitted immutable keys on failure.

    The adapter additionally checks the committed projection, so an uncertain
    database commit never authorizes deletion of a visible revision/delivery.
    Parent retention remains the fallback when safe cleanup cannot be proved.
    """
    paths: list[str] = []
    try:
        yield paths
    except BaseException:

        async def cleanup() -> None:
            lease = current_review_lease.get()
            if lease is not None:
                # The generator's caller is still suspended with its revision
                # bytes, and a failed worker can retain arrays in tracebacks.
                # Free worker slots now, keeping memory until that method has
                # unwound and _execute_reserved clears inactive failed frames.
                await lease.release_expensive(memory_quiescent=False)
            if context.discard is not None and paths:
                try:
                    await context.discard(paths)
                except Exception:
                    LOGGER.warning(
                        "Unable to discard uncommitted review publication for job %s", context.job_id, exc_info=True
                    )

        # A shield alone lets repeated cancellation abandon the coroutine.
        # Drain cleanup before its owner's scratch/ticket can be released.
        await drain_owned(cleanup())
        raise


def envelope(data: JsonObject) -> JSONResponse:
    return JSONResponse(
        ApiEnvelope[dict](schema="tp.photographic_review.v1", success=True, data=data).model_dump(mode="json"), headers=HEADERS
    )


def base_url(context: ReviewContext) -> str:
    return f"/v1/jobs/{quote(context.job_id, safe='')}/photographic-review"


def revision_refs(state: JsonObject) -> list[JsonObject]:
    from transformation_portal.photographic_delivery.presentation import revision_policy

    result = []
    for entry in state["revisions"]:
        ref = deepcopy(entry["variant"])
        ref.pop("presentation_policy", None)
        ref.pop("presentation_policy_sha256", None)
        policy = revision_policy(entry.get("receipt"), ref["sha256"])
        if policy is not None:
            ref.update(presentation_policy=policy.to_payload(), presentation_policy_sha256=policy.sha256)
        result.append(ref)
    return result


def presentation_policy(ref: JsonObject) -> PresentationPolicy | None:
    """Read only adapter/committed-receipt policy, independently of source approval."""
    from transformation_portal.photographic_delivery import PresentationPolicy

    payload = ref.get("presentation_policy")
    if payload is None:
        if ref.get("presentation_policy_sha256") is not None:
            raise ReviewAccessError("Recorded presentation authority is incomplete")
        return None
    policy = PresentationPolicy.from_payload(payload)
    identity = source_identity(ref)
    if identity is None:
        raise ReviewAccessError("Recorded presentation requires its source identity")
    policy.validate_binding(ref["sha256"], identity)
    if policy.sha256 != ref.get("presentation_policy_sha256"):
        raise ReviewAccessError("Recorded presentation policy changed; reload before inspection")
    return policy


def presentation_metadata(ref: JsonObject) -> JsonObject:
    policy = presentation_policy(ref)
    return {
        "available_views": ["recorded", "clip_srgb"] if policy is not None else ["clip_srgb"],
        "policy_sha256": policy.sha256 if policy is not None else None,
        "policy": policy.to_payload() if policy is not None else None,
        "unavailable_reason": None if policy is not None else "recorded_presentation_authority_not_retained",
    }


def public_state(state: JsonObject, context: ReviewContext | None = None) -> JsonObject:
    """Never expose storage keys, physical bindings or scratch paths."""
    result: JsonObject = {
        key: deepcopy(state[key]) for key in ("state_version", "decisions", "groups", "cohesion") if key in state
    }
    if context is not None:
        result["decisions"] = current_decisions(context, state)
    return result


def source_identity(ref: JsonObject) -> str | None:
    return ref.get("source_identity_sha256", ref.get("sha256"))


def current_decisions(context: ReviewContext, state: JsonObject) -> list[JsonObject]:
    refs = [*context.references, *revision_refs(state)]
    return [
        deepcopy(row)
        for row in state["decisions"]
        if any(
            ref["artifact_path"] == row["artifact_path"]
            and ref.get("sha256") == row["sha256"]
            and source_identity(ref) == row.get("source_identity_sha256", row["sha256"])
            for ref in refs
        )
    ]


def delivery_is_approved(context: ReviewContext, state: JsonObject, delivery: JsonObject) -> bool:
    def identity(row: JsonObject) -> tuple[str, str, str | None]:
        return row["artifact_path"], row["sha256"], row.get("source_identity_sha256", row["sha256"])

    approved = {identity(row) for row in current_decisions(context, state) if row["status"] == "approved"}
    saved = delivery.get("approvals", [])
    return (
        bool(saved)
        and {row["sha256"] for row in saved} == set(delivery["approved_sha256"])
        and all(identity(row) in approved for row in saved)
    )


def delivery_disk_budget(decisions: Sequence[JsonObject], refs: Sequence[JsonObject], presets: Sequence[ExportPreset]) -> int:
    """Bound spool/export/ZIP/publication scratch using decoded geometry.

    Compressed JPEG/PNG bytes do not bound TIFF export size. Five product
    copies cover products, worst-case ZIP expansion and verified local
    publication snapshots, with a separate operational margin. This is a
    point-in-time preflight, never a filesystem-space reservation.
    """
    sources, products = 0, 32 * 1024**2
    for decision, ref in zip(decisions, refs, strict=True):
        inspection = decision["inspection"]
        width, height = inspection["width"], inspection["height"]
        if any(type(v) is not int or v < 1 or v > 65536 for v in (width, height)) or width * height > 40_000_000:
            raise ReviewInputError("Approved source geometry exceeds the managed delivery budget")
        components = [ref, *(row for row in (ref.get("native") or {}).values() if row is not None)]
        closure = 0
        for component in components:
            size = component.get("size_bytes", MAX_IMAGE_BYTES)
            if type(size) is not int or not 0 < size <= MAX_IMAGE_BYTES:
                raise ReviewInputError("Approved source size exceeds the managed delivery budget")
            closure += size
        sources += closure
        for preset in presets:
            if preset.format == "source":
                products += closure + (width * height * 16 if ref.get("native") else 0)
            elif preset.format in {"tiff16", "tiff32"}:
                products += width * height * (8 if preset.format == "tiff16" else 16) + 65536
            else:
                scale = min(1, preset.long_edge / max(width, height)) if preset.long_edge else 1
                products += max(1, math.ceil(width * scale)) * max(1, math.ceil(height * scale)) * 4 + 65536
    return sources + products * 5 + 2 * 1024**3


def resolve(
    context: ReviewContext,
    state: JsonObject,
    artifact_path: str,
    sha256: str,
    identity: str | None = None,
    *,
    require_identity: bool = False,
) -> JsonObject:
    matches = [ref for ref in [*context.references, *revision_refs(state)] if ref["artifact_path"] == artifact_path]
    if len(matches) != 1:
        raise ReviewAccessError("Photographic artifact is unavailable", status=404, code="NOT_FOUND")
    ref = deepcopy(matches[0])
    if ref.get("sha256") not in (None, sha256):
        raise ReviewAccessError("Artifact changed; reload and review the current bytes")
    ref["sha256"] = sha256
    if (identity is not None and identity != source_identity(ref)) or (
        require_identity and source_identity(ref) != sha256 and identity is None
    ):
        raise ReviewAccessError("Working master identity changed or is missing; reload before review")
    return ref


def _check_geometry(path: Path) -> tuple[int, int]:
    if path.suffix.lower() == ".npy":
        import numpy as np

        with path.open("rb") as source:
            version = np.lib.format.read_magic(source)
            if version not in {(1, 0), (2, 0)}:
                raise ReviewInputError("Working masters require NPY v1 or v2")
            reader = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
            shape, fortran, dtype = reader(source)
        if len(shape) != 3 or shape[2] != 3 or fortran or dtype != np.dtype("float32"):
            raise ReviewInputError("Working masters require C-layout float32 RGB samples")
        height, width = shape[:2]
    elif path.suffix.lower() in {".tif", ".tiff"}:
        import tifffile

        with tifffile.TiffFile(path) as source:
            page = source.pages[0]
            if not isinstance(page, tifffile.TiffPage):
                raise ReviewInputError("Working TIFFs require a complete first image page")
            width, height = page.imagewidth, page.imagelength
    else:
        from PIL import Image

        with Image.open(path) as source:
            width, height = source.size
    if min(width, height) < 1 or width * height > 40_000_000 or max(width, height) > 65536:
        raise ReviewAccessError("Image exceeds the managed review pixel budget", status=413, code="RESOURCE_EXHAUSTED")
    return width, height


async def spool_image(context: ReviewContext, ref: JsonObject, directory: Path) -> DeliveryImage:
    """Keep multi-image operations path-backed, with verified native companions."""
    from transformation_portal.photographic_delivery import DeliveryImage

    directory.mkdir(parents=True, exist_ok=False)
    path = directory / Path(ref["source_name"]).name
    observed = await context.materialize(ref, path, MAX_IMAGE_BYTES)
    if observed["sha256"] != ref["sha256"]:
        raise ReviewAccessError("Artifact changed during photographic inspection")
    width, height = await _owned_thread(_check_geometry, path)
    lease = current_review_lease.get()
    if lease is not None and width * height > lease.cost.max_pixels:
        raise ReviewOverload("admitted_geometry_changed", status=413)
    native = None
    if ref.get("native") is not None:
        from transformation_portal.photographic_delivery import NativeWorkingMaster

        attachments = ref["native"]
        if (
            not isinstance(attachments, dict)
            or set(attachments) - {"descriptor", "alpha", "source_icc"}
            or "descriptor" not in attachments
        ):
            raise ReviewAccessError("Working master companions are incomplete")
        values: JsonObject = {}
        for name, companion in attachments.items():
            if companion is None and name in {"alpha", "source_icc"}:
                continue
            if not isinstance(companion, dict):
                raise ReviewAccessError("Working master companions are incomplete")
            companion_path = (
                directory / {"descriptor": "photograph.json", "alpha": "alpha.npy", "source_icc": "source.icc"}[name]
            )
            budget = {"descriptor": 2 * 1024**2, "alpha": MAX_IMAGE_BYTES, "source_icc": 16 * 1024**2}[name]
            actual = await context.materialize(companion, companion_path, budget)
            if actual["sha256"] != companion["sha256"]:
                raise ReviewAccessError("Working master companion changed during inspection")
            values[name + "_sha256"] = actual["sha256"]
            values[name + "_path"] = companion_path
        native = NativeWorkingMaster(**values)
    image = DeliveryImage(
        source_name=path.name,
        source_sha256=observed["sha256"],
        source_path=path,
        scene=ref["scene_id"],
        version=ref.get("version", "v1"),
        input_color=ref.get("input_color", "auto"),
        native=native,
        presentation=presentation_policy(ref),
    )
    if getattr(image, "source_identity_sha256", image.source_sha256) != source_identity(ref):
        raise ReviewAccessError("Working master identity changed; reload before review")
    return image


async def load_image(context: ReviewContext, ref: JsonObject) -> DeliveryImage:
    image: DeliveryImage | None = None
    data: bytes | None = None
    native = None
    values: JsonObject = {}
    try:
        with owned_review_directory(prefix="tp-photo-read-") as directory:
            image = await spool_image(context, ref, Path(directory).resolve() / "source")
            data = await _owned_thread(image.snapshot)
            native = getattr(image, "native", None)
            if native is not None:
                values = {}
                for name in ("descriptor", "alpha", "source_icc"):
                    path = getattr(native, name + "_path", None)
                    if path is not None:
                        values[name + "_bytes"] = await _owned_thread(path.read_bytes)
                        values[name + "_path"] = None
                native = replace(native, **values)
            return replace(image, source_path=None, source_bytes=data, native=native)
    finally:
        # A companion read can fail after the full source snapshot exists.
        # The returned image owns successful snapshots; inactive loader frames
        # must not retain them after the caller releases its reservation.
        del image, data, native, values


def _display(
    image: DeliveryImage, crop: NativeCrop | None = None, max_edge: int = 1600, view: str = "clip_srgb"
) -> tuple[bytes, int, int]:
    import numpy as np
    from PIL import Image

    from transformation_portal.core.image_artifact import ImageMaster
    from transformation_portal.lux_depth_v4.photography import linear_to_srgb, output_srgb_icc
    from transformation_portal.lux_depth_v5.preview import preview_samples
    from transformation_portal.photographic_delivery.presentation import render_presentation

    if view not in {"recorded", "clip_srgb"}:
        raise ReviewInputError("Unsupported photographic presentation view")
    if view == "recorded" and image.presentation is None:
        raise ReviewAccessError(
            "Recorded presentation is unavailable; choose the explicitly labelled clipping view",
            code="PRESENTATION_UNAVAILABLE",
        )

    master = image.decode()
    height, width = master.shape
    if crop is not None:
        x, y, w, h = crop
        if any(type(v) is not int for v in crop) or x < 0 or y < 0 or not 1 <= w <= 2048 or not 1 <= h <= 2048:
            raise ReviewInputError("Native crop requires integer coordinates and dimensions in [1,2048]")
        if x + w > width or y + h > height:
            raise ReviewInputError("Native crop must remain inside the source geometry")
        values = master.pixels[y : y + h, x : x + w]
        if view == "recorded":
            assert image.presentation is not None  # Checked before decoding; keep crop rendering policy explicit.
            selected = ImageMaster(
                values,
                master.source_sha256,
                master.source_bit_depth,
                None if master.alpha is None else master.alpha[y : y + h, x : x + w],
                master.metadata,
            )
            selected, _ = render_presentation(selected, image.presentation, image.presentation.recipe)
            values = selected.pixels
        samples = np.rint(np.clip(linear_to_srgb(values), 0, 1) * 255).astype(np.uint8)
        if master.alpha is not None:
            alpha = np.rint(master.alpha[y : y + h, x : x + w] * 255).astype(np.uint8)
            samples = np.concatenate([samples, alpha[..., None]], axis=2)
    else:
        if not 1 <= max_edge <= 1600:
            raise ReviewInputError("Fit previews require max_edge in [1,1600]")
        if view == "recorded":
            assert image.presentation is not None
            master, _ = render_presentation(master, image.presentation, image.presentation.recipe)
            samples = preview_samples(master, max_edge=max_edge)
        else:
            samples = preview_samples(master)
        if max(samples.shape[:2]) > max_edge:
            scale = max_edge / max(samples.shape[:2])
            with Image.fromarray(samples) as source:
                samples = np.asarray(
                    source.resize(
                        (max(1, int(samples.shape[1] * scale)), max(1, int(samples.shape[0] * scale))), Image.Resampling.BOX
                    )
                )
    with Image.fromarray(samples) as output:
        buffer = io.BytesIO()
        output.save(buffer, format="PNG", icc_profile=output_srgb_icc())
    return buffer.getvalue(), samples.shape[1], samples.shape[0]


def _metadata(image: DeliveryImage) -> JsonObject:
    from transformation_portal.photographic_delivery import inspect_image

    return inspect_image(image)


class _LegacyExpensiveGate:
    """Keep disabled/direct-call compatibility; an admitted operation owns its vector."""

    def __init__(self) -> None:
        self._semaphore = asyncio.Semaphore(1)

    async def __aenter__(self) -> None:
        if current_review_lease.get() is None:
            await self._semaphore.acquire()

    async def __aexit__(self, *_error: Any) -> None:
        if current_review_lease.get() is None:
            self._semaphore.release()


class PhotographicReviewService:
    def __init__(self, admission_config: ReviewAdmissionConfig | None = None) -> None:
        self.expensive = _LegacyExpensiveGate()
        self.admission = PhotographicAdmissionController(admission_config)
        # Planning metadata only: never image bytes, approvals or completed
        # render results. The service/router lifecycle owns this bounded LRU.
        self._inspection_costs: OrderedDict[InspectionKey, JsonObject] = OrderedDict()

    async def close(self) -> None:
        await self.admission.close()
        self._inspection_costs.clear()

    def _remember_inspection(self, context: ReviewContext, ref: JsonObject, inspection: JsonObject) -> None:
        if not self.admission.config.enabled or not context.binding_identity:
            return
        width, height, alpha = (inspection.get(name) for name in ("width", "height", "alpha"))
        if (
            type(width) is not int
            or type(height) is not int
            or min(width, height) <= 0
            or width * height > 40_000_000
            or type(alpha) is not bool
        ):
            return
        identity = source_identity(ref)
        if identity is None:
            return
        key = (
            context.tenant_id,
            context.job_id,
            context.binding_identity,
            ref["artifact_path"],
            ref["sha256"],
            identity,
        )
        self._inspection_costs[key] = {"width": width, "height": height, "alpha": alpha}
        self._inspection_costs.move_to_end(key)
        tenant_keys = [item for item in self._inspection_costs if item[0] == context.tenant_id]
        for stale in tenant_keys[:-_TENANT_INSPECTION_CACHE_LIMIT]:
            del self._inspection_costs[stale]
        while len(self._inspection_costs) > _INSPECTION_CACHE_LIMIT:
            self._inspection_costs.popitem(last=False)

    def _admission_inspections(
        self, context: ReviewContext, state: JsonObject, operation: str, args: tuple[Any, ...], kwargs: JsonObject
    ) -> list[JsonObject] | None:
        saved = self._cost_inspections(state, operation, args, kwargs)
        if saved is not None or not self.admission.config.enabled or not context.binding_identity:
            return saved
        prefix = context.tenant_id, context.job_id, context.binding_identity
        candidates = {
            key[3:]: {
                "artifact_path": key[3],
                "sha256": key[4],
                "source_identity_sha256": key[5],
                "inspection": dict(inspection),
            }
            for key, inspection in self._inspection_costs.items()
            if key[:3] == prefix
        }
        # Reuse the exact selector rules used for persisted inspections. This
        # projection is local planning data and cannot become a saved decision.
        for row in state.get("decisions", []):
            if isinstance(row.get("inspection"), dict):
                candidates[row["artifact_path"], row["sha256"], row.get("source_identity_sha256", row["sha256"])] = row
        result = self._cost_inspections({**state, "decisions": list(candidates.values())}, operation, args, kwargs)
        if result is not None:
            selected = {id(inspection) for inspection in result}
            for key in list(self._inspection_costs):
                candidate = candidates.get(key[3:]) if key[:3] == prefix else None
                if candidate is not None and id(candidate["inspection"]) in selected:
                    self._inspection_costs.move_to_end(key)
        return result

    async def execute(self, context: ReviewContext, method: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Authorize before queueing; reserve once through all IO and publication."""
        if not self.admission.config.enabled:
            return await method(context, *args, **kwargs)
        operation = method.__name__
        ticket = await self.admission.begin_request(context.tenant_id)
        transferred = False
        try:
            await self.revalidate(context)
            if operation == "render" and context.binding_identity is not None:
                state = await self.state(context)
                key = (
                    context.tenant_id,
                    context.job_id,
                    context.binding_identity,
                    state["state_version"],
                    "photographic-render-v1",
                    args,
                    canonicalize_json(kwargs),
                )

                async def shared_render() -> Any:
                    return await self._execute_reserved(context, method, *args, _shared_render=True, **kwargs)

                try:
                    result = await self.admission.coalesce(key, shared_render)
                except ReviewAccessError as error:
                    if error.status not in {401, 403}:
                        raise
                    # An earlier subscriber may lose access before its source
                    # read completes. Its owner has already drained and released
                    # the lease. Make one private, equally bounded attempt so
                    # another follower's later revocation cannot poison it.
                    await self.revalidate(context)
                    result = await self._execute_reserved(context, method, *args, **kwargs)
                await self.revalidate(context)
            else:
                result = await self._execute_reserved(context, method, *args, **kwargs)
            if isinstance(result, _OwnedDownloadResponse):
                result.admission_ticket = ticket
                transferred = True
            return result
        finally:
            if not transferred:
                await ticket.release()

    async def _execute_reserved(
        self, context: ReviewContext, method: Callable[..., Any], *args: Any, _shared_render: bool = False, **kwargs: Any
    ) -> Any:
        state = await self.state(context)
        operation = method.__name__
        self._validate_admission_request(state, operation, args)
        inspections = self._admission_inspections(context, state, operation, args, kwargs)
        cost = self._resource_cost(context, state, operation, args, inspections)
        lease = await self.admission.acquire(context.tenant_id, cost)
        token: Token[ReviewLease | None] | None = current_review_lease.set(lease)
        transferred = False
        result = None
        pending_operation = None
        method_frame = None
        try:
            # Reference discovery has bounded streamed snapshots but no image
            # decode. Complete output-space preflight follows resolved geometry.
            lease.check_scratch(min(lease.cost.scratch_bytes, 2 * 1024**3), preparing=True)
            await self.revalidate(context)
            if context.discover_references is not None:
                context = replace(context, references=await context.discover_references())
            if method.__name__ == "catalogue" and token is not None:
                # Discovery and each image take separate complete turns. A
                # catalogue may not monopolize the protected inspection slot.
                current_review_lease.reset(token)
                token = None
                await lease.release()
            if _shared_render and method == self.render:
                pending_operation = self._render_response(context, *args, **kwargs)
            else:
                pending_operation = method(context, *args, **kwargs)
            if inspect.iscoroutine(pending_operation):
                method_frame = pending_operation.cr_frame
            result = await pending_operation
            if (
                method_frame is not None
                and inspect.iscoroutine(pending_operation)
                and inspect.getcoroutinestate(pending_operation) == inspect.CORO_CLOSED
            ):
                method_frame.clear()
                method_frame = None
            # A shared read retains source/materialization checks here, while
            # execute validates each subscriber's actor and source binding after
            # the shared bytes exist. The leader cannot authorize its followers.
            if not _shared_render:
                await self.revalidate(context)
            if isinstance(result, _OwnedDownloadResponse):
                result.admission_lease = lease
                transferred = True
            return result
        except BaseException as error:
            try:
                if isinstance(result, _OwnedDownloadResponse):
                    await _owned_thread(shutil.rmtree, result._directory)
            finally:
                result = None
                # The awaited method and its blocking workers are quiescent.
                # Keep exception types/messages/stack locations, but remove
                # their now-inactive buffer-owning locals before releasing
                # the final memory reservation, even if cleanup itself fails.
                completed = (
                    (method_frame,)
                    if method_frame is not None
                    and inspect.iscoroutine(pending_operation)
                    and inspect.getcoroutinestate(pending_operation) == inspect.CORO_CLOSED
                    else ()
                )
                _clear_failed_frames(error, completed)
            raise
        finally:
            if token is not None:
                current_review_lease.reset(token)
            if not transferred:
                await lease.release()

    @staticmethod
    def _validate_admission_request(state: JsonObject, operation: str, args: tuple[Any, ...]) -> None:
        """Reject cheap stale/approval errors before estimating expensive work.

        Full source authorization and current-decision checks still run after
        discovery; this preflight cannot authorize an operation or publication.
        """
        if operation in {"decision", "groups", "revision", "delivery"} and args:
            if args[0].expected_version != state["state_version"]:
                raise ReviewConflict("Review changed; reload before continuing")
        if operation == "delivery" and args:
            payload = args[0]
            approved = [row for row in state.get("decisions", []) if row["status"] == "approved"]
            if payload.approved_sources:
                valid = all(
                    any(
                        row["artifact_path"] == item.artifact_path
                        and row["sha256"] == item.sha256
                        and (
                            item.source_identity_sha256 is None
                            or row.get("source_identity_sha256", row["sha256"]) == item.source_identity_sha256
                        )
                        for row in approved
                    )
                    for item in payload.approved_sources
                )
            else:
                valid = all(any(row["sha256"] == sha for row in approved) for sha in payload.approved_sha256)
            if not valid:
                raise ReviewInputError("Delivery requires current approval for every exact source identity")

    def _resource_cost(
        self,
        context: ReviewContext,
        state: JsonObject,
        operation: str,
        args: tuple[Any, ...],
        inspections: list[JsonObject] | None,
    ) -> ReviewCost:
        """Freeze geometry-derived temporary and publication filesystem claims."""
        cost = operation_cost(operation, self.admission.config, inspections)
        rows = inspections or [{"width": 8000, "height": 5000, "alpha": True}]

        def source_bound(row: JsonObject) -> JsonObject:
            # Before descriptor discovery, reserve every permitted companion.
            # Known no-alpha inspections can omit only the alpha byte bound.
            pixels = row["width"] * row["height"]
            return {
                "size_bytes": MAX_IMAGE_BYTES,
                "native": {
                    "descriptor": {"size_bytes": 2 * 1024**2},
                    "source_icc": {"size_bytes": 16 * 1024**2},
                    "alpha": {"size_bytes": min(MAX_IMAGE_BYTES, pixels * 4 + 4096)} if row.get("alpha", True) else None,
                },
            }

        refs = [source_bound(row) for row in rows]
        closures = [
            sum(part["size_bytes"] for part in [ref, *(part for part in ref["native"].values() if part)]) for ref in refs
        ]
        source = max(closures)
        workspace, publication_bytes = source + 256 * 1024**2, 0
        if operation == "delivery":
            from transformation_portal.photographic_delivery import ExportPreset

            payload = args[0] if args else None
            count = len(payload.approved_sources or payload.approved_sha256) if payload is not None else len(rows)
            if inspections is None:
                rows, refs = rows * count, refs * count
            presets = (
                [ExportPreset.from_payload(value) for value in payload.presets]
                if payload is not None
                else [ExportPreset.from_payload("retouch_print")]
            )
            complete = delivery_disk_budget([{"inspection": row} for row in rows], refs, presets)
            sources = sum(closures) if inspections is not None else sum(closures) * count
            products = (complete - sources - 2 * 1024**3) // 5
            workspace = sources + products * 3 + 2 * 1024**3
            publication_bytes = products * 2
        elif operation == "revision":
            product = cost.max_pixels * 16 + 65536
            workspace += product * 2
            publication_bytes = product * 2
        elif operation == "groups":
            count = sum(len(group.members) for group in args[0].groups) if args else len(rows)
            workspace = (sum(closures) if inspections is not None else source * count) + 256 * 1024**2
        elif operation == "download":
            identity, filename = args if len(args) == 2 else (None, None)
            files = [
                entry
                for delivery in state.get("deliveries", [])
                if delivery["id"] == identity
                for entry in delivery["files"]
                if entry["filename"] == filename
            ]
            size = files[0]["size_bytes"] if len(files) == 1 else MAX_BUNDLE_BYTES
            workspace = max(size, source) + 256 * 1024**2
        temporary = Path(tempfile.gettempdir()).resolve()
        claims = [(temporary, workspace)]
        if publication_bytes:
            if context.publication_root is None:
                claims[0] = (temporary, workspace + publication_bytes)
            else:
                claims.append((context.publication_root, publication_bytes))
        return replace(cost, scratch_bytes=sum(size for _, size in claims), filesystems=tuple(claims))

    @staticmethod
    def _cost_inspections(
        state: JsonObject, operation: str, args: tuple[Any, ...], kwargs: JsonObject
    ) -> list[JsonObject] | None:
        """Use server-authored inspections only for the exact requested hashes."""
        decisions = state.get("decisions", [])
        selectors: list[tuple[str | None, str, str | None]] = []
        if operation in {"render", "quality"} and len(args) >= 2:
            selectors.append((args[0], args[1], kwargs.get("source_identity_sha256")))
            if operation == "quality" and len(args) >= 4:
                selectors.append((args[2], args[3], kwargs.get("reference_identity_sha256")))
        elif operation in {"decision", "revision"} and args:
            selectors.append((args[0].artifact_path, args[0].sha256, args[0].source_identity_sha256))
        elif operation == "delivery" and args:
            payload = args[0]
            if payload.approved_sources:
                selectors.extend(
                    (item.artifact_path, item.sha256, item.source_identity_sha256) for item in payload.approved_sources
                )
            else:
                selectors.extend((None, sha, None) for sha in payload.approved_sha256)
        elif operation == "groups" and args:
            for group in args[0].groups:
                for identity in group.members:
                    matches = [row for row in decisions if row.get("source_identity_sha256", row["sha256"]) == identity]
                    if len(matches) != 1:
                        return None
                    selectors.append((matches[0]["artifact_path"], matches[0]["sha256"], identity))
        if not selectors:
            return None
        result = []
        for path, sha, identity in selectors:
            matches = [
                row
                for row in decisions
                if row["sha256"] == sha
                and (path is None or row["artifact_path"] == path)
                and (identity is None or row.get("source_identity_sha256", sha) == identity)
            ]
            if len(matches) != 1 or not isinstance(matches[0].get("inspection"), dict):
                return None
            result.append(matches[0]["inspection"])
        return result

    @asynccontextmanager
    async def _catalogue_turn(self, context: ReviewContext) -> AsyncIterator[None]:
        if not self.admission.config.enabled:
            async with self.expensive:
                yield
            return
        state = await self.state(context)
        lease = await self.admission.acquire(context.tenant_id, self._resource_cost(context, state, "catalogue", (), None))
        token = current_review_lease.set(lease)
        try:
            lease.check_scratch(lease.cost.scratch_bytes)
            await self.revalidate(context)
            yield
        finally:
            current_review_lease.reset(token)
            await lease.release()

    async def state(self, context: ReviewContext) -> JsonObject:
        return await get_review_store().get(context.job_id, context.tenant_id)

    async def mutate(
        self, context: ReviewContext, expected_version: int, change: ReviewChange, kind: str, details: JsonObject
    ) -> JsonObject:
        lease = current_review_lease.get()
        if lease is not None:
            await lease.release_expensive()
        await self.revalidate(context)
        updated = await get_review_store().mutate(
            context.job_id,
            context.tenant_id,
            expected_version,
            change,
            {"kind": kind, "actor": context.actor, "created_at": time.time(), "details": details},
        )
        await self.revalidate(context)
        return updated

    @staticmethod
    async def revalidate(context: ReviewContext) -> None:
        if context.revalidate is not None:
            await context.revalidate()

    async def catalogue(self, context: ReviewContext) -> JsonObject:
        from transformation_portal.photographic_delivery import destination_presets

        state = await self.state(context)
        scenes: dict[str, JsonObject] = {}
        for ref in [*context.references, *revision_refs(state)]:
            async with self._catalogue_turn(context):
                if len(scenes) >= 32 and ref["scene_id"] not in scenes:
                    raise ReviewInputError("Photographic review is bounded to 32 scenes")
                image = await load_image(context, ref)
                try:
                    metadata = await _owned_thread(_metadata, image)
                    self._remember_inspection(context, ref, metadata)
                    scene = scenes.setdefault(
                        ref["scene_id"],
                        {
                            "scene_id": ref["scene_id"],
                            "label": ref.get("label", ref["scene_id"]),
                            "variants": [],
                            "original_unavailable_reason": ref.get("original_unavailable_reason"),
                        },
                    )
                    scene["variants"].append(
                        {
                            **metadata,
                            "intended_use": (
                                "Retained input / comparison reference"
                                if ref["role"] == "original"
                                else (
                                    "Native float working master"
                                    if ref["role"] == "working_master"
                                    else metadata["intended_use"]
                                )
                            ),
                            "id": ref["artifact_path"],
                            "role": ref["role"],
                            "artifact_path": ref["artifact_path"],
                            "sha256": image.source_sha256,
                            "source_name": ref["source_name"],
                            "version": ref.get("version", "v1"),
                            "source_identity_sha256": source_identity(ref),
                            "identity_status": ref.get("identity_status", "verified_indexed_bytes"),
                            "preview_url": base_url(context) + "/render",
                            "native_crop_url": base_url(context) + "/native-crop",
                            "presentation": presentation_metadata(ref),
                            "presentation_policy_sha256": ref.get("presentation_policy_sha256"),
                        }
                    )
                finally:
                    # This source owns a separate lease. Drop its snapshot
                    # before that lease exits, including metadata failure or
                    # cancellation after the blocking worker has drained.
                    del image
        await self.revalidate(context)
        for scene in scenes.values():
            finals = [v for v in scene["variants"] if v["role"] not in {"original", "working_master"}]
            for variant in finals[:-1]:
                variant["role"] = "previous_revision"
            if finals:
                finals[-1]["role"] = "final"
                scene["decision"] = next(
                    (
                        d
                        for d in current_decisions(context, state)
                        if d["artifact_path"] == finals[-1]["artifact_path"]
                        and d["sha256"] == finals[-1]["sha256"]
                        and d.get("source_identity_sha256", d["sha256"]) == finals[-1]["source_identity_sha256"]
                    ),
                    None,
                )
            group = next(
                (g for g in state["groups"] if any(v["source_identity_sha256"] in g["members"] for v in scene["variants"])),
                None,
            )
            scene["lighting_group"] = group["name"] if group else None
            scene["reference_sha256"] = group["reference_sha256"] if group else None
        return {
            **public_state(state, context),
            "scenes": list(scenes.values()),
            "presets": [p.to_payload() for p in destination_presets()],
            "scope": "human_review_required",
            "native_display": "8bit_srgb_derivative_one_source_pixel_per_css_pixel",
            "revision_constraints": {"exposure_stops": [-1, 1], "white_balance": [0.8, 1.25], "saturation": [0.8, 1.2]},
        }

    async def render(
        self,
        context: ReviewContext,
        artifact_path: str,
        sha256: str,
        crop: NativeCrop | None = None,
        max_edge: int = 1600,
        source_identity_sha256: str | None = None,
        view: str = "clip_srgb",
    ) -> Response:
        result = await self._render_response(context, artifact_path, sha256, crop, max_edge, source_identity_sha256, view)
        await self.revalidate(context)
        return result

    async def _render_response(
        self,
        context: ReviewContext,
        artifact_path: str,
        sha256: str,
        crop: NativeCrop | None = None,
        max_edge: int = 1600,
        source_identity_sha256: str | None = None,
        view: str = "clip_srgb",
    ) -> Response:
        """Compute a verified read; its public/subscriber boundary checks access."""
        image: DeliveryImage | None = None
        raw: bytes | None = None
        try:
            state = await self.state(context)
            ref = resolve(context, state, artifact_path, sha256, source_identity_sha256, require_identity=True)
            async with self.expensive:
                image = await load_image(context, ref)
                raw, width, height = await _owned_thread(_display, image, crop, max_edge, view)
            return Response(
                raw,
                media_type="image/png",
                headers={
                    **HEADERS,
                    "X-TP-Source-SHA256": sha256,
                    "X-TP-Source-Identity-SHA256": image.source_identity_sha256,
                    "X-TP-Native-Width": str(width),
                    "X-TP-Native-Height": str(height),
                    "X-TP-Display-Transform": (
                        "linear_srgb_to_8bit_srgb_clip_v1" if view == "clip_srgb" else "recorded_presentation_to_8bit_srgb_v1"
                    ),
                    "X-TP-Presentation-View": view,
                    "X-TP-Presentation-Policy-SHA256": image.presentation.sha256 if image.presentation is not None else "none",
                },
            )
        except BaseException:
            # A successful worker's result can remain in its wakeup callback
            # when response construction fails. Keep the lease through that
            # handoff before the failed nested render drops its own snapshots.
            await drain_owned(asyncio.sleep(0))
            raise
        finally:
            # Private render/fallback calls are nested beneath the captured
            # public method. Drop this owner's snapshots before its lease can
            # exit, even when rendering or response construction fails.
            del image, raw

    async def decision(self, context: ReviewContext, payload: ReviewDecisionRequest) -> JsonObject:
        state = await self.state(context)
        ref = resolve(
            context, state, payload.artifact_path, payload.sha256, payload.source_identity_sha256, require_identity=True
        )
        if ref["role"] == "original":
            raise ReviewInputError("Originals are comparison references; select a final master for delivery review")
        async with self.expensive:
            image = await load_image(context, ref)
            inspection = await _owned_thread(_metadata, image)
            del image
        decision: JsonObject = {
            "artifact_path": payload.artifact_path,
            "sha256": payload.sha256,
            "artifact_sha256": payload.sha256,
            "source_identity_sha256": source_identity(ref),
            "status": payload.status,
            "note": payload.note,
            "actor": context.actor,
            "created_at": time.time(),
            "inspection": inspection,
        }
        if payload.inspection_presentation is not None:
            audit = payload.inspection_presentation.model_dump()
            policy = presentation_policy(ref)
            expected = policy.sha256 if policy is not None else None
            if audit["policy_sha256"] != expected or (audit["view"] == "recorded" and policy is None):
                raise ReviewAccessError("Inspected presentation changed; reload before recording the decision")
            decision["inspection_presentation"] = audit

        def change(current: JsonObject) -> JsonObject:
            current["decisions"] = [row for row in current["decisions"] if row["artifact_path"] != payload.artifact_path] + [
                decision
            ]
            return current

        return public_state(await self.mutate(context, payload.expected_version, change, "decision", decision), context)

    async def groups(self, context: ReviewContext, payload: LightingGroupsRequest) -> JsonObject:
        from transformation_portal.photographic_delivery import cohesion_diagnostics

        state = await self.state(context)
        groups = [g.model_dump() for g in payload.groups]
        members = [member for group in groups for member in group["members"]]
        if len(set(members)) != len(members) or len({g["name"] for g in groups}) != len(groups):
            raise ReviewInputError("Each photograph must belong to one uniquely named lighting group")
        for group in groups:
            if group["reference_sha256"] not in group["members"]:
                raise ReviewInputError("Lighting reference must be an explicit group member")
        refs = [*context.references, *revision_refs(state)]
        async with self.expensive:
            with owned_review_directory(prefix="tp-photo-groups-") as directory:
                images: list[DeliveryImage] = []
                for index, sha in enumerate(members):
                    matches = [ref for ref in refs if source_identity(ref) == sha]
                    if len(matches) != 1:
                        raise ReviewInputError("Lighting groups require exact current artifact hashes")
                    images.append(await spool_image(context, matches[0], Path(directory).resolve() / str(index)))
                diagnostics = await _owned_thread(cohesion_diagnostics, images, groups)

        def change(current: JsonObject) -> JsonObject:
            current["groups"], current["cohesion"] = groups, diagnostics
            return current

        return public_state(
            await self.mutate(
                context, payload.expected_version, change, "lighting_groups", {"groups": groups, "diagnostics": diagnostics}
            ),
            context,
        )

    async def quality(
        self,
        context: ReviewContext,
        artifact_path: str,
        sha256: str,
        reference_path: str,
        reference_sha256: str,
        source_identity_sha256: str | None = None,
        reference_identity_sha256: str | None = None,
    ) -> JsonObject:
        from transformation_portal.evals.apex_metrics import compute_apex_metrics

        state = await self.state(context)
        async with self.expensive:
            candidate = await load_image(
                context, resolve(context, state, artifact_path, sha256, source_identity_sha256, require_identity=True)
            )
            reference = await load_image(
                context,
                resolve(context, state, reference_path, reference_sha256, reference_identity_sha256, require_identity=True),
            )

            def compute() -> JsonObject:
                import numpy as np

                from transformation_portal.lux_depth_v4.photography import linear_to_srgb

                left, right = reference.decode(), candidate.decode()
                if left.shape != right.shape:
                    raise ReviewInputError("Quality comparison requires matching native geometry; images are never resized")
                return compute_apex_metrics(
                    np.clip(linear_to_srgb(left.pixels), 0, 1),
                    np.clip(linear_to_srgb(right.pixels), 0, 1),
                    working_color_space="srgb",
                    working_transfer_function="srgb",
                )

            metrics = await _owned_thread(compute)
        await self.revalidate(context)
        return {
            "metrics": metrics,
            "algorithm_revision": "spatial_v2",
            "source_sha256": sha256,
            "reference_sha256": reference_sha256,
            "source_identity_sha256": candidate.source_identity_sha256,
            "reference_identity_sha256": reference.source_identity_sha256,
            "scope": "reference_relative_not_photographic_acceptance",
        }

    async def revision(self, context: ReviewContext, payload: RevisionRequest) -> JsonObject:
        from transformation_portal.photographic_delivery import RevisionRecipe, create_revision

        state = await self.state(context)
        if payload.expected_version != state["state_version"]:
            raise ReviewConflict("Review changed; reload before creating a revision")
        ref = resolve(
            context, state, payload.artifact_path, payload.sha256, payload.source_identity_sha256, require_identity=True
        )
        if payload.scene_id != ref["scene_id"]:
            raise ReviewInputError("Revision scene must match its exact source artifact")
        recipe = RevisionRecipe.from_payload(payload.recipe)
        policy = presentation_policy(ref)
        if "presentation_policy_sha256" in payload.model_fields_set and payload.presentation_policy_sha256 != (
            policy.sha256 if policy is not None else None
        ):
            raise ReviewAccessError("Revision presentation changed; reload before creating a revision")
        if payload.reference_sha256 is not None and not any(
            payload.reference_sha256 in group["members"] and group["reference_sha256"] == payload.reference_sha256
            for group in state["groups"]
        ):
            raise ReviewInputError("Revision reference must be a saved lighting-group reference")
        async with self.expensive:
            image = await load_image(context, ref)
            result = await _owned_thread(create_revision, image, recipe, reference_sha256=payload.reference_sha256)
            del image
        preview = "data:image/png;base64," + base64.b64encode(result["preview_bytes"]).decode("ascii")
        if payload.preview_only:
            del result["source_bytes"], result["preview_bytes"]
            await self.revalidate(context)
            return {"preview_data_url": preview, "receipt": result["receipt"], "state_version": state["state_version"]}
        if len(state["revisions"]) >= 128:
            raise ReviewInputError("Review is bounded to 128 revisions per job")
        identity = uuid.uuid4().hex
        storage = f"photographic-review/{identity}/master.tif"
        variant: JsonObject = {
            "artifact_path": f"revisions/{identity}/master.tif",
            "sha256": result["receipt"]["revision_sha256"],
            "size_bytes": len(result["source_bytes"]),
            "source_name": "master.tif",
            "version": f"r{len(state['revisions']) + 1:03}",
            "scene_id": payload.scene_id,
            "label": ref.get("label", payload.scene_id),
            "role": "final",
            "kind": "review_asset",
            "storage_path": storage,
            "input_color": "auto",
        }
        async with publication(context) as publication_paths:
            with owned_review_directory(prefix="tp-photo-revision-") as directory:
                path = Path(directory).resolve() / "master.tif"
                await _owned_thread(path.write_bytes, result["source_bytes"])
                await context.publish(storage, path, "image/tiff")
                publication_paths.append(storage)
            entry: JsonObject = {
                "id": identity,
                "variant": variant,
                "receipt": result["receipt"],
                "actor": context.actor,
                "created_at": time.time(),
            }

            def change(current: JsonObject) -> JsonObject:
                current["revisions"].append(entry)
                return current

            del result["source_bytes"], result["preview_bytes"]
            updated = await self.mutate(context, payload.expected_version, change, "revision", entry)
        return {
            **public_state(updated, context),
            "revision": {
                "id": identity,
                "artifact_path": variant["artifact_path"],
                "sha256": variant["sha256"],
                "review_status": "unreviewed",
                "receipt": result["receipt"],
            },
            "preview_data_url": preview,
        }

    async def delivery(self, context: ReviewContext, payload: DeliveryRequest) -> JsonObject:
        from transformation_portal.orchestrator.photographic_delivery_history import (
            APPLIED_SCHEMA,
            INTENT_SCHEMA,
            applied_derivatives,
            delivery_intent,
        )
        from transformation_portal.photographic_delivery import ExportPreset, export_delivery, package_delivery

        state = await self.state(context)
        if state["state_version"] != payload.expected_version:
            raise ReviewConflict("Review changed; reload before delivery")
        if len(set(payload.approved_sha256)) != len(payload.approved_sha256):
            raise ReviewInputError("An approved master may appear only once")
        approved = [row for row in current_decisions(context, state) if row["status"] == "approved"]
        if payload.approved_sources is not None:
            decisions: list[JsonObject] = []
            identities = set()
            for selection in payload.approved_sources:
                ref = resolve(
                    context,
                    state,
                    selection.artifact_path,
                    selection.sha256,
                    selection.source_identity_sha256,
                    require_identity=True,
                )
                key = (ref["artifact_path"], ref["sha256"], source_identity(ref))
                if key in identities:
                    raise ReviewInputError("An approved source may appear only once")
                identities.add(key)
                matches = [
                    row
                    for row in approved
                    if (row["artifact_path"], row["sha256"], row.get("source_identity_sha256", row["sha256"])) == key
                ]
                if len(matches) != 1:
                    raise ReviewInputError("Delivery requires current approval for every exact source identity")
                decisions.append(matches[0])
            hashes = [row["sha256"] for row in decisions]
            if payload.approved_sha256 and set(payload.approved_sha256) != set(hashes):
                raise ReviewInputError("Delivery source selectors and legacy hashes disagree")
        else:
            hashes = payload.approved_sha256
            decisions = [row for row in approved if row["sha256"] in hashes]
            if {row["sha256"] for row in decisions} != set(hashes) or len(decisions) != len(hashes):
                raise ReviewInputError(
                    "Delivery requires explicit current approval for every exact master hash; use exact source selectors for shared hashes"
                )
        presets = [ExportPreset.from_payload(preset) for preset in payload.presets]
        if len({preset.name for preset in presets}) != len(presets):
            raise ReviewInputError("Delivery presets must be unique")
        if len(state["deliveries"]) >= 64:
            raise ReviewInputError("Review is bounded to 64 deliveries per job")
        selected_scene_ids = {
            resolve(context, state, decision["artifact_path"], decision["sha256"])["scene_id"] for decision in decisions
        }
        if set(payload.scene_labels) - selected_scene_ids:
            raise ReviewInputError("Delivery labels must name selected approved scenes only")
        intent, intent_sha256 = delivery_intent(
            context,
            payload,
            [resolve(context, state, row["artifact_path"], row["sha256"]) for row in decisions],
            presets,
        )
        identity = uuid.uuid4().hex
        async with publication(context) as publication_paths:
            async with self.expensive:
                source_refs = [resolve(context, state, row["artifact_path"], row["sha256"]) for row in decisions]
                required = delivery_disk_budget(decisions, source_refs, presets)
                lease = current_review_lease.get()
                if lease is not None:
                    lease.check_scratch(required)
                elif shutil.disk_usage(tempfile.gettempdir()).free < required:
                    raise ReviewInputError(
                        f"Insufficient scratch disk for verified delivery; require {required / 1024**3:.1f} GiB free"
                    )
                with owned_review_directory(prefix="tp-approved-delivery-") as directory:
                    root = Path(directory).resolve()
                    images: list[DeliveryImage] = []
                    for index, decision in enumerate(decisions):
                        ref = resolve(context, state, decision["artifact_path"], decision["sha256"])
                        image = await spool_image(context, ref, root / "sources" / str(index))
                        if decision.get("source_identity_sha256", decision["sha256"]) != getattr(
                            image, "source_identity_sha256", image.source_sha256
                        ):
                            raise ReviewAccessError("Approved master identity changed; review the current bytes")
                        source_label: str = ref["label"] if "label" in ref else ref["scene_id"]
                        images.append(replace(image, scene=payload.scene_labels.get(ref["scene_id"], source_label)))
                    inventory = await _owned_thread(
                        export_delivery, images, root / "products", property_name=payload.property_name, export_presets=presets
                    )
                    effective_derivatives = applied_derivatives(inventory)
                    approval = canonicalize_json(
                        {
                            "schema": "tp.photo.approved_set.v1",
                            "job_id": context.job_id,
                            "review_version": payload.expected_version,
                            "masters": decisions,
                            "scope": "explicit_hash_bound_operator_approval_not_automated_photographic_acceptance",
                        }
                    )
                    approval_path = root / "products/approval-record.json"
                    approval_path.write_bytes(approval)
                    import json

                    spec_path = root / "products/package-spec.json"
                    spec = json.loads(spec_path.read_bytes())
                    spec["files"].append(
                        {
                            "source": "approval-record.json",
                            "archive_path": "approval-record.json",
                            "sha256": hashlib.sha256(approval).hexdigest(),
                            "size_bytes": len(approval),
                        }
                    )
                    spec_path.write_bytes(canonicalize_json(spec))
                    from transformation_portal.photographic_delivery.models import slug

                    archive = root / f"{slug(payload.property_name, 'property_name')}-approved-{identity[:12]}.zip"
                    receipt = await _owned_thread(package_delivery, root / "products", archive)
                    files: list[JsonObject] = []
                    for path, content_type in (
                        (archive, "application/zip"),
                        (Path(str(archive) + ".sha256"), "text/plain"),
                        (Path(str(archive) + ".receipt.json"), "application/json"),
                        (root / "products/inventory.json", "application/json"),
                        (root / "products/contact-sheet.jpg", "image/jpeg"),
                    ):
                        with path.open("rb") as stream:
                            digest = await _owned_thread(lambda: hashlib.file_digest(stream, "sha256").hexdigest())
                        row: JsonObject = {
                            "id": identity,
                            "filename": path.name,
                            "storage_path": f"photographic-review/{identity}/{path.name}",
                            "sha256": digest,
                            "size_bytes": path.stat().st_size,
                            "content_type": content_type,
                            "kind": "review_asset",
                            "artifact_path": f"deliveries/{identity}/{path.name}",
                        }
                        await context.publish(row["storage_path"], path, content_type)
                        publication_paths.append(row["storage_path"])
                        files.append(row)
            entry: JsonObject = {
                "id": identity,
                "files": files,
                "approved_sha256": hashes,
                "approvals": decisions,
                "receipt": receipt,
                "inventory": inventory,
                "actor": context.actor,
                "created_at": time.time(),
                "approval_version": payload.expected_version,
                "request_correlation_id": payload.request_correlation_id,
                "request_fingerprint_schema": INTENT_SCHEMA,
                "request_fingerprint_sha256": intent_sha256,
                "intent": intent,
                "applied_derivatives_schema": APPLIED_SCHEMA,
                "applied_derivatives": effective_derivatives,
            }

            def change(current: JsonObject) -> JsonObject:
                current["deliveries"].append(entry)
                return current

            updated = await self.mutate(
                context,
                payload.expected_version,
                change,
                "delivery",
                {
                    "id": identity,
                    "approved_sha256": hashes,
                    "archive_sha256": files[0]["sha256"],
                    "approval_version": payload.expected_version,
                    "request_correlation_id": payload.request_correlation_id,
                    "request_fingerprint_schema": INTENT_SCHEMA,
                    "request_fingerprint_sha256": intent_sha256,
                },
            )
        return {
            "state_version": updated["state_version"],
            "id": identity,
            "request_correlation_id": payload.request_correlation_id,
            "request_fingerprint_schema": INTENT_SCHEMA,
            "request_fingerprint_sha256": intent_sha256,
            "filename": files[0]["filename"],
            "download_url": base_url(context) + f"/files/{identity}/{quote(files[0]['filename'])}",
            "sha256": files[0]["sha256"],
            "size_bytes": files[0]["size_bytes"],
            "inventory": inventory,
            "files": [
                {"filename": f["filename"], "url": base_url(context) + f"/files/{identity}/{quote(f['filename'])}"}
                for f in files
            ],
        }

    async def deliveries(
        self,
        context: ReviewContext,
        *,
        limit: int = 20,
        cursor: str | None = None,
        request_correlation_id: str | None = None,
    ) -> JsonObject:
        from transformation_portal.orchestrator.photographic_delivery_history import history_page

        result = history_page(
            context, await self.state(context), limit=limit, cursor=cursor, request_correlation_id=request_correlation_id
        )
        await self.revalidate(context)
        return result

    async def download(self, context: ReviewContext, identity: str, filename: str) -> FileResponse:
        state = await self.state(context)
        delivery = next((d for d in state["deliveries"] if d["id"] == identity), None)
        if delivery is None:
            raise ReviewAccessError("Delivery is unavailable", status=404, code="NOT_FOUND")
        if not delivery_is_approved(context, state, delivery):
            raise ReviewAccessError("Delivery approval was withdrawn", status=409, code="APPROVAL_REQUIRED")
        ref = next((f for f in delivery["files"] if f["filename"] == filename), None)
        if ref is None:
            raise ReviewAccessError("Delivery file is unavailable", status=404, code="NOT_FOUND")
        directory = Path(tempfile.mkdtemp(prefix="tp-delivery-download-"))
        owner = current_review_lease.get()
        if owner is not None:
            owner.track_path(directory)
        try:
            path = directory / "verified-download"
            lease = current_review_lease.get()
            if lease is not None:
                lease.check_scratch(ref["size_bytes"])
            await context.materialize(ref, path, MAX_BUNDLE_BYTES)
            await self.revalidate(context)
            if not delivery_is_approved(context, await self.state(context), delivery):
                raise ReviewAccessError("Delivery approval was withdrawn", status=409, code="APPROVAL_REQUIRED")
            return _OwnedDownloadResponse(
                path,
                directory=directory,
                media_type=ref["content_type"],
                filename=filename,
                headers={**HEADERS, "X-TP-Artifact-SHA256": ref["sha256"]},
            )
        except BaseException:
            shutil.rmtree(directory)
            raise
