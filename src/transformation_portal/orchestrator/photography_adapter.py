"""Versioned photography adapter behind the shared, fenced job executor.

The plan contains logical intent only. Separate, digest-bound physical bindings
come from trusted admission; neither carrier can specify an executable command.
Workers hydrate the admitted plan without rediscovery or configuration resolution.
"""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

from transformation_portal.core.da3_runtime import repo_local_da3_python_path
from transformation_portal.core.execution_plan import ExecutionPlanError, decode_bounded_json_object
from transformation_portal.core.execution_plan_v4 import ExecutionPlanV4
from transformation_portal.ingest.canonical_json import canonicalize_json
from transformation_portal.orchestrator.artifact_store.base import ArtifactStoreError
from transformation_portal.orchestrator.artifact_store.generation import GenerationPublisher
from transformation_portal.orchestrator.dispatch import DispatchFence

if TYPE_CHECKING:
    from transformation_portal.lux_depth_v5.lifecycle import LuxDepthV5Request, PreparedLuxExecutionV5
    from transformation_portal.lux_depth_v5.pipeline import LuxDepthV5Result

PHOTOGRAPHY_BINDINGS_SCHEMA = "tp.job.photography.bindings.v1"
MAX_PHOTOGRAPHY_BINDINGS_BYTES = 64 * 1024
_PATH_FIELDS = frozenset({"input_root", "runtime_python", "raw_python", "cache_root", "companion_root", "materials_root"})
_OPTIONAL_FIELDS = frozenset({"raw_python", "cache_root", "companion_root", "materials_root"})


def server_runtime_bindings() -> tuple[str, str | None]:
    """Select server-owned interpreters, preserving virtualenv executable spelling."""
    runtime = os.environ.get("TRANSFORMATION_PORTAL_DA3_PYTHON")
    if not runtime:
        candidate = repo_local_da3_python_path(Path(__file__))
        runtime = "" if candidate is None else str(candidate)
    raw = os.environ.get("TRANSFORMATION_PORTAL_RAW_PYTHON")
    return (
        os.path.abspath(os.path.expanduser(runtime)) if runtime else "",
        os.path.abspath(os.path.expanduser(raw)) if raw else None,
    )


def _path(value: Any, *, optional: bool) -> str | None:
    if optional and value is None:
        return None
    if (
        type(value) is not str
        or not value
        or len(value) > 4096
        or "\x00" in value
        or "\\" in value
        or value.startswith("//")
        or not Path(value).is_absolute()
        or ".." in Path(value).parts
        or str(Path(value)) != value
    ):
        raise ExecutionPlanError("Photography bindings require canonical absolute paths")
    return value


@dataclass(frozen=True)
class PhotographyBindings:
    """Canonical immutable carrier with no output, module, argv, or shell fields."""

    canonical_bytes: bytes

    def __post_init__(self) -> None:
        if type(self.canonical_bytes) is not bytes or len(self.canonical_bytes) > MAX_PHOTOGRAPHY_BINDINGS_BYTES:
            raise ExecutionPlanError("Photography bindings must be bounded canonical bytes")
        payload = decode_bounded_json_object(self.canonical_bytes)
        if (
            set(payload) != {"schema", *_PATH_FIELDS}
            or payload["schema"] != PHOTOGRAPHY_BINDINGS_SCHEMA
            or canonicalize_json(payload) != self.canonical_bytes
        ):
            raise ExecutionPlanError("Invalid closed photography bindings carrier")
        for name in _PATH_FIELDS:
            _path(payload[name], optional=name in _OPTIONAL_FIELDS)

    def to_payload(self) -> dict[str, Any]:
        return decode_bounded_json_object(self.canonical_bytes)

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> PhotographyBindings:
        return cls(canonicalize_json(payload))


@dataclass(frozen=True)
class PreparedPhotographyDispatch:
    plan_bytes: bytes
    bindings_bytes: bytes

    def __post_init__(self) -> None:
        validate_photography_dispatch(self.plan_bytes, self.bindings_bytes)


def validate_photography_dispatch(plan_bytes: bytes, bindings_bytes: bytes) -> ExecutionPlanV4:
    """Validate exact admitted bytes without accessing their physical namespaces."""
    plan = ExecutionPlanV4(plan_bytes)
    bindings = PhotographyBindings(bindings_bytes).to_payload()
    payload = plan.to_payload()
    if "publication" not in payload:
        raise ExecutionPlanError("Managed photography requires frozen publication limits")
    if any("companions" in item for item in payload["inputs"]) != (bindings["companion_root"] is not None):
        raise ExecutionPlanError("Photography companion bindings differ from the admitted plan")
    if ("materials_manifest" in payload) != (bindings["materials_root"] is not None):
        raise ExecutionPlanError("Photography material bindings differ from the admitted plan")
    return plan


def prepare_photography_dispatch(request: LuxDepthV5Request, *, publisher: GenerationPublisher) -> PreparedPhotographyDispatch:
    """Freeze one V5 request using only the server's configured interpreters."""
    from transformation_portal.lux_depth_v5.lifecycle import prepare

    runtime, raw = server_runtime_bindings()
    if request.runtime_python not in (None, runtime) or request.raw_python not in (None, raw):
        raise ExecutionPlanError("Photography interpreters must be selected by the server")
    prepared = prepare(replace(request, runtime_python=runtime, raw_python=raw), publisher=publisher)
    bindings = PhotographyBindings.from_payload(
        {
            "schema": PHOTOGRAPHY_BINDINGS_SCHEMA,
            "input_root": str(prepared.input_root),
            "runtime_python": prepared.runtime_python,
            "raw_python": prepared.raw_python,
            "cache_root": None if prepared.cache_root is None else str(prepared.cache_root),
            "companion_root": None if prepared.companion_root is None else str(prepared.companion_root),
            "materials_root": None if prepared.materials_root is None else str(prepared.materials_root),
        }
    )
    return PreparedPhotographyDispatch(prepared.canonical_plan_bytes, bindings.canonical_bytes)


def consume_photography_dispatch(plan_bytes: bytes, bindings_bytes: bytes, *, execution_root: Path) -> PreparedLuxExecutionV5:
    """Hydrate exact intent; never prepare, discover inputs, or select a new model."""
    from transformation_portal.lux_depth_v5.lifecycle import PreparedLuxExecutionV5, validate_prepared_bindings

    plan = validate_photography_dispatch(plan_bytes, bindings_bytes)
    bound = PhotographyBindings(bindings_bytes).to_payload()
    runtime, raw = server_runtime_bindings()
    if (bound["runtime_python"], bound["raw_python"]) != (runtime, raw):
        raise ExecutionPlanError("Photography runtime bindings differ from the current server policy")
    prepared = PreparedLuxExecutionV5(
        plan,
        Path(bound["input_root"]),
        execution_root,
        bound["runtime_python"],
        bound["raw_python"],
        None if bound["cache_root"] is None else Path(bound["cache_root"]),
        None if bound["companion_root"] is None else Path(bound["companion_root"]),
        None if bound["materials_root"] is None else Path(bound["materials_root"]),
    )
    validate_prepared_bindings(prepared)
    return prepared


def verify_photography_dispatch_result(plan_bytes: bytes, *, output_root: Path) -> LuxDepthV5Result:
    """Reconstruct completion exclusively from independently verified output bytes."""
    from transformation_portal.lux_depth_v4.io import snapshot
    from transformation_portal.lux_depth_v5.evidence import verify_execution_evidence_v3
    from transformation_portal.lux_depth_v5.pipeline import LuxDepthV5Result

    plan = ExecutionPlanV4(plan_bytes)
    observed, _ = snapshot(output_root, output_root / "execution-plan.json", maximum_bytes=len(plan_bytes))
    if observed != plan_bytes:
        raise ExecutionPlanError("Photography completion differs from the exact admitted plan bytes")
    verified = verify_execution_evidence_v3(output_root, expected_plan_sha256=plan.plan_fingerprint_sha256)
    verified_plan = next((record for record in verified.artifacts if record.path == "execution-plan.json"), None)
    if verified_plan is None or (verified_plan.sha256, verified_plan.size_bytes) != (
        hashlib.sha256(plan_bytes).hexdigest(),
        len(plan_bytes),
    ):
        raise ExecutionPlanError("Verified photography plan differs from the exact admitted bytes")
    evidence = verified.to_payload()
    return LuxDepthV5Result(
        output_root,
        output_root / "execution-evidence.json",
        plan.plan_fingerprint_sha256,
        tuple(record.path for record in verified.artifacts),
        len(evidence["inputs"]),
        evidence["cache"]["hits"],
        evidence["cache"]["misses"],
    )


class ManagedPhotographyPublisher(GenerationPublisher):
    """Project the verified V5 inventory into ordinary portal artifact items."""

    async def publish(
        self,
        fence: DispatchFence,
        files: Mapping[str, Path],
        *,
        state: str,
        exit_code: int | None,
        artifacts: dict[str, Any],
        run_summary: dict[str, Any],
        error: dict[str, Any] | None = None,
        expected_file_integrity: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> dict[str, Any]:
        from transformation_portal.portal import job_artifacts

        limits = self.limits
        paths = artifacts.get("paths")
        if (
            artifacts.get("schema") != "tp.lux.delivery.v3"
            or artifacts.get("execution_evidence") != "execution-evidence.json"
            or not isinstance(paths, list)
            or len(paths) != len(files)
            or set(paths) != set(files)
            or expected_file_integrity is None
            or set(expected_file_integrity) != set(files)
            or len(files) > limits.max_files
        ):
            raise ArtifactStoreError("Managed photography projection requires the exact verified V5 inventory")
        items: list[dict[str, Any]] = []
        for relative in sorted(files):
            normalized = job_artifacts._normalize_artifact_relative_path(relative)
            if normalized != relative:
                raise ArtifactStoreError("Managed photography artifact paths must be canonical")
            integrity = expected_file_integrity[relative]
            size, digest = integrity.get("size_bytes"), integrity.get("sha256")
            if (
                set(integrity) != {"size_bytes", "sha256"}
                or type(size) is not int
                or not 0 <= size <= limits.max_file_bytes
                or not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise ArtifactStoreError("Managed photography projection requires complete verified file integrity")
            path = Path(relative)
            content_type = job_artifacts._artifact_content_type(path)
            url = job_artifacts._artifact_url(fence.locator.job_id, relative)
            items.append(
                {
                    "name": path.name,
                    "path": relative,
                    "relative_path": relative,
                    "artifact_type": job_artifacts._infer_artifact_type(path),
                    "media_kind": job_artifacts._artifact_media_kind(path),
                    "previewable": job_artifacts._artifact_is_previewable(path),
                    "browser_previewable": job_artifacts._artifact_is_browser_previewable(path),
                    "content_type": content_type,
                    "mime_type": content_type,
                    "display_hint": job_artifacts._artifact_display_hint(relative, path),
                    "url": url,
                    "download_url": url,
                    "size_bytes": size,
                    "sha256": digest,
                    "fingerprint_status": "ok",
                }
            )
            if re.fullmatch(r"input-[0-9]{4}/delivery\.tif", relative):
                preview = str(path.parent / "preview.png")
                if preview in files:
                    # Associate only the same input's independently verified
                    # preview. Never discover a filesystem sibling or imply
                    # that browsers can decode the archival TIFF itself.
                    items[-1]["preview_url"] = job_artifacts._artifact_url(fence.locator.job_id, preview)
                    items[-1]["preview_mime_type"] = "image/png"
        projected = {**artifacts, "items": items, "indexed_count": len(items), "truncated": False}
        if len(canonicalize_json(projected)) > limits.max_manifest_bytes:
            raise ArtifactStoreError("Managed photography artifact projection exceeds its metadata byte limit")
        return await super().publish(
            fence,
            files,
            state=state,
            exit_code=exit_code,
            artifacts=projected,
            run_summary=run_summary,
            error=error,
            expected_file_integrity=expected_file_integrity,
        )
